"""All-time (1920-2025) QB dataset pipeline.

Run: python -m src.historical_pipeline

What this actually does, honestly:
  - Ingests real 1999-2025 player-season passing data from nflverse (via
    nflreadpy) into the database, computes standardized efficiency metrics,
    and computes era-relative z-scores/percentiles against two baselines
    (single-season/league, and a rolling 5-year/league window).
  - Does NOT fabricate any 1920-1998 or 1960-1969 AFL player rows. No
    legitimately licensed bulk source for that data was found (see
    config/sources.json and src/historical_sources.py); it is recorded in the
    coverage report as not_ingested with the specific reason, per the
    project's instruction to flag missing coverage rather than invent it.
  - The schema (src/db/schema.py) and season_calendar reference table already
    support the full 1920-2025 range and the APFA/NFL/AFL league distinction,
    so a future licensed historical source can be plugged in via
    src/historical_sources.py without a schema change.

Idempotent by construction: each run deletes and reinserts the rows for the
seasons/source it is about to (re)write, inside the same transaction, rather
than attempting dialect-specific upserts (which differ between SQLite and
Postgres). Running it twice back-to-back produces identical row counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src import ingest_nflverse, metrics
from src.db.schema import (
    EraAdjustedStat, League, Player, PlayerSeasonDerived, PlayerSeasonRaw,
    PlayerSeasonTeamStint, SeasonCalendar, Source, Team, get_engine, get_session, init_db,
)
from src import coverage as coverage_mod
from src.era_adjust import rolling_era_baseline, season_league_baseline
from src.historical_sources import load_source_registry
from src.qualification import QualificationRules

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

START_SEASON = 1999
END_SEASON = 2025  # excludes the in-progress 2026 season

# metric -> (weight column used for that metric's baseline). Passing metrics are
# weighted by pass attempts; rushing metrics by carries, since weighting a QB's
# yards-per-carry by his *pass* volume would be the wrong volume proxy.
ERA_METRICS = {
    'yards_per_attempt': 'attempts', 'completion_pct': 'attempts', 'td_pct': 'attempts',
    'int_pct': 'attempts', 'adjusted_yards_per_attempt': 'attempts',
    'adjusted_net_yards_per_attempt': 'attempts', 'passer_rating': 'attempts',
    'yards_per_carry': 'carries', 'rushing_td_pct': 'carries',
    'yards_per_play': 'total_plays',
}

LEAGUE_DEFS = [
    ('APFA', 'American Professional Football Association', 1920, 1921,
     'Renamed National Football League (NFL) starting with the 1922 season.'),
    ('NFL', 'National Football League', 1922, None,
     'Continuous since 1922; absorbed the AFL starting with the 1970 season.'),
    ('AFL', 'American Football League (pre-merger)', 1960, 1969,
     'Operated independently 1960-1969, merged into the NFL beginning 1970. Kept as its own '
     'league_id so pre-merger AFL statistics are never pooled with NFL baselines for the same years.'),
]


def _none_if_nan(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
    except TypeError:
        pass
    return value


def _scalar(value):
    v = float(value)
    return None if np.isnan(v) else v


def seed_reference_tables(session: Session) -> None:
    for s in load_source_registry():
        existing = session.get(Source, s['source_id'])
        if existing is None:
            existing = Source(source_id=s['source_id'])
            session.add(existing)
        existing.name = s['name']
        existing.url = s.get('url')
        existing.access_method = s['access_method']
        existing.license_notes = s.get('license_notes')
        existing.coverage_start_season = s.get('coverage_start_season')
        existing.coverage_end_season = s.get('coverage_end_season')
        existing.status = s['status']
        existing.status_reason = s.get('status_reason')
        existing.verified_on = s.get('verified_on')

    for league_id, name, start, end, notes in LEAGUE_DEFS:
        existing = session.get(League, league_id)
        if existing is None:
            existing = League(league_id=league_id)
            session.add(existing)
        existing.name = name
        existing.start_season = start
        existing.end_season = end
        existing.notes = notes
    session.commit()


def seed_season_calendar(session: Session) -> None:
    session.query(SeasonCalendar).delete(synchronize_session=False)
    cal = json.loads((ROOT / 'config' / 'season_calendar.json').read_text(encoding='utf-8'))

    def league_for(season: int) -> str:
        return 'APFA' if season <= 1921 else 'NFL'

    for band in cal['nfl_apfa_bands']:
        for season in range(band['start'], band['end'] + 1):
            session.add(SeasonCalendar(
                league_id=league_for(season), season=season,
                games_scheduled=band.get('games_scheduled'),
                confidence=band['confidence'], notes=band.get('notes'),
            ))
    for band in cal['afl_bands']:
        for season in range(band['start'], band['end'] + 1):
            session.add(SeasonCalendar(
                league_id='AFL', season=season,
                games_scheduled=band.get('games_scheduled'),
                confidence=band['confidence'], notes=band.get('notes'),
            ))
    session.commit()


def seed_teams(session: Session) -> None:
    teams_df = ingest_nflverse.fetch_teams()
    existing = {(t.league_id, t.abbr) for t in session.query(Team).all()}
    for _, row in teams_df.iterrows():
        key = (row['league_id'], row['abbr'])
        if key in existing:
            continue
        session.add(Team(league_id=row['league_id'], abbr=row['abbr'], name=row['name']))
    session.commit()


def upsert_players(session: Session, season_totals: pd.DataFrame, source_id: str) -> dict:
    existing = {p.source_player_id: p.id for p in
                session.query(Player).filter(Player.source_id == source_id).all()}
    unique_players = season_totals[['source_player_id', 'display_name']].drop_duplicates('source_player_id')
    for _, row in unique_players.iterrows():
        if row['source_player_id'] in existing:
            continue
        player = Player(source_id=source_id, source_player_id=row['source_player_id'],
                         display_name=row['display_name'])
        session.add(player)
        session.flush()
        existing[row['source_player_id']] = player.id
    session.commit()
    return existing


def refresh_player_seasons(session: Session, season_totals: pd.DataFrame, stints: pd.DataFrame,
                            source_id: str, seasons: list) -> None:
    player_id_by_source = upsert_players(session, season_totals, source_id)

    stale = (session.query(PlayerSeasonRaw)
             .filter(PlayerSeasonRaw.source_id == source_id, PlayerSeasonRaw.season.in_(seasons))
             .all())
    for row in stale:
        session.delete(row)
    session.commit()

    stints_by_key = {key: grp for key, grp in stints.groupby(['source_player_id', 'season'])}

    for _, row in season_totals.iterrows():
        player_id = player_id_by_source[row['source_player_id']]
        rec = PlayerSeasonRaw(
            player_id=player_id, source_id=source_id, season=int(row['season']),
            league_id=row['league_id'], season_type=row['season_type'],
            position_listed=row['position_listed'], is_qb_listed=bool(row['is_qb_listed']),
            primary_team=row['primary_team'],
            games=_none_if_nan(row['games']), completions=_none_if_nan(row['completions']),
            attempts=_none_if_nan(row['attempts']), passing_yards=_none_if_nan(row['passing_yards']),
            passing_tds=_none_if_nan(row['passing_tds']), interceptions=_none_if_nan(row['interceptions']),
            sacks_suffered=_none_if_nan(row['sacks_suffered']),
            sack_yards_lost=_none_if_nan(row['sack_yards_lost']),
            carries=_none_if_nan(row['carries']), rushing_yards=_none_if_nan(row['rushing_yards']),
            rushing_tds=_none_if_nan(row['rushing_tds']),
            rushing_fumbles_lost=_none_if_nan(row['rushing_fumbles_lost']),
            completeness_flag=row['completeness_flag'], completeness_notes=row['completeness_notes'],
        )
        session.add(rec)
        session.flush()

        key = (row['source_player_id'], row['season'])
        if key in stints_by_key:
            for _, srow in stints_by_key[key].iterrows():
                session.add(PlayerSeasonTeamStint(
                    player_season_id=rec.id, team_abbr=srow['team_abbr'],
                    games=_none_if_nan(srow['games']), attempts=_none_if_nan(srow['attempts']),
                ))
    session.commit()


def compute_derived(session: Session, rules: QualificationRules) -> None:
    session.query(PlayerSeasonDerived).delete(synchronize_session=False)
    session.commit()
    for r in session.query(PlayerSeasonRaw).all():
        comp_pct = metrics.completion_pct(r.completions, r.attempts)
        ypa = metrics.yards_per_attempt(r.passing_yards, r.attempts)
        tdp = metrics.td_pct(r.passing_tds, r.attempts)
        intp = metrics.int_pct(r.interceptions, r.attempts)
        aya = metrics.adjusted_yards_per_attempt(r.passing_yards, r.passing_tds, r.interceptions, r.attempts)
        anya = metrics.adjusted_net_yards_per_attempt(
            r.passing_yards, r.passing_tds, r.interceptions, r.attempts, r.sacks_suffered, r.sack_yards_lost)
        rating = metrics.passer_rating(r.completions, r.attempts, r.passing_yards, r.passing_tds, r.interceptions)
        ypc = metrics.yards_per_carry(r.rushing_yards, r.carries)
        rush_tdp = metrics.rushing_td_pct(r.rushing_tds, r.carries)
        tot_yards = metrics.total_yards(r.passing_yards, r.rushing_yards)
        tot_tds = metrics.total_tds(r.passing_tds, r.rushing_tds)
        ypp = metrics.yards_per_play(r.passing_yards, r.rushing_yards, r.attempts, r.carries)
        threshold = rules.threshold(r.season, r.league_id)
        qualifies = bool(r.attempts is not None and r.attempts >= threshold)
        session.add(PlayerSeasonDerived(
            player_season_id=r.id,
            completion_pct=_scalar(comp_pct), yards_per_attempt=_scalar(ypa),
            td_pct=_scalar(tdp), int_pct=_scalar(intp),
            adjusted_yards_per_attempt=_scalar(aya), adjusted_net_yards_per_attempt=_scalar(anya),
            passer_rating=_scalar(rating),
            yards_per_carry=_scalar(ypc), rushing_td_pct=_scalar(rush_tdp),
            total_yards=_scalar(tot_yards), total_tds=_scalar(tot_tds), yards_per_play=_scalar(ypp),
            qualifies_leaderboard=qualifies,
            qualification_threshold_attempts=threshold,
        ))
    session.commit()


def compute_era_adjusted(session: Session) -> None:
    session.query(EraAdjustedStat).delete(synchronize_session=False)
    session.commit()

    rows = []
    for r in session.query(PlayerSeasonRaw).all():
        d = r.derived
        if d is None:
            continue
        rows.append({
            'player_season_id': r.id, 'season': r.season, 'league_id': r.league_id,
            'attempts': r.attempts or 0, 'carries': r.carries or 0,
            'total_plays': (r.attempts or 0) + (r.carries or 0),
            'qualifies_leaderboard': d.qualifies_leaderboard,
            'completion_pct': d.completion_pct, 'yards_per_attempt': d.yards_per_attempt,
            'td_pct': d.td_pct, 'int_pct': d.int_pct,
            'adjusted_yards_per_attempt': d.adjusted_yards_per_attempt,
            'adjusted_net_yards_per_attempt': d.adjusted_net_yards_per_attempt,
            'passer_rating': d.passer_rating,
            'yards_per_carry': d.yards_per_carry, 'rushing_td_pct': d.rushing_td_pct,
            'yards_per_play': d.yards_per_play,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        session.commit()
        return

    for metric, weight_col in ERA_METRICS.items():
        season_scored = season_league_baseline(df, metric, weight_col=weight_col)
        _store_era_rows(session, season_scored, metric, 'season_league')

        rolling_scored = rolling_era_baseline(df, metric, weight_col=weight_col, window_years=5)
        _store_era_rows(session, rolling_scored, metric, 'rolling_5yr')
    session.commit()


def _store_era_rows(session: Session, scored_df: pd.DataFrame, metric: str, baseline_method: str) -> None:
    for _, row in scored_df.iterrows():
        session.add(EraAdjustedStat(
            player_season_id=int(row['player_season_id']), metric=metric, baseline_method=baseline_method,
            baseline_group_size=None if pd.isna(row['baseline_group_size']) else int(row['baseline_group_size']),
            baseline_mean=None if pd.isna(row['baseline_mean']) else float(row['baseline_mean']),
            baseline_sd=None if pd.isna(row['baseline_sd']) else float(row['baseline_sd']),
            z_score=None if pd.isna(row['z_score']) else float(row['z_score']),
            percentile=None if pd.isna(row['percentile']) else float(row['percentile']),
        ))


def _table_to_df(session: Session, model) -> pd.DataFrame:
    """Loads an ORM-mapped table into a DataFrame via the ORM layer rather than
    pandas' own SQLAlchemy-connectable detection, which is unreliable across
    pandas/SQLAlchemy version combinations (observed to break read_sql_table
    against both Engine and Connection in this environment)."""
    cols = [c.name for c in model.__table__.columns]
    rows = session.query(model).all()
    records = [{col: getattr(row, col) for col in cols} for row in rows]
    return pd.DataFrame.from_records(records, columns=cols)


def export_parquet(session: Session, data_dir: Path | None = None) -> None:
    """Writes Parquet exports into data_dir (defaults to the project's data/
    directory). Callers that run the pipeline against a non-default database
    (e.g. tests, against a temp DB) MUST pass a non-default data_dir so they
    don't clobber the real project exports."""
    data_dir = data_dir or DATA
    data_dir.mkdir(parents=True, exist_ok=True)

    raw = _table_to_df(session, PlayerSeasonRaw)
    derived = _table_to_df(session, PlayerSeasonDerived)
    era = _table_to_df(session, EraAdjustedStat)
    players = _table_to_df(session, Player)
    coverage = _table_to_df(session, coverage_mod.CoverageReport)

    raw.to_parquet(data_dir / 'player_seasons_raw.parquet', index=False)
    derived.to_parquet(data_dir / 'player_seasons_derived.parquet', index=False)
    era.to_parquet(data_dir / 'era_adjusted_stats.parquet', index=False)
    coverage.to_parquet(data_dir / 'coverage_report.parquet', index=False)

    joined = (raw.merge(players[['id', 'display_name']], left_on='player_id', right_on='id',
                         suffixes=('', '_player'))
              .merge(derived, left_on='id', right_on='player_season_id', suffixes=('', '_derived')))
    joined.to_parquet(data_dir / 'qb_seasons_all_time.parquet', index=False)


def main(db_url: str | None = None, seasons: list | None = None, data_dir: Path | None = None,
         results_dir: Path | None = None) -> dict:
    """seasons defaults to the full 1999-2025 range; pass a short list (e.g. in
    tests) to run the same pipeline faster against a subset. data_dir/results_dir
    default to the project's data/ and results/ directories -- pass separate
    directories (e.g. in tests) whenever db_url is also overridden, so a
    partial/test run never overwrites the real Parquet exports or coverage report."""
    engine = init_db(get_engine(db_url))
    session = get_session(engine)
    try:
        seed_reference_tables(session)
        seed_season_calendar(session)
        seed_teams(session)

        seasons = list(seasons) if seasons is not None else list(range(START_SEASON, END_SEASON + 1))
        season_totals = ingest_nflverse.fetch_season_totals(seasons)
        stints = ingest_nflverse.fetch_team_stints(seasons)
        refresh_player_seasons(session, season_totals, stints, ingest_nflverse.SOURCE_ID, seasons)

        rules = QualificationRules()
        compute_derived(session, rules)
        compute_era_adjusted(session)

        coverage_mod.rebuild_coverage_table(session, ingest_nflverse.SOURCE_ID, seasons)
        report_dest = (results_dir / 'coverage_report.json') if results_dir else None
        summary = coverage_mod.write_report(session, dest=report_dest)

        export_parquet(session, data_dir=data_dir)
        return summary
    finally:
        session.close()


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2, default=str))
