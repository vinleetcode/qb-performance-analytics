"""CFB-to-NFL QB prospect pipeline: NFL Draft board (nflreadpy, always available)
+ CollegeFootballData player-season stats (needs CFBD_API_KEY) + identity
matching + a leakage-safe pre-draft-features / NFL-outcomes export split.

Run: python -m src.prospect_pipeline

This pipeline extends the SAME database used by src/historical_pipeline.py
(new tables only -- see src/db/schema.py) and never touches the existing NFL
tables (players, player_seasons_raw, etc.) except to READ from `players` for
the gsis_id join. It is safe to run with or without a configured CFBD key:
without one, the draft board is still ingested and matched against whatever
CFB data already exists in the DB (possibly none), and the report says so
explicitly rather than silently producing an empty-looking success.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from src import ingest_cfbd, ingest_draft, match_prospects, metrics
from src.cfbd_client import CFBDClient, CFBDRequestError
from src.config import cfbd_api_key_configured
from src.db.schema import (
    CfbPlayer, CfbPlayerSeason, CfbPlayerSeasonDerived, CfbTeamSeasonContext,
    CollegeNflPlayerMap, NflDraftPickRaw, Player, QbDraftProspect,
    get_engine, get_session, init_db,
)
from src.historical_pipeline import seed_reference_tables

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

DRAFT_SOURCE_ID = ingest_draft.SOURCE_ID
CFBD_SOURCE_ID = ingest_cfbd.SOURCE_ID

DEFAULT_DRAFT_SEASONS = list(range(2000, 2027))
DEFAULT_CFB_SEASONS = list(range(2004, 2026))

NFL_OUTCOME_COLUMNS = {
    'nfl_games_played_career', 'career_pass_completions', 'career_pass_attempts',
    'career_pass_yards', 'career_pass_tds', 'career_pass_ints', 'career_rush_attempts',
    'career_rush_yards', 'career_rush_tds', 'allpro', 'probowls', 'seasons_started',
    'career_av', 'hof', 'career_last_season',
}


def _none_if_nan(value):
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
    except TypeError:
        pass
    return value


def ingest_draft_board(session: Session, seasons: list[int]) -> pd.DataFrame:
    """Returns the cleaned QB prospect DataFrame actually written (for reuse by
    the matching step without a second DB round-trip)."""
    raw_df = ingest_draft.fetch_draft_raw(seasons)
    bio_df = ingest_draft.fetch_bio_backfill(seasons)
    qb_df = ingest_draft.build_qb_prospects(raw_df, bio_df)

    stale = (session.query(NflDraftPickRaw)
             .filter(NflDraftPickRaw.source_id == DRAFT_SOURCE_ID, NflDraftPickRaw.season.in_(seasons))
             .all())
    for row in stale:
        session.delete(row)
    session.commit()

    raw_id_by_key: dict[tuple[int, int], int] = {}
    for _, row in raw_df.iterrows():
        rec = NflDraftPickRaw(
            source_id=DRAFT_SOURCE_ID, season=int(row['season']), round=_none_if_nan(row.get('round')),
            pick=int(row['pick']), team=row.get('team'), gsis_id=_none_if_nan(row.get('gsis_id')),
            pfr_player_id=_none_if_nan(row.get('pfr_player_id')),
            cfb_player_id_slug=_none_if_nan(row.get('cfb_player_id_slug')),
            pfr_player_name=_none_if_nan(row.get('pfr_player_name')),
            position_listed=row.get('position_listed'), college=_none_if_nan(row.get('college')),
            age_at_draft=_none_if_nan(row.get('age_at_draft')), hof=_none_if_nan(row.get('hof')),
            career_last_season=_none_if_nan(row.get('career_last_season')),
            allpro=_none_if_nan(row.get('allpro')), probowls=_none_if_nan(row.get('probowls')),
            seasons_started=_none_if_nan(row.get('seasons_started')),
            career_av=_none_if_nan(row.get('career_av')),
            nfl_games_played_career=_none_if_nan(row.get('nfl_games_played_career')),
            career_pass_completions=_none_if_nan(row.get('career_pass_completions')),
            career_pass_attempts=_none_if_nan(row.get('career_pass_attempts')),
            career_pass_yards=_none_if_nan(row.get('career_pass_yards')),
            career_pass_tds=_none_if_nan(row.get('career_pass_tds')),
            career_pass_ints=_none_if_nan(row.get('career_pass_ints')),
            career_rush_attempts=_none_if_nan(row.get('career_rush_attempts')),
            career_rush_yards=_none_if_nan(row.get('career_rush_yards')),
            career_rush_tds=_none_if_nan(row.get('career_rush_tds')),
        )
        session.add(rec)
        session.flush()
        raw_id_by_key[(rec.season, rec.pick)] = rec.id
    session.commit()

    stale_qb = (session.query(QbDraftProspect)
                .join(NflDraftPickRaw, QbDraftProspect.draft_pick_raw_id == NflDraftPickRaw.id)
                .filter(NflDraftPickRaw.source_id == DRAFT_SOURCE_ID, NflDraftPickRaw.season.in_(seasons))
                .all())
    for row in stale_qb:
        session.delete(row)
    session.commit()

    for _, row in qb_df.iterrows():
        raw_id = raw_id_by_key.get((int(row['draft_year']), int(row['pick'])))
        if raw_id is None:
            continue
        session.add(QbDraftProspect(
            draft_pick_raw_id=raw_id, source_id=DRAFT_SOURCE_ID, draft_year=int(row['draft_year']),
            round=_none_if_nan(row.get('round')), pick=int(row['pick']), nfl_team=row.get('nfl_team'),
            player_name=row['player_name'], gsis_id=_none_if_nan(row.get('gsis_id')),
            pfr_player_id=_none_if_nan(row.get('pfr_player_id')),
            cfb_player_id_slug=_none_if_nan(row.get('cfb_player_id_slug')), college=row.get('college'),
            position_listed=row.get('position_listed'), height_inches=_none_if_nan(row.get('height_inches')),
            weight_lbs=_none_if_nan(row.get('weight_lbs')), birthdate=_none_if_nan(row.get('birthdate')),
        ))
    session.commit()
    return qb_df


def ingest_cfb_stats(session: Session, seasons: list[int]) -> list[str]:
    """Returns the list of unrecognized CFBD statType strings encountered (empty
    if none, or if CFBD wasn't called at all)."""
    if not cfbd_api_key_configured():
        return []

    client = CFBDClient()
    try:
        long_df = ingest_cfbd.fetch_player_stats_long(seasons, client)
    except CFBDRequestError:
        raise
    wide_df, unrecognized = ingest_cfbd.pivot_player_stats(long_df)
    if wide_df.empty:
        return unrecognized

    roster_df = ingest_cfbd.fetch_roster_bio(seasons, client)
    ratings_df = ingest_cfbd.fetch_team_ratings(seasons, client)

    # CFBD's roster can list more than one row for the same (season, team,
    # name) -- e.g. a position change mid-season. Drop to one bio row per key
    # (preferring a row that actually has height/weight) BEFORE merging, or a
    # fan-out here would silently duplicate player-season rows downstream.
    roster_key = ['season', 'school', 'player_name']
    roster_df = (roster_df.sort_values(by=['height_inches', 'weight_lbs'], na_position='last')
                 .drop_duplicates(subset=roster_key, keep='first'))

    wide_df = wide_df.merge(
        roster_df.rename(columns={'school': 'team'}), left_on=['season', 'team', 'player'],
        right_on=['season', 'team', 'player_name'], how='left', validate='many_to_one',
    )

    stale = (session.query(CfbPlayerSeason)
             .filter(CfbPlayerSeason.source_id == CFBD_SOURCE_ID, CfbPlayerSeason.season.in_(seasons))
             .all())
    for row in stale:
        session.delete(row)
    session.commit()

    existing_players = {p.cfbd_player_id: p.id for p in
                         session.query(CfbPlayer).filter(CfbPlayer.source_id == CFBD_SOURCE_ID).all()}

    for _, row in wide_df.iterrows():
        pid = str(row['playerId'])
        if pid not in existing_players:
            cfb_player = CfbPlayer(source_id=CFBD_SOURCE_ID, cfbd_player_id=pid, display_name=row['player'])
            session.add(cfb_player)
            session.flush()
            existing_players[pid] = cfb_player.id

        rec = CfbPlayerSeason(
            player_id=existing_players[pid], source_id=CFBD_SOURCE_ID, season=int(row['season']),
            school=row.get('team'), conference=row.get('conference'), position_listed=row.get('position'),
            completions=_none_if_nan(row.get('completions')), attempts=_none_if_nan(row.get('attempts')),
            passing_yards=_none_if_nan(row.get('passing_yards')), passing_tds=_none_if_nan(row.get('passing_tds')),
            interceptions=_none_if_nan(row.get('interceptions')),
            rushing_attempts=_none_if_nan(row.get('rushing_attempts')),
            rushing_yards=_none_if_nan(row.get('rushing_yards')), rushing_tds=_none_if_nan(row.get('rushing_tds')),
            height_inches=_none_if_nan(row.get('height_inches')), weight_lbs=_none_if_nan(row.get('weight_lbs')),
        )
        session.add(rec)
        session.flush()

        comp_pct = metrics.completion_pct(rec.completions, rec.attempts)
        ypa = metrics.yards_per_attempt(rec.passing_yards, rec.attempts)
        tdp = metrics.td_pct(rec.passing_tds, rec.attempts)
        intp = metrics.int_pct(rec.interceptions, rec.attempts)
        rating = metrics.ncaa_passer_rating(rec.completions, rec.attempts, rec.passing_yards,
                                             rec.passing_tds, rec.interceptions)
        ypc = metrics.yards_per_carry(rec.rushing_yards, rec.rushing_attempts)
        session.add(CfbPlayerSeasonDerived(
            cfb_player_season_id=rec.id,
            completion_pct=_float_or_none(comp_pct), yards_per_attempt=_float_or_none(ypa),
            td_pct=_float_or_none(tdp), int_pct=_float_or_none(intp),
            ncaa_passer_rating=_float_or_none(rating), yards_per_carry=_float_or_none(ypc),
        ))

    stale_ctx = (session.query(CfbTeamSeasonContext)
                 .filter(CfbTeamSeasonContext.source_id == CFBD_SOURCE_ID,
                         CfbTeamSeasonContext.season.in_(seasons))
                 .all())
    for row in stale_ctx:
        session.delete(row)
    session.commit()
    for _, row in ratings_df.iterrows():
        session.add(CfbTeamSeasonContext(
            source_id=CFBD_SOURCE_ID, season=int(row['season']), school=row['school'],
            conference=_none_if_nan(row.get('conference')), sp_rating=_none_if_nan(row.get('sp_rating')),
            sp_ranking=_none_if_nan(row.get('sp_ranking')),
            strength_of_schedule=_none_if_nan(row.get('strength_of_schedule')),
        ))
    session.commit()
    return unrecognized


def _float_or_none(value):
    v = float(value)
    return None if np.isnan(v) else v


def run_matching(session: Session) -> None:
    cfb_candidates_by_identity: dict[int, match_prospects.CfbCandidate] = {}
    for player in session.query(CfbPlayer).all():
        schools = {match_prospects.normalize_school(s.school) for s in player.seasons if s.school}
        seasons = {s.season for s in player.seasons}
        cfb_candidates_by_identity[player.id] = match_prospects.CfbCandidate(
            cfb_player_id=player.id, name=player.display_name, schools=schools, seasons=seasons)
    candidates = list(cfb_candidates_by_identity.values())

    nfl_players_by_gsis = {p.source_player_id: p.id for p in
                            session.query(Player).filter(Player.source_id == 'nflverse_player_stats').all()}

    prospects = session.query(QbDraftProspect).all()
    for prospect in prospects:
        existing_map = session.query(CollegeNflPlayerMap).filter(
            CollegeNflPlayerMap.qb_draft_prospect_id == prospect.id).first()
        if existing_map:
            session.delete(existing_map)
    session.commit()

    for prospect in prospects:
        if prospect.gsis_id and prospect.gsis_id in nfl_players_by_gsis:
            nfl_player_id, nfl_method = nfl_players_by_gsis[prospect.gsis_id], 'stable_gsis_id'
        elif prospect.gsis_id:
            nfl_player_id, nfl_method = None, 'no_nfl_passing_record'
        else:
            nfl_player_id, nfl_method = None, 'no_gsis_id'

        result = match_prospects.match_prospect_to_cfb(
            prospect.player_name, prospect.college, prospect.draft_year, candidates)

        status = 'matched' if result.method in ('exact_name_school', 'fuzzy_name_school') else (
            'ambiguous' if result.method == 'ambiguous' else 'unmatched')

        session.add(CollegeNflPlayerMap(
            qb_draft_prospect_id=prospect.id, nfl_player_id=nfl_player_id, cfb_player_id=result.cfb_player_id,
            nfl_match_method=nfl_method, cfb_match_method=result.method, cfb_match_confidence=result.confidence,
            match_status=status, notes=result.notes,
        ))
    session.commit()


def export_parquet(session: Session, data_dir: Path | None = None) -> None:
    data_dir = data_dir or DATA
    data_dir.mkdir(parents=True, exist_ok=True)

    def df_of(model, cols=None):
        cols = cols or [c.name for c in model.__table__.columns]
        rows = session.query(model).all()
        return pd.DataFrame.from_records([{c: getattr(r, c) for c in cols} for r in rows], columns=cols)

    raw_draft = df_of(NflDraftPickRaw)
    qb_prospects = df_of(QbDraftProspect)
    cfb_seasons = df_of(CfbPlayerSeason)
    cfb_derived = df_of(CfbPlayerSeasonDerived)
    cfb_players = df_of(CfbPlayer)
    team_ctx = df_of(CfbTeamSeasonContext)
    id_map = df_of(CollegeNflPlayerMap)

    raw_draft.to_parquet(data_dir / 'nfl_draft_picks_raw.parquet', index=False)
    qb_prospects.to_parquet(data_dir / 'qb_draft_prospects.parquet', index=False)
    cfb_joined = cfb_seasons.merge(cfb_derived, left_on='id', right_on='cfb_player_season_id',
                                    suffixes=('', '_derived'), how='left')
    cfb_joined.to_parquet(data_dir / 'cfb_player_seasons.parquet', index=False)
    team_ctx.to_parquet(data_dir / 'cfb_team_season_context.parquet', index=False)
    id_map.to_parquet(data_dir / 'college_nfl_player_map.parquet', index=False)

    # --- Leakage-safe unified prospect export -------------------------------
    # Pre-draft features: bio + draft slot + matched college career stats +
    # team context. NO NFL career-outcome columns, by construction (NFL_OUTCOME_COLUMNS
    # never enters this frame -- see the explicit assertion below).
    prospect_cols = ['id', 'draft_year', 'round', 'pick', 'nfl_team', 'player_name', 'college',
                      'position_listed', 'height_inches', 'weight_lbs', 'birthdate']
    pre_draft = qb_prospects[prospect_cols].rename(columns={'id': 'qb_draft_prospect_id'})
    pre_draft = pre_draft.merge(
        id_map[['qb_draft_prospect_id', 'cfb_player_id', 'match_status', 'cfb_match_method',
                'cfb_match_confidence']], on='qb_draft_prospect_id', how='left')

    if not cfb_players.empty and not cfb_joined.empty:
        career = (cfb_joined.merge(cfb_players[['id', 'cfbd_player_id']], left_on='player_id', right_on='id',
                                    suffixes=('', '_player'))
                  .groupby('player_id', as_index=False)
                  .agg(cfb_seasons_played=('season', 'nunique'),
                       cfb_career_pass_attempts=('attempts', 'sum'),
                       cfb_career_pass_completions=('completions', 'sum'),
                       cfb_career_pass_yards=('passing_yards', 'sum'),
                       cfb_career_pass_tds=('passing_tds', 'sum'),
                       cfb_career_interceptions=('interceptions', 'sum'),
                       cfb_career_rush_yards=('rushing_yards', 'sum'),
                       cfb_final_season=('season', 'max')))
        pre_draft = pre_draft.merge(career, left_on='cfb_player_id', right_on='player_id', how='left') \
            .drop(columns=['player_id'], errors='ignore')

    leaked = NFL_OUTCOME_COLUMNS & set(pre_draft.columns)
    assert not leaked, f'NFL outcome columns leaked into pre-draft feature export: {leaked}'
    pre_draft.to_parquet(data_dir / 'prospects_pre_draft_features.parquet', index=False)

    # NFL outcomes, kept in a SEPARATE file, joinable by qb_draft_prospect_id/draft_pick_raw_id only.
    outcome_cols = ['id'] + sorted(NFL_OUTCOME_COLUMNS)
    outcomes = raw_draft[[c for c in outcome_cols if c in raw_draft.columns]].rename(
        columns={'id': 'draft_pick_raw_id'})
    outcomes = qb_prospects[['id', 'draft_pick_raw_id']].rename(columns={'id': 'qb_draft_prospect_id'}) \
        .merge(outcomes, on='draft_pick_raw_id', how='left')
    outcomes.to_parquet(data_dir / 'prospects_nfl_outcomes.parquet', index=False)


def build_report(session: Session, unrecognized_stat_types: list[str]) -> dict:
    total_qb_prospects = session.query(QbDraftProspect).count()
    total_cfb_players = session.query(CfbPlayer).count()
    total_cfb_player_seasons = session.query(CfbPlayerSeason).count()

    maps = session.query(CollegeNflPlayerMap).all()
    matched = sum(1 for m in maps if m.match_status == 'matched')
    ambiguous = sum(1 for m in maps if m.match_status == 'ambiguous')
    unmatched = sum(1 for m in maps if m.match_status == 'unmatched')
    nfl_stable_matches = sum(1 for m in maps if m.nfl_match_method == 'stable_gsis_id')
    zero_nfl_stat_prospects = sum(1 for m in maps if m.nfl_match_method in
                                   ('no_nfl_passing_record', 'no_gsis_id'))

    draft_years = sorted({p.draft_year for p in session.query(QbDraftProspect.draft_year).all()})
    cfb_seasons = sorted({s[0] for s in session.query(CfbPlayerSeason.season).distinct().all()})

    row_counts = {
        'nfl_draft_picks_raw': session.query(NflDraftPickRaw).count(),
        'qb_draft_prospects': total_qb_prospects,
        'cfb_players': total_cfb_players,
        'cfb_player_seasons': total_cfb_player_seasons,
        'cfb_team_season_context': session.query(CfbTeamSeasonContext).count(),
        'college_nfl_player_map': len(maps),
    }

    return {
        'cfbd_api_key_configured': cfbd_api_key_configured(),
        'unrecognized_cfbd_stat_types': unrecognized_stat_types,
        'draft_years_covered': [min(draft_years), max(draft_years)] if draft_years else None,
        'cfb_seasons_covered': [min(cfb_seasons), max(cfb_seasons)] if cfb_seasons else None,
        'totals': {
            'total_drafted_quarterbacks': total_qb_prospects,
            'total_college_quarterbacks_identified': total_cfb_players,
            'college_player_season_records': total_cfb_player_seasons,
            'matched': matched,
            'ambiguous': ambiguous,
            'unmatched': unmatched,
            'nfl_side_stable_id_matches': nfl_stable_matches,
            'prospects_with_zero_tracked_nfl_passing_stats': zero_nfl_stat_prospects,
        },
        'row_counts': row_counts,
    }


def write_report(report: dict, dest: Path | None = None) -> None:
    dest = dest or (RESULTS / 'prospect_coverage_report.json')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, default=str))


def main(db_url: str | None = None, draft_seasons: list | None = None, cfb_seasons: list | None = None,
         data_dir: Path | None = None, results_dir: Path | None = None) -> dict:
    engine = init_db(get_engine(db_url))
    session = get_session(engine)
    try:
        seed_reference_tables(session)

        draft_seasons = list(draft_seasons) if draft_seasons is not None else DEFAULT_DRAFT_SEASONS
        cfb_seasons = list(cfb_seasons) if cfb_seasons is not None else DEFAULT_CFB_SEASONS

        ingest_draft_board(session, draft_seasons)
        unrecognized = ingest_cfb_stats(session, cfb_seasons)
        run_matching(session)
        export_parquet(session, data_dir=data_dir)

        report = build_report(session, unrecognized)
        write_report(report, dest=(results_dir / 'prospect_coverage_report.json') if results_dir else None)
        return report
    finally:
        session.close()


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, default=str))
