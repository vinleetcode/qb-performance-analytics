"""Phase 3: advanced CFB-to-NFL QB dataset (college context, athletic profile,
NFL outcomes, survival/similarity/draft-value preparation).

Run:  python -m src.advanced_pipeline            (reads data/cfbd_cache; makes CFBD
                                                  calls only for cache misses, capped)
      python -m src.ingest_cfbd_advanced [budget] (optional: warm the cache first)

Extends the SAME database as Phases 1-2. Existing tables are only READ, with one
deliberate exception: college_nfl_player_map is re-derived by the existing
Phase-2 matcher (src.prospect_pipeline.run_matching), which now also consults
CFBD's own draft record (stable collegeAthleteId) -- see README "Identity".
Existing NFL statistics (players, player_seasons_raw, ...) are never written.

Idempotent: every Phase-3 table is rewritten with delete-then-insert inside the
run, so two consecutive runs produce identical row counts and values.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, insert
from sqlalchemy.orm import Session

from src import ingest_cfbd_advanced as A
from src import metrics, prospect_pipeline
from src.cfbd_client import CFBDClient
from src.db.schema import (
    CfbdDraftPick, CfbGame, CfbGamePassing, CfbGameRushing, CfbOffensiveSchemeLabel, CfbPlayer, CfbPlayerSeason,
    CfbPlayerSeasonDerived, CfbReceivingSeason, CfbRecruit, CfbTeamRosterTalent, CfbTeamSeasonContext,
    CfbTeamSeasonEnvironment, CfbTeamSeasonStats, CollegeNflPlayerMap, DraftValuePrediction,
    FeatureDefinition, NflCareerOutcome, NflCombineMeasurement, NflDraftPickRaw, NflProspectSeason,
    ProspectCollegeSeason, ProspectFeatureValue, ProspectSimilarity, QbDraftProspect, SurvivalRecord,
    get_engine, get_session, init_db,
)
from src.features import athletic, development, environment, opponent_adjust, registry, targets
from src.historical_pipeline import seed_reference_tables
from src.modeling import preprocess, similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
CONFIG = ROOT / 'config'
DEFAULT_CALL_BUDGET = 750

PHASE3_TABLES = [CfbGame, CfbGamePassing, CfbGameRushing, CfbTeamSeasonStats, CfbRecruit, CfbReceivingSeason, CfbdDraftPick,
                 NflCombineMeasurement, CfbOffensiveSchemeLabel, CfbTeamRosterTalent, CfbTeamSeasonEnvironment,
                 ProspectCollegeSeason, FeatureDefinition, ProspectFeatureValue, ProspectSimilarity,
                 NflProspectSeason, NflCareerOutcome, SurvivalRecord, DraftValuePrediction]


# ---------------------------------------------------------------------------
# persistence helpers
# ---------------------------------------------------------------------------

def _records(model, df: pd.DataFrame) -> list[dict]:
    cols = [c.name for c in model.__table__.columns if c.name != 'id' and c.name in df.columns]
    if df.empty:
        return []
    sub = df[cols].astype(object)
    sub = sub.where(pd.notna(sub), None)
    recs = sub.to_dict('records')
    date_cols = {c.name for c in model.__table__.columns if c.type.python_type is date}
    bool_cols = {c.name for c in model.__table__.columns if c.type.python_type is bool}
    int_cols = {c.name for c in model.__table__.columns if c.type.python_type is int}
    for r in recs:
        for k in date_cols & r.keys():
            if r[k] is not None:
                r[k] = pd.Timestamp(r[k]).date()
        for k in bool_cols & r.keys():
            if r[k] is not None:
                r[k] = bool(r[k])
        for k in int_cols & r.keys():
            if r[k] is not None:
                r[k] = int(r[k])
    return recs


def replace_table(session: Session, model, df: pd.DataFrame, where=None, chunk: int = 5000) -> int:
    session.execute(delete(model).where(where) if where is not None else delete(model))
    recs = _records(model, df)
    for i in range(0, len(recs), chunk):
        session.execute(insert(model), recs[i:i + chunk])
    session.commit()
    return len(recs)


def table_df(session: Session, model) -> pd.DataFrame:
    cols = [c.name for c in model.__table__.columns]
    rows = session.execute(model.__table__.select()).all()
    return pd.DataFrame(rows, columns=cols)


def load_draft_dates() -> dict:
    return json.loads((CONFIG / 'nfl_draft_dates.json').read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# 1) ingestion of raw / standardized source tables
# ---------------------------------------------------------------------------

def ingest_cfbd_sources(session: Session, client: CFBDClient, cfb_seasons, recruit_years, draft_years) -> dict:
    out: dict = {}
    games = A.fetch_games(cfb_seasons, client)
    out['games'] = games
    if not games.empty:
        g = games.rename(columns={'id': 'game_id', 'seasonType': 'season_type', 'startDate': 'start_date',
                                  'neutralSite': 'neutral_site', 'conferenceGame': 'conference_game',
                                  'homeTeam': 'home_team', 'homeConference': 'home_conference',
                                  'homeClassification': 'home_classification', 'homePoints': 'home_points',
                                  'awayTeam': 'away_team', 'awayConference': 'away_conference',
                                  'awayClassification': 'away_classification', 'awayPoints': 'away_points'})
        g['source_id'] = A.SOURCE_IDS['games']
        replace_table(session, CfbGame, g)

    gp = A.fetch_game_passing(cfb_seasons, games, client) if not games.empty else pd.DataFrame(columns=A.GAME_PASSING_COLUMNS)
    gp['athlete_key'] = gp['cfbd_player_id'].where(gp['cfbd_player_id'].notna(), 'TEAM:' + gp['player_name'].fillna(''))
    gp = gp.drop_duplicates(['game_id', 'team', 'athlete_key'])
    gp['source_id'] = A.SOURCE_IDS['game_passing']
    replace_table(session, CfbGamePassing, gp)
    out['game_passing'] = gp

    rush_seasons = [s for s in cfb_seasons if s in A.BOX_RUSHING_SEASONS]
    gr = A.fetch_game_rushing(rush_seasons, games, client) if (rush_seasons and not games.empty) \
        else pd.DataFrame(columns=A.GAME_RUSHING_COLUMNS)
    gr = gr.drop_duplicates(['game_id', 'team', 'cfbd_player_id'])
    gr['source_id'] = A.SOURCE_IDS['game_rushing']
    replace_table(session, CfbGameRushing, gr)
    out['game_rushing'] = gr

    ts, unmapped = A.fetch_team_season_stats(cfb_seasons, client)
    adv = A.fetch_team_advanced(cfb_seasons, client)
    if not ts.empty:
        ts = ts.merge(adv, on=['season', 'team'], how='left') if not adv.empty else ts
        ts = environment.gate_advanced(ts)
        ts['source_id'] = A.SOURCE_IDS['team_stats']
        replace_table(session, CfbTeamSeasonStats, ts)
    out['team_stats'], out['unmapped_team_stats'] = ts, unmapped

    out['talent'] = A.fetch_talent(cfb_seasons, client)
    out['recruiting_teams'] = A.fetch_recruiting_teams(recruit_years, client)
    rec = A.fetch_recruits(recruit_years, client)
    if not rec.empty:
        rec = rec.dropna(subset=['recruit_id']).drop_duplicates('recruit_id')
        rec['recruit_id'] = rec['recruit_id'].map(A.id_str)
        rec['athlete_id'] = rec['athlete_id'].map(A.id_str)
        rec['source_id'] = A.SOURCE_IDS['recruits']
        replace_table(session, CfbRecruit, rec)
    out['recruits'] = rec

    recv = A.fetch_receiving(cfb_seasons, client)
    if not recv.empty:
        recv['cfbd_player_id'] = recv['cfbd_player_id'].map(A.id_str)
        recv = recv.drop_duplicates(['season', 'cfbd_player_id', 'team'])
        recv['source_id'] = A.SOURCE_IDS['receiving']
        replace_table(session, CfbReceivingSeason, recv)
    out['receiving'] = recv
    out['returning'] = A.fetch_returning(cfb_seasons, client)

    dp = A.fetch_draft_picks(draft_years, client)
    if not dp.empty:
        dp = dp.dropna(subset=['year', 'overall']).drop_duplicates(['year', 'overall'])
        dp['college_athlete_id'] = dp['college_athlete_id'].map(A.id_str)
        dp['nfl_athlete_id'] = dp['nfl_athlete_id'].map(A.id_str)
        for c in ('height', 'weight', 'pre_draft_grade'):
            dp[c] = pd.to_numeric(dp[c], errors='coerce')
        dp['source_id'] = A.SOURCE_IDS['draft_picks']
        replace_table(session, CfbdDraftPick, dp)
    out['draft_picks'] = dp

    # SP+ detail (offense/defense sub-ratings) re-read from the Phase-2 cache.
    sp_rows = []
    for season in cfb_seasons:
        for r in client.get('/ratings/sp', {'year': season}) or []:
            off, de = r.get('offense') or {}, r.get('defense') or {}
            sp_rows.append({'season': r.get('year'), 'team': r.get('team'), 'sp_offense_rating': off.get('rating'),
                            'sp_pace': off.get('pace'), 'sp_run_rate': off.get('runRate'),
                            'sp_defense_rating': de.get('rating')})
    out['sp_detail'] = pd.DataFrame(sp_rows, columns=['season', 'team', 'sp_offense_rating', 'sp_pace',
                                                      'sp_run_rate', 'sp_defense_rating'])
    return out


def ingest_combine(session: Session, seasons, raw: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw is None:
        import nflreadpy as nfl
        raw = _cached_parquet(f'combine_{date.today().isoformat()}', lambda: nfl.load_combine().to_pandas(),
                              NFLVERSE_CACHE)
    raw = raw[raw['season'].isin(list(seasons))]
    comb = athletic.standardize_combine(raw)
    comb['source_id'] = 'nflverse_combine'
    comb['measurement_event'] = 'nfl_scouting_combine'
    comb['measurement_date_precision'] = 'season'
    comb['measurement_window'] = 'late February - early March of combine_season (exact test date not in source)'
    replace_table(session, NflCombineMeasurement, comb)
    return comb


def migrate_scheme_label_table(engine) -> str:
    """The v1 cfb_offensive_scheme_labels table (season_start/season_end/label) was
    created empty in Phase 3. Drop it ONLY if it is still empty so the v2 schema can be
    created; refuse (never silently discard) if anyone put rows in it."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if 'cfb_offensive_scheme_labels' not in insp.get_table_names():
        return 'absent'
    cols = {c['name'] for c in insp.get_columns('cfb_offensive_scheme_labels')}
    if 'season_start' not in cols:
        return 'current'
    with engine.begin() as conn:
        n = conn.execute(text('select count(*) from cfb_offensive_scheme_labels')).scalar()
        if n:
            raise RuntimeError(f'v1 cfb_offensive_scheme_labels has {n} rows; migrate them by hand before upgrading')
        conn.execute(text('drop table cfb_offensive_scheme_labels'))
    return 'migrated_empty_v1'


EMPTY_INFOBOXES = pd.DataFrame(columns=['team', 'season', 'exists', 'page_title', 'revid', 'rev_timestamp', 'off_scheme',
                                        'off_scheme_has_citation'])


def build_schemes(session: Session, client, pcs: pd.DataFrame, env: pd.DataFrame, prospects: pd.DataFrame,
                  fetch_wikipedia: bool = True, plays: pd.DataFrame | None = None) -> dict:
    """Offensive-scheme tiers for every relevant team-season, then per-QB features."""
    from src import ingest_cfbd_plays, ingest_wikipedia_schemes
    from src.db.schema import CfbTeamSeasonScheme
    from src.features import schemes
    tax = schemes.load_taxonomy()
    team_seasons = pcs[['team', 'season']].drop_duplicates() if len(pcs) else pd.DataFrame(columns=['team', 'season'])
    if fetch_wikipedia and len(team_seasons):
        mascots = pd.DataFrame(client.get('/teams'))[['school', 'mascot']]
        ib = ingest_wikipedia_schemes.fetch_team_season_infoboxes(team_seasons, mascots)
    else:
        ib = EMPTY_INFOBOXES.copy()
    labels = pd.concat([schemes.load_manual_labels(), schemes.wikipedia_labels(ib, tax)], ignore_index=True)
    schemes.validate_labels(labels, tax)
    replace_table(session, CfbOffensiveSchemeLabel, labels)
    plays = plays if plays is not None else ingest_cfbd_plays.load_plays()
    prof = schemes.play_profile(plays)
    sch = schemes.team_season_schemes(team_seasons, labels, ib, tax) if len(team_seasons) else pd.DataFrame()
    if len(sch):
        sch = sch.merge(prof, on=['season', 'team'], how='left') if len(prof) else sch
        for c in ('pbp_plays_nongarbage', 'pbp_reliable', 'pass_rate_nongarbage', 'standard_down_pass_rate',
                  'passing_down_pass_rate', 'early_down_pass_rate', 'standard_down_share', 'plays_per_game_nongarbage'):
            if c not in sch.columns:
                sch[c] = np.nan  # no play-by-play for these seasons: unobserved, not zero
        sch['taxonomy_version'] = tax['taxonomy_version']
        replace_table(session, CfbTeamSeasonScheme, sch)
    seasons = schemes.qb_scheme_seasons(pcs, sch, prof, env) if len(sch) else pd.DataFrame()
    dy = prospects.set_index('qb_draft_prospect_id')['draft_year']
    agg = schemes.qb_scheme_features(seasons, dy) if len(seasons) else pd.DataFrame(columns=['qb_draft_prospect_id'])
    return {'labels': labels, 'team_seasons': sch, 'qb_seasons': seasons, 'qb_features': agg, 'infoboxes': ib,
            'taxonomy_version': tax['taxonomy_version'], 'play_profile': prof}


# ---------------------------------------------------------------------------
# 2) feature construction
# ---------------------------------------------------------------------------

def prospect_frame(session: Session) -> pd.DataFrame:
    q = table_df(session, QbDraftProspect).rename(columns={'id': 'qb_draft_prospect_id'})
    m = table_df(session, CollegeNflPlayerMap)[['qb_draft_prospect_id', 'cfb_player_id', 'match_status',
                                                 'cfb_match_method', 'nfl_match_method']]
    q = q.merge(m, on='qb_draft_prospect_id', how='left')
    q.loc[q['match_status'] != 'matched', 'cfb_player_id'] = np.nan
    players = table_df(session, CfbPlayer)[['id', 'cfbd_player_id']].rename(columns={'id': 'cfb_player_id'})
    return q.merge(players, on='cfb_player_id', how='left')


def cfb_season_table(session: Session) -> pd.DataFrame:
    s = table_df(session, CfbPlayerSeason)
    d = table_df(session, CfbPlayerSeasonDerived).drop(columns=['id'])
    players = table_df(session, CfbPlayer)[['id', 'cfbd_player_id']].rename(columns={'id': 'player_id'})
    s = s.merge(d, left_on='id', right_on='cfb_player_season_id', how='left').merge(players, on='player_id', how='left')
    return s


SEASON_STAT_COLS = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions',
                    'rushing_attempts', 'rushing_yards', 'rushing_tds']


def box_season_totals(gp: pd.DataFrame, gr: pd.DataFrame) -> pd.DataFrame:
    """Season passing (and, where rushing box scores were fetched, rushing) totals
    per CFBD athlete, aggregated from box scores. Used ONLY where CFBD's season
    stats lack that player-season (mostly 2004-2008)."""
    p = gp[gp['cfbd_player_id'].notna()].copy()
    if p.empty:
        return pd.DataFrame(columns=['cfbd_player_id', 'season', 'team', 'player_name'] + SEASON_STAT_COLS)
    p = p.sort_values(['season', 'week'])
    agg = p.groupby(['cfbd_player_id', 'season']).agg(
        player_name=('player_name', lambda s: s.mode().iat[0]),
        team=('team', lambda s: '; '.join(dict.fromkeys(s))),
        conference=('conference', 'first'),
        **{c: (c, lambda s: s.sum(min_count=1)) for c in ['completions', 'attempts', 'passing_yards', 'passing_tds',
                                                          'interceptions']}).reset_index()
    if gr is not None and not gr.empty:
        r = gr.groupby(['cfbd_player_id', 'season']).agg(
            rushing_attempts=('carries', lambda s: s.sum(min_count=1)),
            rushing_yards=('rushing_yards', lambda s: s.sum(min_count=1)),
            rushing_tds=('rushing_tds', lambda s: s.sum(min_count=1))).reset_index()
        agg = agg.merge(r, on=['cfbd_player_id', 'season'], how='left')
        # a season with rushing box scores but no carries for this passer = 0 carries (observed);
        # a season WITHOUT rushing box scores stays NaN (unobserved)
        covered = agg['season'].isin(set(gr['season']))
        for c in ('rushing_attempts', 'rushing_yards', 'rushing_tds'):
            agg.loc[covered & agg[c].isna(), c] = 0.0
    for c in ('rushing_attempts', 'rushing_yards', 'rushing_tds'):
        if c not in agg.columns:
            agg[c] = np.nan
    return agg


def all_passer_seasons(cfb_all: pd.DataFrame, box: pd.DataFrame) -> pd.DataFrame:
    """Union of CFBD season stats and box-score season totals (box rows only for
    player-seasons CFBD's season stats lack), with derived rates recomputed the
    same way for both and provenance in `season_stats_source`."""
    ss = cfb_all.rename(columns={'school': 'team', 'player_id': 'cfb_player_id'})
    ss = ss[['cfbd_player_id', 'cfb_player_id', 'season', 'team', 'conference', 'height_inches', 'weight_lbs']
            + SEASON_STAT_COLS].copy()
    ss['season_stats_source'] = 'cfbd_season_stats'
    have = set(zip(ss['cfbd_player_id'], ss['season']))
    bx = box[[(i, s) not in have for i, s in zip(box['cfbd_player_id'], box['season'])]].copy()
    id_to_cfb = dict(zip(cfb_all['cfbd_player_id'], cfb_all['player_id']))
    bx['cfb_player_id'] = bx['cfbd_player_id'].map(id_to_cfb)
    bx['season_stats_source'] = 'box_score_aggregate'
    out = pd.concat([ss, bx.drop(columns=['player_name'])], ignore_index=True)
    out['completion_pct'] = metrics.completion_pct(out['completions'], out['attempts'])
    out['yards_per_attempt'] = metrics.yards_per_attempt(out['passing_yards'], out['attempts'])
    out['td_pct'] = metrics.td_pct(out['passing_tds'], out['attempts'])
    out['int_pct'] = metrics.int_pct(out['interceptions'], out['attempts'])
    out['ncaa_passer_rating'] = metrics.ncaa_passer_rating(out['completions'], out['attempts'], out['passing_yards'],
                                                           out['passing_tds'], out['interceptions'])
    return out


def resolve_identities(prospects: pd.DataFrame, box: pd.DataFrame, draft_picks: pd.DataFrame) -> pd.DataFrame:
    """Phase-3 identity: which CFBD athlete id each prospect's college seasons
    belong to. Order of evidence:
      1. Phase-2 map (CFBD season-stat player; name/school or CFBD draft id);
      2. CFBD draft-pick collegeAthleteId found among box-score passers with a
         plausibly matching name;
      3. the SAME Phase-2 name+school matcher, run against box-score passers
         (needed for 2004-2008, where CFBD season stats are nearly empty).
    Prospects left ambiguous by Phase 2 (name vs id conflict) are not re-matched."""
    from src.match_prospects import CfbCandidate, _plausibly_same_person, match_prospect_to_cfb, normalize_school
    rows = []
    box_names = box.groupby('cfbd_player_id')['player_name'].first().to_dict() if len(box) else {}
    cands = [CfbCandidate(cfb_player_id=int(pid), name=g['player_name'].iat[0],
                          schools={normalize_school(t) for ts in g['team'] for t in str(ts).split('; ')},
                          seasons=set(g['season'].astype(int)))
             for pid, g in box.groupby('cfbd_player_id')] if len(box) else []
    stable = {}
    if len(draft_picks):
        stable = {(int(r.year), int(r.overall)): A.id_str(r.college_athlete_id)
                  for r in draft_picks.itertuples() if A.id_str(r.college_athlete_id)}
    for _, p in prospects.iterrows():
        rec = {'qb_draft_prospect_id': p['qb_draft_prospect_id'], 'cfbd_player_id': None, 'identity_source': None,
               'identity_notes': None}
        if p.get('match_status') == 'matched' and pd.notna(p.get('cfbd_player_id')):
            rec.update(cfbd_player_id=A.id_str(p['cfbd_player_id']), identity_source=f"phase2_map:{p['cfb_match_method']}")
        elif p.get('match_status') == 'ambiguous':
            rec['identity_notes'] = 'left ambiguous by Phase-2 matching; not re-matched'
        else:
            sid = stable.get((int(p['draft_year']), int(p['pick'])))
            if sid and sid in box_names and _plausibly_same_person(p['player_name'], box_names[sid]):
                rec.update(cfbd_player_id=sid, identity_source='cfbd_draft_athlete_id_box_scores')
            else:
                res = match_prospect_to_cfb(p['player_name'], p['college'], int(p['draft_year']), cands)
                if res.method in ('exact_name_school', 'fuzzy_name_school'):
                    rec.update(cfbd_player_id=str(res.cfb_player_id), identity_source=f'box_score_name_school:{res.method}',
                               identity_notes=None if res.confidence == 1.0 else f'similarity {res.confidence:.3f}')
                else:
                    rec['identity_notes'] = f'box-score match {res.method}: {res.notes}'
        rows.append(rec)
    return pd.DataFrame(rows)


BOX_MIN_TEAM_COVERAGE = 0.8
PER_GAME_COLS = ['box_attempts', 'games_with_pass_attempt', 'games_as_leading_passer', 'opp_adj_attempt_coverage',
                 'opp_adj_ypa', 'opp_adj_completion_pct', 'opp_adj_td_pct', 'opp_adj_int_pct', 'sos_pass_defense',
                 'vs_top_def_games', 'vs_top_def_attempts', 'vs_top_def_ypa', 'vs_top_def_completion_pct',
                 'vs_top_def_rating', 'opp_def_sp_rating_avg']


def team_box_coverage(games: pd.DataFrame, box_games: pd.DataFrame) -> pd.DataFrame:
    """Share of each team's completed games that have a passing box score."""
    tgp = environment.team_games_played(games) if not games.empty else pd.DataFrame(columns=['season', 'team', 'team_games_played'])
    if box_games is None or box_games.empty:
        return tgp.assign(team_box_score_coverage=np.nan)[['season', 'team', 'team_box_score_coverage']]
    boxed = box_games.groupby(['season', 'team'])['game_id'].nunique().rename('boxed').reset_index()
    out = tgp.merge(boxed, on=['season', 'team'], how='left')
    out['team_box_score_coverage'] = out['boxed'].fillna(0) / out['team_games_played']
    return out[['season', 'team', 'team_box_score_coverage']]


def build_prospect_college_seasons(prospects, passer_seasons, qb_season_games, conf_rel, team_stats, games,
                                   sp_ctx, box_games: pd.DataFrame | None = None) -> pd.DataFrame:
    linked = prospects.dropna(subset=['cfbd_player_id'])
    s = passer_seasons.merge(linked[['qb_draft_prospect_id', 'cfbd_player_id', 'identity_source', 'draft_year']],
                             on='cfbd_player_id', how='inner')
    s = s[s['season'] < s['draft_year']].copy()  # pre-draft seasons only
    s['seasons_before_draft'] = s['draft_year'] - s['season']
    s['is_final_college_season'] = s['season'] == s.groupby('qb_draft_prospect_id')['season'].transform('max')
    s = s.merge(qb_season_games, on=['cfbd_player_id', 'season'], how='left')
    s = s.merge(conf_rel, on=['cfbd_player_id', 'season'], how='left')
    tgp = environment.team_games_played(games) if not games.empty else pd.DataFrame(columns=['season', 'team', 'team_games_played'])
    s = s.merge(tgp, on=['season', 'team'], how='left')
    ends = environment.season_end_dates(games) if not games.empty else pd.DataFrame(columns=['season', 'team', 'season_end_date'])
    s = s.merge(ends, on=['season', 'team'], how='left')
    s['available_as_of'] = s['season_end_date'].fillna(pd.to_datetime(s['season'].astype(int).astype(str) + '-01-31')
                                                       + pd.DateOffset(years=1))
    if not team_stats.empty:
        s = s.merge(team_stats[['season', 'team', 'rushing_attempts']].rename(columns={'rushing_attempts': 'team_rush_att'}),
                    on=['season', 'team'], how='left')
        s['qb_rush_share'] = np.where(s['team_rush_att'] > 0, s['rushing_attempts'] / s['team_rush_att'], np.nan)
    else:
        s['qb_rush_share'] = np.nan
    s = s.merge(sp_ctx.rename(columns={'school': 'team', 'strength_of_schedule': 'sp_strength_of_schedule'})
                [['season', 'team', 'sp_strength_of_schedule']], on=['season', 'team'], how='left')
    # Box scores exist only for games involving an FBS team, so an FCS team's
    # season is mostly missing from them. Where under BOX_MIN_TEAM_COVERAGE of a
    # team's games are box-scored: box-derived season totals are dropped and all
    # per-game measures are blanked (never reported from a 1-2 game sample).
    s = s.merge(team_box_coverage(games, box_games), on=['season', 'team'], how='left')
    partial = s['team_box_score_coverage'] < BOX_MIN_TEAM_COVERAGE
    s = s[~(partial & s['season_stats_source'].eq('box_score_aggregate'))].copy()
    partial = s['team_box_score_coverage'] < BOX_MIN_TEAM_COVERAGE
    for c in PER_GAME_COLS:
        if c in s.columns:
            s.loc[partial, c] = np.nan
    s['is_final_college_season'] = s['season'] == s.groupby('qb_draft_prospect_id')['season'].transform('max')
    s['participation_rate'] = np.where(s['team_games_played'] > 0,
                                       s['games_with_pass_attempt'] / s['team_games_played'], np.nan)
    s['box_season_attempt_ratio'] = np.where(s['attempts'] > 0, s['box_attempts'] / s['attempts'], np.nan)
    s['feature_version'] = registry.FEATURE_VERSION
    return s


def season_rating_medians(passer_seasons: pd.DataFrame) -> pd.Series:
    q = passer_seasons[passer_seasons['attempts'] >= development.PRODUCTIVE_MIN_ATTEMPTS]
    return q.groupby('season')['ncaa_passer_rating'].median()


def career_game_aggregates(pcs: pd.DataFrame, qb_games: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Career attempt-weighted opponent-adjusted rates and pooled top-defense splits."""
    rows = []
    for pid, g in pcs.groupby('qb_draft_prospect_id'):
        rec = {'qb_draft_prospect_id': pid}
        w = g['box_attempts'] * g['opp_adj_attempt_coverage']
        for c in ('opp_adj_ypa', 'opp_adj_completion_pct', 'opp_adj_td_pct', 'opp_adj_int_pct', 'sos_pass_defense'):
            m = g[c].notna() & (w > 0)
            rec[f'career_{c}'] = float(np.average(g.loc[m, c], weights=w[m])) if m.any() else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    if qb_games.empty or out.empty:
        return out
    keyed = qb_games.merge(id_map, on=['cfbd_player_id'], how='inner')
    keyed = keyed[keyed['season'] < keyed['draft_year']]
    top = keyed[keyed['vs_top_defense']]
    agg = top.groupby('qb_draft_prospect_id')[list(opponent_adjust.RATE_STATS)].sum().reset_index()
    agg['career_vs_top_def_attempts'] = agg['attempts']
    agg['career_vs_top_def_ypa'] = metrics.yards_per_attempt(agg['passing_yards'], agg['attempts'])
    agg['career_vs_top_def_rating'] = metrics.ncaa_passer_rating(agg['completions'], agg['attempts'],
                                                                 agg['passing_yards'], agg['passing_tds'],
                                                                 agg['interceptions'])
    out = out.merge(agg[['qb_draft_prospect_id', 'career_vs_top_def_attempts', 'career_vs_top_def_ypa',
                         'career_vs_top_def_rating']], on='qb_draft_prospect_id', how='left')
    has_box = set(pcs.loc[pcs['box_attempts'].fillna(0) > 0, 'qb_draft_prospect_id'])
    out.loc[out['qb_draft_prospect_id'].isin(has_box) & out['career_vs_top_def_attempts'].isna(),
            'career_vs_top_def_attempts'] = 0.0
    return out


FINAL_SEASON_COLS = {
    'attempts': 'final_attempts', 'completion_pct': 'final_completion_pct',
    'yards_per_attempt': 'final_yards_per_attempt', 'td_pct': 'final_td_pct', 'int_pct': 'final_int_pct',
    'ncaa_passer_rating': 'final_ncaa_passer_rating', 'rushing_attempts': 'final_rushing_attempts',
    'rushing_yards': 'final_rushing_yards', 'rushing_tds': 'final_rushing_tds', 'qb_rush_share': 'final_qb_rush_share',
    'participation_rate': 'final_participation_rate',
    'opp_adj_ypa': 'final_opp_adj_ypa', 'opp_adj_completion_pct': 'final_opp_adj_completion_pct',
    'opp_adj_td_pct': 'final_opp_adj_td_pct', 'opp_adj_int_pct': 'final_opp_adj_int_pct',
    'opp_adj_attempt_coverage': 'final_opp_adj_attempt_coverage', 'sos_pass_defense': 'final_sos_pass_defense',
    'sp_strength_of_schedule': 'final_sp_strength_of_schedule', 'opp_def_sp_rating_avg': 'final_opp_def_sp_rating_avg',
    'conf_rel_rating_z': 'final_conf_rel_rating_z', 'conf_rel_ypa_z': 'final_conf_rel_ypa_z',
    'conf_rel_rating_pctile': 'final_conf_rel_rating_pctile', 'vs_top_def_attempts': 'final_vs_top_def_attempts',
    'vs_top_def_ypa': 'final_vs_top_def_ypa', 'vs_top_def_rating': 'final_vs_top_def_rating',
    'height_inches': 'bio_college_height_inches', 'weight_lbs': 'bio_college_weight_lbs',
}
ENV_COLS = {
    'plays_per_game': 'env_plays_per_game', 'pass_rate': 'env_pass_rate', 'seconds_per_play': 'env_seconds_per_play',
    'sp_pace': 'env_sp_pace', 'sp_run_rate': 'env_sp_run_rate', 'off_ppa': 'env_off_ppa',
    'off_success_rate': 'env_off_success_rate', 'sp_offense_rating': 'env_sp_offense_rating',
    'sack_rate_allowed': 'ol_sack_rate_allowed', 'opp_adj_sack_rate': 'ol_opp_adj_sack_rate',
    'team_rush_ypc': 'ol_team_rush_ypc', 'off_stuff_rate': 'ol_off_stuff_rate', 'off_line_yards': 'ol_off_line_yards',
    'off_havoc_allowed': 'ol_off_havoc_allowed', 'team_receiving_yards': 'wr_team_receiving_yards',
    'top_target_reception_share': 'wr_top_target_reception_share', 'top3_reception_share': 'wr_top3_reception_share',
    'top3_receivers_avg_recruit_rating': 'wr_top3_avg_recruit_rating',
}
TALENT_COLS = {
    'returning_receiving_ppa_pct': 'wr_returning_receiving_ppa_pct', 'talent_composite': 'talent_composite',
    'talent_composite_pctile': 'talent_composite_pctile', 'recruiting_points_4yr_avg': 'talent_recruiting_points_4yr_avg',
    'recruiting_4yr_pctile': 'talent_recruiting_4yr_pctile', 'returning_ppa_pct': 'talent_returning_ppa_pct',
    'returning_usage': 'talent_returning_usage',
}


def assemble_features(prospects, pcs, dev, ages, career_games, env, talent, combine_join, draft_picks,
                      scheme_features: pd.DataFrame | None = None) -> pd.DataFrame:
    base = prospects[['qb_draft_prospect_id', 'draft_year', 'round', 'pick', 'nfl_team', 'player_name', 'college',
                      'match_status', 'cfb_match_method', 'cfbd_player_id', 'identity_source']].copy()
    final = pcs[pcs['is_final_college_season']].drop_duplicates('qb_draft_prospect_id') if len(pcs) else pcs
    fin = final.reindex(columns=['qb_draft_prospect_id', 'season', 'team', 'available_as_of'] + list(FINAL_SEASON_COLS)).rename(
        columns={**FINAL_SEASON_COLS, 'season': 'final_season', 'team': 'final_team',
                 'available_as_of': 'college_available_as_of'})
    f = base.merge(fin, on='qb_draft_prospect_id', how='left')
    if not dev.empty:
        f = f.merge(dev.drop(columns=['final_college_season'], errors='ignore'), on='qb_draft_prospect_id', how='left')
    f = f.merge(ages, on='qb_draft_prospect_id', how='left')
    if not career_games.empty:
        f = f.merge(career_games, on='qb_draft_prospect_id', how='left')
    if not env.empty:
        e = env[['season', 'team'] + list(ENV_COLS)].rename(columns={**ENV_COLS, 'season': 'final_season', 'team': 'final_team'})
        f = f.merge(e, on=['final_season', 'final_team'], how='left')
    if not talent.empty:
        t = talent[['season', 'team'] + [c for c in TALENT_COLS if c in talent.columns]].rename(
            columns={**TALENT_COLS, 'season': 'final_season', 'team': 'final_team'})
        f = f.merge(t, on=['final_season', 'final_team'], how='left')
    f = f.merge(combine_join.drop(columns=['combine_join_method']), on='qb_draft_prospect_id', how='left')
    if scheme_features is not None and len(scheme_features):
        f = f.merge(scheme_features, on='qb_draft_prospect_id', how='left')
    if not draft_picks.empty:
        sc = draft_picks[['year', 'overall', 'pre_draft_ranking', 'pre_draft_position_ranking', 'pre_draft_grade']].rename(
            columns={'year': 'draft_year', 'overall': 'pick', 'pre_draft_ranking': 'scout_pre_draft_ranking',
                     'pre_draft_position_ranking': 'scout_pre_draft_position_ranking',
                     'pre_draft_grade': 'scout_pre_draft_grade'})
        f = f.merge(sc, on=['draft_year', 'pick'], how='left')
    f['draft_round'] = f['round']
    f['draft_overall_pick'] = f['pick']
    for c in registry.FEATURES:
        if c not in f.columns:
            f[c] = np.nan
    f['split'] = f['draft_year'].map(lambda y: preprocess.assign_split(int(y)))
    return f


def feature_availability_dates(features: pd.DataFrame, draft_dates: dict) -> pd.DataFrame:
    """Per prospect, the date each availability rule resolves to."""
    dd = pd.to_datetime(features['draft_year'].astype(int).astype(str).map(draft_dates['draft_first_day']))
    combine_md = draft_dates['combine_known_by']['month_day']
    return pd.DataFrame({
        'qb_draft_prospect_id': features['qb_draft_prospect_id'],
        registry.AVAIL_SEASON_END: pd.to_datetime(features['college_available_as_of']),
        registry.AVAIL_PRE_DRAFT: dd - pd.Timedelta(days=1),
        registry.AVAIL_COMBINE: pd.to_datetime(features['draft_year'].astype(int).astype(str) + '-' + combine_md),
        registry.AVAIL_DRAFT_DAY: dd,
        'draft_date': dd,
    })


def long_feature_values(features: pd.DataFrame, draft_dates: dict) -> pd.DataFrame:
    avail = feature_availability_dates(features, draft_dates).set_index('qb_draft_prospect_id')
    rows = []
    for name, fdef in registry.FEATURES.items():
        col = features[['qb_draft_prospect_id', name]].dropna()
        if col.empty:
            continue
        a = avail.loc[col['qb_draft_prospect_id'], fdef.availability_rule].to_numpy()
        # age_* features are deterministic functions of birthdate and the
        # (pre-announced) draft date -> knowable before draft day
        if name.startswith('age_'):
            a = avail.loc[col['qb_draft_prospect_id'], registry.AVAIL_PRE_DRAFT].to_numpy()
        rows.append(pd.DataFrame({'qb_draft_prospect_id': col['qb_draft_prospect_id'].to_numpy(),
                                  'feature_name': name, 'feature_version': fdef.version,
                                  'value': col[name].astype(float).to_numpy(), 'available_as_of': a}))
    out = pd.concat(rows, ignore_index=True)
    # A value without a resolvable availability date cannot be proven leak-free,
    # so it is not stored (count reported in the leakage audit).
    out.attrs['dropped_no_availability_date'] = int(out['available_as_of'].isna().sum())
    return out.dropna(subset=['available_as_of'])


def leakage_audit(long_values: pd.DataFrame, features: pd.DataFrame, draft_dates: dict) -> dict:
    """Every pre-draft-scope value must be knowable strictly before draft day."""
    dd = feature_availability_dates(features, draft_dates).set_index('qb_draft_prospect_id')['draft_date']
    lv = long_values.copy()
    lv['draft_date'] = lv['qb_draft_prospect_id'].map(dd)
    lv['min_scope'] = lv['feature_name'].map(lambda n: registry.FEATURES[n].min_scope)
    pre = lv[lv['min_scope'] != 'post_draft']
    viol = pre[pd.to_datetime(pre['available_as_of']) >= pre['draft_date']]
    post = lv[lv['min_scope'] == 'post_draft']
    return {'pre_draft_values_checked': int(len(pre)), 'violations': int(len(viol)),
            'values_dropped_no_availability_date': int(long_values.attrs.get('dropped_no_availability_date', 0)),
            'violation_examples': viol.head(10)[['qb_draft_prospect_id', 'feature_name', 'available_as_of']]
            .astype(str).to_dict('records'),
            'post_draft_values': int(len(post)),
            'post_draft_values_on_or_after_draft_day': int((pd.to_datetime(post['available_as_of']) >= post['draft_date']).sum())}


# ---------------------------------------------------------------------------
# 3) NFL outcomes
# ---------------------------------------------------------------------------

NFLVERSE_CACHE = DATA / 'nflverse_cache'
def _with_retries(fn, attempts: int = 4, base_sleep: float = 5.0):
    import time
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # nflreadpy wraps HTTP errors in ConnectionError
            last = exc
            time.sleep(base_sleep * (i + 1))
    raise last


def _cached_parquet(name: str, loader, cache_dir: Path) -> pd.DataFrame:
    """Completed-season nflverse files never change, so they are cached locally
    (data/ is gitignored) and a rerun does not re-download them."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f'{name}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    df = _with_retries(loader)
    df.to_parquet(path, index=False)
    return df


def existing_nfl_season_totals(session: Session, seasons) -> pd.DataFrame:
    """NFL regular-season totals from the EXISTING Phase-1 tables (players +
    player_seasons_raw, nflverse 1999-2025). Reused rather than re-ingested.
    Limitation: Phase 1 kept only player-seasons with >= 1 pass attempt, so an
    appearance without a pass attempt is not counted (starts are added back
    from the schedule-based starting-QB data)."""
    from src.db.schema import Player, PlayerSeasonRaw
    q = (session.query(Player.source_player_id, PlayerSeasonRaw.season, PlayerSeasonRaw.games,
                       PlayerSeasonRaw.attempts, PlayerSeasonRaw.completions, PlayerSeasonRaw.passing_yards,
                       PlayerSeasonRaw.passing_tds, PlayerSeasonRaw.interceptions, PlayerSeasonRaw.sacks_suffered,
                       PlayerSeasonRaw.sack_yards_lost)
         .join(PlayerSeasonRaw, PlayerSeasonRaw.player_id == Player.id)
         .filter(Player.source_id == 'nflverse_player_stats', PlayerSeasonRaw.season_type == 'REG',
                 PlayerSeasonRaw.season.in_(list(seasons))))
    return pd.DataFrame(q.all(), columns=targets.SEASON_TOTAL_COLS)


def load_nfl_outcome_inputs(session: Session, seasons, cache_dir: Path | None = None) -> dict:
    import nflreadpy as nfl
    cache_dir = cache_dir or NFLVERSE_CACHE
    season_totals = existing_nfl_season_totals(session, seasons)
    # nflverse ships ONE all-seasons schedule file, so download it once and filter.
    lo, hi = min(seasons), max(seasons)
    sched = _cached_parquet(f'schedules_{lo}_{hi}', lambda: nfl.load_schedules(seasons=list(seasons)).to_pandas()
                            [['season', 'game_type', 'week', 'home_qb_id', 'away_qb_id', 'result']], cache_dir)
    sched = sched[sched['result'].notna()]
    sched['game_type'] = np.where(sched['game_type'].eq('REG'), 'REG', 'POST')
    st = pd.concat([sched[['season', 'game_type', 'home_qb_id']].rename(columns={'home_qb_id': 'gsis_id'}),
                    sched[['season', 'game_type', 'away_qb_id']].rename(columns={'away_qb_id': 'gsis_id'})])
    starts = st.dropna(subset=['gsis_id']).groupby(['gsis_id', 'season', 'game_type']).size() \
        .rename('games_started').reset_index()
    # roster status changes over time -> cached per pull date, and the date is reported
    players = _cached_parquet(f'players_status_{date.today().isoformat()}',
                              lambda: nfl.load_players().to_pandas()[['gsis_id', 'status']], cache_dir)
    players = players.dropna(subset=['gsis_id'])
    return {'season_totals': season_totals, 'starts': starts, 'roster_status': players}


# ---------------------------------------------------------------------------
# 4) orchestration
# ---------------------------------------------------------------------------

def main(db_url: str | None = None, cfb_seasons=None, recruit_years=None, draft_years=None,
         nfl_seasons=None, data_dir: Path | None = None, results_dir: Path | None = None,
         call_budget: int = DEFAULT_CALL_BUDGET, client: CFBDClient | None = None,
         nfl_inputs: dict | None = None, combine_seasons=None, combine_raw: pd.DataFrame | None = None,
         fetch_wikipedia: bool = True, plays: pd.DataFrame | None = None) -> dict:
    data_dir = Path(data_dir) if data_dir else DATA
    results_dir = Path(results_dir) if results_dir else RESULTS
    cfb_seasons = list(cfb_seasons) if cfb_seasons is not None else A.CFB_SEASONS
    recruit_years = list(recruit_years) if recruit_years is not None else A.RECRUIT_YEARS
    draft_years = list(draft_years) if draft_years is not None else A.DRAFT_YEARS
    nfl_seasons = list(nfl_seasons) if nfl_seasons is not None else list(range(2000, targets.OBS_END_SEASON + 1))
    combine_seasons = list(combine_seasons) if combine_seasons is not None else list(range(2000, 2027))
    draft_dates = load_draft_dates()

    engine = get_engine(db_url)
    scheme_migration = migrate_scheme_label_table(engine)
    engine = init_db(engine)
    session = get_session(engine)
    client = client or CFBDClient(max_network_calls=call_budget)
    try:
        seed_reference_tables(session)
        existing_before = _existing_counts(session)

        src = ingest_cfbd_sources(session, client, cfb_seasons, recruit_years, draft_years)
        combine = ingest_combine(session, combine_seasons, combine_raw)

        # identity: re-derive the map with the stable CFBD draft-pick id bridge,
        # then refresh the Phase-2 exports/report so they stay consistent with it
        prospect_pipeline.run_matching(session)
        prior = results_dir / 'prospect_coverage_report.json'
        unrecognized = json.loads(prior.read_text()).get('unrecognized_cfbd_stat_types', []) if prior.exists() else []
        prospect_pipeline.export_parquet(session, data_dir=data_dir)
        prospect_pipeline.write_report(prospect_pipeline.build_report(session, unrecognized), dest=prior)

        prospects = prospect_frame(session)
        cfb_all = cfb_season_table(session)
        games, gp = src['games'], src['game_passing']
        fbs = environment.fbs_teams(games) if not games.empty else pd.DataFrame(columns=['season', 'team'])
        box = box_season_totals(gp, src['game_rushing'])
        passer_seasons = all_passer_seasons(cfb_all, box)
        identity = resolve_identities(prospects, box, src['draft_picks'])
        prospects = prospects.drop(columns=['cfbd_player_id']).merge(identity, on='qb_draft_prospect_id', how='left')

        sp = src['sp_detail']
        sp_def = sp[['season', 'team', 'sp_defense_rating']].rename(columns={'sp_defense_rating': 'opp_def_sp_rating'})
        if not gp.empty:
            qb_games = opponent_adjust.qb_game_adjustments(gp, fbs, sp_def)
            tg = opponent_adjust.team_game_totals(gp)
            season_def = opponent_adjust.season_defense_table(tg, fbs)
            ids = set(prospects['cfbd_player_id'].dropna())
            qb_season_games = opponent_adjust.qb_season_from_games(qb_games[qb_games['cfbd_player_id'].isin(ids)], tg)
        else:
            qb_games, season_def = pd.DataFrame(), pd.DataFrame(columns=['season', 'defense', 'ypa', 'cmp_pct', 'def_pass_rank'])
            qb_season_games = pd.DataFrame(columns=['cfbd_player_id', 'season'])
        conf_rel = opponent_adjust.conference_relative(passer_seasons)

        ts = src['team_stats']
        receivers = environment.receiver_context(src['receiving'], src['recruits']) \
            if not src['receiving'].empty else pd.DataFrame(columns=['season', 'team'])
        env = environment.team_environment(ts, games, receivers, season_def, sp) if not ts.empty else pd.DataFrame()
        if not env.empty:
            env['feature_version'] = registry.FEATURE_VERSION
            replace_table(session, CfbTeamSeasonEnvironment, env)
        talent = environment.roster_talent(src['talent'], src['recruiting_teams'], src['returning'], cfb_seasons) \
            if not src['recruiting_teams'].empty else pd.DataFrame()
        if not talent.empty:
            talent['feature_version'] = registry.FEATURE_VERSION
            replace_table(session, CfbTeamRosterTalent, talent)

        sp_ctx = table_df(session, CfbTeamSeasonContext)
        pcs = build_prospect_college_seasons(prospects, passer_seasons, qb_season_games, conf_rel, ts, games, sp_ctx,
                                             box_games=gp)
        replace_table(session, ProspectCollegeSeason, pcs)

        dev = development.prospect_development(pcs, season_rating_medians(passer_seasons)) if not pcs.empty else pd.DataFrame()
        final_seasons = pcs[pcs['is_final_college_season']].set_index('qb_draft_prospect_id')['season'] \
            if not pcs.empty else pd.Series(dtype=float)
        ages = development.prospect_ages(prospects, draft_dates['draft_first_day'], final_seasons)
        id_map = prospects.dropna(subset=['cfbd_player_id'])[['qb_draft_prospect_id', 'cfbd_player_id', 'draft_year']]
        career_games = career_game_aggregates(pcs, qb_games, id_map) if not pcs.empty else pd.DataFrame()
        combine_join = athletic.join_combine_to_prospects(prospects, combine)
        scheme = build_schemes(session, client, pcs, env, prospects, fetch_wikipedia=fetch_wikipedia, plays=plays)

        features = assemble_features(prospects, pcs, dev, ages, career_games, env, talent, combine_join,
                                     src['draft_picks'], scheme['qb_features'])
        replace_table(session, FeatureDefinition, pd.DataFrame(registry.registry_records()))
        long_vals = long_feature_values(features, draft_dates)
        replace_table(session, ProspectFeatureValue, long_vals)
        audit = leakage_audit(long_vals, features, draft_dates)

        # NFL outcomes / targets / survival
        nfl_in = nfl_inputs or load_nfl_outcome_inputs(session, nfl_seasons)
        raw = table_df(session, NflDraftPickRaw)[['id', 'career_last_season', 'nfl_games_played_career']]
        pfr = prospects[['qb_draft_prospect_id', 'draft_pick_raw_id']].merge(
            raw.rename(columns={'id': 'draft_pick_raw_id', 'career_last_season': 'pfr_last_season',
                                'nfl_games_played_career': 'pfr_games'}), on='draft_pick_raw_id', how='left')
        nfl_seasons_df = targets.build_prospect_seasons(prospects, nfl_in['season_totals'], nfl_in['starts'])
        replace_table(session, NflProspectSeason, nfl_seasons_df)
        outcomes = targets.build_career_outcomes(prospects, nfl_seasons_df, nfl_in['roster_status'], pfr)
        replace_table(session, NflCareerOutcome, outcomes)
        surv = targets.build_survival_records(outcomes, nfl_seasons_df, prospects)
        replace_table(session, SurvivalRecord, surv)

        # similarity (pre-draft college features only, earlier classes only)
        sim_pool = features[features['split'] != 'pre_coverage']
        comps = similarity.historical_comps(sim_pool)
        comps['method'] = 'nan_euclidean_standardized_expanding_pool'
        comps['feature_version'] = registry.FEATURE_VERSION
        replace_table(session, ProspectSimilarity, comps)
        session.execute(delete(DraftValuePrediction))  # predictions come from future model runs
        session.commit()

        exports = export_datasets(features, outcomes, surv, comps, pcs, env, talent, nfl_seasons_df, data_dir)
        exports.update(export_scheme_datasets(scheme, features, env, src['team_stats'], data_dir))
        report = build_report(session, features, pcs, outcomes, surv, comps, audit, src, combine, combine_join,
                              scheme['labels'], existing_before, client, sim_pool, exports)
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / 'advanced_coverage_report.json').write_text(json.dumps(report, indent=2, default=str))
        scheme_report = build_scheme_report(scheme, features, outcomes, scheme_migration)
        (results_dir / 'offensive_scheme_coverage.json').write_text(json.dumps(scheme_report, indent=2, default=str))
        return report
    finally:
        session.close()


def row_counts(session: Session) -> dict:
    from sqlalchemy import func, select
    from src.db.schema import Base
    return {t.name: int(session.execute(select(func.count()).select_from(t)).scalar())
            for t in Base.metadata.sorted_tables}


def _existing_counts(session: Session) -> dict:
    from src.db.schema import Player, PlayerSeasonRaw, PlayerSeasonDerived, EraAdjustedStat
    models = [Player, PlayerSeasonRaw, PlayerSeasonDerived, EraAdjustedStat, NflDraftPickRaw, QbDraftProspect,
              CfbPlayer, CfbPlayerSeason, CfbTeamSeasonContext]
    return {m.__tablename__: session.query(m).count() for m in models}


def export_datasets(features, outcomes, surv, comps, pcs, env, talent, nfl_seasons_df, data_dir: Path) -> dict:
    out = data_dir / 'model_ready'
    out.mkdir(parents=True, exist_ok=True)
    meta = ['qb_draft_prospect_id', 'draft_year', 'player_name', 'college', 'final_season', 'identity_source', 'split']
    written = {}
    for scope in registry.SCOPES:
        cols = registry.features_for_scope(scope)
        registry.assert_scope_safe(cols, scope)
        df = features[meta + cols]
        df.to_parquet(out / f'features_{scope}.parquet', index=False)
        written[f'features_{scope}'] = len(df)
    tgt = outcomes.merge(features[['qb_draft_prospect_id', 'draft_year', 'split']], on='qb_draft_prospect_id', how='left')
    tgt.to_parquet(out / 'nfl_targets.parquet', index=False)
    surv.merge(features[['qb_draft_prospect_id', 'draft_year', 'split']], on='qb_draft_prospect_id', how='left') \
        .to_parquet(out / 'survival_records.parquet', index=False)
    comps.to_parquet(out / 'prospect_similarity.parquet', index=False)
    pcs.drop(columns=['season_end_date'], errors='ignore').to_parquet(out / 'prospect_college_seasons.parquet', index=False)
    if not env.empty:
        env.to_parquet(out / 'team_season_environment.parquet', index=False)
    if not talent.empty:
        talent.to_parquet(out / 'team_roster_talent.parquet', index=False)
    nfl_seasons_df.to_parquet(out / 'nfl_prospect_seasons.parquet', index=False)
    dv = features[['qb_draft_prospect_id', 'draft_year', 'draft_round', 'draft_overall_pick', 'nfl_team', 'player_name',
                   'college', 'split']].merge(
        outcomes[['qb_draft_prospect_id', 'window_5yr_complete', 'started_any_game', 'started_16plus_5yr',
                  'started_32plus_5yr', 'games_started_5yr', 'pass_yards_5yr', 'career_end_status']],
        on='qb_draft_prospect_id', how='left')
    dv['pred_pre_draft_college'] = np.nan   # filled by future, separately validated models
    dv['pred_post_draft'] = np.nan
    dv['outcome_observed_5yr'] = dv['window_5yr_complete'].fillna(False)
    dv.to_parquet(out / 'draft_value_base.parquet', index=False)
    pd.DataFrame(registry.registry_records()).to_csv(out / 'feature_dictionary.csv', index=False)
    written.update({'nfl_targets': len(tgt), 'survival_records': len(surv), 'prospect_similarity': len(comps),
                    'draft_value_base': len(dv)})
    return written


SCHEME_FEATURE_PREFIXES = ('scheme_',)
QB_ENV_CONTEXT = ['final_qb_rush_share', 'env_pass_rate', 'env_plays_per_game', 'env_seconds_per_play', 'env_sp_pace',
                  'env_off_ppa', 'env_off_success_rate', 'final_observed_season_gap']


def export_scheme_datasets(scheme: dict, features: pd.DataFrame, env: pd.DataFrame, team_stats: pd.DataFrame,
                           data_dir: Path) -> dict:
    """data/processed/: college_offensive_schemes (team-season), qb_scheme_features
    (per prospect, aggregated), qb_scheme_seasons (per prospect-season), labels."""
    out = data_dir / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    ts = scheme['team_seasons']
    if len(ts):
        q = env[['season', 'team', 'pass_rate', 'rush_rate', 'plays_per_game', 'seconds_per_play', 'sp_pace', 'sp_run_rate',
                 'off_ppa', 'off_success_rate']] if len(env) else pd.DataFrame(columns=['season', 'team'])
        if len(team_stats):
            t = team_stats[['season', 'team']].copy()
            t['team_pass_ypa'] = np.where(team_stats['pass_attempts'] > 0,
                                          team_stats['net_passing_yards'] / team_stats['pass_attempts'], np.nan)
            t['off_explosiveness'] = team_stats.get('off_explosiveness')
            t['off_passing_ppa'] = team_stats.get('off_passing_ppa')
            q = q.merge(t, on=['season', 'team'], how='outer')
        ts = ts.merge(q, on=['season', 'team'], how='left')
    ts.to_parquet(out / 'college_offensive_schemes.parquet', index=False)
    scheme_cols = [c for c in features.columns if c.startswith(SCHEME_FEATURE_PREFIXES)]
    meta = ['qb_draft_prospect_id', 'draft_year', 'player_name', 'college', 'final_season', 'final_team', 'split']
    qf = features[[c for c in meta + scheme_cols + QB_ENV_CONTEXT if c in features.columns]]
    qf.to_parquet(out / 'qb_scheme_features.parquet', index=False)
    scheme['qb_seasons'].to_parquet(out / 'qb_scheme_seasons.parquet', index=False)
    scheme['labels'].to_parquet(out / 'offensive_scheme_labels_all.parquet', index=False)
    return {'college_offensive_schemes': len(ts), 'qb_scheme_features': len(qf),
            'qb_scheme_seasons': len(scheme['qb_seasons']), 'offensive_scheme_labels_all': len(scheme['labels'])}


def build_scheme_report(scheme: dict, features: pd.DataFrame, outcomes: pd.DataFrame, migration: str) -> dict:
    ts, labels, qs = scheme['team_seasons'], scheme['labels'], scheme['qb_seasons']
    rep: dict = {'generated_on': str(date.today()), 'taxonomy_version': scheme['taxonomy_version'],
                 'feature_version': registry.FEATURE_VERSION, 'label_table_migration': migration}
    if not len(ts):
        return rep
    ts = ts.copy()
    ts['decade'] = (ts['season'] // 10 * 10).astype(int).astype(str) + 's'
    rep['team_seasons'] = {
        'relevant': int(len(ts)),
        'manual_research_status': ts['manual_research_status'].value_counts().astype(int).to_dict(),
        'reported_label_status': ts['reported_label_status'].value_counts().astype(int).to_dict(),
        'with_verified_family': int(ts['verified_offensive_family'].notna().sum()),
        'with_any_verified_label': int(ts[[c for c in ts.columns if c.startswith('verified_') and c != 'verified_spread_family']]
                                       .notna().any(axis=1).sum()),
        'with_reported_family': int(ts['reported_offensive_family'].notna().sum()),
        'with_pbp_profile': int(ts['standard_down_pass_rate'].notna().sum()) if 'standard_down_pass_rate' in ts else 0,
    }
    by_dec = ts.groupby('decade').agg(team_seasons=('team', 'size'),
                                      verified_family=('verified_offensive_family', lambda s: int(s.notna().sum())),
                                      reported_family=('reported_offensive_family', lambda s: int(s.notna().sum())),
                                      pbp_profile=('standard_down_pass_rate', lambda s: int(s.notna().sum())),
                                      env_pass_rate=('pass_rate_nongarbage', lambda s: int(s.notna().sum())))
    rep['coverage_by_decade'] = by_dec.reset_index().to_dict('records')
    fam_counts: dict = {}
    for v in ts['reported_offensive_family'].dropna():
        for f in v.split('|'):
            fam_counts[f] = fam_counts.get(f, 0) + 1
    rep['reported_family_counts'] = dict(sorted(fam_counts.items(), key=lambda x: -x[1]))
    unm = ts[ts['reported_label_status'] == 'unmapped_text']['wikipedia_off_scheme_raw'].value_counts()
    rep['unmapped_reported_text'] = unm.astype(int).to_dict()
    manual = labels[labels['annotation_method'] == 'manual_source_review_v1']
    rep['manual'] = {
        'rows': int(len(manual)),
        'status_counts': manual['verification_status'].value_counts().astype(int).to_dict(),
        'team_seasons_researched': int(manual[['team', 'season']].drop_duplicates().shape[0]),
        'rows_detail': manual.fillna('').to_dict('records'),
    }
    rep['wikipedia'] = {
        'rows': int((labels['annotation_method'] == 'wikipedia_infobox_automated_v1').sum()),
        'pages_found': int(scheme['infoboxes']['exists'].fillna(False).astype(bool).sum()),
        'pages_with_field': int(scheme['infoboxes']['off_scheme'].notna().sum()),
        'fields_with_inline_citation': int(scheme['infoboxes']['off_scheme_has_citation'].fillna(False).astype(bool).sum()),
        'revision_dates': [str(scheme['infoboxes']['rev_timestamp'].min())[:10],
                           str(scheme['infoboxes']['rev_timestamp'].max())[:10]] if len(scheme['infoboxes']) else None,
    }
    both = ts[ts['verified_offensive_family'].notna()]
    rep['verified_vs_reported'] = both[['team', 'season', 'verified_offensive_family', 'reported_offensive_family',
                                        'wikipedia_off_scheme_raw']].fillna('').to_dict('records')
    mu = features[features['split'] != 'pre_coverage']

    def pct(mask, base):
        return round(100.0 * float(mask.sum()) / len(base), 1) if len(base) else None
    usable = {}
    for name, frame in (('all_drafted_qbs', features), ('model_universe_2005plus', mu)):
        usable[name] = {
            'n': int(len(frame)),
            'with_college_seasons': pct(frame['final_attempts'].notna(), frame),
            'verified_final_family': pct(frame['scheme_ver_final_known'] == 1, frame),
            'reported_final_family': pct(frame['scheme_rep_final_known'] == 1, frame),
            'any_reported_career_label': pct(frame['scheme_rep_career_label_coverage'] > 0, frame),
            'pbp_final_season_profile': pct(frame['scheme_q_final_standard_down_pass_rate'].notna(), frame),
            'env_pass_rate_final': pct(frame['env_pass_rate'].notna(), frame),
            'any_scheme_feature': pct((frame['scheme_rep_final_known'] == 1)
                                      | frame['scheme_q_final_standard_down_pass_rate'].notna()
                                      | (frame['scheme_ver_final_known'] == 1), frame),
        }
    rep['usable_by_qb'] = usable
    scheme_cols = [n for n, f in registry.FEATURES.items() if f.group.startswith('scheme_')]
    rep['feature_coverage_model_universe_pct'] = {c: round(100.0 * float(mu[c].notna().mean()), 1) for c in scheme_cols}
    tq = ['pass_rate_nongarbage', 'standard_down_pass_rate', 'passing_down_pass_rate', 'early_down_pass_rate',
          'plays_per_game_nongarbage']
    rep['team_season_quant_coverage_pct'] = {c: round(100.0 * float(ts[c].notna().mean()), 1) for c in tq if c in ts}
    rep['qb_seasons'] = {'rows': int(len(qs)), 'transfer_qbs_multi_school': int((qs.groupby('qb_draft_prospect_id')['team']
                                                                                 .nunique() > 1).sum()),
                         'final_observed_gap_positive': int((features['final_observed_season_gap'] > 0).sum())}
    rep['leakage'] = {'qb_scheme_seasons_on_or_after_draft_year': int((qs['seasons_before_draft'] < 1).sum())}
    tj = outcomes[['qb_draft_prospect_id', 'started_16plus_5yr']].merge(
        mu[['qb_draft_prospect_id', 'scheme_rep_final_known', 'scheme_ver_final_known', 'split']], on='qb_draft_prospect_id')
    tj = tj[tj['split'].isin(['train', 'validation', 'test']) & tj['started_16plus_5yr'].notna()]
    rep['missingness_vs_outcome'] = {
        'reported_known_positive_rate': round(float(tj.loc[tj['scheme_rep_final_known'] == 1, 'started_16plus_5yr'].astype(float).mean()), 3),
        'reported_unknown_positive_rate': round(float(tj.loc[tj['scheme_rep_final_known'] != 1, 'started_16plus_5yr'].astype(float).mean()), 3),
        'n_known': int((tj['scheme_rep_final_known'] == 1).sum()), 'n_unknown': int((tj['scheme_rep_final_known'] != 1).sum()),
    }
    return rep


def _pct(n, d):
    return round(100.0 * n / d, 1) if d else None


def build_report(session, features, pcs, outcomes, surv, comps, audit, src, combine, combine_join, labels,
                 existing_before, client, sim_pool, exports) -> dict:
    model_universe = features[features['split'] != 'pre_coverage']
    rep: dict = {'feature_version': registry.FEATURE_VERSION, 'target_version': targets.TARGET_VERSION,
                 'survival_version': targets.SURVIVAL_VERSION, 'observation_end_season': targets.OBS_END_SEASON,
                 'generated_on': str(date.today())}
    rep['cfbd_calls'] = {'network_calls_this_run': client.network_calls, 'cache_hits_this_run': client.cache_hits,
                         'per_run_budget': client.max_network_calls}
    rep['prospects'] = {
        'total': int(len(features)),
        'draft_classes': [int(features['draft_year'].min()), int(features['draft_year'].max())],
        'by_class': features.groupby('draft_year').size().astype(int).to_dict(),
        'by_split': features['split'].value_counts().astype(int).to_dict(),
    }
    m = table_df(session, CollegeNflPlayerMap)
    rep['identity'] = {
        'match_status': m['match_status'].value_counts().astype(int).to_dict(),
        'cfb_match_method': m['cfb_match_method'].value_counts().astype(int).to_dict(),
        'confirmed_by_cfbd_draft_id': int(m['notes'].fillna('').str.contains('confirmed by cfbd_draft_athlete_id').sum()),
        'conflicts_name_vs_cfbd_id': int(m['notes'].fillna('').str.startswith('conflict').sum()),
        'cfbd_id_without_stat_rows': int(m['notes'].fillna('').str.contains('has no CFBD season-stat rows').sum()),
        'matched_in_model_universe': int((model_universe['match_status'] == 'matched').sum()),
        'model_universe_size': int(len(model_universe)),
        'phase3_identity_source': features['identity_source'].fillna('unresolved').value_counts().astype(int).to_dict(),
        'phase3_linked_in_model_universe': int(model_universe['identity_source'].notna().sum()),
    }
    # college stats coverage
    has_cfb = pcs.groupby('qb_draft_prospect_id').size()
    rep['college_stats'] = {
        'prospects_with_any_pre_draft_cfbd_season': int(len(has_cfb)),
        'prospect_college_season_rows': int(len(pcs)),
        'season_stats_source': pcs['season_stats_source'].value_counts().astype(int).to_dict() if len(pcs) else {},
        'box_rushing_seasons': A.BOX_RUSHING_SEASONS,
        'by_class_pct_with_college_stats': (features.assign(h=features['final_attempts'].notna())
                                            .groupby('draft_year')['h'].mean().mul(100).round(1).to_dict()),
        'career_possibly_truncated': int(features['career_possibly_truncated'].fillna(0).sum()),
        'transferred': int(features['transferred'].fillna(0).sum()),
        'box_score_attempt_ratio_median': float(pcs['box_season_attempt_ratio'].median()) if len(pcs) else None,
        'box_score_attempt_ratio_p10_p90': [float(pcs['box_season_attempt_ratio'].quantile(q)) for q in (0.1, 0.9)] if len(pcs) else None,
    }
    gp = src['game_passing']
    rep['box_scores'] = {
        'rows': int(len(gp)), 'games': int(gp['game_id'].nunique()) if len(gp) else 0,
        'seasons': sorted(int(s) for s in gp['season'].unique()) if len(gp) else [],
        'team_pseudo_rows': int(gp['cfbd_player_id'].isna().sum()) if len(gp) else 0,
    }
    q = pcs[pcs['attempts'].fillna(0) >= 100]
    rep['opponent_adjusted'] = {
        'prospect_seasons_100plus_att': int(len(q)),
        'with_opp_adj_ypa': int(q['opp_adj_ypa'].notna().sum()),
        'mean_attempt_coverage': float(q['opp_adj_attempt_coverage'].mean()) if len(q) else None,
        'by_season_pct_with_opp_adj': (q.assign(h=q['opp_adj_ypa'].notna()).groupby('season')['h'].mean()
                                       .mul(100).round(1).to_dict()),
        'by_season_pct_with_sp_opp_def': (q.assign(h=q['opp_def_sp_rating_avg'].notna()).groupby('season')['h'].mean()
                                          .mul(100).round(1).to_dict()) if 'opp_def_sp_rating_avg' in q else {},
        'prospects_final_season_opp_adj': int(features['final_opp_adj_ypa'].notna().sum()),
    }
    ts = src['team_stats']
    if not ts.empty:
        rep['team_stats'] = {
            'rows': int(len(ts)),
            'unmapped_stat_names': src['unmapped_team_stats'],
            'by_season': {int(s): {'teams': int(len(g)), 'sacks_allowed_available': int(g['sacks_allowed'].notna().sum()),
                                   'advanced_reliable': int(g['advanced_reliable'].sum()),
                                   'havoc_available': int(g['off_havoc_allowed'].notna().sum())}
                          for s, g in ts.groupby('season')},
        }
    tal = src['talent']
    rep['team_talent'] = {
        'talent_composite_seasons': sorted(int(s) for s in tal['season'].dropna().unique()) if len(tal) else [],
        'recruiting_team_class_years': sorted(int(s) for s in src['recruiting_teams']['class_year'].dropna().unique())
        if len(src['recruiting_teams']) else [],
        'returning_production_seasons': sorted(int(s) for s in src['returning']['season'].dropna().unique())
        if len(src['returning']) else [],
        'recruits_rows': int(len(src['recruits'])),
        'recruits_with_athlete_id_pct': _pct(src['recruits']['athlete_id'].notna().sum(), len(src['recruits']))
        if len(src['recruits']) else None,
        'prospects_with_talent_composite': int(features['talent_composite'].notna().sum()),
        'prospects_with_recruiting_4yr': int(features['talent_recruiting_points_4yr_avg'].notna().sum()),
    }
    rep['receivers'] = {
        'receiving_rows': int(len(src['receiving'])),
        'prospects_with_top3_recruit_rating': int(features['wr_top3_avg_recruit_rating'].notna().sum()),
        'prospects_with_concentration': int(features['wr_top_target_reception_share'].notna().sum()),
    }
    rep['offensive_scheme'] = {
        'quantitative_profiles_team_seasons': int(len(table_df(session, CfbTeamSeasonEnvironment))),
        'prospects_with_pass_rate': int(features['env_pass_rate'].notna().sum()),
        'prospects_with_sp_pace': int(features['env_sp_pace'].notna().sum()),
        'qualitative_label_rows': int((labels['verification_status'] == 'verified').sum()),
        'reported_label_rows_low_confidence': int((labels['verification_status'] == 'reported').sum()),
    }
    found = combine_join['combine_record_found'] == 1
    rep['combine'] = {
        'combine_rows_all_positions': int(len(combine)),
        'prospects_with_combine_record': int(found.sum()),
        'join_methods': combine_join['combine_join_method'].value_counts().astype(int).to_dict(),
        'drill_participation_pct_given_record': {d: _pct(combine_join.loc[found, f'combine_{d}'].notna().sum(), found.sum())
                                                 for d in athletic.DRILLS},
        'by_class_pct_with_record': (features.groupby('draft_year')['combine_record_found']
                                     .apply(lambda s: s.fillna(0).mean() * 100).round(1).to_dict()),
        'pro_day_rows': 0,
    }
    rep['injury'] = {'rows': 0, 'status': 'not_ingested',
                     'reason': 'No verified, legally bulk-accessible historical college injury source (see config/sources.json: college_injury_availability).'}
    rep['nfl_outcomes'] = {
        'career_end_status': outcomes['career_end_status'].value_counts().astype(int).to_dict(),
        'window_5yr_complete': int(outcomes['window_5yr_complete'].sum()),
        'appeared_any_game': int(outcomes['appeared_any_game'].fillna(False).sum()),
        'nflverse_status_RET': int((outcomes['nflverse_roster_status'] == 'RET').sum()),
    }
    tcols = ['started_any_game', 'started_1plus_5yr', 'started_16plus_5yr', 'started_32plus_5yr', 'started_64plus_5yr',
             'games_started_5yr', 'pass_yards_5yr', 'pass_tds_5yr', 'any_a_5yr', 'passer_rating_5yr',
             'seasons_to_final_appearance', 'seasons_to_final_start', 'career_span_seasons']
    tj = outcomes.merge(features[['qb_draft_prospect_id', 'split', 'final_attempts']], on='qb_draft_prospect_id')
    usable = {}
    for t in tcols:
        nn = tj[tj[t].notna()]
        modelable = nn[nn['split'].isin(['train', 'validation', 'test']) & nn['final_attempts'].notna()]
        entry = {'non_null_all_classes': int(len(nn)),
                 'model_ready_with_college_features': {s: int((modelable['split'] == s).sum())
                                                       for s in ('train', 'validation', 'test')}}
        if t.startswith('started_') and len(nn):
            entry['positive_rate_non_null'] = round(float(nn[t].astype(float).mean()), 3)
        usable[t] = entry
    rep['target_usable_observations'] = usable
    rep['survival'] = {ev: {'n': int(len(g)), 'events': int(g['event_observed'].sum()),
                            'censored': int((~g['event_observed'].astype(bool)).sum())}
                       for ev, g in surv.groupby('event_name')} if len(surv) else {}
    rep['similarity'] = {
        'prospects_with_comps': int(comps['qb_draft_prospect_id'].nunique()) if len(comps) else 0,
        'features': similarity.DEFAULT_SIMILARITY_FEATURES,
        'scaling_effect': similarity.scaling_effect(sim_pool).round(4).to_dict('records'),
    }
    key_feats = ['final_attempts', 'final_opp_adj_ypa', 'final_conf_rel_rating_z', 'dev_slope_rating', 'age_at_draft',
                 'env_pass_rate', 'ol_sack_rate_allowed', 'wr_top3_avg_recruit_rating', 'talent_composite',
                 'talent_recruiting_points_4yr_avg', 'combine_forty', 'combine_height_inches', 'scout_pre_draft_grade']
    rep['missingness_by_class_pct'] = (features.groupby('draft_year')[key_feats].apply(lambda d: d.isna().mean() * 100)
                                       .round(0).astype(int).to_dict(orient='index'))
    group_miss = {}
    for gname in sorted(set(f.group for f in registry.FEATURES.values())):
        cols = [n for n, f in registry.FEATURES.items() if f.group == gname]
        group_miss[gname] = round(float(model_universe[cols].isna().mean().mean() * 100), 1)
    rep['missingness_by_group_model_universe_pct'] = group_miss
    rep['leakage_audit'] = audit
    rep['existing_table_counts_before'] = existing_before
    rep['existing_table_counts_after'] = _existing_counts(session)
    rep['row_counts'] = row_counts(session)
    rep['exports'] = exports
    return rep


if __name__ == '__main__':
    r = main()
    print(json.dumps({k: r[k] for k in ('prospects', 'identity', 'cfbd_calls', 'leakage_audit')}, indent=2, default=str))
