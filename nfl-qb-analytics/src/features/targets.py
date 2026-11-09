"""NFL career outcomes, prediction targets and survival records.

Definitions (TARGET_VERSION 1.0.0):
  * NFL season index: 1 = the season of the draft year (rookie season).
  * Observation ends with the last COMPLETED regular season (OBS_END_SEASON).
  * Games started: regular-season games in which nflverse's schedule lists the
    player as his team's starting QB (home_qb_id/away_qb_id). Postseason starts
    are recorded separately and excluded from the window targets.
  * Appearance: a regular-season week with any recorded statistic for the
    player in nflverse weekly player stats.
  * Five-season window = seasons draft_year .. draft_year+4. It is complete only
    if draft_year + 4 <= OBS_END_SEASON. If incomplete, every window target is
    NULL (right-censored) -- never 0 -- because the career is not finished.
    Exception: a positive label already achieved inside a partial window is a
    known fact (e.g. 16 starts reached in year 2), so it is recorded as True.
  * Passing efficiency over the window is defined only with >= EFFICIENCY_MIN_ATTEMPTS.
  * Inferred career end: no appearance in each of the last INACTIVE_SEASONS_FOR_END
    observed seasons. This is an INFERENCE, not a retirement confirmation; QBs do
    return after a gap. "Official" confirmation uses nflverse roster status 'RET',
    which is sparsely populated, so most careers have no official end date.
  * Drafted QBs who never appeared are kept, with status never_appeared_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics

TARGET_VERSION = '1.0.0'
SURVIVAL_VERSION = '1.0.0'
OBS_END_SEASON = 2025
WINDOW_SEASONS = 5
EFFICIENCY_MIN_ATTEMPTS = 300
INACTIVE_SEASONS_FOR_END = 2
START_THRESHOLDS = (1, 16, 32, 64)


SEASON_TOTAL_COLS = ['gsis_id', 'season', 'games', 'attempts', 'completions', 'passing_yards', 'passing_tds',
                     'interceptions', 'sacks_suffered', 'sack_yards_lost']


def weekly_to_season_totals(weekly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate week-level nflverse rows (REG) into the season-total shape."""
    return (weekly.groupby(['gsis_id', 'season'])
            .agg(games=('week', 'nunique'), attempts=('attempts', 'sum'), completions=('completions', 'sum'),
                 passing_yards=('passing_yards', 'sum'), passing_tds=('passing_tds', 'sum'),
                 interceptions=('passing_interceptions', 'sum'), sacks_suffered=('sacks_suffered', 'sum'),
                 sack_yards_lost=('sack_yards_lost', 'sum'))
            .reset_index())


def build_prospect_seasons(prospects: pd.DataFrame, season_totals: pd.DataFrame, starts: pd.DataFrame,
                           obs_end: int = OBS_END_SEASON) -> pd.DataFrame:
    """Explicit (prospect x NFL season) grid with zeros for non-appearance.
    `prospects`: qb_draft_prospect_id, draft_year, gsis_id.
    `season_totals`: SEASON_TOTAL_COLS, regular season (in production: the
        existing Phase-1 player_seasons_raw table -- reused, not re-ingested).
    `starts`: gsis_id, season, game_type ('REG'/'POST'), games_started."""
    rows = []
    linked = prospects.dropna(subset=['gsis_id'])
    for _, p in linked.iterrows():
        for s in range(int(p['draft_year']), obs_end + 1):
            rows.append((p['qb_draft_prospect_id'], p['gsis_id'], s, s - int(p['draft_year']) + 1))
    grid = pd.DataFrame(rows, columns=['qb_draft_prospect_id', 'gsis_id', 'nfl_season', 'season_index'])
    if grid.empty:
        return grid
    w = season_totals[season_totals['gsis_id'].isin(set(linked['gsis_id']))]
    agg = (w.groupby(['gsis_id', 'season'], as_index=False)
           .agg(games_with_stats=('games', 'sum'), pass_attempts=('attempts', 'sum'),
                pass_completions=('completions', 'sum'), passing_yards=('passing_yards', 'sum'),
                passing_tds=('passing_tds', 'sum'), interceptions=('interceptions', 'sum'),
                sacks=('sacks_suffered', 'sum'), sack_yards=('sack_yards_lost', 'sum'))
           .rename(columns={'season': 'nfl_season'}))
    grid = grid.merge(agg, on=['gsis_id', 'nfl_season'], how='left')
    st = starts.pivot_table(index=['gsis_id', 'season'], columns='game_type', values='games_started',
                            aggfunc='sum', fill_value=0).reset_index().rename(columns={'season': 'nfl_season'})
    st.columns.name = None
    for col in ('REG', 'POST'):
        if col not in st.columns:
            st[col] = 0
    grid = grid.merge(st[['gsis_id', 'nfl_season', 'REG', 'POST']], on=['gsis_id', 'nfl_season'], how='left')
    grid = grid.rename(columns={'REG': 'games_started', 'POST': 'postseason_starts'})
    for c in ('games_with_stats', 'games_started', 'postseason_starts'):
        grid[c] = grid[c].fillna(0).astype(int)
    for c in ('pass_attempts', 'pass_completions', 'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards'):
        grid[c] = grid[c].fillna(0.0)
    # A start without a recorded stat row (e.g. injured on first series) still counts as an appearance.
    grid['games_with_stats'] = np.maximum(grid['games_with_stats'], grid['games_started'])
    return grid


def _window_bool(achieved: bool, complete: bool):
    if achieved:
        return True
    return False if complete else None


def build_career_outcomes(prospects: pd.DataFrame, seasons: pd.DataFrame, roster_status: pd.DataFrame,
                          pfr: pd.DataFrame, obs_end: int = OBS_END_SEASON) -> pd.DataFrame:
    """One target row per prospect (including drafted QBs never linked to an NFL
    stat record). `roster_status`: gsis_id, status. `pfr`: qb_draft_prospect_id,
    pfr_last_season, pfr_games (nflverse draft-board career fields, used only to
    tell 'never played' apart from 'played but not linkable')."""
    status_map = dict(zip(roster_status['gsis_id'], roster_status['status'])) if not roster_status.empty else {}
    pfr = pfr.set_index('qb_draft_prospect_id') if not pfr.empty else pfr
    by_pid = {k: g.sort_values('nfl_season') for k, g in seasons.groupby('qb_draft_prospect_id')} if not seasons.empty else {}
    out = []
    for _, p in prospects.iterrows():
        pid, dy = p['qb_draft_prospect_id'], int(p['draft_year'])
        seasons_observed = max(0, obs_end - dy + 1)
        complete = dy + WINDOW_SEASONS - 1 <= obs_end
        g = by_pid.get(pid)
        pfr_last = pfr['pfr_last_season'].get(pid) if len(pfr) else None
        pfr_games = pfr['pfr_games'].get(pid) if len(pfr) else None
        rec = {'qb_draft_prospect_id': pid, 'target_version': TARGET_VERSION, 'observation_end_season': obs_end,
               'seasons_observed': seasons_observed, 'window_5yr_complete': complete,
               'efficiency_min_attempts': EFFICIENCY_MIN_ATTEMPTS,
               'nflverse_roster_status': status_map.get(p.get('gsis_id')),
               'pfr_last_season': None if pd.isna(pfr_last) else int(pfr_last)}
        if seasons_observed == 0:
            rec.update({'appeared_any_game': False, 'games_started_observed': 0,
                        'career_end_status': 'not_yet_observed'})
            out.append(rec)
            continue
        if g is None:
            # Not linkable to nflverse stats. Only call it "never appeared" when the
            # draft board's own career record agrees (no games / no last season).
            never = (pfr_games is None or pd.isna(pfr_games) or pfr_games == 0) and (pfr_last is None or pd.isna(pfr_last))
            rec.update({'appeared_any_game': False if never else None, 'games_started_observed': 0})
            if never:
                rec['started_any_game'] = False if complete else None
                for t in START_THRESHOLDS:
                    rec[f'started_{t}plus_5yr'] = False if complete else None
                if complete:
                    rec.update({'games_started_5yr': 0.0, 'pass_yards_5yr': 0.0, 'pass_tds_5yr': 0.0,
                                'pass_attempts_5yr': 0.0})
                rec['career_end_status'] = 'never_appeared_window_complete' if complete else 'never_appeared_censored'
            else:
                rec['career_end_status'] = 'unlinked_nfl_record'
            out.append(rec)
            continue

        win = g[g['season_index'] <= WINDOW_SEASONS]
        gs_win = int(win['games_started'].sum())
        appeared = g[g['games_with_stats'] > 0]
        started = g[g['games_started'] > 0]
        rec['appeared_any_game'] = bool(len(appeared))
        rec['games_started_observed'] = int(g['games_started'].sum())
        rec['started_any_game'] = True if len(started) else (False if complete else None)
        for t in START_THRESHOLDS:
            rec[f'started_{t}plus_5yr'] = _window_bool(gs_win >= t, complete)
        if complete:
            s = win[['pass_attempts', 'pass_completions', 'passing_yards', 'passing_tds', 'interceptions',
                     'sacks', 'sack_yards']].sum()
            rec.update({'games_started_5yr': float(gs_win), 'pass_yards_5yr': float(s['passing_yards']),
                        'pass_tds_5yr': float(s['passing_tds']), 'pass_attempts_5yr': float(s['pass_attempts'])})
            if s['pass_attempts'] >= EFFICIENCY_MIN_ATTEMPTS:
                rec['any_a_5yr'] = float(metrics.adjusted_net_yards_per_attempt(
                    s['passing_yards'], s['passing_tds'], s['interceptions'], s['pass_attempts'],
                    s['sacks'], s['sack_yards']))
                rec['passer_rating_5yr'] = float(metrics.passer_rating(
                    s['pass_completions'], s['pass_attempts'], s['passing_yards'], s['passing_tds'],
                    s['interceptions']))
        if len(appeared):
            first, last = int(appeared['nfl_season'].min()), int(appeared['nfl_season'].max())
            rec.update({'first_appearance_season': first, 'last_appearance_season': last})
        if len(started):
            rec['last_start_season'] = int(started['nfl_season'].max())
        inferred_end = len(appeared) and rec['last_appearance_season'] <= obs_end - INACTIVE_SEASONS_FOR_END
        if not len(appeared):
            rec['career_end_status'] = 'never_appeared_window_complete' if complete else 'never_appeared_censored'
        elif inferred_end:
            rec['career_end_status'] = 'inferred_inactive'
            rec['seasons_to_final_appearance'] = rec['last_appearance_season'] - dy + 1
            # duration only for (inferred) completed careers; active careers are censored -> survival_records
            rec['career_span_seasons'] = rec['last_appearance_season'] - rec['first_appearance_season'] + 1
            if len(started):
                rec['seasons_to_final_start'] = rec['last_start_season'] - dy + 1
        else:
            rec['career_end_status'] = 'active_or_censored'
        out.append(rec)
    return pd.DataFrame(out).reindex(columns=OUTCOME_COLUMNS)


OUTCOME_COLUMNS = [
    'qb_draft_prospect_id', 'target_version', 'observation_end_season', 'seasons_observed', 'window_5yr_complete',
    'appeared_any_game', 'started_any_game', 'started_1plus_5yr', 'started_16plus_5yr', 'started_32plus_5yr',
    'started_64plus_5yr', 'games_started_5yr', 'pass_yards_5yr', 'pass_tds_5yr', 'pass_attempts_5yr', 'any_a_5yr',
    'passer_rating_5yr', 'efficiency_min_attempts', 'games_started_observed', 'first_appearance_season',
    'last_appearance_season', 'last_start_season', 'seasons_to_final_appearance', 'seasons_to_final_start',
    'career_span_seasons', 'career_end_status', 'nflverse_roster_status', 'pfr_last_season',
]


def build_survival_records(outcomes: pd.DataFrame, seasons: pd.DataFrame, prospects: pd.DataFrame,
                           obs_end: int = OBS_END_SEASON) -> pd.DataFrame:
    """Long-format survival rows. Durations are in NFL seasons (1 = rookie year).
    Rows are only produced for prospects whose NFL record is linkable (or who are
    confirmed never to have appeared, for the first-start event)."""
    dy = prospects.set_index('qb_draft_prospect_id')['draft_year'].astype(int)
    by_pid = {k: g for k, g in seasons.groupby('qb_draft_prospect_id')} if not seasons.empty else {}
    rows = []
    for _, o in outcomes.iterrows():
        pid = o['qb_draft_prospect_id']
        d = int(dy[pid])
        observed = int(o['seasons_observed'])
        if observed <= 0 or o['career_end_status'] in ('unlinked_nfl_record', 'not_yet_observed'):
            continue
        g = by_pid.get(pid)
        base = {'qb_draft_prospect_id': pid, 'definition_version': SURVIVAL_VERSION, 'last_observed_season': obs_end}
        # 1) time to first regular-season start (at risk: every linkable drafted QB)
        first_start = None if g is None else g.loc[g['games_started'] > 0, 'season_index'].min()
        if pd.notna(first_start):
            rows.append({**base, 'event_name': 'first_nfl_start', 'duration_seasons': float(first_start),
                         'event_observed': True, 'event_season': d + int(first_start) - 1,
                         'at_risk_population': 'all_linkable_drafted_qbs'})
        else:
            rows.append({**base, 'event_name': 'first_nfl_start', 'duration_seasons': float(observed),
                         'event_observed': False, 'censor_reason': f'no start observed through {obs_end}',
                         'at_risk_population': 'all_linkable_drafted_qbs'})
        if not o['appeared_any_game']:
            continue
        # 2) inferred final appearance (at risk: QBs who appeared at least once)
        ended = o['career_end_status'] == 'inferred_inactive'
        official = o['nflverse_roster_status'] == 'RET'
        last_app_idx = int(o['last_appearance_season']) - d + 1
        rows.append({**base, 'event_name': 'final_appearance_inferred',
                     'duration_seasons': float(last_app_idx if ended else observed), 'event_observed': bool(ended),
                     'event_season': int(o['last_appearance_season']) if ended else None,
                     'censor_reason': None if ended else f'appeared within last {INACTIVE_SEASONS_FOR_END} observed seasons',
                     'officially_confirmed': True if (ended and official) else None,
                     'at_risk_population': 'appeared_at_least_once'})
        # 3) official retirement status (sparse -> mostly censored, by design)
        rows.append({**base, 'event_name': 'career_end_official',
                     'duration_seasons': float(last_app_idx if official else observed), 'event_observed': bool(official),
                     'event_season': int(o['last_appearance_season']) if official else None,
                     'censor_reason': None if official else 'no official retirement status in nflverse roster data',
                     'officially_confirmed': bool(official), 'at_risk_population': 'appeared_at_least_once'})
        # 4) inferred final start (at risk: QBs who started at least once)
        if pd.notna(o.get('last_start_season')):
            last_start_idx = int(o['last_start_season']) - d + 1
            start_ended = int(o['last_start_season']) <= obs_end - INACTIVE_SEASONS_FOR_END
            rows.append({**base, 'event_name': 'final_start_inferred',
                         'duration_seasons': float(last_start_idx if start_ended else observed),
                         'event_observed': bool(start_ended),
                         'event_season': int(o['last_start_season']) if start_ended else None,
                         'censor_reason': None if start_ended else f'started within last {INACTIVE_SEASONS_FOR_END} observed seasons',
                         'at_risk_population': 'started_at_least_once'})
    return pd.DataFrame(rows)
