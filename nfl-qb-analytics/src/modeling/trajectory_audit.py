"""Career trajectory and longevity audit (Phase 5.5).

Run:  python -m src.modeling.trajectory_audit        (local data only)

Checks the QB x NFL-season table (data/modeling/nfl_qb_seasons_v5.0.0.parquet) and the survival frame:
  indexing       season_index 1 = the draft year; one row per QB per season; no gaps in the grid
  truncation     every QB has a row for every season from his draft through OBS_END_SEASON
  constraints    games started <= games played <= the season's game cap; no negative counts
  missing years  a season without an appearance followed by a later appearance is NOT a career end
  censoring      QBs who appeared in either of the last two observed seasons are right-censored
  reconciliation season sums vs the career totals on the nflverse/PFR draft board, for careers that ended
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.targets import INACTIVE_SEASONS_FOR_END, OBS_END_SEASON
from src.modeling import career_trajectory as CT
from src.modeling.reporting import md_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5_5'
SECTION = RESULTS / 'trajectory_audit_section.md'
RECONCILE = {'pass_attempts': 'career_pass_attempts', 'pass_completions': 'career_pass_completions',
             'pass_yards': 'career_pass_yards', 'pass_tds': 'career_pass_tds', 'interceptions': 'career_pass_ints',
             'rush_attempts': 'career_rush_attempts', 'rush_yards': 'career_rush_yards', 'rush_tds': 'career_rush_tds'}


def checks(seasons: pd.DataFrame, sv: pd.DataFrame) -> dict:
    s = seasons
    obs = s[s['status'] == 'observed']
    per_qb = s.groupby('qb_draft_prospect_id').agg(draft_year=('draft_year', 'first'), n=('season_index', 'size'),
                                                     min_i=('season_index', 'min'), max_i=('season_index', 'max'),
                                                     max_obs=('nfl_season', lambda x: x[s.loc[x.index, 'status'] == 'observed'].max()))
    expected_obs = OBS_END_SEASON - per_qb['draft_year'] + 1
    obs_rows = obs.groupby('qb_draft_prospect_id').size().reindex(per_qb.index, fill_value=0)
    unverified = s.groupby('qb_draft_prospect_id')['status'].apply(lambda x: (x == 'unverified').all())
    idx_ok = bool((s['nfl_season'] == s['draft_year'] + s['season_index'] - 1).all())
    dup = int(s.duplicated(['qb_draft_prospect_id', 'season_index']).sum())
    gaps = int((per_qb['n'] != per_qb['max_i'] - per_qb['min_i'] + 1).sum())
    trunc = int(((obs_rows < expected_obs) & ~unverified).sum())
    cons = {'starts_le_games_played': bool((obs['games_started'] <= obs['games_played']).all()),
            'games_le_cap': bool((obs['games_played'] <= obs['games_cap']).all()),
            'no_negative_counts': bool((obs[['games_played', 'games_started', 'pass_attempts', 'pass_completions', 'pass_tds',
                                             'interceptions', 'rush_attempts', 'rush_tds']] >= 0).all().all()),
            'completions_le_attempts': bool((obs['pass_completions'] <= obs['pass_attempts']).all()),
            'future_rows_have_no_values': bool(s.loc[s['status'] == 'future', 'games_played'].isna().all()),
            'unverified_rows_have_no_values': bool(s.loc[s['status'] == 'unverified', 'games_played'].isna().all())}
    # gap seasons
    o = obs.sort_values(['qb_draft_prospect_id', 'season_index'])
    gap_qbs = []
    for pid, g in o.groupby('qb_draft_prospect_id'):
        app = g['games_played'].to_numpy() > 0
        idx = np.flatnonzero(app)
        if len(idx) >= 2 and (np.diff(idx) > 1).any():
            gap_qbs.append(pid)
    svi = sv.set_index('qb_draft_prospect_id')
    gq = svi.loc[svi.index.intersection(gap_qbs)]
    last_app = o[o['games_played'] > 0].groupby('qb_draft_prospect_id')['nfl_season'].max()
    recent = last_app[last_app > OBS_END_SEASON - INACTIVE_SEASONS_FOR_END].index
    app_sv = svi[svi['appeared'].fillna(False).astype(bool)]
    censor_ok = bool((~app_sv.loc[app_sv.index.intersection(recent), 'longevity_event'].astype(bool)).all())
    ended_ok = bool(app_sv.loc[app_sv.index.difference(recent), 'longevity_event'].astype(bool).all())
    # QBs with a gap: their longevity time is the LAST appearance, not the first missing season
    gap_time_ok = True
    for pid in gq.index:
        if bool(gq.loc[pid, 'longevity_event']):
            la = o[(o['qb_draft_prospect_id'] == pid) & (o['games_played'] > 0)]['season_index'].max()
            gap_time_ok &= float(gq.loc[pid, 'longevity_time']) == float(la)
    longest = (o[o['games_played'] > 0].groupby(['qb_draft_prospect_id', 'player_name'])['season_index'].max()
               .sort_values(ascending=False).head(8).reset_index().rename(columns={'season_index': 'last_season_index_with_appearance'}))
    return {'rows': int(len(s)), 'qbs': int(s['qb_draft_prospect_id'].nunique()),
            'status_counts': s['status'].value_counts().to_dict(),
            'indexing_season1_is_draft_year': idx_ok, 'duplicate_qb_season_rows': dup, 'qbs_with_grid_gaps': gaps,
            'qbs_with_truncated_observed_seasons': trunc, 'constraints': cons,
            'qbs_with_a_missed_season_then_return': len(gap_qbs),
            'gap_qbs_marked_ended_at_last_appearance_not_at_gap': gap_time_ok,
            'appeared_in_last_two_seasons_all_censored': censor_ok,
            'others_who_appeared_all_marked_ended': ended_ok,
            'qbs_censored_longevity': int((~app_sv['longevity_event'].astype(bool)).sum()),
            'qbs_ended_longevity': int(app_sv['longevity_event'].astype(bool).sum()),
            'never_appeared_excluded_from_longevity': int((~svi['appeared'].fillna(False).astype(bool)).sum()),
            'longest_careers': longest.to_dict('records')}


def reconcile(seasons: pd.DataFrame) -> pd.DataFrame:
    """Season sums vs draft-board career totals for QBs whose careers ended (last appearance <= OBS_END - 2)."""
    con = sqlite3.connect(f'file:{CT.DB.as_posix()}?mode=ro', uri=True)
    try:
        board = pd.read_sql('select q.id as qb_draft_prospect_id, r.career_last_season, ' +
                            ', '.join(f'r.{c}' for c in RECONCILE.values()) +
                            ' from qb_draft_prospects q join nfl_draft_picks_raw r on r.id = q.draft_pick_raw_id', con)
    finally:
        con.close()
    obs = seasons[seasons['status'] == 'observed']
    sums = obs.groupby('qb_draft_prospect_id')[list(RECONCILE)].sum()
    m = sums.join(board.set_index('qb_draft_prospect_id'), how='inner')
    m = m[m['career_last_season'].notna() & (m['career_last_season'] <= OBS_END_SEASON - INACTIVE_SEASONS_FOR_END)]
    rows = []
    for ours, theirs in RECONCILE.items():
        ok = m[theirs].notna()
        diff = (m.loc[ok, ours] - m.loc[ok, theirs]).abs()
        tol = np.maximum(2, 0.01 * m.loc[ok, theirs].abs())
        rows.append({'stat': ours, 'qbs_compared': int(ok.sum()), 'exact_match': float((diff == 0).mean()),
                     'within_1pct_or_2': float((diff <= tol).mean()), 'median_abs_diff': float(diff.median()),
                     'max_abs_diff': float(diff.max())})
    return pd.DataFrame(rows)


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    seasons = pd.read_parquet(CT.OUT)
    sv = CT.survival_frame(seasons)
    c = checks(seasons, sv)
    R = reconcile(seasons)
    R.to_csv(RESULTS / 'trajectory_reconciliation.csv', index=False)
    (RESULTS / 'trajectory_audit.json').write_text(json.dumps(c, indent=2, default=str), encoding='utf-8')
    SECTION.write_text(render(c, R), encoding='utf-8')
    print(json.dumps(c, indent=2, default=str)); print(R.round(3).to_string())
    return c


def render(c, R) -> str:
    cons = c['constraints']
    yn = lambda b: 'pass' if b else '**FAIL**'  # noqa: E731
    L = ['## Career trajectories and longevity', '',
         f'Table: {c["rows"]:,} QB-season rows for {c["qbs"]} drafted QBs (2000-2026 classes); status counts '
         + ', '.join(f'{k} {v:,}' for k, v in c['status_counts'].items()) + '.', '',
         '| Check | Result |', '|---|---|',
         f'| Season 1 = draft year, for every row | {yn(c["indexing_season1_is_draft_year"])} |',
         f'| One row per QB per season; no gaps in the grid | {yn(c["duplicate_qb_season_rows"] == 0 and c["qbs_with_grid_gaps"] == 0)} |',
         f'| Long careers not truncated (every observed season through {OBS_END_SEASON} present) | {yn(c["qbs_with_truncated_observed_seasons"] == 0)} |',
         f'| Games started <= games played | {yn(cons["starts_le_games_played"])} |',
         f'| Games played <= season cap (16 before 2021, 17 after) | {yn(cons["games_le_cap"])} |',
         f'| Completions <= attempts; no negative counts | {yn(cons["completions_le_attempts"] and cons["no_negative_counts"])} |',
         f'| Future and unverified rows carry no values (censored, not zero) | {yn(cons["future_rows_have_no_values"] and cons["unverified_rows_have_no_values"])} |',
         f'| Missed seasons are not retirements: {c["qbs_with_a_missed_season_then_return"]} QBs missed a season and came back; their end is dated at the last appearance | {yn(c["gap_qbs_marked_ended_at_last_appearance_not_at_gap"])} |',
         f'| QBs who appeared in {OBS_END_SEASON - 1} or {OBS_END_SEASON} are right-censored | {yn(c["appeared_in_last_two_seasons_all_censored"])} |',
         f'| Others who ever appeared are counted as ended (inferred, two missed seasons) | {yn(c["others_who_appeared_all_marked_ended"])} |', '',
         f'Longevity: {c["qbs_ended_longevity"]} inferred career ends, {c["qbs_censored_longevity"]} censored, '
         f'{c["never_appeared_excluded_from_longevity"]} never appeared (excluded from the longevity curve).', '',
         'Longest careers in the table (last season index with an appearance):', '',
         md_table(pd.DataFrame(c['longest_careers'])), '',
         '**Reconciliation with the draft board.** Season sums against the nflverse/PFR career totals on the draft board, '
         'for QBs whose careers had ended:', '',
         md_table(R), '',
         'Differences come from sources, not from the table logic: the board\'s career totals are PFR\'s, the season rows '
         'are nflverse play-by-play aggregates, and Phase 1 kept only player-seasons with at least one pass attempt (a '
         'rushing-only season is missing). The five-season totals used for modeling matched the Phase-3 targets exactly (0 '
         'mismatches, Phase 5).', '',
         '**Three kinds of numbers, labelled separately in every output:**', '',
         '* **Observed**: what a QB actually did, from the season table (status `observed`).',
         '* **Historical descriptive**: averages and Kaplan-Meier curves by draft round (`career_trajectory_evaluation.md`). '
         'They describe groups and are never presented as a player\'s forecast.',
         '* **Predicted**: the career models\' forecasts for one QB (`year_by_year` in the forecast interface), labelled with '
         'the model version and their backtest status.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
