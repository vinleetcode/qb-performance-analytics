"""NFL career trajectories of drafted QBs: one row per QB per NFL season, plus
descriptive career-year profiles and survival analysis (Phase 5).

Run:  python -m src.modeling.career_trajectory

SOURCES (local only): nflverse regular-season player stats already in the Phase-1
tables (players + player_seasons_raw: passing AND rushing), starts from the nflverse
schedule's starting-QB fields (data/model_ready/nfl_prospect_seasons.parquet),
draft board (data/qb_draft_prospects.parquet), Phase-3 outcome table for verified
never-played QBs.

ROW STATUS
  observed            NFL season <= OBS_END_SEASON and the QB's NFL record is linked (or he is
                      verified to have never played): zeros are REAL zeros.
  future              season not played yet (after OBS_END_SEASON): no values (censored).
  unverified          QB has no linked NFL record and is not verified as never playing: values NA.
Limitation: Phase 1 kept player-seasons with >= 1 pass attempt, so a season with only rushing
plays and no pass attempt or start counts as 0 games (rare for drafted QBs; noted in the report).

SURVIVAL (Kaplan-Meier with Greenwood CIs, log-rank tests), durations in NFL seasons since the draft
(season 1 = rookie season):
  first start              event = first regular-season start; censored at the last observed season
  16 / 32 career starts    event = cumulative starts reach the milestone
  last appearance          event = last season with an appearance, ONLY when the QB then missed the two
                           most recent observed seasons (an inference, not a confirmed retirement);
                           QBs who appeared in either of the last two seasons are censored.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sst

from src import metrics
from src.features.targets import INACTIVE_SEASONS_FOR_END, OBS_END_SEASON
from src.ingest_cfbd_advanced import id_str
from src.modeling import survival

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'data' / 'qb_history.db'
MODEL_READY = ROOT / 'data' / 'model_ready'
OUT = ROOT / 'data' / 'modeling' / 'nfl_qb_seasons_v5.0.0.parquet'
REPORT = ROOT / 'reports' / 'career_trajectory_evaluation.md'
RESULTS = ROOT / 'results' / 'phase5'
STAT_COLS = ['games_played', 'games_started', 'pass_attempts', 'pass_completions', 'pass_yards', 'pass_tds',
             'interceptions', 'rush_attempts', 'rush_yards', 'rush_tds', 'sacks', 'sack_yards']
ROUND_GROUPS = [(1, 1, 'Round 1'), (2, 3, 'Rounds 2-3'), (4, 7, 'Rounds 4-7')]


def round_group(r) -> str:
    for lo, hi, name in ROUND_GROUPS:
        if lo <= r <= hi:
            return name
    return 'unknown'


def season_games_cap(season: int) -> int:
    return 17 if season >= 2021 else 16


def load_nfl_player_seasons(db: Path = DB) -> pd.DataFrame:
    con = sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)
    try:
        q = pd.read_sql(
            "select p.source_player_id as gsis_id, s.season, s.games, s.attempts, s.completions, s.passing_yards, "
            "s.passing_tds, s.interceptions, s.sacks_suffered, s.sack_yards_lost, s.carries, s.rushing_yards, "
            "s.rushing_tds from player_seasons_raw s join players p on p.id = s.player_id "
            "where p.source_id = 'nflverse_player_stats' and s.season_type = 'REG'", con)
    finally:
        con.close()
    return q.groupby(['gsis_id', 'season'], as_index=False).sum(numeric_only=True)


def build_seasons(prospects: pd.DataFrame, grid: pd.DataFrame, stats: pd.DataFrame, outcomes: pd.DataFrame,
                  obs_end: int = OBS_END_SEASON, max_index: int = 10) -> pd.DataFrame:
    """One row per drafted QB x NFL season index 1..max_index."""
    p = prospects.rename(columns={'id': 'qb_draft_prospect_id'})[
        ['qb_draft_prospect_id', 'draft_year', 'round', 'pick', 'player_name', 'college', 'gsis_id']].copy()
    status = outcomes.set_index('qb_draft_prospect_id')['career_end_status']
    p['career_end_status'] = p['qb_draft_prospect_id'].map(status)
    linked = set(grid['qb_draft_prospect_id'])
    never = p['career_end_status'].isin(['never_appeared_window_complete', 'never_appeared_censored'])
    rows = []
    for _, r in p.iterrows():
        for k in range(1, max_index + 1):
            rows.append((r['qb_draft_prospect_id'], int(r['draft_year']) + k - 1, k))
    s = pd.DataFrame(rows, columns=['qb_draft_prospect_id', 'nfl_season', 'season_index']).merge(p, on='qb_draft_prospect_id')
    st = stats.rename(columns={'season': 'nfl_season', 'games': 'gp_stats', 'attempts': 'pass_attempts',
                               'completions': 'pass_completions', 'passing_yards': 'pass_yards', 'passing_tds': 'pass_tds',
                               'sacks_suffered': 'sacks', 'sack_yards_lost': 'sack_yards', 'carries': 'rush_attempts',
                               'rushing_yards': 'rush_yards', 'rushing_tds': 'rush_tds'})
    s = s.merge(st, on=['gsis_id', 'nfl_season'], how='left')
    s = s.merge(grid[['qb_draft_prospect_id', 'nfl_season', 'games_started']], on=['qb_draft_prospect_id', 'nfl_season'],
                how='left')
    s['status'] = np.where(s['nfl_season'] > obs_end, 'future',
                           np.where(s['qb_draft_prospect_id'].isin(linked) | s['qb_draft_prospect_id'].map(
                               dict(zip(p['qb_draft_prospect_id'], never))).fillna(False), 'observed', 'unverified'))
    obs = s['status'] == 'observed'
    for c in ['gp_stats', 'games_started', 'pass_attempts', 'pass_completions', 'pass_yards', 'pass_tds', 'interceptions',
              'rush_attempts', 'rush_yards', 'rush_tds', 'sacks', 'sack_yards']:
        s[c] = pd.to_numeric(s[c], errors='coerce')
        s.loc[obs, c] = s.loc[obs, c].fillna(0)
        s.loc[~obs, c] = np.nan
    # a start without a stat row is still a game played; games played can never be below starts
    s['games_played'] = np.where(obs, np.maximum(s['gp_stats'], s['games_started']), np.nan)
    s['games_cap'] = s['nfl_season'].map(season_games_cap)
    s['round_group'] = s['round'].map(round_group)
    att = s['pass_attempts']
    s['completion_pct'] = np.where(att >= 50, metrics.completion_pct(s['pass_completions'], att), np.nan)
    s['yards_per_attempt'] = np.where(att >= 50, metrics.yards_per_attempt(s['pass_yards'], att), np.nan)
    s['passer_rating'] = np.where(att >= 50, metrics.passer_rating(s['pass_completions'], att, s['pass_yards'],
                                                                  s['pass_tds'], s['interceptions']), np.nan)
    return s.drop(columns=['gp_stats'])


def five_year_totals(seasons: pd.DataFrame, obs_end: int = OBS_END_SEASON) -> pd.DataFrame:
    """Sum of NFL seasons 1-5 for QBs whose window is complete and whose record is observed."""
    w = seasons[seasons['season_index'] <= 5]
    g = w.groupby('qb_draft_prospect_id')
    complete = g['nfl_season'].max() <= obs_end
    observed = g['status'].apply(lambda s: (s == 'observed').all())
    tot = g[STAT_COLS].sum(min_count=5)
    tot = tot[complete & observed].reset_index()
    tot = tot.rename(columns={c: f'{c}_5yr' for c in STAT_COLS})
    a = tot['pass_attempts_5yr']
    tot['completion_pct_5yr'] = np.where(a >= 300, 100 * tot['pass_completions_5yr'] / a.where(a > 0), np.nan)
    tot['yards_per_attempt_5yr'] = np.where(a >= 300, tot['pass_yards_5yr'] / a.where(a > 0), np.nan)
    tot['td_rate_5yr'] = np.where(a >= 300, 100 * tot['pass_tds_5yr'] / a.where(a > 0), np.nan)
    tot['int_rate_5yr'] = np.where(a >= 300, 100 * tot['interceptions_5yr'] / a.where(a > 0), np.nan)
    r = tot['rush_attempts_5yr']
    tot['rush_ypc_5yr'] = np.where(r >= 50, tot['rush_yards_5yr'] / r.where(r > 0), np.nan)
    return tot


def build(db: Path = DB):
    prospects = pd.read_parquet(ROOT / 'data' / 'qb_draft_prospects.parquet')
    grid = pd.read_parquet(MODEL_READY / 'nfl_prospect_seasons.parquet')
    outcomes = pd.read_parquet(MODEL_READY / 'nfl_targets.parquet')
    stats = load_nfl_player_seasons(db)
    # the grid must reach the last observed season for every class (2000 class -> 26 seasons), otherwise a QB still
    # playing beyond the grid would be "inferred inactive" at the grid edge
    seasons = build_seasons(prospects, grid, stats, outcomes, max_index=OBS_END_SEASON - 2000 + 1)
    return seasons, five_year_totals(seasons), outcomes


# ------------------------------------------------------------------ analysis
def logrank(durations, events, groups) -> dict:
    """K-sample log-rank test (chi-square on observed vs expected events at each event time)."""
    d = pd.DataFrame({'t': durations, 'e': events.astype(bool), 'g': groups})
    gs = sorted(d['g'].unique())
    times = np.sort(d.loc[d['e'], 't'].unique())
    O = np.zeros(len(gs)); E = np.zeros(len(gs)); V = np.zeros((len(gs), len(gs)))
    for t in times:
        at = d[d['t'] >= t]
        n = len(at); dt = int(((d['t'] == t) & d['e']).sum())
        if n < 2:
            continue
        ni = np.array([(at['g'] == g).sum() for g in gs], dtype=float)
        di = np.array([((d['t'] == t) & d['e'] & (d['g'] == g)).sum() for g in gs], dtype=float)
        O += di
        E += dt * ni / n
        f = dt * (n - dt) / (n * n * (n - 1)) if n > 1 else 0
        V += f * (np.diag(ni * n) - np.outer(ni, ni))
    k = len(gs) - 1
    diff = (O - E)[:k]
    chi = float(diff @ np.linalg.pinv(V[:k, :k]) @ diff)
    return {'chi2': chi, 'df': k, 'p_value': float(sst.chi2.sf(chi, k)), 'observed': dict(zip(gs, O)),
            'expected': dict(zip(gs, E))}


def survival_frame(seasons: pd.DataFrame, obs_end: int = OBS_END_SEASON) -> pd.DataFrame:
    o = seasons[seasons['status'] == 'observed'].sort_values(['qb_draft_prospect_id', 'season_index'])
    rows = []
    for pid, g in o.groupby('qb_draft_prospect_id'):
        last_idx = int(g['season_index'].max())
        cum = g['games_started'].cumsum().to_numpy()
        idx = g['season_index'].to_numpy()
        rec = {'qb_draft_prospect_id': pid, 'draft_year': int(g['draft_year'].iloc[0]), 'round': g['round'].iloc[0],
               'round_group': g['round_group'].iloc[0], 'seasons_observed': last_idx}
        for name, thr in (('first_start', 1), ('starts_16', 16), ('starts_32', 32)):
            hit = np.flatnonzero(cum >= thr)
            rec[f'{name}_time'] = float(idx[hit[0]]) if len(hit) else float(last_idx)
            rec[f'{name}_event'] = bool(len(hit))
        app = g[g['games_played'] > 0]
        if len(app):
            last_app = int(app['nfl_season'].max())
            ended = last_app <= obs_end - INACTIVE_SEASONS_FOR_END
            rec['longevity_time'] = float(app['season_index'].max() if ended else last_idx)
            rec['longevity_event'] = bool(ended)
            rec['appeared'] = True
        else:
            rec['appeared'] = False
        rows.append(rec)
    return pd.DataFrame(rows)


def career_year_profile(seasons: pd.DataFrame, max_index: int = 8) -> pd.DataFrame:
    o = seasons[(seasons['status'] == 'observed') & (seasons['season_index'] <= max_index) & (seasons['draft_year'] >= 2000)]
    agg = []
    for (rg, k), g in o.groupby(['round_group', 'season_index']):
        agg.append({'round_group': rg, 'season_index': k, 'n_qbs': len(g),
                    'share_any_start': float((g['games_started'] > 0).mean()),
                    'share_8plus_starts': float((g['games_started'] >= 8).mean()),
                    'share_appeared': float((g['games_played'] > 0).mean()),
                    'mean_starts': float(g['games_started'].mean()), 'mean_attempts': float(g['pass_attempts'].mean()),
                    'mean_pass_yards': float(g['pass_yards'].mean()), 'mean_rush_yards': float(g['rush_yards'].mean()),
                    'median_ypa_qualified': float(g['yards_per_attempt'].median())})
    return pd.DataFrame(agg)


def fig_profiles(prof: pd.DataFrame, km: dict):
    from src.modeling.evaluate import INK, INK_2, SURFACE, fig_ax, save
    colors = {'Round 1': '#2a78d6', 'Rounds 2-3': '#eb6834', 'Rounds 4-7': '#1baf7a'}
    fig, axes = fig_ax(13, 4.2, ncols=3)
    for rg, c in colors.items():
        g = prof[prof['round_group'] == rg]
        for ax, col in zip(axes, ('share_any_start', 'mean_starts', 'mean_pass_yards')):
            ax.plot(g['season_index'], g[col], color=c, linewidth=2, marker='o', markersize=5, markeredgecolor=SURFACE,
                    label=rg)
    for ax, t in zip(axes, ('Share with at least one start', 'Mean regular-season starts', 'Mean passing yards')):
        ax.set_title(t, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('NFL season since draft (1 = rookie)', color=INK, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Drafted QBs 2000-2025 by draft round: career-year profile (observed seasons only)', color=INK,
                 fontsize=11, x=0.01, ha='left')
    save(fig, 'p5_career_year_profiles.png')
    fig, axes = fig_ax(13, 4.2, ncols=3)
    for ax, (name, title) in zip(axes, (('first_start', 'Time to first NFL start'), ('starts_32', 'Time to 32 career starts'),
                                         ('longevity', 'Time to inferred last appearance'))):
        for rg, c in colors.items():
            k = km[name].get(rg)
            if k is None or k.empty:
                continue
            curve = 1 - k['survival'] if name != 'longevity' else k['survival']
            ax.step(k['time'], curve, where='post', color=c, linewidth=2, label=rg)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('NFL seasons since draft', color=INK, fontsize=8.5)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('Cumulative share reaching event', color=INK, fontsize=8.5)
    axes[2].set_ylabel('Share not yet inferred inactive', color=INK, fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Kaplan-Meier estimates (right-censored at the 2025 season); last appearance is inferred, not retirement',
                 color=INK, fontsize=10.5, x=0.01, ha='left')
    save(fig, 'p5_career_survival.png')


def main() -> dict:
    seasons, tot, outcomes = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seasons.to_parquet(OUT, index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # cross-checks against the Phase-3 outcome table
    chk = tot.merge(outcomes[['qb_draft_prospect_id', 'games_started_5yr', 'pass_attempts_5yr', 'pass_yards_5yr']],
                    on='qb_draft_prospect_id', how='inner', suffixes=('', '_p3'))
    mism = {c: int((chk[f'{c}_5yr'] != chk[f'{c}_5yr_p3']).sum()) for c in ('games_started', 'pass_attempts', 'pass_yards')
            if f'{c}_5yr_p3' in chk.columns}
    sv = survival_frame(seasons[seasons['draft_year'] >= 2000])
    sv.to_csv(RESULTS / 'survival_frame.csv', index=False)
    prof = career_year_profile(seasons)
    prof.to_csv(RESULTS / 'career_year_profile.csv', index=False)
    km, lr, med = {}, {}, {}
    for name in ('first_start', 'starts_16', 'starts_32', 'longevity'):
        km[name] = {}
        d = sv.dropna(subset=[f'{name}_time']) if f'{name}_time' in sv else sv
        for rg in ('Round 1', 'Rounds 2-3', 'Rounds 4-7'):
            g = d[d['round_group'] == rg]
            km[name][rg] = survival.kaplan_meier(g[f'{name}_time'], g[f'{name}_event'])
        lr[name] = logrank(d[f'{name}_time'].to_numpy(), d[f'{name}_event'].to_numpy(), d['round_group'].to_numpy())
    fig_profiles(prof, km)
    ob = seasons[seasons['status'] == 'observed']          # unobserved rows are NaN by design; NaN <= NaN is False
    cons = {'starts_le_games': bool((ob['games_started'] <= ob['games_played']).all()),
            'completions_le_attempts': bool((ob['pass_completions'] <= ob['pass_attempts']).all()),
            'nonnegative_counts': bool((seasons[['pass_attempts', 'pass_tds', 'rush_attempts', 'rush_tds']].fillna(0) >= 0).all().all())}
    summary = {'qb_seasons_rows': int(len(seasons)), 'qbs': int(seasons['qb_draft_prospect_id'].nunique()),
               'status_counts': seasons['status'].value_counts().to_dict(), 'five_year_complete_qbs': int(len(tot)),
               'crosscheck_rows': int(len(chk)), 'crosscheck_mismatches': mism, 'consistency': cons,
               'logrank': lr, 'survival_n': int(len(sv))}
    (RESULTS / 'trajectory_summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    REPORT.write_text(render(summary, prof, km, sv, seasons), encoding='utf-8')
    print(f'seasons {len(seasons)} rows | 5-yr complete {len(tot)} | mismatches {mism}')
    return summary


def _km_at(k: pd.DataFrame, t: float) -> float:
    s = k[k['time'] <= t]['survival']
    return float(s.iloc[-1]) if len(s) else 1.0


def render(summary, prof, km, sv, seasons) -> str:
    L = ['# NFL career trajectories and longevity of drafted QBs', '',
         'Generated by `python -m src.modeling.career_trajectory`. **Descriptive analysis** of drafted QBs from the 2000-2026 '
         'classes, one row per QB per NFL season, observed through the 2025 regular season. Nothing here is a validated '
         'forecast; the forecasting models are evaluated in `career_production_evaluation.md`.', '',
         '## Dataset', '',
         f'* {summary["qb_seasons_rows"]:,} QB-season rows for {summary["qbs"]} drafted QBs (every NFL season from the draft '
         'through 2025, plus not-yet-played seasons marked `future`). '
         'Status: ' + ', '.join(f'{k} {v:,}' for k, v in summary['status_counts'].items()) + '.',
         '* `observed` rows hold real values, and a zero is a real zero. That includes drafted QBs verified (by the '
         'draft board) to have never played. `future` rows are seasons not yet played (censored). `unverified` rows '
         'belong to QBs with no linked NFL record who are not verified as never playing; their values are left '
         'missing, not zero.',
         f'* {summary["five_year_complete_qbs"]} QBs have a complete, observed five-season window. Their five-season '
         f'totals match the Phase-3 outcome table (mismatches: {summary["crosscheck_mismatches"]}, over '
         f'{summary["crosscheck_rows"]} rows).',
         f'* Consistency checks: starts <= games played {summary["consistency"]["starts_le_games"]}; completions <= '
         f'attempts {summary["consistency"]["completions_le_attempts"]}; non-negative counts '
         f'{summary["consistency"]["nonnegative_counts"]}.',
         '* Limitation: the Phase-1 stat table keeps NFL seasons with at least 1 pass attempt. A season in which a QB '
         'only ran the ball and never threw or started counts as 0 games. Starts always count, from the schedule.', '',
         '## Career-year profile by draft round', '',
         '![Career-year profiles](figures/p5_career_year_profiles.png)', '',
         '| Round group | Season | QBs observed | Share with any start | Share with 8+ starts | Mean starts | Mean pass yards |',
         '|---|---|---|---|---|---|---|']
    for _, r in prof[prof['season_index'] <= 6].iterrows():
        L.append(f'| {r["round_group"]} | {int(r["season_index"])} | {int(r["n_qbs"])} | {r["share_any_start"]:.2f} | '
                 f'{r["share_8plus_starts"]:.2f} | {r["mean_starts"]:.1f} | {r["mean_pass_yards"]:.0f} |')
    pv = prof.pivot_table(index='season_index', columns='round_group', values='share_any_start')
    L += ['', f'Share of QBs with at least one start, by season (seasons 1-6). **Round 1:** {pv.loc[1, "Round 1"]:.0%} as rookies, '
          f'{pv.loc[2, "Round 1"]:.0%} in season 2, then down to {pv.loc[6, "Round 1"]:.0%} by season 6. **Rounds 2-3:** '
          f'{pv.loc[1:6, "Rounds 2-3"].min():.0%}-{pv.loc[1:6, "Rounds 2-3"].max():.0%} in every season. **Rounds 4-7:** '
          f'{pv.loc[1:6, "Rounds 4-7"].min():.0%}-{pv.loc[1:6, "Rounds 4-7"].max():.0%}. These are averages over QBs still '
          'observed. Individual paths vary widely, and a single start can be an injury fill-in.', '',
          '## Time-to-event (Kaplan-Meier)', '',
          '![Survival](figures/p5_career_survival.png)', '',
          '| Event | Round group | QBs | Events | Share reached by season 2 | by season 5 | Log-rank across groups |',
          '|---|---|---|---|---|---|---|']
    for name, lab in (('first_start', 'First start'), ('starts_16', '16 career starts'), ('starts_32', '32 career starts')):
        for rg in ('Round 1', 'Rounds 2-3', 'Rounds 4-7'):
            k = km[name][rg]
            g = sv[sv['round_group'] == rg]
            L.append(f'| {lab} | {rg} | {len(g)} | {int(g[f"{name}_event"].sum())} | {1 - _km_at(k, 2):.2f} | '
                     f'{1 - _km_at(k, 5):.2f} | chi2={summary["logrank"][name]["chi2"]:.1f}, p={summary["logrank"][name]["p_value"]:.2g} |')
    lg = sv[sv['appeared']]
    L += ['', 'Shares are 1 - KM survival, so QBs still waiting (censored at the 2025 season) are handled correctly '
          'rather than counted as never reaching the event.', '',
          '### Longevity (inferred)', '',
          f'Among {len(lg)} drafted QBs who appeared in at least one game, {int(lg["longevity_event"].sum())} are '
          f'**inferred** inactive: no appearance in the {INACTIVE_SEASONS_FOR_END} most recent observed seasons. The rest '
          'are censored. This is not a confirmed retirement, since QBs do return after a gap, and nflverse\'s '
          'official status field is too sparse to confirm it.', '',
          '| Round group | QBs | Inferred ends | Share still appearing after 5 seasons (KM) | after 10 seasons |', '|---|---|---|---|---|']
    for rg in ('Round 1', 'Rounds 2-3', 'Rounds 4-7'):
        k = km['longevity'][rg]
        g = lg[lg['round_group'] == rg]
        L.append(f'| {rg} | {len(g)} | {int(g["longevity_event"].sum())} | {_km_at(k, 5):.2f} | {_km_at(k, 10):.2f} |')
    L += ['', f'Log-rank across round groups: chi2={summary["logrank"]["longevity"]["chi2"]:.1f}, '
          f'p={summary["logrank"]["longevity"]["p_value"]:.2g}. Round differences are associations. Draft slot buys '
          'roster time, so they are not evidence that early picks are better players.', '',
          '## Why this stays descriptive', '',
          '* The 2000-2004 classes have no college covariates, so a predictive survival model with college features '
          '(for example Cox) could use only the 2005+ classes. Draft round, the dominant covariate, is itself a '
          'cause of opportunity rather than a pre-draft trait. The forecasting question ("how many starts, and '
          'when?") is answered and evaluated with embargoed backtests in `career_production_evaluation.md`. Here, '
          'Kaplan-Meier curves and log-rank tests describe the history without claiming predictive validity.',
          '* The year-by-year **forecasts** in `career_production_evaluation.md` come from a simulation model that is '
          'evaluated in chronological backtests. It draws on these same QB-season rows, used only from classes whose '
          'outcomes were observable at each forecast date.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
