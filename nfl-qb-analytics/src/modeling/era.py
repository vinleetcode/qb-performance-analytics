"""Season-relative normalization of college QB statistics (Phase 4, NORMALIZATION_VERSION).

REFERENCE POPULATION (who a QB is compared with). Built from ALL college passers
in the local database -- not only drafted QBs -- by the same union the advanced
pipeline uses (CFBD season stats; for player-seasons CFBD's season stats lack,
mostly 2004-2008, season totals summed from CFBD box scores). A passer-season
enters the reference population when:
  * his team is classified FBS by CFBD in that season (from the games table),
  * he has >= REF_MIN_ATTEMPTS pass attempts that season, and
  * he has >= REF_MIN_GAMES games with a pass attempt (box scores).
Each qualifying passer counts once (unweighted), so the baseline describes a
typical FBS starter-level passer; an attempt-weighted option exists but is not
the default. FCS seasons are never in the reference population.

SCORES (higher = better on every normalized scale):
  season_z    (x - season mean) / season SD, clipped to +/- Z_CLIP
  season_pct  empirical percentile among the season's reference passers (mid-rank ties), 0..1
  conf_z      z-score within the (season, conference) reference group; groups with
              < CONF_MIN_GROUP passers fall back to the season baseline (flagged).
              The conference is the team's conference IN THAT SEASON, so realignment
              is handled by construction.
  Interceptions and schedule strength are sign-flipped: a higher int score means
  FEWER interceptions; a higher sos score means a TOUGHER schedule (lower YPA allowed).

SMALL SAMPLES / MISSING / ZERO VARIANCE. A scored season needs >= PROSPECT_MIN_ATTEMPTS
attempts (rate stats) or >= 1 game with a pass attempt (per-game stats); otherwise the
score is NaN. A baseline with < 2 passers or SD <= 1e-9 yields NaN, never 0 or inf.
Missing inputs stay NaN. Raw values are always kept next to the scores.

LEAKAGE. A season-s baseline uses only season-s passers (or seasons s-k+1..s with the
`window` option), all of whom completed that season before the next NFL draft. The
baselines are fixed functions of contemporaneous public statistics, not parameters
fitted to the modeled QBs or their NFL outcomes; learned preprocessing (imputation,
scaling) still happens inside each training fold's Pipeline.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / 'data' / 'qb_history.db'
NORMALIZATION_VERSION = '2.0.0'
REF_FILE = ROOT / 'data' / 'modeling' / f'college_reference_population_v{NORMALIZATION_VERSION}.parquet'

REF_MIN_ATTEMPTS = 100
REF_MIN_GAMES = 4
PROSPECT_MIN_ATTEMPTS = 50
CONF_MIN_GROUP = 6
Z_CLIP = 4.0
EPS = 1e-9

# stat -> (direction, needs_attempts). direction -1 flips so that higher = better.
NORM_STATS = {
    'completion_pct': (1, True),
    'yards_per_attempt': (1, True),
    'td_pct': (1, True),
    'int_pct': (-1, True),
    'attempts_per_game': (1, False),
    'rush_yards_per_game': (1, False),
    'opp_adj_ypa': (1, True),
    'sos_pass_defense': (-1, True),
}
METHODS = ('season_z', 'season_pct', 'conf_z')


# --------------------------------------------------------------------- core formulas
def zscore(x, mean, sd, direction: int = 1, clip: float | None = Z_CLIP):
    """(x - mean) / sd * direction; NaN where sd is missing/<=EPS or x missing."""
    x = np.asarray(x, dtype=float)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        z = np.where(np.isfinite(sd) & (sd > EPS), (x - mean) / sd, np.nan) * direction
    if clip is not None:
        z = np.where(np.isnan(z), np.nan, np.clip(z, -clip, clip))
    return z


def ecdf_percentile(x, ref, direction: int = 1):
    """Mid-rank empirical percentile of each x among `ref` values: (#<x + 0.5 #==x)/n.
    Works for x that are not members of ref. direction -1 returns 1 - percentile."""
    ref = np.sort(np.asarray(ref, dtype=float)[np.isfinite(np.asarray(ref, dtype=float))])
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if len(ref) == 0:
        return out
    ok = np.isfinite(x)
    lo = np.searchsorted(ref, x[ok], side='left')
    hi = np.searchsorted(ref, x[ok], side='right')
    p = (lo + 0.5 * (hi - lo)) / len(ref)
    out[ok] = p if direction == 1 else 1 - p
    return out


def weighted_mean_sd(values, weights=None) -> tuple[float, float, int]:
    v = np.asarray(values, dtype=float)
    m = np.isfinite(v)
    if weights is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(weights, dtype=float)
        m &= np.isfinite(w) & (w > 0)
    v, w = v[m], w[m]
    n = int(len(v))
    if n < 2:
        return (float(v.mean()) if n else np.nan), np.nan, n
    mu = float(np.average(v, weights=w))
    if weights is None:
        sd = float(np.std(v, ddof=1))
    else:
        sd = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
    return mu, sd, n


def group_baselines(ref: pd.DataFrame, stat: str, by=('season',), window: int = 1,
                    weight: str | None = None) -> pd.DataFrame:
    """Baseline (n, mean, sd) per group. `window`>1 pools seasons s-window+1..s (past
    and current only -- never later seasons). `weight` names a column (e.g. attempts)."""
    by = list(by)
    keys = ref[by].drop_duplicates().dropna()
    rows = []
    for key in keys.itertuples(index=False, name=None):
        k = dict(zip(by, key))
        s = k['season']
        m = ref['season'].between(s - window + 1, s)
        for c in by:
            if c != 'season':
                m &= ref[c] == k[c]
        g = ref[m]
        mu, sd, n = weighted_mean_sd(g[stat], g[weight] if weight else None)
        rows.append({**k, 'stat': stat, 'n': n, 'mean': mu, 'sd': sd})
    return pd.DataFrame(rows, columns=by + ['stat', 'n', 'mean', 'sd'])


def _eligible_mask(df: pd.DataFrame, stat: str) -> np.ndarray:
    _, needs_att = NORM_STATS.get(stat, (1, True))
    if needs_att:
        return (df['attempts'].fillna(0) >= PROSPECT_MIN_ATTEMPTS).to_numpy()
    return (df['games_with_pass_attempt'].fillna(0) >= 1).to_numpy()


def normalize_frame(df: pd.DataFrame, ref: pd.DataFrame, stats=None, methods=METHODS,
                    window: int = 1, weight: str | None = None) -> pd.DataFrame:
    """Add `{stat}__{method}` columns (and `{stat}__conf_fallback`) to a copy of `df`.
    `df` needs season, conference, attempts, games_with_pass_attempt and the stats.
    `ref` is the reference population (in_reference rows only are used)."""
    stats = list(stats or NORM_STATS)
    ref = ref[ref['in_reference']] if 'in_reference' in ref.columns else ref
    out = df.copy()
    for stat in stats:
        direction, _ = NORM_STATS.get(stat, (1, True))
        ok = _eligible_mask(out, stat)
        x = out[stat].to_numpy(dtype=float)
        if 'season_z' in methods or 'conf_z' in methods:
            sb = group_baselines(ref, stat, ('season',), window, weight).set_index('season')
            mu = out['season'].map(sb['mean']).to_numpy(dtype=float)
            sd = out['season'].map(sb['sd']).to_numpy(dtype=float)
            season_z = np.where(ok, zscore(x, mu, sd, direction), np.nan)
            if 'season_z' in methods:
                out[f'{stat}__season_z'] = season_z
        if 'season_pct' in methods:
            pct = np.full(len(out), np.nan)
            for s, idx in out.groupby('season').indices.items():
                r = ref.loc[ref['season'].between(s - window + 1, s), stat]
                pct[idx] = ecdf_percentile(x[idx], r, direction)
            out[f'{stat}__season_pct'] = np.where(ok, pct, np.nan)
        if 'conf_z' in methods:
            cb = group_baselines(ref.dropna(subset=['conference']), stat, ('season', 'conference'), window, weight)
            cb = cb.set_index(['season', 'conference'])
            keys = list(zip(out['season'], out['conference']))
            n = np.array([cb['n'].get(k, 0) for k in keys], dtype=float)
            cmu = np.array([cb['mean'].get(k, np.nan) for k in keys], dtype=float)
            csd = np.array([cb['sd'].get(k, np.nan) for k in keys], dtype=float)
            use_conf = n >= CONF_MIN_GROUP
            cz = np.where(use_conf, zscore(x, cmu, csd, direction), season_z)
            out[f'{stat}__conf_z'] = np.where(ok, cz, np.nan)
            out[f'{stat}__conf_fallback'] = ~use_conf
    return out


# --------------------------------------------------------------------- reference population
def _read(con, sql: str) -> pd.DataFrame:
    return pd.read_sql(sql, con)


def load_tables(db_path: Path = DB_PATH) -> dict:
    """Read-only access to the local SQLite database (mode=ro: cannot modify it)."""
    con = sqlite3.connect(f'file:{Path(db_path).as_posix()}?mode=ro', uri=True)
    try:
        t = {
            'season_stats': _read(con, 'select * from cfb_player_seasons'),
            'players': _read(con, 'select id as player_id, cfbd_player_id from cfb_players'),
            'game_passing': _read(con, 'select * from cfb_game_passing'),
            'game_rushing': _read(con, 'select * from cfb_game_rushing'),
            'games': _read(con, 'select season, home_team, away_team, home_classification, away_classification, '
                                'home_conference, away_conference from cfb_games'),
            'team_stats': _read(con, 'select season, team, conference, games, pass_attempts, rushing_attempts, '
                                     'off_passing_play_rate from cfb_team_season_stats'),
        }
    finally:
        con.close()
    return t


def fbs_team_seasons(games: pd.DataFrame) -> pd.DataFrame:
    a = games.loc[games['home_classification'].eq('fbs'), ['season', 'home_team']].rename(columns={'home_team': 'team'})
    b = games.loc[games['away_classification'].eq('fbs'), ['season', 'away_team']].rename(columns={'away_team': 'team'})
    return pd.concat([a, b]).drop_duplicates().reset_index(drop=True)


def build_reference_population(tables: dict) -> pd.DataFrame:
    """All college passer-seasons with raw stats, per-game rates, opponent-adjusted
    YPA and schedule strength, and the `in_reference` flag."""
    from src import advanced_pipeline as AP
    from src.features import opponent_adjust as OA
    from src.ingest_cfbd_advanced import id_str

    ss = tables['season_stats'].merge(tables['players'], on='player_id', how='left')
    ss['cfbd_player_id'] = ss['cfbd_player_id'].map(id_str)
    gp = tables['game_passing'].copy()
    gp['cfbd_player_id'] = gp['cfbd_player_id'].map(id_str)
    gr = tables['game_rushing'].copy()
    gr['cfbd_player_id'] = gr['cfbd_player_id'].map(id_str)
    box = AP.box_season_totals(gp, gr)
    passer = AP.all_passer_seasons(ss, box)
    passer = passer[passer['attempts'].fillna(0) > 0].copy()

    fbs = fbs_team_seasons(tables['games'])
    tg = OA.team_game_totals(gp)
    qg = OA.qb_game_adjustments(gp, fbs)
    qs = OA.qb_season_from_games(qg, tg)[['cfbd_player_id', 'season', 'games_with_pass_attempt', 'opp_adj_ypa',
                                         'sos_pass_defense', 'opp_adj_attempt_coverage']]
    pop = passer.merge(qs, on=['cfbd_player_id', 'season'], how='left')
    pop['team_first'] = pop['team'].astype(str).str.split('; ').str[0]
    key = set(zip(fbs['season'], fbs['team']))
    pop['is_fbs'] = [(s, t) in key for s, t in zip(pop['season'], pop['team_first'])]
    g = pop['games_with_pass_attempt']
    pop['attempts_per_game'] = np.where(g > 0, pop['attempts'] / g, np.nan)
    pop['rush_yards_per_game'] = np.where(g > 0, pop['rushing_yards'] / g, np.nan)
    pop['in_reference'] = pop['is_fbs'] & (pop['attempts'] >= REF_MIN_ATTEMPTS) & (g >= REF_MIN_GAMES)
    keep = ['cfbd_player_id', 'season', 'team', 'team_first', 'conference', 'season_stats_source', 'is_fbs',
            'in_reference', 'attempts', 'completions', 'passing_yards', 'passing_tds', 'interceptions',
            'rushing_attempts', 'rushing_yards', 'games_with_pass_attempt', 'completion_pct', 'yards_per_attempt',
            'td_pct', 'int_pct', 'attempts_per_game', 'rush_yards_per_game', 'opp_adj_ypa', 'sos_pass_defense',
            'opp_adj_attempt_coverage']
    return pop[keep].reset_index(drop=True)


def team_pass_rates(team_stats: pd.DataFrame, fbs: pd.DataFrame) -> pd.DataFrame:
    """FBS team pass rate = pass attempts / (pass attempts + rush attempts). College
    rushing counts sacks as rushes, so this understates dropback rate; consistently
    so in every season."""
    t = team_stats.copy()
    key = set(zip(fbs['season'], fbs['team']))
    t = t[[(s, x) in key for s, x in zip(t['season'], t['team'])]]
    tot = t['pass_attempts'] + t['rushing_attempts']
    t['pass_rate'] = np.where(tot > 0, t['pass_attempts'] / tot, np.nan)
    return t[['season', 'team', 'conference', 'pass_rate', 'off_passing_play_rate']]


def load_or_build_reference(path: Path = REF_FILE, rebuild: bool = False, db_path: Path = DB_PATH) -> pd.DataFrame:
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    pop = build_reference_population(load_tables(db_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pop.to_parquet(path, index=False)
    return pop
