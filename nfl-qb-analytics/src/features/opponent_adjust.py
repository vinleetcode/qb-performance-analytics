"""Opponent-adjusted college passing, built from per-game box scores.

Method (documented in reports/advanced_features_coverage.md):
  1. For every game, each defense's passing allowed = sum of all opposing passers'
     lines (including team pseudo-rows).
  2. For a QB's game g vs defense D, D's baseline is D's allowed totals over its
     OTHER games that season (leave-one-out), so the QB's own performance never
     contaminates the yardstick it is measured against. Baselines need at least
     MIN_OTHER_GAMES other games; otherwise that game is excluded from the
     adjustment (typical for FCS opponents, which have few FBS box scores).
  3. League baseline = attempt-weighted FBS mean allowed, per season.
  4. Additive adjustment per game: adj = qb_rate - (D_allowed_loo - league_mean),
     aggregated attempt-weighted over the games that had a valid baseline.

Timing: every input belongs to a COMPLETED college season, which ends (bowls,
early January) before the NFL draft (late April). Full-season opponent data is
therefore knowable at the pre-draft prediction cutoff; it would NOT be valid for
an in-season forecast, which this project does not make.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import metrics

MIN_OTHER_GAMES = 4          # opponent must have >= this many other box-scored games
TOP_DEFENSE_N = 25           # "top-ranked defense" = top 25 FBS pass defenses by YPA allowed
MIN_GAMES_FOR_RANK = 6       # a defense needs this many games to be ranked
CONF_REL_MIN_ATTEMPTS = 100  # min attempts to enter a conference-season comparison pool

RATE_STATS = ('completions', 'attempts', 'passing_yards', 'passing_tds', 'interceptions')


def team_game_totals(game_passing: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense team) with summed passing, plus the defense faced."""
    g = game_passing.copy()
    for c in RATE_STATS:
        g[c] = pd.to_numeric(g[c], errors='coerce')
    tot = (g.groupby(['season', 'game_id', 'team', 'opponent'], as_index=False, dropna=False)[list(RATE_STATS)]
           .sum(min_count=1))
    return tot.rename(columns={'team': 'offense', 'opponent': 'defense'})


def defense_allowed_loo(team_games: pd.DataFrame) -> pd.DataFrame:
    """Per (season, game, defense): the defense's allowed totals in its OTHER games."""
    tg = team_games.dropna(subset=['defense']).copy()
    season_tot = tg.groupby(['season', 'defense'])[list(RATE_STATS)].sum(min_count=1)
    season_games = tg.groupby(['season', 'defense'])['game_id'].nunique().rename('def_games')
    season_tot = season_tot.join(season_games).reset_index()
    merged = tg.merge(season_tot, on=['season', 'defense'], suffixes=('', '_season'))
    out = merged[['season', 'game_id', 'defense']].copy()
    for c in RATE_STATS:
        out[f'loo_{c}'] = merged[f'{c}_season'] - merged[c].fillna(0)
    out['loo_games'] = merged['def_games'] - 1
    return out.drop_duplicates(['season', 'game_id', 'defense'])


def _rates(df: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
    att = df[f'{prefix}attempts']
    return pd.DataFrame({
        'cmp_pct': metrics.completion_pct(df[f'{prefix}completions'], att),
        'ypa': metrics.yards_per_attempt(df[f'{prefix}passing_yards'], att),
        'td_pct': metrics.td_pct(df[f'{prefix}passing_tds'], att),
        'int_pct': metrics.int_pct(df[f'{prefix}interceptions'], att),
    }, index=df.index)


def season_defense_table(team_games: pd.DataFrame, fbs_teams: pd.DataFrame) -> pd.DataFrame:
    """Full-season pass defense per (season, defense) with FBS ranks by YPA allowed.
    `fbs_teams` has columns season, team (teams classified FBS that season)."""
    tg = team_games.dropna(subset=['defense'])
    d = tg.groupby(['season', 'defense'])[list(RATE_STATS)].sum(min_count=1)
    d['games'] = tg.groupby(['season', 'defense'])['game_id'].nunique()
    d = d.reset_index()
    d = d.join(_rates(d))
    fbs_key = set(map(tuple, fbs_teams[['season', 'team']].itertuples(index=False, name=None)))
    d['is_fbs'] = [(s, t) in fbs_key for s, t in zip(d['season'], d['defense'])]
    rankable = d['is_fbs'] & (d['games'] >= MIN_GAMES_FOR_RANK)
    d['def_pass_rank'] = np.nan
    d.loc[rankable, 'def_pass_rank'] = d[rankable].groupby('season')['ypa'].rank(method='min')
    return d


def league_baselines(season_def: pd.DataFrame) -> pd.DataFrame:
    """Attempt-weighted FBS mean allowed per season (pooled totals)."""
    fbs = season_def[season_def['is_fbs']]
    pooled = fbs.groupby('season')[list(RATE_STATS)].sum(min_count=1).reset_index()
    rates = _rates(pooled)
    rates.columns = [f'league_{c}' for c in rates.columns]
    return pd.concat([pooled[['season']], rates], axis=1)


def qb_game_adjustments(game_passing: pd.DataFrame, fbs_teams: pd.DataFrame,
                        sp_defense: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per QB-game rows with the opponent's leave-one-out baseline and the
    adjusted rates. Only rows with a real cfbd_player_id (never team rows)."""
    tg = team_game_totals(game_passing)
    loo = defense_allowed_loo(tg)
    sdef = season_defense_table(tg, fbs_teams)
    league = league_baselines(sdef)

    # top-defense threshold: the YPA allowed of the N-th ranked FBS defense
    thr = (sdef.dropna(subset=['def_pass_rank'])
           .query('def_pass_rank <= @TOP_DEFENSE_N')
           .groupby('season')['ypa'].max().rename('top_def_ypa_threshold').reset_index())

    qb = game_passing[game_passing['cfbd_player_id'].notna()].copy()
    for c in RATE_STATS:
        qb[c] = pd.to_numeric(qb[c], errors='coerce')
    qb = qb[qb['attempts'].fillna(0) > 0]
    qb = qb.merge(loo, left_on=['season', 'game_id', 'opponent'], right_on=['season', 'game_id', 'defense'],
                  how='left').merge(league, on='season', how='left').merge(thr, on='season', how='left')

    opp_rates = _rates(qb, prefix='loo_')
    opp_rates.columns = [f'opp_{c}' for c in opp_rates.columns]
    qb_rates = _rates(qb)
    qb = pd.concat([qb, opp_rates, qb_rates], axis=1)

    valid = (qb['loo_games'] >= MIN_OTHER_GAMES) & (qb['loo_attempts'] > 0)
    qb['has_opp_baseline'] = valid
    for r in ('cmp_pct', 'ypa', 'td_pct', 'int_pct'):
        dev = qb[f'opp_{r}'] - qb[f'league_{r}']
        qb[f'adj_{r}'] = np.where(valid, qb[r] - dev, np.nan)
        qb[f'opp_dev_{r}'] = np.where(valid, dev, np.nan)
    qb['vs_top_defense'] = valid & (qb['opp_ypa'] <= qb['top_def_ypa_threshold'])

    if sp_defense is not None and not sp_defense.empty:
        qb = qb.merge(sp_defense.rename(columns={'team': 'opponent'}), on=['season', 'opponent'], how='left')
    return qb


def _wavg(values: pd.Series, weights: pd.Series) -> float:
    m = values.notna() & weights.notna() & (weights > 0)
    if not m.any():
        return np.nan
    return float(np.average(values[m], weights=weights[m]))


def qb_season_from_games(qb_games: pd.DataFrame, team_game_totals_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per (cfbd_player_id, season): opponent-adjusted rates, schedule
    strength, top-defense splits, and participation counts."""
    tg = team_game_totals_df
    # leading passer per (game, team): most attempts; ties -> not credited to anyone
    gp = qb_games[['season', 'game_id', 'team', 'cfbd_player_id', 'attempts']]
    mx = gp.groupby(['game_id', 'team'])['attempts'].transform('max')
    n_at_max = gp.assign(is_max=gp['attempts'].eq(mx)).groupby(['game_id', 'team'])['is_max'].transform('sum')
    qb_games = qb_games.assign(is_leading_passer=gp['attempts'].eq(mx) & n_at_max.eq(1))

    rows = []
    for (pid, season), g in qb_games.groupby(['cfbd_player_id', 'season']):
        v = g[g['has_opp_baseline']]
        att_total = g['attempts'].sum()
        att_v = v['attempts'].sum()
        top = g[g['vs_top_defense']]
        rec = {
            'cfbd_player_id': pid, 'season': season,
            'box_attempts': att_total,
            'games_with_pass_attempt': int(g['game_id'].nunique()),
            'games_as_leading_passer': int(g.loc[g['is_leading_passer'], 'game_id'].nunique()),
            'opp_adj_attempt_coverage': (att_v / att_total) if att_total else np.nan,
            'opp_adj_ypa': _wavg(v['adj_ypa'], v['attempts']),
            'opp_adj_completion_pct': _wavg(v['adj_cmp_pct'], v['attempts']),
            'opp_adj_td_pct': _wavg(v['adj_td_pct'], v['attempts']),
            'opp_adj_int_pct': _wavg(v['adj_int_pct'], v['attempts']),
            'sos_pass_defense': _wavg(v['opp_dev_ypa'], v['attempts']),
            'vs_top_def_games': int(top['game_id'].nunique()),
            'vs_top_def_attempts': float(top['attempts'].sum()),
        }
        if top['attempts'].sum() > 0:
            s = top[list(RATE_STATS)].sum()
            rec['vs_top_def_ypa'] = float(metrics.yards_per_attempt(s['passing_yards'], s['attempts']))
            rec['vs_top_def_completion_pct'] = float(metrics.completion_pct(s['completions'], s['attempts']))
            rec['vs_top_def_rating'] = float(metrics.ncaa_passer_rating(
                s['completions'], s['attempts'], s['passing_yards'], s['passing_tds'], s['interceptions']))
        else:
            rec['vs_top_def_ypa'] = rec['vs_top_def_completion_pct'] = rec['vs_top_def_rating'] = np.nan
        if 'opp_def_sp_rating' in g.columns:
            rec['opp_def_sp_rating_avg'] = _wavg(g['opp_def_sp_rating'], g['attempts'])
        rows.append(rec)
    return pd.DataFrame(rows)


def conference_relative(seasons: pd.DataFrame, min_attempts: int = CONF_REL_MIN_ATTEMPTS) -> pd.DataFrame:
    """Attempt-weighted z-scores of NCAA rating and YPA relative to qualifying QBs
    in the same conference-season (pool built from ALL CFBD passers, not just
    drafted ones, so the comparison group is the real competitive context).
    Expects columns: season, conference, cfbd_player_id, attempts, ncaa_passer_rating,
    yards_per_attempt. Rows below `min_attempts` get NaN (too small to compare)."""
    df = seasons.copy()
    out = []
    for (season, conf), g in df.groupby(['season', 'conference'], dropna=True):
        pool = g[g['attempts'] >= min_attempts]
        if len(pool) < 3:
            continue
        res = g[['cfbd_player_id', 'season']].copy()
        for col, name in (('ncaa_passer_rating', 'conf_rel_rating_z'), ('yards_per_attempt', 'conf_rel_ypa_z')):
            w = pool['attempts']
            mu = np.average(pool[col], weights=w)
            sd = np.sqrt(np.average((pool[col] - mu) ** 2, weights=w))
            res[name] = np.where(g['attempts'] >= min_attempts, (g[col] - mu) / sd if sd > 0 else np.nan, np.nan)
        pct = pool['ncaa_passer_rating'].rank(pct=True)
        res['conf_rel_rating_pctile'] = np.nan
        res.loc[pool.index, 'conf_rel_rating_pctile'] = pct
        out.append(res)
    if not out:
        return pd.DataFrame(columns=['cfbd_player_id', 'season', 'conf_rel_rating_z', 'conf_rel_ypa_z',
                                     'conf_rel_rating_pctile'])
    return pd.concat(out, ignore_index=True)
