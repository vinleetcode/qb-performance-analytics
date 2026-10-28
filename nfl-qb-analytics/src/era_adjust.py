"""Era-relative z-scores and percentiles.

Default baseline: attempt-weighted mean/sd among *qualifying* players within the
same (season, league) -- i.e. compare a QB only to peers from their own season
and league, never pooling eras or leagues together. An alternative rolling-era
baseline (a trailing window of N seasons, still split by league) is provided
for sensitivity testing, per the project's instruction to test alternative
rolling-era baselines separately rather than only shipping one fixed choice.
Both are stored side-by-side (see EraAdjustedStat.baseline_method) so neither
silently overwrites the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _weighted_mean_sd(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mask = ~np.isnan(values) & ~np.isnan(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.nan, np.nan
    v, w = values[mask], weights[mask]
    mean = np.average(v, weights=w)
    sd = np.sqrt(np.average((v - mean) ** 2, weights=w))
    return float(mean), float(sd)


def _percentile_rank(values: np.ndarray, target: float) -> float:
    valid = values[~np.isnan(values)]
    if len(valid) == 0 or np.isnan(target):
        return np.nan
    return float((valid <= target).sum() / len(valid) * 100)


def season_league_baseline(df: pd.DataFrame, metric: str, weight_col: str,
                            group_cols=('season', 'league_id'),
                            eligible_mask_col: str = 'qualifies_leaderboard') -> pd.DataFrame:
    """Within each (season, league) group, compute weighted mean/sd of `metric`
    among eligible rows, then z-score and percentile EVERY row (eligible or not)
    against that eligible-only baseline."""
    out = df.copy()
    for col in ('baseline_mean', 'baseline_sd', 'baseline_group_size', 'z_score', 'percentile'):
        out[col] = np.nan

    for _, group in out.groupby(list(group_cols)):
        eligible = group[group[eligible_mask_col].fillna(False)]
        values = eligible[metric].to_numpy(dtype=float)
        weights = eligible[weight_col].to_numpy(dtype=float)
        mean, sd = _weighted_mean_sd(values, weights)
        n = int((~np.isnan(values)).sum())

        all_vals = group[metric].to_numpy(dtype=float)
        for idx, val in zip(group.index, all_vals):
            out.at[idx, 'baseline_mean'] = mean
            out.at[idx, 'baseline_sd'] = sd
            out.at[idx, 'baseline_group_size'] = n
            if np.isnan(val) or np.isnan(mean) or not sd or sd == 0 or np.isnan(sd):
                out.at[idx, 'z_score'] = 0.0 if (not np.isnan(val) and (sd == 0)) else np.nan
            else:
                out.at[idx, 'z_score'] = (val - mean) / sd
            out.at[idx, 'percentile'] = _percentile_rank(values, val)
    return out


def rolling_era_baseline(df: pd.DataFrame, metric: str, weight_col: str, window_years: int,
                          group_cols=('league_id',), season_col: str = 'season',
                          eligible_mask_col: str = 'qualifies_leaderboard') -> pd.DataFrame:
    """Same idea as season_league_baseline but the comparison group is a trailing
    +/- window of seasons (within the same league) rather than a single season,
    which smooths small-sample seasons and lets eras be compared on a rolling basis."""
    out = df.copy()
    for col in ('baseline_mean', 'baseline_sd', 'baseline_group_size', 'z_score', 'percentile'):
        out[col] = np.nan

    half = window_years // 2
    for key, league_group in out.groupby(list(group_cols)):
        seasons = sorted(league_group[season_col].unique())
        for season in seasons:
            window = league_group[(league_group[season_col] >= season - half) &
                                   (league_group[season_col] <= season + half)]
            eligible = window[window[eligible_mask_col].fillna(False)]
            values = eligible[metric].to_numpy(dtype=float)
            weights = eligible[weight_col].to_numpy(dtype=float)
            mean, sd = _weighted_mean_sd(values, weights)
            n = int((~np.isnan(values)).sum())

            this_season_rows = league_group[league_group[season_col] == season]
            for idx, val in zip(this_season_rows.index, this_season_rows[metric].to_numpy(dtype=float)):
                out.at[idx, 'baseline_mean'] = mean
                out.at[idx, 'baseline_sd'] = sd
                out.at[idx, 'baseline_group_size'] = n
                if np.isnan(val) or np.isnan(mean) or not sd or sd == 0 or np.isnan(sd):
                    out.at[idx, 'z_score'] = np.nan
                else:
                    out.at[idx, 'z_score'] = (val - mean) / sd
                out.at[idx, 'percentile'] = _percentile_rank(values, val)
    return out
