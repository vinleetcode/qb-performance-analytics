"""Uncertainty and calibration utilities (Phase 5).

* Cluster bootstrap by DRAFT CLASS: QBs in one class share an NFL/college environment
  and the same evaluation fold, so resampling QBs alone understates uncertainty.
  `cluster_bootstrap` resamples whole classes; with only ~10 classes the intervals
  are themselves rough (they are reported as such).
* Empirical prediction intervals: quantiles of OUT-OF-SAMPLE residuals (never
  in-sample ones), added to a point forecast; their coverage is then MEASURED on
  later test classes rather than assumed.
* Calibration summaries for probabilities: expected calibration error, calibration
  intercept (calibration-in-the-large) and slope.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def cluster_bootstrap(metric: Callable, df: pd.DataFrame, cluster_col: str = 'draft_year', n_boot: int = 1000,
                      seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile CI of metric(sub_df) resampling whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster_col)}
    keys = list(groups)
    point = metric(df)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        v = metric(pd.concat([groups[k] for k in pick]))
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {'estimate': point, 'lower': np.nan, 'upper': np.nan, 'n_clusters': len(keys)}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {'estimate': float(point), 'lower': float(lo), 'upper': float(hi), 'n_clusters': len(keys),
            'n_boot_valid': len(vals)}


def paired_cluster_bootstrap(metric: Callable, df: pd.DataFrame, col_a: str, col_b: str, y_col: str = 'y',
                             cluster_col: str = 'draft_year', n_boot: int = 1000, seed: int = 0) -> dict:
    """metric(y, b) - metric(y, a) with class-level resampling; also P(b better) for a loss metric."""
    def diff(d):
        return metric(d[y_col].to_numpy(), d[col_b].to_numpy()) - metric(d[y_col].to_numpy(), d[col_a].to_numpy())
    r = cluster_bootstrap(diff, df, cluster_col, n_boot, seed)
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in df.groupby(cluster_col)}
    keys = list(groups)
    better = 0
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        better += diff(pd.concat([groups[k] for k in pick])) < 0
    r['p_b_lower'] = better / n_boot
    return r


def residual_quantiles(residuals, levels=(0.1, 0.9)) -> np.ndarray:
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.full(len(levels), np.nan)
    return np.quantile(r, levels)


def empirical_interval(pred, residuals, levels=(0.1, 0.9), lower_bound: float | None = 0.0):
    """[pred + q_lo, pred + q_hi] from out-of-sample residuals (y - pred); clipped at lower_bound."""
    q = residual_quantiles(residuals, levels)
    p = np.asarray(pred, dtype=float)
    lo, hi = p + q[0], p + q[-1]
    if lower_bound is not None:
        lo, hi = np.maximum(lo, lower_bound), np.maximum(hi, lower_bound)
    return lo, hi


def coverage(y, lo, hi) -> float:
    y, lo, hi = (np.asarray(a, dtype=float) for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else float('nan')


def calibration_summary(y, p, n_bins: int = 10) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {'mean_predicted': float(p.mean()), 'observed_rate': float(y.mean()),
           'calibration_in_the_large': float(y.mean() - p.mean())}
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, max(len(edges) - 2, 0))
    ece = 0.0
    for b in np.unique(idx):
        m = idx == b
        ece += m.mean() * abs(p[m].mean() - y[m].mean())
    out['ece'] = float(ece)
    if len(np.unique(y)) == 2:
        lg = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(lg, y)
        out['calibration_slope'] = float(lr.coef_[0, 0])
        out['calibration_intercept'] = float(lr.intercept_[0])
    else:
        out['calibration_slope'] = out['calibration_intercept'] = float('nan')
    return out


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, max(len(edges) - 2, 0))
    return (pd.DataFrame({'bin': idx, 'p': p, 'y': y}).groupby('bin')
            .agg(n=('y', 'size'), mean_predicted=('p', 'mean'), observed=('y', 'mean')).reset_index())
