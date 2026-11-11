"""Reusable evaluation and uncertainty utilities. These compute metrics for
whatever predictions they are given. Model results live in
reports/model_evaluation.md (computed by src/modeling/evaluate.py).

Classification: Brier score, log loss, ROC-AUC, calibration curve (reliability
table), expected calibration error, reliability diagram.
Uncertainty: percentile bootstrap CIs for any metric (optionally resampling
whole draft classes, since prospects in a class share an environment), split-
conformal and residual-bootstrap prediction intervals for regression.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def brier_score(y_true, p) -> float:
    return float(skm.brier_score_loss(np.asarray(y_true, dtype=int), np.asarray(p, dtype=float)))


def log_loss(y_true, p, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return float(skm.log_loss(np.asarray(y_true, dtype=int), p, labels=[0, 1]))


def roc_auc(y_true, p) -> float:
    y = np.asarray(y_true, dtype=int)
    if len(np.unique(y)) < 2:
        return float('nan')  # undefined with a single class
    return float(skm.roc_auc_score(y, np.asarray(p, dtype=float)))


def calibration_table(y_true, p, n_bins: int = 10, strategy: str = 'quantile') -> pd.DataFrame:
    """Reliability table: per bin, mean predicted probability, observed rate, count."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if strategy == 'quantile':
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    idx = np.clip(np.searchsorted(edges, p, side='right') - 1, 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            rows.append({'bin': b, 'bin_lower': edges[b], 'bin_upper': edges[b + 1], 'n': int(m.sum()),
                         'mean_predicted': float(p[m].mean()), 'observed_rate': float(y[m].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, p, n_bins: int = 10, strategy: str = 'quantile') -> float:
    t = calibration_table(y_true, p, n_bins, strategy)
    if t.empty:
        return float('nan')
    return float((t['n'] * (t['mean_predicted'] - t['observed_rate']).abs()).sum() / t['n'].sum())


def reliability_diagram(y_true, p, ax=None, n_bins: int = 10, strategy: str = 'quantile', label: str | None = None):
    import matplotlib.pyplot as plt
    t = calibration_table(y_true, p, n_bins, strategy)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle='--', color='#888888', linewidth=1, label='perfect calibration')
    ax.plot(t['mean_predicted'], t['observed_rate'], marker='o', linewidth=2, label=label or 'model')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    return ax


def classification_summary(y_true, p) -> dict:
    return {'n': int(len(y_true)), 'positive_rate': float(np.mean(y_true)),
            'brier': brier_score(y_true, p), 'log_loss': log_loss(y_true, p), 'roc_auc': roc_auc(y_true, p),
            'ece': expected_calibration_error(y_true, p)}


def bootstrap_ci(metric: Callable, y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 groups=None) -> dict:
    """Percentile bootstrap CI for metric(y_true, y_pred). If `groups` (e.g. draft
    year) is given, whole groups are resampled (cluster bootstrap)."""
    y = np.asarray(y_true)
    yp = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            v = metric(y[i], yp[i])
            if np.isfinite(v):
                stats.append(v)
    else:
        g = np.asarray(groups)
        uniq = np.unique(g)
        members = {u: np.flatnonzero(g == u) for u in uniq}
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            i = np.concatenate([members[u] for u in pick])
            v = metric(y[i], yp[i])
            if np.isfinite(v):
                stats.append(v)
    point = metric(y, yp)
    if not stats:
        return {'estimate': point, 'lower': float('nan'), 'upper': float('nan'), 'n_boot_valid': 0}
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {'estimate': float(point), 'lower': float(lo), 'upper': float(hi), 'n_boot_valid': len(stats)}


def conformal_interval(residuals_calibration, predictions, alpha: float = 0.1) -> pd.DataFrame:
    """Split-conformal prediction intervals: absolute residuals from a held-out
    CALIBRATION set (never the test set) define a symmetric band with
    finite-sample (1-alpha) marginal coverage under exchangeability."""
    r = np.abs(np.asarray(residuals_calibration, dtype=float))
    n = len(r)
    if n == 0:
        raise ValueError('need calibration residuals')
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(r, q_level, method='higher'))
    pred = np.asarray(predictions, dtype=float)
    return pd.DataFrame({'prediction': pred, 'lower': pred - q, 'upper': pred + q})


def bootstrap_prediction_interval(fit_predict: Callable, X_train, y_train, X_new, n_boot: int = 500,
                                  alpha: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Residual-bootstrap prediction intervals. `fit_predict(X_tr, y_tr, X_eval)`
    must return predictions for X_eval; it is refit on each bootstrap resample of
    the TRAINING data only, and a resampled training residual is added."""
    rng = np.random.default_rng(seed)
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    X_new = np.asarray(X_new, dtype=float)
    base_resid = y_train - np.asarray(fit_predict(X_train, y_train, X_train))
    sims = np.empty((n_boot, len(X_new)))
    n = len(y_train)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        pred = np.asarray(fit_predict(X_train[i], y_train[i], X_new))
        sims[b] = pred + rng.choice(base_resid, size=len(X_new), replace=True)
    lo, hi = np.quantile(sims, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame({'prediction': np.asarray(fit_predict(X_train, y_train, X_new)), 'lower': lo, 'upper': hi})


def regression_summary(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return {'n': int(len(y)), 'rmse': float(np.sqrt(np.mean((y - p) ** 2))), 'mae': float(np.mean(np.abs(y - p))),
            'r2': float(skm.r2_score(y, p)) if len(y) > 1 else float('nan')}


def incremental_value_table(y_true, p_base, p_full, metric: Callable = brier_score, n_boot: int = 2000,
                            seed: int = 0, groups=None) -> dict:
    """Paired comparison of a baseline (e.g. draft-position-only) and a fuller
    model on the SAME rows: bootstrap CI of metric(base) - metric(full). For
    loss metrics (Brier/log loss) a positive difference favours the fuller model."""
    y = np.asarray(y_true)
    pb = np.asarray(p_base, dtype=float)
    pf = np.asarray(p_full, dtype=float)
    stacked = np.column_stack([pb, pf])

    def diff(yy, pp):
        return metric(yy, pp[:, 0]) - metric(yy, pp[:, 1])
    return bootstrap_ci(diff, y, stacked, n_boot=n_boot, seed=seed, groups=groups)
