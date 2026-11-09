"""Model-explainability helpers (for use once models exist).

All of these describe PREDICTIVE ASSOCIATIONS learned by a model on
observational data. None of them identify causal effects: e.g. a high
importance for age_at_draft says the model leans on age, not that being younger
causes NFL success (age is confounded with early-entry decisions, which are
themselves driven by perceived talent).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def permutation_importance_table(model, X: pd.DataFrame, y, scoring: str | None = None, n_repeats: int = 30,
                                 seed: int = 0) -> pd.DataFrame:
    """sklearn permutation importance on a HELD-OUT set (pass validation/test
    data, not training data, to measure generalisation importance)."""
    from sklearn.inspection import permutation_importance
    r = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed)
    return (pd.DataFrame({'feature': X.columns, 'importance_mean': r.importances_mean,
                          'importance_std': r.importances_std})
            .sort_values('importance_mean', ascending=False).reset_index(drop=True))


def shap_values(model, X: pd.DataFrame, background: pd.DataFrame | None = None, max_background: int = 100):
    """SHAP explanations if the optional `shap` package is installed. Raises a
    clear ImportError otherwise (shap is optional, not a hard dependency)."""
    try:
        import shap  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError('SHAP explanations need the optional package: pip install shap') from exc
    import shap
    bg = background if background is not None else X
    if len(bg) > max_background:
        bg = bg.sample(max_background, random_state=0)
    explainer = shap.Explainer(model, bg)
    return explainer(X)


def correlation_analysis(X: pd.DataFrame, method: str = 'spearman', threshold: float = 0.8) -> dict:
    """Pairwise correlations (pairwise-complete) and the list of highly
    correlated feature pairs -- relevant because correlated features split
    permutation importance between them."""
    corr = X.corr(method=method, min_periods=10)
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= threshold:
                pairs.append({'feature_a': a, 'feature_b': b, 'correlation': float(v)})
    return {'matrix': corr, 'high_pairs': pd.DataFrame(pairs, columns=['feature_a', 'feature_b', 'correlation'])}


def partial_dependence_table(model, X: pd.DataFrame, feature: str, grid_resolution: int = 20) -> pd.DataFrame:
    from sklearn.inspection import partial_dependence
    r = partial_dependence(model, X, [feature], grid_resolution=grid_resolution, kind='average')
    grid = r['grid_values'][0] if 'grid_values' in r else r['values'][0]
    return pd.DataFrame({feature: grid, 'partial_dependence': r['average'][0]})


def residual_analysis(y_true, y_pred, groups: pd.Series | None = None) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    resid = y - p
    out = {'summary': {'mean_residual': float(resid.mean()), 'sd_residual': float(resid.std(ddof=1)) if len(resid) > 1 else float('nan'),
                       'corr_fitted_residual': float(np.corrcoef(p, resid)[0, 1]) if len(resid) > 2 else float('nan')},
           'residuals': pd.DataFrame({'fitted': p, 'residual': resid})}
    if groups is not None:
        df = out['residuals'].assign(group=np.asarray(groups))
        out['by_group'] = df.groupby('group')['residual'].agg(['count', 'mean', 'std']).reset_index()
    return out
