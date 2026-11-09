"""Survival-analysis preparation.

* kaplan_meier: self-contained product-limit estimator with Greenwood CIs (no
  lifelines dependency), handling right-censoring and tied event times.
* to_lifelines_frame / to_sksurv_arrays: hand the survival_records table to
  lifelines (CoxPHFitter, KaplanMeierFitter) or scikit-survival
  (RandomSurvivalForest) when those optional packages are installed.
* cox_readiness: events-per-variable check before fitting a Cox model.

Censoring semantics come from survival_records (src/features/targets.py):
event_observed=False means the player was still at risk when observation ended
(e.g. appeared in the last two seasons) -- it is NOT a failure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def kaplan_meier(durations, events, alpha: float = 0.05) -> pd.DataFrame:
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    if len(t) == 0:
        return pd.DataFrame(columns=['time', 'at_risk', 'events', 'censored', 'survival', 'lower', 'upper'])
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    times = np.unique(t)
    rows, s, var_sum = [], 1.0, 0.0
    for ti in times:
        at_risk = int((t >= ti).sum())
        d = int(((t == ti) & e).sum())
        c = int(((t == ti) & ~e).sum())
        if at_risk > 0 and d > 0:
            s *= 1 - d / at_risk
            if at_risk > d:
                var_sum += d / (at_risk * (at_risk - d))
        se = s * np.sqrt(var_sum)
        rows.append({'time': ti, 'at_risk': at_risk, 'events': d, 'censored': c, 'survival': s,
                     'lower': max(0.0, s - z * se), 'upper': min(1.0, s + z * se)})
    return pd.DataFrame(rows)


def median_survival(km: pd.DataFrame) -> float:
    below = km[km['survival'] <= 0.5]
    return float(below['time'].iloc[0]) if len(below) else float('nan')


def to_lifelines_frame(records: pd.DataFrame, event_name: str, covariates: pd.DataFrame | None = None) -> pd.DataFrame:
    df = records[records['event_name'] == event_name][['qb_draft_prospect_id', 'duration_seasons', 'event_observed']]
    df = df.rename(columns={'duration_seasons': 'duration', 'event_observed': 'event'})
    df['event'] = df['event'].astype(int)
    if covariates is not None:
        df = df.merge(covariates, on='qb_draft_prospect_id', how='left')
    return df.reset_index(drop=True)


def to_sksurv_arrays(records: pd.DataFrame, event_name: str):
    """Structured array (event: bool, time: float) in scikit-survival's format,
    built with numpy only, plus the aligned prospect ids."""
    df = records[records['event_name'] == event_name]
    y = np.array(list(zip(df['event_observed'].astype(bool), df['duration_seasons'].astype(float))),
                 dtype=[('event', bool), ('time', float)])
    return y, df['qb_draft_prospect_id'].to_numpy()


def cox_readiness(records: pd.DataFrame, event_name: str, n_covariates: int, min_epv: int = 10) -> dict:
    df = records[records['event_name'] == event_name]
    events = int(df['event_observed'].sum())
    return {'event_name': event_name, 'n': int(len(df)), 'events': events,
            'censored': int(len(df) - events), 'events_per_variable': events / max(n_covariates, 1),
            'adequate': events / max(n_covariates, 1) >= min_epv}
