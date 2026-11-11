"""Chronological backtesting with label-availability embargoes (Phase 5).

A forecast for draft class T is made just before the T draft. A training row from
draft class d may be used only if its LABEL was observable by then:

    observable(d, T)  <=>  d + label_lag <= T

  * draft outcomes (drafted? round? pick?)     label_lag = 1  (known on draft day d < T)
  * five-season NFL outcomes                    label_lag = 5  (seasons d..d+4 end before the d+5 draft)
  * three-season NFL outcomes                   label_lag = 3

Features of the TEST rows are pre-draft by construction (checked by the callers'
feature registries); this module guarantees the training side. Every fold records
the training classes it used, so reports can show exactly what was known when.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

LAG_DRAFT = 1
LAG_FIVE_SEASON = 5
LAG_THREE_SEASON = 3


def observable(train_year, test_year, label_lag: int) -> np.ndarray:
    return np.asarray(train_year) + label_lag <= test_year


def folds(years: pd.Series, test_years, label_lag: int, min_train: int = 1):
    """Yield (test_year, train_index, test_index) with the embargo applied."""
    y = np.asarray(years)
    for ty in test_years:
        tr = np.flatnonzero(observable(y, ty, label_lag))
        te = np.flatnonzero(y == ty)
        if len(te) == 0 or len(tr) < min_train:
            continue
        assert (y[tr] + label_lag <= ty).all() and y[tr].max() < ty
        yield ty, tr, te


def run(df: pd.DataFrame, year_col: str, test_years, label_lag: int,
        fit_predict: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame], min_train: int = 1) -> pd.DataFrame:
    """Apply `fit_predict(train, test) -> DataFrame of predictions aligned to test` per fold
    and stack the results with fold metadata."""
    df = df.reset_index(drop=True)
    out = []
    for ty, tr, te in folds(df[year_col], test_years, label_lag, min_train):
        train, test = df.iloc[tr], df.iloc[te]
        p = fit_predict(train, test).copy()
        p.index = test.index
        p['test_year'] = ty
        p['n_train'] = len(train)
        p['train_first_year'] = int(train[year_col].min())
        p['train_last_year'] = int(train[year_col].max())
        out.append(p)
    return pd.concat(out) if out else pd.DataFrame()


def availability_table(years: pd.Series, test_years, label_lag: int) -> pd.DataFrame:
    y = pd.Series(years)
    rows = []
    for ty in test_years:
        m = observable(y, ty, label_lag)
        rows.append({'test_year': ty, 'label_lag': label_lag, 'train_rows': int(m.sum()),
                     'latest_observable_class': ty - label_lag, 'test_rows': int((y == ty).sum())})
    return pd.DataFrame(rows)
