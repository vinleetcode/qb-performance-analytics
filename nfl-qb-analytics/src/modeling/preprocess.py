"""Chronological splits and train-only preprocessing.

* Splits are by DRAFT CLASS (config/modeling_splits.json), never random.
* TrainOnlyPreprocessor refuses to be fit on anything but train rows, and
  learns medians / means / SDs from train only; validation and test rows are
  only ever transformed. Missing-value indicators are added so that imputation
  never silently hides systematic missingness (e.g. combine drills).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import registry

ROOT = Path(__file__).resolve().parents[2]
SPLIT_CONFIG = ROOT / 'config' / 'modeling_splits.json'


def load_split_config(path: Path | None = None) -> dict:
    return json.loads((path or SPLIT_CONFIG).read_text(encoding='utf-8'))


def assign_split(draft_year: int, config: dict | None = None) -> str:
    cfg = config or load_split_config()
    if draft_year < cfg['college_feature_floor_draft_year']:
        return 'pre_coverage'
    for name, (lo, hi) in cfg['splits'].items():
        if lo <= draft_year <= hi:
            return name
    lo, hi = cfg['censored_holdout']
    if lo <= draft_year <= hi:
        return 'censored_holdout'
    return 'unassigned'


def build_model_matrix(features: pd.DataFrame, targets: pd.DataFrame, scope: str, target: str,
                       config: dict | None = None):
    """Returns (X, y, meta) for one target and model scope.
    Rows: prospects in train/validation/test splits with a NON-NULL target
    (NULL = right-censored/unobservable, so it is excluded rather than coded 0).
    Columns: exactly the registry features allowed in `scope` that exist in `features`."""
    cfg = config or load_split_config()
    cols = [c for c in registry.features_for_scope(scope) if c in features.columns]
    registry.assert_scope_safe(cols, scope)
    df = features[['qb_draft_prospect_id', 'draft_year'] + cols].merge(
        targets[['qb_draft_prospect_id', target]], on='qb_draft_prospect_id', how='inner')
    df['split'] = df['draft_year'].map(lambda y: assign_split(int(y), cfg))
    df = df[df['split'].isin(['train', 'validation', 'test']) & df[target].notna()]
    X = df[cols].astype(float)
    y = df[target].astype(float)
    meta = df[['qb_draft_prospect_id', 'draft_year', 'split']]
    return X.reset_index(drop=True), y.reset_index(drop=True), meta.reset_index(drop=True)


class TrainOnlyPreprocessor:
    """Median imputation + missing indicators + standardization, fit on train only."""

    def __init__(self, add_indicators: bool = True, min_non_missing_frac: float = 0.0):
        self.add_indicators = add_indicators
        self.min_non_missing_frac = min_non_missing_frac
        self.fitted_ = False

    def fit(self, X: pd.DataFrame, splits: pd.Series):
        splits = pd.Series(splits).reset_index(drop=True)
        if len(splits) != len(X):
            raise ValueError('splits must align with X')
        if not splits.eq('train').all():
            raise ValueError('TrainOnlyPreprocessor.fit received non-train rows '
                             f'({sorted(set(splits) - {"train"})}); fit on the train split only.')
        X = X.reset_index(drop=True)
        frac = X.notna().mean()
        self.columns_ = [c for c in X.columns if frac[c] > self.min_non_missing_frac]
        self.dropped_all_missing_ = [c for c in X.columns if c not in self.columns_]
        Xc = X[self.columns_]
        self.medians_ = Xc.median()
        self.indicator_cols_ = [c for c in self.columns_ if Xc[c].isna().any()] if self.add_indicators else []
        filled = Xc.fillna(self.medians_)
        self.means_ = filled.mean()
        self.sds_ = filled.std(ddof=0).replace(0, 1.0)
        self.n_train_ = len(X)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError('call fit() first')
        X = X.reset_index(drop=True)
        Xc = X.reindex(columns=self.columns_)
        out = (Xc.fillna(self.medians_) - self.means_) / self.sds_
        if self.indicator_cols_:
            ind = Xc[self.indicator_cols_].isna().astype(float).add_suffix('__missing')
            out = pd.concat([out, ind], axis=1)
        return out


def missingness_by_class(features: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Share of missing values per feature per draft class (for coverage reports)."""
    cols = columns or [c for c in features.columns if c in registry.FEATURES]
    return features.groupby('draft_year')[cols].apply(lambda d: d.isna().mean())
