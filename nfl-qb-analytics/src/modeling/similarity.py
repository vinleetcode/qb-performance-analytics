"""Historical prospect similarity from standardized PRE-DRAFT features only.

Leakage controls:
  * NFL outcomes are never inputs (feature list is checked against the registry
    and must be inside the 'pre_draft_college' or 'pre_draft_full' scope).
  * For a prospect drafted in year Y, the reference pool is prospects drafted
    BEFORE Y, and the standardization (means/SDs) is fit on that pool only, so
    a comp list could have been produced on Y's draft day.

Scaling: raw features live on wildly different scales (attempts ~1,000s vs
rates ~0.1s); unscaled Euclidean distance is dominated by the largest-variance
column. `scaling_effect` quantifies this.

Missing values: distances use only features present for BOTH prospects,
rescaled by sqrt(n_total / n_shared) (sklearn's nan_euclidean convention), and a
pair needs >= MIN_OVERLAP_FRAC shared features to be compared at all. No
imputation is done for distances, so a missing combine drill never looks like
an average (or poor) one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import registry

DEFAULT_SIMILARITY_FEATURES = [
    'final_yards_per_attempt', 'final_completion_pct', 'final_td_pct', 'final_int_pct',
    'final_opp_adj_ypa', 'final_conf_rel_rating_z', 'final_sos_pass_defense',
    'career_pass_attempts', 'n_college_seasons_observed', 'final_qb_rush_share',
    'dev_slope_rating', 'age_at_draft', 'talent_recruiting_4yr_pctile',
]
MIN_OVERLAP_FRAC = 0.7


def check_features(features: list[str]) -> None:
    allowed = set(registry.features_for_scope('pre_draft_full'))
    bad = [f for f in features if f not in allowed]
    if bad:
        raise ValueError(f'similarity features must be pre-draft registry features: {bad}')


def nan_euclidean(a: np.ndarray, B: np.ndarray, min_overlap_frac: float = MIN_OVERLAP_FRAC):
    """Distances from vector a to each row of B, ignoring pairwise-missing
    features. Returns (distances, n_shared); NaN distance if overlap too small."""
    mask = ~np.isnan(a)[None, :] & ~np.isnan(B)
    n_shared = mask.sum(axis=1)
    diff = np.where(mask, B - a[None, :], 0.0)
    d2 = (diff ** 2).sum(axis=1)
    n_total = a.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.sqrt(d2 * n_total / n_shared)
    d[n_shared < np.ceil(min_overlap_frac * n_total)] = np.nan
    return d, n_shared


def historical_comps(df: pd.DataFrame, features: list[str] | None = None, k: int = 5,
                     min_pool: int = 15) -> pd.DataFrame:
    """For each prospect, the k nearest prospects from strictly earlier draft
    classes. `df` needs qb_draft_prospect_id, draft_year and the feature columns."""
    features = features or DEFAULT_SIMILARITY_FEATURES
    check_features(features)
    rows = []
    for _, target in df.iterrows():
        pool = df[df['draft_year'] < target['draft_year']]
        if len(pool) < min_pool:
            continue
        mu = pool[features].mean()
        sd = pool[features].std(ddof=0).replace(0, np.nan)
        Z = ((pool[features] - mu) / sd).to_numpy(dtype=float)
        z = ((target[features] - mu) / sd).to_numpy(dtype=float)
        if np.isnan(z).mean() > 1 - MIN_OVERLAP_FRAC:
            continue
        d, n_shared = nan_euclidean(z, Z)
        order = [i for i in np.argsort(d) if np.isfinite(d[i])][:k]
        for rank, i in enumerate(order, start=1):
            rows.append({'qb_draft_prospect_id': int(target['qb_draft_prospect_id']),
                         'comp_qb_draft_prospect_id': int(pool.iloc[i]['qb_draft_prospect_id']),
                         'rank': rank, 'distance': float(d[i]), 'n_features_compared': int(n_shared[i])})
    return pd.DataFrame(rows, columns=['qb_draft_prospect_id', 'comp_qb_draft_prospect_id', 'rank', 'distance',
                                       'n_features_compared'])


def scaling_effect(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Share of total (unscaled vs standardized) squared-distance variance
    contributed by each feature -- documents why standardization is required."""
    features = features or DEFAULT_SIMILARITY_FEATURES
    X = df[features].astype(float)
    raw_var = X.var()
    z_var = ((X - X.mean()) / X.std(ddof=0)).var()
    return pd.DataFrame({'feature': features,
                         'share_unscaled': (raw_var / raw_var.sum()).values,
                         'share_standardized': (z_var / z_var.sum()).values,
                         'missing_frac': X.isna().mean().values}).sort_values('share_unscaled', ascending=False)


def fit_pca(train: pd.DataFrame, features: list[str] | None = None, n_components: int = 3):
    """PCA fit on TRAIN rows only (median-imputed + standardized with train
    statistics). Returns (pca, transform_fn) -- apply transform_fn to any split."""
    from sklearn.decomposition import PCA
    features = features or DEFAULT_SIMILARITY_FEATURES
    check_features(features)
    med = train[features].median()
    filled = train[features].fillna(med)
    mu, sd = filled.mean(), filled.std(ddof=0).replace(0, 1.0)
    pca = PCA(n_components=n_components, random_state=0).fit(((filled - mu) / sd).to_numpy())

    def transform(df: pd.DataFrame) -> np.ndarray:
        return pca.transform(((df[features].fillna(med) - mu) / sd).to_numpy())
    return pca, transform


def fit_kmeans(train: pd.DataFrame, features: list[str] | None = None, n_clusters: int = 4, seed: int = 0):
    """K-means archetypes fit on TRAIN rows only (same imputation/scaling as PCA)."""
    from sklearn.cluster import KMeans
    features = features or DEFAULT_SIMILARITY_FEATURES
    check_features(features)
    med = train[features].median()
    filled = train[features].fillna(med)
    mu, sd = filled.mean(), filled.std(ddof=0).replace(0, 1.0)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(((filled - mu) / sd).to_numpy())

    def predict(df: pd.DataFrame) -> np.ndarray:
        return km.predict(((df[features].fillna(med) - mu) / sd).to_numpy())
    return km, predict
