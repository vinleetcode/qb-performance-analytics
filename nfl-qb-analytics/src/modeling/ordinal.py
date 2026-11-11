"""Generic ordinal and nested-probability tools (Phase 5.5).

* ProportionalOddsK   proportional-odds (cumulative logit) model for K ordered classes 0..K-1, with one
                      coefficient vector, strictly ordered thresholds and an L2 penalty on the
                      coefficients (sklearn C convention). Class probabilities are non-negative and sum
                      to 1 by construction. Used for draft rounds (7 or 3 groups) and for NFL-start bins.
* nested_from_ordinal turns class probabilities over start bins into P(>= threshold) that are nested
                      by construction: P(64+) <= P(32+) <= P(16+).
* pava_decreasing     row-wise projection of independently estimated exceedance probabilities onto the
                      nested (non-increasing) set: least-squares isotonic regression (pool-adjacent-
                      violators). Leaves already-coherent rows unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ProportionalOddsK:
    def __init__(self, n_classes: int, C: float = 0.5):
        self.K, self.C = int(n_classes), C

    @staticmethod
    def _theta(a, d):
        return a + np.concatenate([[0.0], np.cumsum(np.exp(d))])

    def fit(self, X, y):
        self.prep = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())])
        Z = self.prep.fit_transform(X)
        y = np.asarray(y, dtype=int)
        if y.min() < 0 or y.max() > self.K - 1:
            raise ValueError('classes must be 0..K-1')
        n_f = Z.shape[1]
        freq = np.clip(np.cumsum(np.bincount(y, minlength=self.K))[:-1] / len(y), 0.02, 0.98)
        start = np.log(freq / (1 - freq))
        x0 = np.concatenate([np.zeros(n_f), [start[0]], np.log(np.maximum(np.diff(start), 0.05))])
        idx = np.arange(len(y))

        def nll(w):
            beta, a, d = w[:n_f], w[n_f], w[n_f + 1:]
            th = self._theta(a, d)
            cum = 1 / (1 + np.exp(-(th[None, :] - (Z @ beta)[:, None])))
            cum = np.column_stack([np.zeros(len(Z)), cum, np.ones(len(Z))])
            pr = np.clip(cum[idx, y + 1] - cum[idx, y], 1e-12, 1)
            return -np.log(pr).sum() + (beta @ beta) / (2 * self.C)
        r = minimize(nll, x0, method='L-BFGS-B')
        self.beta, self.a, self.d = r.x[:n_f], r.x[n_f], r.x[n_f + 1:]
        self.converged = bool(r.success)
        return self

    def predict_proba(self, X):
        Z = self.prep.transform(X)
        th = self._theta(self.a, self.d)
        cum = 1 / (1 + np.exp(-(th[None, :] - (Z @ self.beta)[:, None])))
        cum = np.column_stack([np.zeros(len(Z)), cum, np.ones(len(Z))])
        return np.clip(np.diff(cum, axis=1), 0, 1)


def nested_from_ordinal(P: np.ndarray) -> np.ndarray:
    """Class probabilities over ordered bins -> P(class >= k) for k = 1..K-1 (columns), nested by construction."""
    P = np.asarray(P, dtype=float)
    return np.column_stack([P[:, k:].sum(1) for k in range(1, P.shape[1])])


def pava_decreasing(p: np.ndarray) -> np.ndarray:
    """Row-wise least-squares projection onto non-increasing sequences (columns ordered by increasing threshold)."""
    p = np.asarray(p, dtype=float)
    out = p.copy()
    for i, row in enumerate(p):
        blocks = [[v, 1] for v in row]          # (mean, size)
        j = 0
        while j < len(blocks) - 1:
            if blocks[j][0] < blocks[j + 1][0]:  # violation of non-increasing
                m = (blocks[j][0] * blocks[j][1] + blocks[j + 1][0] * blocks[j + 1][1]) / (blocks[j][1] + blocks[j + 1][1])
                blocks[j] = [m, blocks[j][1] + blocks[j + 1][1]]
                del blocks[j + 1]
                j = max(j - 1, 0)
            else:
                j += 1
        out[i] = np.concatenate([[m] * n for m, n in blocks])
    return out


def is_nested(p: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """True per row when exceedance probabilities are non-increasing in the threshold."""
    p = np.asarray(p, dtype=float)
    return (np.diff(p, axis=1) <= tol).all(axis=1)
