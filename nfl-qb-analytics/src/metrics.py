"""Standardized QB passing-efficiency formulas.

All functions accept scalars, numpy arrays, or pandas Series and return the same
type (vectorized throughout). Attempts of 0 (or NaN) yield NaN rather than a
divide-by-zero error, so missing/zero-attempt rows stay null instead of crashing
a batch computation.
"""
from __future__ import annotations

import numpy as np


def _safe_div(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(denominator > 0, numerator / denominator, np.nan)
    return out


def completion_pct(completions, attempts):
    return _safe_div(completions, attempts) * 100


def yards_per_attempt(passing_yards, attempts):
    return _safe_div(passing_yards, attempts)


def td_pct(passing_tds, attempts):
    return _safe_div(passing_tds, attempts) * 100


def int_pct(interceptions, attempts):
    return _safe_div(interceptions, attempts) * 100


def adjusted_yards_per_attempt(passing_yards, passing_tds, interceptions, attempts):
    """AY/A = (yards + 20*TD - 45*INT) / attempts. Does not require sack data."""
    numerator = np.asarray(passing_yards, dtype=float) + 20 * np.asarray(passing_tds, dtype=float) \
        - 45 * np.asarray(interceptions, dtype=float)
    return _safe_div(numerator, attempts)


def adjusted_net_yards_per_attempt(passing_yards, passing_tds, interceptions, attempts,
                                    sacks_suffered, sack_yards_lost):
    """ANY/A = (yards + 20*TD - 45*INT - sack_yards_lost) / (attempts + sacks).
    Returns NaN wherever sack data is missing rather than silently treating
    missing sacks as zero sacks (which would overstate the rating)."""
    sacks_suffered = np.asarray(sacks_suffered, dtype=float)
    sack_yards_lost = np.asarray(sack_yards_lost, dtype=float)
    attempts_arr = np.asarray(attempts, dtype=float)
    have_sacks = ~np.isnan(sacks_suffered) & ~np.isnan(sack_yards_lost)
    numerator = np.asarray(passing_yards, dtype=float) + 20 * np.asarray(passing_tds, dtype=float) \
        - 45 * np.asarray(interceptions, dtype=float) - sack_yards_lost
    denominator = attempts_arr + sacks_suffered
    result = _safe_div(numerator, denominator)
    return np.where(have_sacks, result, np.nan)


def yards_per_carry(rushing_yards, carries):
    return _safe_div(rushing_yards, carries)


def rushing_td_pct(rushing_tds, carries):
    return _safe_div(rushing_tds, carries) * 100


def total_yards(passing_yards, rushing_yards):
    return np.asarray(passing_yards, dtype=float) + np.asarray(rushing_yards, dtype=float)


def total_tds(passing_tds, rushing_tds):
    return np.asarray(passing_tds, dtype=float) + np.asarray(rushing_tds, dtype=float)


def yards_per_play(passing_yards, rushing_yards, attempts, carries):
    """Combined passing+rushing efficiency per snap of production -- useful for
    comparing dual-threat QBs who generate value outside of pure pass attempts."""
    total_y = total_yards(passing_yards, rushing_yards)
    total_plays = np.asarray(attempts, dtype=float) + np.asarray(carries, dtype=float)
    return _safe_div(total_y, total_plays)


def passer_rating(completions, attempts, passing_yards, passing_tds, interceptions):
    """NFL's official passer rating formula (four components, each clipped to [0, 2.375])."""
    completions = np.asarray(completions, dtype=float)
    attempts_f = np.asarray(attempts, dtype=float)
    passing_yards = np.asarray(passing_yards, dtype=float)
    passing_tds = np.asarray(passing_tds, dtype=float)
    interceptions = np.asarray(interceptions, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid = attempts_f > 0
        a = np.where(valid, ((completions / attempts_f) - 0.3) * 5, np.nan)
        b = np.where(valid, ((passing_yards / attempts_f) - 3) * 0.25, np.nan)
        c = np.where(valid, (passing_tds / attempts_f) * 20, np.nan)
        d = np.where(valid, 2.375 - ((interceptions / attempts_f) * 25), np.nan)

    a = np.clip(a, 0, 2.375)
    b = np.clip(b, 0, 2.375)
    c = np.clip(c, 0, 2.375)
    d = np.clip(d, 0, 2.375)
    return (a + b + c + d) / 6 * 100


def ncaa_passer_rating(completions, attempts, passing_yards, passing_tds, interceptions):
    """NCAA college passer (efficiency) rating -- a different formula from the NFL's,
    with no clipping: ((8.4*Yds) + (330*TD) + (100*Comp) - (200*Int)) / Attempts."""
    numerator = (8.4 * np.asarray(passing_yards, dtype=float)
                 + 330 * np.asarray(passing_tds, dtype=float)
                 + 100 * np.asarray(completions, dtype=float)
                 - 200 * np.asarray(interceptions, dtype=float))
    return _safe_div(numerator, attempts)
