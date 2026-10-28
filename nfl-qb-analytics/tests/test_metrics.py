import math

import numpy as np

from src import metrics


def test_passer_rating_hand_computed_example():
    # comp=20/30 (66.67%), yards=300 (10.0 ypa), td=3 (10.0%), int=1 (3.33%)
    # a=(0.6667-0.3)*5=1.8333, b=(10-3)*0.25=1.75, c=0.10*20=2.0, d=2.375-(0.0333*25)=1.5417
    # rating=(1.8333+1.75+2.0+1.5417)/6*100 = 118.75
    rating = metrics.passer_rating(20, 30, 300, 3, 1)
    assert math.isclose(float(rating), 118.75, abs_tol=0.05)


def test_passer_rating_clips_components_to_max():
    # An absurdly efficient game should cap at the max rating of 158.3, not exceed it.
    rating = metrics.passer_rating(20, 20, 400, 10, 0)
    assert math.isclose(float(rating), 158.3, abs_tol=0.1)


def test_rate_stats_zero_attempts_are_nan_not_crash():
    assert np.isnan(metrics.yards_per_attempt(0, 0))
    assert np.isnan(metrics.completion_pct(0, 0))
    assert np.isnan(metrics.td_pct(0, 0))
    assert np.isnan(metrics.int_pct(0, 0))


def test_adjusted_yards_per_attempt_formula():
    # (300 + 20*3 - 45*1) / 30 = (300+60-45)/30 = 315/30 = 10.5
    aya = metrics.adjusted_yards_per_attempt(300, 3, 1, 30)
    assert math.isclose(float(aya), 10.5, abs_tol=1e-6)


def test_any_a_requires_sack_data():
    # With sacks known: (300+60-45-20) / (30+2) = 295/32
    anya = metrics.adjusted_net_yards_per_attempt(300, 3, 1, 30, sacks_suffered=2, sack_yards_lost=20)
    assert math.isclose(float(anya), 295 / 32, abs_tol=1e-6)
    # With sacks unknown (NaN): must return NaN, never silently assume zero sacks.
    anya_missing = metrics.adjusted_net_yards_per_attempt(300, 3, 1, 30, sacks_suffered=np.nan,
                                                            sack_yards_lost=np.nan)
    assert np.isnan(float(anya_missing))


def test_rushing_metrics():
    assert math.isclose(float(metrics.yards_per_carry(60, 10)), 6.0, abs_tol=1e-9)
    assert math.isclose(float(metrics.rushing_td_pct(2, 10)), 20.0, abs_tol=1e-9)
    assert np.isnan(float(metrics.yards_per_carry(0, 0)))
    # A pure pocket passer with zero carries must not crash and must be NaN, not zero
    # (zero would falsely imply "ran for zero yards on his carries").
    assert np.isnan(float(metrics.yards_per_carry(None, 0)))


def test_total_yards_and_yards_per_play():
    assert math.isclose(float(metrics.total_yards(300, 50)), 350.0, abs_tol=1e-9)
    assert math.isclose(float(metrics.total_tds(2, 1)), 3.0, abs_tol=1e-9)
    # (300 passing + 50 rushing) / (30 attempts + 10 carries) = 350/40 = 8.75
    ypp = metrics.yards_per_play(300, 50, 30, 10)
    assert math.isclose(float(ypp), 8.75, abs_tol=1e-9)
