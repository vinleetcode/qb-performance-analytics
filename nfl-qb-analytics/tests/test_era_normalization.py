"""Tests for season-relative normalization (src/modeling/era.py): formulas, reference
population, conference fallback, percentiles, zero variance, missing values,
low-attempt QBs, rolling/weighted baselines and the no-future-data property."""
import numpy as np
import pandas as pd
import pytest

from src.modeling import era


def _ref(season, conf, values, attempts=200, games=12, stat='completion_pct', in_ref=True):
    n = len(values)
    return pd.DataFrame({'season': season, 'conference': conf, stat: values, 'attempts': attempts,
                         'games_with_pass_attempt': games, 'in_reference': in_ref,
                         'cfbd_player_id': [f'{season}{conf}{i}' for i in range(n)]})


def _prospect(season, conf, value, attempts=300, games=12, stat='completion_pct'):
    return pd.DataFrame({'season': [season], 'conference': [conf], stat: [value], 'attempts': [attempts],
                         'games_with_pass_attempt': [games]})


# ---------------------------------------------------------------- formulas
def test_zscore_formula_direction_and_clip():
    assert era.zscore(70.0, 60.0, 5.0) == pytest.approx(2.0)
    assert era.zscore(3.0, 2.5, 0.5, direction=-1) == pytest.approx(-1.0)     # INT: more interceptions -> lower score
    assert era.zscore(100.0, 0.0, 1.0, clip=4.0) == pytest.approx(4.0)
    assert era.zscore(-100.0, 0.0, 1.0, clip=4.0) == pytest.approx(-4.0)


def test_zero_or_missing_sd_gives_nan_not_inf():
    assert np.isnan(era.zscore(5.0, 5.0, 0.0))
    assert np.isnan(era.zscore(5.0, 5.0, np.nan))
    assert np.isnan(era.zscore(np.nan, 5.0, 1.0))
    mu, sd, n = era.weighted_mean_sd([7.0])           # a single passer: no spread
    assert n == 1 and np.isnan(sd)


def test_season_z_matches_hand_computation():
    vals = [50.0, 55.0, 60.0, 65.0, 70.0]
    ref = _ref(2010, 'X', vals)
    out = era.normalize_frame(_prospect(2010, 'X', 66.0), ref, stats=['completion_pct'], methods=('season_z',))
    expected = (66.0 - np.mean(vals)) / np.std(vals, ddof=1)
    assert out['completion_pct__season_z'].iloc[0] == pytest.approx(expected)


def test_int_pct_is_sign_flipped_so_higher_is_better():
    ref = _ref(2010, 'X', [1.0, 2.0, 3.0, 4.0, 5.0], stat='int_pct')
    out = era.normalize_frame(_prospect(2010, 'X', 1.0, stat='int_pct'), ref, stats=['int_pct'],
                              methods=('season_z', 'season_pct'))
    assert out['int_pct__season_z'].iloc[0] > 0           # few interceptions -> positive score
    assert out['int_pct__season_pct'].iloc[0] > 0.5


# ---------------------------------------------------------------- percentiles
def test_ecdf_percentile_midrank_ties_and_non_members():
    ref = [1, 2, 2, 3]
    np.testing.assert_allclose(era.ecdf_percentile([2], ref), [(1 + 0.5 * 2) / 4])
    np.testing.assert_allclose(era.ecdf_percentile([0, 10], ref), [0.0, 1.0])  # values outside the reference
    np.testing.assert_allclose(era.ecdf_percentile([2.5], ref), [3 / 4])        # x not in ref
    np.testing.assert_allclose(era.ecdf_percentile([1], ref, direction=-1), [1 - 0.5 / 4])
    assert np.isnan(era.ecdf_percentile([np.nan], ref)[0])
    assert np.isnan(era.ecdf_percentile([1.0], [])[0])


# ---------------------------------------------------------------- reference population
def test_only_reference_rows_define_the_baseline():
    good = _ref(2012, 'X', [58.0, 60.0, 62.0])
    junk = _ref(2012, 'X', [5.0, 99.0], in_ref=False)     # e.g. FCS or < 100-attempt passers
    a = era.normalize_frame(_prospect(2012, 'X', 61.0), good, stats=['completion_pct'], methods=('season_z',))
    b = era.normalize_frame(_prospect(2012, 'X', 61.0), pd.concat([good, junk]), stats=['completion_pct'],
                            methods=('season_z',))
    assert a['completion_pct__season_z'].iloc[0] == pytest.approx(b['completion_pct__season_z'].iloc[0])


def test_reference_population_rule_on_real_data():
    if not era.REF_FILE.exists():
        pytest.skip('run python -m src.modeling.build_dataset_v2 first')
    ref = pd.read_parquet(era.REF_FILE)
    r = ref[ref['in_reference']]
    assert r['is_fbs'].all()
    assert (r['attempts'] >= era.REF_MIN_ATTEMPTS).all() and (r['games_with_pass_attempt'] >= era.REF_MIN_GAMES).all()
    counts = r.groupby('season').size()
    assert counts.index.min() == 2004 and counts.index.max() == 2025 and counts.min() >= 100
    # all college passers, not just drafted QBs: far more passer-seasons than drafted QBs
    assert len(r) > 3000


# ---------------------------------------------------------------- conference
def test_conference_z_uses_conference_group_and_falls_back_when_small():
    big = _ref(2015, 'BIG', [60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0])   # 7 >= CONF_MIN_GROUP
    small = _ref(2015, 'TINY', [40.0, 42.0])                               # 2 < CONF_MIN_GROUP
    ref = pd.concat([big, small], ignore_index=True)
    df = pd.concat([_prospect(2015, 'BIG', 66.0), _prospect(2015, 'TINY', 66.0)], ignore_index=True)
    out = era.normalize_frame(df, ref, stats=['completion_pct'], methods=('season_z', 'conf_z'))
    v = big['completion_pct']
    assert out.loc[0, 'completion_pct__conf_z'] == pytest.approx((66.0 - v.mean()) / v.std(ddof=1))
    assert not out.loc[0, 'completion_pct__conf_fallback']
    assert out.loc[1, 'completion_pct__conf_fallback']
    assert out.loc[1, 'completion_pct__conf_z'] == pytest.approx(out.loc[1, 'completion_pct__season_z'])


def test_conference_is_the_conference_of_that_season():
    # a team that changed conference: 2013 in A, 2014 in B -- each season compares with its own conference
    ref = pd.concat([_ref(2013, 'A', np.linspace(50, 56, 7)), _ref(2013, 'B', np.linspace(60, 66, 7)),
                     _ref(2014, 'A', np.linspace(50, 56, 7)), _ref(2014, 'B', np.linspace(60, 66, 7))])
    df = pd.concat([_prospect(2013, 'A', 58.0), _prospect(2014, 'B', 58.0)], ignore_index=True)
    out = era.normalize_frame(df, ref, stats=['completion_pct'], methods=('conf_z',))
    assert out.loc[0, 'completion_pct__conf_z'] > 0 > out.loc[1, 'completion_pct__conf_z']


# ---------------------------------------------------------------- small samples / missing
def test_low_attempt_seasons_get_no_rate_score_but_keep_raw():
    ref = _ref(2011, 'X', [55.0, 60.0, 65.0])
    df = _prospect(2011, 'X', 80.0, attempts=era.PROSPECT_MIN_ATTEMPTS - 1)
    out = era.normalize_frame(df, ref, stats=['completion_pct'])
    assert out[['completion_pct__season_z', 'completion_pct__season_pct', 'completion_pct__conf_z']].isna().all(axis=None)
    assert out['completion_pct'].iloc[0] == 80.0                        # raw value preserved


def test_per_game_stats_need_games_not_attempts():
    ref = _ref(2011, 'X', [10.0, 20.0, 30.0], stat='rush_yards_per_game')
    ok = era.normalize_frame(_prospect(2011, 'X', 25.0, attempts=10, stat='rush_yards_per_game'), ref,
                             stats=['rush_yards_per_game'], methods=('season_z',))
    none = era.normalize_frame(_prospect(2011, 'X', 25.0, games=0, stat='rush_yards_per_game'), ref,
                               stats=['rush_yards_per_game'], methods=('season_z',))
    assert np.isfinite(ok['rush_yards_per_game__season_z'].iloc[0])
    assert np.isnan(none['rush_yards_per_game__season_z'].iloc[0])


def test_missing_values_and_unknown_seasons_stay_nan():
    ref = _ref(2011, 'X', [55.0, 60.0, 65.0])
    df = pd.concat([_prospect(2011, 'X', np.nan), _prospect(1999, 'X', 60.0)], ignore_index=True)
    out = era.normalize_frame(df, ref, stats=['completion_pct'])
    assert out[['completion_pct__season_z', 'completion_pct__season_pct']].isna().all(axis=None)


def test_zero_variance_season_gives_nan():
    ref = _ref(2011, 'X', [60.0, 60.0, 60.0])
    out = era.normalize_frame(_prospect(2011, 'X', 62.0), ref, stats=['completion_pct'], methods=('season_z',))
    assert np.isnan(out['completion_pct__season_z'].iloc[0])


# ---------------------------------------------------------------- weighting / rolling / leakage
def test_weighted_baseline():
    mu, sd, n = era.weighted_mean_sd([1.0, 3.0], [1.0, 3.0])
    assert mu == pytest.approx(2.5) and sd == pytest.approx(np.sqrt(0.75)) and n == 2


def test_rolling_window_uses_only_past_and_current_seasons():
    ref = pd.concat([_ref(2010, 'X', [50.0, 52.0, 54.0]), _ref(2011, 'X', [60.0, 62.0, 64.0]),
                     _ref(2012, 'X', [90.0, 92.0, 94.0])], ignore_index=True)
    b = era.group_baselines(ref, 'completion_pct', ('season',), window=2).set_index('season')
    assert b.loc[2011, 'mean'] == pytest.approx(np.mean([50, 52, 54, 60, 62, 64]))   # 2010-2011, never 2012
    assert b.loc[2011, 'n'] == 6


def test_future_seasons_cannot_change_a_season_score():
    base = pd.concat([_ref(2014, 'X', [55.0, 60.0, 65.0, 70.0]), _ref(2015, 'X', [56.0, 61.0, 66.0, 71.0])])
    changed = pd.concat([_ref(2014, 'X', [55.0, 60.0, 65.0, 70.0]), _ref(2015, 'X', [10.0, 99.0, 30.0, 80.0])])
    df = _prospect(2014, 'X', 64.0)
    for window in (1, 3):
        a = era.normalize_frame(df, base, stats=['completion_pct'], window=window)
        b = era.normalize_frame(df, changed, stats=['completion_pct'], window=window)
        for m in ('season_z', 'season_pct', 'conf_z'):
            assert a[f'completion_pct__{m}'].iloc[0] == pytest.approx(b[f'completion_pct__{m}'].iloc[0])
