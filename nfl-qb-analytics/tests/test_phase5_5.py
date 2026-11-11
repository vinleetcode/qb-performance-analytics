"""Phase 5.5 tests: eligibility, Combine bias, round probabilities, nested start probabilities,
interval calibration, missing features, similarity stability, cutoffs, censoring, artifacts,
version preservation, forecast schema, historical backtest validity, leakage, secrets."""
from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from src.modeling import artifacts as A
from src.modeling import career_production as CP
from src.modeling import career_production_v11 as V11
from src.modeling import draft_eligibility as DE
from src.modeling import ordinal as O
from src.modeling import round_audit as RA
from src.modeling import similarity_audit as SA

ROOT = A.ROOT
# hashes recorded at the start of Phase 5.5 (results/phase5_5/baseline_model_hashes.txt)
PRESERVED = {
    'models/qb_career_production/v1.0.0/pre_draft.joblib': '8b9d799b3bb79ea4fe29ac416ae0d20a803ba39cfc8dd24e78b7c05179c0b1a4',
    'models/qb_career_production/v1.0.0/post_draft.joblib': 'febf1da327673cef4d18418ccb711ad2d5f86407e44a56d121560338d852b1c4',
    'models/qb_career_production/v1.0.0/metadata.json': '236f73dbb04e6c3f6d3bf4580a2f97f5ef90dcfde8848750cf68a9e4066d2396',
    'models/qb_draft/v1.0.0/prob_college_l2.joblib': '2b7148ba4a6257f39433f61386676fc7492f90c4d92f265fad0ec8efd02e72e9',
    'models/qb_draft/v1.0.0/prob_college_combine_l2.joblib': '35c6b1e30991f43cf53d57728de9767d3c575afb98dc64706b46c1e8d580a34a',
    'models/qb_draft/v1.0.0/round_pick.joblib': '660add757c1d0184212906b2f5488a9820528578aca8bef64fb3e014acba5257',
    'models/qb_draft/v1.0.0/metadata.json': 'fbfd784012c8289efa04fa5a2ba85510a858c217cfb92cb1612eb41aaf561441',
    'models/qb_similarity/v1.0.0/scaler.joblib': 'a5d2c57ca00f3549fa8a81b3ca2742fe0ffa4c45770a55dc639a753a0d6a7cfc',
    'models/qb_similarity/v1.0.0/metadata.json': '3b6791f2ff10d2b2313028d3eac5f8676926f194a78f07344ce81a644483878d',
    'models/qb_career_success/v2.0.0/metadata.json': 'd54774733cf77806b2c2321e5b97258093528ed63a1bbab519d3462e00bb694a',
}
HAVE_V11 = (V11.MODEL_DIR / 'metadata.json').exists() and (RA.MODEL_DIR / 'metadata.json').exists()
needs_v11 = pytest.mark.skipif(not HAVE_V11, reason='run the Phase 5.5 modules first')
HAVE_DATA = (ROOT / 'data' / 'qb_history.db').exists()
needs_data = pytest.mark.skipif(not (HAVE_V11 and HAVE_DATA), reason='needs local data and Phase 5.5 artifacts')


# ---------------------------------------------------------------- 1. draft eligibility
def _elig_toy():
    ref = pd.DataFrame({'cfbd_player_id': ['a', 'a', 'b', 'c', 'd', 'e', 'f', 'g'],
                        'season': [2018, 2019, 2019, 2019, 2019, 2019, 2025, 2019],
                        'team_first': ['X', 'X', 'Y', 'Z', 'W', 'V', 'U', 'T'], 'attempts': [300] * 8, 'is_fbs': [True] * 8})
    pres = pd.DataFrame({'cfbd_player_id': ['a', 'a', 'b', 'b', 'c', 'd', 'e', 'f', 'g', 'g', 'g', 'g', 'z'],
                         'season': [2018, 2019, 2019, 2020, 2019, 2019, 2019, 2025, 2016, 2017, 2018, 2019, 2025],
                         'team': ['X', 'X', 'Y', 'Q', 'Z', 'W', 'V', 'U', 'T', 'T', 'T', 'T', 'R'],
                         'classification': ['fbs'] * 3 + ['fcs'] + ['fbs'] * 9})
    rec = pd.DataFrame({'athlete_id': ['a', 'c', 'd', 'e'], 'class_year': [2015, 2017, 2017, 2015], 'rating': [0.9] * 4})
    drafted = pd.DataFrame({'cfbd_player_id': ['c'], 'draft_year': [2020]})
    return DE.classify(ref, pres, rec, drafted)


def test_eligibility_classes_cover_every_case():
    c = _elig_toy().set_index(['cfbd_player_id', 'season'])['next_status']
    assert c[('a', 2018)] == 'returned_same_school'
    assert c[('b', 2019)] == 'transferred'
    assert c[('c', 2019)] == 'drafted_early_entrant'            # 3 years since HS, drafted next spring
    assert c[('d', 2019)] == 'eligibility_unknown_underclass_exit'
    assert c[('e', 2019)] == 'career_ended_4plus_years'
    assert c[('f', 2025)] == 'final_season_not_verifiable_yet'  # latest season in the data: nothing later to check
    assert c[('g', 2019)] == 'career_ended_4plus_years'         # four observed seasons, no recruit link


def test_missing_future_record_is_never_a_verified_declaration():
    c = _elig_toy()
    unknown = c[c['next_status'].str.startswith('eligibility_unknown') | (c['next_status'] == 'final_season_not_verifiable_yet')]
    assert (unknown['eligibility_class'] == 'eligibility_unknown').all()
    assert not c['eligibility_class'].str.contains('declared').any()


def test_population_definitions_are_nested_where_claimed():
    u = pd.read_parquet(ROOT / 'data' / 'modeling' / 'draft_universe_v5.0.0.parquet') if HAVE_DATA else None
    if u is None:
        pytest.skip('no universe')
    d = DE.population_definitions(u)
    cur, rec, exh, naive = d.values()
    assert not (rec & ~cur).any() and not (exh & ~cur).any() and not (cur & ~naive).any()


# ---------------------------------------------------------------- 2. Combine participation bias
def test_combine_variants_scored_on_identical_rows():
    from src.modeling import combine_audit as CA
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(rng.normal(size=(n, len(CA.DP.COLLEGE_FEATURES))), columns=CA.DP.COLLEGE_FEATURES)
    df['draft_year'] = np.repeat(np.arange(2010, 2026), 25)
    df['combine_invited'] = (rng.random(n) < 0.3).astype(float)
    df['drafted'] = ((df['combine_invited'] + rng.normal(0, 0.5, n)) > 0.8).astype(int)
    for c in CA.MEASURES:
        df[c] = np.where(df['combine_invited'] == 1, rng.normal(size=n), np.nan)
    df['cfbd_player_id'] = [str(i) for i in range(n)]
    b = CA.backtest(df, CA.VARIANTS)
    assert b[list(CA.VARIANTS)].notna().all().all()             # every variant predicted every row
    assert set(b['test_year']) <= set(CA.DP.PROB_TEST_YEARS)


def test_pre_combine_model_uses_no_combine_information():
    from src.modeling import draft_predictor as DP
    assert not set(DP.COLLEGE_FEATURES) & set(DP.COMBINE_FEATURES)
    if HAVE_V11:
        import joblib
        m = joblib.load(DP.MODEL_DIR / 'prob_college_l2.joblib')
        assert not set(m['features']) & set(DP.COMBINE_FEATURES)


# ---------------------------------------------------------------- 3. draft-round probabilities
def _rxy(n=200, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list('abcd'))
    y = np.clip(np.rint(3 + 1.5 * X['a'] + rng.normal(0, 1, n)), 0, 6).astype(int)
    return X, y


@pytest.mark.parametrize('K', [7, 3])
def test_round_probabilities_sum_to_one_for_every_candidate(K):
    X, y = _rxy()
    y = y if K == 7 else np.array([RA.GROUPS[v + 1] for v in y])
    for name, mk in RA.candidates(K).items():
        P = mk().fit(X, y).predict_proba(X)
        assert P.shape == (len(X), K), name
        assert np.allclose(P.sum(1), 1) and (P >= 0).all(), name


def test_proportional_odds_cumulative_probabilities_are_monotone():
    X, y = _rxy()
    P = O.ProportionalOddsK(7, C=0.5).fit(X, y).predict_proba(X)
    assert (np.diff(np.cumsum(P, 1), axis=1) >= -1e-12).all()


def test_rps_is_zero_for_a_perfect_forecast_and_grows_with_distance():
    y = np.array([0, 3, 6])
    assert RA.rps_k(np.eye(7)[y], y) == 0
    near = RA.rps_k(np.eye(7)[np.minimum(y + 1, 6)], y)
    far = RA.rps_k(np.eye(7)[np.minimum(y + 3, 6)], y)
    assert 0 < near < far


# ---------------------------------------------------------------- 4. nested start probabilities
def test_pava_enforces_nesting_and_leaves_coherent_rows_alone():
    p = np.array([[0.6, 0.4, 0.1], [0.3, 0.5, 0.2], [0.2, 0.25, 0.3]])
    q = O.pava_decreasing(p)
    assert O.is_nested(q).all()
    np.testing.assert_allclose(q[0], p[0])
    np.testing.assert_allclose(q[1], [0.4, 0.4, 0.2])
    np.testing.assert_allclose(q.sum(1), p.sum(1))            # least-squares projection preserves the mean


def test_ordinal_exceedance_probabilities_are_nested():
    X, y = _rxy()
    P = O.ProportionalOddsK(4, 0.5).fit(X, np.clip(y // 2, 0, 3)).predict_proba(X)
    assert O.is_nested(O.nested_from_ordinal(P)).all()


# ---------------------------------------------------------------- 5. interval calibration calculations
def test_winkler_score_and_interval_nesting():
    y = np.array([5.0, 15.0])
    assert V11.winkler(y, np.array([0, 0.]), np.array([10, 10.]), 0.8) == pytest.approx((10 + 10 + 2 / 0.2 * 5) / 2)
    spec = {'method': 'oof_scaled', 'quantiles': dict(zip(V11.QS, [-3, -2, -1, 1, 2, 3]))}
    iv = V11.apply_interval(spec, pd.DataFrame(index=range(3)), np.array([0, 10, 100.]), 'pass_yards')
    for i in range(3):
        assert iv[0.9][0][i] <= iv[0.8][0][i] <= iv[0.5][0][i] <= iv[0.5][1][i] <= iv[0.8][1][i] <= iv[0.9][1][i]
    assert (iv[0.9][0] >= 0).all()                               # counts never below zero
    neg = V11.apply_interval(spec, pd.DataFrame(index=range(1)), np.array([0.0]), 'rush_yards')
    assert neg[0.9][0][0] < 0                                    # rushing yards may be negative


def test_randomized_pit_is_uniform_for_a_correct_model():
    rng = np.random.default_rng(0)
    sims = rng.poisson(5, size=(2000, 400)).astype(float)
    y = rng.poisson(5, size=2000).astype(float)
    pit = V11.randomized_pit(sims, y, rng)
    assert abs(np.mean(pit) - 0.5) < 0.02 and abs(np.mean(pit < 0.1) - 0.1) < 0.02


def test_interval_specs_store_only_aggregates():
    d, paths = _toy()
    spec = V11.fit_interval('oof_abs', 'ridge', d, ['x1', 'x2'], 'pass_yards')
    assert set(spec) == {'method', 'quantiles', 'n_residuals'} and len(spec['quantiles']) == len(V11.QS)


# ---------------------------------------------------------------- 6. missing features
def _toy(n=80, seed=0):
    from tests.test_phase5 import _toy_careers
    d, paths = _toy_careers(n, seed)
    d['round_group'] = np.where(d['x1'] > 1.5, 'Round 1', 'Rounds 4-7')
    d['player_name'] = [f'p{i}' for i in range(len(d))]
    return d, paths


def test_point_models_handle_missing_features():
    d, _ = _toy()
    test = d.head(3).copy()
    test.loc[:, 'x1'] = np.nan
    for kind in ('median', 'ridge', 'poisson', 'hurdle', 'round_mean'):
        p = V11.PointModel(kind).fit(d, ['x1', 'x2'], 'pass_yards').predict(test)
        assert np.isfinite(p).all() and (p >= 0).all(), kind


def test_coherence_projection():
    t = pd.DataFrame({'games_started': [90.0, 10], 'games_played': [20.0, 30], 'pass_attempts': [100.0, 50],
                      'pass_completions': [120.0, 30], 'pass_tds': [5.0, 60], 'interceptions': [1.0, 2], 'rush_attempts': [10.0, 5],
                      'rush_tds': [12.0, 1]})
    c = V11.enforce_coherence(t)
    assert (c['games_started'] <= c['games_played']).all() and (c['games_played'] <= 85).all()
    assert (c['pass_completions'] <= c['pass_attempts']).all() and (c['rush_tds'] <= c['rush_attempts']).all()
    pd.testing.assert_series_equal(c['interceptions'], t['interceptions'])


# ---------------------------------------------------------------- 7. similarity stability
def test_similarity_overlap_and_noise_stability():
    from tests.test_phase5 import _pool
    pool = _pool()                                               # 40 QBs, two per class 2005-2024
    base = SA.topk(pool, SA.PS.SIM_FEATURES, years=range(2015, 2025))
    assert SA.overlap(base, base)['mean_jaccard_top5'] == 1.0
    tiny = SA.topk(pool, SA.PS.SIM_FEATURES, years=range(2015, 2025), noise=1e-6, rng=np.random.default_rng(0))
    assert SA.overlap(base, tiny)['mean_jaccard_top5'] > 0.95


# ---------------------------------------------------------------- 8. historical prediction cutoffs
def test_inner_folds_never_train_on_the_validation_class_or_later():
    d, _ = _toy(120)
    for tr, va in V11.inner_folds(d):
        assert tr['draft_year'].max() < va['draft_year'].min()
        assert va['draft_year'].nunique() == 1


def test_selection_ignores_outer_test_outcomes():
    d, _ = _toy(120)
    dev = d[d['draft_year'] <= 2011]
    a = V11.select(dev, ['x1', 'x2'], 'pass_yards', 'pre_draft')
    d2 = d.copy()
    d2.loc[d2['draft_year'] > 2011, [f'{t}_5yr' for t in CP.TARGETS]] = 1e6      # outer outcomes only
    b = V11.select(d2[d2['draft_year'] <= 2011], ['x1', 'x2'], 'pass_yards', 'pre_draft')
    assert a == b


# ---------------------------------------------------------------- 9. trajectory censoring
@needs_data
def test_trajectory_checks_pass_on_real_data():
    from src.modeling import career_trajectory as CT
    from src.modeling import trajectory_audit as TA
    s = pd.read_parquet(CT.OUT)
    c = TA.checks(s, CT.survival_frame(s))
    assert c['indexing_season1_is_draft_year'] and c['duplicate_qb_season_rows'] == 0
    assert c['qbs_with_truncated_observed_seasons'] == 0 and all(c['constraints'].values())
    assert c['appeared_in_last_two_seasons_all_censored'] and c['gap_qbs_marked_ended_at_last_appearance_not_at_gap']


def test_missed_season_is_not_retirement():
    from src.modeling import career_trajectory as CT
    rows = []
    for k, gp in enumerate([16, 0, 0, 10, 0], start=1):          # missed seasons 2-3, came back in 4
        rows.append({'qb_draft_prospect_id': 1, 'draft_year': 2019, 'round': 2, 'round_group': 'Rounds 2-3', 'season_index': k,
                     'nfl_season': 2018 + k, 'status': 'observed', 'games_played': gp, 'games_started': min(gp, 5)})
    sv = CT.survival_frame(pd.DataFrame(rows), obs_end=2023)
    assert sv.iloc[0]['longevity_time'] == 4 or not sv.iloc[0]['longevity_event']


# ---------------------------------------------------------------- 10/11. artifacts and versions
def test_preserved_model_files_unchanged():
    for rel, h in PRESERVED.items():
        p = ROOT / rel
        if p.exists():
            assert A.sha256(p) == h, rel


@needs_v11
def test_every_pinned_model_and_data_file_verifies():
    bad = [r for r in A.check_model_files() + A.check_all() if not (r['exists'] and r['sha256_ok'])]
    assert not bad, bad


@needs_v11
def test_v11_artifacts_have_no_embedded_player_level_arrays():
    import joblib
    for fs in V11.MODES if hasattr(V11, 'MODES') else ('pre_draft', 'post_draft'):
        found = A._walk(joblib.load(V11.MODEL_DIR / f'{fs}.joblib'), min_rows=30)
        assert not found, found[:5]
    found = A._walk(joblib.load(RA.MODEL_DIR / 'round_pick.joblib'), min_rows=30)
    assert not found, found[:5]


@needs_v11
def test_v11_simulation_reproduces_v100_exactly():
    import joblib
    d, _ = CP.load_data()
    for fs, cols in CP.feature_sets().items():
        old = joblib.load(CP.MODEL_DIR / f'{fs}.joblib')['structured']
        new = V11.load(fs)['structured']
        X = d[cols].head(6)
        np.testing.assert_array_equal(old.simulate(X, n_sims=200, seed=5), new.simulate(X, n_sims=200, seed=5))


@needs_v11
def test_v11_without_reference_tables_still_gives_totals_but_not_simulation():
    art = V11.load('pre_draft', with_reference=False)
    assert art['structured'].meaningful_paths is None and not art['reference_tables_loaded']
    d, _ = CP.load_data()
    tot = V11.predict_totals(art, d.head(3))
    assert all(np.isfinite(tot[t]['expected']).all() for t in CP.TARGETS)


@needs_v11
def test_v11_direct_models_match_a_refit():
    d, _ = CP.load_data()
    art = V11.load('pre_draft', with_reference=False)
    for t in ('games_started', 'pass_yards'):
        pm = art['point_models'][t]
        refit = V11.PointModel(pm.kind).fit(d, art['features'], t)
        np.testing.assert_allclose(pm.predict(d), refit.predict(d))


# ---------------------------------------------------------------- 12. unified forecast schema
@needs_data
@pytest.mark.parametrize('kw', [dict(name='Fernando Mendoza', draft_year=2026, mode='pre_draft'),
                                dict(name='Fernando Mendoza', draft_year=2026, mode='post_draft'),
                                dict(name='Jaxson Dart', draft_year=2025, mode='post_draft'),
                                dict(name='Fernando Mendoza', draft_year=2026, mode='post_draft', assumed_pick=40)])
def test_forecast_schema_is_valid(kw):
    from src.modeling.prediction_api import forecast_prospect, validate
    f = forecast_prospect(**kw, n_sims=300)
    assert validate(f) == [], validate(f)
    assert f['career']['status'] == 'available'
    if kw.get('assumed_pick'):
        assert f['career']['hypothetical'] is True


@needs_data
def test_unsupported_predictions_are_structured_unavailable():
    from src.modeling.prediction_api import forecast_prospect, validate
    f = forecast_prospect(cfbd_player_id='4606194', final_season=2024, mode='post_draft', n_sims=200)
    assert validate(f) == []
    assert f['career']['status'] == 'unavailable' and f['career']['reason']
    g = forecast_prospect(name='Fernando Mendoza', draft_year=2026, n_sims=200)
    assert g['draft']['probability']['status'] == 'unavailable'      # 2025 final season: not verifiable yet
    assert 'pre_combine' not in g['draft']['probability']
    h = forecast_prospect(name='Nobody Atall', draft_year=2026)
    assert h['status'] == 'unavailable' and h['reason']


# ---------------------------------------------------------------- 13. historical backtest validity
@needs_data
def test_historical_backtest_labels():
    from src.modeling.historical_backtest import historical_backtest
    a = historical_backtest('Josh Allen', 2018)
    assert a['validity']['status'] == 'valid_out_of_sample_backtest' and not a['validity']['in_sample']
    assert a['training_classes'][1] <= 2018 - 5
    b = historical_backtest('Matt Ryan', 2008)
    assert b['validity']['status'] == 'retrospective_illustration_in_sample' and b['validity']['in_sample']
    c = historical_backtest('C.J. Stroud', 2023)
    assert c['validity']['status'] == 'out_of_sample_forecast_outcome_incomplete'
    assert all(v['error'] is None for v in c['views']['pre_draft']['comparison'].values())


@needs_data
def test_historical_backtest_recompute_reproduces_stored_fold():
    from src.modeling.historical_backtest import historical_backtest
    a = historical_backtest('Josh Allen', 2018)
    b = historical_backtest('Josh Allen', 2018, recompute=True)
    for t, v in a['views']['pre_draft']['comparison'].items():
        assert v['predicted'] == pytest.approx(b['views']['pre_draft']['comparison'][t]['predicted'], rel=1e-9)


# ---------------------------------------------------------------- 14/15. leakage and secrets
def test_career_features_exclude_outcomes_and_draft_slot_pre_draft():
    fs = CP.feature_sets() if HAVE_DATA else None
    if fs is None:
        pytest.skip('no data')
    bad = ('nfl_', 'career_success', 'success_', 'games_', 'overall_pick', 'round')
    assert not [c for c in fs['pre_draft'] if c.startswith(bad)]
    assert set(fs['post_draft']) - set(fs['pre_draft']) == {'log_overall_pick'}


def _secret_values():
    env = ROOT / '.env'
    if not env.exists():
        return []
    vals = []
    for line in env.read_text(encoding='utf-8', errors='ignore').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            v = v.strip().strip('"').strip("'")
            if any(w in k.upper() for w in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) and len(v) >= 8:
                vals.append(v.encode())
    return vals


def test_env_is_gitignored_and_no_committable_file_contains_a_secret():
    r = subprocess.run(['git', 'check-ignore', '.env'], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == '.env'
    secrets = _secret_values()
    if not secrets:
        pytest.skip('no secrets configured locally')
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                           capture_output=True, text=True).stdout.split('\n')
    leaks = []
    for f in filter(None, files):
        p = ROOT / f
        if p.is_file() and p.stat().st_size < 50_000_000:
            b = p.read_bytes()
            leaks += [f for s in secrets if s in b]
    assert not leaks, f'{len(leaks)} committable file(s) contain a configured secret'


def test_no_databases_or_raw_data_would_be_committed():
    files = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT,
                           capture_output=True, text=True).stdout.split()
    bad = [f for f in files if f.endswith(('.db', '.sqlite', '.sqlite3', '.parquet', '.feather', '.npz', '.env'))
           or f.startswith(('data/', 'results/'))]
    assert not bad, bad[:10]
