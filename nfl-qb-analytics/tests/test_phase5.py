"""Phase-5 tests: Git-safe artifact paths, v1/v2 preservation, draft universe and labels,
conditional round/pick models, career outcome datasets (zeros vs missing vs censored),
pre/post-draft separation, the structured simulation's consistency and leakage safety,
the comparison engine, uncertainty utilities, chronological folds, serialization and
the integrated forecast."""
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.modeling import artifacts as A
from src.modeling import backtesting as BT
from src.modeling import career_production as CP
from src.modeling import career_trajectory as CT
from src.modeling import draft_predictor as DP
from src.modeling import prospect_similarity as PS
from src.modeling import uncertainty as U

ROOT = A.ROOT

# Hashes captured BEFORE the Phase-5 cleanup (2026-11-10). Model artifacts must never change.
V1_HASHES = {
    'A_basic_enet.joblib': '41c0c6625c1c30885520b87d2a5bf6ec9e2e18e6afa0f93ae902e2a83c9ec6ec',
    'B_context_l2.joblib': '11ebe97b0758ebf58ec6dce6ebe928a52b527b637364473d3f1ef0c66e2f50fa',
    'C_playtype_l2.joblib': 'f2ca6ff3cb11792d861bc3575e84f2ab2c597b1f205f6b0b88e83108ce33b351',
    'draft_reference_l2.joblib': '22f033129137d144245a36850d80a2cff455163db59baf4fd0682fdc6bd03b4a',
    'metadata.json': 'e7aa734e9ee5b131e57bfacf9eb4bcb952aa2a9b728bab6861ecfdda40744a48',
}
V2_HASHES = {'historical_selected.joblib': '8b9d38adef24a05ceb3154b19606a7414939b41033508434a90b69159bad8cae',
             'playtype_stage2.joblib': '8772c9b1d686da0f2c2915854be1b76b68cda95c9cc08fc4e2be33e3651052ab'}
V1 = ROOT / 'models' / 'qb_career_success' / 'v1.0.0'
V2 = ROOT / 'models' / 'qb_career_success' / 'v2.0.0'


# ---------------------------------------------------------------- git cleanup / preservation
def test_v1_and_v2_model_artifacts_unchanged():
    for name, h in V1_HASHES.items():
        assert A.sha256(V1 / name) == h, name
    for name, h in V2_HASHES.items():
        assert A.sha256(V2 / name) == h, name
    assert A.load_metadata(V2)['artifact_sha256'] == V2_HASHES


def test_no_derived_data_files_inside_models():
    bad = [p for p in (ROOT / 'models').rglob('*') if p.suffix in ('.parquet', '.csv', '.feather', '.db')]
    assert not bad, bad


def test_external_data_paths_are_relative_gitignored_and_hash_verified():
    rows = A.check_all()
    assert rows, 'no external data files registered'
    for r in rows:
        assert not r['path'].startswith('/') and ':' not in r['path'] and r['path'].startswith('data/')
        assert r['exists'] and r['sha256_ok'], r
    paths = [r['path'] for r in rows]
    res = subprocess.run(['git', 'check-ignore', *paths], cwd=ROOT, capture_output=True, text=True)
    assert set(res.stdout.split()) == set(paths), 'every external data file must be gitignored'


def test_resolve_explains_how_to_regenerate_missing_files(tmp_path):
    md = tmp_path / 'm'
    md.mkdir()
    (md / 'metadata.json').write_text(json.dumps({'external_data_files': [
        {'role': 'x', 'path': 'data/does_not_exist.parquet', 'sha256': '0', 'regenerate': 'python -m something'}]}))
    with pytest.raises(FileNotFoundError, match='python -m something'):
        A.resolve(md, 'x')


def test_v2_still_loads_and_reproduces_predictions():
    import joblib
    art = joblib.load(V2 / 'historical_selected.joblib')
    chk = pd.read_csv(A.resolve(V2, 'reproducibility_check'))
    np.testing.assert_allclose(art['pipeline'].predict_proba(chk[art['features']])[:, 1], chk['p_success_final_model'], atol=1e-10)
    m = A.load_metadata(V2)
    assert m['normalization']['baselines_file'].startswith('data/model_artifacts/')
    assert A.resolve(V2, 'season_baselines').exists() and A.resolve(V2, 'reference_population').exists()


# ---------------------------------------------------------------- chronological folds / uncertainty
def test_folds_apply_label_embargo():
    years = pd.Series(np.repeat(np.arange(2005, 2022), 3))
    for ty, tr, te in BT.folds(years, [2012, 2018, 2021], BT.LAG_FIVE_SEASON):
        assert years.iloc[tr].max() == ty - 5 and set(years.iloc[te]) == {ty}
    for ty, tr, te in BT.folds(years, [2012], BT.LAG_DRAFT):
        assert years.iloc[tr].max() == 2011
    t = BT.availability_table(years, [2015], 5)
    assert t.iloc[0]['latest_observable_class'] == 2010 and t.iloc[0]['train_rows'] == 3 * 6


def test_interval_and_coverage_helpers():
    lo, hi = U.empirical_interval([10.0, 0.5], residuals=np.arange(-10, 11), levels=(0.1, 0.9), lower_bound=0)
    assert lo[1] == 0 and hi[0] == pytest.approx(10 + 8) and lo[0] == pytest.approx(10 - 8)
    assert U.coverage([1, 5, 9], [0, 0, 0], [2, 6, 8]) == pytest.approx(2 / 3)


def test_cluster_bootstrap_resamples_whole_classes():
    df = pd.DataFrame({'draft_year': [1, 1, 2, 2, 3, 3], 'v': [0, 0, 1, 1, 2, 2]})
    r = U.cluster_bootstrap(lambda d: d['v'].mean(), df, n_boot=300, seed=1)
    assert r['estimate'] == 1 and r['n_clusters'] == 3 and 0 <= r['lower'] <= r['upper'] <= 2


def test_calibration_summary_perfect_model():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    c = U.calibration_summary(y, p)
    assert abs(c['calibration_slope'] - 1) < 0.15 and abs(c['calibration_in_the_large']) < 0.03


# ---------------------------------------------------------------- draft models
def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=['a', 'b', 'c'])
    y = np.clip(np.rint(4 - 1.5 * X['a'] + rng.normal(size=n)), 1, 7).astype(int)
    return X, y


def test_round_probabilities_sum_to_one_and_are_nonnegative():
    X, y = _xy()
    for m in (DP.OrdinalLogit(), DP.Multinomial(), DP.ProportionalOdds()):
        P = m.fit(X, y).predict_proba(X)
        assert P.shape == (len(X), 7) and (P >= -1e-12).all()
        np.testing.assert_allclose(P.sum(1), 1, atol=1e-9)
    P = DP.OrdinalLogit().fit(X, y).predict_proba(X)
    assert (np.diff(np.cumsum(P, 1), axis=1) >= -1e-12).all()        # cumulative probabilities are monotone
    po = DP.ProportionalOdds().fit(X, y)
    assert po.converged and (np.diff(po._theta(po.a, po.d)) > 0).all()  # ordered thresholds
    assert po.beta[0] < 0                                            # higher 'a' -> earlier round in the toy data


def test_ranked_probability_score():
    y = np.array([1, 4])
    perfect = np.eye(7)[y - 1]
    assert DP.rps(perfect, y) == 0
    assert DP.rps(np.eye(7)[[1, 3]], y) < DP.rps(np.eye(7)[[6, 0]], y)   # near misses cost less than far misses


def test_pick_range_intervals_are_ordered_and_clipped():
    X, y = _xy()
    df = X.assign(overall=np.clip((y * 32 - 16) + np.random.default_rng(1).normal(0, 10, len(X)), 1, 262),
                  draft_year=np.repeat(np.arange(2005, 2025), 15))
    pr = DP.PickRange(['a', 'b', 'c']).fit(df).predict(df)
    assert (pr['q10'] <= pr['q25'] + 1e-9).all() and (pr['q75'] <= pr['q90'] + 1e-9).all()
    inside = pr['median_pick'].between(1, 262)
    assert ((pr.loc[inside, 'q25'] <= pr.loc[inside, 'median_pick'] + 1e-9)
            & (pr.loc[inside, 'median_pick'] <= pr.loc[inside, 'q75'] + 1e-9)).all()
    assert pr[['q10', 'q90']].min().min() >= 1 and pr[['q10', 'q90']].max().max() <= 262


def test_draft_features_exclude_draft_outcomes():
    for cols in (DP.COLLEGE_FEATURES, DP.COMBINE_FEATURES, DP.ROUND_FEATURES):
        assert not set(cols) & DP.FORBIDDEN
        assert not [c for c in cols if c.startswith(('nfl_', 'career_', 'success'))]


needs_universe = pytest.mark.skipif(not DP.DATA_FILE.exists(), reason='run python -m src.modeling.draft_predictor')


@needs_universe
def test_draft_universe_rules():
    u = pd.read_parquet(DP.DATA_FILE)
    p = u[u['in_population']]
    assert p['is_fbs'].all() and (p['final_attempts'] >= DP.MIN_FINAL_ATTEMPTS).all()
    assert (p['season'] <= DP.LAST_VERIFIABLE_FINAL).all()
    assert ((p['years_since_hs'] >= 4) | (p['n_seasons_present'] >= 4)).all()
    assert (p['draft_year'] == p['season'] + 1).all()
    # every drafted label comes from an identity source; undrafted rows have no draft info
    assert p.loc[p['drafted'] == 1, 'label_source'].notna().all()
    assert p.loc[p['drafted'] == 0, 'round'].isna().all()
    # underclassmen without the 4-year evidence are never silently labelled undrafted
    assert not u.loc[u['population_status'] == 'three_or_fewer_years_since_high_school', 'in_population'].any()


@needs_universe
def test_draft_universe_final_season_really_is_final():
    src = DP.load_sources()
    pres = DP.presence(src)
    p = pd.read_parquet(DP.DATA_FILE)
    p = p[p['in_population']][['cfbd_player_id', 'season']]
    later = pres.merge(p, on='cfbd_player_id', suffixes=('', '_final'))
    assert not (later['season'] > later['season_final']).any()


@needs_universe
def test_identity_bridge_labels_known_drafted_qbs():
    u = pd.read_parquet(DP.DATA_FILE)
    ryan = u[(u['name'] == 'Matt Ryan') & (u['draft_year'] == 2008)]
    assert len(ryan) == 1 and ryan['drafted'].iloc[0] == 1 and ryan['round'].iloc[0] == 1   # CFBD pick ids differ for old players
    from src.match_prospects import normalize_name
    d = u[(u['drafted'] == 1) & (u['label_source'] == 'qb_identity_bridge')]
    pr = pd.read_parquet(ROOT / 'data' / 'qb_draft_prospects.parquet')
    names = {(r.draft_year, int(r.pick)): normalize_name(r.player_name) for r in pr.itertuples()}
    ok = [names.get((r.draft_year, int(r.overall))) == normalize_name(r.name) for r in d.itertuples()]
    assert np.mean(ok) >= 0.9


# ---------------------------------------------------------------- career outcome datasets
def _toy_inputs():
    prospects = pd.DataFrame({'id': [1, 2, 3, 4], 'draft_year': [2019, 2019, 2019, 2024], 'round': [1, 7, 6, 1],
                              'pick': [5, 240, 200, 3], 'player_name': list('abcd'), 'college': 'x',
                              'gsis_id': ['g1', None, None, 'g4']})
    grid = pd.DataFrame({'qb_draft_prospect_id': [1] * 7 + [4] * 2, 'nfl_season': list(range(2019, 2026)) + [2024, 2025],
                         'games_started': [4, 16, 16, 0, 2, 0, 0, 10, 17]})
    stats = pd.DataFrame({'gsis_id': ['g1', 'g1', 'g4'], 'season': [2019, 2020, 2024], 'games': [6, 16, 11],
                          'attempts': [150, 520, 300], 'completions': [90, 330, 190], 'passing_yards': [1000, 4000, 2100],
                          'passing_tds': [5, 30, 12], 'interceptions': [5, 9, 6], 'sacks_suffered': [9, 20, 15],
                          'sack_yards_lost': [60, 140, 90], 'carries': [20, 50, 60], 'rushing_yards': [80, 200, 300],
                          'rushing_tds': [1, 3, 4]})
    outcomes = pd.DataFrame({'qb_draft_prospect_id': [1, 2, 3, 4],
                             'career_end_status': ['active_or_censored', 'never_appeared_window_complete', 'unlinked_nfl_record',
                                                   'active_or_censored']})
    return prospects, grid, stats, outcomes


def test_zero_production_missing_and_censored_are_distinct():
    s = CT.build_seasons(*_toy_inputs(), obs_end=2025, max_index=6)
    never = s[s['qb_draft_prospect_id'] == 2]
    assert (never.loc[never['status'] == 'observed', ['games_played', 'pass_yards', 'rush_yards']] == 0).all().all()
    unver = s[s['qb_draft_prospect_id'] == 3]
    assert (unver['status'] == 'unverified').all() and unver['pass_yards'].isna().all()
    fut = s[(s['qb_draft_prospect_id'] == 4) & (s['nfl_season'] > 2025)]
    assert (fut['status'] == 'future').all() and fut['games_started'].isna().all()
    one = s[(s['qb_draft_prospect_id'] == 1) & (s['nfl_season'] == 2019)].iloc[0]
    assert one['games_played'] == 6 and one['games_started'] == 4 and one['rush_yards'] == 80
    ob = s[s['status'] == 'observed']
    assert (ob['games_started'] <= ob['games_played']).all()


def test_five_year_totals_only_for_complete_observed_windows():
    s = CT.build_seasons(*_toy_inputs(), obs_end=2025, max_index=6)
    t = CT.five_year_totals(s, obs_end=2025).set_index('qb_draft_prospect_id')
    assert set(t.index) == {1, 2}                       # 3 unverified, 4 incomplete window -> excluded, never 0
    assert t.loc[1, 'games_started_5yr'] == 4 + 16 + 16 + 0 + 2 and t.loc[1, 'pass_yards_5yr'] == 5000
    assert t.loc[2, 'pass_attempts_5yr'] == 0           # verified never-played QB: a real zero


# ---------------------------------------------------------------- pre/post-draft separation
@pytest.mark.skipif(not (ROOT / 'data' / 'modeling' / 'dataset_manifest_v2.0.0.json').exists(), reason='needs v2 dataset')
def test_pre_draft_features_exclude_draft_slot():
    fs = CP.feature_sets()
    assert 'log_overall_pick' not in fs['pre_draft'] and not {'round', 'pick', 'overall_pick'} & set(fs['pre_draft'])
    assert fs['post_draft'] == fs['pre_draft'] + ['log_overall_pick']
    assert not [c for c in fs['post_draft'] if c.startswith(('nfl_', 'career_success', 'success_'))]


# ---------------------------------------------------------------- structured simulation
def _toy_careers(n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows, paths = [], []
    for i in range(n):
        x1, x2 = rng.normal(), rng.normal()
        kind = 0 if i % 5 == 0 else (1 if i % 5 == 1 else 2)
        tot = dict.fromkeys(CP.TARGETS, 0.0)
        for k in range(1, 6):
            line = dict.fromkeys(CP.TARGETS, 0.0)
            if kind == 1 and k == 2:
                line.update(games_played=2, pass_attempts=20, pass_completions=11, pass_yards=130, pass_tds=1, rush_attempts=3, rush_yards=8)
            if kind == 2:
                st = int(rng.integers(0, 17))
                att = int(st * 32 + rng.integers(0, 30))
                line.update(games_played=min(17, st + 1), games_started=st, pass_attempts=att, pass_completions=int(att * 0.62),
                            pass_yards=int(att * 7), pass_tds=int(att * 0.045), interceptions=int(att * 0.025),
                            rush_attempts=int(att * 0.1), rush_yards=int(att * 0.4), rush_tds=int(att * 0.004))
            paths.append({'qb_draft_prospect_id': i, 'season_index': k, **line})
            for t in CP.TARGETS:
                tot[t] += line[t]
        rows.append({'qb_draft_prospect_id': i, 'draft_year': 2005 + i % 10, 'x1': x1 + kind, 'x2': x2,
                     **{f'{t}_5yr': v for t, v in tot.items()}})
    return pd.DataFrame(rows), pd.DataFrame(paths)


def test_structured_simulation_is_internally_consistent():
    d, paths = _toy_careers()
    sm = CP.StructuredModel(['x1', 'x2']).fit(d, paths)
    sims = sm.simulate(d[['x1', 'x2']].head(10), n_sims=300)
    assert all(CP.consistency_checks(sims).values())
    s = sm.summarize(sims)
    assert (s['lo'] <= s['hi']).all() and (s['mean'] >= 0).all()
    np.testing.assert_allclose(s['year_mean'].sum(axis=1), s['mean'], rtol=1e-9)   # totals = sum of seasons


def test_structured_predictions_ignore_test_outcomes():
    d, paths = _toy_careers()
    sm = CP.StructuredModel(['x1', 'x2']).fit(d, paths)
    test = d.head(5).copy()
    a = sm.summarize(sm.simulate(test[['x1', 'x2']], n_sims=200, seed=3))['mean']
    test[[f'{t}_5yr' for t in CP.TARGETS]] = 999999                                # change outcomes only
    b = sm.summarize(sm.simulate(test[['x1', 'x2']], n_sims=200, seed=3))['mean']
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------- comparison engine
def _pool(n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(PS.SIM_FEATURES))), columns=PS.SIM_FEATURES)
    df['qb_draft_prospect_id'] = range(n)
    df['player_name'] = [f'p{i}' for i in range(n)]
    df['college'] = 'x'
    df['draft_year'] = np.repeat(np.arange(2005, 2025), 2)
    df['round'] = 1
    df['overall_pick'] = 10
    df['nfl_starts_first_5_seasons'] = rng.integers(0, 80, n).astype(float)
    df['career_success'] = (df['nfl_starts_first_5_seasons'] >= 32).astype(float)
    df['success_5yr_16'] = 1.0
    df['success_5yr_64'] = 0.0
    df['nfl_starts_first_5_seasons_to_date'] = df['nfl_starts_first_5_seasons']
    df['window_5yr_complete'] = True
    return df


def test_nearest_neighbour_is_the_identical_profile_and_scaler_uses_pool_only():
    pool = _pool()
    c = PS.Comparator().fit(pool)
    q = pool.iloc[7].copy()
    q['qb_draft_prospect_id'] = -1
    r = c.query(q, k=3)
    assert r.iloc[0]['player_name'] == 'p7' and r.iloc[0]['distance'] == pytest.approx(0)
    assert (np.diff(r['distance']) >= 0).all() and (np.diff(r['closeness_index_heuristic']) <= 0).all()
    np.testing.assert_allclose(c.scaler.mean_, pool[PS.SIM_FEATURES].mean().to_numpy())


def test_comparisons_exclude_future_players_and_hide_unobservable_outcomes():
    pool = _pool()
    q = pool[pool['draft_year'] == 2018].iloc[0]
    c = PS.comps_for(pool, q, k=10, backtest=True)
    assert (c['draft_year'] < 2018).all()
    assert c.loc[c['draft_year'] > 2013, 'nfl_starts_first_5_seasons'].isna().all()
    assert c.loc[c['draft_year'] <= 2013, 'nfl_starts_first_5_seasons'].notna().all()


def test_similarity_features_are_pre_draft_only():
    assert not [f for f in PS.SIM_FEATURES if f.startswith(('nfl_', 'career_success', 'success_', 'log_overall'))]
    assert 'overall_pick' not in PS.SIM_FEATURES and 'round' not in PS.SIM_FEATURES


# ---------------------------------------------------------------- serialization + integrated forecast
needs_models = pytest.mark.skipif(not all((ROOT / 'models' / m / 'v1.0.0' / 'metadata.json').exists()
                                          for m in ('qb_draft', 'qb_career_production', 'qb_similarity')),
                                  reason='run the Phase-5 modules first')


@needs_models
def test_phase5_artifacts_load_and_hashes_match():
    import joblib
    for m in ('qb_draft', 'qb_career_production', 'qb_similarity'):
        d = ROOT / 'models' / m / 'v1.0.0'
        meta = A.load_metadata(d)
        for name, h in meta['artifact_sha256'].items():
            assert A.sha256(d / name) == h
            joblib.load(d / name)


@needs_models
def test_integrated_forecast_outputs():
    from src.modeling.forecast import forecast_prospect
    f = forecast_prospect(name='Fernando Mendoza', draft_year=2026, n_sims=300)
    assert set(f) == {'prospect', 'draft', 'career', 'comps', 'evaluation_status'}
    d = f['draft']
    assert abs(sum(d['round_probabilities_if_drafted'].values()) - 1) < 1e-9
    lo, hi = d['pick_if_drafted']['interval_80']
    assert 1 <= lo <= hi <= 262
    for fs in ('pre_draft', 'post_draft'):
        blk = f['career'][fs]
        for k in (16, 32, 64):
            assert 0 <= blk['starter'][f'p_{k}plus_starts_logit'] <= 1
        for t, v in blk['five_year'].items():
            if t in CP.TARGETS:
                assert v['interval_80'][0] <= v['interval_80'][1]
        assert len(blk['year_by_year']) == 5
    assert len(f['comps']) == PS.K and all(c['draft_year'] < 2026 for c in f['comps'])
    assert 0 <= f['career']['p32_v2_reference'] <= 1
