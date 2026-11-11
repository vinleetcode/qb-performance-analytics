"""Tests for the Phase-4 historical model: label-availability embargo, chronological
splits, eligibility and identity, feature-set hygiene, threshold targets, preservation
of v1, model serialization and reproducible predictions."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.match_prospects import normalize_school, split_schools
from src.modeling import build_dataset_v2 as B2
from src.modeling import features as F
from src.modeling import train as T
from src.modeling import train_v2 as T2


# ---------------------------------------------------------------- label availability / splits
def test_five_year_label_availability_rule():
    assert B2.label_observable_at(2021, 2016)      # seasons 2016-2020 complete before the 2021 draft
    assert not B2.label_observable_at(2021, 2017)  # 2017-2021 window ends after the 2021 draft
    assert not B2.label_observable_at(2019, 2018)


def _frame(years, per=4):
    rows = []
    for y in years:
        for i in range(per):
            rows.append({'draft_year': y, 'qb_draft_prospect_id': y * 10 + i, 'v2_historical': True,
                         'comparison_cohort': y >= 2017, 'career_success': i % 2})
    return pd.DataFrame(rows)


def test_embargoed_training_uses_only_observable_labels():
    df = _frame(range(2005, 2022))
    for ty in (2012, 2016, 2019, 2021):
        tr = df[T2.train_mask(df, ty, 'embargo')]
        assert (tr['draft_year'] + 5 <= ty).all() and tr['draft_year'].max() == ty - 5
        retro = df[T2.train_mask(df, ty, 'retro')]
        assert retro['draft_year'].max() == ty - 1
        modern = df[T2.train_mask(df, ty, 'modern_retro')]
        assert modern.empty or (modern['draft_year'].between(2017, ty - 1)).all()


def test_modern_only_model_cannot_exist_under_the_embargo():
    df = _frame(range(2005, 2022))
    for ty in (2019, 2020, 2021):
        emb = df[T2.train_mask(df, ty, 'embargo')]
        assert not (emb['draft_year'] >= 2017).any()


def test_label_availability_table_counts():
    df = _frame(range(2005, 2022), per=2)
    t = B2.label_availability_table(df, [2019])
    r = t.iloc[0]
    assert r['latest_class_with_observable_label'] == 2014
    assert r['embargoed_train_n'] == 2 * len(range(2005, 2015))
    assert r['retrospective_train_n'] == 2 * len(range(2005, 2019))


def test_run_never_trains_on_test_or_later_classes():
    rng = np.random.default_rng(0)
    df = _frame(range(2005, 2022), per=6)
    df['x1'] = rng.normal(size=len(df))
    df['x2'] = rng.normal(size=len(df))
    df['player_name'] = 'qb'
    cfg = {'m': (['x1', 'x2'], T.l2, 's', 'l', 'l2')}
    for regime in ('embargo', 'retro'):
        p, _ = T2.run(df, cfg, [2016, 2017], regime, 'x', np.ones(len(df), bool))
        assert (p['train_last_class'] < p['test_year']).all()
        if regime == 'embargo':
            assert (p['train_last_class'] + 5 <= p['test_year']).all()


def test_degenerate_training_fold_falls_back_to_training_rate():
    df = _frame(range(2005, 2010), per=4)
    df['career_success'] = 0
    df['x1'] = np.arange(len(df), dtype=float)
    pipe, p, degenerate = T2.fit_predict(df, df.head(3), ['x1'], T.l2, 'career_success')
    assert degenerate and np.allclose(p, 0.0)


def test_selection_prefers_simplest_within_tolerance():
    configs = {'raw|basic|l2': None, 'season_z|basic|l2': None, 'raw|context|gbdt': None}
    dev = pd.DataFrame({'model': ['raw|basic|l2'] * 2 + ['season_z|basic|l2'] * 2 + ['raw|context|gbdt'] * 2,
                        'y_true': [1, 0] * 3, 'p_success': [0.70, 0.30, 0.701, 0.299, 0.90, 0.10]})
    chosen, tab = T2.select(dev, configs)
    assert chosen == 'raw|context|gbdt'                    # clearly better (beyond tolerance) wins
    # now season_z and gbdt are best (Brier 0.0894) and raw is 0.0006 behind -> all tied -> simplest (raw) wins
    dev.loc[dev['model'] == 'raw|context|gbdt', 'p_success'] = [0.701, 0.299]
    chosen, tab = T2.select(dev, configs)
    assert tab['tied_with_best'].all() and chosen == 'raw|basic|l2'
    # a normalized strategy that wins by more than the tolerance is selected
    dev.loc[dev['model'] == 'season_z|basic|l2', 'p_success'] = [0.80, 0.20]
    chosen, _ = T2.select(dev, configs)
    assert chosen == 'season_z|basic|l2'


# ---------------------------------------------------------------- feature hygiene / targets
def test_v2_feature_sets_exclude_draft_slot_and_outcomes():
    sets = B2.feature_sets({s: 'season_z' for s in B2.CONTEXT_SHORT})
    for cols in sets.values():
        assert not set(cols) & F.POST_DRAFT_COLUMNS
        assert not [c for c in cols if c.startswith(('nfl_', 'success_', 'career_success', 'started_', 'pt_'))]
    assert len(sets[('season_z', 'basic')]) == 6 and len(sets[('season_z', 'context')]) == 11


def test_combo_rule_uses_only_reference_drift():
    drift = pd.DataFrame({'feature': B2.CONTEXT_SHORT, 'std_drift': [0.5, -0.4, 0.1, 0.0, 0.29, 0.31, -0.3, 0.0]})
    c = B2.combo_choice(drift)
    assert c['cmp'] == 'season_z' and c['ypa'] == 'season_z' and c['td'] == 'raw'
    assert c['att_pg'] == 'raw' and c['rush_pg'] == 'season_z' and c['opp_ypa'] == 'season_z'


def test_threshold_targets_respect_censoring():
    t = pd.DataFrame({'nfl_starts_first_5_seasons': [10.0, 40.0, np.nan, np.nan],
                      'nfl_starts_first_5_seasons_to_date': [10.0, 40.0, 20.0, np.nan],
                      'career_success': pd.array([0, 1, pd.NA, pd.NA], dtype='Int64')})
    out = B2.threshold_targets(t)
    assert out.loc[0, 'success_5yr_16'] == 0 and out.loc[1, 'success_5yr_64'] == 0
    assert out.loc[2, 'success_5yr_16'] == 1                 # reached in a partial window: a known positive
    assert pd.isna(out.loc[2, 'success_5yr_32'])            # not reached and window incomplete: censored
    assert pd.isna(out.loc[3, 'success_5yr_16'])


# ---------------------------------------------------------------- real data
needs_v2 = pytest.mark.skipif(not (B2.FEATURES_FILE.exists() and B2.TARGETS_FILE.exists()),
                              reason='run python -m src.modeling.build_dataset_v2 first')
needs_model = pytest.mark.skipif(not (T2.MODEL_DIR / 'metadata.json').exists(),
                                 reason='run python -m src.modeling.train_v2 first')


@needs_v2
def test_historical_cohort_integrity_and_identity():
    d = T2.load()
    h = d[d['v2_historical']]
    assert h['draft_year'].between(2005, 2021).all() and h['window_5yr_complete'].all()
    assert h['career_success'].notna().all() and h['identity_source'].notna().all()
    assert (h['v2_final_season'] == h['draft_year'] - 1).all() and h['v2_final_is_fbs'].all()
    assert len(h) >= 150 and h['draft_year'].nunique() == 17
    # identity: the final college team is one of the draft board's schools
    ok = [normalize_school(r.v2_final_team) in {normalize_school(s) for s in split_schools(r.college)}
          for r in h.itertuples()]
    assert np.mean(ok) >= 0.97
    # targets agree with starts, never-played verified QBs are kept as 0 starts
    assert ((h['nfl_starts_first_5_seasons'] >= 32).astype(int) == h['career_success']).all()
    assert (h['target_source'] != 'unlinked').all()


@needs_v2
def test_common_holdout_is_the_v1_evaluation_set():
    d = T2.load()
    v1_hold = d[d['comparison_cohort'] & d['draft_year'].between(2019, 2021)]
    assert set(d.loc[d['v2_common_holdout'], 'qb_draft_prospect_id']) == set(v1_hold['qb_draft_prospect_id'])


@needs_v2
def test_v2_features_have_no_outcome_columns_and_keep_raw_values():
    f = pd.read_parquet(B2.FEATURES_FILE)
    assert not [c for c in f.columns if c.startswith(('started_', 'games_started', 'career_success', 'success_', 'nfl_starts'))]
    for short in B2.CONTEXT_SHORT:
        for m in ('raw', 'season_z', 'season_pct', 'conf_z'):
            assert f'{short}__{m}' in f.columns


@needs_v2
@needs_model
def test_v2_model_serialization_and_reproducible_predictions():
    art = joblib.load(T2.MODEL_DIR / 'historical_selected.joblib')
    assert {'pipeline', 'features', 'model_version', 'threshold'} <= set(art) and art['model_version'] == '2.0.0'
    from src.modeling import artifacts as A
    check = pd.read_csv(A.resolve(T2.MODEL_DIR, 'reproducibility_check'))   # gitignored, hash-verified
    p = art['pipeline'].predict_proba(check[art['features']])[:, 1]
    np.testing.assert_allclose(p, check['p_success_final_model'], rtol=0, atol=1e-10)
    # refitting the same pipeline on the same rows is deterministic
    d = T2.load()
    h = d[d['v2_historical']]
    refit = T.make_pipeline(T.l2() if art['family'] == 'l2' else T.enet()).fit(h[art['features']], h['career_success'].astype(int))
    if art['family'] in ('l2', 'enet'):
        np.testing.assert_allclose(refit.predict_proba(h[art['features']])[:, 1],
                                   art['pipeline'].predict_proba(h[art['features']])[:, 1], atol=1e-10)
    meta = json.loads((T2.MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
    for k in ('training_draft_classes', 'holdout_classes', 'normalization', 'dataset', 'model_params', 'random_seed',
              'package_versions', 'selected_configuration'):
        assert k in meta
    assert A.resolve(T2.MODEL_DIR, 'season_baselines').exists()          # moved to data/model_artifacts (gitignored)


def test_v1_model_directory_is_preserved():
    v1 = T2.V1_MODEL_DIR / 'metadata.json'
    if not v1.exists():
        pytest.skip('v1 not trained')
    meta = json.loads(v1.read_text(encoding='utf-8'))
    assert meta['model_version'] == '1.0.0' and T2.MODEL_DIR != T2.V1_MODEL_DIR
    assert (T2.V1_MODEL_DIR / 'B_context_l2.joblib').exists()
