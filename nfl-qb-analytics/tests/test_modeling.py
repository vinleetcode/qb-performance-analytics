"""Tests for the career-success modeling pipeline: target construction (censoring,
never-played handling), chronological folds, leakage safety, the play-type profile,
feature-set hygiene, and checks on the real built dataset / saved models."""
import json

import numpy as np
import pandas as pd
import pytest

from src.modeling import build_dataset as BD
from src.modeling import evaluate as EV
from src.modeling import features as F
from src.modeling import train as T


# ---------------------------------------------------------------- targets
def _season_rows(pid, starts_by_index):
    return [{'qb_draft_prospect_id': pid, 'season_index': i, 'games_started': s} for i, s in starts_by_index.items()]


def test_targets_censoring_and_never_played():
    prospects = pd.DataFrame({
        'qb_draft_prospect_id': [1, 2, 3, 4, 5, 6],
        'draft_year': [2018, 2018, 2023, 2023, 2019, 2024],
        'gsis_id': ['a', 'b', 'c', 'd', None, None]})
    nfl = pd.DataFrame(
        _season_rows(1, {1: 10, 2: 10, 3: 10, 4: 1, 5: 0, 6: 17})      # 31 in window; season 6 must not count
        + _season_rows(2, {1: 16, 2: 16, 3: 0, 4: 0, 5: 0})            # 32 -> success
        + _season_rows(3, {1: 17, 2: 17, 3: 0})                        # incomplete window, already 34 -> success
        + _season_rows(4, {1: 3, 2: 0, 3: 0}))                         # incomplete, not reached -> NA
    outcomes = pd.DataFrame({'qb_draft_prospect_id': [1, 2, 3, 4, 5, 6],
                             'career_end_status': ['inferred_inactive', 'active_or_censored', 'active_or_censored',
                                                   'active_or_censored', 'never_appeared_window_complete',
                                                   'unlinked_nfl_record']})
    t = BD.build_targets(prospects, nfl, outcomes, obs_end=2025).set_index('qb_draft_prospect_id')
    assert t.loc[1, 'nfl_starts_first_5_seasons'] == 31 and t.loc[1, 'career_success'] == 0
    assert t.loc[2, 'career_success'] == 1
    assert t.loc[3, 'career_success'] == 1 and t.loc[3, 'career_success_status'] == 'achieved_in_partial_window'
    assert pd.isna(t.loc[3, 'nfl_starts_first_5_seasons'])           # no final window total while incomplete
    assert pd.isna(t.loc[4, 'career_success'])                       # censored, NOT a failure
    assert t.loc[4, 'career_success_status'] == 'censored_incomplete_window'
    assert t.loc[5, 'career_success'] == 0 and t.loc[5, 'nfl_starts_first_5_seasons'] == 0   # verified never played
    assert pd.isna(t.loc[6, 'career_success'])                       # missing NFL data is not zero starts
    assert t.loc[6, 'career_success_status'] == 'unverified_nfl_record'


def test_class_drafted_after_observation_end_is_not_yet_observed():
    prospects = pd.DataFrame({'qb_draft_prospect_id': [9], 'draft_year': [2026], 'gsis_id': ['z']})
    outcomes = pd.DataFrame({'qb_draft_prospect_id': [9], 'career_end_status': ['not_yet_observed']})
    nfl = pd.DataFrame(columns=['qb_draft_prospect_id', 'season_index', 'games_started'])
    t = BD.build_targets(prospects, nfl, outcomes, obs_end=2025).iloc[0]
    assert t['career_success_status'] == 'not_yet_observed' and pd.isna(t['career_success'])


def test_three_season_supplementary_target():
    prospects = pd.DataFrame({'qb_draft_prospect_id': [1], 'draft_year': [2023], 'gsis_id': ['a']})
    nfl = pd.DataFrame(_season_rows(1, {1: 8, 2: 8, 3: 0}))
    outcomes = pd.DataFrame({'qb_draft_prospect_id': [1], 'career_end_status': ['active_or_censored']})
    t = BD.build_targets(prospects, nfl, outcomes, obs_end=2025).iloc[0]
    assert t['window_3yr_complete'] and t['nfl_starts_first_3_seasons'] == 16
    assert t['success_3yr_16starts'] == 1 and pd.isna(t['career_success'])


# ---------------------------------------------------------------- folds / leakage
def test_rolling_folds_are_strictly_chronological():
    dy = pd.Series([2017] * 3 + [2018] * 3 + [2019] * 3 + [2020] * 3 + [2021] * 3)
    for ty, tr, te in T.rolling_folds(dy, [2019, 2020, 2021]):
        assert dy.iloc[tr].max() < ty and set(dy.iloc[te]) == {ty}
        assert not set(tr) & set(te)


def _synthetic_cohort(n_per_year=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2017, 2022):
        for i in range(n_per_year):
            r = {c: rng.normal() for c in F.FEATURE_SETS['C_playtype'] + F.DRAFT_REFERENCE}
            r.update({'draft_year': y, 'qb_draft_prospect_id': y * 100 + i, 'player_name': f'qb{y}{i}'})
            r['career_success'] = int(r['final_yards_per_attempt'] + rng.normal() > 0.3)
            rows.append(r)
    return pd.DataFrame(rows)


def test_future_classes_cannot_change_earlier_predictions():
    df = _synthetic_cohort()
    models = T.specs(['A_basic_l2', 'C_playtype_l2', 'C_playtype_gbdt'])
    base, _ = T.run_rolling(df, models, 'career_success', [2019, 2020, 2021], 'x')
    poisoned = df.copy()
    late = poisoned['draft_year'] == 2021
    poisoned.loc[late, F.FEATURE_SETS['C_playtype']] = 1e6       # wreck the latest class (incl. imputer/scaler inputs)
    poisoned.loc[late, 'career_success'] = 1 - poisoned.loc[late, 'career_success']
    after, _ = T.run_rolling(poisoned, models, 'career_success', [2019, 2020, 2021], 'x')
    early = base['test_year'] < 2021
    np.testing.assert_allclose(base.loc[early, 'p_success'].to_numpy(), after.loc[early, 'p_success'].to_numpy())


def test_preprocessing_is_fit_on_training_rows_only():
    df = _synthetic_cohort()
    tr = df[df['draft_year'] < 2021]
    pipe = T.make_pipeline(T.l2()).fit(tr[F.GROUP_A_BASIC], tr['career_success'])
    np.testing.assert_allclose(pipe.named_steps['impute'].statistics_, tr[F.GROUP_A_BASIC].median().to_numpy())
    np.testing.assert_allclose(pipe.named_steps['scale'].mean_, tr[F.GROUP_A_BASIC].mean().to_numpy())


def test_threshold_is_training_prevalence():
    df = _synthetic_cohort()
    p, _ = T.run_rolling(df, T.specs(['A_basic_l2']), 'career_success', [2020], 'x')
    expected = df.loc[df['draft_year'] < 2020, 'career_success'].mean()
    assert np.allclose(p['threshold'], expected)


# ---------------------------------------------------------------- features
def test_feature_sets_are_nested_and_college_only():
    a, b, c = (set(F.FEATURE_SETS[k]) for k in ('A_basic', 'B_context', 'C_playtype'))
    assert a < b < c
    assert c - b == set(F.GROUP_C_PLAYTYPE)
    assert not c & F.POST_DRAFT_COLUMNS
    assert len(c) <= 20                      # deliberately small for ~50 QBs
    d = F.feature_dictionary()
    assert set(d.loc[d['selected'], 'feature']) == c | set(F.DRAFT_REFERENCE)
    assert d.loc[d['selected'], 'availability_cutoff'].str.len().gt(0).all()


def _plays(team, n, pass_share, period=1, margin=0, yards=5, down=1, distance=10, season=2019):
    k = int(round(n * pass_share))
    types = ['Pass Reception'] * k + ['Rush'] * (n - k)
    return pd.DataFrame({'gameId': np.arange(n) % 10, 'offense': team, 'defense': 'X', 'offenseScore': margin,
                         'defenseScore': 0, 'period': period, 'down': down, 'distance': distance, 'yardsToGoal': 50,
                         'yardsGained': yards, 'playType': types, 'ppa': 0.0, 'season': season})


def test_play_type_profile_rates_garbage_time_and_reliability():
    big = pd.concat([_plays('T', 400, 0.5),                               # standard downs, 50% pass
                     _plays('T', 100, 1.0, period=4, margin=30),          # garbage time: excluded from rates
                     _plays('T', 100, 0.8, down=3, distance=9)], ignore_index=True)  # passing downs, 80% pass
    big.loc[big.index[:40], 'yardsGained'] = 25                            # 40 explosive pass plays (first rows are passes)
    small = _plays('FCS', 50, 0.5)
    prof = F.team_play_type_profile(pd.concat([big, small], ignore_index=True)).set_index('team')
    t = prof.loc['T']
    assert t['pt_standard_down_pass_rate'] == pytest.approx(0.5)
    assert t['pt_passing_down_pass_rate'] == pytest.approx(0.8)
    assert t['pt_n_plays'] == 500                                          # garbage-time plays dropped
    assert t['pt_explosive_pass_rate'] == pytest.approx(40 / (200 + 80))
    assert t['pt_plays_per_game'] == pytest.approx(600 / 10)               # tempo counts every scrimmage play
    assert not prof.loc['FCS', 'pt_reliable'] and pd.isna(prof.loc['FCS', 'pt_standard_down_pass_rate'])


# ---------------------------------------------------------------- evaluation helpers
def test_threshold_metrics_counts():
    m = EV.threshold_metrics(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.6, 0.1]), np.array([0.5] * 4))
    assert (m['tp'], m['fn'], m['fp'], m['tn']) == (1, 1, 1, 1)
    assert m['precision'] == 0.5 and m['recall'] == 0.5


def test_fast_bootstrap_metrics_match_sklearn():
    from src.modeling import evaluation as E
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40)
    p = np.round(rng.random(40), 1)              # rounding creates ties
    assert EV.fast_auc(y, p) == pytest.approx(E.roc_auc(y, p))
    assert EV.fast_brier(y, p) == pytest.approx(E.brier_score(y, p))
    assert EV.fast_log_loss(y, np.clip(p, 0.01, 0.99)) == pytest.approx(E.log_loss(y, np.clip(p, 0.01, 0.99)))
    assert np.isnan(EV.fast_auc(np.ones(5), np.random.random(5)))


def test_paired_difference_requires_identical_qbs():
    o = pd.DataFrame({'experiment': 'e', 'model': ['a'] * 4 + ['b'] * 3, 'qb_draft_prospect_id': [1, 2, 3, 4, 1, 2, 3],
                      'y_true': [1, 0, 1, 0, 1, 0, 1], 'p_success': [0.6, 0.4, 0.7, 0.2, 0.5, 0.5, 0.5]})
    with pytest.raises(AssertionError):
        EV.paired_difference(o, 'e', 'a', 'b')


# ---------------------------------------------------------------- real data (skipped if not built)
needs_data = pytest.mark.skipif(not (BD.FEATURES_FILE.exists() and BD.TARGETS_FILE.exists()),
                                reason='run python -m src.modeling.build_dataset first')
needs_models = pytest.mark.skipif(not (T.MODEL_DIR / 'metadata.json').exists(), reason='run python -m src.modeling.train first')


@needs_data
def test_real_dataset_cohort_and_target_integrity():
    d = T.load_data()
    el = d[d['eligible_primary']]
    assert el['draft_year'].between(2017, 2021).all() and el['window_5yr_complete'].all()
    assert el['career_success'].notna().all()
    coh = d[d['comparison_cohort']]
    assert len(coh) >= 50 and coh[F.GROUP_C_PLAYTYPE].notna().all().all()
    assert not coh['final_fcs'].any()        # training support is FBS-only; predict.py relies on this
    # every eligible success is in the cohort (exclusions are FCS non-successes only)
    assert (el['career_success'] == 1).sum() == (coh['career_success'] == 1).sum()
    # success label agrees with the starts count
    ok = d['nfl_starts_first_5_seasons'].notna()
    assert ((d.loc[ok, 'nfl_starts_first_5_seasons'] >= 32).astype(int) == d.loc[ok, 'career_success']).all()
    # censored classes are never labelled failures
    assert not ((d['draft_year'] >= 2022) & (d['career_success'] == 0)).any()


@needs_data
def test_real_targets_match_phase3_outcomes_and_features_have_no_outcomes():
    f = pd.read_parquet(BD.FEATURES_FILE)
    assert not [c for c in f.columns if c.startswith(('started_', 'games_started', 'career_success', 'nfl_starts'))]
    t = pd.read_parquet(BD.TARGETS_FILE)
    o = pd.read_parquet(BD.MODEL_READY / 'nfl_targets.parquet')
    m = t.merge(o[['qb_draft_prospect_id', 'games_started_5yr']], on='qb_draft_prospect_id')
    both = m['nfl_starts_first_5_seasons'].notna() & m['games_started_5yr'].notna()
    assert both.sum() > 200
    assert (m.loc[both, 'nfl_starts_first_5_seasons'] == m.loc[both, 'games_started_5yr']).all()


@needs_data
@needs_models
def test_saved_models_load_and_predict_probabilities():
    from src.modeling import predict as P
    meta = json.loads((T.MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['random_seed'] == T.SEED and 'scikit-learn' in meta['package_versions']
    assert meta['training_cutoff']['latest_draft_class'] == 2021
    d = T.load_data()
    rows = d[d['comparison_cohort']].head(5)
    for name in ('C_playtype_l2', 'draft_reference_l2'):
        s = P.score(rows, P.load_model(name))
        assert s['p_success'].between(0, 1).all() and len(s) == 5
