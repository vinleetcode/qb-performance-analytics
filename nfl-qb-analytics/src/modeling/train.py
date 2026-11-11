"""Train and validate the NFL QB career-success models (chronological, leakage-safe).

Run:  python -m src.modeling.build_dataset && python -m src.modeling.train

DESIGN (fixed before looking at any holdout result)
* Rows: the same-QB comparison cohort (eligible 2017-2021 classes with box-score
  AND play-type features). Every model is fit and scored on exactly these rows.
* Validation: rolling-origin by DRAFT CLASS. Test year 2019 is predicted from
  2017-2018, 2020 from 2017-2019, 2021 from 2017-2020. No QB is ever split at
  random, and nothing from a test class or a later class reaches training,
  imputation, scaling, or hyperparameter choice.
* Preprocessing lives inside sklearn Pipelines (median imputer -> scaler ->
  model), so it is re-fit on each fold's training rows only.
* Hyperparameters are PRE-SPECIFIED, not tuned: with 22-43 training QBs per fold,
  an inner tuning loop would mostly fit noise. A C-sensitivity grid is reported
  afterwards, and it is NOT used to pick a model.
* Decision threshold: the training fold's success rate (prevalence), not 0.5.
* XGBoost is not used: it is not installed, and with ~19 positive events it is
  not statistically justified. sklearn GradientBoostingClassifier (depth 2,
  shallow, subsampled) is included as the tree-ensemble comparison, labelled
  exploratory.

Outputs: results/modeling/*.csv|json and models/qb_career_success/v{V}/*.joblib + metadata.json
"""
from __future__ import annotations

import json
import platform
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import build_dataset as BD
from src.modeling import features as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'modeling'
MODEL_VERSION = '1.0.0'
MODEL_DIR = ROOT / 'models' / 'qb_career_success' / f'v{MODEL_VERSION}'
SEED = 20261110
TEST_YEARS = [2019, 2020, 2021]
SUPP_TEST_YEARS = [2019, 2020, 2021, 2022, 2023]
L2_C = 0.5                      # pre-specified
ENET_C, ENET_L1_RATIO = 0.5, 0.5
SENSITIVITY_C = [0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]
N_BOOT_COEF = 1000
TARGET = 'career_success'
SUPP_TARGET = f'success_{BD.SUPP_HORIZON}yr_{BD.SUPP_STARTS}starts'


def l2(c: float = L2_C) -> LogisticRegression:
    return LogisticRegression(C=c, max_iter=5000)


def enet() -> LogisticRegression:
    return LogisticRegression(C=ENET_C, l1_ratio=ENET_L1_RATIO, solver='saga', max_iter=20000, random_state=SEED)


def gbdt() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=4, random_state=SEED)


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', estimator)])


# name -> (feature set, estimator factory, role)
MODEL_SPECS = {
    'prior_rate': ('A_basic', lambda: DummyClassifier(strategy='prior'), 'Baseline A: training success rate'),
    'A_basic_l2': ('A_basic', l2, 'Baseline B / Model A: basic college stats, L2 logistic'),
    'B_context_l2': ('B_context', l2, 'Baseline C / Model B: + opponent/team context, L2 logistic'),
    'C_playtype_l2': ('C_playtype', l2, 'Model C: + play-type profile, L2 logistic'),
    'A_basic_enet': ('A_basic', enet, 'elastic-net logistic'),
    'B_context_enet': ('B_context', enet, 'elastic-net logistic'),
    'C_playtype_enet': ('C_playtype', enet, 'elastic-net logistic'),
    'A_basic_gbdt': ('A_basic', gbdt, 'gradient-boosted trees (exploratory)'),
    'B_context_gbdt': ('B_context', gbdt, 'gradient-boosted trees (exploratory)'),
    'C_playtype_gbdt': ('C_playtype', gbdt, 'gradient-boosted trees (exploratory)'),
    'draft_reference_l2': ('draft_reference', l2, 'REFERENCE only: log(draft pick); uses draft-day information'),
}
PRIMARY_MODELS = ['prior_rate', 'A_basic_l2', 'B_context_l2', 'C_playtype_l2', 'draft_reference_l2']


def load_data(features_file: Path = BD.FEATURES_FILE, targets_file: Path = BD.TARGETS_FILE) -> pd.DataFrame:
    f = pd.read_parquet(features_file)
    t = pd.read_parquet(targets_file)
    return f.merge(t, on=['qb_draft_prospect_id', 'draft_year'], how='inner', validate='one_to_one')


def rolling_folds(draft_years: pd.Series, test_years) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(test_year, train_idx, test_idx): train strictly earlier classes, test exactly one class."""
    dy = np.asarray(draft_years)
    return [(ty, np.flatnonzero(dy < ty), np.flatnonzero(dy == ty)) for ty in test_years]


def run_rolling(df: pd.DataFrame, models: dict, target: str, test_years, experiment: str,
                train_extra: pd.DataFrame | None = None, importance_for=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample predictions for every model on every rolling fold.
    `train_extra`: additional earlier-class rows appended to every fold's training set
    (they are always older than the test class)."""
    df = df.reset_index(drop=True)
    preds, imps = [], []
    for ty, tr, te in rolling_folds(df['draft_year'], test_years):
        train = df.iloc[tr]
        if train_extra is not None:
            train = pd.concat([train_extra[train_extra['draft_year'] < ty], train], ignore_index=True)
        test = df.iloc[te]
        y_tr = train[target].astype(int).to_numpy()
        y_te = test[target].astype(int).to_numpy()
        assert train['draft_year'].max() < ty <= test['draft_year'].min()
        for name, (fset, factory) in models.items():
            cols = F.FEATURE_SETS[fset]
            pipe = make_pipeline(factory())
            pipe.fit(train[cols], y_tr)
            p = pipe.predict_proba(test[cols])[:, 1]
            preds.append(pd.DataFrame({
                'experiment': experiment, 'model': name, 'feature_set': fset, 'test_year': ty,
                'qb_draft_prospect_id': test['qb_draft_prospect_id'].to_numpy(), 'player_name': test['player_name'].to_numpy(),
                'y_true': y_te, 'p_success': p, 'threshold': float(y_tr.mean()), 'n_train': len(train),
                'n_train_pos': int(y_tr.sum())}))
            if name in importance_for:
                r = permutation_importance(pipe, test[cols], y_te, scoring='neg_brier_score', n_repeats=50,
                                           random_state=SEED)
                imps.append(pd.DataFrame({'experiment': experiment, 'model': name, 'test_year': ty, 'n_test': len(test),
                                          'feature': cols, 'importance_mean': r.importances_mean,
                                          'importance_std': r.importances_std}))
    return pd.concat(preds, ignore_index=True), (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


def specs(names) -> dict:
    return {n: (MODEL_SPECS[n][0], MODEL_SPECS[n][1]) for n in names}


def coefficient_table(pipe: Pipeline, cols) -> pd.DataFrame:
    m = pipe.named_steps['model']
    return pd.DataFrame({'feature': cols, 'coef_standardized': m.coef_[0]})


def bootstrap_coefficients(df: pd.DataFrame, cols, target: str, n_boot: int = N_BOOT_COEF) -> pd.DataFrame:
    """Coefficient stability of the final L2 model under QB-level bootstrap resampling."""
    rng = np.random.default_rng(SEED)
    X, y = df[cols], df[target].astype(int).to_numpy()
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(df), len(df))
        if len(np.unique(y[i])) < 2:
            continue
        pipe = make_pipeline(l2()).fit(X.iloc[i], y[i])
        draws.append(pipe.named_steps['model'].coef_[0])
    d = np.array(draws)
    return pd.DataFrame({'feature': cols, 'boot_median': np.median(d, 0), 'boot_p05': np.quantile(d, 0.05, 0),
                         'boot_p95': np.quantile(d, 0.95, 0), 'share_positive': (d > 0).mean(0),
                         'n_boot_valid': len(d)})


def linear_contributions(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Exact SHAP values for a linear-logit model under feature independence: coef_j * (x_j - mean_j)
    on the standardized scale (log-odds units). Computed directly; the optional `shap` package is not needed."""
    Z = pipe.named_steps['scale'].transform(pipe.named_steps['impute'].transform(X))
    return pd.DataFrame(Z * pipe.named_steps['model'].coef_[0], columns=X.columns, index=X.index)


def package_versions() -> dict:
    import matplotlib
    import scipy
    import sklearn
    return {'python': sys.version.split()[0], 'platform': platform.platform(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scikit-learn': sklearn.__version__, 'scipy': scipy.__version__,
            'joblib': joblib.__version__, 'matplotlib': matplotlib.__version__}


def main() -> dict:
    data = load_data()
    cohort = data[data['comparison_cohort']].sort_values(['draft_year', 'overall_pick']).reset_index(drop=True)
    assert cohort[TARGET].notna().all()
    for fset in ('A_basic', 'B_context', 'C_playtype'):
        assert not set(F.FEATURE_SETS[fset]) & F.POST_DRAFT_COLUMNS, 'draft slot leaked into a college model'
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1) primary experiment: all models, same QBs, rolling origin 2019-2021
    oof, imp = run_rolling(cohort, specs(MODEL_SPECS), TARGET, TEST_YEARS, 'primary_5yr_32starts',
                           importance_for=('C_playtype_l2', 'C_playtype_gbdt', 'B_context_l2'))

    # 2) C sensitivity (reported, not used for selection)
    sens = []
    for c in SENSITIVITY_C:
        m = {f'{fs}_l2_C{c}': (fs, (lambda c=c: l2(c))) for fs in ('A_basic', 'B_context', 'C_playtype')}
        s, _ = run_rolling(cohort, m, TARGET, TEST_YEARS, f'sensitivity_C={c}')
        s['C'] = c
        sens.append(s)
    sens = pd.concat(sens, ignore_index=True)

    # 3) supplementary: add 2005-2016 classes (no play-type data) to training for A/B/draft reference
    hist = data[data['eligible_history']].copy()
    hist_oof, _ = run_rolling(cohort, specs(['prior_rate', 'A_basic_l2', 'B_context_l2', 'draft_reference_l2']),
                              TARGET, TEST_YEARS, 'supp_history_augmented', train_extra=hist)

    # 4) supplementary: shorter horizon, >=16 starts in seasons 1-3, classes 2017-2023
    supp = data[data['eligible_supp_3yr']].sort_values(['draft_year', 'overall_pick']).reset_index(drop=True)
    supp_oof, _ = run_rolling(supp, specs(PRIMARY_MODELS), SUPP_TARGET, SUPP_TEST_YEARS, 'supp_3yr_16starts')

    allp = pd.concat([oof, hist_oof, supp_oof], ignore_index=True)
    allp.to_csv(RESULTS / 'oof_predictions.csv', index=False)
    sens.to_csv(RESULTS / 'sensitivity_predictions.csv', index=False)
    imp.to_csv(RESULTS / 'permutation_importance_by_fold.csv', index=False)

    # 5) final models on all 2017-2021 cohort QBs (for predict.py and explanation)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    y = cohort[TARGET].astype(int).to_numpy()
    saved, coefs = {}, []
    for name, (fset, factory, role) in MODEL_SPECS.items():
        cols = F.FEATURE_SETS[fset]
        pipe = make_pipeline(factory()).fit(cohort[cols], y)
        path = MODEL_DIR / f'{name}.joblib'
        joblib.dump({'pipeline': pipe, 'features': cols, 'model_name': name, 'model_version': MODEL_VERSION,
                     'threshold': float(y.mean())}, path)
        saved[name] = {'file': path.name, 'feature_set': fset, 'features': cols, 'role': role,
                       'params': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                                  for k, v in pipe.named_steps['model'].get_params().items()}}
        if name.endswith('_l2'):
            coefs.append(coefficient_table(pipe, cols).assign(model=name))
    pd.concat(coefs).to_csv(RESULTS / 'final_model_coefficients.csv', index=False)
    cfull = F.FEATURE_SETS['C_playtype']
    bootstrap_coefficients(cohort, cfull, TARGET).to_csv(RESULTS / 'coefficient_bootstrap_C_playtype_l2.csv', index=False)
    final_c = joblib.load(MODEL_DIR / 'C_playtype_l2.joblib')['pipeline']
    contrib = linear_contributions(final_c, cohort[cfull])
    contrib.insert(0, 'player_name', cohort['player_name'])
    contrib.insert(0, 'draft_year', cohort['draft_year'])
    contrib.to_csv(RESULTS / 'linear_contributions_C_playtype_l2.csv', index=False)
    cohort[['qb_draft_prospect_id', 'draft_year', 'player_name'] + cfull + F.DRAFT_REFERENCE + [TARGET]].to_csv(
        RESULTS / 'cohort_features_snapshot.csv', index=False)

    manifest = json.loads((BD.OUT_DIR / 'dataset_manifest.json').read_text(encoding='utf-8'))
    meta = {
        'model_version': MODEL_VERSION, 'created': date.today().isoformat(), 'random_seed': SEED,
        'target': f'{TARGET} = nfl_starts_first_5_seasons >= {BD.SUCCESS_STARTS}',
        'training_rows': int(len(cohort)), 'training_positive': int(y.sum()),
        'training_draft_classes': sorted(cohort['draft_year'].unique().tolist()),
        'training_cutoff': {'latest_draft_class': int(cohort['draft_year'].max()),
                            'outcomes_observed_through': f'{BD.OBS_END_SEASON} NFL regular season',
                            'features_available_by': 'draft day of each class (pre-draft college/combine data only)'},
        'dataset': manifest,
        'feature_schema': {k: v for k, v in F.FEATURE_SETS.items()},
        'feature_version': F.MODEL_FEATURE_VERSION,
        'validation': {'scheme': 'rolling-origin by draft class', 'test_years': TEST_YEARS,
                       'hyperparameters': 'pre-specified; no tuning on holdout data',
                       'threshold_rule': 'training-fold success rate'},
        'models': saved, 'package_versions': package_versions(),
        'format': 'joblib dict {pipeline, features, model_name, model_version, threshold}; load with joblib of the same sklearn version',
    }
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    print(f'cohort {len(cohort)} ({int(y.sum())}+) | oof rows {len(allp)} | models saved {len(saved)} -> {MODEL_DIR}')
    return meta


if __name__ == '__main__':
    main()
