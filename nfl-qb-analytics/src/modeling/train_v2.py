"""Phase 4: historical (2005-2021) training with era-normalized college features.

Run:  python -m src.modeling.build_dataset_v2 && python -m src.modeling.train_v2

PRE-REGISTERED PROTOCOL (fixed in this file before any holdout number was looked at)
1. Development = embargoed expanding window over test classes 2012-2018. For test
   class T the training set is every historical-eligible class d with d + 5 <= T,
   i.e. only classes whose five-season label was OBSERVABLE at the T draft.
2. Selection (normalization strategy x feature level x model family) uses ONLY the
   pooled development Brier score. Configurations within SIMPLICITY_TOL of the best
   are treated as tied, and the simplest wins, in this order: strategy raw_v1, raw,
   season_z, season_pct, conf_z, combo; level basic before context; family L2,
   elastic net, gradient boosting. Normalization therefore has to earn its place.
3. Final holdout = draft classes 2019-2021, restricted to the same 31 QBs v1 was
   evaluated on (the common cohort). Untouched until step 2 was complete. Two regimes:
     retrospective: train on every eligible class < T (what v1 did -- note that its
                    later training labels were not yet observable at the T draft);
     embargoed:     train on classes d <= T - 5 (deployable at the T draft). The
                    modern-only model cannot exist in this regime (no 2017+ class has
                    an observable five-season label before the 2022 draft).
4. Preference rule: v2 replaces v1 as the preferred model only if, on the holdout
   (retrospective regime, the only one where v1 exists), its Brier AND log loss are
   both lower than v1's B_context_l2. Otherwise v1 stays preferred. Differences are
   reported with paired bootstrap intervals; nothing is called significant unless
   the 95% interval excludes zero.
5. Play-type extension (recent classes only): a two-stage scheme. Stage 1 = the
   selected historical model, forward cross-fitted (each recent class is scored by a
   model trained only on earlier classes). Stage 2 = L2 logistic on recent cohort
   classes < T with [stage-1 logit] vs [stage-1 logit + 5 play-type features]. Older
   classes are never discarded: they are what stage 1 is trained on.

Outputs: results/modeling_v2/*.csv|json and models/qb_career_success/v2.0.0/
(v1.0.0 is never written to.)
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance

from src.modeling import build_dataset_v2 as B2
from src.modeling import era
from src.modeling import features as F
from src.modeling import train as T

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'modeling_v2'
MODEL_VERSION = '2.0.0'
MODEL_DIR = ROOT / 'models' / 'qb_career_success' / f'v{MODEL_VERSION}'
V1_MODEL_DIR = ROOT / 'models' / 'qb_career_success' / 'v1.0.0'
SEED = T.SEED
DEV_TEST_YEARS = list(range(2012, 2019))
HOLDOUT_YEARS = [2019, 2020, 2021]
SIMPLICITY_TOL = 0.002
FAMILIES = {'l2': T.l2, 'enet': T.enet, 'gbdt': T.gbdt}
LEVELS = ('basic', 'context')
TARGET = 'career_success'


def config_name(strategy, level, family) -> str:
    return f'{strategy}|{level}|{family}'


def all_configs(sets: dict) -> dict:
    """name -> (features, estimator factory, strategy, level, family), in simplicity order."""
    out = {}
    for s in B2.STRATEGIES:
        for lv in LEVELS:
            for fam, fac in FAMILIES.items():
                out[config_name(s, lv, fam)] = (sets[(s, lv)], fac, s, lv, fam)
    return out


def train_mask(df: pd.DataFrame, ty: int, regime: str) -> np.ndarray:
    dy = df['draft_year']
    if regime == 'embargo':
        return (df['v2_historical'] & (dy + B2.LABEL_EMBARGO_YEARS <= ty)).to_numpy()
    if regime == 'retro':
        return (df['v2_historical'] & (dy < ty)).to_numpy()
    if regime == 'modern_retro':
        return (df['comparison_cohort'] & dy.between(2017, ty - 1)).to_numpy()
    if regime == 'v1_history_pool_retro':
        # exactly v1's supplementary pool: 2005-2016 'eligible_history' (any QB with box stats, incl. FCS
        # finals and QBs whose last observed season is not the pre-draft season) + v1 cohort classes < T
        return ((df['eligible_history'] | df['comparison_cohort']) & (dy < ty)).to_numpy()
    raise ValueError(regime)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols, factory, target: str):
    y = train[target].astype(int).to_numpy()
    if len(np.unique(y)) < 2:       # degenerate fold (possible at extreme thresholds): predict the training rate
        pipe = T.make_pipeline(DummyClassifier(strategy='prior')).fit(train[cols], y)
        return pipe, np.full(len(test), float(y.mean())), True
    pipe = T.make_pipeline(factory()).fit(train[cols], y)
    return pipe, pipe.predict_proba(test[cols])[:, 1], False


def run(df: pd.DataFrame, configs: dict, test_years, regime: str, experiment: str, test_mask, target: str = TARGET,
        importance_for=()) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds, imps = [], []
    for ty in test_years:
        tr = df[train_mask(df, ty, regime)]
        te = df[test_mask & (df['draft_year'] == ty).to_numpy()]
        if te.empty:
            continue
        tr = tr[tr[target].notna()]
        te = te[te[target].notna()]
        assert tr['draft_year'].max() < ty
        if regime == 'embargo':
            assert (tr['draft_year'] + B2.LABEL_EMBARGO_YEARS <= ty).all()
        y_tr = tr[target].astype(int).to_numpy()
        for name, (cols, fac, s, lv, fam) in configs.items():
            pipe, p, degenerate = fit_predict(tr, te, cols, fac, target)
            preds.append(pd.DataFrame({
                'experiment': experiment, 'regime': regime, 'model': name, 'strategy': s, 'level': lv, 'family': fam,
                'feature_set': lv, 'target': target, 'test_year': ty,
                'qb_draft_prospect_id': te['qb_draft_prospect_id'].to_numpy(), 'player_name': te['player_name'].to_numpy(),
                'y_true': te[target].astype(int).to_numpy(), 'p_success': p, 'threshold': float(y_tr.mean()),
                'n_train': len(tr), 'n_train_pos': int(y_tr.sum()), 'train_first_class': int(tr['draft_year'].min()),
                'train_last_class': int(tr['draft_year'].max()), 'degenerate_fold': degenerate}))
            if name in importance_for and not degenerate:
                r = permutation_importance(pipe, te[cols], te[target].astype(int), scoring='neg_brier_score',
                                           n_repeats=50, random_state=SEED)
                imps.append(pd.DataFrame({'experiment': experiment, 'model': name, 'test_year': ty, 'n_test': len(te),
                                          'feature': cols, 'importance_mean': r.importances_mean}))
    return pd.concat(preds, ignore_index=True), (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


def prior_config(sets) -> dict:
    return {'base_rate': (sets[('raw_v1', 'basic')], lambda: DummyClassifier(strategy='prior'), 'none', 'none', 'prior')}


def brier(g: pd.DataFrame) -> float:
    return float(np.mean((g['p_success'] - g['y_true']) ** 2))


def logloss(g: pd.DataFrame) -> float:
    p = np.clip(g['p_success'].to_numpy(), 1e-15, 1 - 1e-15)
    y = g['y_true'].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def select(dev: pd.DataFrame, configs: dict) -> tuple[str, pd.DataFrame]:
    tab = dev.groupby('model', sort=False).apply(
        lambda g: pd.Series({'dev_brier': brier(g), 'dev_log_loss': logloss(g), 'n': len(g)}), include_groups=False)
    tab = tab.loc[[c for c in configs if c in tab.index]]
    best = tab['dev_brier'].min()
    tied = tab[tab['dev_brier'] <= best + SIMPLICITY_TOL]
    chosen = tied.index[0]                      # configs are already in simplicity order
    tab['tied_with_best'] = tab.index.isin(tied.index)
    tab['selected'] = tab.index == chosen
    return chosen, tab.reset_index()


def stage1_logits(df: pd.DataFrame, cols, factory, classes) -> pd.Series:
    """Forward cross-fitted stage-1 logit for every comparison-cohort row of the given classes."""
    out = {}
    for c in classes:
        tr = df[train_mask(df, c, 'retro')]
        rows = df[df['comparison_cohort'] & (df['draft_year'] == c)]
        _, p, _ = fit_predict(tr, rows, cols, factory, TARGET)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        out.update(dict(zip(rows['qb_draft_prospect_id'], np.log(p / (1 - p)))))
    return pd.Series(out, name='stage1_logit')


def playtype_extension(df: pd.DataFrame, cols, factory) -> pd.DataFrame:
    d = df.copy()
    d['stage1_logit'] = d['qb_draft_prospect_id'].map(stage1_logits(d, cols, factory, range(2017, 2022)))
    d['stage1_p'] = 1 / (1 + np.exp(-d['stage1_logit']))
    preds = []
    for ty in HOLDOUT_YEARS:
        tr = d[d['comparison_cohort'] & d['draft_year'].between(2017, ty - 1)]
        te = d[d['v2_common_holdout'] & (d['draft_year'] == ty)]
        y_tr = tr[TARGET].astype(int).to_numpy()
        base = {'experiment': 'playtype_extension', 'regime': 'retro', 'target': TARGET, 'test_year': ty,
                'qb_draft_prospect_id': te['qb_draft_prospect_id'].to_numpy(), 'player_name': te['player_name'].to_numpy(),
                'y_true': te[TARGET].astype(int).to_numpy(), 'threshold': float(y_tr.mean()), 'n_train': len(tr),
                'n_train_pos': int(y_tr.sum())}
        preds.append(pd.DataFrame({**base, 'model': 'stage1_historical', 'feature_set': 'stage1',
                                   'p_success': te['stage1_p'].to_numpy()}))
        for name, c in (('stage2_recalibrated', ['stage1_logit']),
                        ('stage2_plus_playtype', ['stage1_logit'] + F.GROUP_C_PLAYTYPE)):
            pipe = T.make_pipeline(T.l2()).fit(tr[c], y_tr)
            preds.append(pd.DataFrame({**base, 'model': name, 'feature_set': name,
                                       'p_success': pipe.predict_proba(te[c])[:, 1]}))
    return pd.concat(preds, ignore_index=True), d


def bootstrap_coefficients(df, cols, factory, n_boot=1000) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    X, y = df[cols], df[TARGET].astype(int).to_numpy()
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(df), len(df))
        if len(np.unique(y[i])) < 2:
            continue
        m = T.make_pipeline(factory()).fit(X.iloc[i], y[i]).named_steps['model']
        if hasattr(m, 'coef_'):
            draws.append(m.coef_[0])
    if not draws:
        return pd.DataFrame()
    d = np.array(draws)
    return pd.DataFrame({'feature': cols, 'boot_median': np.median(d, 0), 'boot_p05': np.quantile(d, .05, 0),
                         'boot_p95': np.quantile(d, .95, 0), 'share_positive': (d > 0).mean(0), 'n_boot_valid': len(d)})


def load() -> pd.DataFrame:
    f = pd.read_parquet(B2.FEATURES_FILE)
    t = pd.read_parquet(B2.TARGETS_FILE)
    return f.merge(t, on=['qb_draft_prospect_id', 'draft_year'], how='inner', validate='one_to_one')


def main() -> dict:
    manifest = json.loads((B2.OUT_DIR / f'dataset_manifest_v{B2.DATASET_VERSION}.json').read_text(encoding='utf-8'))
    sets = B2.feature_sets(manifest['combo_choice'])
    for (s, lv), cols in sets.items():
        assert not set(cols) & F.POST_DRAFT_COLUMNS, f'draft slot in {s}|{lv}'
        assert not [c for c in cols if c.startswith(('pt_', 'nfl_', 'success_', 'career_success'))], cols
    df = load().sort_values(['draft_year', 'overall_pick']).reset_index(drop=True)
    H = df['v2_historical'].to_numpy()
    hold = df['v2_common_holdout'].to_numpy()
    configs = all_configs(sets)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1) development (embargoed) + selection -- holdout classes are never touched here
    dev, _ = run(df, {**prior_config(sets), **configs}, DEV_TEST_YEARS, 'embargo', 'dev_embargo', H)
    assert dev['test_year'].max() < HOLDOUT_YEARS[0] and dev['train_last_class'].max() < HOLDOUT_YEARS[0]
    chosen, sel_tab = select(dev, configs)
    sel_tab.to_csv(RESULTS / 'dev_selection_table.csv', index=False)
    cols, fac, s_strat, s_level, s_fam = configs[chosen]
    # pre-registered secondary question, also answered on dev only: best normalized vs raw at fixed L2/context
    print(f'selected on development folds: {chosen}')

    # 2) final holdout, both regimes, full ablation (labelled; not used for selection)
    sel = {chosen: configs[chosen]}
    ref_draft = {'draft_reference|l2': (F.DRAFT_REFERENCE, T.l2, 'draft_reference', 'reference', 'l2')}
    h_retro, imp = run(df, {**prior_config(sets), **configs, **ref_draft}, HOLDOUT_YEARS, 'retro', 'holdout_retro', hold,
                       importance_for=(chosen,))
    h_emb, _ = run(df, {**prior_config(sets), **configs, **ref_draft}, HOLDOUT_YEARS, 'embargo', 'holdout_embargo', hold)
    modern = {'v1_A_basic_l2': (F.FEATURE_SETS['A_basic'], T.l2, 'raw_v1', 'basic', 'l2'),
              'v1_B_context_l2': (F.FEATURE_SETS['B_context'], T.l2, 'raw_v1', 'context', 'l2'),
              'v1_C_playtype_l2': (F.FEATURE_SETS['C_playtype'], T.l2, 'raw_v1', 'playtype', 'l2')}
    h_mod, _ = run(df, {**prior_config(sets), **modern, **{'draft_reference|l2': ref_draft['draft_reference|l2']}},
                   HOLDOUT_YEARS, 'modern_retro', 'holdout_modern_retro', hold)
    # diagnostic: v1's history pool vs the v2 eligibility rules, same raw_v1 features, same holdout QBs
    diag_cfg = {k: configs[k] for k in ('raw_v1|basic|l2', 'raw_v1|context|l2')}
    h_diag, _ = run(df, diag_cfg, HOLDOUT_YEARS, 'v1_history_pool_retro', 'diagnostic_v1_history_pool', hold)

    # 3) thresholds (same pre-selected configuration; complete windows only)
    thr = []
    for k in B2.THRESHOLDS:
        tgt = f'success_5yr_{k}'
        for regime, mods, exp in (('retro', sel, 'retro'), ('embargo', sel, 'embargo'),
                                  ('modern_retro', {'v1_B_context_l2': modern['v1_B_context_l2']}, 'modern_retro'),
                                  ('retro', prior_config(sets), 'retro')):
            p, _ = run(df, mods, HOLDOUT_YEARS, regime, f'threshold_{k}_{exp}', hold, target=tgt)
            thr.append(p.assign(threshold_starts=k))
            if regime == 'embargo':
                pd_, _ = run(df, mods, DEV_TEST_YEARS, 'embargo', f'threshold_dev_{k}', H, target=tgt)
                thr.append(pd_.assign(threshold_starts=k))
    thr = pd.concat(thr, ignore_index=True)

    # 4) play-type extension on recent prospects
    pt, d_stage = playtype_extension(df, cols, fac)

    allp = pd.concat([dev, h_retro, h_emb, h_mod, h_diag, pt], ignore_index=True)
    allp.to_csv(RESULTS / 'oof_predictions.csv', index=False)
    thr.to_csv(RESULTS / 'threshold_predictions.csv', index=False)
    imp.to_csv(RESULTS / 'permutation_importance_holdout.csv', index=False)
    avail = B2.label_availability_table(df, DEV_TEST_YEARS + HOLDOUT_YEARS)
    avail.to_csv(RESULTS / 'label_availability.csv', index=False)

    # 5) final artifacts. Existing model directories are never overwritten; CFBD-derived player-level
    #    files go to the gitignored data/model_artifacts/, listed in metadata with path + SHA-256.
    from src.modeling import artifacts as A
    full = df[H]
    y = full[TARGET].astype(int).to_numpy()
    final = T.make_pipeline(fac()).fit(full[cols], y)
    coefs = bootstrap_coefficients(full, cols, fac)
    coefs.to_csv(RESULTS / 'coefficient_bootstrap_selected.csv', index=False)
    if hasattr(final.named_steps['model'], 'coef_'):
        pd.DataFrame({'feature': cols, 'coef_standardized': final.named_steps['model'].coef_[0]}).to_csv(
            RESULTS / 'final_model_coefficients.csv', index=False)
    if (MODEL_DIR / 'metadata.json').exists():
        print(f'{MODEL_DIR.relative_to(ROOT).as_posix()} already exists: model artifacts NOT overwritten '
              '(results/modeling_v2 refreshed). Bump MODEL_VERSION to save a new model.')
        return A.load_metadata(MODEL_DIR)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = A.DATA_ARTIFACTS / 'qb_career_success' / f'v{MODEL_VERSION}'
    data_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'pipeline': final, 'features': cols, 'model_name': chosen, 'model_version': MODEL_VERSION,
                 'threshold': float(y.mean()), 'strategy': s_strat, 'level': s_level, 'family': s_fam},
                MODEL_DIR / 'historical_selected.joblib')
    # stage-2 play-type model (saved for reproducibility; see the report for whether it helps)
    st = d_stage[d_stage['comparison_cohort'] & d_stage['draft_year'].between(2017, 2021)]
    s2 = T.make_pipeline(T.l2()).fit(st[['stage1_logit'] + F.GROUP_C_PLAYTYPE], st[TARGET].astype(int))
    joblib.dump({'pipeline': s2, 'features': ['stage1_logit'] + F.GROUP_C_PLAYTYPE, 'model_name': 'stage2_plus_playtype',
                 'model_version': MODEL_VERSION, 'threshold': float(st[TARGET].mean()),
                 'stage1': 'historical_selected.joblib (stage1_logit = its log-odds)'},
                MODEL_DIR / 'playtype_stage2.joblib')
    for p in (B2.BASELINES_FILE, era.REF_FILE):
        shutil.copy2(p, data_dir / p.name)
    snap = full[['qb_draft_prospect_id', 'draft_year', 'player_name'] + cols].copy()
    snap['p_success_final_model'] = final.predict_proba(full[cols])[:, 1]
    snap.to_csv(data_dir / 'reproducibility_check.csv', index=False)
    external = [{'role': role, 'path': A.rel(data_dir / name), 'sha256': A.sha256(data_dir / name), 'regenerate': regen,
                 'reason': 'player-level data derived from CFBD; kept out of Git'} for name, role, regen in A.V2_MOVES]

    v1_meta = json.loads((V1_MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
    meta = {
        'model_version': MODEL_VERSION, 'created': date.today().isoformat(), 'random_seed': SEED,
        'selected_configuration': {'name': chosen, 'strategy': s_strat, 'level': s_level, 'family': s_fam,
                                   'features': cols, 'selected_on': 'pooled Brier, embargoed development folds 2012-2018',
                                   'simplicity_tolerance_brier': SIMPLICITY_TOL},
        'target': f'{TARGET} = nfl_starts_first_5_seasons >= 32 (thresholds 16-64 evaluated as sensitivity)',
        'training_draft_classes': sorted(full['draft_year'].unique().tolist()),
        'training_rows': int(len(full)), 'training_positive': int(y.sum()),
        'development_test_classes': DEV_TEST_YEARS, 'holdout_classes': HOLDOUT_YEARS,
        'holdout_rows': int(hold.sum()),
        'label_embargo_years': B2.LABEL_EMBARGO_YEARS,
        'training_cutoff': {'latest_draft_class': int(full['draft_year'].max()),
                            'outcomes_observed_through': '2025 NFL regular season',
                            'features_available_by': 'draft day of each class'},
        'normalization': {'version': era.NORMALIZATION_VERSION, **manifest['reference_population_rule'],
                          'methods': list(era.METHODS), 'combo_choice': manifest['combo_choice'],
                          'baselines_file': A.rel(data_dir / B2.BASELINES_FILE.name),
                          'reference_population_file': A.rel(data_dir / era.REF_FILE.name)},
        'dataset': {'version': B2.DATASET_VERSION, 'files': manifest['files']},
        'feature_sets': manifest['feature_sets'],
        'model_params': {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in final.named_steps['model'].get_params().items()},
        'preference_rule': 'v2 preferred over v1 only if holdout (retrospective) Brier AND log loss are both lower than v1 B_context_l2',
        'v1_reference': {'model_version': v1_meta['model_version'], 'dir': V1_MODEL_DIR.relative_to(ROOT).as_posix()},
        'package_versions': T.package_versions(),
        'artifacts': ['historical_selected.joblib', 'playtype_stage2.joblib', 'metadata.json'],
        'artifact_sha256': {n: A.sha256(MODEL_DIR / n) for n in ('historical_selected.joblib', 'playtype_stage2.joblib')},
        'external_data_files': external,
    }
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    print(f'historical training rows {len(full)} ({int(y.sum())}+) | holdout {int(hold.sum())} | saved -> {MODEL_DIR}')
    return meta


if __name__ == '__main__':
    main()
