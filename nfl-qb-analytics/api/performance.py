"""Build api/static/model_performance.json: curated, aggregate-only evaluation results for the API.

    python -m api.performance        (reads results/phase5 and results/phase5_5; writes a tracked JSON file)

Every number is copied from an existing evaluation output; nothing is recomputed or invented. Each section
says how it was evaluated and how far the evaluation is from an untouched final test.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R5, R55 = ROOT / 'results' / 'phase5', ROOT / 'results' / 'phase5_5'
OUT = ROOT / 'api' / 'static' / 'model_performance.json'


def rows(df: pd.DataFrame, cols) -> list[dict]:
    d = df[cols].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].round(4)
    return json.loads(d.to_json(orient='records'))


def build() -> dict:
    meta = json.loads((ROOT / 'models' / 'qb_career_production' / 'v1.1.0' / 'metadata.json').read_text(encoding='utf-8'))
    shipped = {fs: next(iter(v.values())) for fs, v in meta['selection'].items()}
    cm = pd.read_csv(R55 / 'combine_metrics.csv')
    base = pd.read_csv(R5 / 'draft_prob_metrics.csv').set_index('model')
    r7, r3 = pd.read_csv(R55 / 'round7_metrics.csv'), pd.read_csv(R55 / 'round3_metrics.csv')
    pk, pc = pd.read_csv(R55 / 'pick_metrics.csv'), pd.read_csv(R55 / 'pick_coverage.csv')
    pr = pd.read_csv(R55 / 'v11_prob_metrics.csv')
    pm = pd.read_csv(R55 / 'v11_point_metrics.csv')
    im = pd.read_csv(R55 / 'v11_interval_metrics.csv')
    sim = pd.read_csv(R5 / 'similarity_backtest.csv') if (R5 / 'similarity_backtest.csv').exists() else None
    prod = []
    for fs, s in shipped.items():
        p = pm[pm['feature_set'] == fs].pivot_table(index='target', columns='model', values='mae')
        for t, r in p.iterrows():
            prod.append({'view': fs, 'statistic': t, 'mae_shipped_family': round(float(r[s['point']]), 2),
                         'mae_nested_selection': round(float(r['selected']), 2), 'mae_historical_median': round(float(r['median']), 2),
                         'shipped_family': s['point']})
    cov = im[im['method'] == f'direct_{shipped["pre_draft"]["interval"]}'].copy()
    sim_auc = None
    if sim is not None:
        from src.modeling.evaluate import fast_auc
        sim_auc = round(float(fast_auc(sim['y'], sim['knn_rate'])), 3)
    out = {
        'generated_from': f'python -m api.performance on {date.today().isoformat()} from results/phase5 and results/phase5_5',
        'evaluation_status': {
            'draft_probability': 'chronological backtest, test drafts 2014-2025; feature sets were compared on these same drafts '
                                 '(Phase 5) - exploratory, not an untouched final test',
            'draft_round_and_pick': 'nested chronological validation, test drafts 2012-2026: model family and penalty chosen '
                                    'inside earlier drafts only',
            'starter_probabilities': 'nested chronological backtest with a five-season label embargo, test classes 2012-2021; '
                                     'the 2019-2021 classes were also examined in earlier phases',
            'production': 'nested chronological backtest, test classes 2012-2021. The selection rule was revised once after '
                          'its first nested run; the shipped family matches the final inner choice. Shipped-family errors are '
                          'hindsight numbers (optimistic); nested-selection errors are conservative.',
            'intervals': 'coverage measured on later classes; not guaranteed under drift. The pre-draft interval method is a '
                         'reviewed override of the automated choice (backtest-informed, toward wider intervals).',
            'comparisons': 'explanatory only; the comps\' outcome rate is a weak predictor and is never served as a probability'},
        'sections': {
            'draft_probability': {
                'population': 'FBS QBs whose college career verifiably ended after >= 4 years (556 final seasons, 138 drafted)',
                'test_rows': int(cm['n'].iloc[0]), 'test_drafted': int(cm['n_drafted'].iloc[0]),
                'models': rows(cm[cm['variant'].isin(['A_college', 'B_college_invited', 'C_college_invited_measures'])],
                               ['variant', 'auc', 'auc_lo', 'auc_hi', 'brier', 'log_loss', 'calibration_slope']),
                'baseline_brier_historical_rate': round(float(base.loc['base_rate', 'brier']), 4),
                'note': 'A = pre-Combine (served as pre_combine); C = post-Combine (served as post_combine). Most of the gain from A '
                        'to C comes from Combine participation (B), not measurements.'},
            'draft_round': {'seven_rounds': rows(r7, ['model', 'n', 'rps', 'rps_lo', 'rps_hi', 'log_loss', 'top_class_accuracy']),
                            'three_groups': rows(r3, ['model', 'n', 'rps', 'rps_lo', 'rps_hi', 'log_loss', 'top_class_accuracy']),
                            'note': '`selected` = nested choice; `freq` = historical round frequencies (baseline).'},
            'pick': {'errors': rows(pk, ['model', 'mae_picks', 'median_abs_error_picks']),
                     'coverage': rows(pc, ['interval_method', 'coverage50', 'coverage80', 'coverage90']),
                     'served_interval_method': 'groupkfold'},
            'starter_probabilities': rows(pr[pr['model'].isin(['nested_logit_pava', 'base_rate'])],
                                          ['feature_set', 'model', 'threshold', 'n', 'n_pos', 'auc', 'brier', 'log_loss',
                                           'calibration_slope', 'calibration_intercept']),
            'production': {'test_qbs': 107, 'test_classes': [2012, 2021], 'errors': prod},
            'interval_calibration': {'served_method': {fs: s['interval'] for fs, s in shipped.items()},
                                     'coverage_by_statistic': rows(cov, ['feature_set', 'target', 'coverage50', 'coverage80', 'coverage90']),
                                     'mean_coverage': {fs: {k: round(float(g[k].mean()), 3) for k in ('coverage50', 'coverage80', 'coverage90')}
                                                       for fs, g in cov.groupby('feature_set')}},
            'comparisons': {'knn_outcome_rate_auc_32plus_starts': sim_auc,
                            'note': 'Phase-5 exploratory backtest; comps are explanatory'},
            'model_versions': {'draft_probability': 'qb_draft v1.0.0', 'draft_round_pick': 'qb_draft v1.1.0',
                               'career': 'qb_career_production v1.1.0', 'comparisons': 'qb_similarity v1.0.0',
                               'reference': 'qb_career_success v2.0.0'}},
        'limitations': [
            'About 10 QBs per draft class and ten test classes: every metric has wide class-bootstrap intervals.',
            'Draft eligibility, declarations and transfers are inferred, not observed; P(drafted) covers only the verified population.',
            'Pre-draft five-season totals are only modestly better than the historical median (not statistically established).',
            'P(64+ starts) pre-draft is poorly calibrated (few events).',
            'Rushing-total intervals under-cover; pick intervals under-cover (about 75% at nominal 80%).',
            'NFL efficiency rates are not forecast: they were not predictable from college data.',
            'Drafting team, coaching and injuries are not modelled.'],
    }
    return out


def _check(o):
    if isinstance(o, dict):
        for v in o.values():
            _check(v)
    elif isinstance(o, list):
        for v in o:
            _check(v)
    elif isinstance(o, float):
        assert np.isfinite(o), 'non-finite number in performance summary'


if __name__ == '__main__':
    o = build()
    s = json.dumps(o, indent=2)
    _check(json.loads(s))
    OUT.write_text(s, encoding='utf-8')
    print(f'wrote {OUT.relative_to(ROOT).as_posix()} ({len(s):,} bytes)')
