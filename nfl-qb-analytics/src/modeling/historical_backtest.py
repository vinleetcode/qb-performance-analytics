"""Per-QB historical backtests for the dashboard (Phase 5.5).

    from src.modeling.historical_backtest import historical_backtest
    b = historical_backtest(name='Josh Allen', draft_year=2018)
    python -m src.modeling.historical_backtest --name "Josh Allen" --draft-year 2018

For a drafted QB of class T:
  1. prediction date: just before the T draft; inputs are his pre-draft college profile (and, in the
     post-draft view, his actual pick)
  2. training classes: those whose five-season outcome was observable by then (classes <= T-5)
  3. models: the v1.1.0 procedure (inner chronological selection of the point model and interval
     method, then a fit) applied to those classes only
  4. compare with what he actually did in NFL seasons 1-5

Validity labels (never mixed up):
  valid_out_of_sample_backtest                class 2012-2021 in the 183-QB cohort: the prediction was
                                              produced without his outcome, and his window is complete
  out_of_sample_forecast_outcome_incomplete   class 2022+: the forecast is out of sample, but his five
                                              seasons are not finished (actuals are to date)
  retrospective_illustration_in_sample        classes before 2012 (too few earlier classes for a fair
                                              fit): shown with the final model, which was trained on
                                              this QB. NOT a backtest
Stored results (from `python -m src.modeling.career_production_v11`) are used for the cohort;
`recompute=True` refits the fold and must reproduce them.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from src.modeling import artifacts as A
from src.modeling import backtesting as BT
from src.modeling import career_production as CP
from src.modeling import career_production_v11 as V11
from src.modeling import train_v2 as T2

FIRST_VALID = min(CP.TEST_YEARS)
LAST_COMPLETE = max(CP.TEST_YEARS)
COMPARE = ['games_started', 'games_played', 'pass_attempts', 'pass_yards', 'pass_tds', 'interceptions', 'rush_attempts',
           'rush_yards', 'rush_tds']


@lru_cache(maxsize=1)
def _data():
    d, paths = CP.load_data()
    allq = T2.load()
    seasons = pd.read_parquet(CP.CT.OUT)
    fsets = CP.feature_sets()
    try:
        stored = pd.read_parquet(V11.DATA_DIR / 'historical_backtest_points.parquet')
        stored_p = pd.read_parquet(V11.DATA_DIR / 'historical_backtest_start_probs.parquet')
    except FileNotFoundError:
        stored = stored_p = None
    return d, paths, allq, seasons, fsets, stored, stored_p


@lru_cache(maxsize=64)
def _fold_models(train_last: int, fs: str):
    """Models of one outer fold (training classes <= train_last), fitted once and reused for every QB of that class."""
    d, _, _, _, fsets, _, _ = _data()
    dev = d[d['draft_year'] <= train_last]
    cols = fsets[fs]
    chosen = V11.select_all(dev, cols, fs)
    fitted = {}
    for t in CP.TARGETS:
        s = chosen[t]
        fitted[t] = (s, V11.PointModel(s['point']).fit(dev, cols, t), V11.fit_interval(s['interval'], s['point'], dev, cols, t))
    return fitted, V11.start_prob_models(dev, cols)


def _fold_predict(train_last: int, row: pd.DataFrame, cols, fs) -> tuple[dict, dict]:
    fitted, spm = _fold_models(train_last, fs)
    preds = {}
    for t in CP.TARGETS:
        s, pm, spec = fitted[t]
        p = pm.predict(row)
        iv = V11.apply_interval(spec, row, p, t)
        preds[t] = {'predicted': float(p[0]), 'interval_80': [float(iv[0.8][0][0]), float(iv[0.8][1][0])],
                    'model': s['point'], 'interval_method': s['interval']}
    sp = V11.start_probs(spm, row[cols])['nested_logit_pava'][0]
    return preds, dict(zip(V11.THRESHOLDS, map(float, sp)))


def _actuals(pid, seasons: pd.DataFrame) -> tuple[dict, bool, int]:
    s = seasons[(seasons['qb_draft_prospect_id'] == pid) & (seasons['season_index'] <= CP.YEARS) & (seasons['status'] == 'observed')]
    complete = len(s) == CP.YEARS
    return {t: float(s[t].sum()) for t in CP.TARGETS}, complete, int(len(s))


def historical_backtest(name: str | None = None, draft_year: int | None = None, qb_draft_prospect_id: int | None = None,
                        recompute: bool = False) -> dict:
    d, paths, allq, seasons, fsets, stored, stored_p = _data()
    q = allq
    if qb_draft_prospect_id is not None:
        q = q[q['qb_draft_prospect_id'] == qb_draft_prospect_id]
    else:
        q = q[(q['player_name'] == name) & ((q['draft_year'] == draft_year) if draft_year else True)]
    if q.empty:
        return {'status': 'unavailable', 'reason': 'QB not found among drafted QBs with a college profile'}
    r = q.iloc[0]
    pid, T = int(r['qb_draft_prospect_id']), int(r['draft_year'])
    in_cohort = pid in set(d['qb_draft_prospect_id'])
    actual, complete, n_obs = _actuals(pid, seasons)
    if pd.isna(r.get('overall_pick')):
        return {'status': 'unavailable', 'reason': 'no draft record'}
    train_last = T - BT.LAG_FIVE_SEASON
    dev = d[d['draft_year'] <= train_last]
    if FIRST_VALID <= T <= LAST_COMPLETE and in_cohort:
        validity, expl = 'valid_out_of_sample_backtest', (f'models trained only on classes {int(dev["draft_year"].min())}-{train_last} '
                                                           f'({len(dev)} QBs), whose outcomes were known before the {T} draft')
    elif T > LAST_COMPLETE and len(dev) >= V11.MIN_INNER_TRAIN:
        validity, expl = 'out_of_sample_forecast_outcome_incomplete', (
            f'out of sample (trained on classes <= {train_last}), but only {n_obs} of 5 NFL seasons are observed; actuals are to date')
    else:
        validity, expl = 'retrospective_illustration_in_sample', (
            f'too few classes before {T} for a fair fit; shown with the final model, which was TRAINED ON THIS QB. '
            'Not a backtest: errors understate real forecast error')
    row = q.copy()
    row['log_overall_pick'] = np.log(float(r['overall_pick']))
    row['round_group'] = CP.CT.round_group(int(r['round']))
    out = {'status': 'available', 'player': {'name': r['player_name'], 'college': r['college'], 'draft_year': T,
                                             'round': int(r['round']), 'overall_pick': int(r['overall_pick']),
                                             'qb_draft_prospect_id': pid},
           'validity': {'status': validity, 'explanation': expl, 'in_sample': validity.startswith('retrospective')},
           'prediction_date': f'before the {T} NFL Draft (late April {T})',
           'training_classes': None if validity.startswith('retrospective') else [int(dev['draft_year'].min()), train_last],
           'n_training_qbs': None if validity.startswith('retrospective') else int(len(dev)),
           'label_embargo_years': BT.LAG_FIVE_SEASON, 'actual_window_complete': complete, 'observed_seasons': n_obs,
           'model_version': f'qb_career_production v{V11.MODEL_VERSION} procedure', 'views': {}}
    for fs, cols in fsets.items():
        if validity == 'valid_out_of_sample_backtest' and stored is not None and not recompute:
            g = stored[(stored['qb_draft_prospect_id'] == pid) & (stored['feature_set'] == fs)].set_index('target')
            preds = {t: {'predicted': float(g.loc[t, 'pred']), 'interval_80': [float(g.loc[t, 'selected_lo80']), float(g.loc[t, 'selected_hi80'])],
                         'model': g.loc[t, 'selected_point'], 'interval_method': g.loc[t, 'selected_interval']} for t in CP.TARGETS}
            sp_ = stored_p[(stored_p['qb_draft_prospect_id'] == pid) & (stored_p['feature_set'] == fs)
                           & (stored_p['model'] == 'nested_logit_pava')].set_index('threshold')['p']
            sp = {int(k): float(v) for k, v in sp_.items()}
            source = 'stored nested backtest'
        elif validity.startswith('retrospective'):
            art = V11.load(fs, with_reference=False)
            tot = V11.predict_totals(art, row)
            preds = {t: {'predicted': float(tot[t]['raw'][0]), 'interval_80': [float(tot[t]['intervals'][0.8][0][0]),
                                                                             float(tot[t]['intervals'][0.8][1][0])],
                         'model': tot[t]['model'], 'interval_method': tot[t]['interval_method']} for t in CP.TARGETS}
            sp = dict(zip(V11.THRESHOLDS, map(float, V11.predict_start_probs(art, row)['nested'][0])))
            source = 'final v1.1.0 model (in-sample)'
        else:
            preds, sp = _fold_predict(train_last, row, cols, fs)
            source = 'refit on the fold'
        comp = {}
        cap = float(sum(CP.CT.season_games_cap(T + k) for k in range(CP.YEARS)))
        for t in COMPARE:
            p, a = preds[t], actual[t]
            lo, hi = p['interval_80']
            if t in ('games_started', 'games_played'):        # feasibility: no more games than were scheduled
                lo, hi = min(lo, cap), min(hi, cap)
            comp[t] = {'predicted': p['predicted'], 'interval_80': [lo, hi], 'actual': a,
                       'actual_label': 'observed (five seasons)' if complete else f'observed to date ({n_obs} seasons)',
                       'error': (p['predicted'] - a) if complete else None,
                       'inside_80': (lo <= a <= hi) if complete else None, 'model': p['model']}
        starts = actual['games_started']
        out['views'][fs] = {'source': source, 'comparison': comp,
                            'start_probabilities': {str(k): {'predicted': v, 'actual_event': (starts >= k) if complete else
                                                             (True if starts >= k else None)} for k, v in sp.items()}}
    return out


def summarize(b: dict) -> str:
    if b['status'] != 'available':
        return b['reason']
    p, v = b['player'], b['validity']
    L = [f'{p["name"]} ({p["draft_year"]}, pick {p["overall_pick"]}): {v["status"]}', f'  {v["explanation"]}']
    for fs, view in b['views'].items():
        c = view['comparison']
        L.append(f'  [{fs}, {view["source"]}] ' + '; '.join(
            f'{t}: pred {c[t]["predicted"]:.0f} ({c[t]["interval_80"][0]:.0f}-{c[t]["interval_80"][1]:.0f}) vs actual {c[t]["actual"]:.0f}'
            for t in ('games_started', 'pass_yards', 'pass_tds', 'rush_yards')))
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--draft-year', type=int)
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    b = historical_backtest(a.name, a.draft_year, recompute=a.recompute)
    print(json.dumps(b, indent=2, default=str) if a.json else summarize(b))
