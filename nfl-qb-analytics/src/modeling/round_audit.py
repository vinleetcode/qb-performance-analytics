"""Draft-round and pick audit with NESTED chronological validation (Phase 5.5).

Run:  python -m src.modeling.round_audit        (local data only)

Phase 5 chose the proportional-odds round model after seeing the first ordinal model fail on the
same backtest, so its result is provisional. Here every choice is made inside development data:

  outer loop   test draft T = 2012..2026; development data = drafts < T (draft labels known on draft day)
  inner loop   validation drafts T-4..T-1, each predicted by a model trained on drafts before it
  selection    the candidate (family + penalty) with the lowest pooled inner ranked probability score
               is refit on all development drafts and scores draft T once

Candidates: historical round frequencies (raw and add-one smoothed), L2 multinomial logistic and
proportional-odds logistic regression, each with C in {0.05, 0.2, 0.5}. The same procedure runs on
seven rounds and on three groups (round 1, rounds 2-3, rounds 4-7). Pick ranges: ridge on log(pick)
with the penalty chosen in the inner loop; 50/80/90% intervals from inner (chronological) residuals,
compared with the Phase-5 GroupKFold residual intervals and the historical pick distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from src.modeling import draft_predictor as DP
from src.modeling import uncertainty as U
from src.modeling.ordinal import ProportionalOddsK
from src.modeling.reporting import md_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5_5'
REPORT_SECTION = RESULTS / 'round_audit_section.md'
SEED = DP.SEED
OUTER = DP.ROUND_TEST_YEARS
N_INNER = 4
CS = (0.05, 0.2, 0.5)
GROUPS = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}
GROUP_LABELS = ['Round 1', 'Rounds 2-3', 'Rounds 4-7']
LEVELS = (0.5, 0.8, 0.9)
FEATURES = DP.ROUND_FEATURES


class Freq:
    def __init__(self, K, smooth=0.0):
        self.K, self.smooth = K, smooth

    def fit(self, X, y):
        c = np.bincount(np.asarray(y, int), minlength=self.K) + self.smooth
        self.p = c / c.sum()
        return self

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


class MultinomialK:
    def __init__(self, K, C):
        self.K, self.C = K, C

    def fit(self, X, y):
        y = np.asarray(y, int)
        self.m = DP.pipe(LogisticRegression(C=self.C, max_iter=5000)).fit(X, y)
        self.cls = self.m.named_steps['model'].classes_
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), self.K))
        p[:, self.cls] = self.m.predict_proba(X)
        return p


def candidates(K: int) -> dict:
    c = {'freq': lambda: Freq(K), 'freq_smoothed': lambda: Freq(K, 1.0)}
    for C in CS:
        c[f'multinomial_C{C}'] = (lambda C=C: MultinomialK(K, C))
        c[f'prop_odds_C{C}'] = (lambda C=C: ProportionalOddsK(K, C))
    return c


def rps_k(P, y):
    K = P.shape[1]
    cp = np.cumsum(P, 1)
    co = np.cumsum(np.eye(K)[np.asarray(y, int)], 1)
    return float(np.mean(((cp - co) ** 2).sum(1) / (K - 1)))


def log_loss_k(P, y, eps=1e-6):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), np.asarray(y, int)], eps, 1))))


def targets(df, K):
    r = df['round'].astype(int).to_numpy()
    return r - 1 if K == 7 else np.array([GROUPS[x] for x in r])


def nested_rounds(dr: pd.DataFrame, K: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cands = candidates(K)
    yrs = dr['draft_year'].to_numpy()
    rows, sel = [], []
    for T in OUTER:
        te = yrs == T
        if not te.any():
            continue
        dev = dr[yrs < T]
        dy = dev['draft_year'].to_numpy()
        inner_years = sorted(set(dy))[-N_INNER:]
        scores = {k: [] for k in cands}
        for iy in inner_years:
            tr, va = dev[dy < iy], dev[dy == iy]
            if len(tr) < 30:
                continue
            for k, mk in cands.items():
                P = mk().fit(tr[FEATURES], targets(tr, K)).predict_proba(va[FEATURES])
                scores[k].append((rps_k(P, targets(va, K)), len(va)))
        inner = {k: (sum(s * n for s, n in v) / sum(n for _, n in v)) if v else np.nan for k, v in scores.items()}
        best = min((k for k in inner if np.isfinite(inner[k])), key=lambda k: inner[k], default='freq')
        test = dr[te]
        yt = targets(test, K)
        out = {'draft_year': T, 'name': test['name'].to_numpy(), 'y': yt, 'n_dev': len(dev)}
        P_sel = cands[best]().fit(dev[FEATURES], targets(dev, K)).predict_proba(test[FEATURES])
        fixed = {'selected': P_sel,
                 'freq': cands['freq']().fit(dev[FEATURES], targets(dev, K)).predict_proba(test[FEATURES]),
                 'phase5_prop_odds_C0.5': cands['prop_odds_C0.5']().fit(dev[FEATURES], targets(dev, K)).predict_proba(test[FEATURES]),
                 'multinomial_C0.5': cands['multinomial_C0.5']().fit(dev[FEATURES], targets(dev, K)).predict_proba(test[FEATURES])}
        f = pd.DataFrame({'draft_year': T, 'name': test['name'].to_numpy(), 'y': yt, 'n_dev': len(dev)})
        for k, P in fixed.items():
            for j in range(K):
                f[f'{k}_p{j}'] = P[:, j]
        rows.append(f)
        sel.append({'draft_year': T, 'K': K, 'selected': best, 'n_dev': len(dev),
                    **{f'inner_rps_{k}': v for k, v in inner.items()}})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sel)


def unimodal_share(P) -> float:
    ok = 0
    for row in P:
        d = np.sign(np.diff(row))
        d = d[d != 0]
        ok += (np.sum(np.diff(d) > 0) == 0)          # never goes down then up again
    return float(ok / len(P))


def round_metrics(b: pd.DataFrame, K: int, models) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = b['y'].to_numpy()
    rows, cal = [], []
    for m in models:
        P = b[[f'{m}_p{j}' for j in range(K)]].to_numpy()
        exp_r = P @ np.arange(K)
        d = b.assign(_i=np.arange(len(b)))
        ci = U.cluster_bootstrap(lambda x, P=P: rps_k(P[x['_i'].to_numpy()], x['y'].to_numpy()), d, n_boot=1000, seed=SEED)
        rows.append({'model': m, 'n': len(y), 'rps': rps_k(P, y), 'rps_lo': ci['lower'], 'rps_hi': ci['upper'],
                     'log_loss': log_loss_k(P, y), 'top_class_accuracy': float((P.argmax(1) == y).mean()),
                     'mae_expected_class': float(np.mean(np.abs(exp_r - y))),
                     'min_prob_assigned_to_actual': float(P[np.arange(len(y)), y].min()),
                     'prob_sum_max_dev': float(np.abs(P.sum(1) - 1).max()), 'min_prob': float(P.min())})
        for j in range(K):
            cal.append({'model': m, 'class': j + 1 if K == 7 else GROUP_LABELS[j], 'n_actual': int((y == j).sum()),
                        'observed_share': float((y == j).mean()), 'mean_predicted': float(P[:, j].mean()),
                        'brier_one_vs_rest': float(np.mean((P[:, j] - (y == j)) ** 2))})
    return pd.DataFrame(rows), pd.DataFrame(cal)


def aggregate_to_groups(b7: pd.DataFrame, models) -> pd.DataFrame:
    g = pd.DataFrame({'draft_year': b7['draft_year'], 'name': b7['name'],
                      'y': [GROUPS[int(r) + 1] for r in b7['y']]})
    for m in models:
        for j, rs in enumerate(([1], [2, 3], [4, 5, 6, 7])):
            g[f'{m}_from7_p{j}'] = b7[[f'{m}_p{r - 1}' for r in rs]].sum(axis=1).to_numpy()
    return g


# ------------------------------------------------------------------ pick ranges
def _ridge(alpha):
    return DP.pipe(Ridge(alpha=alpha))


def nested_picks(dr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    yrs = dr['draft_year'].to_numpy()
    out, sel = [], []
    qs = sorted({(1 - l) / 2 for l in LEVELS} | {1 - (1 - l) / 2 for l in LEVELS})
    for T in OUTER:
        te = yrs == T
        if not te.any():
            continue
        dev = dr[yrs < T]
        dy = dev['draft_year'].to_numpy()
        inner_years = sorted(set(dy))[-N_INNER:]
        res = {a: [] for a in (1.0, 10.0, 100.0)}
        for iy in inner_years:
            tr, va = dev[dy < iy], dev[dy == iy]
            if len(tr) < 30:
                continue
            for a in res:
                p = _ridge(a).fit(tr[FEATURES], np.log(tr['overall'])).predict(va[FEATURES])
                res[a].append(np.log(va['overall'].to_numpy()) - p)
        mae_in = {a: float(np.mean(np.abs(np.concatenate(v)))) if v else np.inf for a, v in res.items()}
        a_best = min(mae_in, key=mae_in.get)
        chrono_res = np.concatenate(res[a_best]) if res[a_best] else np.array([0.0])
        m = _ridge(a_best).fit(dev[FEATURES], np.log(dev['overall']))
        test = dr[te]
        mu = m.predict(test[FEATURES])
        # Phase-5 style: GroupKFold residuals inside development data (not chronological)
        g = dev['draft_year'].to_numpy()
        gk = np.full(len(dev), np.nan)
        for tr, va in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(dev, groups=g):
            gk[va] = np.log(dev['overall'].iloc[va]) - _ridge(a_best).fit(dev[FEATURES].iloc[tr], np.log(dev['overall'].iloc[tr])).predict(dev[FEATURES].iloc[va])
        f = pd.DataFrame({'draft_year': T, 'name': test['name'].to_numpy(), 'overall': test['overall'].to_numpy(float),
                          'pred_median': np.exp(mu), 'baseline_median': float(dev['overall'].median())})
        qc, qg, qb = np.quantile(chrono_res, qs), np.quantile(gk[np.isfinite(gk)], qs), np.quantile(dev['overall'], qs)
        for q, vc, vg, vb in zip(qs, qc, qg, qb):
            f[f'chrono_q{q:.2f}'] = np.clip(np.exp(mu + vc), 1, 262)
            f[f'groupkfold_q{q:.2f}'] = np.clip(np.exp(mu + vg), 1, 262)
            f[f'baseline_q{q:.2f}'] = vb
        out.append(f)
        sel.append({'draft_year': T, 'alpha': a_best, 'n_chrono_residuals': len(chrono_res), **{f'inner_mae_log_alpha{a}': v for a, v in mae_in.items()}})
    return pd.concat(out, ignore_index=True), pd.DataFrame(sel)


def pick_metrics(pk: pd.DataFrame) -> pd.DataFrame:
    o = pk['overall'].to_numpy()
    rows = []
    for name, pred in (('ridge_log_pick_nested', pk['pred_median']), ('historical_median', pk['baseline_median'])):
        rows.append({'model': name, 'mae_picks': float(np.mean(np.abs(pred - o))),
                     'median_abs_error_picks': float(np.median(np.abs(pred - o)))})
    P = pd.DataFrame(rows)
    cov = []
    for meth in ('chrono', 'groupkfold', 'baseline'):
        r = {'interval_method': meth}
        for lv in LEVELS:
            lo, hi = f'{meth}_q{(1 - lv) / 2:.2f}', f'{meth}_q{1 - (1 - lv) / 2:.2f}'
            r[f'coverage{int(lv * 100)}'] = U.coverage(o, pk[lo], pk[hi])
            r[f'mean_width{int(lv * 100)}'] = float((pk[hi] - pk[lo]).mean())
        cov.append(r)
    return P, pd.DataFrame(cov)


MODEL_VERSION = '1.1.0'
MODEL_DIR = ROOT / 'models' / 'qb_draft' / f'v{MODEL_VERSION}'


def select_final(dr: pd.DataFrame, K: int) -> tuple[str, dict]:
    """The same inner rule, applied as if the next draft were the test draft."""
    cands = candidates(K)
    dy = dr['draft_year'].to_numpy()
    scores = {k: [] for k in cands}
    for iy in sorted(set(dy))[-N_INNER:]:
        tr, va = dr[dy < iy], dr[dy == iy]
        for k, mk in cands.items():
            scores[k].append((rps_k(mk().fit(tr[FEATURES], targets(tr, K)).predict_proba(va[FEATURES]), targets(va, K)), len(va)))
    inner = {k: sum(a * n for a, n in v) / sum(n for _, n in v) for k, v in scores.items()}
    return min(inner, key=inner.get), inner


def fit_final(dr: pd.DataFrame, PC: pd.DataFrame, ps: pd.DataFrame, R7, R3) -> dict:
    from src.modeling import artifacts as A
    from src.modeling.train import package_versions
    if (MODEL_DIR / 'metadata.json').exists():
        raise FileExistsError(f'{A.rel(MODEL_DIR)} exists; versioned models are never overwritten')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    b7, i7 = select_final(dr, 7)
    b3, i3 = select_final(dr, 3)
    m7 = candidates(7)[b7]().fit(dr[FEATURES], targets(dr, 7))
    m3 = candidates(3)[b3]().fit(dr[FEATURES], targets(dr, 3))
    # pick: alpha by the inner rule; GroupKFold residual quantiles (closest to nominal in the nested backtest)
    dy = dr['draft_year'].to_numpy()
    mae = {}
    for a in (1.0, 10.0, 100.0):
        e = []
        for iy in sorted(set(dy))[-N_INNER:]:
            tr, va = dr[dy < iy], dr[dy == iy]
            e.append(np.log(va['overall'].to_numpy()) - _ridge(a).fit(tr[FEATURES], np.log(tr['overall'])).predict(va[FEATURES]))
        mae[a] = float(np.mean(np.abs(np.concatenate(e))))
    alpha = min(mae, key=mae.get)
    gk = np.full(len(dr), np.nan)
    for tr, va in GroupKFold(n_splits=5).split(dr, groups=dy):
        gk[va] = np.log(dr['overall'].iloc[va]) - _ridge(alpha).fit(dr[FEATURES].iloc[tr], np.log(dr['overall'].iloc[tr])).predict(dr[FEATURES].iloc[va])
    qs = sorted({(1 - l) / 2 for l in LEVELS} | {1 - (1 - l) / 2 for l in LEVELS})
    art = {'model_version': MODEL_VERSION, 'features': FEATURES,
           'round7': m7, 'round7_choice': b7, 'round7_inner_rps': i7,
           'round3': m3, 'round3_choice': b3, 'round3_inner_rps': i3, 'round3_labels': GROUP_LABELS,
           'pick_model': _ridge(alpha).fit(dr[FEATURES], np.log(dr['overall'])), 'pick_alpha': alpha,
           'pick_log_residual_quantiles': dict(zip([round(q, 2) for q in qs], np.quantile(gk, qs).tolist())),
           'levels': LEVELS}
    joblib.dump(art, MODEL_DIR / 'round_pick.joblib')
    meta = {'model_version': MODEL_VERSION, 'parent_version': '1.0.0', 'created_by': 'python -m src.modeling.round_audit',
            'random_seed': SEED, 'training_rows': int(len(dr)), 'training_draft_years': sorted(int(x) for x in set(dy)),
            'what_changed_vs_v1.0.0': ['round family and penalty chosen by inner chronological validation (RPS)',
                                       'three-group round model (1, 2-3, 4-7) added; primary for display',
                                       'pick-range ridge penalty chosen by the inner rule; 50/80/90% bounds stored',
                                       'draft-probability models are NOT re-versioned: use models/qb_draft/v1.0.0/prob_*.joblib'],
            'choices': {'round7': b7, 'round3': b3, 'pick_alpha': alpha},
            'measured_backtest': {'round7_rps_selected': float(R7.set_index('model').loc['selected', 'rps']),
                                  'round3_rps_selected': float(R3.set_index('model').loc['selected', 'rps']),
                                  'pick_coverage_groupkfold': PC[PC['interval_method'] == 'groupkfold'].iloc[0][
                                      ['coverage50', 'coverage80', 'coverage90']].round(3).to_dict()},
            'artifacts': ['metadata.json', 'round_pick.joblib'],
            'artifact_sha256': {'round_pick.joblib': A.sha256(MODEL_DIR / 'round_pick.joblib')},
            'external_data_files': [], 'package_versions': package_versions()}
    (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    return meta


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    src = DP.load_sources()
    dr = DP.build_drafted(src)
    dr = dr[dr['draft_year'] >= 2005].reset_index(drop=True)
    b7, s7 = nested_rounds(dr, 7)
    b3, s3 = nested_rounds(dr, 3)
    models = ['selected', 'freq', 'phase5_prop_odds_C0.5', 'multinomial_C0.5']
    R7, C7 = round_metrics(b7, 7, models)
    R3, C3 = round_metrics(b3, 3, models)
    g = aggregate_to_groups(b7, models)
    Rg, Cg = round_metrics(g.rename(columns={f'{m}_from7_p{j}': f'{m}_from7_p{j}' for m in models for j in range(3)}), 3,
                           [f'{m}_from7' for m in models])
    pk, ps = nested_picks(dr)
    PK, PC = pick_metrics(pk)
    # paired: nested-selected vs frequencies (RPS), class bootstrap
    pr = []
    for K, b in ((7, b7), (3, b3)):
        d = b.assign(_i=np.arange(len(b)))
        Ps = b[[f'selected_p{j}' for j in range(K)]].to_numpy()
        Pf = b[[f'freq_p{j}' for j in range(K)]].to_numpy()
        r = U.cluster_bootstrap(lambda x: rps_k(Ps[x['_i']], x['y'].to_numpy()) - rps_k(Pf[x['_i']], x['y'].to_numpy()),
                                d, n_boot=1000, seed=SEED)
        pr.append({'granularity': f'{K} classes', 'rps_selected_minus_freq': r['estimate'], 'lo': r['lower'], 'hi': r['upper']})
    PR = pd.DataFrame(pr)
    for n, t in (('round7_nested_backtest', b7), ('round3_nested_backtest', b3), ('round7_selection', s7), ('round3_selection', s3),
                 ('round7_metrics', R7), ('round3_metrics', R3), ('round7_calibration', C7), ('round3_calibration', C3),
                 ('round7_to_groups_metrics', Rg), ('pick_nested_backtest', pk), ('pick_selection', ps),
                 ('pick_metrics', PK), ('pick_coverage', PC), ('round_paired', PR)):
        t.to_csv(RESULTS / f'{n}.csv', index=False)
    train_counts = dr['round'].value_counts().sort_index()
    summary = {'rows': int(len(dr)), 'outer_rows': int(len(b7)), 'train_round_counts': train_counts.to_dict(),
               'selected_7': s7['selected'].value_counts().to_dict(), 'selected_3': s3['selected'].value_counts().to_dict()}
    (RESULTS / 'round_audit_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    REPORT_SECTION.write_text(render(summary, R7, C7, R3, C3, Rg, PK, PC, PR, s7, s3, ps, dr), encoding='utf-8')
    if not (MODEL_DIR / 'metadata.json').exists():
        summary['final'] = fit_final(dr, PC, ps, R7, R3)['choices']
    print(R7.round(3).to_string()); print(R3.round(3).to_string()); print(Rg.round(3).to_string())
    print(PK.round(1).to_string()); print(PC.round(3).to_string()); print(PR.round(4).to_string())
    print(s7['selected'].value_counts(), s3['selected'].value_counts())
    return summary


def render(summary, R7, C7, R3, C3, Rg, PK, PC, PR, s7, s3, ps, dr) -> str:
    r7, r3, rg = R7.set_index('model'), R3.set_index('model'), Rg.set_index('model')
    pr = PR.set_index('granularity')
    per_year = dr.groupby('draft_year').size()
    L = ['## Draft round and pick: nested chronological validation', '',
         f'Data: {summary["rows"]} drafted QBs (drafts 2005-2026) with a final college passer-season. Outer test drafts '
         f'{OUTER[0]}-{OUTER[-1]} ({summary["outer_rows"]} QBs). About {per_year.mean():.0f} QBs per draft; rounds 2-7 average '
         f'{dr["round"].between(2, 7).sum() / per_year.size / 6:.1f} QBs per round per year.', '',
         'Each outer draft is predicted by the candidate that won the **inner** validation on the four previous drafts. The '
         'outer drafts never influence a choice.', '',
         '### Seven rounds', '', md_table(R7), '',
         'Models: `selected` = nested choice; `freq` = historical round frequencies; `phase5_prop_odds_C0.5` = the Phase-5 '
         'model refit in the same folds; `multinomial_C0.5` = the Phase-5 multinomial.', '',
         f'Nested-selected minus frequencies, RPS: {pr.loc["7 classes", "rps_selected_minus_freq"]:+.4f} '
         f'(class-bootstrap 95% interval {pr.loc["7 classes", "lo"]:+.4f} to {pr.loc["7 classes", "hi"]:+.4f}).', '',
         'Choices made by the inner loop: ' + ', '.join(f'{k} x{v}' for k, v in s7['selected'].value_counts().items()) + '.', '',
         'Calibration by round (mean predicted probability vs observed share, outer drafts):', '',
         md_table(C7[C7['model'].isin(['selected', 'freq'])]), '',
         '### Three groups (round 1, rounds 2-3, rounds 4-7)', '', md_table(R3), '',
         f'Nested-selected minus frequencies, RPS: {pr.loc["3 classes", "rps_selected_minus_freq"]:+.4f} '
         f'({pr.loc["3 classes", "lo"]:+.4f} to {pr.loc["3 classes", "hi"]:+.4f}).', '',
         'Choices made by the inner loop: ' + ', '.join(f'{k} x{v}' for k, v in s3['selected'].value_counts().items()) + '.', '',
         md_table(C3[C3['model'].isin(['selected', 'freq'])]), '',
         'Seven-round forecasts summed into the same three groups (same rows, same scoring):', '', md_table(Rg), '',
         '### Pick ranges', '', md_table(PK, '.1f'), '',
         'Interval coverage at nominal 50/80/90% (chronological = residuals from the inner forward-chained folds; '
         'groupkfold = the Phase-5 method; baseline = quantiles of historical picks):', '', md_table(PC), '',
         'Ridge penalty chosen by the inner loop: ' + ', '.join(f'{k:g} x{v}' for k, v in ps['alpha'].value_counts().items()) + '.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    from src.modeling import round_audit as _m      # pickled classes resolve to the package module, not __main__
    _m.main()
