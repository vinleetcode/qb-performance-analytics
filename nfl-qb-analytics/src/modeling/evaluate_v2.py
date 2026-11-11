"""Evaluate Phase-4 predictions -> reports/historical_model_comparison.md,
reports/normalization_ablation.md and reports/figures/v2_*.png.

Run:  python -m src.modeling.evaluate_v2      (after train_v2)

All numbers are out-of-sample. Development numbers come from embargoed folds
(test classes 2012-2018); holdout numbers from 2019-2021 (the same 31 QBs v1 used).
Bootstrap intervals resample QBs (2,000 draws, fixed seed).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn import metrics as skm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.modeling import build_dataset_v2 as B2  # noqa: E402
from src.modeling import evaluate as EV  # noqa: E402
from src.modeling import evaluation as E  # noqa: E402
from src.modeling import train_v2 as T2  # noqa: E402
from src.modeling.evaluate import INK, INK_2, SURFACE, fig_ax, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = T2.RESULTS
REPORTS = ROOT / 'reports'
N_BOOT = EV.N_BOOT
SEED = T2.SEED
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
STRAT_LABEL = {'raw_v1': 'Raw, v1 definitions', 'raw': 'Raw, robust definitions', 'season_z': 'Season z-score',
               'season_pct': 'Season percentile', 'conf_z': 'Conference z-score', 'combo': 'Combination'}


# ---------------------------------------------------------------- metric helpers
def select_rows(o, experiment, model):
    g = o[(o['experiment'] == experiment) & (o['model'] == model)]
    assert len(g), (experiment, model)
    return g


def metrics_with_ci(g: pd.DataFrame) -> dict:
    r = EV.model_metrics(g)
    y, p = g['y_true'].to_numpy(), g['p_success'].to_numpy()
    r['mean_predicted'] = float(p.mean())
    r['calibration_slope'] = calibration_slope(y, p)
    return r


def calibration_slope(y, p) -> float:
    if len(np.unique(y)) < 2:
        return float('nan')
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(C=1e6, max_iter=1000).fit(lg, y).coef_[0, 0])


def paired(a: pd.DataFrame, b: pd.DataFrame, label: str) -> dict:
    """metric(b) - metric(a) on identical QBs; AUC positive / Brier & log loss negative = b better."""
    a = a.set_index('qb_draft_prospect_id')
    b = b.set_index('qb_draft_prospect_id')
    assert set(a.index) == set(b.index), f'{label}: not the same QBs'
    b = b.loc[a.index]
    assert (a['y_true'] == b['y_true']).all()
    y = a['y_true'].to_numpy()
    P = np.column_stack([a['p_success'].to_numpy(), b['p_success'].to_numpy()])
    rng = np.random.default_rng(SEED)
    out = {'comparison': label, 'n': len(y), 'n_pos': int(y.sum())}
    idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    for k, fn in EV.FAST:
        d = np.array([fn(y[i], P[i, 1]) - fn(y[i], P[i, 0]) for i in idx])
        d = d[np.isfinite(d)]
        better = (d > 0) if k == 'roc_auc' else (d < 0)
        out.update({f'{k}_diff': fn(y, P[:, 1]) - fn(y, P[:, 0]), f'{k}_lo': float(np.quantile(d, .025)),
                    f'{k}_hi': float(np.quantile(d, .975)), f'{k}_p_better': float(better.mean())})
    return out


def per_class(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ty, x in g.groupby('test_year'):
        rows.append({'test_year': ty, 'n': len(x), 'n_pos': int(x['y_true'].sum()), 'n_train': int(x['n_train'].iloc[0]),
                     'train_classes': f'{int(x["train_first_class"].iloc[0])}-{int(x["train_last_class"].iloc[0])}'
                     if 'train_first_class' in x and pd.notna(x['train_first_class'].iloc[0]) else '',
                     'roc_auc': E.roc_auc(x['y_true'], x['p_success']), 'brier': E.brier_score(x['y_true'], x['p_success'])})
    return pd.DataFrame(rows)


def f3(v):
    return '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.3f}'


def ci(r, k):
    return f'{f3(r[k])} [{f3(r[k + "_lo"])}, {f3(r[k + "_hi"])}]'


def dci(r, k):
    return f'{r[k + "_diff"]:+.3f} [{r[k + "_lo"]:+.3f}, {r[k + "_hi"]:+.3f}]'


# ---------------------------------------------------------------- figures
def fig_cohorts(df: pd.DataFrame, avail: pd.DataFrame):
    h = df[df['v2_historical']]
    c = h.groupby('draft_year')['career_success'].agg(n='size', pos=lambda s: int((s == 1).sum()))
    fig, axes = fig_ax(12, 4.4, ncols=2)
    ax = axes[0]
    ax.bar(c.index, c['pos'], color=C1, width=0.8, edgecolor=SURFACE, linewidth=1, label='32+ starts')
    ax.bar(c.index, c['n'] - c['pos'], bottom=c['pos'], color='#b7d3f6', width=0.8, edgecolor=SURFACE, linewidth=1,
           label='Fewer than 32')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_xticks(range(2005, 2022, 2))
    ax.set_ylim(0, c['n'].max() + 4)
    ax.set_title(f'Historical-eligible drafted QBs per class (n={len(h)})', color=INK, fontsize=10, loc='left')
    ax = axes[1]
    for col, color, lab in (('retrospective_train_n', C1, 'Historical, retrospective (all classes < T)'),
                            ('embargoed_train_n', C2, 'Historical, embargoed (labels observable at T)'),
                            ('modern_only_retrospective_n', C3, 'Modern-only, retrospective (v1)')):
        ax.plot(avail['test_draft_year'], avail[col], color=color, linewidth=2, marker='o', markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    ax.axvspan(2018.5, 2021.5, color='#f0efec', zorder=0)
    ax.text(2020, ax.get_ylim()[1] * 0.95, 'holdout', ha='center', fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.set_title('Training QBs available for each test class T', color=INK, fontsize=10, loc='left')
    ax.set_xlabel('Test draft class', color=INK)
    save(fig, 'v2_training_cohort_sizes.png')


def fig_ablation(dev_m: pd.DataFrame, hold_m: pd.DataFrame):
    fig, axes = fig_ax(12, 5.0, ncols=2)
    strategies = list(B2.STRATEGIES)
    for ax, (tab, title) in zip(axes, ((dev_m, 'Development (embargoed, 2012-2018, n=76)'),
                                       (hold_m, 'Holdout (retrospective, 2019-2021, n=31) - not used to choose'))):
        for j, lv in enumerate(('basic', 'context')):
            for i, s in enumerate(strategies):
                r = tab[(tab['strategy'] == s) & (tab['level'] == lv) & (tab['family'] == 'l2')].iloc[0]
                y = len(strategies) - i + (0.18 if lv == 'basic' else -0.18)
                c = C1 if lv == 'basic' else C2
                ax.plot([r['brier_lo'], r['brier_hi']], [y, y], color=c, linewidth=2, solid_capstyle='round')
                ax.plot(r['brier'], y, 'o', color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=('Basic (6 features)' if lv == 'basic' else 'Context (11 features)') if i == 0 else None)
        base = tab[tab['model'] == 'base_rate'].iloc[0]['brier']
        ax.axvline(base, color=INK_2, linewidth=1, linestyle='--', label='Base rate')
        ax.set_yticks(range(1, len(strategies) + 1))
        ax.set_yticklabels([STRAT_LABEL[s] for s in strategies[::-1]], fontsize=8.5, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('Brier score, L2 logistic (lower is better), 95% bootstrap interval', color=INK, fontsize=8.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Normalization strategies on identical QBs', color=INK, fontsize=11, x=0.01, ha='left')
    fig.subplots_adjust(bottom=0.2)
    save(fig, 'v2_normalization_ablation.png')


def fig_roc_cal(series):
    fig, axes = fig_ax(11.5, 5.0, ncols=2)
    axes[0].plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle='--')
    axes[1].plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle='--', label='Perfect calibration')
    for (lab, g), c in zip(series, (C1, C2, C3, C4)):
        fpr, tpr, _ = skm.roc_curve(g['y_true'], g['p_success'])
        axes[0].step(fpr, tpr, where='post', color=c, linewidth=2,
                     label=f'{lab} (AUC {skm.roc_auc_score(g["y_true"], g["p_success"]):.2f})')
        t = E.calibration_table(g['y_true'], g['p_success'], n_bins=4)
        axes[1].plot(t['mean_predicted'], t['observed_rate'], color=c, linewidth=2, marker='o', markersize=6,
                     markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    axes[0].set_xlabel('False-positive rate', color=INK)
    axes[0].set_ylabel('True-positive rate', color=INK)
    axes[0].set_title('ROC, holdout 2019-2021 (n=31)', color=INK, fontsize=10, loc='left')
    axes[0].legend(frameon=False, fontsize=7.5, loc='lower right')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Mean predicted probability (4 quantile bins)', color=INK)
    axes[1].set_ylabel('Observed rate', color=INK)
    axes[1].set_title('Calibration, holdout', color=INK, fontsize=10, loc='left')
    axes[1].legend(frameon=False, fontsize=7.5, loc='upper left')
    save(fig, 'v2_holdout_roc_calibration.png')


def fig_comparison(rows: list[tuple[str, dict, str]]):
    fig, axes = fig_ax(11.5, 0.5 * len(rows) + 1.6, ncols=2)
    y = np.arange(len(rows))[::-1]
    for ax, k, title in ((axes[0], 'roc_auc', 'ROC-AUC (higher is better)'), (axes[1], 'brier', 'Brier (lower is better)')):
        for yi, (lab, r, c) in zip(y, rows):
            if k == 'roc_auc' and lab.startswith('Base rate'):
                continue
            ax.plot([r[f'{k}_lo'], r[f'{k}_hi']], [yi, yi], color=c, linewidth=2, solid_capstyle='round')
            ax.plot(r[k], yi, 'o', color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows] if ax is axes[0] else [''] * len(rows), fontsize=8.5, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
    fig.suptitle('Holdout 2019-2021, same 31 QBs, 95% bootstrap intervals', color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, 'v2_holdout_model_comparison.png')


def fig_thresholds(tt: pd.DataFrame):
    fig, axes = fig_ax(11.5, 4.4, ncols=2)
    for (lab, key), c in zip((('Historical v2, retrospective', 'retro_v2'), ('Historical v2, embargoed', 'embargo_v2'),
                              ('Modern-only v1 B, retrospective', 'modern_v1B')), (C1, C2, C3)):
        g = tt[tt['series'] == key].sort_values('threshold_starts')
        for ax, k in ((axes[0], 'roc_auc'), (axes[1], 'brier_skill')):
            ax.plot(g['threshold_starts'], g[k], color=c, linewidth=2, marker='o', markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab)
    axes[0].axhline(0.5, color=INK_2, linewidth=1, linestyle='--')
    axes[1].axhline(0, color=INK_2, linewidth=1, linestyle='--')
    for ax, t in ((axes[0], 'Holdout ROC-AUC'), (axes[1], 'Holdout Brier skill vs base rate')):
        ax.axvline(32, color=INK_2, linewidth=1)
        ax.set_xlabel('Success threshold: NFL starts in first 5 seasons', color=INK)
        ax.set_title(t, color=INK, fontsize=10, loc='left')
        ax.set_xticks(B2.THRESHOLDS)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(bottom=0.24)
    save(fig, 'v2_threshold_sensitivity.png')


def fig_importance(boot: pd.DataFrame, imp: pd.DataFrame, labels: dict):
    fig, axes = fig_ax(12, 5.4, ncols=2)
    b = boot.sort_values('boot_median')
    y = np.arange(len(b))
    for yi, (_, r) in zip(y, b.iterrows()):
        axes[0].plot([r['boot_p05'], r['boot_p95']], [yi, yi], color=C1, linewidth=2, solid_capstyle='round')
        axes[0].plot(r['boot_median'], yi, 'o', color=C1, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    axes[0].axvline(0, color=INK_2, linewidth=1)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f'{labels.get(f, f)} ({100 * s:.0f}% > 0)' for f, s in zip(b['feature'], b['share_positive'])],
                            fontsize=8, color=INK)
    axes[0].set_xlabel('Log-odds per SD: bootstrap median, 90% interval', color=INK, fontsize=8.5)
    axes[0].set_title('Final v2 model coefficients (183 QBs)', color=INK, fontsize=10, loc='left')
    w = (imp.assign(wi=imp['importance_mean'] * imp['n_test']).groupby('feature')
         .agg(wi=('wi', 'sum'), n=('n_test', 'sum')))
    w['imp'] = w['wi'] / w['n']
    w = w.sort_values('imp')
    axes[1].barh(np.arange(len(w)), w['imp'], color=C1, height=0.7, edgecolor=SURFACE, linewidth=2)
    axes[1].axvline(0, color=INK_2, linewidth=1)
    axes[1].set_yticks(np.arange(len(w)))
    axes[1].set_yticklabels([labels.get(f, f) for f in w.index], fontsize=8, color=INK)
    axes[1].set_xlabel('Brier increase when permuted (holdout folds)', color=INK, fontsize=8.5)
    axes[1].set_title('Permutation importance, holdout', color=INK, fontsize=10, loc='left')
    fig.suptitle('Associations, not causes', color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, 'v2_feature_importance.png')
    return w


FEATURE_LABEL = {**EV.FEATURE_LABEL,
                 'cmp__raw': 'Completion %', 'ypa__raw': 'Yards/attempt', 'td__raw': 'TD %', 'int__raw': 'INT %',
                 'att_pg__raw': 'Attempts per game', 'rush_pg__raw': 'Rush yds/game (last 3 seasons)',
                 'opp_ypa__raw': 'Opponent-adjusted YPA', 'sos__raw': 'Opp. pass-D YPA vs league'}


# ---------------------------------------------------------------- main / reports
def main() -> dict:
    o = pd.read_csv(RESULTS / 'oof_predictions.csv')
    th = pd.read_csv(RESULTS / 'threshold_predictions.csv')
    sel_tab = pd.read_csv(RESULTS / 'dev_selection_table.csv')
    avail = pd.read_csv(RESULTS / 'label_availability.csv')
    imp = pd.read_csv(RESULTS / 'permutation_importance_holdout.csv')
    boot = pd.read_csv(RESULTS / 'coefficient_bootstrap_selected.csv')
    meta = json.loads((T2.MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
    era_tests = pd.read_csv(RESULTS / 'era_shift_tests.csv') if (RESULTS / 'era_shift_tests.csv').exists() else None
    df = T2.load()
    chosen = meta['selected_configuration']['name']

    # metric tables
    rows = []
    for (exp, model), g in o.groupby(['experiment', 'model'], sort=False):
        r = metrics_with_ci(g)
        first = g.iloc[0]
        rows.append({'experiment': exp, 'model': model, 'strategy': first.get('strategy'), 'level': first.get('level'),
                     'family': first.get('family'), **r})
    M = pd.DataFrame(rows)
    M.to_csv(RESULTS / 'metrics.csv', index=False)
    dev_m = M[M['experiment'] == 'dev_embargo']
    hold_m = M[M['experiment'] == 'holdout_retro']
    emb_m = M[M['experiment'] == 'holdout_embargo']
    get = lambda exp, model: M[(M['experiment'] == exp) & (M['model'] == model)].iloc[0]  # noqa: E731

    v2r = select_rows(o, 'holdout_retro', chosen)
    v2e = select_rows(o, 'holdout_embargo', chosen)
    v1b = select_rows(o, 'holdout_modern_retro', 'v1_B_context_l2')
    v1a = select_rows(o, 'holdout_modern_retro', 'v1_A_basic_l2')
    comps = [
        paired(v1b, v2r, 'v2 selected (historical, retrospective) vs v1 B_context (modern-only)'),
        paired(v1a, select_rows(o, 'holdout_retro', 'raw|basic|l2'), 'historical raw basic vs v1 A_basic (modern-only)'),
        paired(v1b, v2e, 'v2 selected (embargoed) vs v1 B_context (modern-only, retrospective)'),
        paired(v2r, v2e, 'v2 selected: embargoed vs retrospective (cost of the embargo)'),
        paired(select_rows(o, 'diagnostic_v1_history_pool', 'raw_v1|context|l2'),
               select_rows(o, 'holdout_retro', 'raw_v1|context|l2'),
               'v2 eligibility rules vs v1 history pool (same raw_v1 context features)'),
        paired(select_rows(o, 'playtype_extension', 'stage2_recalibrated'),
               select_rows(o, 'playtype_extension', 'stage2_plus_playtype'),
               'play-type extension: + play types vs historical model recalibrated'),
        paired(select_rows(o, 'holdout_retro', 'draft_reference|l2'), v2r, 'v2 selected vs draft-pick reference'),
    ]
    norm_comps = []
    for exp, tag in (('dev_embargo', 'development'), ('holdout_retro', 'holdout')):
        for lv in ('context', 'basic'):
            base = select_rows(o, exp, f'raw|{lv}|l2')
            for s in ('season_z', 'season_pct', 'conf_z', 'combo'):
                norm_comps.append({**paired(base, select_rows(o, exp, f'{s}|{lv}|l2'), f'{s} vs raw'), 'set': tag, 'level': lv,
                                   'strategy': s})
            norm_comps.append({**paired(select_rows(o, exp, f'raw_v1|{lv}|l2'), base, 'raw (robust) vs raw_v1'),
                               'set': tag, 'level': lv, 'strategy': 'raw'})
    PD = pd.DataFrame(comps)
    ND = pd.DataFrame(norm_comps)
    PD.to_csv(RESULTS / 'paired_differences.csv', index=False)
    ND.to_csv(RESULTS / 'normalization_paired_differences.csv', index=False)

    # thresholds
    trow = []
    for k in B2.THRESHOLDS:
        for key, exp, model in (('retro_v2', f'threshold_{k}_retro', chosen), ('embargo_v2', f'threshold_{k}_embargo', chosen),
                                ('modern_v1B', f'threshold_{k}_modern_retro', 'v1_B_context_l2'),
                                ('base_retro', f'threshold_{k}_retro', 'base_rate'),
                                ('dev_embargo_v2', f'threshold_dev_{k}', chosen)):
            g = th[(th['experiment'] == exp) & (th['model'] == model)]
            if g.empty:
                continue
            trow.append({'threshold_starts': k, 'series': key, 'n': len(g), 'n_pos': int(g['y_true'].sum()),
                         'roc_auc': E.roc_auc(g['y_true'], g['p_success']), 'brier': E.brier_score(g['y_true'], g['p_success']),
                         'degenerate_folds': int(g.groupby('test_year')['degenerate_fold'].first().sum())})
    TT = pd.DataFrame(trow)
    base_b = TT[TT['series'] == 'base_retro'].set_index('threshold_starts')['brier']
    TT['brier_skill'] = 1 - TT['brier'] / TT['threshold_starts'].map(base_b)
    TT.to_csv(RESULTS / 'threshold_sensitivity.csv', index=False)

    # figures
    fig_cohorts(df, avail)
    fig_ablation(dev_m, hold_m)
    fig_roc_cal([('Historical v2, retrospective', v2r), ('Historical v2, embargoed', v2e),
                 ('Modern-only v1 B', v1b), ('Draft pick (reference)', select_rows(o, 'holdout_retro', 'draft_reference|l2'))])
    comp_rows = [('Base rate (historical)', get('holdout_retro', 'base_rate'), INK_2),
                 ('v1 A basic, modern-only', get('holdout_modern_retro', 'v1_A_basic_l2'), C3),
                 ('v1 B context, modern-only', get('holdout_modern_retro', 'v1_B_context_l2'), C3),
                 ('Historical raw basic, retro', get('holdout_retro', 'raw|basic|l2'), C1),
                 ('v2 selected, retrospective', get('holdout_retro', chosen), C1),
                 ('Historical season-z context, retro', get('holdout_retro', 'season_z|context|l2'), C1),
                 ('v2 selected, embargoed', get('holdout_embargo', chosen), C2),
                 ('Draft pick (reference), retro', get('holdout_retro', 'draft_reference|l2'), C4)]
    fig_comparison(comp_rows)
    fig_thresholds(TT)
    imp_w = fig_importance(boot, imp, FEATURE_LABEL)

    pref = (get('holdout_retro', chosen)['brier'] < get('holdout_modern_retro', 'v1_B_context_l2')['brier']) and \
           (get('holdout_retro', chosen)['log_loss'] < get('holdout_modern_retro', 'v1_B_context_l2')['log_loss'])
    thr_rows = o[(o['experiment'].isin(['holdout_retro', 'holdout_embargo'])) & (o['model'] == chosen)]
    dev_av = avail[avail['test_draft_year'].isin(T2.DEV_TEST_YEARS)]
    sd = sel_tab.set_index('model')['dev_brier']
    g = TT.pivot(index='threshold_starts', columns='series', values='brier_skill')
    ga = TT.pivot(index='threshold_starts', columns='series', values='roc_auc')
    retro_ok = bool(((g['retro_v2'] >= g['modern_v1B']) & (ga['retro_v2'] >= ga['modern_v1B'])).all())
    gap = float((g['modern_v1B'] - g['embargo_v2']).max())
    thr_claim = (('At every threshold the retrospective v2 has AUC and Brier skill at least as high as v1 B. ' if retro_ok else
                  'The retrospective v2 does not beat v1 B at every threshold (see table). ') +
                 (f'The embargoed v2 is never more than {gap:.2f} Brier skill behind v1 B.' if gap > 0 else
                  'The embargoed v2 matches or beats v1 B at every threshold.'))
    ctx = dict(thr_lo=float(thr_rows['threshold'].min()), thr_hi=float(thr_rows['threshold'].max()),
               gbdt_dev=float(sd['raw|context|gbdt']), sel_dev=float(sd[chosen]),
               dev_ntr_lo=int(dev_av['embargoed_train_n'].min()), dev_ntr_hi=int(dev_av['embargoed_train_n'].max()),
               thr_claim=thr_claim,
               M=M, o=o, PD=PD, ND=ND, TT=TT, sel_tab=sel_tab, avail=avail, boot=boot, imp_w=imp_w, meta=meta,
               chosen=chosen, df=df, get=get, pref=pref, era_tests=era_tests,
               per_dev=per_class(select_rows(o, 'dev_embargo', chosen)),
               per_dev_base=per_class(select_rows(o, 'dev_embargo', 'base_rate')),
               per_hold=per_class(v2r), per_emb=per_class(v2e), per_v1=per_class(v1b))
    (REPORTS / 'historical_model_comparison.md').write_text(render_comparison(ctx), encoding='utf-8')
    (REPORTS / 'normalization_ablation.md').write_text(render_ablation(ctx), encoding='utf-8')
    summary = {'selected': chosen, 'v2_preferred_over_v1': bool(pref),
               'holdout': {m: get('holdout_retro', m)[['roc_auc', 'brier', 'log_loss']].to_dict()
                           for m in (chosen, 'base_rate', 'draft_reference|l2')},
               'paired': PD.to_dict('records')}
    (RESULTS / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    print(f'selected {chosen} | v2 preferred over v1: {pref}')
    print(PD[['comparison', 'roc_auc_diff', 'brier_diff', 'brier_lo', 'brier_hi', 'brier_p_better']].round(3).to_string())
    return summary


def model_row(r, label) -> str:
    auc = 'n/a' if r['model'] == 'base_rate' else ci(r, 'roc_auc')
    return (f'| {label} | {auc} | {ci(r, "brier")} | {f3(r["log_loss"])} | {f3(r["precision"])} | {f3(r["recall"])} | '
            f'{r["mean_predicted"]:.2f} vs {r["prevalence"]:.2f} | {"-" if r["model"] == "base_rate" else f3(r["calibration_slope"])} |')


def render_comparison(c) -> str:
    M, get, chosen, PD, TT, avail, meta = c['M'], c['get'], c['chosen'], c['PD'], c['TT'], c['avail'], c['meta']
    P = PD.set_index('comparison')
    v2r, v2e = get('holdout_retro', chosen), get('holdout_embargo', chosen)
    v1b, v1a = get('holdout_modern_retro', 'v1_B_context_l2'), get('holdout_modern_retro', 'v1_A_basic_l2')
    h = c['df'][c['df']['v2_historical']]
    k1 = P.loc['v2 selected (historical, retrospective) vs v1 B_context (modern-only)']
    k2 = P.loc['v2 selected (embargoed) vs v1 B_context (modern-only, retrospective)']
    k3 = P.loc['v2 selected: embargoed vs retrospective (cost of the embargo)']
    k4 = P.loc['v2 eligibility rules vs v1 history pool (same raw_v1 context features)']
    k5 = P.loc['play-type extension: + play types vs historical model recalibrated']
    k6 = P.loc['v2 selected vs draft-pick reference']
    diag = get('diagnostic_v1_history_pool', 'raw_v1|context|l2')
    st1, st2a, st2b = (get('playtype_extension', m) for m in ('stage1_historical', 'stage2_recalibrated', 'stage2_plus_playtype'))
    L = ['# Historical vs modern-only QB career-success models (Phase 4)', '',
         'Generated by `python -m src.modeling.evaluate_v2`. All numbers are out-of-sample. Development folds are '
         'embargoed (test classes 2012-2018). The holdout is the 2019-2021 classes, restricted to the same 31 QBs v1 was '
         'evaluated on (12 reached 32 starts). Holdout results were computed only after the model was chosen on the '
         'development folds. 95% intervals are QB-level bootstraps.', '',
         '## Bottom line', '',
         f'* **Expanding the training data helped; era normalization did not.** The pre-registered selection on '
         f'development folds chose `{chosen}`: 11 context features in **raw units** with truncation-robust definitions, '
         f'L2 logistic, trained on {meta["training_rows"]} QBs from {len(meta["training_draft_classes"])} classes '
         f'(2005-2021). No normalized strategy beat raw on the development folds (see `normalization_ablation.md`).',
         f'* **Holdout, retrospective regime (directly comparable to v1):** the v2 model scores AUC {ci(v2r, "roc_auc")} '
         f'and Brier {ci(v2r, "brier")}. v1 B_context, modern-only, scores AUC {f3(v1b["roc_auc"])} and Brier '
         f'{f3(v1b["brier"])}. Paired difference: Brier {dci(k1, "brier")} '
         f'(lower in {100 * k1["brier_p_better"]:.0f}% of bootstrap draws), AUC {dci(k1, "roc_auc")}. '
         '**That is consistent with a modest improvement, but not statistically clear:** both intervals include zero.',
         f'* **Pre-registered preference rule:** v2 replaces v1 only if both its holdout Brier and log loss are lower. '
         f'Result: **{"v2 is now the preferred model" if c["pref"] else "v1 stays the preferred model"}** (Brier '
         f'{f3(v2r["brier"])} vs {f3(v1b["brier"])}, log loss {f3(v2r["log_loss"])} vs {f3(v1b["log_loss"])}). v1.0.0 is '
         'kept unchanged.',
         f'* **Deployable (embargoed) version:** trained only on classes whose five-season label was observable at '
         f'each draft, v2 scores AUC {f3(v2e["roc_auc"])} and Brier {f3(v2e["brier"])}. The embargo costs '
         f'{k3["brier_diff"]:+.3f} Brier against retrospective training. v1\'s modern-only design **cannot be '
         'trained at all** under the embargo, because no 2017+ class had an observed five-season label before the '
         '2022 draft. Against v1-retrospective, the embargoed v2 is '
         f'{dci(k2, "brier")} Brier, i.e. about the same, using strictly less information.',
         f'* **The v1 "history hurt" finding came from the sample, not from era drift.** With v1\'s own feature '
         f'definitions, training on v1\'s old history pool gives Brier {f3(diag["brier"])}. The same features on the '
         f'v2-eligible pool give {f3(get("holdout_retro", "raw_v1|context|l2")["brier"])} '
         f'({dci(k4, "brier")}). The v2 pool drops FCS final seasons, QBs whose last observed season was not the '
         'pre-draft season, and QBs with fewer than 50 final-season attempts. v1\'s report attributed the decline to '
         'era drift; this result revises that.',
         f'* **Play types still add nothing detectable:** {dci(k5, "brier")} Brier and {dci(k5, "roc_auc")} AUC when '
         'added on top of the historical model.',
         f'* **Draft position remains the strongest single signal:** AUC {f3(get("holdout_retro", "draft_reference|l2")["roc_auc"])}. '
         f'v2 vs the draft-pick reference: Brier {dci(k6, "brier")}. The gap has narrowed, and the interval includes zero.', '',
         '## Samples', '',
         f'* **Historical-eligible QBs:** {len(h)} drafted 2005-2021 ({int((h.career_success == 1).sum())} reached 32 starts, '
         f'{int((h.career_success == 0).sum())} did not) across {h.draft_year.nunique()} draft classes. Requirements: a linked '
         'college identity, a final observed college season equal to the pre-draft season, at an FBS team, with at '
         'least 50 attempts, and a complete five-season window.',
         '* **Excluded from 2005-2021 (19 QBs):**',
         '  * 8 have no linked CFBD college record: non-FBS schools, or a final college season before 2004.',
         '  * 7 have an FCS final season (including Garoppolo and Wentz).',
         '  * 3 transferred to FCS schools whose final seasons are missing from the data (Flacco, Bomar, Coleman).',
         '  * Matt Cassel had fewer than 50 final-season attempts.',
         '  * Drafted QBs who never played stay in the sample as 0 starts whenever the draft board confirms it.',
         '* **Identity:** college identity comes from the existing matcher (exact or fuzzy name+school, CFBD draft '
         'athlete id, box-score name+school). NFL identity comes from the nflverse draft record itself. Tests check '
         'that each eligible QB\'s final college team matches his draft-board school.', '',
         '![Cohort sizes](figures/v2_training_cohort_sizes.png)', '',
         '## Label availability at each training cutoff', '',
         'A class drafted in year d has a five-season label (seasons d to d+4) only after season d+4 ends, i.e. '
         'by the draft of year d+5. The embargoed regime trains only on those classes. The retrospective regime, '
         'which v1 used, also trains on classes whose labels were not yet observable at the time.', '',
         '| Test class T | Latest class with an observable label | Embargoed training QBs (pos) | Retrospective training QBs (pos) | Modern-only (v1) training QBs | Classes used retrospectively without an observable label |',
         '|---|---|---|---|---|---|']
    for _, a in avail.iterrows():
        L.append(f'| {a["test_draft_year"]} | {a["latest_class_with_observable_label"]} | {a["embargoed_train_n"]} '
                 f'({a["embargoed_train_pos"]}) | {a["retrospective_train_n"]} ({a["retrospective_train_pos"]}) | '
                 f'{a["modern_only_retrospective_n"]} | {a["classes_unobservable_but_used_retrospectively"]} |')
    L += ['', '## Holdout results (2019-2021, the same 31 QBs as v1)', '',
          '| Model | ROC-AUC [95% CI] | Brier [95% CI] | Log loss | Precision | Recall | Mean predicted vs observed | Calibration slope |',
          '|---|---|---|---|---|---|---|---|',
          model_row(get('holdout_retro', 'base_rate'), 'Base rate (historical, retrospective)'),
          model_row(get('holdout_modern_retro', 'base_rate'), 'Base rate (modern-only)'),
          model_row(v1a, 'v1 A basic, modern-only (retrospective)'),
          model_row(v1b, 'v1 B context, modern-only (retrospective) - v1 preferred'),
          model_row(get('holdout_retro', 'raw|basic|l2'), 'Historical raw basic L2 (retrospective)'),
          model_row(v2r, f'**v2 selected `{chosen}`, retrospective**'),
          model_row(v2e, f'v2 selected `{chosen}`, embargoed (deployable)'),
          model_row(get('holdout_retro', 'raw|context|enet'), 'Historical raw context, elastic net (retro)'),
          model_row(get('holdout_retro', 'raw|context|gbdt'), 'Historical raw context, gradient boosting (retro)'),
          model_row(get('holdout_retro', 'draft_reference|l2'), 'Draft pick (reference, historical retro)'), '',
          'Precision and recall use each fold\'s training success rate as the threshold. Calibration slope is the slope '
          'of a logistic recalibration of the outcome on the model\'s log-odds. 1 is ideal; below 1 means '
          'over-confident; above 1 means probabilities are compressed toward the base rate. It is not defined for the '
          'base-rate rows, which are constant within each fold. The historical L2 models have slopes near 2: their '
          'probabilities are under-dispersed. Two reasons: regularisation, and a historical training success rate '
          f'(about {c["thr_lo"]:.0%}-{c["thr_hi"]:.0%}) below the holdout rate (39%), which also pulls mean predictions '
          'below the observed rate. With 31 QBs these slopes are themselves very uncertain. Ranking quality (AUC) is '
          'unaffected, and a recalibration step (as in the play-type stage 2) barely changes Brier.', '',
          f'Gradient boosting on the raw context features has the best holdout numbers in this table (Brier '
          f'{f3(get("holdout_retro", "raw|context|gbdt")["brier"])}). On the development folds it was among the worst '
          f'configurations (Brier {c["gbdt_dev"]:.3f} against {c["sel_dev"]:.3f} for the selected model). That is exactly '
          'the case the protocol guards against: choosing it now would be selecting on the holdout.', '',
          '![Model comparison](figures/v2_holdout_model_comparison.png)', '',
          '![ROC and calibration](figures/v2_holdout_roc_calibration.png)', '',
          '### Paired comparisons (identical QBs)', '',
          '| Comparison | AUC diff [95% CI] | Brier diff [95% CI] | P(boot: second better, Brier) | Log-loss diff [95% CI] |',
          '|---|---|---|---|---|']
    for _, r in PD.iterrows():
        L.append(f'| {r["comparison"]} | {dci(r, "roc_auc")} | {dci(r, "brier")} | {100 * r["brier_p_better"]:.0f}% | {dci(r, "log_loss")} |')
    sig = PD[(PD['brier_lo'] > 0) | (PD['brier_hi'] < 0)]['comparison'].tolist()
    L += ['', 'Positive AUC differences and negative Brier/log-loss differences favour the first-named model. Brier '
          'intervals that exclude zero: ' + ('; '.join(f'"{x}"' for x in sig) if sig else 'none') + '. Every other '
          'difference, including v2 against v1, is not statistically distinguishable from zero. These intervals '
          'resample QBs, not draft classes, so they somewhat understate uncertainty.', '',
          '### Per draft class', '',
          '| Class | Training classes | n (pos) | v2 retro AUC / Brier | v2 embargoed AUC / Brier | v1 B AUC / Brier |', '|---|---|---|---|---|---|']
    for (_, a), (_, b), (_, v) in zip(c['per_hold'].iterrows(), c['per_emb'].iterrows(), c['per_v1'].iterrows()):
        L.append(f'| {a["test_year"]} | retro {a["train_classes"]}, embargo {b["train_classes"]} | {a["n"]} ({a["n_pos"]}) | '
                 f'{f3(a["roc_auc"])} / {f3(a["brier"])} | {f3(b["roc_auc"])} / {f3(b["brier"])} | {f3(v["roc_auc"])} / {f3(v["brier"])} |')
    L += ['', 'Development classes (embargoed, selected model vs base rate):', '',
          '| Class | Training classes | n (pos) | AUC | Brier | Base-rate Brier |', '|---|---|---|---|---|---|']
    for (_, a), (_, b) in zip(c['per_dev'].iterrows(), c['per_dev_base'].iterrows()):
        L.append(f'| {a["test_year"]} | {a["train_classes"]} | {a["n"]} ({a["n_pos"]}) | {f3(a["roc_auc"])} | {f3(a["brier"])} | {f3(b["brier"])} |')
    dv = get('dev_embargo', chosen)
    db = get('dev_embargo', 'base_rate')
    L += ['', f'Pooled development: AUC {ci(dv, "roc_auc")}, Brier {f3(dv["brier"])} against {f3(db["brier"])} for the base rate. '
          f'Development is harder than the holdout: embargoed training sets are small ({c["dev_ntr_lo"]}-{c["dev_ntr_hi"]} QBs), and single '
          'classes have 7-14 QBs, so per-class numbers swing widely.', '',
          '## Threshold sensitivity (complete windows only)', '',
          'The selected configuration is held fixed at every threshold; nothing is re-selected per threshold.', '',
          '| Threshold (starts) | Holdout positives | v2 retro AUC / Brier skill | v2 embargoed AUC / Brier skill | v1 B AUC / Brier skill | Development (embargoed) positives, AUC |',
          '|---|---|---|---|---|---|']
    for k in B2.THRESHOLDS:
        g = TT[TT['threshold_starts'] == k].set_index('series')
        cell = lambda s: f'{f3(g.loc[s, "roc_auc"])} / {g.loc[s, "brier_skill"]:+.2f}' + (  # noqa: E731
            f' ({int(g.loc[s, "degenerate_folds"])} degenerate fold)' if g.loc[s, 'degenerate_folds'] else '')
        L.append(f'| {k}{" (primary)" if k == 32 else ""} | {int(g.loc["retro_v2", "n_pos"])} of {int(g.loc["retro_v2", "n"])} | '
                 f'{cell("retro_v2")} | {cell("embargo_v2")} | {cell("modern_v1B")} | '
                 f'{int(g.loc["dev_embargo_v2", "n_pos"])}, {f3(g.loc["dev_embargo_v2", "roc_auc"])} |')
    L += ['', '![Threshold sensitivity](figures/v2_threshold_sensitivity.png)', '',
          'Brier skill = 1 - Brier / Brier of the historical base rate at that threshold. A "degenerate fold" is one '
          'where the training set contained only one outcome class, so the model fell back to the training rate. '
          f'{c["thr_claim"]} At 64 starts the positives are few, and the estimates are correspondingly unstable.', '',
          '## Play-type features on recent prospects', '',
          'Play-type data exists only for the 2017+ classes. Older data is not discarded. Instead, a two-stage scheme '
          'is used, so each stage is trained where its features exist:', '',
          '1. **Stage 1:** the selected historical model, forward cross-fitted. Each 2017-2021 QB is scored by a model trained only '
          'on classes before his.',
          '2. **Stage 2:** L2 logistic on the 2017+ cohort classes before the test class, using either the stage-1 log-odds alone '
          '(recalibration) or the stage-1 log-odds plus the 5 play-type features.', '',
          '| Model | ROC-AUC [95% CI] | Brier [95% CI] | Log loss |', '|---|---|---|---|',
          f'| Stage 1 only (historical model) | {ci(st1, "roc_auc")} | {ci(st1, "brier")} | {f3(st1["log_loss"])} |',
          f'| Stage 2, recalibrated | {ci(st2a, "roc_auc")} | {ci(st2a, "brier")} | {f3(st2a["log_loss"])} |',
          f'| Stage 2, + play types | {ci(st2b, "roc_auc")} | {ci(st2b, "brier")} | {f3(st2b["log_loss"])} |', '',
          f'Adding play types: Brier {dci(k5, "brier")}, AUC {dci(k5, "roc_auc")}. This matches v1: no detectable value. '
          'Stage 2 trains on only 22-44 recent QBs, so it can mostly add noise. A real test needs play-type data for '
          'older classes (2004-2015 play-by-play is deferred until the CFBD quota resets).', '',
          '## Feature importance (selected model)', '',
          '| Feature | Final coefficient, bootstrap median [90%] | % of resamples > 0 | Holdout permutation importance (Brier) |',
          '|---|---|---|---|']
    b = c['boot'].set_index('feature')
    for f in b.sort_values('boot_median', key=np.abs, ascending=False).index:
        L.append(f'| {FEATURE_LABEL.get(f, f)} | {b.loc[f, "boot_median"]:+.3f} [{b.loc[f, "boot_p05"]:+.3f}, '
                 f'{b.loc[f, "boot_p95"]:+.3f}] | {100 * b.loc[f, "share_positive"]:.0f}% | '
                 f'{c["imp_w"]["imp"].get(f, float("nan")):+.4f} |')
    L += ['', '![Feature importance](figures/v2_feature_importance.png)', '',
          'In this table schedule strength is the raw "opponent pass-defense YPA allowed vs league" measure, so a '
          'negative coefficient means a **tougher** schedule goes with a higher success probability. Five associations keep '
          'their sign in at least 96% of resamples: younger age at draft, tougher schedule, higher completion %, more '
          'rushing yards per game, and greater height.', '',
          'These are associations, not causes. Age at draft partly reflects early-entry decisions, which follow '
          'perceived talent. Correlated features (YPA with opponent-adjusted YPA, rho about 0.9) share their importance, '
          'which is why opponent-adjusted YPA itself gets about zero weight.', '',
          '## Limitations', '',
          '* **The holdout is still small.** It has 31 QBs, and none of the differences between v2 and v1 are statistically clear. The '
          'development folds add 76 QBs, but with smaller, embargoed training sets.',
          '* **Starts measure opportunity as well as ability.** Draft slot drives early starts, which is partly why the '
          'draft-pick reference is so strong.',
          '* **FBS only.** The model has never seen FCS final seasons, so `predict` skips FCS QBs by default.',
          '* **Truncated careers in early classes.** For the 2005-2008 classes, college seasons before 2004 are '
          'missing. v2 sidesteps this with final-season and last-three-season definitions, but some information is '
          'still lost.', '',
          '## Saved artifacts', '',
          f'* `models/qb_career_success/v{meta["model_version"]}/`:',
          '  * `historical_selected.joblib`: the selected Pipeline, trained on all 183 QBs.',
          '  * `playtype_stage2.joblib`.',
          '  * The normalization baselines and reference population (parquet).',
          '  * `reproducibility_check.csv`.',
          '  * `metadata.json`: training and evaluation classes, normalization method, dataset hashes, parameters, seed and package versions.',
          '* `models/qb_career_success/v1.0.0/` is unchanged.', '',
          '## Reproduce', '', '```bash', 'python -m src.modeling.build_dataset_v2   # reference population, normalization, targets (local data only)',
          'python -m src.modeling.era_report         # college era analysis',
          'python -m src.modeling.train_v2           # development selection, holdout, thresholds, play-type extension',
          'python -m src.modeling.evaluate_v2        # metrics, intervals, figures, these reports',
          'python -m pytest -q', '```', '']
    return '\n'.join(L)


def render_ablation(c) -> str:
    M, ND, sel, chosen, get = c['M'], c['ND'], c['sel_tab'], c['chosen'], c['get']
    L = ['# Normalization ablation', '',
         'Generated by `python -m src.modeling.evaluate_v2`. Six feature strategies are compared, each at two levels '
         '(basic: 6 features; context: 11), across three model families. All 36 configurations run on identical QBs '
         'and identical folds. **Selection used only the development folds** (embargoed, test classes 2012-2018, 76 '
         'QBs). The holdout columns are shown for transparency and played no part in the choice.', '',
         '## Strategies', '',
         '| Strategy | Definition |', '|---|---|',
         '| `raw_v1` | v1 features exactly. Career pass attempts and career rushing yards are truncated for the 2005-2008 classes (no CFBD data before 2004). |',
         '| `raw` | Truncation-robust definitions, raw units: final-season rates and attempts per game, plus rushing yards per game over the last 3 seasons. |',
         '| `season_z` | The `raw` features as z-scores against the season\'s FBS reference passers (100+ attempts). |',
         '| `season_pct` | The `raw` features as season percentiles. |',
         '| `conf_z` | The `raw` features as conference-season z-scores, with a season fallback for groups under 6 passers. |',
         '| `combo` | Season z-scores for the features whose reference population shifted at least 0.3 SD (completion %, YPA, INT %, opponent-adjusted YPA), raw for the rest. Chosen from the era analysis before modeling. |', '',
         'Context adds three era-stable features to every strategy: team recruiting percentile, age at draft and height.', '',
         '## Development results (used for selection)', '',
         f'Selection rule: lowest pooled development Brier. Anything within {T2.SIMPLICITY_TOL} of the best counts as tied, '
         'and ties go to the simplest configuration (raw before normalized, basic before context, L2 before elastic '
         f'net before boosting). **Selected: `{chosen}`.**', '',
         '| Configuration | Dev Brier | Dev log loss | Tied with best | Holdout Brier (retro) | Holdout AUC (retro) |',
         '|---|---|---|---|---|---|']
    hold = M[M['experiment'] == 'holdout_retro'].set_index('model')
    for _, r in sel.sort_values('dev_brier').iterrows():
        h = hold.loc[r['model']]
        L.append(f'| `{r["model"]}`{" (selected)" if r["selected"] else ""} | {r["dev_brier"]:.4f} | {r["dev_log_loss"]:.4f} | '
                 f'{"yes" if r["tied_with_best"] else ""} | {h["brier"]:.3f} | {h["roc_auc"]:.3f} |')
    db = get('dev_embargo', 'base_rate')
    L += ['', f'Development base-rate Brier: {db["brier"]:.4f}. Holdout base-rate Brier (retrospective): '
          f'{get("holdout_retro", "base_rate")["brier"]:.3f}.', '',
          '![Ablation](figures/v2_normalization_ablation.png)', '',
          '## Paired comparisons against `raw` (L2 logistic, identical QBs)', '',
          '| Set | Level | Comparison | Brier diff [95% CI] | P(boot: normalized better) | AUC diff [95% CI] |',
          '|---|---|---|---|---|---|']
    for _, r in ND.iterrows():
        L.append(f'| {r["set"]} | {r["level"]} | {r["comparison"]} | {dci(r, "brier")} | {100 * r["brier_p_better"]:.0f}% | {dci(r, "roc_auc")} |')
    dz = ND[(ND['set'] == 'development') & (ND['level'] == 'context') & (ND['strategy'] == 'season_z')].iloc[0]
    rr = ND[(ND['set'] == 'development') & (ND['level'] == 'basic') & (ND['strategy'] == 'raw')].iloc[0]
    L += ['', '## Interpretation', '',
          f'* **Normalization did not improve prediction.** At the context level, season z-scores against raw change '
          f'development Brier by {dci(dz, "brier")}, and no normalized strategy beats raw in any development or holdout '
          'comparison. Several development intervals exclude zero in the *wrong* direction (season z, season '
          'percentile and combo at the context level): normalizing made predictions slightly worse.',
          '* **Two readings of "worse", which the data cannot separate.** (1) Part of the era drift is genuine: QBs '
          'became more accurate, and absolute accuracy still matters in the NFL, so removing the level removes '
          'signal. (2) A z-score against roughly 150 passers per season adds a little noise of its own. Either way, '
          'the season level should not simply be subtracted out. The bootstrap resamples QBs, not classes, so these '
          'intervals are somewhat optimistic.',
          f'* **The robust redefinitions mattered more.** Replacing truncated career totals with final-season and '
          f'recent-season measures (`raw` against `raw_v1`, basic level) changes development Brier by {dci(rr, "brier")}. '
          'It is the only change in the ablation that improves predictions, though its interval includes zero.',
          '* **Why normalization adds little.** The era analysis (`college_era_analysis.md`) found that season '
          'explains at most about 5% of the variance in any college passing statistic. Drift is at most 0.55 SD '
          '(completion %, INT %) over twenty years, against between-QB spreads of a full SD. A logistic model with context features also already '
          'carries era-stable signals (age, recruiting percentile). Removing a small level shift therefore changes '
          'rankings very little. Percentiles and conference z-scores also discard information: percentiles flatten '
          'the tails, and conference groups of 6-14 passers produce noisy baselines.',
          '* **This does not mean era effects are zero.** It means they are too small, relative to between-QB '
          'differences, to change predictions measurably with 183 QBs. A larger sample, or a model that interacts '
          'era with other features, could still find value.', '',
          '## Normalization details', '',
          '* **Reference population:** every FBS passer-season with at least 100 attempts and at least 4 games with a pass '
          'attempt, from all college passers (not only drafted QBs).',
          '* **Formulas:** z = (x - mean) / SD, clipped to +/-4. Percentile = mid-rank empirical CDF. INT % and schedule '
          'strength are flipped so that higher is better.',
          '* **Edge cases:** fewer than 50 attempts gives NaN; zero variance or a group of fewer than 2 gives NaN; '
          'conference groups under 6 fall back to the season baseline.',
          '* **Leakage:** a season-s baseline uses only season-s passers, which is known before the next draft. The '
          'baselines use no NFL outcome. Learned preprocessing (imputation, scaling) is fit inside each training fold.',
          '* Code: `src/modeling/era.py`. Tests: `tests/test_era_normalization.py`.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
