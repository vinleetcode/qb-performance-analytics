"""Evaluate the out-of-sample predictions written by train.py; write figures and the report.

Run:  python -m src.modeling.evaluate        (after build_dataset and train)

Every metric here is computed on OUT-OF-SAMPLE rolling-origin predictions (each
QB scored by a model trained only on earlier draft classes). Confidence
intervals are percentile bootstraps that resample QBs (2,000 draws, fixed seed).
With ~31 holdout QBs they are wide, and the report says so.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn import metrics as skm  # noqa: E402
from sklearn.inspection import partial_dependence  # noqa: E402

from src.modeling import build_dataset as BD  # noqa: E402
from src.modeling import evaluation as E  # noqa: E402
from src.modeling import features as F  # noqa: E402
from src.modeling import train as T  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = T.RESULTS
FIG = ROOT / 'reports' / 'figures'
REPORT = ROOT / 'reports' / 'model_evaluation.md'
N_BOOT = 2000
SEED = T.SEED

# reference palette (light mode), categorical slots in fixed order
SURFACE, INK, INK_2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'A_basic': '#2a78d6', 'B_context': '#eb6834', 'C_playtype': '#1baf7a', 'draft_reference': '#eda100'}
LABEL = {'A_basic': 'A: basic college stats', 'B_context': 'B: + context', 'C_playtype': 'C: + play types',
         'draft_reference': 'Draft pick (reference)', 'prior': 'Training success rate'}
FEATURE_LABEL = {
    'final_completion_pct': 'Final-season completion %', 'final_yards_per_attempt': 'Final-season yards/attempt',
    'final_td_pct': 'Final-season TD %', 'final_int_pct': 'Final-season INT %',
    'log_career_pass_attempts': 'Career pass attempts (log)', 'career_rush_yards': 'Career rushing yards',
    'final_opp_adj_ypa': 'Opponent-adjusted YPA', 'final_sos_pass_defense': 'Opp. pass-defense YPA allowed',
    'talent_recruiting_4yr_pctile': 'Team recruiting percentile', 'age_at_draft': 'Age at draft',
    'height_inches': 'Height', 'pt_standard_down_pass_rate': 'Standard-down pass rate',
    'pt_passing_down_pass_rate': 'Passing-down pass rate', 'pt_explosive_pass_rate': 'Explosive (20+ yd) pass rate',
    'pt_sack_rate': 'Team sack rate', 'pt_plays_per_game': 'Plays per game (tempo)', 'log_overall_pick': 'Draft pick (log)'}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_ax(w=6.5, h=4.2, ncols=1):
    fig, ax = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    for a in np.atleast_1d(ax):
        style(a)
    return fig, ax


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- metrics
# Fast numpy versions for the bootstrap loops (identical values to the sklearn-based
# point estimates in src/modeling/evaluation.py; tested in tests/test_modeling.py).
def fast_auc(y, p) -> float:
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fast_brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def fast_log_loss(y, p, eps: float = 1e-15) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


FAST = (('roc_auc', fast_auc), ('brier', fast_brier), ('log_loss', fast_log_loss))


def threshold_metrics(y, p, thr) -> dict:
    yhat = (np.asarray(p) >= np.asarray(thr)).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {'threshold_rule': 'training-fold success rate', 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'precision': tp / (tp + fp) if tp + fp else float('nan'), 'recall': tp / (tp + fn) if tp + fn else float('nan'),
            'specificity': tn / (tn + fp) if tn + fp else float('nan'), 'accuracy': (tp + tn) / len(y)}


def model_metrics(g: pd.DataFrame, boot: bool = True) -> dict:
    y, p = g['y_true'].to_numpy(), g['p_success'].to_numpy()
    out = {'n': len(g), 'n_pos': int(y.sum()), 'prevalence': float(y.mean()),
           'roc_auc': E.roc_auc(y, p), 'brier': E.brier_score(y, p), 'log_loss': E.log_loss(y, p),
           'ece_4bin': E.expected_calibration_error(y, p, n_bins=4)}
    if boot:
        for k, fn in FAST:
            ci = E.bootstrap_ci(fn, y, p, n_boot=N_BOOT, seed=SEED)
            out[f'{k}_lo'], out[f'{k}_hi'] = ci['lower'], ci['upper']
    out.update(threshold_metrics(y, p, g['threshold'].to_numpy()))
    return out


def metrics_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, model), g in oof.groupby(['experiment', 'model'], sort=False):
        rows.append({'experiment': exp, 'model': model, 'feature_set': g['feature_set'].iloc[0], **model_metrics(g)})
    t = pd.DataFrame(rows)
    prior = t[t['model'] == 'prior_rate'].set_index('experiment')['brier']
    t['brier_skill_vs_prior'] = 1 - t['brier'] / t['experiment'].map(prior)
    return t


def per_fold_table(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (exp, model, ty), g in oof.groupby(['experiment', 'model', 'test_year'], sort=False):
        rows.append({'experiment': exp, 'model': model, 'test_year': ty, 'n': len(g), 'n_pos': int(g['y_true'].sum()),
                     'n_train': int(g['n_train'].iloc[0]), 'roc_auc': E.roc_auc(g['y_true'], g['p_success']),
                     'brier': E.brier_score(g['y_true'], g['p_success'])})
    return pd.DataFrame(rows)


def paired_difference(oof: pd.DataFrame, exp: str, base: str, full: str) -> dict:
    """Bootstrap (QB resampling) of metric(full) - metric(base) on the same QBs.
    AUC: positive favours `full`. Brier: NEGATIVE favours `full`."""
    a = oof[(oof['experiment'] == exp) & (oof['model'] == base)].set_index('qb_draft_prospect_id')
    b = oof[(oof['experiment'] == exp) & (oof['model'] == full)].set_index('qb_draft_prospect_id')
    assert set(a.index) == set(b.index), 'paired comparison needs identical QBs'
    b = b.loc[a.index]
    y = a['y_true'].to_numpy()
    P = np.column_stack([a['p_success'].to_numpy(), b['p_success'].to_numpy()])
    rng = np.random.default_rng(SEED)
    out = {'experiment': exp, 'base': base, 'full': full, 'n': len(y)}
    for k, fn in FAST:
        point = fn(y, P[:, 1]) - fn(y, P[:, 0])
        d = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            v = fn(y[i], P[i, 1]) - fn(y[i], P[i, 0])
            if np.isfinite(v):
                d.append(v)
        d = np.array(d)
        better = (d > 0) if k == 'roc_auc' else (d < 0)
        out.update({f'{k}_diff': point, f'{k}_diff_lo': float(np.quantile(d, 0.025)),
                    f'{k}_diff_hi': float(np.quantile(d, 0.975)), f'{k}_share_full_better': float(better.mean())})
    return out


COMPARISONS = [
    ('primary_5yr_32starts', 'A_basic_l2', 'B_context_l2'),
    ('primary_5yr_32starts', 'B_context_l2', 'C_playtype_l2'),
    ('primary_5yr_32starts', 'A_basic_l2', 'C_playtype_l2'),
    ('primary_5yr_32starts', 'B_context_enet', 'C_playtype_enet'),
    ('primary_5yr_32starts', 'B_context_gbdt', 'C_playtype_gbdt'),
    ('primary_5yr_32starts', 'C_playtype_l2', 'draft_reference_l2'),
    ('supp_3yr_16starts', 'A_basic_l2', 'B_context_l2'),
    ('supp_3yr_16starts', 'B_context_l2', 'C_playtype_l2'),
]


# ---------------------------------------------------------------- figures
def fig_roc(oof):
    fig, ax = fig_ax(5.6, 5.0)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle='--', label='No skill')
    p = oof[oof['experiment'] == 'primary_5yr_32starts']
    for m, fs in (('A_basic_l2', 'A_basic'), ('B_context_l2', 'B_context'), ('C_playtype_l2', 'C_playtype'),
                  ('draft_reference_l2', 'draft_reference')):
        g = p[p['model'] == m]
        fpr, tpr, _ = skm.roc_curve(g['y_true'], g['p_success'])
        auc = skm.roc_auc_score(g['y_true'], g['p_success'])
        ax.step(fpr, tpr, where='post', color=SERIES[fs], linewidth=2, label=f'{LABEL[fs]} (AUC {auc:.2f})')
    ax.set_xlabel('False-positive rate', color=INK)
    ax.set_ylabel('True-positive rate', color=INK)
    ax.set_title('ROC, out-of-sample 2019-2021 classes (n=31)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    save(fig, 'model_roc_curves.png')


def fig_calibration(oof):
    fig, ax = fig_ax(5.6, 5.0)
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle='--', label='Perfect calibration')
    p = oof[oof['experiment'] == 'primary_5yr_32starts']
    for m, fs in (('A_basic_l2', 'A_basic'), ('B_context_l2', 'B_context'), ('C_playtype_l2', 'C_playtype'),
                  ('draft_reference_l2', 'draft_reference')):
        g = p[p['model'] == m]
        t = E.calibration_table(g['y_true'], g['p_success'], n_bins=4)
        ax.plot(t['mean_predicted'], t['observed_rate'], color=SERIES[fs], linewidth=2, marker='o', markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[fs])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean predicted probability (4 quantile bins, ~8 QBs each)', color=INK)
    ax.set_ylabel('Observed success rate', color=INK)
    ax.set_title('Calibration, out-of-sample (n=31)', color=INK, fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    save(fig, 'model_calibration.png')


def fig_metric_comparison(mt):
    p = mt[mt['experiment'] == 'primary_5yr_32starts'].copy()
    order = ['prior_rate', 'A_basic_l2', 'A_basic_enet', 'A_basic_gbdt', 'B_context_l2', 'B_context_enet',
             'B_context_gbdt', 'C_playtype_l2', 'C_playtype_enet', 'C_playtype_gbdt', 'draft_reference_l2']
    p = p.set_index('model').loc[order].reset_index()
    fig, axes = fig_ax(10, 5.2, ncols=2)
    ypos = np.arange(len(p))[::-1]
    for ax, k, title in ((axes[0], 'roc_auc', 'ROC-AUC (higher is better)'), (axes[1], 'brier', 'Brier score (lower is better)')):
        for yi, (_, r) in zip(ypos, p.iterrows()):
            c = INK_2 if r['model'] == 'prior_rate' else SERIES[r['feature_set']]
            ax.plot([r[f'{k}_lo'], r[f'{k}_hi']], [yi, yi], color=c, linewidth=2, solid_capstyle='round')
            ax.plot(r[k], yi, 'o', color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(p['model'] if ax is axes[0] else [''] * len(p), fontsize=8, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
    axes[0].axvline(0.5, color=INK_2, linewidth=1, linestyle='--')
    fig.suptitle('Out-of-sample performance with 95% bootstrap intervals (31 QBs, 12 successes)', color=INK,
                 fontsize=11, x=0.01, ha='left')
    save(fig, 'model_metric_comparison.png')


def fig_increment(pd_tab):
    t = pd_tab.copy()
    short = {'A_basic_l2': 'A', 'B_context_l2': 'B', 'C_playtype_l2': 'C', 'B_context_enet': 'B enet',
             'C_playtype_enet': 'C enet', 'B_context_gbdt': 'B boost', 'C_playtype_gbdt': 'C boost',
             'draft_reference_l2': 'Draft pick'}
    t['label'] = t['experiment'].map({'primary_5yr_32starts': '5-yr', 'supp_3yr_16starts': '3-yr'}) + \
        ': ' + t['full'].map(short) + ' vs ' + t['base'].map(short)
    fig, axes = fig_ax(10, 4.6, ncols=2)
    ypos = np.arange(len(t))[::-1]
    for ax, k, title in ((axes[0], 'roc_auc', 'AUC difference (right = second model better)'),
                         (axes[1], 'brier', 'Brier difference (left = second model better)')):
        for yi, (_, r) in zip(ypos, t.iterrows()):
            ax.plot([r[f'{k}_diff_lo'], r[f'{k}_diff_hi']], [yi, yi], color=SERIES['C_playtype'] if 'C_playtype' in r['full'] and 'C_playtype' not in r['base'] else SERIES['A_basic'],
                    linewidth=2, solid_capstyle='round')
            ax.plot(r[f'{k}_diff'], yi, 'o', color=INK, markersize=6)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels(t['label'] if ax is axes[0] else [''] * len(t), fontsize=8.5, color=INK)
        ax.set_title(title, color=INK, fontsize=10, loc='left')
    fig.suptitle('Paired differences on identical QBs, 95% bootstrap intervals (green = adding play-type features)',
                 color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, 'model_playtype_increment.png')


def fig_coefficients(boot):
    b = boot.sort_values('boot_median')
    fig, ax = fig_ax(9.5, 6.0)
    y = np.arange(len(b))
    colors = [SERIES['C_playtype'] if f in F.GROUP_C_PLAYTYPE else SERIES['B_context'] if f in F.GROUP_B_CONTEXT
              else SERIES['A_basic'] for f in b['feature']]
    for yi, (_, r), c in zip(y, b.iterrows(), colors):
        ax.plot([r['boot_p05'], r['boot_p95']], [yi, yi], color=c, linewidth=2, solid_capstyle='round')
        ax.plot(r['boot_median'], yi, 'o', color=c, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{FEATURE_LABEL.get(f, f)} ({100 * s:.0f}% > 0)" for f, s in zip(b['feature'], b['share_positive'])],
                       fontsize=8, color=INK)
    ax.set_xlabel('Log-odds per SD: bootstrap median and 90% interval', color=INK, fontsize=9)
    fig.suptitle('Model C coefficients: associations, not causes (blue = basic, orange = context, green = play-type)',
                 color=INK, fontsize=10.5, x=0.01, ha='left')
    save(fig, 'model_coefficients_C.png')


def fig_importance(imp):
    if imp.empty:
        return
    w = (imp.assign(wi=imp['importance_mean'] * imp['n_test'])
         .groupby(['model', 'feature']).agg(wi=('wi', 'sum'), n=('n_test', 'sum')).reset_index())
    w['importance'] = w['wi'] / w['n']
    models = [m for m in ('B_context_l2', 'C_playtype_l2', 'C_playtype_gbdt') if m in set(w['model'])]
    fig, axes = fig_ax(5.0 * len(models), 5.4, ncols=len(models))
    for ax, m in zip(np.atleast_1d(axes), models):
        g = w[w['model'] == m].sort_values('importance')
        colors = [SERIES['C_playtype'] if f in F.GROUP_C_PLAYTYPE else SERIES['B_context'] if f in F.GROUP_B_CONTEXT
                  else SERIES['A_basic'] for f in g['feature']]
        ax.barh(np.arange(len(g)), g['importance'], color=colors, height=0.7, edgecolor=SURFACE, linewidth=2)
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(np.arange(len(g)))
        ax.set_yticklabels([FEATURE_LABEL.get(f, f) for f in g['feature']], fontsize=8, color=INK)
        ax.set_title(m, color=INK, fontsize=10, loc='left')
        ax.set_xlabel('Brier increase when permuted (held-out folds)', color=INK, fontsize=8)
    fig.suptitle('Out-of-sample permutation importance, averaged over the 2019/2020/2021 folds', color=INK,
                 fontsize=11, x=0.01, ha='left')
    save(fig, 'model_permutation_importance.png')


def fig_pdp(cohort, features):
    m_l2 = joblib.load(T.MODEL_DIR / 'C_playtype_l2.joblib')
    m_gb = joblib.load(T.MODEL_DIR / 'C_playtype_gbdt.joblib')
    X = cohort[m_l2['features']]
    fig, axes = fig_ax(4.0 * len(features), 3.8, ncols=len(features))
    for ax, f in zip(np.atleast_1d(axes), features):
        for art, ls, lab in ((m_l2, '-', 'L2 logistic'), (m_gb, '--', 'Gradient boosting')):
            r = partial_dependence(art['pipeline'], X, [f], grid_resolution=20, kind='average')
            ax.plot(r['grid_values'][0], r['average'][0], color=INK, linestyle=ls, linewidth=2, label=lab)
        ax.plot(X[f], np.full(len(X), ax.get_ylim()[0]), '|', color=INK_2, markersize=8)
        ax.set_title(FEATURE_LABEL.get(f, f), color=INK, fontsize=10, loc='left')
        ax.set_ylabel('Average predicted P(success)', color=INK, fontsize=8)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Partial dependence, final model C fit on all 53 QBs (ticks = observed values; in-sample)',
                 color=INK, fontsize=11, x=0.01, ha='left')
    save(fig, 'model_partial_dependence.png')


def fig_correlations(cohort):
    cols = F.FEATURE_SETS['C_playtype'] + F.DRAFT_REFERENCE
    corr = cohort[cols].corr(method='spearman')
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('div', ['#2a78d6', '#f0efec', '#eb6834'])
    fig, ax = plt.subplots(figsize=(8.6, 7.4), facecolor=SURFACE)
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    labels = [FEATURE_LABEL.get(c, c) for c in cols]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=7.5, color=INK)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(labels, fontsize=7.5, color=INK)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            if i != j and abs(v) >= 0.4:
                ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=6.5, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label('Spearman correlation (|r| >= 0.4 labelled)', fontsize=8)
    ax.set_title('Feature correlations, comparison cohort (n=53)', color=INK, fontsize=11, loc='left')
    save(fig, 'model_feature_correlations.png')
    return corr


# ---------------------------------------------------------------- report
def f3(v):
    return '-' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.3f}'


def ci(r, k):
    return f'{r[k]:.3f} [{r[k + "_lo"]:.3f}, {r[k + "_hi"]:.3f}]'


def render(mt, folds, pdiff, sens_t, boot, coefs, imp_w, contrib_abs, corr, meta, ready) -> str:
    P = mt[mt['experiment'] == 'primary_5yr_32starts'].set_index('model')
    PD = pdiff.set_index(['experiment', 'base', 'full'])
    bc = PD.loc[('primary_5yr_32starts', 'B_context_l2', 'C_playtype_l2')]
    ab = PD.loc[('primary_5yr_32starts', 'A_basic_l2', 'B_context_l2')]
    s3 = PD.loc[('supp_3yr_16starts', 'B_context_l2', 'C_playtype_l2')]
    gb = PD.loc[('primary_5yr_32starts', 'B_context_gbdt', 'C_playtype_gbdt')]
    en = PD.loc[('primary_5yr_32starts', 'B_context_enet', 'C_playtype_enet')]
    L = ['# NFL QB career-success model: evaluation', '',
         f'Generated by `python -m src.modeling.evaluate` from out-of-sample predictions written by `train.py` '
         f'(model version {meta["model_version"]}, seed {meta["random_seed"]}, dataset {meta["dataset"]["dataset_version"]}). '
         'Every number below is out-of-sample: each QB is scored by a model trained only on earlier draft classes.', '',
         '## Bottom line', '',
         f'* **Target:** 32+ regular-season NFL starts in the first five seasons. **Sample:** {meta["training_rows"]} '
         f'drafted QBs from the 2017-2021 classes ({meta["training_positive"]} successes). Of those, '
         f'{int(P.loc["A_basic_l2", "n"])} QBs from the 2019-2021 classes ({int(P.loc["A_basic_l2", "n_pos"])} successes) are scored out-of-sample.',
         f'* **College statistics carry real signal.** Basic stats reach AUC {ci(P.loc["A_basic_l2"], "roc_auc")}, '
         f'against 0.5 for no skill. Adding opponent/team context gives AUC {ci(P.loc["B_context_l2"], "roc_auc")} and '
         f'Brier {P.loc["B_context_l2", "brier"]:.3f}, versus {P.loc["prior_rate", "brier"]:.3f} for the base-rate baseline. '
         f'The A-to-B gain (AUC {ab["roc_auc_diff"]:+.3f}, 95% CI [{ab["roc_auc_diff_lo"]:+.3f}, {ab["roc_auc_diff_hi"]:+.3f}]) '
         'points in the right direction but is not statistically clear with 31 QBs.',
         f'* **Play-type features did not improve the primary model.** B to C with L2 logistic changes AUC by '
         f'{bc["roc_auc_diff"]:+.3f} [{bc["roc_auc_diff_lo"]:+.3f}, {bc["roc_auc_diff_hi"]:+.3f}] and Brier by '
         f'{bc["brier_diff"]:+.3f} [{bc["brier_diff_lo"]:+.3f}, {bc["brier_diff_hi"]:+.3f}] (positive Brier = worse). '
         f'Elastic net agrees (AUC {en["roc_auc_diff"]:+.3f}). With gradient boosting, play types improved Brier '
         f'({gb["brier_diff"]:+.3f} [{gb["brier_diff_lo"]:+.3f}, {gb["brier_diff_hi"]:+.3f}]) but lowered AUC '
         f'({gb["roc_auc_diff"]:+.3f}). On the larger 3-season task the change is AUC {s3["roc_auc_diff"]:+.3f} '
         f'[{s3["roc_auc_diff_lo"]:+.3f}, {s3["roc_auc_diff_hi"]:+.3f}]. **Verdict: no detectable improvement.** Every '
         'interval includes zero, and the AUC point estimate is slightly negative in all three model families and on '
         'the 3-season task. There is no sign of a benefit, but with this sample a small one cannot be ruled out.',
         f'* **Draft position (reference only, not a college model)** scores AUC {ci(P.loc["draft_reference_l2"], "roc_auc")} '
         f'and Brier {P.loc["draft_reference_l2", "brier"]:.3f}. NFL teams\' own evaluation (and the playing time a high '
         'pick is given) already encodes more than these college features do. Part of this is self-fulfilling: '
         'early picks are handed starts.',
         '* **No model is production-ready.** With 31 holdout QBs, AUC intervals span roughly +/-0.1 to +/-0.17. Treat '
         'these numbers as an early estimate, not a validated accuracy figure. Boosting with context features '
         '(`B_context_gbdt`) has the highest AUC in the table, but picking the best of 11 models on the same 31 QBs '
         'would overfit the holdout. The pre-specified primary comparison is the L2 logistic series.', '',
         '## Design', '',
         '* **Rows:** the same-QB comparison cohort from `reports/model_training_readiness.md`. Every model in every '
         'table is fit and scored on exactly these QBs.',
         '* **Validation:** rolling-origin by draft class. 2019 is predicted from 2017-2018 (22 training QBs), 2020 from '
         '2017-2019 (32), and 2021 from 2017-2020 (44). There is no random splitting.',
         '* **Leakage control:** preprocessing (median imputation, standardisation) sits inside sklearn Pipelines, so '
         'it is re-fit on each fold\'s training classes only. Features are pre-draft only. Draft slot appears only in '
         'the reference model. Tests check that training classes are always earlier than the test class.',
         f'* **Hyperparameters were pre-specified, not tuned:** L2 C={T.L2_C}; elastic net C={T.ENET_C}, '
         f'l1_ratio={T.ENET_L1_RATIO}; gradient boosting with 100 trees, depth 2, learning rate 0.05, subsample 0.8 '
         'and min leaf 4. A C-sensitivity grid is shown below and was not used for selection.',
         '* **Decision threshold:** each fold\'s training success rate (0.31-0.34), not 0.5. Successes are a '
         'third of the sample, and a 0.5 cut-off would trade recall away arbitrarily.',
         '* **XGBoost was not run.** It is not installed and is not statistically justified with about 19 positive '
         'events. sklearn gradient boosting stands in as an exploratory tree model. No neural networks.', '',
         '## Primary results (out-of-sample, 2019-2021 classes)', '',
         '| Model | Features | ROC-AUC [95% CI] | Brier [95% CI] | Log loss | Brier skill vs base rate | Precision | Recall | TP/FP/TN/FN |',
         '|---|---|---|---|---|---|---|---|---|']
    for m in ['prior_rate', 'A_basic_l2', 'B_context_l2', 'C_playtype_l2', 'A_basic_enet', 'B_context_enet',
              'C_playtype_enet', 'A_basic_gbdt', 'B_context_gbdt', 'C_playtype_gbdt', 'draft_reference_l2']:
        r = P.loc[m]
        nfeat = 0 if m == 'prior_rate' else len(F.FEATURE_SETS[r['feature_set']])
        auc_s = 'n/a (constant within fold)' if m == 'prior_rate' else ci(r, 'roc_auc')
        L.append(f'| `{m}` | {nfeat} | {auc_s} | {ci(r, "brier")} | {f3(r["log_loss"])} | '
                 f'{r["brier_skill_vs_prior"]:+.2f} | {f3(r["precision"])} | {f3(r["recall"])} | '
                 f'{r["tp"]}/{r["fp"]}/{r["tn"]}/{r["fn"]} |')
    L += ['', 'Precision and recall use the documented threshold (training success rate). Accuracy is deliberately '
          'not headlined: with a 39% success rate in the holdout, always predicting "no" scores 61%. The base-rate '
          'model\'s pooled AUC is not meaningful, because its prediction is constant within each fold.', '',
          '![ROC curves](figures/model_roc_curves.png)', '', '![Calibration](figures/model_calibration.png)', '',
          '![Metric comparison](figures/model_metric_comparison.png)', '',
          '### By test class', '', '| Model | 2019 AUC / Brier | 2020 AUC / Brier | 2021 AUC / Brier (latest holdout) |',
          '|---|---|---|---|']
    pf = folds[folds['experiment'] == 'primary_5yr_32starts']
    for m in ['A_basic_l2', 'B_context_l2', 'C_playtype_l2', 'C_playtype_gbdt', 'draft_reference_l2']:
        cells = []
        for ty in T.TEST_YEARS:
            r = pf[(pf['model'] == m) & (pf['test_year'] == ty)].iloc[0]
            cells.append(f'{f3(r["roc_auc"])} / {f3(r["brier"])} (n={r["n"]}, {r["n_pos"]}+)')
        L.append(f'| `{m}` | ' + ' | '.join(cells) + ' |')
    L += ['', 'Single-class results rest on 9-12 QBs each. The 2021 column is the "latest fully observed class" '
          'holdout, shown for completeness. It is far too small to judge a model on its own.', '',
          '### Calibration', '',
          'Calibration is estimated with 4 quantile bins of about 8 QBs (expected calibration error, 4 bins: ' +
          ', '.join(f'{m} {P.loc[m, "ece_4bin"]:.3f}' for m in ['A_basic_l2', 'B_context_l2', 'C_playtype_l2', 'draft_reference_l2']) +
          '). The bins are too coarse to diagnose miscalibration reliably. The Brier score, which combines '
          'calibration and discrimination, is the better summary at this size.', '',
          '## Do detailed play types improve predictions?', '',
          'Paired comparison on identical QBs and folds, with bootstrap resampling of QBs. AUC: a positive difference '
          'means the second model is better. Brier and log loss: a negative difference means the second model is better.', '',
          '| Task | Comparison | AUC diff [95% CI] | P(boot: better AUC) | Brier diff [95% CI] | P(boot: better Brier) | Log-loss diff |',
          '|---|---|---|---|---|---|---|']
    for _, r in pdiff.iterrows():
        task = '5-yr, 32 starts' if r['experiment'].startswith('primary') else '3-yr, 16 starts'
        L.append(f'| {task} | `{r["full"]}` vs `{r["base"]}` | {r["roc_auc_diff"]:+.3f} [{r["roc_auc_diff_lo"]:+.3f}, '
                 f'{r["roc_auc_diff_hi"]:+.3f}] | {r["roc_auc_share_full_better"]:.0%} | {r["brier_diff"]:+.3f} '
                 f'[{r["brier_diff_lo"]:+.3f}, {r["brier_diff_hi"]:+.3f}] | {r["brier_share_full_better"]:.0%} | '
                 f'{r["log_loss_diff"]:+.3f} |')
    L += ['', '![Play-type increment](figures/model_playtype_increment.png)', '',
          'Reading: the context features (B) are the most consistent addition. The five play-type features (C) add '
          'nothing detectable on top of B. The only comparison whose interval clears zero is B over A on the 3-season '
          'task, and only just, with 8 comparisons run. The linear models get slightly worse with play types, which is what five extra noisy '
          'features cost when there are 22-44 training QBs. Gradient boosting improves its Brier but loses AUC. Two '
          'caveats limit how far "no improvement" can be pushed. (1) The play-type profile describes the QB\'s team '
          'offense, not the QB himself: the slim play cache has no passer id, and CFBD play types do not identify '
          'RPO, play-action or formations. (2) With 31 holdout QBs, a true AUC gain under about 0.10 is undetectable.', '',
          '## Supplementary analyses', '',
          '### Shorter horizon: 16+ starts in the first 3 seasons (2017-2023 classes)', '']
    S = mt[mt['experiment'] == 'supp_3yr_16starts'].set_index('model')
    r0 = S.iloc[0]
    L += [f'More classes have complete 3-season windows, so this adds the 2022-2023 classes: {int(r0["n"])} '
          f'out-of-sample QBs ({int(r0["n_pos"])} successes), rolling over the 2019-2023 test classes.', '',
          '| Model | ROC-AUC [95% CI] | Brier [95% CI] | Brier skill |', '|---|---|---|---|']
    for m in ['prior_rate', 'A_basic_l2', 'B_context_l2', 'C_playtype_l2', 'draft_reference_l2']:
        r = S.loc[m]
        L.append(f'| `{m}` | {"n/a" if m == "prior_rate" else ci(r, "roc_auc")} | {ci(r, "brier")} | {r["brier_skill_vs_prior"]:+.2f} |')
    H = mt[mt['experiment'] == 'supp_history_augmented'].set_index('model')
    L += ['', 'Same pattern: context helps, play types add nothing detectable, and draft position is strongest. '
          'Signal is weaker at the shorter horizon, since many good QBs sit behind a veteran for 1-2 years.', '',
          '### Adding the 2005-2016 classes to training (A/B feature sets only)', '',
          f'Older classes have no play-by-play, so this check covers only models without play types. Their '
          f'{ready["history_eligible"]} QBs ({ready["history_positive"]} successes) are added to every fold\'s '
          'training set, and the model is scored on the same 31 holdout QBs.', '',
          '| Model | AUC, cohort-only training | AUC, + 2005-2016 | Brier, cohort-only | Brier, + 2005-2016 |',
          '|---|---|---|---|---|']
    for m in ['A_basic_l2', 'B_context_l2', 'draft_reference_l2']:
        L.append(f'| `{m}` | {f3(P.loc[m, "roc_auc"])} | {f3(H.loc[m, "roc_auc"])} | {f3(P.loc[m, "brier"])} | {f3(H.loc[m, "brier"])} |')
    L += ['', 'Adding 137 older QBs did **not** improve the college-stat models on recent classes: AUC and Brier both got worse '
          'for A, and slightly worse for B. This suggests era drift: college passing numbers inflated sharply after '
          '2010, so the same completion % or YPA means something different. Era-normalising college stats (for '
          'example, z-scores within season) is the next thing to try before pooling eras. Draft position is stable '
          'across eras.', '',
          '### Regularisation sensitivity (not used for model selection)', '',
          '| C | A AUC | B AUC | C AUC | A Brier | B Brier | C Brier |', '|---|---|---|---|---|---|---|']
    for c, r in sens_t.iterrows():
        L.append(f'| {c} | {f3(r[("auc", "A_basic")])} | {f3(r[("auc", "B_context")])} | {f3(r[("auc", "C_playtype")])} | '
                 f'{f3(r[("brier", "A_basic")])} | {f3(r[("brier", "B_context")])} | {f3(r[("brier", "C_playtype")])} |')
    L += ['', 'In AUC, B >= C > A at every regularisation strength. In Brier, C only matches or slightly beats B '
          'under the strongest regularisation (C <= 0.1), where the play-type coefficients are shrunk toward zero. With '
          'weaker regularisation, C falls further behind B. The conclusion, no detectable gain from play types, holds '
          'at every C.', '',
          '## Explainability (associations, not causes)', '',
          'The final models are re-fit on all 53 cohort QBs for these explanations and for `predict.py`. '
          'Coefficients are standardised (log-odds per standard deviation). The bootstrap column shows how often '
          'a coefficient stays positive when the 53 QBs are resampled 1,000 times. Values near 50% mean the sign '
          'itself is unreliable.', '',
          '| Feature | Group | Coefficient (final model C) | Bootstrap median [90% interval] | % of resamples > 0 | Out-of-sample permutation importance (Brier) | Mean abs. log-odds contribution |',
          '|---|---|---|---|---|---|---|']
    cc = coefs[coefs['model'] == 'C_playtype_l2'].set_index('feature')['coef_standardized']
    bb = boot.set_index('feature')
    iw = imp_w[imp_w['model'] == 'C_playtype_l2'].set_index('feature')['importance']
    for f in bb.sort_values('boot_median', key=np.abs, ascending=False).index:
        grp = 'C play-type' if f in F.GROUP_C_PLAYTYPE else 'B context' if f in F.GROUP_B_CONTEXT else 'A basic'
        L.append(f'| {FEATURE_LABEL.get(f, f)} | {grp} | {cc[f]:+.3f} | {bb.loc[f, "boot_median"]:+.3f} '
                 f'[{bb.loc[f, "boot_p05"]:+.3f}, {bb.loc[f, "boot_p95"]:+.3f}] | {bb.loc[f, "share_positive"]:.0%} | '
                 f'{iw.get(f, float("nan")):+.4f} | {contrib_abs[f]:.3f} |')
    high = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.5:
                high.append((a, b, corr.loc[a, b]))
    L += ['', '![Coefficients](figures/model_coefficients_C.png)', '',
          '![Permutation importance](figures/model_permutation_importance.png)', '',
          '![Partial dependence](figures/model_partial_dependence.png)', '',
          'The "mean abs. log-odds contribution" column gives exact SHAP values for the linear model, assuming '
          'independent features: coefficient times standardised value. It was computed directly, because the optional '
          '`shap` package is not installed and is not needed for a linear model. Permutation importance is measured on '
          'the held-out folds, weighted by fold size. Negative values mean that shuffling the feature *improved* '
          'held-out Brier, i.e. the feature was adding noise.', '',
          '**Correlated features** (|Spearman| >= 0.5) split importance between them, so neither member of a pair '
          'should be read alone:', '']
    for a, b, v in sorted(high, key=lambda x: -abs(x[2])):
        L.append(f'* {FEATURE_LABEL.get(a, a)} - {FEATURE_LABEL.get(b, b)}: {v:+.2f}')
    L += ['', '![Correlations](figures/model_feature_correlations.png)', '',
          '**What the model leans on.** Signs that hold in at least 90% of bootstrap resamples: younger age at draft (-), '
          'higher final-season completion % (+), more career rushing yards (+), and a *less* talented recruiting class '
          'around the QB (-, 94%). Two play-type features have positive signs in about 90% of resamples (passing-down '
          'pass rate, explosive pass rate), but the explosive-pass rate is 0.90 correlated with yards per attempt, so the '
          'two share one signal. Out of sample, the play-type features add nothing (see above).', '',
          'None of this is causal. The negative age coefficient does not mean that being younger causes success. '
          'Age at draft reflects early-entry decisions, which follow perceived talent, and it is 0.52 correlated with '
          'draft slot. The negative recruiting coefficient is consistent with "produced without elite teammates", but it '
          'also mixes in the QB\'s own recruitment and selection into the draft. Starts also depend on opportunity '
          '(draft slot, depth chart and team need), not only ability.', '',
          '## Limitations', '',
          '1. **Small sample.** There are 53 QBs and 19 successes in total, and 31 holdout QBs. Intervals are wide '
          'and differences under about 0.10 AUC cannot be detected. No result here is a validated accuracy figure.',
          '2. **Starts measure opportunity as well as ability.** High picks get starts that low picks would not. '
          'That is also why draft position predicts so well, partly by construction.',
          '3. **Threshold sensitivity.** Three QBs sit within 3 starts of 32: Love (33), Wilson (33) and Mills (30). '
          'Single games decide their labels.',
          '4. **The play-type profile is team-level.** It has no passer id, no RPO/play-action/formation data, and '
          'FCS QBs have none (4 eligible QBs excluded, all non-successes).',
          '5. **Era drift.** Pooling the 2005-2016 classes hurt. College statistics are not comparable across eras '
          'without normalisation.',
          '6. **No verified college starts, injury history or pro-day data** in any ingested source.',
          '7. **Hyperparameters were fixed a priori.** Results could shift modestly with other reasonable choices; '
          'the sensitivity grid suggests the ordering is stable.', '',
          '## Saved artifacts', '',
          f'* Models: `models/qb_career_success/v{meta["model_version"]}/*.joblib`. Each is a dict with the fitted sklearn '
          'Pipeline (imputer, scaler, model), the feature list and the threshold.',
          '* `metadata.json` records the training cutoff, dataset hashes, feature schema, all model parameters, the '
          'random seed and package versions.',
          '* Out-of-sample predictions: `results/modeling/oof_predictions.csv`. Metrics: `results/modeling/metrics.csv`, '
          '`paired_differences.csv` and `per_fold_metrics.csv`.', '',
          '## Reproduce', '', '```bash', 'python -m src.modeling.build_dataset   # dataset, targets, readiness report',
          'python -m src.modeling.train           # rolling-origin validation + final models',
          'python -m src.modeling.evaluate        # metrics, figures, this report',
          'python -m src.modeling.predict         # score the censored 2022-2026 classes (not evaluable yet)',
          'python -m pytest -q', '```', '']
    return '\n'.join(L)


def main() -> dict:
    oof = pd.read_csv(RESULTS / 'oof_predictions.csv')
    sens = pd.read_csv(RESULTS / 'sensitivity_predictions.csv')
    imp = pd.read_csv(RESULTS / 'permutation_importance_by_fold.csv')
    boot = pd.read_csv(RESULTS / 'coefficient_bootstrap_C_playtype_l2.csv')
    coefs = pd.read_csv(RESULTS / 'final_model_coefficients.csv')
    contrib = pd.read_csv(RESULTS / 'linear_contributions_C_playtype_l2.csv')
    meta = json.loads((T.MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))
    ready = json.loads((ROOT / 'results' / 'model_training_readiness.json').read_text(encoding='utf-8'))
    data = T.load_data()
    cohort = data[data['comparison_cohort']].reset_index(drop=True)

    mt = metrics_table(oof)
    folds = per_fold_table(oof)
    pdiff = pd.DataFrame([paired_difference(oof, *c) for c in COMPARISONS])
    sens_t = (sens.groupby(['C', 'feature_set'])
              .apply(lambda g: pd.Series({'auc': E.roc_auc(g['y_true'], g['p_success']),
                                          'brier': E.brier_score(g['y_true'], g['p_success'])}), include_groups=False)
              .unstack())
    imp_w = (imp.assign(wi=imp['importance_mean'] * imp['n_test']).groupby(['model', 'feature'])
             .agg(wi=('wi', 'sum'), n=('n_test', 'sum')).reset_index())
    imp_w['importance'] = imp_w['wi'] / imp_w['n']
    contrib_abs = contrib[F.FEATURE_SETS['C_playtype']].abs().mean()

    mt.to_csv(RESULTS / 'metrics.csv', index=False)
    folds.to_csv(RESULTS / 'per_fold_metrics.csv', index=False)
    pdiff.to_csv(RESULTS / 'paired_differences.csv', index=False)
    imp_w.to_csv(RESULTS / 'permutation_importance_weighted.csv', index=False)

    fig_roc(oof)
    fig_calibration(oof)
    fig_metric_comparison(mt)
    fig_increment(pdiff)
    fig_coefficients(boot)
    fig_importance(imp)
    top = (imp_w[imp_w['model'] == 'C_playtype_l2'].sort_values('importance', ascending=False)['feature'].head(3).tolist())
    fig_pdp(cohort, top)
    corr = fig_correlations(cohort)

    REPORT.write_text(render(mt, folds, pdiff, sens_t, boot, coefs, imp_w, contrib_abs, corr, meta, ready),
                      encoding='utf-8')
    summary = {'metrics': mt.to_dict('records'), 'paired_differences': pdiff.to_dict('records')}
    (RESULTS / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')
    p = mt[mt['experiment'] == 'primary_5yr_32starts'].set_index('model')
    print(p[['n', 'n_pos', 'roc_auc', 'roc_auc_lo', 'roc_auc_hi', 'brier', 'precision', 'recall']].round(3).to_string())
    return summary


if __name__ == '__main__':
    main()
