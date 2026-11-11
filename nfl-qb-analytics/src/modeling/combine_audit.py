"""Combine missingness and selection-bias audit (Phase 5.5).

Run:  python -m src.modeling.combine_audit        (local data only)

Question: is the Phase-5 gain from "college + Combine" (AUC 0.878 -> 0.907) athletic information,
or information about the scouting process (who was invited)? All variants are fit and scored on
IDENTICAL rows (the verified prospect universe, same chronological backtest and one-year label lag):

  A  college                       college features only (available right after the final season)
  B  college + invited             + Combine participation indicator (invitations: early February)
  C  college + invited + measures  the Phase-5 model (forty, height, weight; measured early March)
  D  college + measures, no flag   measurements median-imputed for non-participants, NO indicator
  D2 college + height/weight only  the measurements that duplicate roster data, no flag, no forty
Within Combine participants only (participation held fixed):
  I_A college                      vs  I_C college + measurements

Prediction cutoffs: A is the pre-Combine forecast, B the post-invitation one, C/D post-Combine.
No variant uses information from after its cutoff (the draft outcome is never a feature).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.modeling import backtesting as BT
from src.modeling import draft_predictor as DP
from src.modeling import uncertainty as U
from src.modeling.reporting import md_table

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5_5'
REPORT = ROOT / 'reports' / 'combine_selection_bias.md'
SEED = DP.SEED
MEASURES = ['combine_forty', 'combine_height_inches', 'combine_weight_lbs']
VARIANTS = {
    'A_college': DP.COLLEGE_FEATURES,
    'B_college_invited': DP.COLLEGE_FEATURES + ['combine_invited'],
    'C_college_invited_measures': DP.COLLEGE_FEATURES + DP.COMBINE_FEATURES,
    'D_college_measures_no_flag': DP.COLLEGE_FEATURES + MEASURES,
    'D2_college_height_weight_no_flag': DP.COLLEGE_FEATURES + ['combine_height_inches', 'combine_weight_lbs'],
}
CUTOFF = {'A_college': 'after final season (pre-Combine)', 'B_college_invited': 'after invitations (early February)',
          'C_college_invited_measures': 'after the Combine (early March)', 'D_college_measures_no_flag': 'after the Combine',
          'D2_college_height_weight_no_flag': 'after the Combine'}


def logit():
    return DP.pipe(LogisticRegression(C=0.5, max_iter=5000))


def backtest(pop: pd.DataFrame, variants: dict) -> pd.DataFrame:
    def fp(train, test):
        out = pd.DataFrame({'cfbd_player_id': test['cfbd_player_id'].to_numpy(), 'y': test['drafted'].to_numpy(),
                            'invited': test['combine_invited'].to_numpy()})
        for k, cols in variants.items():
            out[k] = logit().fit(train[cols], train['drafted']).predict_proba(test[cols])[:, 1]
        return out
    return BT.run(pop.reset_index(drop=True), 'draft_year', DP.PROB_TEST_YEARS, BT.LAG_DRAFT, fp, min_train=50)


def metrics(b: pd.DataFrame, models) -> pd.DataFrame:
    from src.modeling.evaluate import fast_auc, fast_brier, fast_log_loss
    d = b.rename(columns={'test_year': 'draft_year'})
    rows = []
    for m in models:
        c = U.calibration_summary(d['y'], d[m])
        a = U.cluster_bootstrap(lambda x: fast_auc(x['y'], x[m]), d, n_boot=1000, seed=SEED)
        rows.append({'variant': m, 'n': len(d), 'n_drafted': int(d['y'].sum()), 'auc': fast_auc(d['y'], d[m]),
                     'auc_lo': a['lower'], 'auc_hi': a['upper'], 'brier': fast_brier(d['y'], d[m]),
                     'log_loss': fast_log_loss(d['y'], np.clip(d[m], 1e-4, 1 - 1e-4)),
                     'calibration_slope': c['calibration_slope']})
    return pd.DataFrame(rows)


def paired(b: pd.DataFrame, pairs) -> pd.DataFrame:
    from src.modeling.evaluate import fast_auc, fast_brier
    d = b.rename(columns={'test_year': 'draft_year'})
    rows = []
    for a, c in pairs:
        r = U.paired_cluster_bootstrap(lambda y, p: fast_brier(y, p), d, a, c, n_boot=1000, seed=SEED)
        ra = U.cluster_bootstrap(lambda x: fast_auc(x['y'], x[c]) - fast_auc(x['y'], x[a]), d, n_boot=1000, seed=SEED)
        rows.append({'from': a, 'to': c, 'brier_change': r['estimate'], 'brier_lo': r['lower'], 'brier_hi': r['upper'],
                     'p_to_better': r['p_b_lower'], 'auc_change': ra['estimate'], 'auc_lo': ra['lower'], 'auc_hi': ra['upper']})
    return pd.DataFrame(rows)


def missingness(pop: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inv, g in pop.groupby('combine_invited'):
        rows.append({'combine_participant': bool(inv), 'rows': len(g), 'draft_rate': float(g['drafted'].mean()),
                     **{f'{c}_missing': float(g[c].isna().mean()) for c in MEASURES}})
    inv = pop[pop['combine_invited'] == 1]
    for miss, g in inv.groupby(inv['combine_forty'].isna()):
        rows.append({'combine_participant': f'participant, forty {"missing" if miss else "recorded"}', 'rows': len(g),
                     'draft_rate': float(g['drafted'].mean()), **{f'{c}_missing': float(g[c].isna().mean()) for c in MEASURES}})
    return pd.DataFrame(rows)


def coefficient_table(pop: pd.DataFrame) -> pd.DataFrame:
    m = logit().fit(pop[VARIANTS['C_college_invited_measures']], pop['drafted'])
    coef = m.named_steps['model'].coef_[0]
    t = pd.DataFrame({'feature': VARIANTS['C_college_invited_measures'], 'standardized_coef': coef})
    t['abs'] = t['standardized_coef'].abs()
    return t.sort_values('abs', ascending=False).drop(columns='abs').head(10)


def main() -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(DP.DATA_FILE)
    pop = u[u['in_population']].reset_index(drop=True)
    b = backtest(pop, VARIANTS)
    b.to_csv(RESULTS / 'combine_backtest.csv', index=False)
    M = metrics(b, list(VARIANTS))
    P = paired(b, [('A_college', 'B_college_invited'), ('B_college_invited', 'C_college_invited_measures'),
                   ('A_college', 'C_college_invited_measures'), ('A_college', 'D_college_measures_no_flag'),
                   ('A_college', 'D2_college_height_weight_no_flag')])
    inv = pop[pop['combine_invited'] == 1].reset_index(drop=True)
    bi = backtest(inv, {'I_A_college': DP.COLLEGE_FEATURES, 'I_C_college_measures': DP.COLLEGE_FEATURES + MEASURES})
    MI = metrics(bi, ['I_A_college', 'I_C_college_measures'])
    PI = paired(bi, [('I_A_college', 'I_C_college_measures')])
    # how much of the separation is the participation flag itself?
    from src.modeling.evaluate import fast_auc
    flag_auc = fast_auc(b['y'], b['invited'])
    by_inv = (b.assign(group=np.where(b['invited'] == 1, 'participants', 'non-participants'))
              .groupby('group').apply(lambda g: pd.Series({'rows': len(g), 'draft_rate': g['y'].mean(),
                                                          **{f'mean_p_{k}': g[k].mean() for k in ('A_college', 'B_college_invited', 'C_college_invited_measures')}}),
                                      include_groups=False).reset_index())
    miss = missingness(pop)
    coef = coefficient_table(pop)
    for n, t in (('combine_metrics', M), ('combine_paired', P), ('combine_invited_only_metrics', MI),
                 ('combine_invited_only_paired', PI), ('combine_missingness', miss), ('combine_by_participation', by_inv)):
        t.to_csv(RESULTS / f'{n}.csv', index=False)
    summary = {'rows': int(len(b)), 'drafted': int(b['y'].sum()), 'participation_flag_auc_alone': flag_auc,
               'invited_only_rows': int(len(bi))}
    (RESULTS / 'combine_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    REPORT.write_text(render(summary, M, P, MI, PI, miss, coef, by_inv, pop), encoding='utf-8')
    print(M.round(3).to_string()); print(P.round(3).to_string()); print(MI.round(3).to_string())
    return summary


def render(summary, M, P, MI, PI, miss, coef, by_inv, pop) -> str:
    m = M.set_index('variant')
    p = P.set_index(['from', 'to'])
    ab = p.loc[('A_college', 'B_college_invited')]
    bc = p.loc[('B_college_invited', 'C_college_invited_measures')]
    ad = p.loc[('A_college', 'D_college_measures_no_flag')]
    ic = PI.iloc[0]
    share_flag = (m.loc['B_college_invited', 'auc'] - m.loc['A_college', 'auc']) / max(
        m.loc['C_college_invited_measures', 'auc'] - m.loc['A_college', 'auc'], 1e-9)
    L = ['# Combine selection bias (Phase 5.5)', '',
         'Generated by `python -m src.modeling.combine_audit`. Population: the Phase-5 verified prospect universe '
         f'({len(pop)} final seasons, {int(pop["drafted"].sum())} drafted). Backtest: test drafts '
         f'{DP.PROB_TEST_YEARS[0]}-{DP.PROB_TEST_YEARS[-1]}, training on earlier drafts only, every variant scored on the same '
         f'{summary["rows"]} rows ({summary["drafted"]} drafted).', '',
         '## Answer', '',
         f'* **Most of the Combine gain is participation, not measurement.** Adding only the participation flag (B) moves '
         f'AUC from {m.loc["A_college", "auc"]:.3f} to {m.loc["B_college_invited", "auc"]:.3f}, '
         f'{min(share_flag, 1):.0%} of the full gain to {m.loc["C_college_invited_measures", "auc"]:.3f} (C). Brier changes by '
         f'{ab["brier_change"]:+.4f} (95% class-bootstrap interval {ab["brier_lo"]:+.4f} to {ab["brier_hi"]:+.4f}).',
         f'* Adding forty time, height and weight on top of the flag (B to C) changes Brier by {bc["brier_change"]:+.4f} '
         f'({bc["brier_lo"]:+.4f} to {bc["brier_hi"]:+.4f}).',
         f'* Measurements **without** the flag (D, non-participants median-imputed) change Brier by {ad["brier_change"]:+.4f} '
         f'against college-only ({ad["brier_lo"]:+.4f} to {ad["brier_hi"]:+.4f}): '
         + ('**no improvement**. Without the participation signal, the measurements add nothing a linear model can use.'
            if ad['brier_change'] >= 0 else
            'a small gain, part of which may still be participation leaking through imputation (a median-imputed forty '
            'marks a non-participant).'),
         f'* **Among participants only** (participation fixed, {summary["invited_only_rows"]} backtest rows), adding measurements '
         f'changes AUC by {ic["auc_change"]:+.3f} ({ic["auc_lo"]:+.3f} to {ic["auc_hi"]:+.3f}) and Brier by {ic["brier_change"]:+.4f} '
         f'({ic["brier_lo"]:+.4f} to {ic["brier_hi"]:+.4f}). This is the cleanest test of athletic information.',
         f'* The participation flag on its own ranks drafted vs undrafted QBs with AUC {summary["participation_flag_auc_alone"]:.3f}. '
         'An invitation is the NFL\'s own scouting verdict, published about two months before the draft.', '',
         'So, in the full population, the "college + Combine" model mostly learns **what NFL scouts had already decided**, not '
         'how fast or big a QB is. '
         + ('Among QBs who were measured, the measurements do carry a small, real signal (the participant-only interval '
            'excludes zero), but too small to show up once the flag is in the model. '
            if ic['auc_lo'] > 0 else 'Even among QBs who were measured, the measurements add no clear signal. ')
         + 'Participation is still legitimate pre-draft information at its cutoff, but it should be presented as such: a '
         'post-invitation forecast, not evidence that athletic testing predicts the draft.', '',
         '## All variants on identical rows', '',
         md_table(M.assign(cutoff=M['variant'].map(CUTOFF))), '',
         '## Paired differences (class-cluster bootstrap)', '',
         'Negative Brier change = the second model is better. `p_to_better` = share of bootstrap resamples where it is.', '',
         md_table(P), '',
         '## Participants only', '',
         md_table(MI), '', md_table(PI), '',
         '## Missingness is informative', '',
         md_table(miss), '',
         'Measurements exist only for participants, so "missing forty" in the full population means "not invited". Among '
         'participants, skipping the forty is itself a signal (top prospects often skip drills), so even the '
         'participant-only model has informative missingness. Median imputation hides the flag from a linear model only '
         'partly.', '',
         '## Predicted probabilities by participation', '',
         md_table(by_inv), '',
         '## Largest standardized coefficients, model C (all rows)', '',
         md_table(coef), '',
         '## What changes in the system', '',
         '* The forecast interface reports draft probability at **separate cutoffs**: `pre_combine` (college only, model A '
         '= `models/qb_draft/v1.0.0/prob_college_l2.joblib`) and `post_combine` (model C = `prob_college_combine_l2.joblib`). '
         'The post-Combine number is labelled as driven mainly by participation.',
         '* A QB with no Combine record in a year whose Combine has not happened yet gets the pre-Combine number only. '
         'The interface never treats "no record yet" as "not invited".',
         '* No model artifact was changed. The Phase-5 headline (AUC 0.907) stays correct as a post-Combine backtest '
         'number, but this audit changes what it means.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
