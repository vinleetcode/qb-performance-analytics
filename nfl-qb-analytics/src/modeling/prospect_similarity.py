"""Historical QB prospect comparisons (Phase 5).

Run:  python -m src.modeling.prospect_similarity

For a college QB, find the most similar previously DRAFTED QBs by college profile.

* Profile features (pre-draft only; no NFL outcome ever enters a distance): final-season
  completion %, YPA, TD %, INT %, attempts per game, rushing yards per game (last 3 seasons),
  opponent-adjusted YPA, schedule strength, team pass rate, team recruiting percentile,
  age at draft, height, final-season YPA change, seasons as primary passer.
* Distance: Euclidean distance on z-scores, with the scaler fit ONLY on the reference pool
  (drafted QBs from classes before the query's class). Missing values are skipped and the
  distance rescaled (nan-Euclidean), never imputed with a future-informed value.
* Pool: classes strictly before the query's draft class. In backtests, comps' NFL outcomes
  are used only if they were observable at the query's draft (class <= query - 5).
* `closeness_index` (0-100) is a transparent HEURISTIC, not a probability: the share of all
  pairwise distances among pool QBs that are larger than this pair's distance.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import StandardScaler

from src.modeling import backtesting as BT
from src.modeling import train_v2 as T2

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5'
REPORT = ROOT / 'reports' / 'prospect_comparisons.md'
MODEL_VERSION = '1.0.0'
MODEL_DIR = ROOT / 'models' / 'qb_similarity' / f'v{MODEL_VERSION}'
DATA_DIR = ROOT / 'data' / 'model_artifacts' / 'qb_similarity' / f'v{MODEL_VERSION}'
K = 5
K_OUTCOME = 10
SIM_FEATURES = ['cmp__raw', 'ypa__raw', 'td__raw', 'int__raw', 'att_pg__raw', 'rush_pg__raw', 'opp_ypa__raw', 'sos__raw',
                'env_pass_rate', 'talent_recruiting_4yr_pctile', 'age_at_draft', 'height_inches', 'dev_yoy_ypa_final',
                'seasons_as_primary_passer']
LABELS = {'cmp__raw': 'completion %', 'ypa__raw': 'yards/attempt', 'td__raw': 'TD %', 'int__raw': 'INT %',
          'att_pg__raw': 'attempts/game', 'rush_pg__raw': 'rush yds/game', 'opp_ypa__raw': 'opp.-adjusted YPA',
          'sos__raw': 'opponent pass-D YPA (low = tough)', 'env_pass_rate': 'team pass rate',
          'talent_recruiting_4yr_pctile': 'team recruiting pct.', 'age_at_draft': 'age at draft', 'height_inches': 'height',
          'dev_yoy_ypa_final': 'final-year YPA change', 'seasons_as_primary_passer': 'seasons as primary passer'}
OUTCOME_COLS = ['nfl_starts_first_5_seasons', 'career_success', 'success_5yr_16', 'success_5yr_64',
                'nfl_starts_first_5_seasons_to_date', 'window_5yr_complete']


def load_pool() -> pd.DataFrame:
    d = T2.load()
    d = d[d['v2_feature_eligible'] & d['draft_year'].between(2005, 2026)].copy()
    return d.reset_index(drop=True)


class Comparator:
    def __init__(self, features=SIM_FEATURES):
        self.features = list(features)

    def fit(self, pool: pd.DataFrame):
        assert not [c for c in self.features if c.startswith(('nfl_', 'career_success', 'success_'))]
        self.pool = pool.reset_index(drop=True)
        self.scaler = StandardScaler().fit(self.pool[self.features])     # NaN-aware: ignores missing values
        self.Z = self.scaler.transform(self.pool[self.features])
        D = nan_euclidean_distances(self.Z)
        iu = np.triu_indices(len(self.Z), 1)
        self.pair_d = np.sort(D[iu][np.isfinite(D[iu])])
        return self

    def closeness_index(self, d):
        """Heuristic 0-100: % of pool pairs farther apart than d (NOT a probability)."""
        return 100 * (1 - np.searchsorted(self.pair_d, d, side='left') / len(self.pair_d))

    def query(self, row: pd.Series, k: int = K, as_of_year: int | None = None) -> pd.DataFrame:
        z = self.scaler.transform(row[self.features].to_frame().T.astype(float))
        d = nan_euclidean_distances(z, self.Z)[0]
        out = self.pool[['qb_draft_prospect_id', 'player_name', 'college', 'draft_year', 'round', 'overall_pick'] + OUTCOME_COLS].copy()
        out['distance'] = d
        out = out[np.isfinite(out['distance']) & (out['qb_draft_prospect_id'] != row.get('qb_draft_prospect_id'))]
        out = out.sort_values('distance').head(k).copy()
        out['closeness_index_heuristic'] = self.closeness_index(out['distance'].to_numpy())
        if as_of_year is not None:     # hide outcomes that were not yet observable at the query's draft
            hidden = out['draft_year'] + BT.LAG_FIVE_SEASON > as_of_year
            for c in ('nfl_starts_first_5_seasons', 'career_success', 'success_5yr_16', 'success_5yr_64'):
                out[c] = out[c].astype('float').where(~hidden)
        zc = self.scaler.transform(self.pool.loc[out.index, self.features])
        diff = zc - z
        sims, difs = [], []
        for r in diff:
            a = np.abs(r)
            ok = np.flatnonzero(np.isfinite(a))
            order = ok[np.argsort(a[ok])]
            sims.append(', '.join(LABELS[self.features[j]] for j in order[:3]))
            difs.append('; '.join(f'{LABELS[self.features[j]]} {"higher" if r[j] > 0 else "lower"} ({r[j]:+.1f} SD)'
                                  for j in order[::-1][:3]))
        out['most_similar_on'] = sims
        out['biggest_differences_comp_vs_query'] = difs
        return out


def comps_for(pool: pd.DataFrame, row: pd.Series, k: int = K, backtest: bool = False) -> pd.DataFrame:
    """Comps from classes strictly before the query's class; scaler fit on that pool only."""
    y = int(row['draft_year'])
    ref = pool[pool['draft_year'] < y]
    return Comparator().fit(ref).query(row, k, as_of_year=y if backtest else None)


def outcome_distribution(c: pd.DataFrame) -> dict:
    done = c[c['window_5yr_complete'] & c['nfl_starts_first_5_seasons'].notna()]
    if done.empty:
        return {'n_with_complete_outcome': 0}
    s = done['nfl_starts_first_5_seasons']
    return {'n_with_complete_outcome': int(len(done)), 'median_starts_5yr': float(s.median()),
            'share_16plus': float((s >= 16).mean()), 'share_32plus': float((s >= 32).mean()),
            'share_64plus': float((s >= 64).mean()), 'iqr_starts': [float(s.quantile(.25)), float(s.quantile(.75))]}


def backtest(pool: pd.DataFrame) -> pd.DataFrame:
    """Exploratory: does the comps' observable outcome rate predict 32+ starts? Test classes 2012-2021;
    comps only from classes whose five-season outcome was observable (<= T - 5)."""
    rows = []
    hist = pool[pool['v2_historical']]
    for ty in range(2012, 2022):
        ref = pool[(pool['draft_year'] + BT.LAG_FIVE_SEASON <= ty) & pool['v2_historical']]
        cmpr = Comparator().fit(ref)
        for _, r in hist[hist['draft_year'] == ty].iterrows():
            c = cmpr.query(r, K_OUTCOME)
            rows.append({'test_year': ty, 'qb_draft_prospect_id': r['qb_draft_prospect_id'], 'y': int(r['career_success']),
                         'knn_rate': float(c['career_success'].astype(float).mean()),
                         'knn_rate_shrunk': float((c['career_success'].astype(float).sum() + ref['career_success'].mean() * 2) / (len(c) + 2)),
                         'base_rate': float(ref['career_success'].mean()), 'mean_distance': float(c['distance'].mean())})
    return pd.DataFrame(rows)


def fig_pca(pool: pd.DataFrame, example: pd.Series, comps: pd.DataFrame):
    from src.modeling.evaluate import INK, INK_2, SURFACE, fig_ax, save
    ref = pool[pool['draft_year'] < example['draft_year']]
    sc = StandardScaler().fit(ref[SIM_FEATURES])
    Z = np.nan_to_num(sc.transform(ref[SIM_FEATURES]))          # zero = pool mean, for the 2-D picture only
    pca = PCA(n_components=2, random_state=0).fit(Z)
    P = pca.transform(Z)
    q = pca.transform(np.nan_to_num(sc.transform(example[SIM_FEATURES].to_frame().T.astype(float))))
    fig, ax = fig_ax(7.5, 5.8)
    succ = ref['career_success'].astype(float)
    ax.scatter(P[:, 0], P[:, 1], s=18, color=np.where(succ == 1, '#2a78d6', np.where(succ == 0, '#b7d3f6', '#d9d8d4')),
               edgecolor=SURFACE, linewidth=0.5)
    idx = ref.reset_index(drop=True).index[ref['qb_draft_prospect_id'].isin(comps['qb_draft_prospect_id']).to_numpy()]
    ax.scatter(P[idx, 0], P[idx, 1], s=60, facecolor='none', edgecolor='#eb6834', linewidth=2)
    for i in idx:
        ax.annotate(ref.iloc[i]['player_name'], (P[i, 0], P[i, 1]), fontsize=7.5, color=INK, xytext=(4, 3), textcoords='offset points')
    ax.scatter(q[:, 0], q[:, 1], s=120, marker='*', color='#eb6834', edgecolor=INK, linewidth=0.8)
    ax.annotate(example['player_name'], (q[0, 0], q[0, 1]), fontsize=9, color=INK, xytext=(6, -10), textcoords='offset points')
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({100 * ev[0]:.0f}% of variance)', color=INK)
    ax.set_ylabel(f'PC2 ({100 * ev[1]:.0f}%)', color=INK)
    ax.set_title(f'College-profile map of drafted QBs {int(ref.draft_year.min())}-{int(ref.draft_year.max())}: dark = 32+ starts, '
                 'light = fewer, grey = not yet known', color=INK, fontsize=9.5, loc='left')
    save(fig, 'p5_similarity_pca.png')
    return ev


def main() -> dict:
    from src.modeling import artifacts as A
    from src.modeling.evaluate import fast_auc, fast_brier
    pool = load_pool()
    RESULTS.mkdir(parents=True, exist_ok=True)
    bt = backtest(pool)
    bt.to_csv(RESULTS / 'similarity_backtest.csv', index=False)
    ev = {m: {'auc': fast_auc(bt['y'], bt[m]) if m != 'base_rate' else float('nan'), 'brier': fast_brier(bt['y'], bt[m])}
          for m in ('base_rate', 'knn_rate', 'knn_rate_shrunk')}
    # examples: the 2026 first-round QBs (current), plus two historical backtest queries
    cur = pool[pool['draft_year'] == 2026].sort_values('overall_pick').head(3)
    hist = pool[pool['player_name'].isin(['Josh Allen', 'Joe Burrow'])]
    examples = []
    for _, r in pd.concat([cur, hist]).iterrows():
        backtest_mode = r['draft_year'] <= 2021
        c = comps_for(pool, r, backtest=backtest_mode)
        c10 = comps_for(pool, r, k=K_OUTCOME, backtest=backtest_mode)
        examples.append({'query': r, 'comps': c, 'dist': outcome_distribution(c10), 'backtest': backtest_mode})
    ex0 = examples[0]
    pca_ev = fig_pca(pool, ex0['query'], ex0['comps'])
    exists = (MODEL_DIR / 'metadata.json').exists()
    if not exists:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        full = Comparator().fit(pool[pool['draft_year'] <= 2026])
        joblib.dump({'scaler': full.scaler, 'features': SIM_FEATURES, 'model_version': MODEL_VERSION}, MODEL_DIR / 'scaler.joblib')
        pool[['qb_draft_prospect_id', 'player_name', 'college', 'draft_year', 'round', 'overall_pick'] + SIM_FEATURES + OUTCOME_COLS
             + ['v2_historical']].to_parquet(DATA_DIR / 'pool.parquet', index=False)
        meta = {'model_version': MODEL_VERSION, 'features': SIM_FEATURES, 'k': K, 'distance': 'nan-Euclidean on pool z-scores',
                'pool': 'drafted QBs 2005-2026 with v2 features; queries use only classes before their own',
                'closeness_index': 'heuristic: % of pool pairwise distances larger than the pair distance; not a probability',
                'artifacts': ['scaler.joblib', 'metadata.json'], 'artifact_sha256': {'scaler.joblib': A.sha256(MODEL_DIR / 'scaler.joblib')},
                'external_data_files': [{'role': 'pool', 'path': A.rel(DATA_DIR / 'pool.parquet'), 'sha256': A.sha256(DATA_DIR / 'pool.parquet'),
                                         'regenerate': 'python -m src.modeling.prospect_similarity',
                                         'reason': 'player-level CFBD/nflverse-derived data; kept out of Git'}]}
        (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    REPORT.write_text(render(examples, ev, bt, pool, pca_ev), encoding='utf-8')
    print(ev)
    return {'eval': ev, 'n_pool': len(pool)}


def render(examples, ev, bt, pool, pca_ev) -> str:
    L = ['# Historical QB prospect comparisons', '',
         'Generated by `python -m src.modeling.prospect_similarity`. For a college QB, the engine finds the most similar '
         'previously drafted QBs by **college profile only**. NFL outcomes never enter the distance; they are shown '
         'afterwards, for context.', '',
         '## Method', '',
         f'* **Pool:** {len(pool)} drafted QBs from 2005-2026 with the v2 college features. A query is compared only with QBs '
         'drafted **before** its own class.',
         f'* **Profile ({len(SIM_FEATURES)} features):** ' + ', '.join(LABELS[f] for f in SIM_FEATURES) + '.',
         '* **Distance:** Euclidean distance on z-scores. The scaler is fit on the query\'s pool only. Missing values are '
         'skipped and the distance rescaled (nan-Euclidean), so no future-informed imputation is involved. All features '
         'have equal weight. This is a modelling choice, not a learned metric.',
         '* **Closeness index (0-100):** the share of all pairwise distances between pool QBs that are larger than this '
         'pair\'s distance. It is a **heuristic ranking aid, not a probability** or a calibrated similarity percentage.',
         '* **Backtests:** comps\' outcomes are shown only if they were observable at the query\'s draft (comp class <= '
         'query class - 5). Outcomes not yet known at the time show as blank.', '',
         '## Do comparable players\' outcomes predict NFL success? (exploratory)', '',
         'Test classes 2012-2021 (107 QBs). Prediction = the share of the 10 nearest comps (from classes with observable '
         'outcomes) who reached 32 starts.', '',
         '| Predictor | ROC-AUC | Brier |', '|---|---|---|',
         f'| Historical base rate | n/a | {ev["base_rate"]["brier"]:.3f} |',
         f'| 10-nearest-comp success rate | {ev["knn_rate"]["auc"]:.3f} | {ev["knn_rate"]["brier"]:.3f} |',
         f'| Same, shrunk toward the base rate (2 pseudo-comps) | {ev["knn_rate_shrunk"]["auc"]:.3f} | {ev["knn_rate_shrunk"]["brier"]:.3f} |', '',
         'For comparison, the pre-draft logistic model on the same test QBs (`career_production_evaluation.md`) is stronger. '
         'Comps are a way to explain a profile, not a better forecaster. Their outcome spread is shown as a descriptive '
         'distribution, never as a probability.', '',
         '![PCA map](figures/p5_similarity_pca.png)', '',
         f'The 2-D map is a PCA of the standardised profiles (PC1 {100 * pca_ev[0]:.0f}%, PC2 {100 * pca_ev[1]:.0f}% of '
         'variance), fit on the pool before the highlighted query. Missing values are set to the pool mean for the picture '
         'only. Distances in the map are approximate; the table distances use all features.', '',
         '## Examples', '']
    for e in examples:
        q, c, dist = e['query'], e['comps'], e['dist']
        L += [f'### {q["player_name"]} ({q["college"]}, drafted {int(q["draft_year"])}, round {int(q["round"])}, pick {int(q["overall_pick"])})', '',
              ('*Historical backtest: comps are from classes before ' + str(int(q['draft_year'])) + '. Outcomes are shown only for '
               'comps whose five seasons had been played by then.*' if e['backtest'] else
               '*Current query: comps are from 2005-' + str(int(q['draft_year']) - 1) + '. Outcomes shown where the five-season '
               'window is complete; otherwise starts to date.*'), '',
              '| Comp | College | Drafted | Distance | Closeness index (heuristic) | Similar on | Biggest differences (comp vs query) | NFL starts, seasons 1-5 |',
              '|---|---|---|---|---|---|---|---|']
        for _, r in c.iterrows():
            if pd.notna(r['nfl_starts_first_5_seasons']):
                out = f'{int(r["nfl_starts_first_5_seasons"])}'
            elif not e['backtest'] and pd.notna(r['nfl_starts_first_5_seasons_to_date']):
                out = f'{int(r["nfl_starts_first_5_seasons_to_date"])} so far (window incomplete)'
            else:
                out = 'not yet observable'
            L.append(f'| {r["player_name"]} | {r["college"]} | {int(r["draft_year"])} R{int(r["round"])} #{int(r["overall_pick"])} | '
                     f'{r["distance"]:.2f} | {r["closeness_index_heuristic"]:.0f} | {r["most_similar_on"]} | '
                     f'{r["biggest_differences_comp_vs_query"]} | {out} |')
        if dist.get('n_with_complete_outcome'):
            L += ['', f'Among the 10 nearest comps with complete outcomes ({dist["n_with_complete_outcome"]}): median '
                  f'{dist["median_starts_5yr"]:.0f} starts in seasons 1-5 (IQR {dist["iqr_starts"][0]:.0f}-{dist["iqr_starts"][1]:.0f}); '
                  f'{100 * dist["share_16plus"]:.0f}% reached 16+, {100 * dist["share_32plus"]:.0f}% reached 32+, '
                  f'{100 * dist["share_64plus"]:.0f}% reached 64+. This describes the comps; it is not a probability for the query.', '']
        else:
            L += ['', 'None of the 10 nearest comps had an observable five-season outcome at this date.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    main()
