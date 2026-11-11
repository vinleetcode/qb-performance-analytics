"""Five-season NFL production forecasts for drafted QBs (Phase 5).

Run:  python -m src.modeling.career_trajectory && python -m src.modeling.career_production

POPULATION: the 183 v2 historical-eligible drafted QBs (classes 2005-2021, complete,
verified five-season windows; never-played QBs are real zeros).
FEATURES: pre-draft = the v2 selected 11 college/context features (raw, robust definitions);
post-draft = pre-draft + log(overall pick). Drafting team is NOT used: 32 teams for 183 QBs
(about 6 per team, spread over 17 years of changing staffs) cannot support team effects.
TARGETS (seasons 1-5 after the draft, regular season): games played/started, pass attempts,
completions, yards, TDs, INTs, rush attempts, yards, TDs; derived efficiency where the
denominators allow (>= 300 pass attempts, >= 50 rushes); P(16+/32+/64+ starts).

MODELS
  mean / round_mean     historical baselines (round_mean uses draft round: post-draft only)
  ridge                 L2 linear regression on the raw total (clipped at 0)
  poisson               quasi-Poisson GLM (sklearn PoissonRegressor). Only the MEAN model is Poisson;
                        variance/mean ratios are in the hundreds (overdispersed), so no Poisson-based
                        intervals are used -- intervals come from out-of-sample residuals
  gbdt                  shallow gradient boosting with Poisson loss (exploratory)
  hurdle                two-stage: P(meaningful opportunity) x E[total | opportunity] + (1-P) x E[total | none]
  structured            simulation: P(appear), P(meaningful | appear); for meaningful careers, models for
                        log attempts and starts plus completion %, YPA, TD and INT rates, rush attempts per
                        pass attempt, rush YPC and rush TD rate. Residuals are drawn JOINTLY from one historical
                        "donor" QB, which keeps the empirical dependence between volume and efficiency, and the
                        donor's own year-by-year shape spreads the totals over seasons 1-5 (no common growth
                        curve). Counts are drawn as integers (binomial / Poisson), so completions <= attempts,
                        TDs are non-negative integers, starts <= games played <= the season cap, five-year totals
                        equal the sum of the seasons, and a zero-opportunity draw has zero production.
VALIDATION: expanding-window backtest by draft class, test classes 2012-2021, five-season label
embargo (a class is used for training only once its five seasons had been played). The 2019-2021
classes have been examined in earlier phases, so these results are labelled exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.modeling import backtesting as BT
from src.modeling import build_dataset_v2 as B2
from src.modeling import career_trajectory as CT
from src.modeling import train_v2 as T2
from src.modeling import uncertainty as U

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results' / 'phase5'
MODEL_VERSION = '1.0.0'
MODEL_DIR = ROOT / 'models' / 'qb_career_production' / f'v{MODEL_VERSION}'
DATA_DIR = ROOT / 'data' / 'model_artifacts' / 'qb_career_production' / f'v{MODEL_VERSION}'
SEED = 20261110
TEST_YEARS = list(range(2012, 2022))
N_SIMS = 1000
MEANINGFUL_ATTEMPTS = 100
YEARS = 5
SEASON_CAP = 17
INTERVAL = (0.10, 0.90)
THRESHOLDS = (16, 32, 64)
TARGETS = ['games_played', 'games_started', 'pass_attempts', 'pass_completions', 'pass_yards', 'pass_tds',
           'interceptions', 'rush_attempts', 'rush_yards', 'rush_tds']
EFFICIENCY = {'completion_pct': ('pass_completions', 'pass_attempts', 300, 100), 'yards_per_attempt': ('pass_yards', 'pass_attempts', 300, 1),
              'td_rate': ('pass_tds', 'pass_attempts', 300, 100), 'int_rate': ('interceptions', 'pass_attempts', 300, 100),
              'rush_ypc': ('rush_yards', 'rush_attempts', 50, 1)}
RATE_SPECS = [('cmp', 'pass_completions', 'pass_attempts', (0.35, 0.75)), ('ypa', 'pass_yards', 'pass_attempts', (3.0, 10.0)),
              ('td', 'pass_tds', 'pass_attempts', (0.0, 0.10)), ('int', 'interceptions', 'pass_attempts', (0.0, 0.08)),
              ('rush_ratio', 'rush_attempts', 'pass_attempts', (0.0, 0.6)), ('rypc', 'rush_yards', 'rush_attempts', (-2.0, 9.0)),
              ('rtd', 'rush_tds', 'rush_attempts', (0.0, 0.2))]


def feature_sets() -> dict:
    manifest = json.loads((B2.OUT_DIR / f'dataset_manifest_v{B2.DATASET_VERSION}.json').read_text(encoding='utf-8'))
    pre = B2.feature_sets(manifest['combo_choice'])[('raw', 'context')]
    return {'pre_draft': pre, 'post_draft': pre + ['log_overall_pick']}


def pipe(est) -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', est)])


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = T2.load()
    d = d[d['v2_historical']].copy()
    seasons = pd.read_parquet(CT.OUT)
    tot = CT.five_year_totals(seasons)
    d = d.merge(tot, on='qb_draft_prospect_id', how='inner', validate='one_to_one')
    d['round_group'] = d['round'].map(CT.round_group)
    paths = seasons[(seasons['season_index'] <= YEARS) & seasons['qb_draft_prospect_id'].isin(d['qb_draft_prospect_id'])]
    return d.reset_index(drop=True), paths.sort_values(['qb_draft_prospect_id', 'season_index'])


def path_array(paths: pd.DataFrame, ids) -> np.ndarray:
    """(n_qbs, 5, n_stats) array of actual yearly lines, in TARGETS order."""
    p = paths.set_index(['qb_draft_prospect_id', 'season_index'])[TARGETS]
    out = np.zeros((len(ids), YEARS, len(TARGETS)))
    for i, pid in enumerate(ids):
        out[i] = p.loc[pid].reindex(range(1, YEARS + 1)).fillna(0).to_numpy()
    return out


# ------------------------------------------------------------------ helpers
def _logit(X, y, C=0.5):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return ('const', float(y.mean()))
    return ('model', pipe(LogisticRegression(C=C, max_iter=5000)).fit(X, y))


def _prob(m, X):
    return np.full(len(X), m[1]) if m[0] == 'const' else m[1].predict_proba(X)[:, 1]


def _oof_residuals(make, X, y, groups, weights=None, n_splits=5):
    """Out-of-fold residuals inside the TRAINING data (grouped by draft class)."""
    y = np.asarray(y, dtype=float)
    res = np.full(len(y), np.nan)
    k = min(n_splits, len(np.unique(groups)))
    if k < 2:
        return res
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        m = make()
        kw = {'model__sample_weight': weights[tr]} if weights is not None else {}
        m.fit(X.iloc[tr], y[tr], **kw)
        res[te] = y[te] - m.predict(X.iloc[te])
    return res


# ------------------------------------------------------------------ structured simulation model
class StructuredModel:
    """Opportunity -> volume -> efficiency simulation (see module docstring)."""

    def __init__(self, cols, seed=SEED):
        self.cols, self.seed = cols, seed

    def fit(self, train: pd.DataFrame, paths: pd.DataFrame):
        X = train[self.cols]
        appeared = (train['games_played_5yr'] > 0).to_numpy()
        meaningful = (train['pass_attempts_5yr'] >= MEANINGFUL_ATTEMPTS).to_numpy()
        self.m_appear = _logit(X, appeared)
        self.m_meaningful = _logit(X[appeared], meaningful[appeared]) if appeared.sum() > 1 else ('const', 0.0)
        ids = train['qb_draft_prospect_id'].to_numpy()
        arr = path_array(paths, ids)
        self.limited_paths = arr[appeared & ~meaningful]
        self.none_paths = arr[~appeared]
        mt = train[meaningful].reset_index(drop=True)
        self.meaningful_paths = arr[meaningful]
        Xm, g = mt[self.cols], mt['draft_year'].to_numpy()
        ridge = lambda: pipe(Ridge(alpha=10.0))  # noqa: E731
        self.m_att = ridge().fit(Xm, np.log(mt['pass_attempts_5yr']))
        self.e_att = _oof_residuals(ridge, Xm, np.log(mt['pass_attempts_5yr']), g)
        self.m_st = ridge().fit(Xm, np.log1p(mt['games_started_5yr']))
        self.e_st = _oof_residuals(ridge, Xm, np.log1p(mt['games_started_5yr']), g)
        self.rate_models, self.rate_res, self.rate_fallback = {}, {}, {}
        for name, num, den, (lo, hi) in RATE_SPECS:
            n_, d_ = mt[f'{num}_5yr'].to_numpy(float), mt[f'{den}_5yr'].to_numpy(float)
            ok = d_ >= (10 if den == 'rush_attempts' else 1)
            r = np.where(ok, n_ / np.where(d_ > 0, d_, 1), np.nan)
            self.rate_fallback[name] = float(np.nansum(n_[ok]) / np.nansum(d_[ok])) if ok.any() else 0.0
            r = np.where(ok, r, self.rate_fallback[name])
            w = np.where(ok, d_, 1.0)
            m = ridge().fit(Xm, r, model__sample_weight=w)
            self.rate_models[name] = m
            res = _oof_residuals(ridge, Xm, r, g, weights=w)
            self.rate_res[name] = np.where(np.isfinite(res), res, 0.0)
        self.e_att = np.where(np.isfinite(self.e_att), self.e_att, 0.0)
        self.e_st = np.where(np.isfinite(self.e_st), self.e_st, 0.0)
        self.n_meaningful = int(meaningful.sum())
        return self

    def simulate(self, X: pd.DataFrame, n_sims: int = N_SIMS, seed: int | None = None) -> np.ndarray:
        """(n_qbs, n_sims, 5, n_stats) simulated yearly lines."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        pa = _prob(self.m_appear, X)
        pm = _prob(self.m_meaningful, X)
        mu_att, mu_st = self.m_att.predict(X), self.m_st.predict(X)
        mu_r = {k: m.predict(X) for k, m in self.rate_models.items()}
        out = np.zeros((len(X), n_sims, YEARS, len(TARGETS)))
        ti = {t: i for i, t in enumerate(TARGETS)}
        nm = self.n_meaningful
        for i in range(len(X)):
            u = rng.random(n_sims)
            state = np.where(u < 1 - pa[i], 0, np.where(u < 1 - pa[i] + pa[i] * (1 - pm[i]), 1, 2))
            s1 = np.flatnonzero(state == 1)
            if len(s1) and len(self.limited_paths):
                out[i, s1] = self.limited_paths[rng.integers(0, len(self.limited_paths), len(s1))]
            s2 = np.flatnonzero(state == 2)
            if not len(s2) or nm == 0:
                continue
            j = rng.integers(0, nm, len(s2))                       # one donor per draw: joint residuals
            donor = self.meaningful_paths[j]                        # (k, 5, stats)
            att_tot = np.maximum(np.rint(np.exp(mu_att[i] + self.e_att[j])), 1).astype(int)
            st_tot = np.maximum(np.rint(np.expm1(mu_st[i] + self.e_st[j])), 0).astype(int)
            rates = {}
            for name, _, _, (lo, hi) in RATE_SPECS:
                rates[name] = np.clip(mu_r[name][i] + self.rate_res[name][j], lo, hi)
            d_att = donor[:, :, ti['pass_attempts']]
            sh = np.where(d_att.sum(1, keepdims=True) > 0, d_att / np.maximum(d_att.sum(1, keepdims=True), 1), 1 / YEARS)
            att_y = np.array([rng.multinomial(a, s / s.sum()) for a, s in zip(att_tot, sh)])
            d_st = donor[:, :, ti['games_started']]
            ssh = np.where(d_st.sum(1, keepdims=True) > 0, d_st / np.maximum(d_st.sum(1, keepdims=True), 1), sh)
            st_y = np.minimum(np.array([rng.multinomial(a, s / s.sum()) for a, s in zip(st_tot, ssh)]), SEASON_CAP)
            nonstart = np.maximum(donor[:, :, ti['games_played']] - d_st, 0)
            gp_y = np.minimum(st_y + nonstart, SEASON_CAP)
            gp_y = np.where((att_y > 0) & (gp_y == 0), 1, gp_y)
            cmp_y = rng.binomial(att_y, rates['cmp'][:, None])
            td_y = rng.binomial(att_y, rates['td'][:, None])
            int_y = rng.binomial(att_y, rates['int'][:, None])
            yds_y = np.rint(att_y * rates['ypa'][:, None])
            ra_y = rng.poisson(att_y * rates['rush_ratio'][:, None])
            ry_y = np.rint(ra_y * rates['rypc'][:, None])
            rtd_y = rng.binomial(ra_y, rates['rtd'][:, None])
            block = np.stack([gp_y, st_y, att_y, cmp_y, yds_y, td_y, int_y, ra_y, ry_y, rtd_y], axis=-1)
            out[i, s2] = block
        return out

    @staticmethod
    def summarize(sims: np.ndarray) -> dict:
        tot = sims.sum(axis=2)                                    # (n, sims, stats) five-year totals
        q = np.quantile(tot, INTERVAL, axis=1)
        summ = {'mean': tot.mean(axis=1), 'median': np.median(tot, axis=1), 'lo': q[0], 'hi': q[1],
                'p_starts': {k: (tot[:, :, TARGETS.index('games_started')] >= k).mean(1) for k in THRESHOLDS},
                'year_mean': sims.mean(axis=1), 'year_lo': np.quantile(sims, INTERVAL[0], axis=1),
                'year_hi': np.quantile(sims, INTERVAL[1], axis=1)}
        return summ


# ------------------------------------------------------------------ direct models
def direct_models(train: pd.DataFrame, cols, target: str):
    X, y = train[cols], train[f'{target}_5yr'].to_numpy(float)
    yp = np.maximum(y, 0)
    g = train['draft_year'].to_numpy()
    out = {}
    makers = {'ridge': lambda: pipe(Ridge(alpha=10.0)),
              'poisson': lambda: pipe(PoissonRegressor(alpha=1.0, max_iter=3000)),
              'gbdt': lambda: Pipeline([('impute', SimpleImputer(strategy='median')),
                                        ('model', HistGradientBoostingRegressor(loss='poisson', max_depth=2, learning_rate=0.05,
                                                                                max_iter=150, min_samples_leaf=10,
                                                                                random_state=SEED))])}
    for name, mk in makers.items():
        yy = y if name == 'ridge' else yp
        m = mk().fit(X, yy)
        res = _oof_residuals(mk, X, yy, g) if name != 'gbdt' else None
        out[name] = (m, res)
    meaningful = (train['pass_attempts_5yr'] >= MEANINGFUL_ATTEMPTS).to_numpy()
    hm = _logit(X, meaningful)
    hp = pipe(PoissonRegressor(alpha=1.0, max_iter=3000)).fit(X[meaningful], yp[meaningful]) if meaningful.sum() > 3 else None
    none_mean = float(y[~meaningful].mean()) if (~meaningful).any() else 0.0
    out['hurdle'] = (hm, hp, none_mean)
    return out


def predict_direct(models, X):
    p = {k: np.maximum(models[k][0].predict(X), 0) for k in ('ridge', 'poisson', 'gbdt')}
    hm, hp, nm = models['hurdle']
    prob = _prob(hm, X)
    cond = hp.predict(X) if hp is not None else np.zeros(len(X))
    p['hurdle'] = prob * cond + (1 - prob) * nm
    return p


# ------------------------------------------------------------------ backtest
def backtest(d: pd.DataFrame, paths: pd.DataFrame, fsets: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, prob_rows, year_rows = [], [], []
    for ty, tr, te in BT.folds(d['draft_year'], TEST_YEARS, BT.LAG_FIVE_SEASON):
        train, test = d.iloc[tr], d.iloc[te]
        base_round = train.groupby('round_group')
        for fs, cols in fsets.items():
            sm = StructuredModel(cols).fit(train, paths)
            summ = sm.summarize(sm.simulate(test[cols], seed=SEED + ty))
            for target in TARGETS:
                ti = TARGETS.index(target)
                y = test[f'{target}_5yr'].to_numpy(float)
                dm = direct_models(train, cols, target)
                pr = predict_direct(dm, test[cols])
                mean_b = float(train[f'{target}_5yr'].mean())
                rm = base_round[f'{target}_5yr'].mean()
                preds = {'mean': np.full(len(test), mean_b),
                         'median': np.full(len(test), float(train[f'{target}_5yr'].median())), **pr,
                         'structured': summ['mean'][:, ti], 'structured_median': summ['median'][:, ti]}
                if fs == 'post_draft':
                    preds['round_mean'] = test['round_group'].map(rm).fillna(mean_b).to_numpy()
                ints = {'structured': (summ['lo'][:, ti], summ['hi'][:, ti])}
                for k in ('ridge', 'poisson'):
                    ints[k] = U.empirical_interval(pr[k], dm[k][1], INTERVAL, lower_bound=0 if target != 'rush_yards' else None)
                for k, p in preds.items():
                    lo, hi = ints.get(k, (np.full(len(y), np.nan),) * 2)
                    rows.append(pd.DataFrame({'feature_set': fs, 'target': target, 'model': k, 'test_year': ty,
                                              'qb_draft_prospect_id': test['qb_draft_prospect_id'].to_numpy(),
                                              'player_name': test['player_name'].to_numpy(),
                                              'round_group': test['round_group'].to_numpy(), 'y': y, 'pred': p,
                                              'lo': lo, 'hi': hi, 'n_train': len(train)}))
            starts = test['games_started_5yr'].to_numpy()
            for k in THRESHOLDS:
                yk = (starts >= k).astype(int)
                lg = _logit(train[cols], train['games_started_5yr'] >= k)
                for name, p in (('base_rate', np.full(len(test), float((train['games_started_5yr'] >= k).mean()))),
                                ('logit', _prob(lg, test[cols])), ('structured', summ['p_starts'][k])):
                    prob_rows.append(pd.DataFrame({'feature_set': fs, 'threshold': k, 'model': name, 'test_year': ty,
                                                   'qb_draft_prospect_id': test['qb_draft_prospect_id'].to_numpy(),
                                                   'y': yk, 'p': p}))
            actual_y = path_array(paths, test['qb_draft_prospect_id'].to_numpy())
            tp = path_array(paths, train['qb_draft_prospect_id'].to_numpy()).mean(axis=0)
            for yi in range(YEARS):
                for stat in ('games_started', 'pass_attempts', 'pass_yards', 'rush_yards'):
                    si = TARGETS.index(stat)
                    year_rows.append(pd.DataFrame({'feature_set': fs, 'season_index': yi + 1, 'stat': stat, 'test_year': ty,
                                                   'qb_draft_prospect_id': test['qb_draft_prospect_id'].to_numpy(),
                                                   'y': actual_y[:, yi, si], 'structured': summ['year_mean'][:, yi, si],
                                                   'lo': summ['year_lo'][:, yi, si], 'hi': summ['year_hi'][:, yi, si],
                                                   'year_mean_baseline': np.full(len(test), tp[yi, si])}))
    return pd.concat(rows, ignore_index=True), pd.concat(prob_rows, ignore_index=True), pd.concat(year_rows, ignore_index=True)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def evaluate(pred: pd.DataFrame, probs: pd.DataFrame, years: pd.DataFrame, d: pd.DataFrame) -> dict:
    tab = []
    for (fs, tgt, m), g in pred.groupby(['feature_set', 'target', 'model'], sort=False):
        tab.append({'feature_set': fs, 'target': tgt, 'model': m, 'n': len(g), 'mae': mae(g['y'], g['pred']),
                    'rmse': rmse(g['y'], g['pred']), 'bias': float((g['pred'] - g['y']).mean()),
                    'coverage80': U.coverage(g['y'], g['lo'], g['hi']) if g['lo'].notna().any() else np.nan,
                    'mean_width80': float((g['hi'] - g['lo']).mean()) if g['lo'].notna().any() else np.nan})
    T = pd.DataFrame(tab)
    base = T[T['model'] == 'mean'].set_index(['feature_set', 'target'])['mae']
    T['mae_skill_vs_mean'] = 1 - T['mae'] / [base.loc[(a, b)] for a, b in zip(T['feature_set'], T['target'])]
    med = T[T['model'] == 'median'].set_index(['feature_set', 'target'])['mae']
    T['mae_skill_vs_median'] = 1 - T['mae'] / [med.loc[(a, b)] for a, b in zip(T['feature_set'], T['target'])]
    # cluster bootstrap (by draft class) of MAE difference vs the historical mean
    cb = []
    for fs in ('pre_draft', 'post_draft'):
        for tgt in TARGETS:
            w = pred[(pred['feature_set'] == fs) & (pred['target'] == tgt)].pivot_table(
                index=['qb_draft_prospect_id', 'test_year', 'y'], columns='model', values='pred').reset_index()
            w = w.rename(columns={'test_year': 'draft_year'})
            for m in ('structured', 'structured_median', 'poisson', 'hurdle', 'ridge') + (('round_mean',) if fs == 'post_draft' else ()):
                for base_name in ('mean', 'median'):
                    r = U.paired_cluster_bootstrap(mae, w, base_name, m, y_col='y', n_boot=1000, seed=SEED)
                    cb.append({'feature_set': fs, 'target': tgt, 'model': m, 'baseline': base_name,
                               'mae_diff': r['estimate'], 'lo': r['lower'], 'hi': r['upper'], 'p_lower_mae': r['p_b_lower']})
    CB = pd.DataFrame(cb)
    # pre vs post (structured)
    pp = []
    for tgt in TARGETS:
        a = pred[(pred['feature_set'] == 'pre_draft') & (pred['target'] == tgt) & (pred['model'] == 'structured')]
        b = pred[(pred['feature_set'] == 'post_draft') & (pred['target'] == tgt) & (pred['model'] == 'structured')]
        w = a[['qb_draft_prospect_id', 'test_year', 'y', 'pred']].merge(
            b[['qb_draft_prospect_id', 'pred']], on='qb_draft_prospect_id', suffixes=('_pre', '_post')).rename(columns={'test_year': 'draft_year'})
        r = U.paired_cluster_bootstrap(mae, w, 'pred_pre', 'pred_post', y_col='y', n_boot=1000, seed=SEED)
        pp.append({'target': tgt, 'mae_pre': mae(w['y'], w['pred_pre']), 'mae_post': mae(w['y'], w['pred_post']),
                   'diff': r['estimate'], 'lo': r['lower'], 'hi': r['upper'], 'p_post_better': r['p_b_lower']})
    PP = pd.DataFrame(pp)
    # probabilities
    pt = []
    for (fs, k, m), g in probs.groupby(['feature_set', 'threshold', 'model'], sort=False):
        from src.modeling.evaluate import fast_auc, fast_brier, fast_log_loss
        c = U.calibration_summary(g['y'], g['p']) if g['y'].nunique() == 2 else {}
        bci = U.cluster_bootstrap(lambda x: fast_brier(x['y'], x['p']), g.rename(columns={'test_year': 'draft_year'}),
                                  n_boot=1000, seed=SEED)
        pt.append({'feature_set': fs, 'threshold': k, 'model': m, 'n': len(g), 'n_pos': int(g['y'].sum()),
                   'auc': fast_auc(g['y'], g['p']), 'brier': fast_brier(g['y'], g['p']), 'brier_lo': bci['lower'],
                   'brier_hi': bci['upper'], 'log_loss': fast_log_loss(g['y'], np.clip(g['p'], 1e-4, 1 - 1e-4)),
                   'calibration_slope': c.get('calibration_slope', np.nan),
                   'calibration_in_the_large': c.get('calibration_in_the_large', np.nan)})
    PT = pd.DataFrame(pt)
    # year-by-year
    yt = []
    for (fs, k, stat), g in years.groupby(['feature_set', 'season_index', 'stat']):
        yt.append({'feature_set': fs, 'season_index': k, 'stat': stat, 'n': len(g), 'mae_structured': mae(g['y'], g['structured']),
                   'mae_baseline': mae(g['y'], g['year_mean_baseline']), 'coverage80': U.coverage(g['y'], g['lo'], g['hi'])})
    YT = pd.DataFrame(yt)
    # bias by round group and by class (structured and poisson)
    bias = (pred[pred['model'].isin(['structured', 'poisson', 'mean'])]
            .assign(err=lambda x: x['pred'] - x['y'])
            .groupby(['feature_set', 'target', 'model', 'round_group'])['err'].mean().reset_index())
    bias_class = (pred[pred['model'].isin(['structured'])].assign(err=lambda x: x['pred'] - x['y'])
                  .groupby(['feature_set', 'target', 'test_year'])['err'].mean().reset_index())
    # efficiency on QBs with sufficient denominators (structured implied rates vs mean baseline)
    eff = []
    for fs in ('pre_draft', 'post_draft'):
        s = pred[(pred['feature_set'] == fs) & (pred['model'].isin(['structured', 'mean']))]
        w = s.pivot_table(index=['qb_draft_prospect_id', 'model'], columns='target', values='pred').reset_index()
        a = s.pivot_table(index='qb_draft_prospect_id', columns='target', values='y')
        for name, (num, den, min_den, scale) in EFFICIENCY.items():
            ok = a[den] >= min_den
            act = (a.loc[ok, num] / a.loc[ok, den]) * scale
            for m in ('structured', 'mean'):
                pw = w[w['model'] == m].set_index('qb_draft_prospect_id').loc[act.index]
                prd = (pw[num] / pw[den].where(pw[den] > 0)) * scale
                eff.append({'feature_set': fs, 'metric': name, 'model': m, 'n': int(ok.sum()), 'mae': mae(act, prd.fillna(prd.mean()))})
    EF = pd.DataFrame(eff)
    od = pd.DataFrame([{'target': t, 'mean': d[f'{t}_5yr'].mean(), 'variance': d[f'{t}_5yr'].var(),
                        'var_to_mean': d[f'{t}_5yr'].var() / max(d[f'{t}_5yr'].mean(), 1e-9),
                        'share_zero': float((d[f'{t}_5yr'] == 0).mean())} for t in TARGETS])
    return {'metrics': T, 'cluster': CB, 'pre_post': PP, 'probs': PT, 'years': YT, 'bias_round': bias,
            'bias_class': bias_class, 'efficiency': EF, 'overdispersion': od}


def consistency_checks(sims: np.ndarray) -> dict:
    ti = {t: i for i, t in enumerate(TARGETS)}
    s = sims
    zero_opp = (s[..., ti['games_played']].sum(-1) == 0)
    return {
        'completions_le_attempts': bool((s[..., ti['pass_completions']] <= s[..., ti['pass_attempts']]).all()),
        'td_nonneg_integer': bool(((s[..., ti['pass_tds']] >= 0) & (s[..., ti['pass_tds']] == np.rint(s[..., ti['pass_tds']]))).all()
                                  and ((s[..., ti['rush_tds']] >= 0) & (s[..., ti['rush_tds']] == np.rint(s[..., ti['rush_tds']]))).all()),
        'starts_le_games_played': bool((s[..., ti['games_started']] <= s[..., ti['games_played']]).all()),
        'games_le_cap': bool((s[..., ti['games_played']] <= SEASON_CAP).all()),
        'zero_opportunity_zero_production': bool((np.abs(s[zero_opp]).sum() == 0) if zero_opp.any() else True),
        'interceptions_le_attempts': bool((s[..., ti['interceptions']] <= s[..., ti['pass_attempts']]).all()),
    }


def fit_final(d: pd.DataFrame, paths: pd.DataFrame, fsets: dict) -> dict:
    """Final models on all 183 QBs (classes 2005-2021, all labels observable now)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    for fs, cols in fsets.items():
        sm = StructuredModel(cols).fit(d, paths)
        thr = {k: _logit(d[cols], d['games_started_5yr'] >= k) for k in THRESHOLDS}
        direct = {t: direct_models(d, cols, t) for t in TARGETS}
        art = {'structured': sm, 'threshold_logits': thr, 'direct': direct, 'features': cols, 'feature_set': fs,
               'model_version': MODEL_VERSION, 'targets': TARGETS}
        joblib.dump(art, MODEL_DIR / f'{fs}.joblib')
        saved[fs] = f'{fs}.joblib'
    return saved


def main() -> dict:
    from src.modeling import artifacts as A
    from src.modeling.train import package_versions
    d, paths = load_data()
    fsets = feature_sets()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pred, probs, years = backtest(d, paths, fsets)
    pred.to_csv(RESULTS / 'production_backtest_predictions.csv', index=False)
    probs.to_csv(RESULTS / 'production_backtest_probabilities.csv', index=False)
    years.to_csv(RESULTS / 'production_backtest_years.csv', index=False)
    ev = evaluate(pred, probs, years, d)
    for k, v in ev.items():
        v.to_csv(RESULTS / f'production_{k}.csv', index=False)
    sm = StructuredModel(fsets['pre_draft']).fit(d, paths)
    cons = consistency_checks(sm.simulate(d[fsets['pre_draft']].head(40), n_sims=500))
    exists = (MODEL_DIR / 'metadata.json').exists()
    saved = {} if exists else fit_final(d, paths, fsets)
    if not exists:
        paths.to_parquet(DATA_DIR / 'training_paths.parquet', index=False)
        meta = {'model_version': MODEL_VERSION, 'random_seed': SEED, 'feature_sets': fsets, 'targets': TARGETS,
                'training_draft_classes': sorted(d['draft_year'].unique().tolist()), 'training_rows': int(len(d)),
                'backtest': {'test_classes': TEST_YEARS, 'label_embargo_years': BT.LAG_FIVE_SEASON},
                'structured_model': {'meaningful_attempts': MEANINGFUL_ATTEMPTS, 'n_sims': N_SIMS, 'interval': INTERVAL,
                                     'season_cap': SEASON_CAP},
                'drafting_team': 'not used: about 6 QBs per team over 17 seasons cannot support team effects',
                'artifacts': sorted(saved.values()) + ['metadata.json'],
                'artifact_sha256': {v: A.sha256(MODEL_DIR / v) for v in saved.values()},
                'external_data_files': [{'role': 'training_paths', 'path': A.rel(DATA_DIR / 'training_paths.parquet'),
                                         'sha256': A.sha256(DATA_DIR / 'training_paths.parquet'),
                                         'regenerate': 'python -m src.modeling.career_trajectory',
                                         'reason': 'player-level nflverse-derived seasons; kept out of Git (the structured '
                                                   'model objects also embed training career paths, see report)'}],
                'consistency_checks': cons, 'package_versions': package_versions()}
        (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    (RESULTS / 'production_consistency.json').write_text(json.dumps(cons, indent=2), encoding='utf-8')
    M = ev['metrics']
    print(M[(M['target'].isin(['games_started', 'pass_yards', 'pass_tds', 'rush_yards']))].pivot_table(
        index=['feature_set', 'target'], columns='model', values='mae').round(1).to_string())
    print(cons)
    return ev


if __name__ == '__main__':
    # run through the package module so pickled classes resolve as src.modeling.career_production, not __main__
    from src.modeling import career_production as _m
    _m.main()
