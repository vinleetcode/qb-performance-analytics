"""NFL Draft projection for college QBs (Phase 5).

Run:  python -m src.modeling.draft_predictor

THE PROSPECT UNIVERSE (who could have been drafted). The draft tables list only drafted
players, so "everyone not in the draft table" is NOT a valid set of undrafted prospects
(it would include QBs who returned to school, transferred down, or quit). CFBD has no
"declared for the draft" record. This module therefore builds a population whose draft
status can be verified from local data:

  1. a college passer-season with >= 100 attempts at an FBS team (CFBD),
  2. that is the athlete's LAST season anywhere in the local CFBD data (season stats of any
     kind, passing or rushing box scores), checked against at least one later season that
     is in the data (so final seasons up to 2024; 2025 cannot be verified yet),
  3. whose high-school recruiting class Y is known (CFBD recruiting record linked by athlete id),
     with years since high school s - Y + 1 >= 4. Every such QB is NFL-draft eligible (three
     years removed from high school), and he did not continue in college.
  Drafted = selected in the very next draft (any position), by CFBD athlete id.

  Excluded and reported: no recruiting link (walk-ons, JUCO, and most pre-2011 seasons: the
  athlete-id link is sparse early), true juniors and younger (s - Y + 1 <= 3: the only way to
  leave is to declare, and undrafted early entrants are not identifiable), final 2025 seasons.
  Remaining risk: a fourth/fifth-year QB who transferred to a level CFBD did not cover (FCS
  before 2022) or simply stopped playing would be labelled "undrafted"; and recruit linkage
  may differ by prospect quality. The probability is therefore P(drafted | an FBS QB's college
  career verifiably ended after >= 4 years), not P(drafted) for every college QB.

ROUND / PICK (conditional on being drafted): all drafted QBs 2005-2026 with a final college
passer-season in the data. Round probabilities: multinomial and ordinal (cumulative-logit
style) logistic models; both sum to 1 by construction. Overall pick: ridge regression on
log(pick) with prediction intervals from out-of-sample residuals; coverage is measured.

Inputs are pre-draft only (final college season, earlier seasons, recruiting, roster size,
team context, NFL Combine in late February/March). Actual draft position is never an input.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingest_cfbd_advanced import id_str
from src.match_prospects import normalize_name, normalize_school
from src.modeling import backtesting as BT
from src.modeling import era
from src.modeling import uncertainty as U

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / 'data' / 'qb_history.db'
RESULTS = ROOT / 'results' / 'phase5'
REPORT = ROOT / 'reports' / 'draft_model_evaluation.md'
MODEL_VERSION = '1.0.0'
MODEL_DIR = ROOT / 'models' / 'qb_draft' / f'v{MODEL_VERSION}'
DATA_FILE = ROOT / 'data' / 'modeling' / 'draft_universe_v5.0.0.parquet'
SEED = 20261110
MIN_FINAL_ATTEMPTS = 100
MIN_YEARS_SINCE_HS = 4
LAST_VERIFIABLE_FINAL = 2024
PROB_TEST_YEARS = list(range(2014, 2026))
ROUND_TEST_YEARS = list(range(2012, 2027))
POWER = {'SEC', 'Big Ten', 'Big 12', 'ACC', 'Pac-12', 'Pac-10'}

COLLEGE_FEATURES = ['completion_pct', 'yards_per_attempt', 'td_pct', 'int_pct', 'attempts_per_game', 'log_attempts',
                    'opp_adj_ypa', 'sos_pass_defense', 'rush_ypg_last3', 'n_seasons_100att', 'ypa_change_vs_prior',
                    'recruit_rating', 'years_since_hs', 'n_seasons_present', 'height_inches', 'weight_lbs',
                    'team_recruiting_pctile', 'power_conference', 'team_win_pct']
COMBINE_FEATURES = ['combine_invited', 'combine_forty', 'combine_height_inches', 'combine_weight_lbs']
SIMPLE_FEATURES = ['yards_per_attempt', 'log_attempts', 'power_conference']
ROUND_FEATURES = [c for c in COLLEGE_FEATURES if c not in ('years_since_hs', 'n_seasons_present')] + ['age_at_draft'] + COMBINE_FEATURES[1:]
FORBIDDEN = {'drafted', 'round', 'pick', 'overall', 'overall_pick', 'draft_round', 'draft_ovr', 'nfl_team'}


def pipe(est):
    # keep_empty_features: early draft years have no roster weights at all; keep the column (filled 0) so shapes match
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler()),
                     ('model', est)])


def _ro(db: Path = DB):
    return sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)


# ------------------------------------------------------------------ data
def load_sources(db: Path = DB) -> dict:
    con = _ro(db)
    try:
        s = {
            'season_rows': pd.read_sql('select s.season, s.school, s.height_inches, s.weight_lbs, s.attempts, p.cfbd_player_id, '
                                       'p.display_name from cfb_player_seasons s join cfb_players p on p.id = s.player_id', con),
            'box_pass': pd.read_sql('select distinct season, cfbd_player_id from cfb_game_passing where cfbd_player_id is not null', con),
            'box_names': pd.read_sql('select cfbd_player_id, player_name, count(*) n from cfb_game_passing '
                                     'where cfbd_player_id is not null group by 1, 2', con),
            'box_rush': pd.read_sql('select distinct season, cfbd_player_id from cfb_game_rushing where cfbd_player_id is not null', con),
            'recruits': pd.read_sql('select athlete_id, class_year, rating, stars, position from cfb_recruits where athlete_id is not null', con),
            'picks': pd.read_sql('select year, overall, round, pick, name, position, college_athlete_id, college_team from cfbd_draft_picks', con),
            'games': pd.read_sql('select season, home_team, away_team, home_points, away_points, completed from cfb_games', con),
        }
    finally:
        con.close()
    for k in ('season_rows', 'box_pass', 'box_rush', 'box_names'):
        s[k]['cfbd_player_id'] = s[k]['cfbd_player_id'].map(id_str)
    s['recruits']['athlete_id'] = s['recruits']['athlete_id'].map(id_str)
    s['picks']['college_athlete_id'] = s['picks']['college_athlete_id'].map(id_str)
    s['ref'] = era.load_or_build_reference()
    s['talent'] = pd.read_parquet(ROOT / 'data' / 'model_ready' / 'team_roster_talent.parquet')
    s['combine'] = pd.read_parquet(ROOT / 'data' / 'nflverse_cache' / 'combine_2026-11-09.parquet')
    s['prospects'] = pd.read_parquet(ROOT / 'data' / 'qb_draft_prospects.parquet')
    # Phase-3 college identity of drafted QBs (name+school / box-score / CFBD draft-id bridge). Needed because CFBD's
    # draft-pick athlete ids do NOT match its box-score athlete ids for older players (e.g. Matt Ryan 2125735 vs 145026).
    pcs = pd.read_parquet(ROOT / 'data' / 'model_ready' / 'prospect_college_seasons.parquet')
    idm = pcs[['qb_draft_prospect_id', 'cfbd_player_id']].dropna().drop_duplicates('qb_draft_prospect_id')
    idm['cfbd_player_id'] = idm['cfbd_player_id'].map(id_str)
    pr = s['prospects'].rename(columns={'id': 'qb_draft_prospect_id'})
    s['identity'] = idm.merge(pr[['qb_draft_prospect_id', 'draft_year', 'round', 'pick', 'player_name', 'birthdate']],
                              on='qb_draft_prospect_id')
    return s


def team_win_pct(games: pd.DataFrame) -> pd.DataFrame:
    g = games[(games['completed'] == 1) & games['home_points'].notna() & games['away_points'].notna()]
    h = pd.DataFrame({'season': g['season'], 'team': g['home_team'], 'win': (g['home_points'] > g['away_points']).astype(float)})
    a = pd.DataFrame({'season': g['season'], 'team': g['away_team'], 'win': (g['away_points'] > g['home_points']).astype(float)})
    return pd.concat([h, a]).groupby(['season', 'team'])['win'].mean().rename('team_win_pct').reset_index()


def presence(src: dict) -> pd.DataFrame:
    p = pd.concat([src['season_rows'][['cfbd_player_id', 'season']], src['box_pass'], src['box_rush'],
                   src['ref'][['cfbd_player_id', 'season']]]).dropna().drop_duplicates()
    return p


def ht_inches(h):
    if not isinstance(h, str) or '-' not in h:
        return np.nan
    f, i = h.split('-')[:2]
    try:
        return 12 * int(f) + int(i)
    except ValueError:
        return np.nan


def passer_features(ref: pd.DataFrame, rows: pd.DataFrame, src: dict) -> pd.DataFrame:
    """Features for given (cfbd_player_id, season = final college season) rows, computed the same
    way for drafted and undrafted QBs, from the all-passer reference table."""
    r = ref.sort_values(['cfbd_player_id', 'season'])
    by = {k: g for k, g in r.groupby('cfbd_player_id')}
    out = []
    for _, x in rows.iterrows():
        g = by.get(x['cfbd_player_id'])
        if g is None:
            out.append({}); continue
        g = g[g['season'] <= x['season']]
        if g.empty or g['season'].max() != x['season']:
            out.append({}); continue
        f = g[g['season'] == x['season']].sort_values('attempts', ascending=False).iloc[0]
        prev = g[(g['season'] == x['season'] - 1) & (g['attempts'] >= 100)]
        last3 = g[(g['season'] > x['season'] - 3) & (g['games_with_pass_attempt'] > 0)]
        w = last3['games_with_pass_attempt']
        out.append({
            'team': f['team_first'], 'conference': f['conference'], 'is_fbs': bool(f['is_fbs']),
            'final_attempts': f['attempts'], 'completion_pct': f['completion_pct'], 'yards_per_attempt': f['yards_per_attempt'],
            'td_pct': f['td_pct'], 'int_pct': f['int_pct'], 'attempts_per_game': f['attempts_per_game'],
            'log_attempts': np.log1p(f['attempts']), 'opp_adj_ypa': f['opp_adj_ypa'], 'sos_pass_defense': f['sos_pass_defense'],
            'rush_ypg_last3': float(np.average(last3['rush_yards_per_game'].fillna(0), weights=w)) if w.sum() > 0 else np.nan,
            'n_seasons_100att': int((g['attempts'] >= 100).sum()),
            'ypa_change_vs_prior': (f['yards_per_attempt'] - prev['yards_per_attempt'].iloc[0]) if len(prev) else np.nan,
            'power_conference': float(f['conference'] in POWER or f['team_first'] == 'Notre Dame')})
    F = pd.DataFrame(out, index=rows.index)
    res = pd.concat([rows, F], axis=1)
    # recruiting
    rec = (src['recruits'].sort_values('rating', ascending=False).drop_duplicates('athlete_id')
           .rename(columns={'athlete_id': 'cfbd_player_id', 'class_year': 'hs_class_year', 'rating': 'recruit_rating'}))
    res = res.merge(rec[['cfbd_player_id', 'hs_class_year', 'recruit_rating', 'stars']], on='cfbd_player_id', how='left')
    res['years_since_hs'] = (res['season'] - res['hs_class_year'] + 1).clip(upper=6)
    # roster height / weight (latest season with a value)
    sr = src['season_rows'].dropna(subset=['cfbd_player_id']).sort_values('season')
    hw = sr.groupby('cfbd_player_id').agg(height_inches=('height_inches', 'last'), weight_lbs=('weight_lbs', 'last'),
                                          name=('display_name', 'last'))
    if 'name' in res.columns:
        res = res.drop(columns=['name'])
    res = res.merge(hw, left_on='cfbd_player_id', right_index=True, how='left')
    bn = src['box_names'].sort_values('n', ascending=False).drop_duplicates('cfbd_player_id').set_index('cfbd_player_id')['player_name']
    res['name'] = res['name'].fillna(res['cfbd_player_id'].map(bn))
    # team context
    tal = src['talent'][['season', 'team', 'recruiting_4yr_pctile']].rename(columns={'recruiting_4yr_pctile': 'team_recruiting_pctile'})
    res = res.merge(tal, on=['season', 'team'], how='left').merge(team_win_pct(src['games']), on=['season', 'team'], how='left')
    return res


def attach_combine(df: pd.DataFrame, combine: pd.DataFrame, year_col: str = 'draft_year') -> pd.DataFrame:
    c = combine[combine['pos'] == 'QB'].copy()
    c['k_name'] = c['player_name'].map(normalize_name)
    c['k_school'] = c['school'].map(normalize_school)
    c = c.drop_duplicates(['season', 'k_name', 'k_school'])
    d = df.copy()
    d['k_name'] = d['name'].fillna('').map(normalize_name)
    d['k_school'] = d['team'].fillna('').map(normalize_school)
    m = d.merge(c[['season', 'k_name', 'k_school', 'forty', 'ht', 'wt']].rename(columns={'season': year_col}),
                on=[year_col, 'k_name', 'k_school'], how='left', indicator=True)
    m['combine_invited'] = (m['_merge'] == 'both').astype(float)
    m['combine_forty'] = m['forty']
    m['combine_height_inches'] = m['ht'].map(ht_inches)
    m['combine_weight_lbs'] = m['wt']
    return m.drop(columns=['_merge', 'forty', 'ht', 'wt', 'k_name', 'k_school'])


def build_universe(src: dict) -> pd.DataFrame:
    ref = src['ref']
    pres = presence(src)
    last = pres.groupby('cfbd_player_id')['season'].max().rename('last_season_any')
    data_max = int(pres['season'].max())
    cand = ref[(ref['attempts'] >= MIN_FINAL_ATTEMPTS) & ref['is_fbs']][['cfbd_player_id', 'season']].drop_duplicates()
    cand = cand.merge(last, left_on='cfbd_player_id', right_index=True)
    cand = cand[cand['season'] == cand['last_season_any']].drop(columns='last_season_any')
    cand['draft_year'] = cand['season'] + 1
    seasons_present = pres.merge(cand[['cfbd_player_id', 'season']].rename(columns={'season': 'final'}), on='cfbd_player_id')
    seasons_present = seasons_present[seasons_present['season'] <= seasons_present['final']].groupby('cfbd_player_id')['season'].nunique()
    cand['n_seasons_present'] = cand['cfbd_player_id'].map(seasons_present).fillna(1).astype(int)
    u = passer_features(ref, cand.reset_index(drop=True), src)
    picks = src['picks']
    ident = src['identity'].rename(columns={'pick': 'overall'}).assign(position='Quarterback', label_source='qb_identity_bridge')
    nxt_pick = picks.rename(columns={'college_athlete_id': 'cfbd_player_id', 'year': 'draft_year'})[
        ['cfbd_player_id', 'draft_year', 'round', 'overall', 'position']].assign(label_source='cfbd_draft_athlete_id')
    nxt = (pd.concat([ident[['cfbd_player_id', 'draft_year', 'round', 'overall', 'position', 'label_source']], nxt_pick])
           .drop_duplicates(['cfbd_player_id', 'draft_year']))
    u = u.merge(nxt, on=['cfbd_player_id', 'draft_year'], how='left')
    u['drafted'] = u['round'].notna().astype(int)
    ever = pd.concat([ident[['cfbd_player_id', 'draft_year']], nxt_pick[['cfbd_player_id', 'draft_year']]])
    later = ever.groupby('cfbd_player_id')['draft_year'].min()
    u['drafted_other_year'] = (u['drafted'] == 0) & u['cfbd_player_id'].map(later).notna()
    by_recruit = u['years_since_hs'] >= MIN_YEARS_SINCE_HS
    by_seasons = u['n_seasons_present'] >= MIN_YEARS_SINCE_HS
    u['eligibility_basis'] = np.select([by_recruit, by_seasons], ['recruit_class_year', 'observed_college_seasons'], 'none')
    reason = np.select(
        [u['season'] > min(LAST_VERIFIABLE_FINAL, data_max - 1), u['drafted_other_year'], by_recruit | by_seasons,
         u['hs_class_year'].notna()],
        ['final_season_not_verifiable_yet', 'drafted_in_a_later_draft', 'in_population',
         'three_or_fewer_years_since_high_school'], default='no_evidence_of_four_college_years')
    u['population_status'] = reason
    u['in_population'] = reason == 'in_population'
    return attach_combine(u, src['combine'])


def build_drafted(src: dict) -> pd.DataFrame:
    """All drafted QBs 2005-2026 from the nflverse draft board, linked to their college athlete id by the
    Phase-3 identity bridge, with features from their final college passer-season (<= draft year - 1)."""
    q = src['identity'][src['identity']['draft_year'].between(2005, 2026)].rename(columns={'pick': 'overall'}).copy()
    ref = src['ref']
    lastp = ref.groupby('cfbd_player_id')['season'].max()
    q['season'] = [min(lastp[i], y - 1) if i in lastp.index else np.nan for i, y in zip(q['cfbd_player_id'], q['draft_year'])]
    q = q.dropna(subset=['season'])
    q['season'] = q['season'].astype(int)
    f = passer_features(ref, q[['cfbd_player_id', 'season', 'draft_year', 'round', 'overall', 'qb_draft_prospect_id',
                                'player_name', 'birthdate']].reset_index(drop=True), src)
    f = f[f['final_attempts'] >= 50]
    bd = pd.to_datetime(f['birthdate'], errors='coerce')
    f['age_at_draft'] = (pd.to_datetime(f['draft_year'].astype(str) + '-04-25') - bd).dt.days / 365.25
    f['name'] = f['player_name'].fillna(f['name'])
    f = attach_combine(f, src['combine'])
    return f.reset_index(drop=True)


# ------------------------------------------------------------------ probability models
def prob_models() -> dict:
    return {
        'base_rate': (None, None),
        'simple_l2': (SIMPLE_FEATURES, lambda: pipe(LogisticRegression(C=0.5, max_iter=5000))),
        'college_l2': (COLLEGE_FEATURES, lambda: pipe(LogisticRegression(C=0.5, max_iter=5000))),
        'college_combine_l2': (COLLEGE_FEATURES + COMBINE_FEATURES, lambda: pipe(LogisticRegression(C=0.5, max_iter=5000))),
        'college_combine_gbdt': (COLLEGE_FEATURES + COMBINE_FEATURES, lambda: Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('model', HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=20,
                                                     random_state=SEED))])),
    }


def prob_backtest(u: pd.DataFrame) -> pd.DataFrame:
    pop = u[u['in_population']].reset_index(drop=True)
    specs = prob_models()

    def fp(train, test):
        out = pd.DataFrame({'cfbd_player_id': test['cfbd_player_id'].to_numpy(), 'name': test['name'].to_numpy(),
                            'y': test['drafted'].to_numpy()})
        for name, (cols, mk) in specs.items():
            out[name] = (np.full(len(test), train['drafted'].mean()) if mk is None
                         else mk().fit(train[cols], train['drafted']).predict_proba(test[cols])[:, 1])
        return out
    return BT.run(pop, 'draft_year', PROB_TEST_YEARS, BT.LAG_DRAFT, fp, min_train=50)


# ------------------------------------------------------------------ round / pick models
ROUNDS = np.arange(1, 8)


class OrdinalLogit:
    """Cumulative binary logistic models P(round <= k), k = 1..6, made monotone in k, then
    differenced into class probabilities (non-negative, summing to 1)."""

    def __init__(self, C=0.5):
        self.C = C

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        self.models = []
        for k in ROUNDS[:-1]:
            t = (y <= k).astype(int)
            self.models.append(('const', t.mean()) if len(np.unique(t)) < 2
                               else ('m', pipe(LogisticRegression(C=self.C, max_iter=5000)).fit(X, t)))
        return self

    def predict_proba(self, X):
        cum = np.column_stack([np.full(len(X), m[1]) if m[0] == 'const' else m[1].predict_proba(X)[:, 1]
                               for m in self.models])
        cum = np.maximum.accumulate(np.clip(cum, 0, 1), axis=1)
        cum = np.column_stack([cum, np.ones(len(X))])
        p = np.diff(np.column_stack([np.zeros(len(X)), cum]), axis=1)
        return p / p.sum(1, keepdims=True)


class ProportionalOdds:
    """Proportional-odds (cumulative logit) model: P(round <= k) = sigmoid(theta_k - x.beta) with ONE coefficient
    vector and ordered thresholds (theta_k = theta_1 + sum exp(d_j)). L2 penalty on beta matching sklearn's C
    convention (loss = sum NLL + ||beta||^2 / (2C)). Added in Phase 5 after the separate-fit ordinal model above was
    seen to produce incoherent (bimodal) round distributions; both are reported."""

    def __init__(self, C=0.5):
        self.C = C

    def _theta(self, a, d):
        return a + np.concatenate([[0.0], np.cumsum(np.exp(d))])

    def fit(self, X, y):
        from scipy.optimize import minimize
        self.prep = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('scale', StandardScaler())])
        Z = self.prep.fit_transform(X)
        y = np.asarray(y, dtype=int)
        k, n_f = len(ROUNDS) - 1, Z.shape[1]
        freq = np.clip(np.cumsum(np.bincount(y, minlength=8)[1:])[:-1] / len(y), 0.02, 0.98)
        start = np.log(freq / (1 - freq))
        x0 = np.concatenate([np.zeros(n_f), [start[0]], np.log(np.maximum(np.diff(start), 0.05))])

        def nll(w):
            beta, a, d = w[:n_f], w[n_f], w[n_f + 1:]
            th = self._theta(a, d)
            eta = Z @ beta
            cum = 1 / (1 + np.exp(-(th[None, :] - eta[:, None])))
            cum = np.column_stack([np.zeros(len(Z)), cum, np.ones(len(Z))])
            pr = np.clip(cum[np.arange(len(y)), y] - cum[np.arange(len(y)), y - 1], 1e-12, 1)
            return -np.log(pr).sum() + (beta @ beta) / (2 * self.C)
        r = minimize(nll, x0, method='L-BFGS-B')
        self.beta, self.a, self.d = r.x[:n_f], r.x[n_f], r.x[n_f + 1:]
        self.converged = bool(r.success)
        return self

    def predict_proba(self, X):
        Z = self.prep.transform(X)
        th = self._theta(self.a, self.d)
        cum = 1 / (1 + np.exp(-(th[None, :] - (Z @ self.beta)[:, None])))
        cum = np.column_stack([np.zeros(len(Z)), cum, np.ones(len(Z))])
        return np.diff(cum, axis=1)


class Multinomial:
    def __init__(self, C=0.5):
        self.C = C

    def fit(self, X, y):
        self.m = pipe(LogisticRegression(C=self.C, max_iter=5000)).fit(X, np.asarray(y, dtype=int))
        self.classes_ = self.m.named_steps['model'].classes_
        return self

    def predict_proba(self, X):
        p = np.zeros((len(X), len(ROUNDS)))
        p[:, self.classes_ - 1] = self.m.predict_proba(X)
        return p


class PickRange:
    """Ridge on log(overall pick); intervals from out-of-sample residuals (GroupKFold by class)."""

    def __init__(self, cols, levels=(0.1, 0.25, 0.75, 0.9)):
        self.cols, self.levels = cols, levels

    def fit(self, df):
        X, y = df[self.cols], np.log(df['overall'].astype(float))
        self.m = pipe(Ridge(alpha=10.0)).fit(X, y)
        res = np.full(len(df), np.nan)
        g = df['draft_year'].to_numpy()
        for tr, te in GroupKFold(n_splits=min(5, len(np.unique(g)))).split(X, y, g):
            res[te] = y.iloc[te] - pipe(Ridge(alpha=10.0)).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        self.q = dict(zip(self.levels, np.quantile(res[np.isfinite(res)], self.levels)))
        self.max_pick = float(df['overall'].max())
        return self

    def predict(self, X):
        mu = self.m.predict(X[self.cols])
        out = {'median_pick': np.exp(mu)}
        for lv, q in self.q.items():
            out[f'q{int(lv * 100)}'] = np.clip(np.exp(mu + q), 1, 262)
        return pd.DataFrame(out)


def rps(P, y):
    """Ranked probability score over ordered rounds (lower is better)."""
    cp = np.cumsum(P, axis=1)
    co = np.cumsum(np.eye(len(ROUNDS))[np.asarray(y, int) - 1], axis=1)
    return float(np.mean(((cp - co) ** 2).sum(1) / (len(ROUNDS) - 1)))


def round_backtest(dr: pd.DataFrame) -> pd.DataFrame:
    def fp(train, test):
        y = train['round'].astype(int).to_numpy()
        base = np.bincount(y, minlength=8)[1:] / len(y)
        out = {'cfbd_player_id': test['cfbd_player_id'].to_numpy(), 'name': test['name'].to_numpy(),
               'round': test['round'].astype(int).to_numpy(), 'overall': test['overall'].to_numpy()}
        probs = {'base_rate': np.tile(base, (len(test), 1)),
                 'multinomial': Multinomial().fit(train[ROUND_FEATURES], y).predict_proba(test[ROUND_FEATURES]),
                 'ordinal': OrdinalLogit().fit(train[ROUND_FEATURES], y).predict_proba(test[ROUND_FEATURES]),
                 'prop_odds': ProportionalOdds().fit(train[ROUND_FEATURES], y).predict_proba(test[ROUND_FEATURES])}
        for k, P in probs.items():
            for r in ROUNDS:
                out[f'{k}_p{r}'] = P[:, r - 1]
        pk = PickRange(ROUND_FEATURES).fit(train).predict(test)
        for c in pk.columns:
            out[f'pick_{c}'] = pk[c].to_numpy()
        out['pick_baseline_median'] = np.full(len(test), float(train['overall'].median()))
        bl = np.quantile(train['overall'], [0.1, 0.9])
        out['pick_baseline_q10'], out['pick_baseline_q90'] = bl[0], bl[1]
        return pd.DataFrame(out)
    return BT.run(dr, 'draft_year', ROUND_TEST_YEARS, BT.LAG_DRAFT, fp, min_train=40)


# ------------------------------------------------------------------ evaluation
def evaluate_prob(pb: pd.DataFrame) -> pd.DataFrame:
    from src.modeling.evaluate import fast_auc, fast_brier, fast_log_loss
    rows = []
    d = pb.rename(columns={'test_year': 'draft_year'})
    for m in prob_models():
        c = U.calibration_summary(d['y'], d[m])
        b = U.cluster_bootstrap(lambda x: fast_brier(x['y'], x[m]), d, n_boot=1000, seed=SEED)
        a = U.cluster_bootstrap(lambda x: fast_auc(x['y'], x[m]), d, n_boot=1000, seed=SEED)
        rows.append({'model': m, 'n': len(d), 'n_drafted': int(d['y'].sum()), 'auc': fast_auc(d['y'], d[m]),
                     'auc_lo': a['lower'], 'auc_hi': a['upper'], 'brier': fast_brier(d['y'], d[m]),
                     'brier_lo': b['lower'], 'brier_hi': b['upper'],
                     'log_loss': fast_log_loss(d['y'], np.clip(d[m], 1e-4, 1 - 1e-4)), **c})
    return pd.DataFrame(rows)


def evaluate_round(rb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    y = rb['round'].to_numpy()
    d = rb.rename(columns={'test_year': 'draft_year'})
    for m in ('base_rate', 'multinomial', 'ordinal', 'prop_odds'):
        P = rb[[f'{m}_p{r}' for r in ROUNDS]].to_numpy()
        exp_r = P @ ROUNDS
        med_r = np.argmax(np.cumsum(P, 1) >= 0.5, axis=1) + 1
        top = np.argmax(P, 1) + 1
        ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y - 1], 1e-6, 1))))

        def rps_df(x, m=m):
            return rps(x[[f'{m}_p{r}' for r in ROUNDS]].to_numpy(), x['round'].to_numpy())
        ci = U.cluster_bootstrap(rps_df, d, n_boot=1000, seed=SEED)
        rows.append({'model': m, 'n': len(y), 'log_loss': ll, 'rps': rps(P, y), 'rps_lo': ci['lower'], 'rps_hi': ci['upper'],
                     'accuracy_top_round': float((top == y).mean()), 'within_one_round': float((np.abs(med_r - y) <= 1).mean()),
                     'mae_expected_round': float(np.mean(np.abs(exp_r - y))), 'mae_median_round': float(np.mean(np.abs(med_r - y))),
                     'prob_sum_max_dev': float(np.abs(P.sum(1) - 1).max())})
    R = pd.DataFrame(rows)
    o = rb['overall'].to_numpy(float)
    pk = [{'model': 'ridge_log_pick', 'mae_picks': float(np.mean(np.abs(rb['pick_median_pick'] - o))),
           'median_abs_error_picks': float(np.median(np.abs(rb['pick_median_pick'] - o))),
           'coverage50': U.coverage(o, rb['pick_q25'], rb['pick_q75']), 'coverage80': U.coverage(o, rb['pick_q10'], rb['pick_q90']),
           'mean_width80': float((rb['pick_q90'] - rb['pick_q10']).mean())},
          {'model': 'historical_median', 'mae_picks': float(np.mean(np.abs(rb['pick_baseline_median'] - o))),
           'median_abs_error_picks': float(np.median(np.abs(rb['pick_baseline_median'] - o))), 'coverage50': np.nan,
           'coverage80': U.coverage(o, rb['pick_baseline_q10'], rb['pick_baseline_q90']),
           'mean_width80': float((rb['pick_baseline_q90'] - rb['pick_baseline_q10']).mean())}]
    return R, pd.DataFrame(pk)


def fig_draft(pb: pd.DataFrame, rb: pd.DataFrame, u: pd.DataFrame):
    from src.modeling.evaluate import INK, INK_2, SURFACE, fig_ax, save
    fig, axes = fig_ax(13, 4.4, ncols=3)
    pop = u[u['in_population']]
    c = pop.groupby('draft_year')['drafted'].agg(['size', 'sum'])
    axes[0].bar(c.index, c['sum'], color='#2a78d6', width=0.8, edgecolor=SURFACE, label='Drafted (next draft)')
    axes[0].bar(c.index, c['size'] - c['sum'], bottom=c['sum'], color='#b7d3f6', width=0.8, edgecolor=SURFACE, label='Not drafted')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title('Verified prospect universe per draft', color=INK, fontsize=10, loc='left')
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=INK_2, linestyle='--', linewidth=1)
    for m, col in (('college_l2', '#2a78d6'), ('college_combine_l2', '#eb6834'), ('simple_l2', '#1baf7a')):
        t = U.reliability_table(pb['y'], pb[m], n_bins=8)
        ax.plot(t['mean_predicted'], t['observed'], color=col, linewidth=2, marker='o', markersize=5,
                markeredgecolor=SURFACE, label=m)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('Draft-probability calibration (backtest)', color=INK, fontsize=10, loc='left')
    ax.set_xlabel('Predicted P(drafted), 8 quantile bins', color=INK, fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    P = rb[[f'ordinal_p{r}' for r in ROUNDS]].to_numpy()
    exp_r = P @ ROUNDS
    ax.plot(rb['round'] + np.random.default_rng(0).uniform(-0.2, 0.2, len(rb)), exp_r, 'o', color='#2a78d6', markersize=4, alpha=0.7)
    ax.plot([1, 7], [1, 7], color=INK_2, linestyle='--', linewidth=1)
    ax.set_xlabel('Actual round', color=INK, fontsize=8.5)
    ax.set_ylabel('Expected round (ordinal model)', color=INK, fontsize=8.5)
    ax.set_title('Round forecasts, drafted QBs (backtest)', color=INK, fontsize=10, loc='left')
    save(fig, 'p5_draft_model.png')


def fit_final(u: pd.DataFrame, dr: pd.DataFrame) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pop = u[u['in_population']]
    arts = {}
    for name in ('college_l2', 'college_combine_l2'):
        cols, mk = prob_models()[name]
        joblib.dump({'pipeline': mk().fit(pop[cols], pop['drafted']), 'features': cols, 'model_version': MODEL_VERSION,
                     'population': 'FBS QB, final college season verified, >= 4 years since HS class'},
                    MODEL_DIR / f'prob_{name}.joblib')
        arts[name] = f'prob_{name}.joblib'
    y = dr['round'].astype(int)
    joblib.dump({'ordinal': OrdinalLogit().fit(dr[ROUND_FEATURES], y), 'multinomial': Multinomial().fit(dr[ROUND_FEATURES], y),
                 'prop_odds': ProportionalOdds().fit(dr[ROUND_FEATURES], y),
                 'pick': PickRange(ROUND_FEATURES).fit(dr), 'features': ROUND_FEATURES, 'model_version': MODEL_VERSION},
                MODEL_DIR / 'round_pick.joblib')
    arts['round_pick'] = 'round_pick.joblib'
    return arts


def main() -> dict:
    from src.modeling import artifacts as A
    from src.modeling.train import package_versions
    for cols in (COLLEGE_FEATURES, COMBINE_FEATURES, ROUND_FEATURES):
        assert not set(cols) & FORBIDDEN
    src = load_sources()
    u = build_universe(src)
    dr = build_drafted(src)
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    u.to_parquet(DATA_FILE, index=False)
    RESULTS.mkdir(parents=True, exist_ok=True)
    # identity cross-check: every drafted QB in the universe should also be on the nflverse draft board
    q_ids = set(dr['cfbd_player_id'])
    pop = u[u['in_population']]
    drafted_qb_in_pop = pop[(pop['drafted'] == 1) & (pop['position'] == 'Quarterback')]
    ident = {'drafted_in_population': int(pop['drafted'].sum()),
             'drafted_in_population_at_qb': int(len(drafted_qb_in_pop)),
             'drafted_at_other_positions': pop.loc[(pop['drafted'] == 1) & (pop['position'] != 'Quarterback'),
                                                   ['name', 'draft_year', 'position']].to_dict('records'),
             'qb_drafted_in_population_also_in_round_dataset': int(drafted_qb_in_pop['cfbd_player_id'].isin(q_ids).sum()),
             'round_dataset_rows_with_nflverse_board_match': int(dr['qb_draft_prospect_id'].notna().sum()),
             'round_dataset_rows': int(len(dr)),
             'combine_match_rate_drafted_round_dataset': float(dr['combine_invited'].mean())}
    pb = prob_backtest(u)
    rb = round_backtest(dr)
    pb.to_csv(RESULTS / 'draft_prob_backtest.csv', index=False)
    rb.to_csv(RESULTS / 'draft_round_backtest.csv', index=False)
    PM = evaluate_prob(pb)
    RM, PK = evaluate_round(rb)
    for n, t in (('draft_prob_metrics', PM), ('draft_round_metrics', RM), ('draft_pick_metrics', PK)):
        t.to_csv(RESULTS / f'{n}.csv', index=False)
    fig_draft(pb, rb, u)
    status = u.groupby(['population_status']).agg(rows=('drafted', 'size'), drafted=('drafted', 'sum')).reset_index()
    per_year = pop.groupby('draft_year')['drafted'].agg(['size', 'sum']).reset_index()
    exists = (MODEL_DIR / 'metadata.json').exists()
    arts = {} if exists else fit_final(u, dr)
    if not exists:
        meta = {'model_version': MODEL_VERSION, 'random_seed': SEED,
                'population_rule': {'min_final_attempts': MIN_FINAL_ATTEMPTS, 'fbs_final_season': True,
                                    'verified_final_season': f'no later CFBD presence; final seasons <= {LAST_VERIFIABLE_FINAL}',
                                    'min_years_since_hs_class': MIN_YEARS_SINCE_HS, 'label': 'drafted in the next draft, any position'},
                'features': {'college': COLLEGE_FEATURES, 'combine': COMBINE_FEATURES, 'round_pick': ROUND_FEATURES},
                'training': {'probability_rows': int(len(pop)), 'probability_draft_years': sorted(pop['draft_year'].unique().tolist()),
                             'round_rows': int(len(dr)), 'round_draft_years': sorted(dr['draft_year'].unique().tolist())},
                'backtest': {'probability_test_years': PROB_TEST_YEARS, 'round_test_years': ROUND_TEST_YEARS,
                             'label_lag_years': BT.LAG_DRAFT},
                'artifacts': sorted(arts.values()) + ['metadata.json'],
                'artifact_sha256': {v: A.sha256(MODEL_DIR / v) for v in arts.values()},
                'external_data_files': [{'role': 'draft_universe', 'path': A.rel(DATA_FILE), 'sha256': A.sha256(DATA_FILE),
                                         'regenerate': 'python -m src.modeling.draft_predictor',
                                         'reason': 'player-level CFBD-derived data; kept out of Git'}],
                'package_versions': package_versions()}
        (MODEL_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    summary = {'identity': ident, 'status': status.to_dict('records'), 'per_year': per_year.to_dict('records'),
               'n_population': int(len(pop)), 'n_round': int(len(dr))}
    (RESULTS / 'draft_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    REPORT.write_text(render(summary, PM, RM, PK, pb, rb, u, dr), encoding='utf-8')
    print(PM[['model', 'auc', 'brier', 'log_loss', 'calibration_slope']].round(3).to_string())
    print(RM.round(3).to_string()); print(PK.round(2).to_string())
    return summary


def render(summary, PM, RM, PK, pb, rb, u, dr) -> str:
    from src.modeling.evaluate import f3
    P = PM.set_index('model'); R = RM.set_index('model'); K = PK.set_index('model')
    st = pd.DataFrame(summary['status']).set_index('population_status')
    pop = u[u['in_population']]
    yrs = sorted(pop['draft_year'].unique())
    L = ['# NFL Draft projection: evaluation', '',
         'Generated by `python -m src.modeling.draft_predictor`. All metrics come from **chronological backtests**: each draft '
         'class is predicted by models trained only on earlier drafts (a draft outcome is known on draft day). Hyperparameters '
         'were fixed in advance (L2 C=0.5; boosting with depth 2, 150 trees, min leaf 20). The results are exploratory, not '
         'validated accuracy.', '',
         '## 1. Who could have been drafted? (the prospect universe)', '',
         'The draft tables list only drafted players. "Not in the draft table" is not the same as "undrafted": it also '
         'includes QBs who returned to school, transferred down a level, or stopped playing. There is no CFBD record of '
         'draft declarations. The model is therefore trained only on QBs whose draft status can be verified from local data:', '',
         f'1. A passer-season with at least {MIN_FINAL_ATTEMPTS} attempts at an FBS school,',
         '2. that is the athlete\'s **last appearance anywhere** in the local CFBD data (season stats or box scores), checked '
         f'against the following season (so final seasons up to {LAST_VERIFIABLE_FINAL}),',
         f'3. with evidence of **at least {MIN_YEARS_SINCE_HS} college years**: a linked high-school recruiting class at least '
         f'{MIN_YEARS_SINCE_HS} years earlier, or appearances in at least {MIN_YEARS_SINCE_HS} distinct seasons of CFBD data. '
         'Either one makes him NFL-draft-eligible (three years removed from high school), and he did not continue in college.', '',
         '**Label:** drafted in the very next draft, at any position. Drafted QBs are identified through the project\'s '
         'Phase-3 QB identity bridge, because CFBD\'s draft-pick athlete ids do **not** match its box-score athlete ids '
         'for older players (Matt Ryan is 2125735 in one and 145026 in the other). Using the pick ids alone '
         'mislabelled the entire 2008 QB class as undrafted. CFBD pick ids are used only for players drafted at other '
         'positions.', '',
         '| Candidate status | Rows | Drafted |', '|---|---|---|']
    for k, r in st.iterrows():
        L.append(f'| {k} | {int(r["rows"])} | {int(r["drafted"])} |')
    L += ['', f'**Population used: {len(pop)} final seasons ({int(pop["drafted"].sum())} drafted), drafts {yrs[0]}-{yrs[-1]}.** '
          'Eligibility basis: ' + ', '.join(f'{k} {v}' for k, v in pop['eligibility_basis'].value_counts().items()) +
          '. Recruiting links are sparse before about 2011, and the observed-seasons rule fills most of that gap.', '',
          '**What the probability means, and what it does not.** It is P(drafted | an FBS QB\'s college career verifiably '
          'ended after 4+ years). It is **not** a draft probability for every college QB:',
          '* True juniors and younger are excluded. The only way for them to leave is to declare, and undrafted early '
          'entrants cannot be identified locally.',
          '* So are QBs with neither a recruiting link nor four observed seasons (mostly JUCO transfers and short FBS '
          'careers): the "no evidence of four college years" row.',
          '* Two biases remain. A 4th/5th-year QB who moved to a level CFBD did not cover (FCS before 2022) or stopped '
          'playing is labelled "undrafted". And recruiting links may be more complete for higher-rated recruits.', '',
          f'Identity checks: {summary["identity"]["qb_drafted_in_population_also_in_round_dataset"]} of '
          f'{summary["identity"]["drafted_in_population_at_qb"]} drafted QBs in the population are also in the drafted-QB '
          'dataset. Drafted at another position: '
          + (', '.join(f'{x["name"]} ({x["draft_year"]}, {x["position"]})' for x in summary['identity']['drafted_at_other_positions']) or 'none')
          + '.', '',
          '![Draft model](figures/p5_draft_model.png)', '',
          '## 2. Draft probability (backtest, test drafts '
          f'{PROB_TEST_YEARS[0]}-{PROB_TEST_YEARS[-1]})', '',
          '| Model | ROC-AUC [95% CI, by class] | Brier [95% CI] | Log loss | Calibration slope | Calibration-in-the-large | ECE |',
          '|---|---|---|---|---|---|---|']
    lab = {'base_rate': 'Historical draft rate', 'simple_l2': 'Simple college (YPA, attempts, power conference)',
           'college_l2': 'College features (18)', 'college_combine_l2': 'College + Combine (22)',
           'college_combine_gbdt': 'College + Combine, gradient boosting'}
    for m, name in lab.items():
        r = P.loc[m]
        auc = 'n/a' if m == 'base_rate' else f'{r["auc"]:.3f} [{r["auc_lo"]:.3f}, {r["auc_hi"]:.3f}]'
        L.append(f'| {name} | {auc} | {r["brier"]:.3f} [{r["brier_lo"]:.3f}, {r["brier_hi"]:.3f}] | {r["log_loss"]:.3f} | '
                 f'{f3(r["calibration_slope"]) if m != "base_rate" else "-"} | {r["calibration_in_the_large"]:+.3f} | {r["ece"]:.3f} |')
    r0 = P.loc['base_rate']
    L += ['', f'{int(r0["n"])} backtest rows, {int(r0["n_drafted"])} drafted. The confidence intervals resample whole draft '
          'classes. The Combine is a legitimate pre-draft input (it is held in late February or March), but an invitation '
          'already reflects NFL teams\' opinion. "College features" is the cleaner measure of what college production alone '
          'predicts.', '',
          '## 3. Draft round, conditional on being drafted', '',
          f'Every drafted QB from 2005-2026 with a final college passer-season in the data: {len(dr)} QBs. The age-at-draft '
          'birthdate comes from the nflverse draft board. Test drafts: '
          f'{ROUND_TEST_YEARS[0]}-{ROUND_TEST_YEARS[-1]} ({int(R.loc["base_rate", "n"])} QBs).', '',
          '| Model | Log loss | Ranked prob. score [95% CI] | Top-round accuracy | Within one round | MAE, expected round | Max deviation of sum(P) from 1 |',
          '|---|---|---|---|---|---|---|']
    for m, name in (('base_rate', 'Historical round frequencies'), ('multinomial', 'Multinomial logistic'),
                    ('ordinal', 'Ordinal, separate cumulative fits'), ('prop_odds', 'Proportional odds (added in Phase 5, see note)')):
        r = R.loc[m]
        L.append(f'| {name} | {r["log_loss"]:.3f} | {r["rps"]:.3f} [{r["rps_lo"]:.3f}, {r["rps_hi"]:.3f}] | '
                 f'{r["accuracy_top_round"]:.2f} | {r["within_one_round"]:.2f} | {r["mae_expected_round"]:.2f} | {r["prob_sum_max_dev"]:.1e} |')
    L += ['', 'The ranked probability score respects the order of the rounds: predicting round 2 for a round-1 QB costs less '
          'than predicting round 7.', '',
          '**Note on the proportional-odds model.** The pre-specified ordinal model fits six separate cumulative logistic '
          'models. That produced incoherent distributions, for example 0.88 on round 1 and 0.09 on round 6 with almost '
          'nothing between. The proportional-odds model (one coefficient vector, ordered thresholds) was therefore added '
          '**after** seeing that defect, and it is evaluated in the same backtest. Its results are shown for transparency. '
          'Treat any preference between the two as provisional, since it was decided after looking at results.', '',
          '### Overall pick range', '',
          '| Model | MAE (picks) | Median abs. error | 50% interval coverage | 80% interval coverage | Mean 80% width |', '|---|---|---|---|---|---|']
    for m, name in (('ridge_log_pick', 'Ridge on log(pick), out-of-sample residual intervals'), ('historical_median', 'Historical median / 10-90% range')):
        r = K.loc[m]
        L.append(f'| {name} | {r["mae_picks"]:.1f} | {r["median_abs_error_picks"]:.1f} | {f3(r["coverage50"])} | {f3(r["coverage80"])} | {r["mean_width80"]:.0f} |')
    L += ['', f'Median absolute error: {K.loc["ridge_log_pick", "median_abs_error_picks"]:.0f} picks, against '
          f'{K.loc["historical_median", "median_abs_error_picks"]:.0f} for the baseline. But the **mean** absolute error is '
          f'no better ({K.loc["ridge_log_pick", "mae_picks"]:.1f} vs {K.loc["historical_median", "mae_picks"]:.1f}): the model is '
          'usually closer, and occasionally misses by 150+ picks (day-3 QBs projected early, or the reverse). The intervals '
          'are empirical, and their coverage is measured on later drafts: the 80% intervals cover '
          f'{K.loc["ridge_log_pick", "coverage80"]:.0%}, slightly under nominal. Pick ranges are wide. College data '
          'separates likely first-rounders from day-3 QBs, but not pick 5 from pick 25.', '',
          '## 4. Using the models together', '',
          'For a prospect inside the verified universe, P(round = r) = P(drafted) x P(round = r | drafted). For anyone else '
          '(an underclassman, a walk-on, a 2025 final season), `forecast.py` returns only the conditional "if drafted" round and '
          'pick projection, and says that the unconditional draft probability is not defined for him.', '',
          '## Limitations', '',
          '* The universe is a verified subset, not every draft-eligible QB. Probabilities should not be quoted for early '
          'entrants.',
          '* There is no declaration data, no all-star game data, and no pre-draft scouting rankings for undrafted players. '
          'CFBD pre-draft rankings exist only for drafted players, so they are excluded to avoid leakage.',
          f'* The round models train on about {len(dr) / dr["draft_year"].nunique():.0f} QBs per draft. Rounds 2-7 each have '
          'only 1-3 QBs per year, so round-level probabilities are coarse.', '']
    return '\n'.join(L)


if __name__ == '__main__':
    # run through the package module so pickled classes resolve as src.modeling.draft_predictor, not __main__
    from src.modeling import draft_predictor as _m
    _m.main()
