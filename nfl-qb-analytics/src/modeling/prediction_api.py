"""Unified, JSON-serialisable prediction interface for the future dashboard / API (Phase 5.5).

    from src.modeling.prediction_api import forecast_prospect
    f = forecast_prospect(name='Fernando Mendoza', draft_year=2026, mode='pre_draft')
    f = forecast_prospect(name='Jaxson Dart', draft_year=2025, mode='post_draft')
    f = forecast_prospect(cfbd_player_id='4606194', final_season=2024)          # any college passer-season
    f = forecast_prospect(name='Fernando Mendoza', draft_year=2026, mode='post_draft', assumed_pick=10)  # hypothetical

    python -m src.modeling.prediction_api --name "Fernando Mendoza" --draft-year 2026 --mode pre_draft

Works from local artifacts and cached data only: no network access. Every block carries a
`status` ('available' | 'unavailable' | 'partial'); unavailable blocks say why and never contain
numbers. Values are labelled as `observed` (what happened), `predicted` (a model forecast for this QB)
or `historical_descriptive` (group averages), and carry the model version and backtest status.

Modes
  pre_draft   draft projections + career forecasts from pre-draft college/Combine data only
  post_draft  career forecasts that also use the overall pick. Needs the actual pick (drafted QB)
              or `assumed_pick` (labelled hypothetical). The draft block then reports the actual
              result, with the pre-draft projection kept for reference.
"""
from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.modeling import artifacts as A
from src.modeling import career_production as CP
from src.modeling import career_production_v11 as V11
from src.modeling import draft_predictor as DP
from src.modeling import prospect_similarity as PS
from src.modeling import round_audit as RA
from src.modeling.career_trajectory import season_games_cap

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = '1.0.0'
MODES = ('pre_draft', 'post_draft')
STAT_LABELS = {'games_played': 'Games played', 'games_started': 'Games started', 'pass_attempts': 'Pass attempts',
               'pass_completions': 'Completions', 'pass_yards': 'Passing yards', 'pass_tds': 'Passing TDs',
               'interceptions': 'Interceptions', 'rush_attempts': 'Rush attempts', 'rush_yards': 'Rushing yards',
               'rush_tds': 'Rushing TDs'}
DATA_LIMITATIONS = [
    'Draft declarations, transfer-portal entries, class year and remaining eligibility are not in the local data; '
    'draft probability is defined only for FBS QBs whose college career verifiably ended after at least four years.',
    'Career models are trained on 183 drafted QBs (classes 2005-2021). Undrafted QBs are not modelled.',
    'Post-Combine draft probability mostly reflects Combine participation (a scouting decision), not athletic measurements.',
    'NFL efficiency (completion %, YPA, TD and INT rates) is not predictable from college data at this sample size.',
    'Intervals are empirical and measured on later draft classes; coverage under future drift is not guaranteed.',
    'Drafting team, coaching and injuries are not modelled.',
]


def _f(x):
    """JSON-safe float."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) or math.isinf(v) else v


def _i(x):
    v = _f(x)
    return None if v is None else int(round(v))


def unavailable(reason: str, **extra) -> dict:
    return {'status': 'unavailable', 'reason': reason, **extra}


# ------------------------------------------------------------------ engine
class Engine:
    """`source` supplies player profiles (and the draft-probability / v2 models). Default: the Phase-5 engine, which
    assembles profiles from the full local data. The API passes a serving-bundle source with the same interface
    (`profile`, `src`, `draft_prob`, `v2`), so both paths run this same forecasting code."""

    def __init__(self, source=None, combine_seasons=None, hist_max=None):
        if source is None:
            from src.modeling.forecast import Engine as P5Engine
            source = P5Engine()                                 # profile assembly, v1.0.0 draft probability models, v2
        self.p5 = source
        self.prod = {fs: V11.load(fs) for fs in MODES}
        self.prod_meta = A.load_metadata(V11.MODEL_DIR)
        self.round = joblib.load(RA.MODEL_DIR / 'round_pick.joblib')
        self.round_meta = A.load_metadata(RA.MODEL_DIR)
        self.combine_seasons = (set(combine_seasons) if combine_seasons is not None else
                                set(pd.read_parquet(ROOT / 'data' / 'nflverse_cache' / 'combine_2026-11-09.parquet')['season'].unique()))
        try:
            self.pool = pd.read_parquet(A.resolve(PS.MODEL_DIR, 'pool'))
        except (FileNotFoundError, ValueError) as e:
            self.pool, self.pool_error = None, str(e)
        if hist_max is not None:
            self.hist_max = dict(hist_max)
        else:
            try:   # largest five-season totals in the training cohort (aggregates; flags extrapolation)
                d, _ = CP.load_data()
                self.hist_max = {t: float(d[f'{t}_5yr'].max()) for t in CP.TARGETS}
            except Exception:      # noqa: BLE001 - optional context
                self.hist_max = {}

    # -------------------------------------------------------------- identity
    def identity(self, p: dict) -> dict:
        src = self.p5.src()
        v = p['v2']
        pid = v.get('qb_draft_prospect_id')
        pid = None if pid is None or (isinstance(pid, float) and math.isnan(pid)) else int(pid)
        cid = p['draft'].get('cfbd_player_id') if len(p['draft']) else None
        gsis, verification = None, 'cfbd_athlete_id'
        if pid is not None:
            pr = src['prospects']
            row = pr[pr['id'] == pid]
            gsis = row['gsis_id'].iloc[0] if len(row) and pd.notna(row['gsis_id'].iloc[0]) else None
            verification = 'phase3_identity_bridge (nflverse draft board <-> CFBD college record)'
        drafted = p['drafted']
        return {'name': p['name'], 'college': p['college'], 'draft_year': int(p['draft_year']),
                'final_college_season': _i(p['draft'].get('season')) if len(p['draft']) else None,
                'identifiers': {'qb_draft_prospect_id': pid, 'cfbd_player_id': None if cid is None else str(cid), 'gsis_id': gsis},
                'identity_verification': verification,
                'actual_draft': {'label': 'observed', 'drafted': drafted, 'round': _i(p['actual_round']), 'overall_pick': _i(p['actual_pick'])}}

    # -------------------------------------------------------------- draft
    def draft_block(self, p: dict, mode: str) -> dict:
        f = p['draft']
        if f is None or len(f) == 0:
            return unavailable('no college passer-season found for this QB')
        X = f.to_frame().T
        status = f.get('population_status', 'not_in_verified_universe')
        in_pop = bool(f.get('in_population', False))
        # draft probability at two cutoffs
        if in_pop:
            pc = float(self.p5.draft_prob['college_l2']['pipeline'].predict_proba(
                X.reindex(columns=self.p5.draft_prob['college_l2']['features']).astype(float))[:, 1][0])
            combine_held = int(p['draft_year']) in self.combine_seasons
            post = (float(self.p5.draft_prob['college_combine_l2']['pipeline'].predict_proba(
                X.reindex(columns=self.p5.draft_prob['college_combine_l2']['features']).astype(float))[:, 1][0]) if combine_held else None)
            prob = {'status': 'available', 'label': 'predicted', 'pre_combine': pc, 'post_combine': post,
                    'post_combine_note': ('mostly reflects Combine participation (see combine_selection_bias.md)' if combine_held
                                          else f'the {p["draft_year"]} Combine is not in the local data'),
                    'population': 'FBS QB whose college career verifiably ended after >= 4 years',
                    'model_version': 'qb_draft v1.0.0', 'backtest': 'test drafts 2014-2025: AUC 0.878 pre-Combine, 0.907 post-Combine'}
        else:
            prob = unavailable(f'outside the verified prospect universe ({status}); P(drafted) is not extrapolated',
                               population_status=status)
        # round and pick, conditional on being drafted
        Xr = X.reindex(columns=self.round['features']).astype(float)
        p7 = self.round['round7'].predict_proba(Xr)[0]
        p3 = self.round['round3'].predict_proba(Xr)[0]
        mu = float(self.round['pick_model'].predict(Xr)[0])
        q = self.round['pick_log_residual_quantiles']
        pick = {lv: [float(np.clip(np.exp(mu + q[round((1 - lv) / 2, 2)]), 1, 262)),
                     float(np.clip(np.exp(mu + q[round(1 - (1 - lv) / 2, 2)]), 1, 262))] for lv in (0.5, 0.8, 0.9)}
        mb = self.round_meta['measured_backtest']
        rnd = {'status': 'available', 'label': 'predicted', 'conditional_on': 'being drafted',
               'groups': dict(zip(self.round['round3_labels'], map(float, p3))),
               'rounds': {str(r): float(v) for r, v in zip(range(1, 8), p7)},
               'expected_round': float(p7 @ np.arange(1, 8)),
               'display_note': 'show the three groups; single-round probabilities rest on 1-3 QBs per round per year',
               'model_version': 'qb_draft v1.1.0', 'backtest_rps': {'groups': mb['round3_rps_selected'], 'rounds': mb['round7_rps_selected']}}
        pk = {'status': 'available', 'label': 'predicted', 'conditional_on': 'being drafted', 'median': float(np.exp(mu)),
              'interval_50': pick[0.5], 'interval_80': pick[0.8], 'interval_90': pick[0.9],
              'measured_coverage': mb['pick_coverage_groupkfold'], 'model_version': 'qb_draft v1.1.0'}
        out = {'status': 'available', 'probability': prob, 'round': rnd, 'pick': pk,
               'eligibility': {'population_status': status, 'in_verified_universe': in_pop,
                               'classification': ('eligible_career_ended' if in_pop else
                                                  'eligibility_unknown' if status in ('final_season_not_verifiable_yet',
                                                                                      'three_or_fewer_years_since_high_school',
                                                                                      'no_evidence_of_four_college_years',
                                                                                      'not_in_verified_universe') else status)}}
        if mode == 'post_draft':
            out['note'] = 'post-draft view: the actual draft result is in player.actual_draft; these are the pre-draft projections'
        return out

    # -------------------------------------------------------------- career
    def _features(self, p: dict, fs: str, pick, hypothetical: bool = False) -> pd.DataFrame:
        from src.modeling.career_trajectory import round_group
        X = p['v2'].to_frame().T.copy()
        if fs == 'post_draft':
            X['log_overall_pick'] = np.log(float(pick))
        cols = self.prod[fs]['features']
        X = X.reindex(columns=cols).astype(float)
        X['round_group'] = None
        if fs == 'post_draft':
            rnd = p['actual_round'] if not hypothetical and p['actual_round'] is not None and not pd.isna(p['actual_round'])                 else DP_round_from_pick(int(pick))
            X['round_group'] = round_group(int(rnd))
        return X

    def career_block(self, p: dict, mode: str, assumed_pick, n_sims: int, year_by_year: bool) -> dict:
        assumptions = []
        if p['drafted'] is not True:
            assumptions.append('career models are trained only on drafted QBs: this forecast assumes the QB is drafted')
        pick, hypothetical = None, False
        if mode == 'post_draft':
            if assumed_pick is not None:
                pick, hypothetical = int(assumed_pick), True
                assumptions.append(f'hypothetical: assumes overall pick {pick}')
            elif p['actual_pick'] is not None and not pd.isna(p['actual_pick']):
                pick = int(p['actual_pick'])
            else:
                return unavailable('post_draft mode needs an actual overall pick (this QB has none recorded) or assumed_pick')
        art = self.prod[mode]
        X = self._features(p, mode, pick, hypothetical)
        n_missing = int(X[art['features']].isna().sum(axis=1).iloc[0])
        if n_missing > len(art['features']) // 2:
            return unavailable(f'{n_missing} of {len(art["features"])} model features are missing for this QB')
        if n_missing:
            assumptions.append(f'{n_missing} feature(s) missing, imputed with the training median')
        tot = V11.predict_totals(art, X)
        meas = self.prod_meta['measured_backtest']['coverage_by_target']
        game_cap = float(sum(season_games_cap(int(p['draft_year']) + k) for k in range(CP.YEARS)))
        five = {}
        for t in CP.TARGETS:
            iv = tot[t]['intervals']
            cap = game_cap if t in ('games_played', 'games_started') else np.inf
            five[t] = {'label': 'predicted', 'stat': STAT_LABELS[t], 'expected': float(min(tot[t]['expected'][0], cap)),
                       **{f'interval_{int(lv * 100)}': [float(min(iv[lv][0][0], cap)), float(min(iv[lv][1][0], cap))]
                          for lv in V11.LEVELS},
                       'model': tot[t]['model'], 'interval_method': tot[t]['interval_method'],
                       'measured_coverage': meas.get(f'{mode}/{t}')}
            if t in self.hist_max:
                five[t]['training_max_5yr'] = self.hist_max[t]
                five[t]['upper_bound_beyond_training_max'] = bool(five[t]['interval_80'][1] > self.hist_max[t])
        if any(v.get('upper_bound_beyond_training_max') for v in five.values()):
            assumptions.append('some 80% upper bounds exceed the largest five-season total in the 183-QB training cohort; '
                               'treat them as "at least the historical maximum", not as precise values')
        if mode == 'pre_draft':
            assumptions.append('pre-draft five-season totals: the gain over the historical median is modest and not '
                               'statistically established (class-bootstrap intervals include zero); rely on the intervals')
        sp = V11.predict_start_probs(art, X)
        starts = {'label': 'predicted', 'p_16_plus': float(sp['nested'][0, 0]), 'p_32_plus': float(sp['nested'][0, 1]),
                  'p_64_plus': float(sp['nested'][0, 2]), 'window': 'NFL seasons 1-5 (regular season starts)',
                  'nested': True, 'method': 'L2 logistic per threshold, projected onto P(64+) <= P(32+) <= P(16+)',
                  'p_any_nfl_game': float(CP._prob(art['structured'].m_appear, X[art['features']])[0]),
                  'warnings': (['P(64+) pre-draft is poorly calibrated in backtests (slope about 0.24, 15 events): '
                                'treat it as a rough indication only'] if mode == 'pre_draft' else [])}
        out = {'status': 'available', 'mode': mode, 'hypothetical': hypothetical, 'assumptions': assumptions,
               'model_version': f'qb_career_production v{V11.MODEL_VERSION}', 'start_probabilities': starts,
               'five_year_totals': five,
               'efficiency_note': 'efficiency rates are not forecast: they were not predictable in backtests'}
        out['year_by_year'] = (self.year_block(art, X, {**five, '_draft_year': p['draft_year']}, n_sims) if year_by_year
                               else unavailable('not requested'))
        try:
            v2 = self.p5.v2
            out['reference_v2_p32'] = {'label': 'predicted', 'model_version': 'qb_career_success v2.0.0',
                                       'p_32_plus': float(v2['pipeline'].predict_proba(
                                           p['v2'].to_frame().T.reindex(columns=v2['features']).astype(float))[:, 1][0])}
        except Exception as e:      # noqa: BLE001 - reference only
            out['reference_v2_p32'] = unavailable(str(e))
        return out

    def year_block(self, art, X, five: dict, n_sims: int) -> dict:
        if not art.get('reference_tables_loaded'):
            return unavailable('the simulation reference tables are not in this checkout (gitignored player-level data); '
                               'regenerate with: python -m src.modeling.career_production_v11')
        sims = art['structured'].simulate(X[art['features']], n_sims=n_sims, seed=CP.SEED)[0]   # (sims, 5, stats)
        mean = sims.mean(0)
        lo, hi = np.quantile(sims, 0.1, axis=0), np.quantile(sims, 0.9, axis=0)
        years = np.zeros((CP.YEARS, len(CP.TARGETS)))
        ilo, ihi = np.zeros_like(years), np.zeros_like(years)
        for j, t in enumerate(CP.TARGETS):
            tot_sim = mean[:, j].sum()
            target = five[t]['expected']
            r = target / tot_sim if tot_sim > 0 else 0.0
            years[:, j] = mean[:, j] * r if tot_sim > 0 else target / CP.YEARS
            ilo[:, j], ihi[:, j] = lo[:, j] * r, hi[:, j] * r
        years = coherent_seasons(years)
        for j, t in enumerate(CP.TARGETS):
            if t in ('games_played', 'games_started'):
                caps = np.array([season_games_cap(int(five['_draft_year']) + k) for k in range(CP.YEARS)], float) \
                    if '_draft_year' in five else np.full(CP.YEARS, 17.0)
                ilo[:, j], ihi[:, j] = np.minimum(ilo[:, j], caps), np.minimum(ihi[:, j], caps)
        seasons = [{'season': y + 1, **{t: {'expected': float(years[y, j]), 'interval_80': [float(ilo[y, j]), float(ihi[y, j])]}
                                         for j, t in enumerate(CP.TARGETS)}} for y in range(CP.YEARS)]
        return {'status': 'available', 'label': 'predicted', 'seasons': seasons,
                'method': 'seasonal shape from the structured simulation, scaled so each statistic sums to the five-year total',
                'interval_note': 'yearly 80% ranges come from the simulation (scaled); not separately recalibrated per season'}

    # -------------------------------------------------------------- comparisons
    def comps_block(self, p: dict) -> dict:
        if self.pool is None:
            return unavailable(f'comparison pool not in this checkout: {self.pool_error}')
        pool = self.pool[self.pool['draft_year'] < p['draft_year']]
        if len(pool) < 20:
            return unavailable('too few earlier drafted QBs to compare with')
        c = PS.Comparator().fit(pool).query(p['v2'].copy(), PS.K)
        comps = []
        for r in c.itertuples():
            done = bool(r.window_5yr_complete)
            comps.append({'qb_draft_prospect_id': _i(r.qb_draft_prospect_id), 'name': r.player_name, 'college': r.college,
                          'draft_year': int(r.draft_year), 'round': _i(r.round),
                          'overall_pick': _i(r.overall_pick), 'distance': float(r.distance),
                          'closeness_index_heuristic': float(r.closeness_index_heuristic),
                          'most_similar_on': r.most_similar_on, 'biggest_differences': r.biggest_differences_comp_vs_query,
                          'nfl_outcome': {'label': 'observed',
                                          'starts_first_5_seasons': _i(r.nfl_starts_first_5_seasons) if done else None,
                                          'starts_to_date': None if done else _i(r.nfl_starts_first_5_seasons_to_date),
                                          'window_complete': done}})
        return {'status': 'available', 'kind': 'explanatory', 'model_version': 'qb_similarity v1.0.0', 'comps': comps,
                'note': 'nearest earlier drafted QBs by college profile (raw rates, z-scored on the pool). Distances and the '
                        'closeness index are heuristics, not probabilities; NFL outcomes never enter the distance.'}

    # -------------------------------------------------------------- assemble
    def forecast(self, mode='pre_draft', assumed_pick=None, n_sims=2000, year_by_year=True, **key) -> dict:
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        if assumed_pick is not None and not (1 <= int(assumed_pick) <= 262):
            raise ValueError('assumed_pick must be between 1 and 262')
        try:
            p = self.p5.profile(**{k: v for k, v in key.items() if v is not None})
        except (ValueError, IndexError, KeyError) as e:
            return {'schema_version': SCHEMA_VERSION, 'status': 'unavailable', 'reason': f'player not found: {e}', 'query': key}
        out = {'schema_version': SCHEMA_VERSION, 'status': 'available', 'mode': mode, 'player': self.identity(p),
               'draft': self.draft_block(p, mode), 'career': self.career_block(p, mode, assumed_pick, n_sims, year_by_year),
               'comparisons': self.comps_block(p)}
        out['uncertainty'] = {
            'model_versions': {'draft_probability': 'qb_draft v1.0.0', 'draft_round_pick': 'qb_draft v1.1.0',
                               'career': f'qb_career_production v{V11.MODEL_VERSION}', 'comparisons': 'qb_similarity v1.0.0',
                               'reference': 'qb_career_success v2.0.0'},
            'calibration_status': calibration_status(self.prod_meta, self.round_meta),
            'training_coverage': {'career_draft_classes': [2005, 2021], 'career_qbs': self.prod_meta['training_rows'],
                                  'draft_round_years': [2005, 2026], 'draft_round_qbs': self.round_meta['training_rows'],
                                  'draft_probability_population': 556},
            'data_limitations': DATA_LIMITATIONS}
        out['partial'] = [k for k in ('draft', 'career', 'comparisons') if out[k].get('status') != 'available']
        return out


def DP_round_from_pick(pick: int) -> int:
    """Approximate round for a hypothetical pick (32 picks per round plus compensatory picks)."""
    edges = [32, 64, 100, 138, 176, 220]
    return 1 + sum(pick > e for e in edges)


def coherent_seasons(y: np.ndarray) -> np.ndarray:
    """Per-season feasibility (starts <= games, completions/TDs/INTs <= attempts, rush TDs <= rushes) while keeping
    each statistic's five-season sum: excess in a violating season is moved from seasons with slack."""
    y = y.copy()
    ti = {t: i for i, t in enumerate(CP.TARGETS)}
    pairs = [('games_started', 'games_played'), ('pass_completions', 'pass_attempts'), ('pass_tds', 'pass_attempts'),
             ('interceptions', 'pass_attempts'), ('rush_tds', 'rush_attempts')]
    for small, big in pairs:
        a, b = ti[small], ti[big]
        for _ in range(3):
            over = np.maximum(y[:, a] - y[:, b], 0)
            if over.sum() <= 1e-9:
                break
            slack = np.maximum(y[:, b] - y[:, a], 0)
            if slack.sum() <= 1e-9:
                break
            move = min(over.sum(), slack.sum())
            y[:, a] = y[:, a] - over * (move / over.sum()) + slack * (move / slack.sum())
    return y


def calibration_status(prod_meta: dict, round_meta: dict) -> dict:
    cov = prod_meta['measured_backtest']['coverage_selected_intervals']
    return {'five_year_intervals': {fs: {f'nominal_{k[-2:]}': v for k, v in c.items()} for fs, c in cov.items()},
            'pick_intervals': round_meta['measured_backtest']['pick_coverage_groupkfold'],
            'start_probabilities': 'see reports/calibration_improvements.md (slopes, intercepts, reliability by threshold)',
            'note': 'coverage measured on later draft classes in nested chronological backtests; not guaranteed'}


@lru_cache(maxsize=1)
def engine() -> Engine:
    return Engine()


def forecast_prospect(name: str | None = None, draft_year: int | None = None, mode: str = 'pre_draft', *,
                      cfbd_player_id: str | None = None, final_season: int | None = None,
                      qb_draft_prospect_id: int | None = None, assumed_pick: int | None = None,
                      n_sims: int = 2000, year_by_year: bool = True) -> dict:
    key = {'name': name, 'draft_year': draft_year, 'cfbd_player_id': cfbd_player_id, 'final_season': final_season,
           'qb_draft_prospect_id': qb_draft_prospect_id}
    return engine().forecast(mode=mode, assumed_pick=assumed_pick, n_sims=n_sims, year_by_year=year_by_year, **key)


# ------------------------------------------------------------------ schema validation (also used by the tests)
REQUIRED = {
    '': ['schema_version', 'status', 'mode', 'player', 'draft', 'career', 'comparisons', 'uncertainty', 'partial'],
    'player': ['name', 'college', 'draft_year', 'identifiers', 'identity_verification', 'actual_draft'],
    'uncertainty': ['model_versions', 'calibration_status', 'training_coverage', 'data_limitations'],
}


def validate(f: dict) -> list[str]:
    """Problems with a forecast object (empty list = valid)."""
    errs = []
    if f.get('status') == 'unavailable':
        return [] if 'reason' in f else ['unavailable forecast without a reason']
    for path, keys in REQUIRED.items():
        node = f if not path else f.get(path, {})
        errs += [f'missing {path + "." if path else ""}{k}' for k in keys if k not in node]
    for blk in ('draft', 'career', 'comparisons'):
        b = f.get(blk, {})
        if b.get('status') not in ('available', 'unavailable', 'partial'):
            errs.append(f'{blk}.status invalid')
        if b.get('status') == 'unavailable' and not b.get('reason'):
            errs.append(f'{blk} unavailable without a reason')
    d = f.get('draft', {})
    if d.get('status') == 'available':
        for sub in ('probability', 'round', 'pick'):
            if d[sub].get('status') == 'unavailable' and not d[sub].get('reason'):
                errs.append(f'draft.{sub} unavailable without a reason')
        r = d['round']
        if abs(sum(r['groups'].values()) - 1) > 1e-6 or abs(sum(r['rounds'].values()) - 1) > 1e-6:
            errs.append('round probabilities do not sum to 1')
        pk = d['pick']
        if not (1 <= pk['interval_90'][0] <= pk['interval_80'][0] <= pk['interval_50'][0] <= pk['median']
                <= pk['interval_50'][1] <= pk['interval_80'][1] <= pk['interval_90'][1] <= 262):
            errs.append('pick intervals not nested')
    c = f.get('career', {})
    if c.get('status') == 'available':
        s = c['start_probabilities']
        if not (0 <= s['p_64_plus'] <= s['p_32_plus'] <= s['p_16_plus'] <= 1):
            errs.append('start probabilities not nested')
        for t, v in c['five_year_totals'].items():
            if not (v['interval_50'][0] >= v['interval_80'][0] - 1e-9 >= v['interval_90'][0] - 2e-9 and
                    v['interval_50'][1] <= v['interval_80'][1] + 1e-9 <= v['interval_90'][1] + 2e-9):
                errs.append(f'{t} intervals not nested')
        ft = c['five_year_totals']
        if ft['games_started']['expected'] > ft['games_played']['expected'] + 1e-9:
            errs.append('expected starts exceed games played')
        if ft['pass_completions']['expected'] > ft['pass_attempts']['expected'] + 1e-9:
            errs.append('expected completions exceed attempts')
        y = c.get('year_by_year', {})
        if y.get('status') == 'available':
            for t in CP.TARGETS:
                s5 = sum(season[t]['expected'] for season in y['seasons'])
                if abs(s5 - ft[t]['expected']) > 1e-6 * max(1, abs(ft[t]['expected'])):
                    errs.append(f'{t} seasons do not sum to the five-year total')
    try:
        json.dumps(f)
    except (TypeError, ValueError) as e:
        errs.append(f'not JSON-serialisable: {e}')
    return errs


def summarize(f: dict) -> str:
    if f.get('status') == 'unavailable':
        return f'unavailable: {f["reason"]}'
    pl, d, c = f['player'], f['draft'], f['career']
    L = [f'{pl["name"]} ({pl["college"]}), {pl["draft_year"]} draft class [{f["mode"]}]'
         + (f'; actual pick {pl["actual_draft"]["overall_pick"]}' if pl['actual_draft']['overall_pick'] else '')]
    if d.get('status') == 'available':
        pr = d['probability']
        L.append('  P(drafted): ' + (f'pre-Combine {pr["pre_combine"]:.2f}' + (f', post-Combine {pr["post_combine"]:.2f}' if pr['post_combine'] is not None else '')
                                     if pr['status'] == 'available' else f'unavailable ({pr["reason"]})'))
        g = d['round']['groups']
        L.append('  If drafted: ' + ', '.join(f'{k} {v:.2f}' for k, v in g.items()) +
                 f'; pick ~{d["pick"]["median"]:.0f} (80%: {d["pick"]["interval_80"][0]:.0f}-{d["pick"]["interval_80"][1]:.0f})')
    if c.get('status') == 'available':
        s, t = c['start_probabilities'], c['five_year_totals']
        L.append(f'  P(16/32/64+ starts, seasons 1-5) {s["p_16_plus"]:.2f}/{s["p_32_plus"]:.2f}/{s["p_64_plus"]:.2f}')
        for k in ('games_started', 'pass_yards', 'pass_tds', 'interceptions', 'rush_yards', 'rush_tds'):
            L.append(f'  {STAT_LABELS[k]}: {t[k]["expected"]:.0f} (80%: {t[k]["interval_80"][0]:.0f}-{t[k]["interval_80"][1]:.0f}) [{t[k]["model"]}]')
    else:
        L.append(f'  Career: unavailable ({c.get("reason")})')
    if f['comparisons'].get('status') == 'available':
        L.append('  Comps: ' + '; '.join(f'{x["name"]} ({x["draft_year"]})' for x in f['comparisons']['comps']))
    return '\n'.join(L)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name')
    ap.add_argument('--draft-year', type=int)
    ap.add_argument('--cfbd-id')
    ap.add_argument('--final-season', type=int)
    ap.add_argument('--mode', default='pre_draft', choices=MODES)
    ap.add_argument('--assumed-pick', type=int)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    f = forecast_prospect(a.name, a.draft_year, a.mode, cfbd_player_id=a.cfbd_id, final_season=a.final_season,
                          assumed_pick=a.assumed_pick)
    print(json.dumps(f, indent=2) if a.json else summarize(f))
