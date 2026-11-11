"""The only place the API touches the modeling code. Loads the serving bundle and the forecasting engine once
(lazily, thread-safe), verifies them, and turns engine output into the API schema. No retraining, no
network access, no SQL: every request is answered from verified local artifacts."""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd

from api import bundle as B
from api import errors as E
from api.config import API_VERSION, Settings

log = logging.getLogger('qb_api.prediction')
PLAYER_ID_RE = re.compile(r'^(qb_\d{1,6}|cfb_\d{1,12}_\d{4})$')
STAT_ORDER = ['games_played', 'games_started', 'pass_attempts', 'pass_completions', 'pass_yards', 'pass_tds',
              'interceptions', 'rush_attempts', 'rush_yards', 'rush_tds']


def clean(x: Any) -> Any:
    """Recursively convert to JSON-safe Python types: NumPy scalars -> Python, NaN/inf -> None."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if x is None or isinstance(x, (str, int)):
        return x
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return str(x)


def check_player_id(pid: str) -> str:
    if not isinstance(pid, str) or not PLAYER_ID_RE.match(pid):
        raise E.APIError(E.INVALID_PARAMETER, 'player_id must look like qb_<number> or cfb_<number>_<season>')
    return pid


def _ivs(d: dict, levels=(0.5, 0.8, 0.9)) -> list[dict]:
    out = []
    for lv in levels:
        k = f'interval_{int(lv * 100)}'
        if d.get(k) is not None:
            out.append({'level': lv, 'lower': d[k][0], 'upper': d[k][1]})
    return out


class PredictionService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._loaded = False
        self._engine = None
        self._source = None
        self._backtests = None
        self._cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        self.load_error: str | None = None
        self.manifest: dict | None = None

    # -------------------------------------------------------------- loading
    @property
    def bundle_dir(self):
        return self.settings.bundle_dir

    def bundle_present(self) -> bool:
        return (self.bundle_dir / B.MANIFEST).exists()

    def _load_index(self):
        """Prospect index, details and season lines (small); verified against the bundle manifest once."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if not self.bundle_present():
                self.load_error = 'serving bundle not installed'
                raise E.APIError(E.MODEL_UNAVAILABLE, 'Prediction data is not installed on this server.')
            bad = [r['file'] for r in B.verify(B.ROOT, self.bundle_dir) if not (r['exists'] and r['sha256_ok'])]
            if bad:
                self.load_error = f'{len(bad)} bundle/model files failed verification'
                log.error('artifact verification failed for %d files', len(bad))
                raise E.APIError(E.MODEL_UNAVAILABLE, 'Prediction artifacts failed integrity verification.')
            self.manifest = B.load_manifest(self.bundle_dir)
            P = pd.read_parquet(self.bundle_dir / B.PROSPECTS)
            P['shares_name'] = P['name_key'].duplicated(keep=False)
            self.prospects = P.set_index('player_id', drop=False)
            self.details = json.loads((self.bundle_dir / B.DETAILS).read_text(encoding='utf-8'))
            self.seasons = pd.read_parquet(self.bundle_dir / B.SEASONS)
            self.cache_key = self.manifest['created_utc'] + '|' + json.dumps(
                {k: v['artifact_sha256'] for k, v in self.manifest['models'].items()}, sort_keys=True)
            self._loaded = True
            log.info('serving index loaded: %d prospects', len(P))

    def engine(self):
        self._load_index()
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    t = time.time()
                    from src.modeling.prediction_api import Engine
                    src = B.BundleSource(self.bundle_dir)
                    self._source = src
                    self._engine = Engine(source=src, combine_seasons=src.combine_seasons, hist_max=src.hist_max)
                    log.info('forecast engine loaded in %.1fs', time.time() - t)
        return self._engine

    @property
    def engine_loaded(self) -> bool:
        return self._engine is not None

    # -------------------------------------------------------------- prospects
    def summary(self, row) -> dict:
        return clean({'player_id': row['player_id'], 'name': row['name'], 'college': row['college'],
                      'draft_year': int(row['draft_year']), 'final_college_season': row['final_college_season'],
                      'drafted': None if pd.isna(row['drafted']) else bool(row['drafted']), 'draft_round': row['draft_round'],
                      'overall_pick': row['overall_pick'], 'kind': row['kind'],
                      'shares_name_with_other_prospects': bool(row['shares_name'])})

    def _row(self, pid: str):
        self._load_index()
        check_player_id(pid)
        if pid not in self.prospects.index:
            raise E.APIError(E.PROSPECT_NOT_FOUND, f'No prospect with id {pid}.')
        return self.prospects.loc[pid]

    def search(self, q: str | None, draft_year: int | None, limit: int, offset: int, drafted_only: bool = False) -> dict:
        self._load_index()
        P = self.prospects
        key = B.normalize(q) if q else ''
        if key:
            toks = key.split()
            m = np.ones(len(P), bool)
            for t in toks:
                m &= P['name_key'].str.contains(re.escape(t), regex=True).to_numpy()
            P = P[m]
        if draft_year is not None:
            P = P[P['draft_year'] == draft_year]
        if drafted_only:
            P = P[P['kind'] == 'drafted']
        if key:
            rank = np.where(P['name_key'] == key, 0, np.where(P['name_key'].str.startswith(key), 1, 2))
            P = P.assign(_r=rank).sort_values(['_r', 'draft_year', 'name'], ascending=[True, False, True])
        else:
            P = P.sort_values(['draft_year', 'name'], ascending=[False, True])
        total = len(P)
        page = P.iloc[offset: offset + limit]
        return {'query': q, 'draft_year': draft_year, 'total': int(total), 'limit': limit, 'offset': offset,
                'items': [self.summary(r) for _, r in page.iterrows()]}

    def resolve(self, name: str, draft_year: int | None) -> dict:
        self._load_index()
        key = B.normalize(name)
        P = self.prospects[self.prospects['name_key'] == key]
        if draft_year is not None:
            P = P[P['draft_year'] == draft_year]
        if P.empty:
            raise E.APIError(E.PROSPECT_NOT_FOUND, 'No prospect with that name' + (f' in the {draft_year} class.' if draft_year else '.'))
        if len(P) > 1:
            raise E.APIError(E.AMBIGUOUS_PLAYER, f'{len(P)} prospects share that name; choose one by player_id.',
                             {'candidates': [self.summary(r) for _, r in P.iterrows()]})
        return self.summary(P.iloc[0])

    def detail(self, pid: str) -> dict:
        row = self._row(pid)
        d = self.details.get(pid, {})
        cid = row['cfbd_player_id']
        seasons = []
        if isinstance(cid, str) and pd.notna(row['final_college_season']):
            s = self.seasons[(self.seasons['cfbd_player_id'] == cid) & (self.seasons['season'] <= row['final_college_season'])]
            seasons = [clean(r) for r in s.sort_values('season').rename(columns={'team_first': 'team'}).to_dict('records')]
            for r in seasons:
                r.pop('cfbd_player_id', None)
        gsis = None
        if pd.notna(row['qb_draft_prospect_id']):
            g = self._prospects_gsis()
            gsis = g.get(int(row['qb_draft_prospect_id']))
        drafted = None if pd.isna(row['drafted']) else bool(row['drafted'])
        return clean({
            'prospect': self.summary(row),
            'identifiers': {'qb_draft_prospect_id': row['qb_draft_prospect_id'], 'cfbd_player_id': cid, 'gsis_id': gsis},
            'draft_status': {'label': 'observed', 'drafted': drafted, 'round': row['draft_round'], 'overall_pick': row['overall_pick'],
                             'note': ('not yet verifiable: no later college season exists in the data to confirm this was the '
                                      'final season' if drafted is None else None)},
            'final_season_statistics': d.get('final_season_statistics', {}), 'college_seasons': seasons,
            'offensive_profile': d.get('offensive_profile', {}), 'athletic_measurements': d.get('athletic_measurements', {}),
            'combine_participant': d.get('combine_participant'), 'data_coverage': d.get('data_coverage', {})})

    def _prospects_gsis(self) -> dict:
        if not hasattr(self, '_gsis'):
            import joblib
            g = joblib.load(self.bundle_dir / B.PROFILES)['prospects_gsis']
            self._gsis = {int(i): (str(v) if pd.notna(v) else None) for i, v in zip(g['id'], g['gsis_id'])}
        return self._gsis

    # -------------------------------------------------------------- forecasts
    def forecast(self, pid: str, mode: str, assumed_pick: int | None) -> dict:
        if mode not in ('pre_draft', 'post_draft'):
            raise E.APIError(E.INVALID_PREDICTION_MODE, "mode must be 'pre_draft' or 'post_draft'")
        if assumed_pick is not None and mode != 'post_draft':
            raise E.APIError(E.INVALID_PARAMETER, 'assumed_pick is only allowed with mode=post_draft')
        self._row(pid)
        key = (pid, mode, assumed_pick, self.cache_key)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        eng = self.engine()
        prof_key = self._profile_key(pid)
        f = eng.forecast(mode=mode, assumed_pick=assumed_pick, n_sims=self.settings.n_sims, **prof_key)
        if f.get('status') != 'available':
            raise E.APIError(E.PROSPECT_NOT_FOUND, 'The forecasting engine has no profile for this prospect.')
        out = clean(self.to_schema(pid, f))
        if self.settings.cache_size:
            with self._cache_lock:
                self._cache[key] = out
                while len(self._cache) > self.settings.cache_size:
                    self._cache.popitem(last=False)
        return out

    def _profile_key(self, pid: str) -> dict:
        if pid.startswith('qb_'):
            return {'qb_draft_prospect_id': int(pid[3:])}
        _, cid, season = pid.split('_')
        return {'cfbd_player_id': cid, 'final_season': int(season)}

    @staticmethod
    def to_schema(pid: str, f: dict) -> dict:
        mode = f['mode']
        pl, d, c = f['player'], f['draft'], f['career']
        warnings = []
        player = {'player_id': pid, 'name': pl['name'], 'college': pl['college'], 'draft_year': pl['draft_year'],
                  'final_college_season': pl['final_college_season'], 'identifiers': pl['identifiers'],
                  'identity_verification': pl['identity_verification'],
                  'actual_draft': ({k: pl['actual_draft'][k] for k in ('drafted', 'round', 'overall_pick')}
                                   if mode == 'post_draft' else None)}
        # draft
        if d.get('status') == 'available':
            pr = d['probability']
            prob = ({'status': 'unavailable', 'reason': pr['reason'], 'code': E.PREDICTION_UNSUPPORTED} if pr['status'] != 'available' else
                    {k: pr.get(k) for k in ('status', 'pre_combine', 'post_combine', 'post_combine_note', 'population', 'model_version', 'backtest')})
            if pr['status'] != 'available':
                warnings.append({'code': 'DRAFT_PROBABILITY_UNAVAILABLE', 'message': pr['reason']})
            elif pr.get('post_combine') is not None:
                warnings.append({'code': 'COMBINE_PARTICIPATION', 'message': 'post-Combine draft probability mostly reflects '
                                 'Combine participation (an NFL scouting decision), not athletic measurements'})
            r, k = d['round'], d['pick']
            draft = {'status': 'available', 'probability': prob,
                     'round': {'status': 'available', 'conditional_on': r['conditional_on'], 'groups': r['groups'], 'rounds': r['rounds'],
                               'expected_round': r['expected_round'], 'display_note': r['display_note'],
                               'model_version': r['model_version'], 'backtest_rps': r['backtest_rps']},
                     'pick': {'status': 'available', 'conditional_on': k['conditional_on'], 'median': k['median'], 'intervals': _ivs(k),
                              'measured_coverage': k['measured_coverage'], 'model_version': k['model_version']},
                     'eligibility': d['eligibility']}
        else:
            draft = {'status': 'unavailable', 'reason': d.get('reason'), 'code': E.INSUFFICIENT_DATA}
        # career
        if c.get('status') == 'available':
            s = c['start_probabilities']
            if not (0 <= s['p_64_plus'] <= s['p_32_plus'] <= s['p_16_plus'] <= 1):
                raise E.APIError(E.INTERNAL_ERROR, 'start probabilities failed the nesting check')
            five = []
            for t in STAT_ORDER:
                v = c['five_year_totals'][t]
                five.append({'stat': t, 'label': v['stat'], 'expected': v['expected'], 'intervals': _ivs(v), 'model': v['model'],
                             'interval_method': v['interval_method'], 'measured_coverage': v.get('measured_coverage'),
                             'training_max_5yr': v.get('training_max_5yr'),
                             'upper_bound_beyond_training_max': v.get('upper_bound_beyond_training_max')})
            y = c['year_by_year']
            yby = ({'status': 'available', 'method': y['method'], 'interval_note': y['interval_note'],
                    'seasons': [{'season': ss['season'], 'stats': {t: {'expected': ss[t]['expected'],
                                                                         'interval_80': {'level': 0.8, 'lower': ss[t]['interval_80'][0],
                                                                                         'upper': ss[t]['interval_80'][1]}}
                                                                    for t in STAT_ORDER}} for ss in y['seasons']]}
                   if y.get('status') == 'available' else {'status': 'unavailable', 'reason': y.get('reason')})
            ref = c.get('reference_v2_p32', {})
            career = {'status': 'available', 'hypothetical': c['hypothetical'], 'assumptions': c['assumptions'],
                      'model_version': c['model_version'],
                      'start_probabilities': {'status': 'available', **{k: s[k] for k in ('p_16_plus', 'p_32_plus', 'p_64_plus',
                                                                                           'p_any_nfl_game', 'window', 'method',
                                                                                           'nested', 'warnings')}},
                      'five_year_totals': five, 'year_by_year': yby,
                      'reference_v2': ({'status': 'available', 'model_version': ref['model_version'], 'p_32_plus': ref['p_32_plus']}
                                       if ref.get('label') == 'predicted' else {'status': 'unavailable', 'reason': ref.get('reason')}),
                      'efficiency_note': c.get('efficiency_note')}
            for w in s.get('warnings', []):
                warnings.append({'code': 'CALIBRATION_WARNING', 'message': w})
            beyond = [x['label'] for x in five if x['upper_bound_beyond_training_max']]
            if beyond:
                warnings.append({'code': 'EXTRAPOLATION', 'message': '80% upper bounds exceed the largest five-season total in the '
                                 'training cohort for: ' + ', '.join(beyond) + '. Read them as "at least the historical maximum".'})
            for a in c['assumptions']:
                if a.startswith('some 80% upper bounds'):
                    continue                                   # reported once, as the EXTRAPOLATION warning above
                warnings.append({'code': 'HYPOTHETICAL_PICK' if a.startswith('hypothetical') else 'ASSUMPTION', 'message': a})
        else:
            reason = c.get('reason', '')
            code = E.INSUFFICIENT_DATA if 'features are missing' in reason else E.PREDICTION_UNSUPPORTED
            career = {'status': 'unavailable', 'reason': reason, 'code': code}
        unavailable = [k for k, b in (('draft', draft), ('career', career)) if b['status'] != 'available']
        if draft.get('probability', {}).get('status') == 'unavailable':
            unavailable.append('draft.probability')
        if career.get('year_by_year', {}).get('status') == 'unavailable':
            unavailable.append('career.year_by_year')
        return {'api_version': API_VERSION, 'schema_version': f['schema_version'], 'mode': mode, 'player': player,
                'draft': draft, 'career': career, 'warnings': warnings, 'unavailable_blocks': unavailable,
                'uncertainty': f['uncertainty']}

    # -------------------------------------------------------------- comparisons
    def comparisons(self, pid: str) -> dict:
        self._row(pid)
        eng = self.engine()
        prof = self._source.profile(**self._profile_key(pid))
        c = eng.comps_block(prof)
        if c.get('status') != 'available':
            return clean({'player_id': pid, 'status': 'unavailable', 'reason': c.get('reason')})
        comps = [{'player_id': f'qb_{x["qb_draft_prospect_id"]}' if x.get('qb_draft_prospect_id') is not None else None,
                  **{k: x[k] for k in ('name', 'college', 'draft_year', 'round', 'overall_pick', 'distance',
                                       'closeness_index_heuristic', 'most_similar_on', 'biggest_differences')},
                  'nfl_outcome': {k: x['nfl_outcome'][k] for k in ('label', 'starts_first_5_seasons', 'starts_to_date', 'window_complete')}}
                 for x in c['comps']]
        return clean({'player_id': pid, 'status': 'available', 'kind': c['kind'], 'model_version': c['model_version'],
                      'note': c['note'] + ' Outcomes are as observed through the 2025 NFL season.', 'comparables': comps})

    # -------------------------------------------------------------- backtests
    def backtest(self, pid: str) -> dict:
        row = self._row(pid)
        if row['kind'] != 'drafted':
            raise E.APIError(E.PREDICTION_UNSUPPORTED, 'Historical backtests exist only for drafted QBs with an NFL draft record.')
        if self._backtests is None:
            with self._lock:
                if self._backtests is None:
                    self._backtests = json.loads((self.bundle_dir / B.BACKTESTS).read_text(encoding='utf-8'))
        b = self._backtests.get(pid)
        if not b or b.get('status') != 'available':
            raise E.APIError(E.PREDICTION_UNSUPPORTED, (b or {}).get('reason', 'No historical backtest is available for this QB.'))
        views = {}
        for fs, v in b['views'].items():
            views[fs] = {'source': v['source'],
                         'statistics': [{'stat': t, 'predicted': x['predicted'],
                                         'interval_80': {'level': 0.8, 'lower': x['interval_80'][0], 'upper': x['interval_80'][1]},
                                         'actual': x['actual'], 'actual_label': x['actual_label'], 'error': x['error'],
                                         'inside_80': x['inside_80'], 'model': x['model']} for t, x in v['comparison'].items()],
                         'start_probabilities': [{'threshold': int(k), 'predicted': x['predicted'], 'actual_event': x['actual_event']}
                                                 for k, x in v['start_probabilities'].items()]}
        why = 'the career models forecast five-season totals; season-level forecasts were not backtested'
        return clean({'player_id': pid, 'player': b['player'], 'classification': b['validity'],
                      'prediction_cutoff': b['prediction_date'], 'training_classes': b['training_classes'],
                      'n_training_qbs': b['n_training_qbs'], 'label_embargo_years': b['label_embargo_years'],
                      'model_version': b['model_version'], 'actual_window_complete': b['actual_window_complete'],
                      'observed_seasons': b['observed_seasons'],
                      'horizons': {'five_year': {'status': 'available'},
                                   'one_year': {'status': 'unavailable', 'reason': why},
                                   'three_year': {'status': 'unavailable', 'reason': why}},
                      'views': views})
