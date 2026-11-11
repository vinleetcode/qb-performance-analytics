"""Serving bundle: the prediction-ready data the API needs, without the training database.

    python -m api.bundle build      # from the full local data (once per data/model version)
    python -m api.bundle verify     # SHA-256 of every bundle file and every model file it depends on
    python -m api.bundle pack       # dist/qb_serving_bundle_v<version>.tar.gz for deployment
    python -m api.bundle install dist/qb_serving_bundle_v1.tar.gz   # on the server (safe extraction)

The bundle holds, for every servable prospect, the exact profile objects the forecasting engine
consumes (so the API runs the unchanged `prediction_api.Engine` code), precomputed historical
backtests, college season lines for the details page, and two existing model data files (the
simulation reference tables and the comparison pool). It is player-level data derived from CFBD and
nflverse, so it is gitignored like everything under data/: build it locally and ship it with the
deployment, never commit it. Model files (models/**) are tracked and are verified against the
hashes recorded in the manifest.

Building requires the full local data; serving requires only the bundle + tracked model files.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_VERSION = '1'
BUNDLE_DIR = ROOT / 'data' / 'serving' / f'v{BUNDLE_VERSION}'
DIST = ROOT / 'dist'
PROFILES = 'profiles.joblib'
PROSPECTS = 'prospects.parquet'
SEASONS = 'college_seasons.parquet'
BACKTESTS = 'backtests.json'
DETAILS = 'details.json'
MANIFEST = 'manifest.json'
# existing gitignored model data files the engine reads, shipped at their own repo-relative paths
MODEL_DATA_FILES = ['data/model_artifacts/qb_career_production/v1.1.0/reference_tables.npz',
                    'data/model_artifacts/qb_similarity/v1.0.0/pool.parquet']
MODEL_DIRS = ['models/qb_career_production/v1.1.0', 'models/qb_draft/v1.0.0', 'models/qb_draft/v1.1.0',
              'models/qb_similarity/v1.0.0', 'models/qb_career_success/v2.0.0']
PACK_PREFIXES = ('data/serving/', 'data/model_artifacts/')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def player_id_drafted(qb_draft_prospect_id) -> str:
    return f'qb_{int(qb_draft_prospect_id)}'


def player_id_college(cfbd_player_id, season) -> str:
    return f'cfb_{cfbd_player_id}_{int(season)}'


def normalize(name) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize('NFKD', str(name or '')).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _clean(v):
    """JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v if isinstance(v, (str, int)) else str(v)


# ------------------------------------------------------------------ details (what the prospect page shows)
COLLEGE_FINAL = {'final_attempts': 'attempts', 'completion_pct': 'completion_pct', 'yards_per_attempt': 'yards_per_attempt',
                 'td_pct': 'td_pct', 'int_pct': 'int_pct', 'attempts_per_game': 'attempts_per_game',
                 'opp_adj_ypa': 'opponent_adjusted_ypa', 'sos_pass_defense': 'opponent_pass_defense_ypa',
                 'rush_ypg_last3': 'rush_yards_per_game_last3', 'n_seasons_100att': 'seasons_with_100_attempts',
                 'ypa_change_vs_prior': 'ypa_change_vs_prior_season'}
OFFENSE = {'env_pass_rate': 'team_pass_rate', 'env_plays_per_game': 'team_plays_per_game',
           'env_sp_offense_rating': 'team_sp_offense_rating', 'talent_recruiting_4yr_pctile': 'team_recruiting_percentile',
           'scheme_rep_final_spread_family': 'reported_spread_family_offense'}
ATHLETIC = {'height_inches': ('height_inches', 'college roster'), 'weight_lbs': ('weight_lbs', 'college roster'),
            'combine_height_inches': ('combine_height_inches', 'NFL Combine'), 'combine_weight_lbs': ('combine_weight_lbs', 'NFL Combine'),
            'combine_forty': ('combine_forty_seconds', 'NFL Combine')}
SEASON_COLS = ['season', 'team_first', 'conference', 'is_fbs', 'attempts', 'completions', 'passing_yards', 'passing_tds',
               'interceptions', 'rushing_attempts', 'rushing_yards', 'completion_pct', 'yards_per_attempt', 'td_pct', 'int_pct',
               'opp_adj_ypa']


def build_details(pid: str, prof: dict, career_features: list[str], draft_features: list[str]) -> dict:
    v, f = prof['v2'], prof['draft']
    has_f = f is not None and len(f) > 0
    final = {out: _clean(f.get(src)) for src, out in COLLEGE_FINAL.items()} if has_f else {}
    off = {out: _clean(v.get(src)) for src, out in OFFENSE.items()}
    if has_f:
        off['power_conference'] = _clean(f.get('power_conference'))
        off['conference'] = _clean(f.get('conference'))
        off['team'] = _clean(f.get('team'))
    ath = {}
    for src, (out, source) in ATHLETIC.items():
        val = _clean(f.get(src)) if has_f else None
        if val is None and src in ('height_inches',):
            val = _clean(v.get('height_inches'))
        ath[out] = {'value': val, 'source': source}
    combine = _clean(f.get('combine_invited')) if has_f else None
    present = [c for c in career_features if _clean(v.get(c)) is not None]
    dpresent = [c for c in draft_features if has_f and _clean(f.get(c)) is not None]
    return {'player_id': pid, 'final_season_statistics': final, 'offensive_profile': off, 'athletic_measurements': ath,
            'combine_participant': None if combine is None else bool(combine),
            'data_coverage': {'career_model_features_present': len(present), 'career_model_features_total': len(career_features),
                              'career_model_features_missing': [c for c in career_features if c not in present],
                              'draft_model_features_present': len(dpresent), 'draft_model_features_total': len(draft_features),
                              'has_college_passer_season': bool(has_f),
                              'population_status': _clean(f.get('population_status')) if has_f else None}}


# ------------------------------------------------------------------ build
def _backtest_one(qid: int) -> tuple[int, dict]:
    from src.modeling.historical_backtest import historical_backtest
    return qid, historical_backtest(qb_draft_prospect_id=qid)


def build(n_jobs: int = -1) -> dict:
    from src.modeling import artifacts as A
    from src.modeling import career_production as CP
    from src.modeling import draft_predictor as DP
    from src.modeling import prediction_api as PA
    from src.modeling.forecast import Engine as P5Engine
    t0 = time.time()
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    p5 = P5Engine()
    src = p5.src()
    v2 = p5.v2_data
    ident = src['identity']
    drafted_cfbd = set(ident['cfbd_player_id'].dropna())
    profiles, rows = {}, []
    for qid in v2['qb_draft_prospect_id'].astype(int):
        prof = p5.profile(qb_draft_prospect_id=qid)
        pid = player_id_drafted(qid)
        profiles[pid] = prof
        idr = ident[ident['qb_draft_prospect_id'] == qid]
        f = prof['draft']
        rows.append({'player_id': pid, 'kind': 'drafted', 'name': prof['name'], 'college': prof['college'],
                     'draft_year': int(prof['draft_year']),
                     'final_college_season': int(f['season']) if len(f) and pd.notna(f.get('season')) else None,
                     'qb_draft_prospect_id': qid, 'cfbd_player_id': idr['cfbd_player_id'].iloc[0] if len(idr) else None,
                     'drafted': True, 'draft_round': _clean(prof['actual_round']), 'overall_pick': _clean(prof['actual_pick'])})
    u = p5.universe
    for r in u.itertuples(index=False):
        if r.cfbd_player_id in drafted_cfbd:
            continue                                  # the same person is already listed as a drafted QB
        try:
            prof = p5.profile(cfbd_player_id=r.cfbd_player_id, final_season=int(r.season))
        except Exception:                             # noqa: BLE001 - skip rows the engine cannot profile
            continue
        pid = player_id_college(r.cfbd_player_id, r.season)
        profiles[pid] = prof
        rows.append({'player_id': pid, 'kind': 'college_only', 'name': prof['name'], 'college': prof['college'],
                     'draft_year': int(prof['draft_year']), 'final_college_season': int(r.season), 'qb_draft_prospect_id': None,
                     'cfbd_player_id': r.cfbd_player_id, 'drafted': bool(r.drafted) if r.population_status != 'final_season_not_verifiable_yet' else None,
                     'draft_round': _clean(r.round) if r.drafted else None, 'overall_pick': _clean(r.overall) if r.drafted else None})
    P = pd.DataFrame(rows)
    P['name_key'] = P['name'].map(normalize)
    P['cfbd_player_id'] = P['cfbd_player_id'].astype('string')
    assert P['player_id'].is_unique
    # details
    eng_feats = joblib.load(ROOT / 'models' / 'qb_career_production' / 'v1.1.0' / 'pre_draft.joblib')['features']
    dfeat = DP.COLLEGE_FEATURES + DP.COMBINE_FEATURES
    details = {pid: build_details(pid, prof, eng_feats, dfeat) for pid, prof in profiles.items()}
    # college season lines
    ref = src['ref']
    S = ref[ref['cfbd_player_id'].isin(set(P['cfbd_player_id'].dropna()))][['cfbd_player_id'] + SEASON_COLS].copy()
    S['cfbd_player_id'] = S['cfbd_player_id'].astype('string')
    # backtests (drafted QBs only), parallel; fold fits are cached per class inside each worker
    from joblib import Parallel, delayed
    qids = P.loc[P['kind'] == 'drafted', 'qb_draft_prospect_id'].astype(int).tolist()
    bt = dict(Parallel(n_jobs=n_jobs, verbose=2)(delayed(_backtest_one)(q) for q in qids))
    bt = {player_id_drafted(q): b for q, b in bt.items()}
    gsis = src['prospects'][['id', 'gsis_id']].copy()
    d, _ = CP.load_data()
    serving = {'profiles': profiles, 'prospects_gsis': gsis,
               'combine_seasons': sorted(int(x) for x in pd.read_parquet(ROOT / 'data' / 'nflverse_cache' / 'combine_2026-11-09.parquet')['season'].unique()),
               'hist_max': {t: float(d[f'{t}_5yr'].max()) for t in CP.TARGETS}}
    joblib.dump(serving, BUNDLE_DIR / PROFILES, compress=3)
    P.to_parquet(BUNDLE_DIR / PROSPECTS, index=False)
    S.to_parquet(BUNDLE_DIR / SEASONS, index=False)
    (BUNDLE_DIR / DETAILS).write_text(json.dumps(details, default=_clean), encoding='utf-8')
    (BUNDLE_DIR / BACKTESTS).write_text(json.dumps(bt, default=_clean), encoding='utf-8')
    files = {f'data/serving/v{BUNDLE_VERSION}/{n}': sha256(BUNDLE_DIR / n) for n in (PROFILES, PROSPECTS, SEASONS, DETAILS, BACKTESTS)}
    files.update({p: sha256(ROOT / p) for p in MODEL_DATA_FILES})
    models = {}
    for md in MODEL_DIRS:
        meta = A.load_metadata(ROOT / md)
        models[md] = {'metadata_sha256': sha256(ROOT / md / 'metadata.json'), 'artifact_sha256': meta.get('artifact_sha256', {})}
    manifest = {'bundle_version': BUNDLE_VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                'built_by': 'python -m api.bundle build', 'files': files, 'models': models,
                'counts': {'prospects': int(len(P)), 'drafted': int((P['kind'] == 'drafted').sum()),
                           'college_only': int((P['kind'] == 'college_only').sum()), 'backtests': len(bt),
                           'college_season_rows': int(len(S))},
                'contents_note': 'player-level data derived from CFBD and nflverse; not for redistribution via Git',
                'build_seconds': round(time.time() - t0, 1)}
    (BUNDLE_DIR / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


# ------------------------------------------------------------------ verify / pack / install
def load_manifest(bundle_dir: Path = BUNDLE_DIR) -> dict:
    return json.loads((bundle_dir / MANIFEST).read_text(encoding='utf-8'))


def verify(root: Path = ROOT, bundle_dir: Path = BUNDLE_DIR) -> list[dict]:
    """Every bundle file and every model file it was built against, against the manifest."""
    m = load_manifest(bundle_dir)
    out = []
    for rel, h in m['files'].items():
        p = root / rel
        out.append({'file': rel, 'exists': p.exists(), 'sha256_ok': p.exists() and sha256(p) == h})
    for md, info in m['models'].items():
        p = root / md / 'metadata.json'
        out.append({'file': f'{md}/metadata.json', 'exists': p.exists(), 'sha256_ok': p.exists() and sha256(p) == info['metadata_sha256']})
        for name, h in info['artifact_sha256'].items():
            q = root / md / name
            out.append({'file': f'{md}/{name}', 'exists': q.exists(), 'sha256_ok': q.exists() and sha256(q) == h})
    return out


def pack() -> Path:
    m = load_manifest()
    DIST.mkdir(exist_ok=True)
    out = DIST / f'qb_serving_bundle_v{BUNDLE_VERSION}.tar.gz'
    with tarfile.open(out, 'w:gz') as tar:
        for rel in list(m['files']) + [f'data/serving/v{BUNDLE_VERSION}/{MANIFEST}']:
            tar.add(ROOT / rel, arcname=rel)
    return out


def install(archive: Path, root: Path = ROOT) -> list[str]:
    """Extract only regular files under the expected data prefixes (no absolute paths, no '..', no links)."""
    names = []
    with tarfile.open(archive, 'r:gz') as tar:
        for mbr in tar.getmembers():
            n = mbr.name
            if not mbr.isfile() or n.startswith(('/', '\\')) or '..' in Path(n).parts or not n.startswith(PACK_PREFIXES):
                raise ValueError(f'refusing archive member {n!r}')
        for mbr in tar.getmembers():
            dest = root / mbr.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(mbr) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
            names.append(mbr.name)
    return names


# ------------------------------------------------------------------ serving source (same interface as forecast.Engine)
class BundleSource:
    """Profiles from the bundle; draft-probability and v2 models from the tracked model files."""

    def __init__(self, bundle_dir: Path = BUNDLE_DIR):
        from src.modeling import draft_predictor as DP
        from src.modeling import train_v2 as T2
        s = joblib.load(bundle_dir / PROFILES)
        self.profiles = s['profiles']
        self._src = {'prospects': s['prospects_gsis']}
        self.combine_seasons = s['combine_seasons']
        self.hist_max = s['hist_max']
        self.prospects = pd.read_parquet(bundle_dir / PROSPECTS)
        self.v2 = joblib.load(T2.MODEL_DIR / 'historical_selected.joblib')
        self.draft_prob = {k: joblib.load(DP.MODEL_DIR / f'prob_{k}.joblib') for k in ('college_l2', 'college_combine_l2')}
        self._by_q = dict(zip(self.prospects['qb_draft_prospect_id'].dropna().astype(int), self.prospects.loc[
            self.prospects['qb_draft_prospect_id'].notna(), 'player_id']))

    def src(self):
        return self._src

    def profile(self, qb_draft_prospect_id=None, name=None, draft_year=None, cfbd_player_id=None, final_season=None, player_id=None):
        if player_id is None:
            if qb_draft_prospect_id is not None:
                player_id = self._by_q.get(int(qb_draft_prospect_id))
            elif cfbd_player_id is not None and final_season is not None:
                player_id = player_id_college(cfbd_player_id, final_season)
                if player_id not in self.profiles:
                    m = self.prospects[(self.prospects['cfbd_player_id'] == str(cfbd_player_id))
                                       & (self.prospects['final_college_season'] == int(final_season))]
                    player_id = m['player_id'].iloc[0] if len(m) else None
            elif name is not None:
                m = self.prospects[(self.prospects['name'] == name) & ((self.prospects['draft_year'] == draft_year) if draft_year else True)]
                player_id = m['player_id'].iloc[0] if len(m) else None
        if player_id not in self.profiles:
            raise ValueError('prospect not in the serving bundle')
        return copy.deepcopy(self.profiles[player_id])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['build', 'verify', 'pack', 'install'])
    ap.add_argument('archive', nargs='?')
    a = ap.parse_args()
    if a.command == 'build':
        print(json.dumps(build(), indent=2))
    elif a.command == 'verify':
        rows = verify()
        bad = [r for r in rows if not (r['exists'] and r['sha256_ok'])]
        for r in rows:
            print(r)
        print(f'{len(rows)} files checked; {len(bad)} problems')
        raise SystemExit(1 if bad else 0)
    elif a.command == 'pack':
        print(pack())
    else:
        print('\n'.join(install(Path(a.archive))))
