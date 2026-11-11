"""Model-artifact data files kept OUT of Git.

Trained model artifacts (joblib pipelines + metadata.json) live in models/ and may
be tracked. Player-level data derived from CFBD (the college reference population,
season baselines, per-player reproducibility snapshots) is NOT distributed with the
repository: it lives under the gitignored data/model_artifacts/<model>/<version>/
and each model's metadata.json lists it under `external_data_files` with a
repository-relative path, its SHA-256 and the command that regenerates it.

  python -m src.modeling.artifacts --check            # verify every listed data file AND every pinned model file
  python -m src.modeling.artifacts --scan             # list large arrays embedded in joblib artifacts
  python -m src.modeling.artifacts --migrate-v2       # one-off move done on 2026-11-10 (idempotent)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'models'
DATA_ARTIFACTS = ROOT / 'data' / 'model_artifacts'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def load_metadata(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / 'metadata.json').read_text(encoding='utf-8'))


def external_files(model_dir: Path) -> list[dict]:
    return load_metadata(model_dir).get('external_data_files', [])


def resolve(model_dir: Path, role: str, verify: bool = True) -> Path:
    """Absolute path of an external data file by role; verifies its SHA-256."""
    for e in external_files(model_dir):
        if e['role'] == role:
            p = ROOT / e['path']
            if not p.exists():
                raise FileNotFoundError(f'{e["path"]} is not in this checkout (it is gitignored). '
                                        f'Regenerate it with: {e["regenerate"]}')
            if verify and sha256(p) != e['sha256']:
                raise ValueError(f'{e["path"]} does not match the SHA-256 recorded for {rel(model_dir)}; '
                                 'it was regenerated from different data. Re-train or restore the pinned copy.')
            return p
    raise KeyError(f'no external data file with role {role!r} in {rel(model_dir)}/metadata.json')


def check_all() -> list[dict]:
    rows = []
    for md in sorted(MODELS.glob('*/v*/metadata.json')):
        for e in external_files(md.parent):
            p = ROOT / e['path']
            rows.append({'model': rel(md.parent), 'role': e['role'], 'path': e['path'], 'exists': p.exists(),
                         'sha256_ok': p.exists() and sha256(p) == e['sha256']})
    return rows


def check_model_files() -> list[dict]:
    """Every file listed under `artifact_sha256` in a model's metadata.json exists and matches."""
    rows = []
    for md in sorted(MODELS.glob('*/v*/metadata.json')):
        for name, h in load_metadata(md.parent).get('artifact_sha256', {}).items():
            p = md.parent / name
            rows.append({'model': rel(md.parent), 'file': name, 'exists': p.exists(), 'sha256_ok': p.exists() and sha256(p) == h})
    return rows


def _walk(obj, path='', seen=None, out=None, min_rows=30, depth=0):
    import numpy as np
    import pandas as pd
    seen = set() if seen is None else seen
    out = [] if out is None else out
    if id(obj) in seen or depth > 12:
        return out
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        if obj.ndim >= 1 and obj.shape[0] >= min_rows and obj.dtype != object:
            out.append({'path': path, 'shape': tuple(obj.shape), 'dtype': str(obj.dtype)})
        elif obj.dtype == object:
            for i, v in enumerate(obj.ravel()[:500]):
                _walk(v, f'{path}[{i}]', seen, out, min_rows, depth + 1)
        return out
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if len(obj) >= min_rows:
            out.append({'path': path, 'shape': tuple(obj.shape), 'dtype': 'pandas'})
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f'{path}.{k}' if path else str(k), seen, out, min_rows, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f'{path}[{i}]', seen, out, min_rows, depth + 1)
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        for k, v in vars(obj).items():
            _walk(v, f'{path}.{k}', seen, out, min_rows, depth + 1)
    return out


def scan_embedded(min_rows: int = 30) -> list[dict]:
    """Arrays with >= min_rows rows inside each joblib artifact. Coefficients and scalers are small; long arrays
    are training-derived records (career paths, residuals) or per-value tables (tree bin thresholds)."""
    import joblib
    rows = []
    for p in sorted(MODELS.glob('*/v*/*.joblib')):
        found = _walk(joblib.load(p), min_rows=min_rows)
        rows.append({'artifact': rel(p), 'large_arrays': len(found), 'largest_rows': max((f['shape'][0] for f in found), default=0),
                     'examples': [f['path'] for f in found[:4]]})
    return rows


V2_MOVES = [  # (file in the model dir, role, regenerate command)
    ('college_reference_population_v2.0.0.parquet', 'reference_population',
     'python -m src.modeling.build_dataset_v2 (writes data/modeling/college_reference_population_v2.0.0.parquet)'),
    ('college_season_baselines_v2.0.0.parquet', 'season_baselines',
     'python -m src.modeling.build_dataset_v2 (writes data/modeling/college_season_baselines_v2.0.0.parquet)'),
    ('reproducibility_check.csv', 'reproducibility_check',
     'python -m src.modeling.train_v2 (retrains v2 into a new directory; do not overwrite v2.0.0)'),
]


def migrate_v2(model_dir: Path = MODELS / 'qb_career_success' / 'v2.0.0') -> dict:
    """Move CFBD-derived files out of the tracked model directory. Idempotent.
    joblib artifacts are never touched; their hashes are recorded before and after."""
    meta = load_metadata(model_dir)
    target = DATA_ARTIFACTS / 'qb_career_success' / 'v2.0.0'
    target.mkdir(parents=True, exist_ok=True)
    joblib_before = {p.name: sha256(p) for p in sorted(model_dir.glob('*.joblib'))}
    ext = {e['role']: e for e in meta.get('external_data_files', [])}
    for name, role, regen in V2_MOVES:
        src, dst = model_dir / name, target / name
        if src.exists():
            if dst.exists() and sha256(dst) != sha256(src):
                raise ValueError(f'{dst} exists with different content; refusing to overwrite')
            if not dst.exists():
                shutil.copy2(src, dst)
            assert sha256(dst) == sha256(src)
            src.unlink()
        ext[role] = {'role': role, 'path': rel(dst), 'sha256': sha256(dst), 'regenerate': regen,
                     'reason': 'player-level data derived from CFBD; kept out of Git'}
    joblib_after = {p.name: sha256(p) for p in sorted(model_dir.glob('*.joblib'))}
    assert joblib_before == joblib_after
    meta['external_data_files'] = list(ext.values())
    meta['normalization']['baselines_file'] = ext['season_baselines']['path']
    meta['normalization']['reference_population_file'] = ext['reference_population']['path']
    meta['artifacts'] = sorted(joblib_after) + ['metadata.json']
    meta['artifact_sha256'] = joblib_after
    revs = meta.get('metadata_revisions', [])
    if not any(r.get('change', '').startswith('moved CFBD-derived') for r in revs):
        revs.append({'date': date.today().isoformat(),
                     'change': 'moved CFBD-derived parquet/csv files to data/model_artifacts (gitignored); '
                               'joblib artifacts byte-identical (artifact_sha256)'})
    meta['metadata_revisions'] = revs
    (model_dir / 'metadata.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    return {'moved_to': rel(target), 'joblib_sha256': joblib_after, 'external': list(ext.values())}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--migrate-v2', action='store_true')
    ap.add_argument('--scan', action='store_true')
    a = ap.parse_args()
    if a.migrate_v2:
        print(json.dumps(migrate_v2(), indent=2))
    if a.scan:
        for r in scan_embedded():
            print(r)
    if a.check or not (a.migrate_v2 or a.scan):
        data, files = check_all(), check_model_files()
        for r in data:
            print(r)
        for r in files:
            print(r)
        bad = [r for r in data + files if not (r['exists'] and r['sha256_ok'])]
        print(f'{len(data)} external data files and {len(files)} model files checked; {len(bad)} problems')
        raise SystemExit(1 if bad else 0)
