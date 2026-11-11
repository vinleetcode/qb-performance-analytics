"""Model inventory, artifact verification (once per process) and the curated performance summary."""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

from api.config import ROOT
from src.modeling import artifacts as A

log = logging.getLogger('qb_api.models')
PERFORMANCE_FILE = ROOT / 'api' / 'static' / 'model_performance.json'
# models the API serves (name, directory, role)
SERVED = [
    ('qb_draft', 'models/qb_draft/v1.0.0', 'draft probability (pre- and post-Combine)'),
    ('qb_draft', 'models/qb_draft/v1.1.0', 'draft round groups, rounds, pick range'),
    ('qb_career_production', 'models/qb_career_production/v1.1.0', 'start probabilities and five-season production'),
    ('qb_similarity', 'models/qb_similarity/v1.0.0', 'historical comparisons'),
    ('qb_career_success', 'models/qb_career_success/v2.0.0', 'reference P(32+ starts)'),
]


@lru_cache(maxsize=1)
def inventory() -> tuple[dict, ...]:
    """Verify every served model file against its metadata hashes. Computed once per process."""
    out = []
    for name, rel, role in SERVED:
        d = ROOT / rel
        version = rel.rsplit('/v', 1)[-1]
        try:
            meta = A.load_metadata(d)
            files = meta.get('artifact_sha256', {})
            ok = bool(files) and all((d / f).exists() and A.sha256(d / f) == h for f, h in files.items())
            out.append({'name': name, 'version': version, 'role': role, 'available': True, 'verified': ok,
                        'created_by': meta.get('created_by'), 'training_rows': meta.get('training_rows')})
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            log.warning('model %s v%s unavailable', name, version)
            out.append({'name': name, 'version': version, 'role': role, 'available': False, 'verified': False,
                        'created_by': None, 'training_rows': None})
    return tuple(out)


def all_verified() -> bool:
    return all(m['available'] and m['verified'] for m in inventory())


@lru_cache(maxsize=1)
def performance() -> dict:
    return json.loads(Path(PERFORMANCE_FILE).read_text(encoding='utf-8'))
