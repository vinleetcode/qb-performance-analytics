"""API settings from environment variables (see .env.example). Nothing here is ever sent to clients."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_VERSION = '1.0.0'


def _int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(os.environ.get(name, default))
    except ValueError:
        v = default
    return max(lo, min(hi, v))


def _origins(raw: str) -> list[str]:
    return [o.strip().rstrip('/') for o in raw.split(',') if o.strip()]


def _env(name, default):
    return lambda: os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    frontend_origins: list[str] = field(default_factory=lambda: _origins(os.environ.get('FRONTEND_ORIGINS', 'http://localhost:3000')))
    bundle_dir: Path = field(default_factory=lambda: ROOT / os.environ.get('QB_API_BUNDLE_DIR', 'data/serving/v1'))
    rate_limit_per_minute: int = field(default_factory=lambda: _int('RATE_LIMIT_PER_MINUTE', 120, 0, 100000))  # 0 disables
    n_sims: int = field(default_factory=lambda: _int('QB_API_N_SIMS', 2000, 200, 10000))
    cache_size: int = field(default_factory=lambda: _int('QB_API_CACHE_SIZE', 512, 0, 100000))
    max_page_size: int = field(default_factory=lambda: _int('QB_API_MAX_PAGE_SIZE', 50, 1, 200))
    docs_enabled: bool = field(default_factory=lambda: os.environ.get('QB_API_DOCS_ENABLED', 'true').lower() != 'false')
    log_level: str = field(default_factory=lambda: os.environ.get('LOG_LEVEL', 'INFO').upper())
    environment: str = field(default_factory=_env('APP_ENV', 'development'))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
