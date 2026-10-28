"""Thin, cache-first HTTP client for the CollegeFootballData API.

Design choices driven by the project's data-access policy and CFBD's own
published constraints (free tier: 1,000 calls/month):
  - Every response is cached to disk, keyed by (path, sorted params). A second
    run of the same ingestion makes zero new network calls -- this is the
    "incremental downloads" requirement, and it also makes the client testable
    without hitting the live API or needing a real key.
  - A small fixed delay between *actual* HTTP calls (cache misses only).
  - Bounded retries (not infinite) with exponential backoff on 429/5xx.
  - The API key is read once via src.config.get_cfbd_api_key(), sent only as
    an Authorization header on server-side requests, and never logged, never
    included in an exception message, and never returned to any caller.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from src.config import get_cfbd_api_key

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / 'data' / 'cfbd_cache'
BASE_URL = 'https://api.collegefootballdata.com'

MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0
THROTTLE_SECONDS = 1.0  # minimum spacing between real (non-cached) requests


class CFBDRequestError(RuntimeError):
    """Raised on a non-retryable or retry-exhausted API failure. Message never
    includes the API key."""


class CFBDClient:
    def __init__(self, cache_dir: Path | None = None, api_key: str | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Resolved lazily (not at construction) so building a client in code
        # that never ends up calling the API doesn't require a key to exist.
        self._api_key = api_key
        self._session = requests.Session()
        self._last_request_at = 0.0

    def _auth_header(self) -> dict:
        key = self._api_key or get_cfbd_api_key()
        return {'Authorization': f'Bearer {key}'}

    @staticmethod
    def _cache_key(path: str, params: dict) -> str:
        canonical = json.dumps({'path': path, 'params': params}, sort_keys=True)
        digest = hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]
        safe_path = path.strip('/').replace('/', '_')
        return f'{safe_path}__{digest}.json'

    def get(self, path: str, params: dict | None = None, use_cache: bool = True) -> list | dict:
        params = params or {}
        cache_file = self.cache_dir / self._cache_key(path, params)

        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text(encoding='utf-8'))

        self._throttle()
        url = f'{BASE_URL}{path}'
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params, headers=self._auth_header(), timeout=30)
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
                continue
            self._last_request_at = time.monotonic()

            if resp.status_code == 200:
                # JSON is always UTF-8 per RFC 8259 -- pin it explicitly rather
                # than rely on requests' own charset detection, as cheap
                # insurance against a response that ever omits charset (CFBD's
                # server was verified 2026-10-30 to send it correctly today).
                resp.encoding = 'utf-8'
                data = resp.json()
                cache_file.write_text(json.dumps(data), encoding='utf-8')
                return data
            if resp.status_code in (401, 403):
                # Never echo the response body here -- some APIs reflect request
                # params/headers back in error bodies.
                raise CFBDRequestError(
                    f'CFBD API rejected the request ({resp.status_code}) for {path}. '
                    'Check that CFBD_API_KEY is a valid, active key.'
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = CFBDRequestError(f'CFBD API returned {resp.status_code} for {path}')
                time.sleep(BACKOFF_BASE_SECONDS * attempt)
                continue
            raise CFBDRequestError(f'CFBD API returned unexpected status {resp.status_code} for {path}')

        raise CFBDRequestError(
            f'CFBD API request to {path} failed after {MAX_RETRIES} attempts: {last_exc}'
        )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < THROTTLE_SECONDS:
            time.sleep(THROTTLE_SECONDS - elapsed)
