"""Public smoke test for a deployed QB Forecast Lab (stdlib only; run from nfl-qb-analytics/).

    python deploy/smoke_test.py --site https://<project>.vercel.app --api https://<service>.onrender.com

Checks the frontend pages, security headers and absence of localhost URLs; the API health, search, forecasts,
comparisons and backtests; CORS; that docs and the bundle are not exposed; and, when the serving bundle is installed
locally, that the live forecasts are identical to the local engine's. Times every request (the first one may include
a free-tier wake-up of about a minute). Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAILS: list[str] = []
SAMPLE = [('qb_226', 'pre_draft'), ('qb_226', 'post_draft'), ('qb_319', 'pre_draft')]


def get(url: str, origin: str | None = None, timeout: int = 180):
    h = {'User-Agent': 'qb-forecast-smoke-test', 'Accept': '*/*'}
    if origin:
        h['Origin'] = origin
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout) as r:
            body, status, headers = r.read(), r.status, r.headers
    except urllib.error.HTTPError as e:
        body, status, headers = e.read(), e.code, e.headers
    return status, round(1000 * (time.perf_counter() - t)), body, headers


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', required=True)
    ap.add_argument('--api', required=True)
    a = ap.parse_args()
    site, api = a.site.rstrip('/'), a.api.rstrip('/')
    check('site uses https', site.startswith('https://'))
    check('api uses https', api.startswith('https://'))

    # ---- backend (first call may wake a sleeping free instance)
    s, ms, b, _ = get(f'{api}/api/health')
    j = json.loads(b) if s == 200 else {}
    check('api health ok', s == 200 and j.get('status') == 'ok', f'{ms} ms (includes any wake-up), status={j.get("status")}')
    check('all models verified', bool(j.get('models')) and all(m['verified'] for m in j['models']))
    for _ in range(60):                                       # warm-up thread loads the engine after boot
        if json.loads(get(f'{api}/api/health')[2]).get('engine_loaded'):
            break
        time.sleep(2)
    s, ms, b, h = get(f'{api}/api/prospects?q=allen&limit=5', origin=site)
    items = json.loads(b).get('items', []) if s == 200 else []
    check('search returns real players', s == 200 and any(i['name'] == 'Josh Allen' for i in items), f'{ms} ms')
    check('CORS allows the site', h.get('access-control-allow-origin') == site, str(h.get('access-control-allow-origin')))
    s, ms, b, h = get(f'{api}/api/prospects?q=allen', origin='https://not-the-site.example')
    check('CORS refuses other origins', 'access-control-allow-origin' not in {k.lower() for k in h.keys()})
    s, _, b, _ = get(f'{api}/api/prospects?limit=200')
    check('page size capped', s == 200 and json.loads(b)['limit'] <= 20, str(json.loads(b).get('limit')) if s == 200 else str(s))

    live = {}
    for pid, mode in SAMPLE:
        s, ms, b, _ = get(f'{api}/api/prospects/{pid}/forecast?mode={mode}')
        live[(pid, mode)] = json.loads(b) if s == 200 else None
        pt = next((x['expected'] for x in live[(pid, mode)]['career']['five_year_totals'] if x['stat'] == 'pass_tds'), None) \
            if s == 200 and live[(pid, mode)]['career']['status'] == 'available' else None
        check(f'forecast {pid} {mode}', s == 200, f'{ms} ms, expected pass TDs={pt}')
        s2, ms2, b2, _ = get(f'{api}/api/prospects/{pid}/forecast?mode={mode}')
        check(f'repeat forecast {pid} {mode} identical', s2 == 200 and json.loads(b2) == live[(pid, mode)], f'{ms2} ms')
    s, ms, b, _ = get(f'{api}/api/prospects/qb_226/comparisons')
    check('comparisons load', s == 200 and json.loads(b).get('status') == 'available', f'{ms} ms')
    s, ms, b, _ = get(f'{api}/api/backtests/qb_226')
    check('backtest loads', s == 200 and bool(json.loads(b).get('views')), f'{ms} ms')
    s, ms, b, _ = get(f'{api}/api/models/performance')
    check('model performance loads', s == 200, f'{ms} ms')

    for path in ('/docs', '/openapi.json', '/data/serving/v1/manifest.json', '/dist/qb_serving_bundle_v1.tar.gz',
                 '/api/../data/serving/v1/profiles.joblib', '/.env'):
        s, _, b, _ = get(f'{api}{path}')
        check(f'api does not serve {path}', s in (400, 404), str(s))
    s, _, b, _ = get(f'{api}/api/prospects/qb_999999/forecast')
    check('errors are structured, no traceback', s == 404 and b'Traceback' not in b and b'error' in b, str(s))

    # ---- identical to the local engine (needs the local bundle)
    try:
        from api import bundle as B
        if (B.BUNDLE_DIR / B.MANIFEST).exists():
            # through the local app, so the response model serializes both sides the same way
            import os
            os.environ['RATE_LIMIT_PER_MINUTE'] = '0'
            from fastapi.testclient import TestClient
            from api.main import create_app
            with TestClient(create_app()) as c:
                for (pid, mode), got in live.items():
                    want = c.get(f'/api/prospects/{pid}/forecast', params={'mode': mode}).json()
                    check(f'live forecast {pid} {mode} == local engine', got == want)
        else:
            print('SKIP  local comparison (no local serving bundle)')
    except Exception as exc:                                  # noqa: BLE001
        check('local comparison ran', False, type(exc).__name__)

    # ---- frontend
    pages = ['/', '/prospects', '/prospects/qb_226', '/backtesting', '/methodology', '/case-study', '/credits',
             '/robots.txt', '/opengraph-image', '/icon.svg', '/manifest.webmanifest']
    for p in pages:
        s, ms, b, h = get(f'{site}{p}')
        ok = s == 200 or (p == '/robots.txt' and s == 404)
        check(f'site {p}', ok, f'{s} {ms} ms')
        if p == '/' and s == 200:
            html = b.decode('utf-8', 'replace')
            check('no localhost in homepage HTML', 'localhost' not in html)
            for k in ('content-security-policy', 'x-content-type-options', 'x-frame-options', 'referrer-policy',
                      'strict-transport-security'):
                check(f'header {k}', k in {x.lower() for x in h.keys()})
            csp = h.get('content-security-policy', '')
            check('CSP connect-src names the API', api in csp, '')
    for p in ('/images/hero-stadium-night.jpg', '/brand/icon-192.png'):
        s, _, b, h = get(f'{site}{p}')
        check(f'asset {p}', s == 200 and len(b) > 1000, h.get('content-type', ''))
    for p in ('/data/serving/v1/manifest.json', '/qb_serving_bundle_v1.tar.gz', '/.env', '/.env.local'):
        s, *_ = get(f'{site}{p}')
        check(f'site does not serve {p}', s == 404, str(s))

    print(f'\n{len(FAILS)} failed' + (': ' + ', '.join(FAILS) if FAILS else ''))
    return 1 if FAILS else 0


if __name__ == '__main__':
    raise SystemExit(main())
