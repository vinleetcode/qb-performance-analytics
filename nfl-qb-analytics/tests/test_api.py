"""Phase 6A API tests (FastAPI TestClient; in-process, no network)."""
from __future__ import annotations

import json
import shutil
import socket
import subprocess

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import bundle as B
from api import errors as E
from api.config import ROOT, Settings, get_settings
from api.dependencies import get_service
from api.services.prediction_service import PredictionService

HAVE_BUNDLE = (B.BUNDLE_DIR / B.MANIFEST).exists()
needs_bundle = pytest.mark.skipif(not HAVE_BUNDLE, reason='build the serving bundle first: python -m api.bundle build')
HAVE_LOCAL = (ROOT / 'data' / 'qb_history.db').exists()
MENDOZA, DART, ALLEN, RYAN, STROUD = 'Fernando Mendoza', 'Jaxson Dart', 'Josh Allen', 'Matt Ryan', 'C.J. Stroud'


@pytest.fixture(scope='module')
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def pid_of(client, name, year):
    r = client.get('/api/prospects/resolve', params={'name': name, 'draft_year': year})
    assert r.status_code == 200, r.text
    return r.json()['player_id']


def no_nan(text: str):
    assert 'NaN' not in text and 'Infinity' not in text


# ---------------------------------------------------------------- startup, health, docs
def test_app_starts_and_docs_are_served(client):
    assert client.get('/openapi.json').status_code == 200
    assert client.get('/docs').status_code == 200


@needs_bundle
def test_health_is_cheap_and_reports_verified_models():
    from api.main import app
    svc = PredictionService(get_settings())
    app.dependency_overrides[get_service] = lambda: svc
    try:
        with TestClient(app) as c:
            r = c.get('/api/health')
        j = r.json()
        assert r.status_code == 200 and j['status'] == 'ok' and j['engine_loaded'] is False
        assert all(m['available'] and m['verified'] for m in j['models'])
        assert '\\' not in r.text and 'data/' not in r.text and ':/' not in r.text     # no paths
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------- prospects
@needs_bundle
def test_search_autocomplete_and_pagination(client):
    r = client.get('/api/prospects', params={'q': 'mendo'})
    assert r.status_code == 200
    assert any(i['name'] == MENDOZA for i in r.json()['items'])
    a = client.get('/api/prospects', params={'q': 'a', 'limit': 5, 'offset': 0}).json()
    b = client.get('/api/prospects', params={'q': 'a', 'limit': 5, 'offset': 5}).json()
    assert a['total'] == b['total'] > 10 and len(a['items']) == 5
    assert not {i['player_id'] for i in a['items']} & {i['player_id'] for i in b['items']}
    big = client.get('/api/prospects', params={'limit': 200}).json()
    assert len(big['items']) <= get_settings().max_page_size
    assert client.get('/api/prospects', params={'limit': 0}).status_code == 422
    assert client.get('/api/prospects', params={'q': 'x' * 200}).json()['error']['code'] == E.INVALID_PARAMETER


def test_duplicate_names_are_flagged_and_resolve_is_ambiguous():
    svc = PredictionService(Settings())
    P = pd.DataFrame({'player_id': ['qb_1', 'cfb_9_2019', 'qb_2'], 'name': ['Sam Smith', 'Sam Smith', 'Joe Lee'],
                      'name_key': ['sam smith', 'sam smith', 'joe lee'], 'college': ['A', 'B', 'C'], 'draft_year': [2015, 2020, 2015],
                      'final_college_season': [2014, 2019, 2014], 'drafted': [True, None, True], 'draft_round': [2, None, 1],
                      'overall_pick': [40, None, 5], 'kind': ['drafted', 'college_only', 'drafted'],
                      'qb_draft_prospect_id': [1, None, 2], 'cfbd_player_id': ['1', '9', '2']})
    P['shares_name'] = P['name_key'].duplicated(keep=False)
    svc.prospects, svc._loaded = P.set_index('player_id', drop=False), True
    res = svc.search('sam smith', None, 10, 0)
    assert res['total'] == 2 and all(i['shares_name_with_other_prospects'] for i in res['items'])
    with pytest.raises(E.APIError) as ei:
        svc.resolve('Sam Smith', None)
    assert ei.value.code == E.AMBIGUOUS_PLAYER and len(ei.value.details['candidates']) == 2
    assert svc.resolve('Sam Smith', 2015)['player_id'] == 'qb_1'
    with pytest.raises(E.APIError) as ei:
        svc.resolve('Nobody', None)
    assert ei.value.code == E.PROSPECT_NOT_FOUND


@needs_bundle
def test_prospect_detail(client):
    pid = pid_of(client, MENDOZA, 2026)
    r = client.get(f'/api/prospects/{pid}')
    assert r.status_code == 200
    no_nan(r.text)
    j = r.json()
    assert j['prospect']['name'] == MENDOZA and j['college_seasons']
    assert all(s['season'] <= j['prospect']['final_college_season'] for s in j['college_seasons'])
    assert j['data_coverage']['career_model_features_total'] >= 1


# ---------------------------------------------------------------- forecasts
@needs_bundle
def test_pre_draft_forecast_hides_post_draft_information(client):
    pid = pid_of(client, DART, 2025)
    r = client.get(f'/api/prospects/{pid}/forecast', params={'mode': 'pre_draft'})
    assert r.status_code == 200
    no_nan(r.text)
    j = r.json()
    assert j['mode'] == 'pre_draft' and j['player']['actual_draft'] is None
    s = j['career']['start_probabilities']
    assert 0 <= s['p_64_plus'] <= s['p_32_plus'] <= s['p_16_plus'] <= 1
    assert [x['stat'] for x in j['career']['five_year_totals']] == [
        'games_played', 'games_started', 'pass_attempts', 'pass_completions', 'pass_yards', 'pass_tds', 'interceptions',
        'rush_attempts', 'rush_yards', 'rush_tds']
    for x in j['career']['five_year_totals']:
        lv = {i['level']: i for i in x['intervals']}
        assert lv[0.9]['lower'] <= lv[0.8]['lower'] <= lv[0.5]['lower'] <= lv[0.5]['upper'] <= lv[0.8]['upper'] <= lv[0.9]['upper']
    assert abs(sum(j['draft']['round']['groups'].values()) - 1) < 1e-6
    post = client.get(f'/api/prospects/{pid}/forecast', params={'mode': 'post_draft'}).json()
    assert post['player']['actual_draft']['overall_pick'] == 25
    assert post['career']['five_year_totals'] != j['career']['five_year_totals']


@needs_bundle
def test_unsupported_predictions_are_structured(client):
    m = client.get(f'/api/prospects/{pid_of(client, MENDOZA, 2026)}/forecast').json()
    assert m['draft']['probability']['status'] == 'unavailable' and m['draft']['probability']['reason']
    assert m['draft']['probability']['pre_combine'] is None
    assert 'draft.probability' in m['unavailable_blocks']
    col = client.get('/api/prospects', params={'q': 'henigan'}).json()['items']
    pid = [i['player_id'] for i in col if i['kind'] == 'college_only'][0]
    r = client.get(f'/api/prospects/{pid}/forecast', params={'mode': 'post_draft'})
    assert r.status_code == 200
    c = r.json()['career']
    assert c['status'] == 'unavailable' and c['code'] == E.PREDICTION_UNSUPPORTED and c['five_year_totals'] == []
    h = client.get(f'/api/prospects/{pid}/forecast', params={'mode': 'post_draft', 'assumed_pick': 150}).json()
    assert h['career']['status'] == 'available' and h['career']['hypothetical'] is True
    assert any(w['code'] == 'HYPOTHETICAL_PICK' for w in h['warnings'])


@needs_bundle
@pytest.mark.skipif(not HAVE_LOCAL, reason='needs the full local data for the direct engine')
def test_api_forecast_matches_direct_python_engine(client):
    from src.modeling.prediction_api import forecast_prospect
    for name, year, mode in ((DART, 2025, 'pre_draft'), (MENDOZA, 2026, 'post_draft')):
        api = client.get(f'/api/prospects/{pid_of(client, name, year)}/forecast', params={'mode': mode}).json()
        direct = forecast_prospect(name=name, draft_year=year, mode=mode, n_sims=get_settings().n_sims)
        dc = direct['career']
        assert api['career']['start_probabilities']['p_32_plus'] == pytest.approx(dc['start_probabilities']['p_32_plus'], abs=1e-12)
        for x in api['career']['five_year_totals']:
            assert x['expected'] == pytest.approx(dc['five_year_totals'][x['stat']]['expected'], rel=1e-12)
        y_api = api['career']['year_by_year']['seasons'][2]['stats']['pass_yards']['expected']
        assert y_api == pytest.approx(dc['year_by_year']['seasons'][2]['pass_yards']['expected'], rel=1e-12)
        assert api['draft']['round']['groups'] == pytest.approx(direct['draft']['round']['groups'])


@needs_bundle
def test_forecasts_are_deterministic(client):
    from api.dependencies import get_service as gs
    pid = pid_of(client, DART, 2025)
    a = client.get(f'/api/prospects/{pid}/forecast').json()
    gs()._cache.clear()
    b = client.get(f'/api/prospects/{pid}/forecast').json()
    assert a == b


@needs_bundle
@pytest.mark.parametrize('url,code', [
    ('/api/prospects/qb_7/forecast?mode=draft_day', E.INVALID_PREDICTION_MODE),
    ('/api/prospects/qb_7/forecast?mode=pre_draft&assumed_pick=10', E.INVALID_PARAMETER),
    ('/api/prospects/qb_7/forecast?mode=post_draft&assumed_pick=999', E.INVALID_PARAMETER),
    ('/api/prospects/QB-7/forecast', E.INVALID_PARAMETER),
    ('/api/prospects/qb_999999/forecast', E.PROSPECT_NOT_FOUND),
    ('/api/prospects/../../etc/passwd', E.NOT_FOUND),
    ('/api/nope', E.NOT_FOUND)])
def test_invalid_parameters_get_stable_error_codes(client, url, code):
    r = client.get(url)
    assert r.json()['error']['code'] == code, r.text
    assert r.status_code == E.STATUS[code]


def test_mutating_methods_are_not_allowed(client):
    assert client.post('/api/prospects').status_code == 405


# ---------------------------------------------------------------- comparisons / backtests / performance
@needs_bundle
def test_comparisons_are_explanatory_and_earlier_only(client):
    pid = pid_of(client, DART, 2025)
    j = client.get(f'/api/prospects/{pid}/comparisons').json()
    assert j['status'] == 'available' and j['kind'] == 'explanatory' and len(j['comparables']) == 5
    assert all(c['draft_year'] < 2025 and c['player_id'].startswith('qb_') for c in j['comparables'])
    assert 'probability' not in json.dumps(j['comparables']).lower()


@needs_bundle
def test_backtest_classifications(client):
    a = client.get(f'/api/backtests/{pid_of(client, ALLEN, 2018)}').json()
    assert a['classification']['status'] == 'valid_out_of_sample_backtest' and a['classification']['in_sample'] is False
    assert a['training_classes'][1] <= 2013 and a['horizons']['one_year']['status'] == 'unavailable'
    b = client.get(f'/api/backtests/{pid_of(client, RYAN, 2008)}').json()
    assert b['classification']['status'] == 'retrospective_illustration_in_sample' and b['classification']['in_sample'] is True
    c = client.get(f'/api/backtests/{pid_of(client, STROUD, 2023)}').json()
    assert c['classification']['status'] == 'out_of_sample_forecast_outcome_incomplete'
    assert all(s['error'] is None for s in c['views']['pre_draft']['statistics'])
    col = [i['player_id'] for i in client.get('/api/prospects', params={'q': 'henigan'}).json()['items'] if i['kind'] == 'college_only'][0]
    r = client.get(f'/api/backtests/{col}')
    assert r.status_code == 422 and r.json()['error']['code'] == E.PREDICTION_UNSUPPORTED


def test_model_performance(client):
    r = client.get('/api/models/performance')
    assert r.status_code == 200
    no_nan(r.text)
    j = r.json()
    assert {'draft_probability', 'draft_round', 'starter_probabilities', 'production', 'interval_calibration'} <= set(j['sections'])
    assert j['limitations'] and j['evaluation_status']


# ---------------------------------------------------------------- CORS, rate limit, errors
def _app_with_env(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    from api.main import create_app
    app = create_app()
    get_settings.cache_clear()
    return app


def test_cors_allowlist(monkeypatch):
    app = _app_with_env(monkeypatch, FRONTEND_ORIGINS='*,https://qb.example.com')
    with TestClient(app) as c:
        ok = c.get('/api/models/performance', headers={'Origin': 'https://qb.example.com'})
        bad = c.get('/api/models/performance', headers={'Origin': 'https://evil.example.org'})
        pre = c.options('/api/models/performance', headers={'Origin': 'https://qb.example.com', 'Access-Control-Request-Method': 'GET'})
    assert ok.headers.get('access-control-allow-origin') == 'https://qb.example.com'
    assert 'access-control-allow-origin' not in bad.headers                     # wildcard was ignored
    assert 'access-control-allow-credentials' not in ok.headers and pre.status_code == 200


def test_rate_limit(monkeypatch):
    app = _app_with_env(monkeypatch, RATE_LIMIT_PER_MINUTE='3')
    with TestClient(app) as c:
        codes = [c.get('/api/models/performance').status_code for _ in range(4)]
    assert codes[:3] == [200] * 3 and codes[3] == 429


def test_rate_limited_responses_keep_cors_headers(monkeypatch):
    # a 429 without CORS headers reaches browser code as an opaque network error instead of RATE_LIMITED
    app = _app_with_env(monkeypatch, RATE_LIMIT_PER_MINUTE='1', FRONTEND_ORIGINS='https://qb.example.com')
    with TestClient(app) as c:
        h = {'Origin': 'https://qb.example.com'}
        first, limited = (c.get('/api/models/performance', headers=h) for _ in range(2))
    assert first.status_code == 200 and limited.status_code == 429
    assert limited.headers.get('access-control-allow-origin') == 'https://qb.example.com'
    assert limited.json()['error']['code'] == E.RATE_LIMITED
    assert limited.headers.get('X-Request-ID')


def test_internal_errors_hide_details(client, monkeypatch):
    from api.main import app

    class Boom(PredictionService):
        def search(self, *a, **k):
            raise RuntimeError('C:/secret/path/qb_history.db')
    app.dependency_overrides[get_service] = lambda: Boom(get_settings())
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            r = c.get('/api/prospects', params={'q': 'a'})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500 and r.json()['error']['code'] == E.INTERNAL_ERROR
    assert 'secret' not in r.text and 'Traceback' not in r.text


# ---------------------------------------------------------------- missing / tampered artifacts
def test_missing_bundle_returns_model_unavailable(tmp_path):
    from api.main import app
    svc = PredictionService(Settings(bundle_dir=tmp_path / 'nothing'))
    app.dependency_overrides[get_service] = lambda: svc
    try:
        with TestClient(app) as c:
            r = c.get('/api/prospects/qb_7/forecast')
            h = c.get('/api/health').json()
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and r.json()['error']['code'] == E.MODEL_UNAVAILABLE
    assert h['status'] == 'degraded'


@needs_bundle
def test_tampered_bundle_is_rejected(tmp_path):
    d = tmp_path / 'bundle'
    shutil.copytree(B.BUNDLE_DIR, d)
    m = json.loads((d / B.MANIFEST).read_text(encoding='utf-8'))
    first = next(iter(m['files']))
    m['files'][first] = '0' * 64
    (d / B.MANIFEST).write_text(json.dumps(m), encoding='utf-8')
    svc = PredictionService(Settings(bundle_dir=d))
    with pytest.raises(E.APIError) as ei:
        svc.search('a', None, 5, 0)
    assert ei.value.code == E.MODEL_UNAVAILABLE


@needs_bundle
def test_bundle_and_model_artifacts_verify_and_load():
    rows = B.verify()
    assert rows and all(r['exists'] and r['sha256_ok'] for r in rows)
    from api.services import model_service as M
    assert M.all_verified()
    src = B.BundleSource()
    p = src.profile(player_id=src.prospects['player_id'].iloc[0])
    assert {'v2', 'draft', 'name', 'draft_year'} <= set(p)


def test_bundle_install_refuses_unsafe_archives(tmp_path):
    import io
    import tarfile
    bad = tmp_path / 'bad.tar.gz'
    with tarfile.open(bad, 'w:gz') as t:
        data = b'x'
        info = tarfile.TarInfo('../../evil.py')
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    with pytest.raises(ValueError):
        B.install(bad, tmp_path / 'root')


# ---------------------------------------------------------------- no network, no secrets, gitignore
@needs_bundle
def test_no_external_network_calls_during_inference(client, monkeypatch):
    from api.dependencies import get_service as gs
    gs()._cache.clear()
    attempts = []
    real = socket.socket.connect

    def guard(self, addr):
        host = addr[0] if isinstance(addr, tuple) else str(addr)
        if host not in ('127.0.0.1', '::1', 'localhost'):
            attempts.append(host)
            raise OSError('network disabled in tests')
        return real(self, addr)
    monkeypatch.setattr(socket.socket, 'connect', guard)
    pid = pid_of(client, ALLEN, 2018)
    for url in (f'/api/prospects/{pid}', f'/api/prospects/{pid}/forecast?mode=post_draft', f'/api/prospects/{pid}/comparisons',
                f'/api/backtests/{pid}', '/api/models/performance', '/api/health'):
        assert client.get(url).status_code == 200, url
    assert attempts == []


def _secrets():
    env = ROOT / '.env'
    if not env.exists():
        return []
    out = []
    for line in env.read_text(encoding='utf-8', errors='ignore').splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k, v = line.split('=', 1)
            v = v.strip().strip('"').strip("'")
            if len(v) >= 8 and any(w in k.upper() for w in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                out.append(v)
    return out


@needs_bundle
def test_responses_never_contain_secrets(client):
    secrets = _secrets()
    if not secrets:
        pytest.skip('no local secrets configured')
    pid = pid_of(client, DART, 2025)
    bodies = [client.get(u).text for u in ('/api/health', '/api/models', '/api/models/performance', '/openapi.json',
                                           f'/api/prospects/{pid}', f'/api/prospects/{pid}/forecast')]
    assert not any(s in b for s in secrets for b in bodies)
    bundle_bytes = b''.join((B.BUNDLE_DIR / n).read_bytes() for n in (B.DETAILS, B.BACKTESTS, B.MANIFEST))
    assert not any(s.encode() in bundle_bytes for s in secrets)


def test_generated_data_and_local_config_are_gitignored():
    paths = ['data/serving/v1/manifest.json', 'dist/qb_serving_bundle_v1.tar.gz', '.env', 'results/x.csv',
             'models/qb_career_production/v1.0.0/pre_draft.joblib', '../.claude/settings.local.json']
    r = subprocess.run(['git', 'check-ignore', *paths], cwd=ROOT, capture_output=True, text=True)
    assert set(r.stdout.split()) == set(paths)
    tracked_ok = ['api/main.py', 'api/static/model_performance.json', 'models/qb_career_production/v1.1.0/pre_draft.joblib']
    r = subprocess.run(['git', 'check-ignore', *tracked_ok], cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == ''
