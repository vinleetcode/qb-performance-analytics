"""Tests the CFBD client's caching/retry/security behavior without ever making
a real network call -- requests.Session.get is monkeypatched."""
import json

import pytest

from src.cfbd_client import CFBDClient, CFBDRequestError


class FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.encoding = None  # set by the client, mirrors real requests.Response

    def json(self):
        return self._payload


class FakeEncodingSensitiveResponse:
    """A closer-to-real fake: .json() actually decodes raw bytes using
    self.encoding, the way requests.Response.text/.json() do -- so this
    exercises the real bug (a wrong/default encoding mangling non-ASCII
    names) rather than just returning a pre-built Python object."""
    def __init__(self, status_code, body_obj, wrong_default_encoding='iso-8859-1'):
        self.status_code = status_code
        self._raw_bytes = json.dumps(body_obj, ensure_ascii=False).encode('utf-8')
        self.encoding = wrong_default_encoding  # simulates requests' encoding misdetection

    def json(self):
        import json as _json
        text = self._raw_bytes.decode(self.encoding, errors='replace')
        return _json.loads(text)


def test_non_ascii_names_are_decoded_correctly_not_mangled(tmp_path, monkeypatch):
    """Defensive test, not a reproduction of an observed bug: live verification
    (2026-10-30) showed CFBD's server correctly sends UTF-8 with an explicit
    charset and requests decodes it correctly on its own -- a terminal display
    artifact had briefly looked like corruption but the stored bytes were
    always correct. The client still forces resp.encoding = 'utf-8' before
    parsing (JSON is always UTF-8 per RFC 8259) as cheap insurance against a
    server that ever omits charset; this test exercises that path directly
    so it can't silently regress."""
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return FakeEncodingSensitiveResponse(200, [{'school': 'San José State'}])

    monkeypatch.setattr('requests.Session.get', fake_get)
    client = CFBDClient(cache_dir=tmp_path, api_key='fake-key')
    data = client.get('/roster', {'year': 2024})
    assert data[0]['school'] == 'San José State'


def test_second_call_is_served_from_cache_not_network(tmp_path, monkeypatch):
    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        return FakeResponse(200, [{'season': 2024, 'stat': '100'}])

    monkeypatch.setattr('requests.Session.get', fake_get)
    client = CFBDClient(cache_dir=tmp_path, api_key='fake-key')

    first = client.get('/stats/player/season', {'year': 2024, 'category': 'passing'})
    second = client.get('/stats/player/season', {'year': 2024, 'category': 'passing'})

    assert first == second == [{'season': 2024, 'stat': '100'}]
    assert len(calls) == 1  # the second .get() was a pure cache hit


def test_different_params_are_not_conflated_in_cache(tmp_path, monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return FakeResponse(200, [{'year': params['year']}])

    monkeypatch.setattr('requests.Session.get', fake_get)
    client = CFBDClient(cache_dir=tmp_path, api_key='fake-key')

    r2023 = client.get('/stats/player/season', {'year': 2023, 'category': 'passing'})
    r2024 = client.get('/stats/player/season', {'year': 2024, 'category': 'passing'})
    assert r2023 != r2024


def test_auth_failure_raises_without_retrying_forever(tmp_path, monkeypatch):
    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append(1)
        return FakeResponse(401)

    monkeypatch.setattr('requests.Session.get', fake_get)
    client = CFBDClient(cache_dir=tmp_path, api_key='bad-key')
    with pytest.raises(CFBDRequestError):
        client.get('/stats/player/season', {'year': 2024})
    assert len(calls) == 1  # 401 is not retried


def test_rate_limit_is_retried_a_bounded_number_of_times(tmp_path, monkeypatch):
    calls = []

    def fake_get(self, url, params=None, headers=None, timeout=None):
        calls.append(1)
        return FakeResponse(429)

    monkeypatch.setattr('requests.Session.get', fake_get)
    monkeypatch.setattr('time.sleep', lambda s: None)  # don't actually wait in tests
    client = CFBDClient(cache_dir=tmp_path, api_key='fake-key')
    with pytest.raises(CFBDRequestError):
        client.get('/stats/player/season', {'year': 2024})
    assert len(calls) == 3  # MAX_RETRIES, not unbounded


def test_client_error_message_never_contains_the_api_key(tmp_path, monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return FakeResponse(401)

    monkeypatch.setattr('requests.Session.get', fake_get)
    secret = 'sk_super_secret_value_9999'
    client = CFBDClient(cache_dir=tmp_path, api_key=secret)
    with pytest.raises(CFBDRequestError) as exc_info:
        client.get('/stats/player/season', {'year': 2024})
    assert secret not in str(exc_info.value)


def test_cache_file_on_disk_never_contains_the_api_key(tmp_path, monkeypatch):
    def fake_get(self, url, params=None, headers=None, timeout=None):
        return FakeResponse(200, [{'ok': True}])

    monkeypatch.setattr('requests.Session.get', fake_get)
    secret = 'sk_super_secret_value_9999'
    client = CFBDClient(cache_dir=tmp_path, api_key=secret)
    client.get('/stats/player/season', {'year': 2024})

    for f in tmp_path.glob('*.json'):
        assert secret not in f.read_text(encoding='utf-8')
