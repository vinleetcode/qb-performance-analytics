"""Phase 9 (free deployment) tests: fast profile loading changes no output, the bundle fetch step is safe, the
warm-up hook, and the free-tier deployment configuration."""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile

import pytest
from fastapi.testclient import TestClient

from api import bundle as B
from api import fetch_bundle as FB
from api.config import ROOT, Settings, get_settings
from api.services.prediction_service import PredictionService

HAVE_BUNDLE = (B.BUNDLE_DIR / B.MANIFEST).exists()
needs_bundle = pytest.mark.skipif(not HAVE_BUNDLE, reason='build the serving bundle first: python -m api.bundle build')
REPO = ROOT.parent


@pytest.fixture
def bundle_copy(tmp_path):
    d = tmp_path / 'bundle'
    shutil.copytree(B.BUNDLE_DIR, d, ignore=shutil.ignore_patterns(B.FAST_PROFILES, B.FAST_META))
    yield d
    B._serving_cache.pop(str(d.resolve()), None)


def _outputs(svc: PredictionService, pids: list[str]) -> str:
    out = {}
    for pid in pids:
        out[pid] = {'detail': svc.detail(pid), 'pre': svc.forecast(pid, 'pre_draft', None),
                    'post': svc.forecast(pid, 'post_draft', None), 'pick': svc.forecast(pid, 'post_draft', 40),
                    'comps': svc.comparisons(pid)}
    return json.dumps(out, sort_keys=True)


# ---------------------------------------------------------------- fast profile loading preserves outputs exactly
@needs_bundle
def test_fast_profiles_give_identical_outputs(bundle_copy, monkeypatch):
    pids = ['qb_226', 'qb_319']                                     # drafted QBs (Josh Allen and another)
    college = PredictionService(Settings(bundle_dir=bundle_copy)).search(None, 2026, 50, 0)['items']
    pids += [next(p['player_id'] for p in college if p['kind'] == 'college_only')]
    reference = _outputs(PredictionService(Settings(bundle_dir=bundle_copy)), pids)      # joblib path

    meta = B.optimize(bundle_copy)
    assert meta['source_sha256'] == B.sha256(bundle_copy / B.PROFILES)
    B._serving_cache.pop(str(bundle_copy.resolve()), None)
    real_load = B.joblib.load

    def no_profiles_joblib(path, *a, **k):
        assert B.PROFILES not in str(path), 'fast path not used'
        return real_load(path, *a, **k)
    monkeypatch.setattr(B.joblib, 'load', no_profiles_joblib)
    fast = _outputs(PredictionService(Settings(bundle_dir=bundle_copy)), pids)
    assert fast == reference
    assert B.verify(ROOT, bundle_copy) and all(r['sha256_ok'] for r in B.verify(ROOT, bundle_copy))  # manifest unaffected


@needs_bundle
def test_tampered_fast_profiles_are_never_unpickled(bundle_copy, monkeypatch):
    B.optimize(bundle_copy)
    (bundle_copy / B.FAST_PROFILES).write_bytes(b'not the file that was hashed')
    B._serving_cache.pop(str(bundle_copy.resolve()), None)
    import pickle

    def refuse(*a, **k):
        raise AssertionError('tampered fast file was unpickled')
    monkeypatch.setattr(pickle, 'load', refuse)
    s = B.load_serving(bundle_copy)                                # falls back to the verified joblib file
    assert 'profiles' in s and 'prospects_gsis' in s


@needs_bundle
def test_optimize_refuses_a_profiles_file_that_fails_the_manifest(bundle_copy):
    (bundle_copy / B.PROFILES).write_bytes(b'tampered')
    with pytest.raises(ValueError):
        B.optimize(bundle_copy)


@needs_bundle
def test_profiles_are_loaded_once_per_process(bundle_copy, monkeypatch):
    calls = []
    real_load = B.joblib.load

    def counting(path, *a, **k):
        if B.PROFILES in str(path):
            calls.append(path)
        return real_load(path, *a, **k)
    monkeypatch.setattr(B.joblib, 'load', counting)
    svc = PredictionService(Settings(bundle_dir=bundle_copy))
    svc.detail('qb_226')                         # GSIS lookup
    svc.forecast('qb_226', 'pre_draft', None)    # engine
    assert len(calls) == 1


# ---------------------------------------------------------------- bundle fetch step
def _archive(tmp_path, members: dict[str, bytes]):
    p = tmp_path / 'b.tar.gz'
    with tarfile.open(p, 'w:gz') as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return p


def test_fetch_installs_a_local_archive_matching_the_pin(tmp_path):
    a = _archive(tmp_path, {'data/serving/v1/x.json': b'{}'})
    pin = tmp_path / 'pin'
    pin.write_text(B.sha256(a) + '  qb_serving_bundle_v1.tar.gz\n')
    names = FB.fetch(a, None, None, root=tmp_path / 'root', pin_file=pin)
    assert names == ['data/serving/v1/x.json'] and (tmp_path / 'root/data/serving/v1/x.json').exists()


def test_fetch_rejects_an_archive_that_does_not_match_the_pin(tmp_path):
    a = _archive(tmp_path, {'data/serving/v1/x.json': b'{}'})
    pin = tmp_path / 'pin'
    pin.write_text('0' * 64)
    with pytest.raises(FB.FetchError):
        FB.fetch(a, None, None, root=tmp_path / 'root', pin_file=pin)
    assert not (tmp_path / 'root').exists()


@pytest.mark.parametrize('url', ['http://example.com/b.tar.gz', 'file:///etc/passwd', 'ftp://x/y'])
def test_fetch_requires_https(tmp_path, url):
    pin = tmp_path / 'pin'
    pin.write_text('0' * 64)
    with pytest.raises(FB.FetchError, match='https'):
        FB.fetch(None, url, None, root=tmp_path, pin_file=pin)


def test_fetch_without_a_source_fails_the_build(tmp_path):
    pin = tmp_path / 'pin'
    pin.write_text('0' * 64)
    assert FB.main(['--local', str(tmp_path / 'missing.tar.gz'), '--url', '']) == 1


def test_fetch_errors_do_not_leak_the_token(tmp_path, monkeypatch):
    token = tmp_path / 'tok'
    token.write_text('ghp_SUPERSECRETTOKEN')

    def boom(*a, **k):
        raise OSError('connection refused for ghp_SUPERSECRETTOKEN')
    monkeypatch.setattr(FB.urllib.request, 'urlopen', boom)
    with pytest.raises(FB.FetchError) as ei:
        FB.fetch(None, 'https://api.github.com/repos/o/r/contents/b.tar.gz', token, root=tmp_path,
                 pin_file=_pin(tmp_path))
    assert 'SUPERSECRET' not in str(ei.value)


def _pin(tmp_path):
    p = tmp_path / 'pin2'
    p.write_text('a' * 64)
    return p


def test_pinned_hash_matches_the_local_archive_when_present():
    archive = ROOT / 'dist' / 'qb_serving_bundle_v1.tar.gz'
    if not archive.exists():
        pytest.skip('no local archive')
    assert FB.pinned_sha256() == B.sha256(archive)


# ---------------------------------------------------------------- warm-up
def test_warm_up_runs_only_when_enabled(monkeypatch):
    import api.main as main
    called = []
    monkeypatch.setattr(main, '_warm_up', lambda: called.append(1))
    for flag, expected in (('false', 0), ('true', 1)):
        called.clear()
        monkeypatch.setenv('QB_API_WARMUP', flag)
        get_settings.cache_clear()
        app = main.create_app()
        with TestClient(app):
            pass
        get_settings.cache_clear()
        import time
        for _ in range(50):
            if len(called) >= expected:
                break
            time.sleep(0.02)
        assert len(called) == expected


# ---------------------------------------------------------------- free-tier deployment configuration
def test_dockerfile_binds_render_port_and_never_copies_raw_data():
    d = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    assert '${PORT:-8000}' in d and 'QB_API_DOCS_ENABLED=false' in d and 'QB_API_WARMUP=true' in d
    assert 'COPY data/' not in d and 'id=qb_bundle_token' in d and 'api.bundle verify' in d
    assert 'FROM python:3.14' in d                                   # the pickles need Python 3.14
    ign = (ROOT / '.dockerignore').read_text(encoding='utf-8').split()
    assert 'data/' in ign and 'qb_bundle_token' in ign and '.env' in ign


def test_render_blueprint_uses_the_free_plan_and_no_disk():
    y = (REPO / 'render.yaml').read_text(encoding='utf-8')
    assert 'plan: free' in y and 'disk:' not in y and 'healthCheckPath: /api/health' in y
    assert 'FRONTEND_ORIGINS' in y and 'sync: false' in y and 'QB_API_DOCS_ENABLED' in y


def test_bundle_archive_is_gitignored_but_its_pin_is_not():
    r = subprocess.run(['git', 'check-ignore', 'dist/qb_serving_bundle_v1.tar.gz', 'data/serving/v1/profiles.fast.pkl'],
                       cwd=ROOT, capture_output=True, text=True)
    assert len(r.stdout.split()) == 2
    r = subprocess.run(['git', 'check-ignore', 'deploy/serving_bundle.sha256', 'api/fetch_bundle.py'],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.stdout.strip() == ''


@needs_bundle
def test_committed_model_files_match_the_bundle_manifest_byte_for_byte():
    # Render builds from a Linux checkout: line-ending conversion of metadata.json would fail `api.bundle verify`
    import hashlib
    m = B.load_manifest()
    for md, info in m['models'].items():
        for rel, h in {f'{md}/metadata.json': info['metadata_sha256'],
                       **{f'{md}/{n}': x for n, x in info['artifact_sha256'].items()}}.items():
            blob = subprocess.run(['git', 'show', f':nfl-qb-analytics/{rel}'], cwd=ROOT, capture_output=True).stdout
            assert hashlib.sha256(blob).hexdigest() == h, f'{rel}: bytes in Git differ from the bundle manifest'
