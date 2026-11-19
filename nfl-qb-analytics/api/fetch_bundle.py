"""Supply the serving bundle to a deployment build without committing it or making it public.

    python -m api.fetch_bundle [--local PATH] [--url HTTPS_URL] [--token-file PATH]

The bundle (player-level data derived from CFBD/nflverse) is gitignored, so a host that builds from Git has to fetch
it. Sources, in order:

1. ``--local``: an archive already in the build context (a local ``docker build`` with ``dist/`` present).
2. ``--url`` (or ``QB_BUNDLE_URL``): an HTTPS URL to private storage, read with a bearer token from ``--token-file``
   (a Render secret file mounted as a BuildKit build secret, so it never enters an image layer or the build args).
   For a private GitHub repository the URL is the contents API,
   ``https://api.github.com/repos/<owner>/<repo>/contents/qb_serving_bundle_v1.tar.gz``.

Whatever the source, the archive must match the SHA-256 pinned in ``deploy/serving_bundle.sha256`` (tracked in Git),
then it is extracted with ``api.bundle.install`` (regular files under the data prefixes only). Any problem exits
non-zero, so the image build fails instead of deploying an API without its data.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from api import bundle as B

PIN_FILE = B.ROOT / 'deploy' / 'serving_bundle.sha256'
MAX_BYTES = 100 * 2**20


class FetchError(Exception):
    pass


def pinned_sha256(pin_file: Path = PIN_FILE) -> str:
    try:
        h = pin_file.read_text(encoding='utf-8').split()[0].strip().lower()
    except (OSError, IndexError) as exc:
        raise FetchError(f'missing pinned hash file {pin_file.name}') from exc
    if len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
        raise FetchError('pinned hash is not a SHA-256 hex digest')
    return h


def download(url: str, dest: Path, token: str | None, timeout: int = 120) -> None:
    u = urllib.parse.urlparse(url)
    if u.scheme != 'https' or not u.netloc:
        raise FetchError('bundle URL must be https://')
    headers = {'User-Agent': 'qb-forecast-bundle-fetch'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    if u.netloc == 'api.github.com':
        headers['Accept'] = 'application/vnd.github.raw+json'
        headers['X-GitHub-Api-Version'] = '2022-11-28'
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, 'wb') as f:
            n = 0
            while chunk := r.read(1 << 20):
                n += len(chunk)
                if n > MAX_BYTES:
                    raise FetchError('bundle download exceeds the size limit')
                f.write(chunk)
    except FetchError:
        raise
    except Exception as exc:                       # noqa: BLE001 - never print the token or response body
        raise FetchError(f'bundle download failed ({type(exc).__name__}: {getattr(exc, "code", "")})') from None


def fetch(local: Path | None, url: str | None, token_file: Path | None, root: Path = B.ROOT,
          pin_file: Path = PIN_FILE) -> list[str]:
    want = pinned_sha256(pin_file)
    with tempfile.TemporaryDirectory() as tmp:
        if local is not None and local.is_file():
            archive, origin = local, 'build context'
        elif url:
            token = token_file.read_text(encoding='utf-8').strip() if token_file and token_file.is_file() else None
            archive, origin = Path(tmp) / 'bundle.tar.gz', 'private URL'
            download(url, archive, token)
        else:
            raise FetchError('no serving bundle: set QB_BUNDLE_URL (and the qb_bundle_token secret file) '
                             'or put dist/qb_serving_bundle_v1.tar.gz in the build context')
        got = B.sha256(archive)
        if got != want:
            raise FetchError(f'bundle archive from {origin} does not match the pinned SHA-256')
        return B.install(archive, root=root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--local', type=Path)
    ap.add_argument('--url', default=os.environ.get('QB_BUNDLE_URL', ''))
    ap.add_argument('--token-file', type=Path)
    a = ap.parse_args(argv)
    try:
        names = fetch(a.local, a.url.strip() or None, a.token_file)
    except (FetchError, ValueError) as exc:
        print(f'fetch_bundle: {exc}', file=sys.stderr)
        return 1
    print(f'fetch_bundle: installed {len(names)} files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
