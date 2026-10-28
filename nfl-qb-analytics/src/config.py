"""Environment/secret configuration. Loads .env from the project root and
validates the CollegeFootballData API key without ever printing its value.

Per CollegeFootballData's own guidance, the key is read and used here --
server-side / backend Python -- only. Nothing in this module (or anything
that imports it) should pass the key to a frontend, log it, or include it in
an exception message.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / '.env'

PLACEHOLDER = 'PASTE_YOUR_API_KEY_HERE'

_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        # override=False: real process env vars (e.g. set by a deployment platform)
        # always win over whatever is in the .env file.
        load_dotenv(dotenv_path=ENV_PATH, override=False)
        _loaded = True


class MissingAPIKeyError(RuntimeError):
    """Raised when CFBD_API_KEY is absent or still the placeholder. Never
    include the key value in this exception's message."""


def cfbd_api_key_configured() -> bool:
    """Non-raising check: True only if a real-looking key is present."""
    import os
    _ensure_loaded()
    value = os.environ.get('CFBD_API_KEY', '').strip()
    return bool(value) and value != PLACEHOLDER


def get_cfbd_api_key() -> str:
    """Returns the configured CFBD API key, or raises MissingAPIKeyError with a
    helpful message that names the problem but never the value."""
    import os
    _ensure_loaded()
    value = os.environ.get('CFBD_API_KEY', '').strip()
    if not value:
        raise MissingAPIKeyError(
            f"CFBD_API_KEY is not set. Create {ENV_PATH} (copy .env.example) and set "
            "CFBD_API_KEY=<your real key>, obtained from https://collegefootballdata.com/key"
        )
    if value == PLACEHOLDER:
        raise MissingAPIKeyError(
            f"CFBD_API_KEY in {ENV_PATH} is still the placeholder value. Replace "
            "PASTE_YOUR_API_KEY_HERE with your real key from https://collegefootballdata.com/key"
        )
    return value
