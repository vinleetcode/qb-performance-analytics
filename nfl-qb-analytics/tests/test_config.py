import pytest

from src import config


def test_missing_key_raises_without_printing_value(monkeypatch, capsys):
    monkeypatch.delenv('CFBD_API_KEY', raising=False)
    monkeypatch.setattr(config, '_loaded', True)  # skip re-reading the real .env file
    with pytest.raises(config.MissingAPIKeyError) as exc_info:
        config.get_cfbd_api_key()
    assert 'CFBD_API_KEY' in str(exc_info.value)
    out, err = capsys.readouterr()
    assert out == '' and err == ''


def test_placeholder_key_raises_helpful_error(monkeypatch):
    monkeypatch.setenv('CFBD_API_KEY', config.PLACEHOLDER)
    monkeypatch.setattr(config, '_loaded', True)
    with pytest.raises(config.MissingAPIKeyError) as exc_info:
        config.get_cfbd_api_key()
    # The placeholder string itself isn't a secret (it's the public sentinel
    # value shipped in .env.example), so it's fine for the message to name it --
    # the point of this test is that a REAL key is never in a position to leak
    # here, since get_cfbd_api_key() only reaches this branch for the placeholder.
    assert 'placeholder' in str(exc_info.value).lower()


def test_real_key_never_appears_in_any_raised_exception(monkeypatch):
    """If a real key is configured, get_cfbd_api_key() must not raise at all --
    this guards against a code path that could echo the key into an error."""
    monkeypatch.setenv('CFBD_API_KEY', 'sk_live_totally_real_secret_12345')
    monkeypatch.setattr(config, '_loaded', True)
    key = config.get_cfbd_api_key()
    assert key == 'sk_live_totally_real_secret_12345'


def test_real_looking_key_is_returned_and_configured_check_is_true(monkeypatch):
    monkeypatch.setenv('CFBD_API_KEY', 'a1b2c3d4e5f6realkeylookingvalue')
    monkeypatch.setattr(config, '_loaded', True)
    assert config.cfbd_api_key_configured() is True
    assert config.get_cfbd_api_key() == 'a1b2c3d4e5f6realkeylookingvalue'


def test_placeholder_is_reported_as_not_configured(monkeypatch):
    monkeypatch.setenv('CFBD_API_KEY', config.PLACEHOLDER)
    monkeypatch.setattr(config, '_loaded', True)
    assert config.cfbd_api_key_configured() is False


def test_empty_key_is_reported_as_not_configured(monkeypatch):
    monkeypatch.setenv('CFBD_API_KEY', '   ')
    monkeypatch.setattr(config, '_loaded', True)
    assert config.cfbd_api_key_configured() is False
