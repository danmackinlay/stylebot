"""Tests for stylebot.config — the one home of the path/secret precedence ladder.

Everything resolves out of the process environment (direnv's job in real use),
so `monkeypatch.setenv` is the whole fixture: no file discovery to neutralise,
and no way for a developer-local config to leak into the assertions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stylebot import config


def test_data_dir_flag_beats_env(monkeypatch):
    monkeypatch.setenv("STYLEBOT_DATA_DIR", "/from/env")
    assert config.resolve_data_dir("/from/flag") == Path("/from/flag")


def test_data_dir_env_beats_default(monkeypatch):
    monkeypatch.setenv("STYLEBOT_DATA_DIR", "/from/env")
    assert config.resolve_data_dir() == Path("/from/env")


def test_get_key_blank_is_none(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "")
    assert config.get_key("SOME_API_KEY") is None
    monkeypatch.setenv("SOME_API_KEY", "sk-123")
    assert config.get_key("SOME_API_KEY") == "sk-123"


def test_require_key_missing_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NOPE_API_KEY"):
        config.require_key("NOPE_API_KEY")


def test_div_policy_is_a_flag_only(monkeypatch):
    """`--div-policy` is a module:attr or None; no environment variable feeds it (D101)."""
    from stylebot import config

    monkeypatch.setenv("STYLEBOT_DIV_POLICY", "operator:truth")
    assert config.resolve_div_policy(None) is None
    assert config.resolve_div_policy("operator:truth")("callout-note") is True
    assert config.resolve_div_policy("operator:not_")("callout-note") is False


def test_div_policy_bad_spec_is_a_config_error():
    from stylebot import config
    from stylebot.errors import ConfigError

    for spec in ("nocolon", "operator:", "no.such.module:x", "operator:no_such_attr"):
        with pytest.raises(ConfigError, match="--div-policy"):
            config.resolve_div_policy(spec)
