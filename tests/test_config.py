"""Tests for stylebot.config — the one home of the path/secret precedence ladder.

`config._LOADED` is forced True so `.env` discovery can't leak developer-local
values into the assertions; the ladder below `load()` is what's under test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stylebot import config


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    monkeypatch.setattr(config, "_LOADED", True)


def test_data_dir_flag_beats_env(monkeypatch):
    monkeypatch.setenv("STYLEBOT_DATA_DIR", "/from/env")
    assert config.resolve_data_dir("/from/flag") == Path("/from/flag")


def test_data_dir_env_beats_default(monkeypatch):
    monkeypatch.setenv("STYLEBOT_DATA_DIR", "/from/env")
    assert config.resolve_data_dir() == Path("/from/env")


def test_data_dir_default(monkeypatch):
    monkeypatch.delenv("STYLEBOT_DATA_DIR", raising=False)
    assert config.resolve_data_dir() == Path("_training_pairs")


def test_get_key_blank_is_none(monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "")
    assert config.get_key("SOME_API_KEY") is None
    monkeypatch.setenv("SOME_API_KEY", "sk-123")
    assert config.get_key("SOME_API_KEY") == "sk-123"


def test_require_key_missing_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NOPE_API_KEY"):
        config.require_key("NOPE_API_KEY")
