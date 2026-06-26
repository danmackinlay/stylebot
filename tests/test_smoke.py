"""Smoke tests: the package imports, the CLI runs, the schema contract holds,
and the corpus location is honoured. Cheap guardrails for the fan-out — a
subagent breaking the pairs.jsonl contract should fail here."""

from __future__ import annotations

import json

from click.testing import CliRunner

from stylebot import config
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style_log import diff_chunks, main


def test_cli_help_runs():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Capture" in result.output


def test_pair_writes_contract_schema(tmp_path, monkeypatch):
    """`pair` must emit the shared chat-completion schema downstream relies on."""
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(tmp_path))
    # Re-import module-level paths so the env override takes effect.
    import importlib

    from stylebot.bin import ai_style_log

    importlib.reload(ai_style_log)

    before = tmp_path / "b.txt"
    after = tmp_path / "a.txt"
    before.write_text("It is worth noting that we delve into the tapestry.\n")
    after.write_text("We dig into it.\n")

    result = CliRunner().invoke(
        ai_style_log.main,
        ["pair", "--before", str(before), "--after", str(after), "--source", "t"],
    )
    assert result.exit_code == 0, result.output

    lines = (tmp_path / "pairs.jsonl").read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == STYLE_SYSTEM
    assert rec["meta"]["source"] == "t"

    # restore default module state for other tests
    importlib.reload(ai_style_log)


def test_diff_chunks_skips_pure_insert():
    # Pure insert (no before content) yields no learnable pair.
    assert diff_chunks("", "brand new paragraph") == []
    # A real replacement yields exactly one pair.
    pairs = diff_chunks("old slop here", "tight prose")
    assert len(pairs) == 1


def test_config_require_key_is_actionable(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_KEY", raising=False)
    config._LOADED = True  # skip .env load in the sandbox
    try:
        config.require_key("DEFINITELY_UNSET_KEY")
    except RuntimeError as exc:
        assert "DEFINITELY_UNSET_KEY" in str(exc)
    else:
        raise AssertionError("expected RuntimeError for missing key")
