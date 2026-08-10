"""Smoke tests: the package imports, the CLI runs, the schema contract holds,
and the corpus location is honoured. Cheap guardrails for the fan-out — a
subagent breaking the pairs.jsonl contract should fail here."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from stylebot import config
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin import ai_style
from stylebot.bin.ai_style_log import diff_chunks, main


def test_cli_help_runs():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Capture" in result.output


def test_pair_writes_contract_schema(tmp_path, monkeypatch):
    """`pair` must emit the shared chat-completion schema downstream relies on."""
    # The logger resolves its corpus per command through `config`, so the env
    # override lands without reimporting anything.
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(tmp_path))

    before = tmp_path / "b.txt"
    after = tmp_path / "a.txt"
    before.write_text("It is worth noting that we delve into the tapestry.\n")
    after.write_text("We dig into it.\n")

    result = CliRunner().invoke(
        main,
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


def test_diff_chunks_skips_pure_insert():
    # Pure insert (no before content) yields no learnable pair.
    assert diff_chunks("", "brand new paragraph") == []
    # A real replacement yields exactly one pair.
    pairs = diff_chunks("old slop here", "tight prose")
    assert len(pairs) == 1


# --- the shared styler-backend surface (`run` and `serve`) ----------------
#
# `BackendOpts` folded twelve duplicated declarations into one bundle. These
# pin what that dedup must not lose: every flag still reaches both commands,
# and the two deliberate per-command differences survive.

_BACKEND_FLAGS = (
    "--backend",
    "--sampler-path",
    "--base-model",
    "--renderer",
    "--base-url",
    "--model",
    "--api-key-env",
    "--temperature",
    "--max-tokens",
    "--max-chunk-chars",
    "--inline-headings",
    "--detector-model",
)


@pytest.mark.parametrize("command", ["run", "serve"])
def test_backend_flags_reach_both_commands(command):
    result = CliRunner().invoke(ai_style.main, [command, "--help"])
    assert result.exit_code == 0, result.output
    missing = [flag for flag in _BACKEND_FLAGS if flag not in result.output]
    assert not missing, missing
    assert "Styler backend" in result.output  # the sectioned --help still renders


def _params(command) -> dict:
    return {p.name: p for p in ai_style.main.commands[command].params}


def test_backend_defaults_differ_where_they_are_meant_to():
    """run defaults to the tinker backend at 0.3; serve to score-only at 0.7
    (interactive best-of wants distinct candidates)."""
    run, serve = _params("run"), _params("serve")
    assert (run["backend"].default, run["temperature"].default) == ("tinker", 0.3)
    assert (serve["backend"].default, serve["temperature"].default) == (None, 0.7)
    # serve cannot serve without its detector; run's is optional (--best-of only).
    assert serve["detector_model"].required is True
    assert run["detector_model"].required is False
    result = CliRunner().invoke(ai_style.main, ["serve"])
    assert result.exit_code == 2
    assert "--detector-model" in result.output


@pytest.mark.parametrize(
    ("args", "exit_code", "message"),
    [
        # A flag combination the user must fix is a UsageError -> exit 2 ...
        (["--backend", "tinker"], 2, "needs --sampler-path and --base-model"),
        (["--backend", "openai"], 2, "needs --base-url and --model"),
        # ... an unset key is an environment failure -> ClickException, exit 1.
        (
            [
                "--backend",
                "openai",
                "--base-url",
                "http://x/v1",
                "--model",
                "m",
                "--api-key-env",
                "NOPE_UNSET",
            ],
            1,
            "NOPE_UNSET is not set",
        ),
    ],
)
def test_run_backend_errors_keep_their_exit_codes(tmp_path, monkeypatch, args, exit_code, message):
    monkeypatch.delenv("NOPE_UNSET", raising=False)
    doc = tmp_path / "d.qmd"
    doc.write_text("Some prose.\n", encoding="utf-8")
    result = CliRunner().invoke(ai_style.main, ["run", str(doc), *args])
    assert result.exit_code == exit_code, result.output
    assert message in result.output


@pytest.mark.xfail(
    reason="depends on worktree A's infer.rewrite_text(inline_anchors=...)",
    raises=TypeError,
    strict=True,
)
def test_run_strict_anchors_reaches_rewrite_text(tmp_path, monkeypatch):
    """--strict-anchors is the CLI face of A's `inline_anchors` guard.

    Deliberately does NOT stub `rewrite_text`: the point is that the real
    signature accepts the keyword. Flip this to a plain test when A merges.
    """
    from stylebot import infer

    monkeypatch.setattr(
        infer,
        "openai_backend",
        lambda *a, **k: lambda messages, num_samples=1: [messages[1]["content"]] * num_samples,
    )
    doc = tmp_path / "d.qmd"
    doc.write_text("## H\n\nSome prose here, long enough to chunk.\n", encoding="utf-8")
    args = ["run", str(doc), "--backend", "openai", "--base-url", "http://x/v1", "--model", "m"]
    result = CliRunner().invoke(ai_style.main, [*args, "--strict-anchors"], catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_config_require_key_is_actionable(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        config.require_key("DEFINITELY_UNSET_KEY")
    msg = str(excinfo.value)
    # Name the key, and say how to supply it — direnv is how Dan's shell does it.
    assert "DEFINITELY_UNSET_KEY" in msg
    assert "direnv" in msg
