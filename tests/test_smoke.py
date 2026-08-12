"""Smoke tests: the package imports, the CLI runs, the schema contract holds,
and the corpus location is honoured. Cheap guardrails for the fan-out — a
subagent breaking the pairs.jsonl contract should fail here."""

from __future__ import annotations

import importlib
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


# --- CLI defaults are the library constants, not copies of them -----------
#
# Four commands used to declare each default three times: `default=None`, a
# hand-written `[default: X]` in the help string, and an `x if x is not None
# else DEFAULT_X` translation at the call site. Nothing tied the three
# together, so a library constant could move and `--help` would keep lying.
# click owns them now; this is the test that keeps it that way.


@pytest.mark.parametrize(
    ("command", "param", "constant"),
    [
        ("train", "base_model", "stylebot.train:DEFAULT_BASE_MODEL"),
        ("train", "lora_rank", "stylebot.train:DEFAULT_LORA_RANK"),
        ("train", "num_epochs", "stylebot.train:DEFAULT_NUM_EPOCHS"),
        ("train", "batch_size", "stylebot.train:DEFAULT_BATCH_SIZE"),
        ("train", "max_length", "stylebot.train:DEFAULT_MAX_LENGTH"),
        ("train", "val_frac", "stylebot.train:DEFAULT_VAL_FRAC"),
        ("train-clf", "embed_model", "stylebot.classify_train:DEFAULT_EMBED_MODEL"),
        ("train-clf", "test_size", "stylebot.classify_train:DEFAULT_TEST_SIZE"),
        ("train-clf", "n_splits", "stylebot.classify_train:DEFAULT_N_SPLITS"),
        ("make-splits", "eval_frac", "stylebot.splits:DEFAULT_EVAL_FRAC"),
        ("make-splits", "detector_frac", "stylebot.splits:DEFAULT_DETECTOR_FRAC"),
    ],
)
def test_cli_default_is_the_library_constant(command, param, constant):
    module_name, attr = constant.split(":")
    module = importlib.import_module(module_name)
    assert _params(command)[param].default is getattr(module, attr)


@pytest.mark.parametrize("command", ["train", "train-clf", "make-splits"])
def test_no_hand_written_default_strings(command):
    """A `[default: …]` in a help string is click's job, and a second copy to
    drift. The survivors name a *computed* default in prose ("Default: …")."""
    result = CliRunner().invoke(ai_style.main, [command, "--help"])
    assert result.exit_code == 0, result.output
    for p in ai_style.main.commands[command].params:
        assert "[default:" not in (getattr(p, "help", None) or ""), p.name


def test_computed_defaults_stay_none_and_say_so():
    """`--max-tokens`/`--max-chunk-chars` resolve against `infer` at backend
    construction (BackendOpts.tokens/chunk_chars), `--run-id` at launch, `--lr`
    and `--renderer` inside the trainer against the chosen base model. None is
    the sentinel that means "compute me", so it must survive — but the help has
    to keep saying what gets computed."""
    from stylebot import infer

    run = _params("run")
    for name in ("max_tokens", "max_chunk_chars"):
        assert run[name].default is None
    # The two numbers the backend help quotes ARE these constants.
    assert str(infer.STYLE_CHARS_PER_CHUNK) in run["max_chunk_chars"].help
    assert str(infer.DEFAULT_MAX_TOKENS) in run["max_tokens"].help

    train = _params("train")
    for name in ("run_id", "learning_rate", "renderer_name"):
        assert train[name].default is None
        assert "efault" in train[name].help


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


def test_run_anchor_log_writes_a_joinable_sidecar(tmp_path, monkeypatch):
    """`--anchor-log` is the CLI face of step 0: the counters must outlive the
    run, or the question masking is gated on stays unanswerable."""
    from stylebot import infer

    monkeypatch.setattr(
        infer,
        "openai_backend",
        lambda *a, **k: (
            lambda messages, num_samples=1: (
                ["Read the notes, invented [here](https://fake.example/x)."] * num_samples
            )
        ),
    )
    doc = tmp_path / "d.qmd"
    doc.write_text("Read [the notes](./real.qmd) whenever you get the chance.\n", encoding="utf-8")
    log = tmp_path / "anchors.jsonl"
    args = ["run", str(doc), "--backend", "openai", "--base-url", "http://x/v1", "--model", "m"]
    result = CliRunner().invoke(
        ai_style.main, [*args, "--anchor-log", str(log)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    (line,) = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert line["source"] == str(doc)
    assert line["source_anchors"] == ["./real.qmd"]
    (cand,) = line["candidates"]
    assert cand["lost"] == ["./real.qmd"] and cand["gained"] == ["https://fake.example/x"]
    # And the run said so, by cause.
    assert "1 lost, 1 invented" in result.output


def test_config_require_key_is_actionable(monkeypatch):
    monkeypatch.delenv("DEFINITELY_UNSET_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        config.require_key("DEFINITELY_UNSET_KEY")
    msg = str(excinfo.value)
    # Name the key, and say how to supply it — direnv is how Dan's shell does it.
    assert "DEFINITELY_UNSET_KEY" in msg
    assert "direnv" in msg
