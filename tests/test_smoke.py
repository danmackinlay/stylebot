"""Two CLI seams nothing else reaches: `log pair` writes the shared
chat-completion schema downstream relies on, and `run --anchor-log` leaves a
joinable sidecar. Exit codes are test_errors'; defaults are click's."""

from __future__ import annotations

import json

from click.testing import CliRunner
from conftest import corpus_records

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin import ai_style
from stylebot.bin.ai_style_log import main


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

    records = corpus_records(tmp_path)
    assert len(records) == 1
    rec = records[0]
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"] == STYLE_SYSTEM
    assert rec["meta"]["source"] == "t"


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
