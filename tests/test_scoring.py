"""`stylebot.scoring`: the corpus-scale pass. Per-passage signals stay in
`test_eval.py`, which also holds the older `score_pairs_file` tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from stylebot import eval as ev
from stylebot import scoring
from stylebot.errors import RunError
from stylebot.jsonl import append_jsonl
from stylebot.pairs import record_id

_VALE = {"available": True, "alerts": 0, "by_severity": {}, "raw": []}


def _record(i: int = 0) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"[slop] Paragraph {i}, long enough to score."},
            {"role": "assistant", "content": f"Paragraph {i}, long enough to score."},
        ],
        "meta": {"source": f"post/p{i}.qmd", "chunk_index": 0, "synth_key": f"k{i}"},
    }


async def _judge(prose: str) -> dict:
    return {"score": 4, "rationale": "fine"}


def test_a_missing_judge_slot_raises_rather_than_storing_a_coroutine():
    """`_score_record` reads the judge pass by field; a hole in it is an error.

    Falling through to `score_candidate`'s inline call would hand the async
    judge's coroutine object to the record as its score.
    """
    rec = _record()
    with pytest.raises(RunError) as excinfo:
        scoring._score_record(
            rec,
            fields=("slop",),
            judge=_judge,
            detector=ev.null_detector,
            vale_config=None,
            vale_by_field={"slop": _VALE},
            detector_by_field={"slop": ev.null_detector("")},
            judge_by_field={},
        )
    message = str(excinfo.value)
    assert record_id(rec) in message and "slop" in message


def test_a_missing_judge_slot_fails_its_record_through_the_async_entry(tmp_path, monkeypatch):
    """The same hole reached through `score_pairs_file_async` lands in `errors`,
    named per record, and never reaches the file (a coroutine is not JSON)."""
    pairs = tmp_path / "pairs.jsonl"
    for i in range(2):
        append_jsonl(pairs, _record(i))
    out = tmp_path / "scores.jsonl"
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)

    async def empty_prepass(records, **kwargs):
        return [{} for _ in records]

    monkeypatch.setattr(scoring, "_judge_prepass", empty_prepass)

    res = asyncio.run(scoring.score_pairs_file_async(pairs, out, judge=_judge, max_workers=2))
    assert res.written == 0
    assert [rid for rid, _ in res.errors] == [record_id(_record(0)), record_id(_record(1))]
    assert all(msg.startswith("RunError:") and "slop" in msg for _, msg in res.errors)
    assert out.read_text() == ""


# --- `run_eval`: the `eval` pass as a library (D78) -------------------------


def _corpus(tmp_path, n: int = 2):
    pairs = tmp_path / "pairs.jsonl"
    for i in range(n):
        rec = _record(i)
        rec["meta"]["slop_strategy"] = "polish" if i % 2 else "catalogue"
        append_jsonl(pairs, rec)
    return pairs


def _recording_emitter():
    lines: list[tuple[str, str]] = []

    def emit(message: str, /, *, level: str = "info") -> None:
        lines.append((level, message))

    return emit, lines


def test_run_eval_scores_keyless_and_facets_the_summary(tmp_path, monkeypatch):
    """Null detector, no judge, Vale absent: the pass still writes, reads back
    and reports — the `--by` lines carry both means in the `det:/judge:` form."""
    from stylebot.verdict import EvalOpts, VerdictOpts

    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path)
    opts = EvalOpts(pairs_path=pairs, by="slop_strategy", summary_path=tmp_path / "summary.json")
    emit, lines = _recording_emitter()

    summary = scoring.run_eval(opts, VerdictOpts(), emit=emit)

    scores = tmp_path / "scores.jsonl"  # the default: beside the corpus
    assert len(scores.read_text().splitlines()) == 2
    assert summary["pairs"] == 2 and set(summary["by"]) == {"catalogue", "polish"}
    assert json.loads(opts.summary_path.read_text()) == summary
    messages = [m for _, m in lines]
    assert f"scored 2 pair(s) (0 already scored, skipped) -> {scores}" in messages
    assert "by slop_strategy:" in messages
    assert "  catalogue: slop=det:None/judge:None, target=det:None/judge:None" in messages
    assert f"wrote summary -> {opts.summary_path}" in messages
    assert ("status", "  ... 2/2 scored") in lines

    # Resumable through the same door: a second pass scores nothing new.
    again = scoring.run_eval(opts, VerdictOpts(), emit=emit)
    assert again == summary
    assert f"scored 0 pair(s) (2 already scored, skipped) -> {scores}" in [m for _, m in lines]


def test_run_eval_scores_a_document_pass_at_its_serving_scale(tmp_path, monkeypatch):
    """A finished `DocumentPass` swaps the corpus, adds `output`, turns pruning
    off and wraps the detector at the pass's `chunk_scale` (D54)."""
    from stylebot.docscale import DocumentPass
    from stylebot.verdict import EvalOpts, VerdictOpts

    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path)
    docs = tmp_path / "documents.jsonl"
    doc = _record(0)
    doc["output"] = "\n\n".join(f"Rewritten paragraph {i}, long enough to score." for i in range(3))
    doc["meta"]["capture_id"], doc["meta"]["scale"] = "doc:cap-1", "document"
    doc["meta"].pop("synth_key")
    append_jsonl(docs, doc)
    seen: list[str] = []

    def detector(prose: str) -> dict:
        seen.append(prose)
        return {"score": 0.5, "configured": True, "name": "stub"}

    detector.identity = {"name": "stub", "embed_model": "e", "git_sha": "abc"}
    passed = DocumentPass(docs, ("slop", "target", "output"), False, {"max_chunk_chars": 60})
    opts = EvalOpts(pairs_path=pairs, out_path=tmp_path / "doc-scores.jsonl")

    summary = scoring.run_eval(opts, VerdictOpts(), detector=detector, documents=passed)

    assert set(summary["fields"]) == {"slop", "target", "output"}
    row = json.loads(opts.out_path.read_text().splitlines()[0])
    assert row["scores"]["output"]["detector"]["n_chunks"] == 3
    assert sum("Rewritten paragraph" in text for text in seen) == 3
    assert opts.out_path.exists() and not (tmp_path / "scores.jsonl").exists()


def test_ai_style_eval_is_the_pass_behind_flags(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from stylebot.bin.ai_style import main

    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path)
    result = CliRunner().invoke(
        main, ["eval", "--pairs", str(pairs), "--by", "slop_strategy", "--field", "output"]
    )
    assert result.exit_code == 0, result.output
    assert "scored 2 pair(s)" in result.output and "by slop_strategy:" in result.output
    assert "output=det:None/judge:None" in result.output

    refused = CliRunner().invoke(main, ["eval", "--pairs", str(pairs), "--documents"])
    assert refused.exit_code == 2 and "--documents-manifest" in refused.output
