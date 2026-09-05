"""`stylebot.scoring`: the corpus-scale pass. Per-passage signals stay in
`test_eval.py`, which also holds the older `score_pairs_file` tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from stylebot import eval as ev
from stylebot import scoring
from stylebot.errors import ConfigError, RunError
from stylebot.jsonl import append_jsonl
from stylebot.pairs import iter_pairs, record_id

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
    assert len(scores.read_text().splitlines()) == 2 * 2  # 2 pairs x (slop, target), vale
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
    row = scoring.load_scores(opts.out_path)[0]
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


# --- rows keyed on `(record_id, field, signal)` (D39, W4 step 5 item 5) -----
#
# One row per signal, so a signal is independently addressable: adding `--judge`
# to a detector-scored file appends judge rows and rewrites nothing.


def _detector(score: float = 0.4):
    identity = {"name": "voice-clf", "embed_model": "sd", "git_sha": "abc1234"}

    def detect(prose: str) -> dict:
        return {"score": score, "configured": True, **identity}

    detect.identity = identity
    return detect


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_score_row_is_keyed_on_the_record_the_field_and_the_signal(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path, 2)
    out = tmp_path / "scores.jsonl"

    scoring.score_pairs_file(pairs, out, detector=_detector())

    keys = {(r["id"], r["field"], r["signal"]) for r in _rows(out)}
    # 2 records x (slop, target) x (vale, detector)
    assert len(keys) == len(_rows(out)) == 2 * 2 * 2
    assert {k[2] for k in keys} == {"vale", "detector"}
    assert scoring.existing_score_keys(out) == keys
    # …and folds back to exactly the nested record every reader still expects.
    rec = scoring.load_scores(out)[0]
    assert set(rec) == {"id", "meta", "scores"}
    assert rec["scores"]["slop"]["detector"]["score"] == 0.4
    assert rec["scores"]["slop"]["judge"] is None


def test_adding_judge_appends_only_judge_rows_and_leaves_the_rest_byte_identical(
    tmp_path, monkeypatch
):
    """The RUNBOOK §4 footgun, as a test: a second pass that adds a signal must
    not re-buy, re-write or skip anything — the detector rows stay byte-for-byte."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path, 3)
    out = tmp_path / "scores.jsonl"

    scoring.score_pairs_file(pairs, out, detector=_detector())
    before = out.read_text()

    res = scoring.score_pairs_file(pairs, out, detector=_detector(), judge=_judge, max_workers=2)

    after = out.read_text()
    assert after.startswith(before)  # nothing existing was rewritten or reordered
    appended = [json.loads(line) for line in after[len(before) :].splitlines() if line.strip()]
    assert {r["signal"] for r in appended} == {"judge"}
    assert len(appended) == 3 * 2 and res.written == 3 and res.skipped_existing == 0
    folded = scoring.load_scores(out)
    assert all(f["judge"]["score"] == 4 for r in folded for f in r["scores"].values())
    assert all(f["detector"]["score"] == 0.4 for r in folded for f in r["scores"].values())
    # A third pass with the same signals has nothing left to do.
    again = scoring.score_pairs_file(pairs, out, detector=_detector(), judge=_judge, max_workers=2)
    assert (again.written, again.skipped_existing) == (0, 3) and out.read_text() == after


def test_an_unconfigured_detector_writes_no_row_so_a_real_head_is_not_skipped(
    tmp_path, monkeypatch
):
    """A null head produces no detector result, so it must claim no key: leaving
    a `configured: false` stub behind would make a later real head a no-op."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path, 2)
    out = tmp_path / "scores.jsonl"

    scoring.score_pairs_file(pairs, out)
    assert {r["signal"] for r in _rows(out)} == {"vale"}

    res = scoring.score_pairs_file(pairs, out, detector=_detector())
    assert res.written == 2 and {r["signal"] for r in _rows(out)} == {"vale", "detector"}
    assert scoring.detector_identities(out) == [
        {
            "name": "voice-clf",
            "embed_model": "sd",
            "git_sha": "abc1234",
            "trained_at": None,
            "n_pairs": None,
        }
    ]


def _legacy_row(rid: str, judge_score: int) -> dict:
    """The pre-D39 shape: every signal of every field nested in one row."""
    signals = {
        "text_chars": 42,
        "vale": {"available": True, "alerts": 0, "by_severity": {}, "raw": []},
        "judge": {"score": judge_score, "rationale": "paid for once"},
        "detector": {
            "score": 0.4,
            "configured": True,
            "name": "voice-clf",
            "embed_model": "sd",
            "git_sha": "abc1234",
        },
        "eyeball": None,
    }
    return {
        "id": rid,
        "meta": {"source": "notebook/a.qmd"},
        "scores": {"slop": dict(signals), "target": dict(signals)},
    }


def test_a_legacy_nested_scores_file_explodes_and_resumes_without_rebuying(tmp_path, monkeypatch):
    """The live file is the only copy of its rows. Exploding it seeds the keyed
    layout losslessly, and resume then skips those records: nothing is re-bought."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path, 2)
    ids = [record_id(rec) for rec in iter_pairs(pairs)]
    out = tmp_path / "scores.jsonl"
    legacy = [_legacy_row(rid, 5) for rid in ids]
    for row in legacy:
        append_jsonl(out, row)

    exploded = scoring.explode_score_row(legacy[0])
    assert len(exploded) == 2 * 3  # (slop, target) x (vale, judge, detector); eyeball is null
    assert scoring.load_scores(out) == legacy  # folds back to the very same records

    calls: list[str] = []

    async def counting_judge(prose: str) -> dict:
        calls.append(prose)
        return {"score": 1, "rationale": "should never run"}

    res = scoring.score_pairs_file(
        pairs, out, detector=_detector(), judge=counting_judge, max_workers=2
    )
    assert (res.written, res.skipped_existing) == (0, 2) and calls == []
    assert scoring.load_scores(out) == legacy


def test_the_detector_guard_runs_before_the_prune_touches_the_file(tmp_path, monkeypatch):
    """D39's guard is a pre-write check: a refused run must not have rewritten
    the file on its way to raising, and the prune is a rewrite."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    pairs = _corpus(tmp_path, 1)
    out = tmp_path / "scores.jsonl"
    append_jsonl(out, _legacy_row(record_id(next(iter(iter_pairs(pairs)))), 5))
    append_jsonl(out, _legacy_row("orphan-not-in-the-corpus", 5))  # a prune would drop this
    before = out.read_text()

    other = _detector()
    other.identity = dict(other.identity, git_sha="9999999")
    with pytest.raises(ConfigError) as excinfo:
        scoring.score_pairs_file(pairs, out, detector=other)

    assert out.read_text() == before
    assert "scores-voice-clf@9999999.jsonl" in str(excinfo.value)  # where THIS head goes


def test_scores_filename_names_the_head_that_wrote_the_rows(tmp_path):
    assert scoring.scores_filename(ev.null_detector) == "scores.jsonl"
    assert scoring.scores_filename(_detector()) == "scores-voice-clf@abc1234.jsonl"
