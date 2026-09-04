"""`stylebot.scoring`: the corpus-scale pass. Per-passage signals stay in
`test_eval.py`, which also holds the older `score_pairs_file` tests."""

from __future__ import annotations

import asyncio

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
