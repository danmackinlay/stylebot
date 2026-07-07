"""Direct tests for stylebot.capture — the Phase-1 mechanism as a library.

The CLI-level behaviour (heading context, opt-outs) is pinned in
test_ai_style_log.py through the `ai-style-log` wrappers; these exercise the
library surface with explicit paths — no module globals, no monkeypatching —
which is exactly what the extraction buys a non-CLI producer.
"""

from __future__ import annotations

import json

import pytest

from stylebot import capture
from stylebot.pairs import validate_pairs_file


def test_capture_pairs_writes_valid_records_to_explicit_path(tmp_path):
    pairs_path = tmp_path / "deep" / "pairs.jsonl"
    written = capture.capture_pairs(
        "## S\n\nIt is worth noting the rich tapestry.",
        "## S\n\nWe dig in.",
        pairs_path=pairs_path,
        source="post/foo.qmd",
        snapped_at="2026-07-07T00:00:00+00:00",
    )
    assert len(written) == 1
    assert written[0]["meta"]["context"] == "## S"
    assert validate_pairs_file(pairs_path) == []
    on_disk = [json.loads(ln) for ln in pairs_path.read_text().splitlines()]
    assert on_disk == written


def test_capture_pairs_raises_when_nothing_changed(tmp_path):
    with pytest.raises(ValueError, match="no changed text chunks"):
        capture.capture_pairs(
            "same text", "same text", pairs_path=tmp_path / "pairs.jsonl",
            source=None, snapped_at=None,
        )


def test_remove_pairs_for_source_roundtrips_corrupt_lines(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    keep = {"messages": [], "meta": {"source": "keep.qmd"}}
    drop = {"messages": [], "meta": {"source": "drop.qmd"}}
    pairs_path.write_text(
        json.dumps(keep) + "\n" + "not json at all\n" + json.dumps(drop) + "\n",
        encoding="utf-8",
    )
    assert capture.remove_pairs_for_source(pairs_path, "drop.qmd") == 1
    lines = pairs_path.read_text(encoding="utf-8").splitlines()
    # The kept record survives AND the corrupt line is preserved verbatim.
    assert lines == [json.dumps(keep, ensure_ascii=False), "not json at all"]
