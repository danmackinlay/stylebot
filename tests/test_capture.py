"""Direct tests for stylebot.capture — the Phase-1 mechanism as a library.

The CLI-level behaviour (heading context, opt-outs) is pinned in
test_ai_style_log.py through the `ai-style log` wrappers; these exercise the
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


def test_capture_pairs_archives_full_texts_for_rechunking(tmp_path):
    """The chunk records are a lossy view; the archive is the re-cuttable raw.

    diff_chunks drops unchanged paragraphs and document offsets, so longer-
    scale re-chunking (v2 multi-paragraph pairs) depends on this archive
    existing from capture day one.
    """
    before = "## S\n\nKept paragraph.\n\nIt is worth noting the rich tapestry.\n\nAlso kept."
    after = "## S\n\nKept paragraph.\n\nWe dig in.\n\nAlso kept."
    written = capture.capture_pairs(
        before,
        after,
        pairs_path=tmp_path / "pairs.jsonl",
        source="post/foo.qmd",
        snapped_at="2026-07-07T00:00:00+00:00",
    )
    cap_id = written[0]["meta"]["capture_id"]
    archive = json.loads((tmp_path / "captures" / f"{cap_id}.json").read_text())
    # Byte-exact full texts — the unchanged paragraphs the chunk view lost.
    assert archive["before_text"] == before
    assert archive["after_text"] == after
    assert archive["source"] == "post/foo.qmd"
    assert archive["whole"] is False


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


# --- gap coalescing (v2 longer rewrites) ----------------------------------

# Two edits with one untouched paragraph between them — the shape of a
# multi-paragraph rewrite that today fragments into two independent pairs.
_KEPT = "An untouched paragraph sitting between the two edits, left exactly as it was."
_BEFORE = f"It is worth noting the rich tapestry.\n\n{_KEPT}\n\nMoreover, we must delve deeper.\n"
_AFTER = f"We dig in.\n\n{_KEPT}\n\nSo we dig further.\n"


def test_diff_chunks_default_does_not_bridge_gaps():
    """Legacy contract: only CONTIGUOUS changed paragraphs coalesce."""
    triples = capture.diff_chunks(_BEFORE, _AFTER)
    assert len(triples) == 2
    assert all(_KEPT not in after for _before, after, _ctx in triples)


def test_diff_chunks_bridges_a_short_gap():
    triples = capture.diff_chunks(
        _BEFORE, _AFTER, coalesce_gap_paras=2, coalesce_gap_chars=800
    )
    assert len(triples) == 1
    before, after, _ctx = triples[0]
    # The bridged paragraph appears on BOTH sides — it is context inside one
    # edit, not a rewrite target of its own.
    assert _KEPT in before and _KEPT in after
    assert "We dig in." in after and "So we dig further." in after


def test_diff_chunks_respects_gap_tolerances():
    """A gap wider than either tolerance leaves the regions separate."""
    long_gap = "\n\n".join(f"Untouched paragraph number {i}, quite long indeed." * 3 for i in range(4))
    before = f"It is worth noting the tapestry.\n\n{long_gap}\n\nMoreover, we delve.\n"
    after = f"We dig in.\n\n{long_gap}\n\nSo we dig further.\n"
    # Blocked by the paragraph count.
    assert len(capture.diff_chunks(before, after, coalesce_gap_paras=2, coalesce_gap_chars=99999)) == 2
    # Blocked by the char budget.
    assert len(capture.diff_chunks(before, after, coalesce_gap_paras=99, coalesce_gap_chars=50)) == 2


def test_diff_chunks_oversized_merge_falls_back_to_parts():
    """A coalesced region over max_chars degrades to the legacy cut rather
    than dropping a real edit on the floor."""
    big_gap = "A long untouched paragraph. " * 40
    before = f"It is worth noting the tapestry.\n\n{big_gap}\n\nMoreover, we delve.\n"
    after = f"We dig in.\n\n{big_gap}\n\nSo we dig further.\n"
    kw = dict(coalesce_gap_paras=2, coalesce_gap_chars=99999)
    assert len(capture.diff_chunks(before, after, **kw)) == 1  # merges when uncapped
    capped = capture.diff_chunks(before, after, max_chars=500, **kw)
    assert len(capped) == 2  # too big merged -> the two original regions
    assert all(big_gap.strip() not in after_ for _b, after_, _c in capped)


def test_coalescing_preserves_pure_insert_delete_skip():
    """Bridging must not resurrect a pure insert/delete as a training pair."""
    before = "It is worth noting the tapestry.\n\nA paragraph that gets deleted entirely.\n"
    after = "We dig in.\n"
    triples = capture.diff_chunks(before, after, coalesce_gap_paras=2, coalesce_gap_chars=800)
    assert all(b.strip() and a.strip() for b, a, _c in triples)


# --- recut from the archive -----------------------------------------------


def _capture(pairs_path, before, after, source):
    return capture.capture_pairs(
        before, after, pairs_path=pairs_path, source=source,
        snapped_at="2026-08-01T00:00:00+00:00",
    )


def test_recut_rederives_pairs_from_the_archive(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    assert len(capture.read_jsonl(pairs_path)) == 2  # default cut fragments

    recut = capture.recut_archives(
        pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0
    )
    assert len(recut) == 1  # bridged into one multi-paragraph pair
    assert recut[0]["meta"]["recut"] == {"gap_paras": 2, "gap_chars": 800}
    # The old fragmented pairs for this capture are gone, not stacked on.
    on_disk = capture.read_jsonl(pairs_path)
    assert len(on_disk) == 1
    assert _KEPT in on_disk[0]["messages"][2]["content"]


def test_recut_is_idempotent(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    kw = dict(coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    capture.recut_archives(pairs_path, **kw)
    first = capture.read_jsonl(pairs_path)
    capture.recut_archives(pairs_path, **kw)
    second = capture.read_jsonl(pairs_path)
    assert len(first) == len(second) == 1
    assert first[0]["messages"] == second[0]["messages"]


def test_recut_leaves_unarchived_captures_alone(tmp_path):
    """Pre-archive captures have nothing to re-cut from, so they keep their
    native scale rather than being dropped."""
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    legacy = capture.build_pair_record(
        before_body="It is worth noting the tapestry.", after_body="We dig in.",
        source="post/legacy.qmd", snapped_at=None,
        before_frontmatter=None, after_frontmatter=None,
        capture_id="deadbeef", chunk_index=0, chunk_total=1,
    )
    capture.append_pair(pairs_path, legacy)

    capture.recut_archives(
        pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0
    )
    ids = {r["meta"]["capture_id"] for r in capture.read_jsonl(pairs_path)}
    assert "deadbeef" in ids  # untouched


def test_recut_dry_run_does_not_write(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    before = pairs_path.read_text(encoding="utf-8")
    recut = capture.recut_archives(
        pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800,
        min_chars=0, dry_run=True,
    )
    assert recut and pairs_path.read_text(encoding="utf-8") == before


def test_recut_output_validates(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    capture.recut_archives(
        pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0
    )
    assert validate_pairs_file(pairs_path) == []
