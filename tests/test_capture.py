"""Direct tests for stylebot.capture — the Phase-1 mechanism as a library.

The `ai-style log` opt-outs are pinned in test_ai_style_log.py through the
CLI wrappers; these exercise the library surface with explicit paths — no
module globals, no monkeypatching — which is exactly what the extraction buys a
non-CLI producer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import corpus_file, corpus_records

from stylebot import capture
from stylebot.errors import RunError
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
    assert validate_pairs_file(corpus_file(pairs_path.parent, tmp_path / "p.jsonl")) == []
    assert corpus_records(pairs_path.parent) == written


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
    archive_dir = tmp_path / "captures" / cap_id
    # The documents are ORDINARY FILES, openable and hand-editable. That is the
    # whole reason for the directory layout: a borked capture can be repaired.
    assert (archive_dir / "before.md").read_text(encoding="utf-8") == before
    assert (archive_dir / "after.md").read_text(encoding="utf-8") == after
    meta = json.loads((archive_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["source"] == "post/foo.qmd"
    assert meta["whole"] is False
    assert meta["capture_id"] == cap_id
    # .md, never .qmd — a .qmd here would be reformatted by the blog's
    # pre-commit hook and indexed by qmd search as if Dan had written it.
    assert not list(archive_dir.glob("*.qmd"))
    # And the flat reader still hands back `read_archive`'s flat shape.
    assert capture.read_archive(archive_dir)["before_text"] == before


def test_capture_pairs_raises_when_nothing_changed(tmp_path):
    with pytest.raises(RunError, match="no changed text chunks"):
        capture.capture_pairs(
            "same text",
            "same text",
            pairs_path=tmp_path / "pairs.jsonl",
            source=None,
            snapped_at=None,
        )


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
    triples = capture.diff_chunks(_BEFORE, _AFTER, coalesce_gap_paras=2, coalesce_gap_chars=800)
    assert len(triples) == 1
    before, after, _ctx = triples[0]
    # The bridged paragraph appears on BOTH sides — it is context inside one
    # edit, not a rewrite target of its own.
    assert _KEPT in before and _KEPT in after
    assert "We dig in." in after and "So we dig further." in after


def test_diff_chunks_respects_gap_tolerances():
    """A gap wider than either tolerance leaves the regions separate."""
    long_gap = "\n\n".join(
        f"Untouched paragraph number {i}, quite long indeed." * 3 for i in range(4)
    )
    before = f"It is worth noting the tapestry.\n\n{long_gap}\n\nMoreover, we delve.\n"
    after = f"We dig in.\n\n{long_gap}\n\nSo we dig further.\n"
    # Blocked by the paragraph count.
    chunks = capture.diff_chunks(before, after, coalesce_gap_paras=2, coalesce_gap_chars=99999)
    assert len(chunks) == 2
    # Blocked by the char budget.
    chunks = capture.diff_chunks(before, after, coalesce_gap_paras=99, coalesce_gap_chars=50)
    assert len(chunks) == 2


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


# ---- protected blocks never enter a pair (D100) ----

_FENCE = "```python\nx = 0\n```"
_MATH = "$$\nE = mc^2\n$$"


def test_diff_chunks_ignores_an_edit_inside_a_fence():
    """A fence between two prose paragraphs, the edit inside the fence only:
    no pair, and `capture_pairs` says so rather than landing code."""
    before = f"Prose one.\n\n{_FENCE}\n\nProse two.\n"
    after = before.replace("x = 0", "x = 1")
    assert capture.diff_chunks(before, after) == []
    with pytest.raises(RunError, match="no changed text chunks"):
        capture.capture_pairs(
            before, after, pairs_path=Path("unused"), source=None, snapped_at=None
        )


def test_diff_chunks_ignores_display_math_and_keeps_a_heading():
    """Display math is protected like a fence; a heading paragraph is content (D64)."""
    before = f"## Old title\n\nProse.\n\n{_MATH}\n"
    after = f"## New title\n\nProse.\n\n{_MATH.replace('2', '3')}\n"
    assert capture.diff_chunks(before, after) == [("## Old title", "## New title", "")]


def test_diff_chunks_pairs_prose_either_side_of_an_edited_fence():
    """Edits in the prose before and after a fence that also changed: two
    prose pairs, the fence in neither, and coalescing never bridges it."""
    before = f"Prose one.\n\n{_FENCE}\n\nProse two.\n"
    after = f"Prose won.\n\n{_FENCE.replace('0', '1')}\n\nProse too.\n"
    for kw in ({}, dict(coalesce_gap_paras=5, coalesce_gap_chars=5000)):
        triples = capture.diff_chunks(before, after, **kw)
        assert [(b, a) for b, a, _c in triples] == [
            ("Prose one.", "Prose won."),
            ("Prose two.", "Prose too."),
        ]


def test_diff_chunks_trims_a_fence_from_a_regions_edge():
    """An edit spanning a prose paragraph and the adjacent fence keeps the
    prose side only, still a byte-exact substring of the before text."""
    before = f"Prose one.\n\n{_FENCE}\n"
    after = f"Prose won.\n\n{_FENCE.replace('0', '1')}\n\nProse two.\n"
    triples = capture.diff_chunks(before, after)
    assert [(b, a) for b, a, _c in triples] == [("Prose one.", "Prose won.")]
    assert capture._span_of(before, triples[0][0]) == (0, len("Prose one."))


def test_diff_chunks_drops_a_region_a_block_left():
    """A fence deleted while the prose around it was rewritten is a content
    move, not a style transform: nothing is emitted for that region."""
    before = f"Prose one.\n\n{_FENCE}\n\nProse two.\n"
    after = "Prose won, then too.\n"
    assert capture.diff_chunks(before, after) == []


def test_diff_chunks_div_policy_decides_a_callout():
    """Every div is protected by default; `div_is_prose` frees a callout's
    prose while its `:::` lines stay structural."""
    before = "::: {.callout-note}\n\nA note.\n\n:::\n"
    after = before.replace("A note.", "The note.")
    assert capture.diff_chunks(before, after) == []
    freed = capture.diff_chunks(before, after, div_is_prose=lambda attrs: "callout" in attrs)
    assert freed == [("A note.", "The note.", "")]


def test_coalescing_preserves_pure_insert_delete_skip():
    """Bridging must not resurrect a pure insert/delete as a training pair."""
    before = "It is worth noting the tapestry.\n\nA paragraph that gets deleted entirely.\n"
    after = "We dig in.\n"
    triples = capture.diff_chunks(before, after, coalesce_gap_paras=2, coalesce_gap_chars=800)
    assert all(b.strip() and a.strip() for b, a, _c in triples)


# --- heading context (D14): the nearest preceding heading, immediate depth --


def test_diff_chunks_resolves_nearest_preceding_heading():
    before = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Section One\n\n"
        "The old prose under section one.\n\n"
        "## Section Two\n\n"
        "Untouched tail."
    )
    after = (
        "# Title\n\n"
        "Intro paragraph.\n\n"
        "## Section One\n\n"
        "The shiny new prose under section one.\n\n"
        "## Section Two\n\n"
        "Untouched tail."
    )
    triples = capture.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    before_chunk, after_chunk, context = triples[0]
    assert context == "## Section One"
    assert "shiny new prose" in after_chunk
    assert "old prose" in before_chunk


def test_diff_chunks_preamble_region_has_empty_context():
    # The changed paragraph sits before any heading -> no context.
    before = "First draft of the opening.\n\n## Later Heading\n\ntail"
    after = "Reworked opening line.\n\n## Later Heading\n\ntail"
    triples = capture.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    _, _, context = triples[0]
    assert context == ""


def test_diff_chunks_off_yields_no_context():
    before = "## H\n\nold body"
    after = "## H\n\nnew body"
    triples = capture.diff_chunks(before, after, heading_context=False)
    assert len(triples) == 1
    assert triples[0][2] == ""


def test_diff_chunks_picks_immediate_not_breadcrumb():
    # Two heading levels precede the change; immediate depth = the nearest one.
    before = "# Top\n\n## Mid\n\n### Leaf\n\nold leaf text"
    after = "# Top\n\n## Mid\n\n### Leaf\n\nnew leaf text"
    triples = capture.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    assert triples[0][2] == "### Leaf"


def test_diff_chunks_coalesces_contiguous_changed_paragraphs():
    # Two adjacent changed paragraphs coalesce into ONE pair (documented rule).
    before = "## H\n\npara one old\n\npara two old\n\nuntouched"
    after = "## H\n\npara one new\n\npara two new\n\nuntouched"
    triples = capture.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    before_chunk, after_chunk, context = triples[0]
    assert context == "## H"
    assert before_chunk == "para one old\n\npara two old"
    assert after_chunk == "para one new\n\npara two new"


# --- the archive -----------------------------------------------------------


def _capture(pairs_path, before, after, source):
    return capture.capture_pairs(
        before,
        after,
        pairs_path=pairs_path,
        source=source,
        snapped_at="2026-08-01T00:00:00+00:00",
    )


def test_archive_write_is_atomic(tmp_path):
    """Three files means a partial write is now possible where the old single
    write_text could only truncate. A half-built archive must never be visible
    to iter_archives, which skips unreadable ones SILENTLY."""
    cap_dir = tmp_path / "captures"
    cap_dir.mkdir()
    # Simulate a crashed write: a leftover .tmp staging dir with partial content.
    stale = cap_dir / "abcd1234.tmp"
    stale.mkdir()
    (stale / "before.md").write_text("half-written", encoding="utf-8")

    assert list(capture.iter_archives(tmp_path / "pairs.jsonl")) == []  # invisible

    # A fresh write over the same id clears the stale staging dir and lands whole.
    capture.write_archive(
        cap_dir / "abcd1234",
        capture_id="abcd1234",
        source="post/a.qmd",
        snapped_at="2026-08-04T00:00:00+00:00",
        whole=False,
        before_text="B",
        after_text="A",
    )
    assert not stale.exists()
    got = list(capture.iter_archives(tmp_path / "pairs.jsonl"))
    assert len(got) == 1 and got[0]["before_text"] == "B"


def test_iter_archives_refuses_legacy_json(tmp_path):
    """The legacy blob reader went away in 0.3, but an unmigrated blob must not
    vanish SILENTLY — a silent skip looks exactly like 'that capture had no
    changed regions', so the corpus would quietly shrink. It raises, naming the
    migration script, and it raises even when good archives sit beside it."""
    cap_dir = tmp_path / "captures"
    cap_dir.mkdir()
    (cap_dir / "deadbeef.json").write_text(
        json.dumps(
            {
                "capture_id": "deadbeef",
                "source": "post/legacy.qmd",
                "snapped_at": "2026-07-01T00:00:00+00:00",
                "whole": False,
                "before_text": "It is worth noting the tapestry.",
                "after_text": "We dig in.",
            }
        ),
        encoding="utf-8",
    )
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")  # a valid archive alongside

    with pytest.raises(RunError) as excinfo:
        list(capture.iter_archives(pairs_path))
    msg = str(excinfo.value)
    assert "1 legacy JSON capture archive" in msg
    assert "deadbeef.json" in msg
    assert "ai-style data" not in msg  # its migration is retired; name no ghost command (D38)


# --- the record contract: schema version, system sha, provenance --------------


def test_captured_records_carry_provenance(tmp_path):
    """W8: a real capture stamps the same five keys a synthetic pair does, and
    reads `source_automation` off the frontmatter it was handed."""
    from stylebot.pairs import PAIR_SCHEMA_VERSION
    from stylebot.provenance import PROVENANCE_KEYS

    pairs_path = tmp_path / "pairs.jsonl"
    written = capture.capture_pairs(
        "---\ntitle: T\nautomation: 0\n---\n\nIt is worth noting the rich tapestry.",
        "---\ntitle: T\nautomation: 0\n---\n\nWe dig in.",
        pairs_path=pairs_path,
        source="post/foo.qmd",
        snapped_at="2026-09-03T00:00:00+00:00",
    )
    meta = written[0]["meta"]
    assert meta["schema_version"] == PAIR_SCHEMA_VERSION
    # The prompt claim is the view's, not the draw's (D36, W4 step 5).
    assert "system_sha" not in meta
    assert sorted(meta["provenance"]) == sorted(PROVENANCE_KEYS)
    assert meta["provenance"]["source_automation"] == 0
    assert meta["provenance"]["policy_digest"]
    assert validate_pairs_file(corpus_file(tmp_path, tmp_path / "p.jsonl")) == []


def test_capture_policy_digest_tracks_the_cut_knobs(tmp_path):
    """Two coalesce policies over the same edit must not share a digest — that
    is what makes 'were these pairs cut the same way?' answerable."""
    before = "P one is worth noting.\n\nUntouched middle.\n\nP two is worth noting."
    after = "P one, dug in.\n\nUntouched middle.\n\nP two, dug in."
    digests = []
    for gap_paras, gap_chars in ((0, 0), (1, 200)):
        pairs_path = tmp_path / f"pairs-{gap_paras}.jsonl"
        written = capture.capture_pairs(
            before,
            after,
            pairs_path=pairs_path,
            source="post/foo.qmd",
            snapped_at="2026-09-03T00:00:00+00:00",
            coalesce_gap_paras=gap_paras,
            coalesce_gap_chars=gap_chars,
        )
        digests.append(written[0]["meta"]["provenance"]["policy_digest"])
    assert digests[0] != digests[1]


# --- a save lands as draws (W4 step 4, D84) ----------------------------------


def test_capture_pairs_lands_a_save_as_draws(tmp_path):
    """A capture is appended as draws and nothing else; the identity view
    renders the pair back (W4 step 5)."""
    from stylebot import draws
    from stylebot.pairs import record_id

    data_dir = tmp_path / "dd"
    (data_dir / "draws").mkdir(parents=True)
    pairs_path = data_dir / "pairs"
    slop = "It is worth noting the rich tapestry."
    before = f"## S\n\nKept.\n\n{slop}\n"
    written = capture.capture_pairs(
        before,
        "## S\n\nKept.\n\nWe dig in.\n",
        pairs_path=pairs_path,
        source="post/foo.qmd",
        snapped_at="2026-09-05T00:00:00+00:00",
    )
    assert len(written) == 1 and not pairs_path.exists()
    got = list(draws.iter_draws(data_dir / "draws"))
    assert [d.record for d in got] == written
    draw = got[0]
    assert draw.outcome == draws.ACCEPTED and draw.draw_id == record_id(written[0])
    assert (draw.blog_sha, draw.blog_dirty, draw.supersedes) == (None, True, None)
    assert draw.span == (before.index(slop), before.index(slop) + len(slop))
    assert draw.drawn_at == written[0]["meta"]["captured_at"]
    # The archive's sidecar carries the save time, so save order is checkable.
    cap_id = written[0]["meta"]["capture_id"]
    meta = json.loads((data_dir / "captures" / cap_id / "meta.json").read_text(encoding="utf-8"))
    assert meta["captured_at"] == written[0]["meta"]["captured_at"]
    assert draws.build_view(got, draws.ViewPolicy()).records == 1


def test_capture_pairs_creates_a_draws_dir_that_is_not_there_yet(tmp_path):
    """W4 step 5: `draws/` is the only landing site, so an absent one is
    created rather than read as the old `pairs/` layout (D84)."""
    from stylebot import draws

    data_dir = tmp_path / "fresh"
    pairs_path = data_dir / "pairs.jsonl"
    written = capture.capture_pairs(
        "## S\n\nIt is worth noting the rich tapestry.",
        "## S\n\nWe dig in.",
        pairs_path=pairs_path,
        source="post/foo.qmd",
        snapped_at="2026-09-05T00:00:00+00:00",
    )
    assert (data_dir / "draws").is_dir()
    assert not pairs_path.exists()
    assert [d.record for d in draws.iter_draws(data_dir / "draws")] == written


def test_capture_stamps_its_source_policys_prompt_unless_told_otherwise(tmp_path):
    """D85: the author slot is the policy's; the default renders the frozen bytes."""
    from stylebot.ai_core import STYLE_SYSTEM, style_system
    from stylebot.lib import SourcePolicy

    before = "---\ntitle: T\n---\n\nIt is worth noting the rich tapestry."
    after = "---\ntitle: T\n---\n\nWe dig in."
    kw = dict(pairs_path=tmp_path / "pairs.jsonl", source="post/foo.md", snapped_at=None)
    ada = capture.capture_pairs(before, after, source_policy=SourcePolicy(author="Ada"), **kw)
    assert ada[0]["messages"][0]["content"] == style_system("Ada")
    told = capture.capture_pairs(before, after, system_prompt="a custom prompt", **kw)
    assert told[0]["messages"][0]["content"] == "a custom prompt"
    assert capture.capture_pairs(before, after, **kw)[0]["messages"][0]["content"] == STYLE_SYSTEM


def test_capture_reads_the_level_through_its_source_policy(tmp_path):
    """W4 step 5: which frontmatter field says "human" is policy, not a literal."""
    from stylebot.lib import SourcePolicy

    written = capture.capture_pairs(
        "---\ntitle: T\nai_level: 1\n---\n\nIt is worth noting the rich tapestry.",
        "---\ntitle: T\nai_level: 1\n---\n\nWe dig in.",
        pairs_path=tmp_path / "pairs.jsonl",
        source="post/foo.md",
        snapped_at="2026-09-05T00:00:00+00:00",
        source_policy=SourcePolicy(meta_field="ai_level"),
    )
    assert written[0]["meta"]["provenance"]["source_automation"] == 1


def _archived_save(tmp_path, after="## S\n\nKept.\n\nWe dig in, teh end.\n"):
    """One `capture_pairs` save in a fresh draws-layout data dir: (data_dir, records)."""
    data_dir = tmp_path / "dd"
    (data_dir / "draws").mkdir(parents=True)
    before = "## S\n\nKept.\n\nIt is worth noting the rich tapestry.\n"
    written = capture.capture_pairs(
        before,
        after,
        pairs_path=data_dir / "pairs",
        source="post/foo.qmd",
        snapped_at="2026-09-05T00:00:00+00:00",
    )
    return data_dir, written


def test_recut_archive_relands_an_edited_archive_under_a_fresh_capture_id(tmp_path):
    """A hand edit to `after.md` shows up as drift; `recut_archive` lands new
    draws naming the archive and superseding the first landing, and the view
    then carries the fix (D96)."""
    from stylebot import draws

    data_dir, first = _archived_save(tmp_path)
    cid = first[0]["meta"]["capture_id"]
    captures = data_dir / "captures"
    after_md = captures / cid / "after.md"
    assert capture.drifted_archives(captures, first) == []
    after_md.write_text(
        after_md.read_text(encoding="utf-8").replace("teh", "the"), encoding="utf-8"
    )
    assert capture.drifted_archives(captures, first) == [(cid, 1)]

    relanded = capture.recut_archive(
        captures / cid, pairs_path=data_dir / "pairs", supersedes=(cid,)
    )
    meta = relanded[0]["meta"]
    assert meta["archive_id"] == cid and meta["capture_id"] != cid
    assert meta["snapped_at"] == "2026-09-05T00:00:00+00:00"
    assert capture.drifted_archives(captures, relanded) == []
    assert sorted(p.name for p in captures.iterdir()) == [cid]

    got = list(draws.iter_draws(data_dir / "draws"))
    assert got[-1].supersedes == (cid,) and got[-1].span == got[0].span
    view = list(draws.view_records(got, draws.ViewPolicy()))
    assert len(view) == 1 and "the end" in view[0]["messages"][2]["content"]


def test_recut_archive_refuses_a_wholly_reverted_archive(tmp_path):
    """Reverting every paragraph leaves no pair to land: a clause retires it."""
    from stylebot import draws

    data_dir, first = _archived_save(tmp_path)
    cid = first[0]["meta"]["capture_id"]
    archive = data_dir / "captures" / cid
    (archive / "after.md").write_text(
        (archive / "before.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    with pytest.raises(RunError, match="identical"):
        capture.recut_archive(archive, pairs_path=data_dir / "pairs")
    assert len(list(draws.iter_draws(data_dir / "draws"))) == 1


# ---- the after-side finisher (D107) ---------------------------------------


def _fix_teh(chunks):
    return [c.replace("teh", "the") for c in chunks]


def test_finish_after_touches_only_the_landed_chunks():
    """Untouched prose and the before side are never sent; the fix is spliced
    back where the chunk sat, and the changed `(old, new)` pairs come back."""
    before = "---\ntitle: T\n---\n## S\n\nKept teh paragraph.\n\nRich tapestry.\n\nAlso kept."
    after = "---\ntitle: T\n---\n## S\n\nKept teh paragraph.\n\nWe dig in, teh end.\n\nAlso kept."
    sent = []

    def finish(chunks):
        sent.extend(chunks)
        return _fix_teh(chunks)

    out, changed = capture.finish_after(before, after, finish)
    assert sent == ["We dig in, teh end."]
    assert out == after.replace("teh end", "the end") and "Kept teh" in out
    assert changed == [("We dig in, teh end.", "We dig in, the end.")]


def test_finish_after_keeps_a_chunk_the_finisher_blanks_or_loses():
    before = "## S\n\nRich tapestry.\n"
    after = "## S\n\nWe dig in.\n"
    out, changed = capture.finish_after(before, after, lambda chunks: ["  "] * len(chunks))
    assert (out, changed) == (after, [])
    with pytest.raises(RunError, match="finisher returned 0"):
        capture.finish_after(before, after, lambda chunks: [])


def test_capture_pairs_with_finish_archives_what_landed(tmp_path):
    """The archive's after.md is the finished text, so the record is a substring
    of it and `drifted_archives` stays quiet (D107)."""
    written = capture.capture_pairs(
        "## S\n\nRich tapestry.\n",
        "## S\n\nWe dig in, teh end.\n",
        pairs_path=tmp_path / "pairs.jsonl",
        source="post/foo.qmd",
        snapped_at="2026-07-07T00:00:00+00:00",
        finish=_fix_teh,
    )
    cap_id = written[0]["meta"]["capture_id"]
    assert written[0]["messages"][2]["content"].endswith("We dig in, the end.")
    after_md = (tmp_path / "captures" / cap_id / "after.md").read_text(encoding="utf-8")
    assert after_md == "## S\n\nWe dig in, the end.\n"
    assert capture.drifted_archives(tmp_path / "captures", written) == []


def test_finish_archive_rewrites_after_md_only(tmp_path):
    archive = tmp_path / "captures" / "abc12345"
    capture.write_archive(
        archive,
        capture_id="abc12345",
        source="post/foo.qmd",
        snapped_at=None,
        whole=False,
        before_text="## S\n\nRich tapestry.\n",
        after_text="## S\n\nWe dig in, teh end.\n",
    )
    meta_before = (archive / "meta.json").read_bytes()
    assert capture.finish_archive(archive, _fix_teh, write=False) == [
        ("We dig in, teh end.", "We dig in, the end.")
    ]
    assert "teh" in (archive / "after.md").read_text(encoding="utf-8")
    assert len(capture.finish_archive(archive, _fix_teh)) == 1
    assert (archive / "after.md").read_text(encoding="utf-8") == "## S\n\nWe dig in, the end.\n"
    assert (archive / "meta.json").read_bytes() == meta_before
    assert not list(archive.glob("*.tmp"))
    with pytest.raises(RunError, match="no readable capture archive"):
        capture.finish_archive(tmp_path / "captures" / "nope", _fix_teh)
