"""Tests for Phase 1 heading-context capture in `ai-style log`.

Covers the `diff_chunks` heading resolution, the context prefix on captured
pairs (gated through the shared `validate_pair_record`), the
`--no-heading-context` / `--whole` opt-outs, and back-compat for heading-less
(preamble) regions. See `_plans/heading-context.md`.
"""

from __future__ import annotations

import pytest

from stylebot.bin import ai_style_log as log
from stylebot.pairs import build_pair_content, validate_pair_record


@pytest.fixture
def temp_corpus(tmp_path, monkeypatch):
    """Point the module's on-disk pairs.jsonl at a temp file.

    `PAIRS_PATH` (and friends) are resolved at import time from
    $STYLEBOT_DATA_DIR, so we patch the module globals directly to isolate
    each test's writes.
    """
    data_dir = tmp_path / "corpus"
    monkeypatch.setattr(log, "TRAINING_PAIRS_DIR", data_dir)
    monkeypatch.setattr(log, "SNAPSHOTS_DIR", data_dir / "snapshots")
    monkeypatch.setattr(log, "PAIRS_PATH", data_dir / "pairs.jsonl")
    return data_dir


# ---- diff_chunks heading resolution --------------------------------------


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
    triples = log.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    before_chunk, after_chunk, context = triples[0]
    assert context == "## Section One"
    assert "shiny new prose" in after_chunk
    assert "old prose" in before_chunk


def test_diff_chunks_preamble_region_has_empty_context():
    # The changed paragraph sits before any heading -> no context.
    before = "First draft of the opening.\n\n## Later Heading\n\ntail"
    after = "Reworked opening line.\n\n## Later Heading\n\ntail"
    triples = log.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    _, _, context = triples[0]
    assert context == ""


def test_diff_chunks_off_yields_no_context():
    before = "## H\n\nold body"
    after = "## H\n\nnew body"
    triples = log.diff_chunks(before, after, heading_context=False)
    assert len(triples) == 1
    assert triples[0][2] == ""


def test_diff_chunks_picks_immediate_not_breadcrumb():
    # Two heading levels precede the change; immediate depth = the nearest one.
    before = (
        "# Top\n\n"
        "## Mid\n\n"
        "### Leaf\n\n"
        "old leaf text"
    )
    after = (
        "# Top\n\n"
        "## Mid\n\n"
        "### Leaf\n\n"
        "new leaf text"
    )
    triples = log.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    assert triples[0][2] == "### Leaf"


def test_diff_chunks_coalesces_contiguous_changed_paragraphs():
    # Two adjacent changed paragraphs coalesce into ONE pair (documented rule).
    before = "## H\n\npara one old\n\npara two old\n\nuntouched"
    after = "## H\n\npara one new\n\npara two new\n\nuntouched"
    triples = log.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 1
    before_chunk, after_chunk, context = triples[0]
    assert context == "## H"
    assert before_chunk == "para one old\n\npara two old"
    assert after_chunk == "para one new\n\npara two new"


def test_diff_chunks_gap_breaks_into_two_pairs():
    # An untouched paragraph between two changed regions -> two pairs.
    before = "## H\n\nA old\n\nmiddle untouched\n\nB old"
    after = "## H\n\nA new\n\nmiddle untouched\n\nB new"
    triples = log.diff_chunks(before, after, heading_context=True)
    assert len(triples) == 2
    # Both changed regions are under the same single heading.
    assert all(t[2] == "## H" for t in triples)


def test_diff_chunks_skips_pure_insert_and_delete():
    # Pure insert (added para) and pure delete (removed para) yield no pair.
    before = "## H\n\nkept body"
    after = "## H\n\nkept body\n\nbrand new added paragraph"
    assert log.diff_chunks(before, after, heading_context=True) == []


# ---- captured pair has context prefix + validates ------------------------


def test_captured_pair_has_heading_prefix_and_validates(temp_corpus):
    before_text = "## Section\n\nIt is worth noting the rich tapestry."
    after_text = "## Section\n\nWe dig in."
    written = log._capture_and_append(
        before_text=before_text,
        after_text=after_text,
        source="post/foo.qmd",
        snapped_at="2026-06-28T00:00:00+00:00",
        whole=False,
        extra_meta=None,
        heading_context=True,
    )
    assert len(written) == 1
    rec = written[0]
    assert rec["meta"]["context"] == "## Section"
    assert rec["meta"]["context_mode"] == "immediate"
    # Heading is a verbatim prefix on BOTH sides.
    assert rec["messages"][1]["content"].startswith("## Section\n\n")
    assert rec["messages"][2]["content"].startswith("## Section\n\n")
    # before/after_chars count the BODY only (excluding the prefix).
    assert rec["meta"]["before_chars"] == len("It is worth noting the rich tapestry.")
    assert rec["meta"]["after_chars"] == len("We dig in.")
    # Gate through the shared validator.
    assert validate_pair_record(rec) == []


def test_build_pair_record_context_matches_build_pair_content():
    rec = log._build_pair_record(
        before_body="slop body",
        after_body="dan body",
        source="x",
        snapped_at=None,
        before_frontmatter=None,
        after_frontmatter=None,
        capture_id="cap12345",
        chunk_index=0,
        chunk_total=1,
        context="## A Heading",
    )
    assert rec["messages"][1]["content"] == build_pair_content("## A Heading", "slop body")
    assert rec["messages"][2]["content"] == build_pair_content("## A Heading", "dan body")
    assert rec["meta"]["context"] == "## A Heading"
    assert rec["meta"]["context_mode"] == "immediate"
    assert validate_pair_record(rec) == []


# ---- opt-outs: --no-heading-context, --whole, preamble -------------------


def test_no_heading_context_leaves_content_bare(temp_corpus):
    before_text = "## Section\n\nold body here."
    after_text = "## Section\n\nnew body here."
    written = log._capture_and_append(
        before_text=before_text,
        after_text=after_text,
        source="post/foo.qmd",
        snapped_at=None,
        whole=False,
        extra_meta=None,
        heading_context=False,
    )
    assert len(written) == 1
    rec = written[0]
    assert "context" not in rec["meta"]
    assert "context_mode" not in rec["meta"]
    # Old behaviour: bodies are bare (no heading prefix).
    assert rec["messages"][1]["content"] == "old body here."
    assert rec["messages"][2]["content"] == "new body here."
    assert validate_pair_record(rec) == []


def test_whole_mode_has_no_context_even_when_on(temp_corpus):
    before_text = "## H\n\nold body\n\n## H2\n\nmore old"
    after_text = "## H\n\nnew body\n\n## H2\n\nmore new"
    written = log._capture_and_append(
        before_text=before_text,
        after_text=after_text,
        source="post/foo.qmd",
        snapped_at=None,
        whole=True,
        extra_meta=None,
        heading_context=True,
    )
    assert len(written) == 1
    rec = written[0]
    assert "context" not in rec["meta"]
    # Whole-file body is the entire (frontmatter-stripped) text, unchanged —
    # headings stay inline, nothing is prepended.
    assert rec["messages"][1]["content"] == before_text
    assert rec["messages"][2]["content"] == after_text
    assert validate_pair_record(rec) == []


def test_preamble_change_is_valid_with_no_context(temp_corpus):
    # Change before any heading -> heading-less pair, no meta.context.
    before_text = "Opening line, original.\n\n## Heading\n\nbody"
    after_text = "Opening line, rewritten.\n\n## Heading\n\nbody"
    written = log._capture_and_append(
        before_text=before_text,
        after_text=after_text,
        source="post/foo.qmd",
        snapped_at=None,
        whole=False,
        extra_meta=None,
        heading_context=True,
    )
    assert len(written) == 1
    rec = written[0]
    assert "context" not in rec["meta"]
    assert rec["messages"][1]["content"] == "Opening line, original."
    assert rec["messages"][2]["content"] == "Opening line, rewritten."
    assert validate_pair_record(rec) == []
