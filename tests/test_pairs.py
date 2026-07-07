"""Tests for the pairs.jsonl schema validator — the enforceable contract
between Phase 1/2 (producers) and Phase 3 (consumer)."""

from __future__ import annotations

import json

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.pairs import build_pair_content, validate_pair_record, validate_pairs_file


def _good_record():
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": "It is worth noting the rich tapestry."},
            {"role": "assistant", "content": "We dig in."},
        ],
        "meta": {
            "source": "post/foo.qmd",
            "captured_at": "2026-06-27T00:00:00+00:00",
            "capture_id": "abcd1234",
            "chunk_index": 0,
            "chunk_total": 1,
        },
    }


def test_good_record_validates():
    assert validate_pair_record(_good_record()) == []


def test_wrong_system_prompt_flagged():
    rec = _good_record()
    rec["messages"][0]["content"] = "you are a helpful assistant"
    errs = validate_pair_record(rec)
    assert any("STYLE_SYSTEM" in e for e in errs)


def test_bad_roles_flagged():
    rec = _good_record()
    rec["messages"] = rec["messages"][:2]  # only 2 messages
    assert validate_pair_record(rec)


def test_empty_assistant_flagged():
    rec = _good_record()
    rec["messages"][2]["content"] = "   "
    assert any("assistant" in e for e in validate_pair_record(rec))


def test_missing_meta_key_flagged():
    rec = _good_record()
    del rec["meta"]["capture_id"]
    assert any("capture_id" in e for e in validate_pair_record(rec))


def test_file_validation_reports_line_numbers(tmp_path):
    p = tmp_path / "pairs.jsonl"
    lines = [
        json.dumps(_good_record()),
        "",  # blank, skipped
        "{not json",
        json.dumps({"messages": [], "meta": {}}),
    ]
    p.write_text("\n".join(lines) + "\n")
    problems = validate_pairs_file(p)
    bad_lines = [ln for ln, _ in problems]
    assert bad_lines == [3, 4]  # line 1 good, line 2 blank


def test_file_all_valid_returns_empty(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps(_good_record()) + "\n")
    assert validate_pairs_file(p) == []


# --- heading-context contract (shared by Phase 1 + Phase 2) ---------------


def test_build_pair_content_prepends_heading():
    assert build_pair_content("## H", "body") == "## H\n\nbody"


def test_build_pair_content_empty_context_passthrough():
    assert build_pair_content("", "body") == "body"
    assert build_pair_content(None, "body") == "body"


def _record_with_context(context="## Heading"):
    rec = _good_record()
    rec["messages"][1]["content"] = build_pair_content(context, "It is worth noting the rich tapestry.")
    rec["messages"][2]["content"] = build_pair_content(context, "We dig in.")
    rec["meta"]["context"] = context
    return rec


def test_context_pair_validates():
    assert validate_pair_record(_record_with_context()) == []


def test_context_must_prefix_both_sides():
    rec = _record_with_context()
    # Heading paraphrased on the user (slop) side only — the failure we guard.
    rec["messages"][1]["content"] = "## A Different Heading\n\nIt is worth noting the rich tapestry."
    errs = validate_pair_record(rec)
    assert any("meta.context" in e for e in errs)


def test_absent_context_unconstrained():
    # No meta.context -> no prefix requirement (back-compat with the 254 pairs).
    assert validate_pair_record(_good_record()) == []


def test_gen_facet_keys_are_a_subset_of_recorded_keys():
    # Every key eval facets by must be one synth actually records — a rename
    # on either side fails here instead of silently emptying facet columns.
    from stylebot.pairs import GEN_FACET_KEYS, GEN_META_KEYS, GEN_SESSION_KEYS

    unknown = set(GEN_FACET_KEYS) - set(GEN_META_KEYS) - set(GEN_SESSION_KEYS)
    assert not unknown, f"facet keys nothing records: {sorted(unknown)}"
