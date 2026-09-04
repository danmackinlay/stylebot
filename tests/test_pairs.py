"""Tests for the pairs.jsonl schema validator — the enforceable contract
between Phase 1/2 (producers) and Phase 3 (consumer)."""

from __future__ import annotations

import json

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.pairs import build_pair_content, record_id, validate_pair_record, validate_pairs_file


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


# --- heading-context contract (shared by Phase 1 + Phase 2) ---------------


def test_build_pair_content_prepends_heading():
    assert build_pair_content("## H", "body") == "## H\n\nbody"


def test_build_pair_content_empty_context_passthrough():
    assert build_pair_content("", "body") == "body"
    assert build_pair_content(None, "body") == "body"


def _record_with_context(context="## Heading"):
    rec = _good_record()
    rec["messages"][1]["content"] = build_pair_content(
        context, "It is worth noting the rich tapestry."
    )
    rec["messages"][2]["content"] = build_pair_content(context, "We dig in.")
    rec["meta"]["context"] = context
    return rec


def test_context_pair_validates():
    assert validate_pair_record(_record_with_context()) == []


def test_context_must_prefix_both_sides():
    rec = _record_with_context()
    # Heading paraphrased on the user (slop) side only — the failure we guard.
    rec["messages"][1]["content"] = (
        "## A Different Heading\n\nIt is worth noting the rich tapestry."
    )
    errs = validate_pair_record(rec)
    assert any("meta.context" in e for e in errs)


def test_cross_heading_record_validates_with_inline_headings():
    """A cross-heading pair carries headings INSIDE both sides and no
    meta.context, so the verbatim-prefix invariant does not apply. Pins that
    the v2 chunk mode is representable in the corpus contract as-is."""
    rec = _good_record()
    passage = "## A Section\n\nSome prose.\n\n### Another\n\nMore prose."
    rec["messages"][1]["content"] = passage.replace(
        "Some prose.", "It is worth noting the tapestry."
    )
    rec["messages"][2]["content"] = passage
    rec["meta"]["merge_max_chars"] = 4500
    rec["meta"]["chunk_mode"] = "cross-heading"
    assert "context" not in rec["meta"]
    assert validate_pair_record(rec) == []


def test_gen_facet_keys_are_a_subset_of_recorded_keys():
    # Every key eval facets by must be one synth actually records — a rename
    # on either side fails here instead of silently emptying facet columns.
    from stylebot.pairs import GEN_FACET_KEYS, GEN_META_KEYS, GEN_SESSION_KEYS

    unknown = set(GEN_FACET_KEYS) - set(GEN_META_KEYS) - set(GEN_SESSION_KEYS)
    assert not unknown, f"facet keys nothing records: {sorted(unknown)}"


# --- meta types and domains ---------------------------------------------------


def test_pre_provenance_record_still_validates():
    """The 8,265 records written before 2026-09 carry no schema_version, no
    system_sha and no provenance. Adding the keys must not invalidate them —
    absence is the version-0 tell, not an error."""
    rec = _good_record()
    assert "schema_version" not in rec["meta"]
    assert validate_pair_record(rec) == []


def test_meta_type_violations_flagged():
    for key, bad in (
        ("chunk_index", "0"),
        ("chunk_total", None),
        ("transform_sim", "0.5"),
        ("tags", "synthetic"),
        ("synthetic", 1),
        ("gen", []),
    ):
        rec = _good_record()
        rec["meta"][key] = bad
        errs = validate_pair_record(rec)
        assert any(f"meta.{key}" in e for e in errs), f"{key}={bad!r} passed validation"


def test_meta_domain_violations_flagged():
    for key, bad in (
        ("chunk_index", -1),
        ("chunk_total", 0),
        ("transform_sim", 1.5),
        ("context_mode", "breadcrumb"),
        ("chunk_mode", "section"),
        ("schema_version", 0),
    ):
        rec = _good_record()
        rec["meta"][key] = bad
        assert any(f"meta.{key}" in e for e in validate_pair_record(rec)), f"{key}={bad!r} passed"


def test_chunk_index_past_chunk_total_flagged():
    rec = _good_record()
    rec["meta"]["chunk_index"] = 3
    rec["meta"]["chunk_total"] = 2
    assert any("chunk_index" in e for e in validate_pair_record(rec))


def test_bool_does_not_satisfy_an_int_field():
    # bool is an int subclass, so a naive isinstance check would let True
    # through as a chunk index.
    rec = _good_record()
    rec["meta"]["chunk_index"] = True
    assert any("chunk_index" in e for e in validate_pair_record(rec))


def test_system_sha_must_match_the_frozen_prompt():
    from stylebot.pairs import PAIR_SCHEMA_VERSION, system_sha

    rec = _good_record()
    rec["meta"]["schema_version"] = PAIR_SCHEMA_VERSION
    rec["meta"]["system_sha"] = system_sha()
    assert validate_pair_record(rec) == []
    rec["meta"]["system_sha"] = "0" * 16
    assert any("system_sha" in e for e in validate_pair_record(rec))


def test_provenance_shape_is_checked():
    from stylebot.provenance import PROVENANCE_KEYS, pair_provenance

    rec = _good_record()
    rec["meta"]["provenance"] = pair_provenance(
        {"blog_sha": "abc1234", "blog_dirty": False, "stylebot_sha": "def5678"},
        policy_digest="0123456789ab",
        source_automation=0,
    )
    assert sorted(rec["meta"]["provenance"]) == sorted(PROVENANCE_KEYS)
    assert validate_pair_record(rec) == []

    del rec["meta"]["provenance"]["policy_digest"]
    assert any("provenance missing" in e for e in validate_pair_record(rec))

    rec["meta"]["provenance"]["policy_digest"] = 7
    assert any("provenance.policy_digest" in e for e in validate_pair_record(rec))


# --- re-cut ids (the `meta.recut` discriminator) ---------------------------
#
# `capture.recut_archives` reuses the archive's `capture_id` and renumbers
# `chunk_index` from 0, so `capture_id:chunk_index` alone can name materially
# different prose before and after a re-cut. `record_id` folds `meta.recut`
# (the gap policy) into the key for re-cut records ONLY. The end-to-end
# capture->recut regression lives in test_capture.py; these pin the id algebra.

_RECUT_A = {"gap_paras": 2, "gap_chars": 800}


def test_record_id_golden_shapes_for_non_recut_records():
    """GOLDEN. Pairs with no `meta.recut` keep their id byte-for-byte.

    Load-bearing, not decorative: a committed `scores.jsonl` lives in the blog
    repo and joins on these strings. Changing either shape orphans every score
    for the never-re-cut captures that are most of the corpus, so the literals
    are pinned rather than merely round-tripped.
    """
    assert record_id({"meta": {"capture_id": "13860ebc", "chunk_index": 0}}) == "13860ebc:0"
    # The no-capture_id fallback: sha256("post/x.qmd\x001").hexdigest()[:16].
    assert record_id({"meta": {"source": "post/x.qmd", "chunk_index": 1}}) == "abfe53300435fc98"


def test_record_id_recut_differs_from_the_original_cut():
    """A re-cut record must not reuse the original cut's id at the same
    capture_id:chunk_index — that collision is the whole bug."""
    base = {"meta": {"capture_id": "13860ebc", "chunk_index": 0}}
    recut = {"meta": {"capture_id": "13860ebc", "chunk_index": 0, "recut": _RECUT_A}}
    assert record_id(recut) != record_id(base)
    # …but it is still recognisably the same capture and chunk, so the id stays
    # greppable and `capture_id` remains the leading grouping key.
    assert record_id(recut).startswith("13860ebc:r")
    assert record_id(recut).endswith(":0")


def test_record_id_is_stable_for_the_same_recut_policy():
    """Re-cutting twice at the SAME policy yields the SAME id — a re-cut has to
    stay idempotent end to end, not just inside pairs.jsonl. Key order in the
    stamp must not matter either (the digest is over canonical JSON)."""
    first = record_id({"meta": {"capture_id": "cap", "chunk_index": 1, "recut": _RECUT_A}})
    again = record_id({"meta": {"capture_id": "cap", "chunk_index": 1, "recut": dict(_RECUT_A)}})
    reordered = record_id(
        {
            "meta": {
                "capture_id": "cap",
                "chunk_index": 1,
                "recut": {"gap_chars": 800, "gap_paras": 2},
            },
        }
    )
    assert first == again == reordered


def test_record_id_synthetic_pairs_ignore_recut():
    """Synthetic pairs key off `synth_key` and return before the re-cut branch:
    they are generated, never re-cut, and their committed scores must not move."""
    meta = {"synth_key": "abc", "capture_id": "cap", "chunk_index": 2, "recut": _RECUT_A}
    assert record_id({"meta": meta}) == "abc"


# --- record_id uniqueness, and the fix ---------------------------------------


def _synth_record(key: str, slop: str = "It is worth noting the tapestry."):
    rec = _good_record()
    rec["messages"][1]["content"] = slop
    rec["meta"]["synth_key"] = key
    rec["meta"]["synthetic"] = True
    return rec


def test_duplicate_record_id_is_reported_against_the_later_line(tmp_path):
    p = tmp_path / "pairs.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                _synth_record("aaa", "first draw"),
                _synth_record("bbb"),
                _synth_record("aaa", "second draw, same cell"),
            )
        )
        + "\n"
    )
    problems = validate_pairs_file(p)
    assert [ln for ln, _ in problems] == [3]
    assert any("duplicate record_id 'aaa'" in e for e in problems[0][1])
    assert any("first seen line 1" in e for e in problems[0][1])
    # A shard is legitimately a partial view of the id space.
    assert validate_pairs_file(p, check_duplicates=False) == []


def test_dedupe_keeps_the_first_draw(tmp_path):
    from stylebot.pairs import dedupe_pairs_file, iter_pairs

    p = tmp_path / "pairs.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                _synth_record("aaa", "first draw"),
                _synth_record("bbb"),
                _synth_record("aaa", "second draw"),
            )
        )
        + "\n"
    )
    dry = dedupe_pairs_file(p)
    assert (dry.total, dry.dropped, dry.kept, dry.duplicate_ids) == (3, 1, 2, 1)
    assert dry.drops == [("aaa", 3, 1)]
    assert not dry.applied
    assert len(p.read_text().splitlines()) == 3  # dry run wrote nothing

    report = dedupe_pairs_file(p, apply=True)
    assert report.applied and report.dropped == 1
    assert report.backup is not None and report.backup.exists()
    bodies = [r["messages"][1]["content"] for r in iter_pairs(p)]
    # The FIRST draw survives: it is the one committed scores already join to.
    assert bodies == ["first draw", "It is worth noting the tapestry."]
    assert validate_pairs_file(p) == []


def test_dedupe_is_a_no_op_on_a_clean_file(tmp_path):
    from stylebot.pairs import dedupe_pairs_file

    p = tmp_path / "pairs.jsonl"
    p.write_text(json.dumps(_synth_record("aaa")) + "\n")
    before = p.read_bytes()
    report = dedupe_pairs_file(p, apply=True)
    assert (report.dropped, report.applied) == (0, False)
    assert p.read_bytes() == before
    assert not (tmp_path / "pairs.jsonl.bak").exists()


# --- the CLI shell ------------------------------------------------------------


def test_validate_pairs_cli_reports_and_fixes(tmp_path):
    from click.testing import CliRunner

    from stylebot.bin.ai_style import main

    p = tmp_path / "pairs.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(r)
            for r in (_synth_record("aaa", "first"), _synth_record("aaa", "second"))
        )
        + "\n"
    )
    runner = CliRunner()
    result = runner.invoke(main, ["validate-pairs", str(p)])
    assert result.exit_code == 1
    assert "1 duplicate record_id" in result.output

    result = runner.invoke(main, ["validate-pairs", str(p), "--fix", "--no-backup"])
    assert result.exit_code == 0, result.output
    assert "dropped 1 of 2" in result.output
    assert not (tmp_path / "pairs.jsonl.bak").exists()
    assert runner.invoke(main, ["validate-pairs", str(p)]).exit_code == 0
