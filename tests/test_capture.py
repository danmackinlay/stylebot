"""Direct tests for stylebot.capture — the Phase-1 mechanism as a library.

The CLI-level behaviour (heading context, opt-outs) is pinned in
test_ai_style_log.py through the `ai-style log` wrappers; these exercise the
library surface with explicit paths — no module globals, no monkeypatching —
which is exactly what the extraction buys a non-CLI producer.
"""

from __future__ import annotations

import json
import threading

import pytest

from stylebot import capture, corpus
from stylebot.errors import RunError
from stylebot.jsonl import append_jsonl
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
    # And the flat reader still hands back the shape recut_archives expects.
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


def test_coalescing_preserves_pure_insert_delete_skip():
    """Bridging must not resurrect a pure insert/delete as a training pair."""
    before = "It is worth noting the tapestry.\n\nA paragraph that gets deleted entirely.\n"
    after = "We dig in.\n"
    triples = capture.diff_chunks(before, after, coalesce_gap_paras=2, coalesce_gap_chars=800)
    assert all(b.strip() and a.strip() for b, a, _c in triples)


# --- recut from the archive -----------------------------------------------


def _capture(pairs_path, before, after, source):
    return capture.capture_pairs(
        before,
        after,
        pairs_path=pairs_path,
        source=source,
        snapped_at="2026-08-01T00:00:00+00:00",
    )


def test_recut_rederives_pairs_from_the_archive(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    assert len(corpus.read_records(pairs_path)) == 2  # default cut fragments

    recut = capture.recut_archives(
        pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0
    )
    assert len(recut) == 1  # bridged into one multi-paragraph pair
    assert recut[0]["meta"]["recut"] == {"gap_paras": 2, "gap_chars": 800}
    # The old fragmented pairs for this capture are gone, not stacked on.
    on_disk = corpus.read_records(pairs_path)
    assert len(on_disk) == 1
    assert _KEPT in on_disk[0]["messages"][2]["content"]


def test_recut_is_idempotent(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    kw = dict(coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    capture.recut_archives(pairs_path, **kw)
    first = corpus.read_records(pairs_path)
    capture.recut_archives(pairs_path, **kw)
    second = corpus.read_records(pairs_path)
    assert len(first) == len(second) == 1
    assert first[0]["messages"] == second[0]["messages"]


def test_recut_leaves_unarchived_captures_alone(tmp_path):
    """Pre-archive captures have nothing to re-cut from, so they keep their
    native scale rather than being dropped."""
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    legacy = capture.build_pair_record(
        before_body="It is worth noting the tapestry.",
        after_body="We dig in.",
        source="post/legacy.qmd",
        snapped_at=None,
        before_frontmatter=None,
        after_frontmatter=None,
        capture_id="deadbeef",
        chunk_index=0,
        chunk_total=1,
    )
    append_jsonl(pairs_path, legacy)

    capture.recut_archives(pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    ids = {r["meta"]["capture_id"] for r in corpus.read_records(pairs_path)}
    assert "deadbeef" in ids  # untouched


def test_recut_dry_run_does_not_write(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    before = pairs_path.read_text(encoding="utf-8")
    recut = capture.recut_archives(
        pairs_path,
        coalesce_gap_paras=2,
        coalesce_gap_chars=800,
        min_chars=0,
        dry_run=True,
    )
    assert recut and pairs_path.read_text(encoding="utf-8") == before


def test_recut_output_validates(tmp_path):
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    capture.recut_archives(pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    assert validate_pairs_file(pairs_path) == []


def test_recut_preserves_a_corrupt_line(tmp_path):
    """`recut` rewrites the whole corpus, so it must round-trip unparseable
    lines like every other rewrite path — the file is an audit trail, and a
    half-written record is evidence, not garbage to silently delete."""
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    corrupt = '{"messages": [{"role": "system"'  # truncated mid-write
    with pairs_path.open("a", encoding="utf-8") as fp:
        fp.write(corrupt + "\n")

    capture.recut_archives(pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    assert corrupt in pairs_path.read_text(encoding="utf-8").splitlines()


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

    # And the same refusal reaches the caller through recut, which is how a user
    # would actually hit it.
    with pytest.raises(RunError, match="legacy JSON capture archive"):
        capture.recut_archives(pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800)


def test_hand_edited_archive_changes_what_recut_emits(tmp_path):
    """The capability this whole layout exists for: open the archive, fix a
    borked capture by hand, re-cut, and the corpus follows."""
    pairs_path = tmp_path / "pairs.jsonl"
    written = _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    cap_id = written[0]["meta"]["capture_id"]

    after_md = tmp_path / "captures" / cap_id / "after.md"
    after_md.write_text(
        after_md.read_text(encoding="utf-8").replace("We dig in.", "We dig in, properly."),
        encoding="utf-8",
    )
    capture.recut_archives(pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    targets = [r["messages"][2]["content"] for r in corpus.read_records(pairs_path)]
    assert any("We dig in, properly." in t for t in targets)


# --- a re-cut must not reuse the previous cut's join keys ------------------


def _ids_to_text(pairs_path) -> dict[str, str]:
    """`{record_id: target text}` for the corpus as it stands on disk."""
    from stylebot.pairs import record_id

    return {record_id(r): r["messages"][2]["content"] for r in corpus.read_records(pairs_path)}


def test_recut_does_not_reuse_the_previous_cuts_record_ids(tmp_path):
    """REGRESSION. A re-cut reuses `capture_id` and renumbers `chunk_index` from
    0, so `<capture_id>:<chunk_index>` alone named DIFFERENT prose before and
    after: `<cap>:0` went from one paragraph to the coalesced multi-paragraph
    pair. Everything that joins or resumes on that key then mismatches —
    `eval.score_pairs_file` skips the pair as "already scored" and leaves stale
    judge/detector numbers pinned to changed text, and
    `report.render_scores_report` renders the new text under the old scores.

    Folding `meta.recut` into the id fixes it: the re-cut keys are disjoint from
    the original cut's, so a stale score can no longer be mistaken for a
    current one.
    """
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    original = _ids_to_text(pairs_path)
    assert len(original) == 2  # the default cut fragments into two pairs

    capture.recut_archives(pairs_path, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)
    recut = _ids_to_text(pairs_path)
    assert len(recut) == 1  # …bridged into one

    # The text genuinely moved (this is what makes reuse a correctness bug and
    # not a cosmetic one) …
    assert _KEPT not in "".join(original.values())
    assert _KEPT in next(iter(recut.values()))
    # … and no id survives the move, so no stale score can join to it.
    assert not (set(original) & set(recut))


def test_recut_ids_are_stable_across_reruns_at_the_same_policy(tmp_path):
    """The other half of the contract: idempotence must survive the id change.
    Re-running the SAME re-cut has to reproduce the same keys, or every re-run
    would re-score the whole corpus from scratch."""
    pairs_path = tmp_path / "pairs.jsonl"
    _capture(pairs_path, _BEFORE, _AFTER, "post/a.qmd")
    kw = dict(coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0)

    capture.recut_archives(pairs_path, **kw)
    first = _ids_to_text(pairs_path)
    capture.recut_archives(pairs_path, **kw)
    assert _ids_to_text(pairs_path) == first

    # A DIFFERENT policy is a different cut, so it must key differently.
    capture.recut_archives(pairs_path, coalesce_gap_paras=99, coalesce_gap_chars=99999, min_chars=0)
    assert not (set(_ids_to_text(pairs_path)) & set(first))


def test_recut_drops_null_edits_that_min_chars_cannot_catch(tmp_path):
    """The two floors fail in different directions: a cross-reference line is
    short, a punctuation touch-up to a long paragraph is not. `max_similarity`
    is the second one, and re-cutting is how a corpus already holding no-ops
    sheds them without anyone hand-culling pairs.jsonl."""
    long_para = (
        "The archive is insurance against an environment shift the training "
        "objective never anticipated, which is the whole reason it earns its "
        "keep across a fleet of otherwise identical policies and their runs."
    )
    before = f"# H\n\n{long_para}\n\nSee [that](./x.qmd).\n\nA third paragraph, untouched here."
    after = f"# H\n\n{long_para}!\n\nSee [this](./x.qmd).\n\nA third paragraph, untouched here."

    loose = capture.diff_chunks(before, after, heading_context=True)
    assert len(loose) >= 1  # the raw cut keeps both edits

    d = tmp_path / "data"
    d.mkdir()
    pairs = d / "pairs.jsonl"
    pairs.write_text("")
    arch = d / "captures" / "cap1"
    arch.mkdir(parents=True)
    (arch / "before.md").write_text(before)
    (arch / "after.md").write_text(after)
    (arch / "meta.json").write_text(
        json.dumps({"capture_id": "cap1", "source": "notebook/x.qmd", "snapped_at": "t"})
    )

    kept = capture.recut_archives(
        pairs, coalesce_gap_paras=2, coalesce_gap_chars=800, min_chars=0, dry_run=True
    )
    floored = capture.recut_archives(
        pairs,
        coalesce_gap_paras=2,
        coalesce_gap_chars=800,
        min_chars=0,
        max_similarity=0.95,
        dry_run=True,
    )
    assert len(floored) < len(kept)
    assert all((r["meta"]["recut"].get("max_similarity")) == 0.95 for r in floored)


# --- the record contract: schema version, system sha, provenance --------------


def test_captured_records_carry_provenance(tmp_path):
    """W8: a real capture stamps the same five keys a synthetic pair does, and
    reads `source_automation` off the frontmatter it was handed."""
    from stylebot.pairs import PAIR_SCHEMA_VERSION, system_sha
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
    assert meta["system_sha"] == system_sha()
    assert sorted(meta["provenance"]) == sorted(PROVENANCE_KEYS)
    assert meta["provenance"]["source_automation"] == 0
    assert meta["provenance"]["policy_digest"]
    assert validate_pairs_file(pairs_path) == []


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


def test_recut_records_carry_provenance(tmp_path):
    from stylebot.provenance import PROVENANCE_KEYS

    pairs_path = tmp_path / "pairs.jsonl"
    capture.capture_pairs(
        "---\nautomation: 1\n---\n\nP one is worth noting.\n\nStill here.\n\n"
        "P two is worth noting.",
        "---\nautomation: 1\n---\n\nP one, dug in.\n\nStill here.\n\nP two, dug in.",
        pairs_path=pairs_path,
        source="post/foo.qmd",
        snapped_at="2026-09-03T00:00:00+00:00",
    )
    records = capture.recut_archives(pairs_path, coalesce_gap_paras=1, coalesce_gap_chars=200)
    assert records
    for rec in records:
        prov = rec["meta"]["provenance"]
        assert sorted(prov) == sorted(PROVENANCE_KEYS)
        assert prov["source_automation"] == 1
        assert prov["policy_digest"]
    assert validate_pairs_file(pairs_path) == []


def test_capture_pairs_appends_under_the_corpus_lock(tmp_path):
    """A `log save` is a locked rewrite, then an append: the append has to wait
    for a concurrent rewrite rather than land inside its critical section and be
    overwritten by it (D60)."""
    pairs_path = tmp_path / "pairs"
    pairs_path.mkdir()
    kw = dict(pairs_path=pairs_path, source="post/foo.qmd", snapped_at="2026-07-07T00:00:00+00:00")
    capture.capture_pairs("## S\n\nIt is worth noting the tapestry.", "## S\n\nWe dig in.", **kw)
    entered, proceed = threading.Event(), threading.Event()

    def tag(records):
        entered.set()
        proceed.wait(5)
        for r in records:
            r["meta"]["tag"] = "rewritten"
        return records

    rewriter = threading.Thread(target=corpus.rewrite_with, args=(pairs_path, tag))
    rewriter.start()
    assert entered.wait(5)
    appender = threading.Thread(
        target=capture.capture_pairs,
        args=("## T\n\nIt is worth noting the rich tapestry.", "## T\n\nWe dig in."),
        kwargs=kw,
    )
    appender.start()
    appender.join(0.5)  # an unlocked append would land here, inside the rewrite
    proceed.set()
    rewriter.join(5)
    appender.join(5)
    on_disk = corpus.read_records(pairs_path)
    assert [r["meta"].get("tag") for r in on_disk] == ["rewritten", None]
