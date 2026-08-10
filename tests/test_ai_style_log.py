"""Tests for Phase 1 heading-context capture in `ai-style log`.

Covers the `diff_chunks` heading resolution, the context prefix on captured
pairs (gated through the shared `validate_pair_record`), the
`--no-heading-context` / `--whole` opt-outs, and back-compat for heading-less
(preamble) regions. See `_plans/decisions.md` D14.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from stylebot import capture
from stylebot.bin import ai_style_log as log
from stylebot.pairs import build_pair_content, validate_pair_record


@pytest.fixture
def temp_corpus(tmp_path, monkeypatch):
    """Point the logger's on-disk state at a temp dir.

    The logger resolves its paths per command through
    `config.resolve_data_dir()`, so setting the env var is all it takes — no
    module globals to patch, no reimport.
    """
    data_dir = tmp_path / "corpus"
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(data_dir))
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
    before = "# Top\n\n## Mid\n\n### Leaf\n\nold leaf text"
    after = "# Top\n\n## Mid\n\n### Leaf\n\nnew leaf text"
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
    rec = capture.build_pair_record(
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


# ---- session redo: close archives, reopen restores, re-save replaces ------


@pytest.fixture
def session_repo(temp_corpus, tmp_path, monkeypatch):
    """A working tree with one .qmd, cwd'd into it, sharing temp_corpus."""
    repo = tmp_path / "repo"
    (repo / "post").mkdir(parents=True)
    monkeypatch.chdir(repo)
    return repo


def _run(*args):
    result = CliRunner().invoke(log.main, ["-q", *args])
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def _sources_and_stamps():
    recs = log._iter_pairs()
    return [
        (r["meta"].get("source"), r["meta"].get("snapped_at"), r["messages"][2]["content"])
        for r in recs
    ]


def test_save_archives_snapshot_instead_of_deleting(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")

    assert not log._snapshot_path(Path("post/foo.qmd")).exists()
    archived = log._closed_snapshot_path(Path("post/foo.qmd"))
    assert archived.exists()
    # The archive holds the ORIGINAL base, not the saved state -- and holds it
    # as an ordinary readable file, not a JSON-escaped string.
    assert "rich tapestry" in (archived / "content.md").read_text(encoding="utf-8")
    assert log._read_snapshot(archived)["source"] == "post/foo.qmd"


def test_redo_replaces_prior_pair_against_same_base(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")
    first = _sources_and_stamps()
    assert len(first) == 1
    assert "typoo" in first[0][2]

    # Fix the typo and redo.
    f.write_text("## S\n\nWe dig in, without a typo.\n", encoding="utf-8")
    _run("save", "--redo", "post/foo.qmd")

    after = _sources_and_stamps()
    # Replaced, not stacked...
    assert len(after) == 1
    assert "typoo" not in after[0][2]
    assert "without a typo" in after[0][2]
    # ...because the session timestamp was preserved across the reopen.
    assert after[0][1] == first[0][1]
    # And the redo's own close re-archived the same base for a further redo.
    payload = log._read_snapshot(log._closed_snapshot_path(Path("post/foo.qmd")))
    assert "rich tapestry" in payload["content"]


def test_reopen_then_save_matches_redo(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")
    stamp = _sources_and_stamps()[0][1]

    out = _run("reopen", "post/foo.qmd")
    assert "reopened session" in out
    assert log._snapshot_path(Path("post/foo.qmd")).exists()
    # Restoring consumes the archive, so it cannot be double-restored.
    assert not log._closed_snapshot_path(Path("post/foo.qmd")).exists()

    f.write_text("## S\n\nWe dig in, without a typo.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")
    after = _sources_and_stamps()
    assert len(after) == 1
    assert after[0][1] == stamp
    assert "without a typo" in after[0][2]


def test_redo_preserves_cross_session_pairs(session_repo):
    """A redo must only replace ITS session's pairs, not an earlier session's."""
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nFirst slop draft here.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nFirst rewrite here.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")

    # Unrelated later session on the same file.
    f.write_text("## S\n\nSecond slop draft here.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nSecond rewrite, typoo.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")
    assert len(_sources_and_stamps()) == 2

    f.write_text("## S\n\nSecond rewrite, fixed.\n", encoding="utf-8")
    _run("save", "--redo", "post/foo.qmd")

    after = _sources_and_stamps()
    assert len(after) == 2
    bodies = sorted(b for _, _, b in after)
    assert any("First rewrite" in b for b in bodies)
    assert any("Second rewrite, fixed" in b for b in bodies)
    assert not any("typoo" in b for b in bodies)


def test_reopen_refuses_when_session_already_open(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nbody one.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    result = CliRunner().invoke(log.main, ["-q", "reopen", "post/foo.qmd"])
    assert result.exit_code == 1
    assert "already open" in result.output


def test_reopen_without_archive_errors(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nbody one.\n", encoding="utf-8")
    result = CliRunner().invoke(log.main, ["-q", "reopen", "post/foo.qmd"])
    assert result.exit_code == 1
    assert "no closed session" in result.output


def test_redo_without_archive_errors(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nbody one.\n", encoding="utf-8")
    result = CliRunner().invoke(log.main, ["-q", "save", "--redo", "post/foo.qmd"])
    assert result.exit_code == 1
    assert "no closed session" in result.output


def test_redo_is_noop_when_session_already_open(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    out = _run("save", "--redo", "post/foo.qmd")
    assert "reopened session" not in out
    assert len(_sources_and_stamps()) == 1


def test_drop_is_recoverable_via_reopen(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    _run("drop", "post/foo.qmd")
    _run("reopen", "post/foo.qmd")
    payload = log._read_snapshot(log._snapshot_path(Path("post/foo.qmd")))
    assert "rich tapestry" in payload["content"]


def test_closed_sessions_are_not_reported_as_open(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")
    out = CliRunner().invoke(log.main, ["list"]).output
    assert "0 open session(s)" in out
    assert "1 closed session(s)" in out


def test_save_reports_sessionless_prior_pairs_without_crashing(session_repo):
    """Regression: pairs logged via `log pair` carry snapped_at=None.

    The post-save report sorted the set of prior snapped_at values, which
    raised `TypeError: '<' not supported between 'str' and 'NoneType'` as soon
    as a source had both session-logged and sessionless pairs.
    """
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nbody\n", encoding="utf-8")
    (session_repo / "b.txt").write_text("It is worth noting the tapestry.", encoding="utf-8")
    (session_repo / "a.txt").write_text("We dig in.", encoding="utf-8")
    # A sessionless pair (snapped_at=None) for this source...
    _run("pair", "--before", "b.txt", "--after", "a.txt", "--source", "post/foo.qmd", "--whole")
    # ...plus an earlier session's pair, so `other_session` mixes both kinds.
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in here.\n", encoding="utf-8")
    _run("save", "post/foo.qmd")

    f.write_text("## S\n\nA later slop draft entirely.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nA later rewrite entirely.\n", encoding="utf-8")
    out = _run("save", "post/foo.qmd")

    assert "1 earlier session(s)" in out
    assert "1 logged outside a session" in out
    assert len(_sources_and_stamps()) == 3


# ---- corpus location comes from the shared config ladder ------------------


def test_env_supplied_data_dir_is_honoured(tmp_path, monkeypatch):
    """Regression: `log` resolves its corpus through `stylebot.config`.

    It used to bake the location into module constants at import time, so an
    environment that named a corpus was ignored and a save silently landed in a
    cwd-relative `_training_pairs/` instead. The env var here is set after the
    module was imported — exactly the case the import-time read got wrong — and
    the cwd deliberately differs from the corpus, so a regression shows up as a
    stray `_training_pairs/` under `repo/`.
    """
    data_dir = tmp_path / "env-corpus"
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(data_dir))

    repo = tmp_path / "repo"
    (repo / "post").mkdir(parents=True)
    monkeypatch.chdir(repo)
    f = repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    _run("open", "post/foo.qmd")
    f.write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    out = _run("save", "post/foo.qmd")

    assert (data_dir / "pairs.jsonl").exists()
    assert (data_dir / "snapshots-closed" / "post" / "foo.qmd").exists()
    # Not in the cwd-relative default, and the confirmation names where it went.
    assert not (repo / "_training_pairs").exists()
    assert str((data_dir / "pairs.jsonl").resolve()) in out


# ---- the error idiom: UsageError -> 2, ClickException -> 1 ----------------
#
# `log` used to `echo(err=True)` then `sys.exit(N)` at 13 sites, which made the
# exit code a hand-maintained constant next to each message. It now raises like
# the rest of `ai-style`, and these pin the two classes apart: a wrong FLAG
# COMBINATION is a UsageError (2), a fine invocation against an unready world is
# a ClickException (1). Downstream scripts branch on that.


def _err(*args):
    """Invoke `log` expecting failure; return (exit_code, output)."""
    result = CliRunner().invoke(log.main, ["-q", *args])
    assert result.exit_code != 0, result.output
    return result.exit_code, result.output


def test_contradictory_flags_are_usage_errors(session_repo):
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nIt is worth noting the rich tapestry.\n", encoding="utf-8")
    CliRunner().invoke(log.main, ["-q", "open", "post/foo.qmd"])
    f.write_text("## S\n\nWe dig in.\n", encoding="utf-8")

    code, out = _err("save", "--replace", "--append", "post/foo.qmd")
    assert code == 2
    assert "mutually exclusive" in out


def test_pair_flag_misuse_is_a_usage_error(temp_corpus, tmp_path):
    before = tmp_path / "b.txt"
    after = tmp_path / "a.txt"
    before.write_text("It is worth noting the rich tapestry.\n", encoding="utf-8")
    after.write_text("We dig in.\n", encoding="utf-8")

    code, out = _err("pair", "--before", "-", "--after", "-")
    assert code == 2
    assert "only one of --before/--after may be '-'" in out

    # --replace without the --source it filters on: the flags contradict.
    code, out = _err("pair", "--before", str(before), "--after", str(after), "--replace")
    assert code == 2
    assert "--replace requires --source" in out


def test_unready_world_is_a_click_exception(session_repo, tmp_path):
    """Flags are fine; there is just nothing to act on. Exit 1, not 2."""
    f = session_repo / "post" / "foo.qmd"
    f.write_text("## S\n\nbody one.\n", encoding="utf-8")

    assert _err("save", "post/foo.qmd")[0] == 1  # no open session
    assert _err("drop", "post/foo.qmd")[0] == 1  # nothing to drop

    CliRunner().invoke(log.main, ["-q", "open", "post/foo.qmd"])
    code, out = _err("save", "post/foo.qmd")  # opened, but nothing edited
    assert code == 1
    assert "nothing to save" in out

    same = tmp_path / "same.txt"
    same.write_text("identical text\n", encoding="utf-8")
    code, out = _err("pair", "--before", str(same), "--after", str(same))
    assert code == 1
    assert "refusing to log" in out
