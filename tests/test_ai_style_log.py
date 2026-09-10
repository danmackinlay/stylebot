"""`ai-style log` as a thin shell: the context prefix on captured pairs (D14),
the `--no-heading-context` / `--whole` opt-outs, the corpus it resolves
through `stylebot.config`, and the two exit-code classes. The session rules
themselves are test_session.py's; `diff_chunks` is test_capture.py's.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from conftest import corpus_records

from stylebot import capture, session
from stylebot.ai_core import STYLE_SYSTEM
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


# ---- captured pair has context prefix + validates ------------------------


def test_captured_pair_has_heading_prefix_and_validates(temp_corpus):
    before_text = "## Section\n\nIt is worth noting the rich tapestry."
    after_text = "## Section\n\nWe dig in."
    written = capture.capture_pairs(
        before_text,
        after_text,
        pairs_path=session.SessionStore(temp_corpus).pairs_path,
        source="post/foo.qmd",
        snapped_at="2026-06-28T00:00:00+00:00",
        whole=False,
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
        system=STYLE_SYSTEM,
        context="## A Heading",
    )
    assert rec["messages"][0]["content"] == STYLE_SYSTEM
    assert rec["messages"][1]["content"] == build_pair_content("## A Heading", "slop body")
    assert rec["messages"][2]["content"] == build_pair_content("## A Heading", "dan body")
    assert rec["meta"]["context"] == "## A Heading"
    assert rec["meta"]["context_mode"] == "immediate"
    assert validate_pair_record(rec) == []


# ---- opt-outs: --no-heading-context, --whole, preamble -------------------


def test_no_heading_context_leaves_content_bare(temp_corpus):
    before_text = "## Section\n\nold body here."
    after_text = "## Section\n\nnew body here."
    written = capture.capture_pairs(
        before_text,
        after_text,
        pairs_path=session.SessionStore(temp_corpus).pairs_path,
        source="post/foo.qmd",
        snapped_at=None,
        whole=False,
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
    written = capture.capture_pairs(
        before_text,
        after_text,
        pairs_path=session.SessionStore(temp_corpus).pairs_path,
        source="post/foo.qmd",
        snapped_at=None,
        whole=True,
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


# ---- the CLI's own state: a working tree, and the corpus it names ---------


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

    assert [r["meta"]["source"] for r in corpus_records(data_dir)] == ["post/foo.qmd"]
    assert (data_dir / "snapshots-closed" / "post" / "foo.qmd").exists()
    # Not in the cwd-relative default, and the confirmation names where it went.
    assert not (repo / "_training_pairs").exists()
    assert str((data_dir / "draws").resolve()) in out


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


def test_div_policy_env_reaches_save(session_repo, temp_corpus, monkeypatch):
    """$STYLEBOT_DIV_POLICY names the differ's div policy for the whole `log` group (D100)."""
    from click.testing import CliRunner

    from stylebot.bin import ai_style_log as log

    post = session_repo / "post" / "callout.qmd"
    callout = "::: {.callout-note}\n\nIt is worth noting the rich tapestry.\n\n:::\n"
    post.write_text(callout, encoding="utf-8")
    runner = CliRunner()
    assert runner.invoke(log.main, ["-q", "open", "post/callout.qmd"]).exit_code == 0
    post.write_text(callout.replace("rich tapestry", "plain weave"), encoding="utf-8")
    monkeypatch.setenv("STYLEBOT_DIV_POLICY", "operator:truth")
    result = runner.invoke(log.main, ["-q", "save", "post/callout.qmd"])
    assert result.exit_code == 0, result.output
    assert "saved 1 pair" in result.output or "1 pair" in result.output
