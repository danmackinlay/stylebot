"""Direct tests for stylebot.lib — the generic frontmatter/markdown layer.

Everything here feeds corpus extraction (Phase 1 diffs and Phase 2 targets both
come through these functions), so regressions corrupt training data silently.
`segment_for_edit`'s protected-block semantics are pinned in test_segment.py;
this file covers the rest plus the lossless round-trip property on inputs
nastier than test_segment's.
"""

from __future__ import annotations

import pytest
from ruamel.yaml import YAMLError

from stylebot.lib import (
    gather_qmd_files,
    is_valid_qmd_file,
    read_w_frontmatter_text,
    segment_for_edit,
    split_paragraphs,
)

# -- read_w_frontmatter_text --


def test_frontmatter_parses_meta_and_body():
    meta, body = read_w_frontmatter_text("---\ntitle: x\nautomation: 0\n---\n\nbody text\n")
    assert meta["title"] == "x"
    assert meta["automation"] == 0
    assert body == "\nbody text\n"


def test_no_frontmatter_returns_whole_text():
    assert read_w_frontmatter_text("just prose\n") == ({}, "just prose\n")


def test_empty_text():
    assert read_w_frontmatter_text("") == ({}, "")


def test_frontmatter_dots_terminator():
    # YAML's document-end marker `...` closes frontmatter like `---` does.
    meta, body = read_w_frontmatter_text("---\ntitle: x\n...\nbody\n")
    assert meta == {"title": "x"}
    assert body == "body\n"


def test_frontmatter_crlf_line_endings():
    meta, body = read_w_frontmatter_text("---\r\ntitle: x\r\n---\r\nbody\r\n")
    assert meta == {"title": "x"}
    assert body == "body\r\n"


def test_empty_frontmatter_block_gives_empty_meta():
    meta, body = read_w_frontmatter_text("---\n---\nbody\n")
    assert meta == {}
    assert body == "body\n"


def test_malformed_frontmatter_raises():
    # Contract: garbage YAML propagates — callers own the guard (see
    # targets._load_meta_body, which treats an unreadable post as unselected).
    with pytest.raises(YAMLError):
        read_w_frontmatter_text("---\ntitle: x\nno closing delimiter")


# -- split_paragraphs --


def test_split_on_blank_lines():
    assert split_paragraphs("one\n\ntwo\n\nthree\n") == ["one", "two", "three"]


def test_multiline_paragraph_stays_together():
    assert split_paragraphs("line a\nline b\n\nnext\n") == ["line a\nline b", "next"]


def test_runs_of_blank_lines_collapse():
    assert split_paragraphs("a\nb\n\n\n\nc\n") == ["a\nb", "c"]


def test_no_trailing_blank_line():
    assert split_paragraphs("only paragraph") == ["only paragraph"]


def test_empty_input_gives_no_paragraphs():
    assert split_paragraphs("") == []
    assert split_paragraphs("\n\n\n") == []


# -- segment_for_edit: the lossless round-trip property on nasty inputs --


@pytest.mark.parametrize(
    "doc",
    [
        # markers inside a fence must not open spans of their own
        "p1\n\n```txt\n::: not a div\n$$ not math $$\n```\n\np2\n",
        # inline math with a Quarto attr block
        "text $$e=mc^2$$ {#eq-x} more\n",
        # blockquote at EOF without trailing newline
        "lead\n\n> quoted\n> more quoted",
        # adjacent protected blocks
        "```a\nx\n```\n$$y$$\n> q\n",
        # unclosed div: nothing protected, but nothing lost either
        "p\n\n::: open\nnever closed\n",
        # empty document
        "",
    ],
)
def test_segment_roundtrip_is_lossless(doc):
    assert "".join(seg for seg, _ in segment_for_edit(doc)) == doc


def test_fence_swallows_inner_markers():
    doc = "p1\n\n```txt\n::: not a div\n$$ not math $$\n```\n\np2\n"
    protected = [seg for seg, editable in segment_for_edit(doc) if not editable]
    assert len(protected) == 1  # ONE fence span, not fence+div+math
    assert "::: not a div" in protected[0]


def test_unclosed_div_stays_editable():
    # No closing ::: -> no protected span; the text must not be dropped.
    segs = segment_for_edit("p\n\n::: open\nnever closed\n")
    assert segs == [("p\n\n::: open\nnever closed\n", True)]


# -- file gathering --


def test_is_valid_qmd_file(tmp_path):
    ok = tmp_path / "post" / "note.qmd"
    ok.parent.mkdir()
    ok.write_text("x", encoding="utf-8")
    assert is_valid_qmd_file(ok) is True
    assert is_valid_qmd_file(ok.parent) is False  # directories excluded
    assert is_valid_qmd_file(tmp_path / "_draft.qmd") is False
    assert is_valid_qmd_file(tmp_path / ".hidden" / "a.qmd") is False
    assert is_valid_qmd_file(tmp_path / "_site" / "a.qmd") is False
    assert is_valid_qmd_file(tmp_path / "renv" / "a.qmd") is False
    assert is_valid_qmd_file(tmp_path / "node_modules" / "p" / "a.qmd") is False


# Every case below passed the pre-2026-08-12 predicate, which tested
# `path.name` — the filename — so an underscore *directory* was invisible to
# it. These feed `iter_targets`, i.e. scratch prose offered up as a training
# target (livingthing `_plans/qmd-walker-2026-08.md`).
@pytest.mark.parametrize(
    "rel",
    ["_tmp/scratch.qmd", "_freeze/a/b.qmd", "_site/index.qmd", "_plans/design.qmd"],
)
def test_is_valid_qmd_file_rejects_underscore_directories(tmp_path, rel):
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_text("x", encoding="utf-8")
    assert is_valid_qmd_file(p) is False


def test_is_valid_qmd_file_base_makes_the_judgement_project_relative(tmp_path):
    """A checkout under a `_`- or `.`-prefixed directory is not disqualified."""
    base = tmp_path / "_checkout"
    ok = base / "post" / "note.qmd"
    ok.parent.mkdir(parents=True)
    ok.write_text("x", encoding="utf-8")
    assert is_valid_qmd_file(ok) is False  # judged whole: `_checkout` bites
    assert is_valid_qmd_file(ok, base) is True  # judged below `base`
    assert gather_qmd_files([], base=base) == [ok]


def test_gather_qmd_files_walk_skips_build_dirs(tmp_path):
    (tmp_path / "post").mkdir()
    (tmp_path / "post" / "keep.qmd").write_text("x", encoding="utf-8")
    (tmp_path / "_site").mkdir()
    (tmp_path / "_site" / "built.qmd").write_text("x", encoding="utf-8")
    (tmp_path / "_tmp" / "eval").mkdir(parents=True)
    (tmp_path / "_tmp" / "eval" / "scratch.qmd").write_text("x", encoding="utf-8")
    (tmp_path / "post" / "_draft.qmd").write_text("x", encoding="utf-8")
    found = gather_qmd_files([], base=tmp_path)
    assert [p.name for p in found] == ["keep.qmd"]


def test_gather_qmd_files_honours_a_non_qmd_glob(tmp_path):
    """The suffix is the caller's glob to choose, not this predicate's to fix."""
    (tmp_path / "post").mkdir()
    (tmp_path / "post" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "post" / "b.qmd").write_text("x", encoding="utf-8")
    found = gather_qmd_files([], base=tmp_path, default_glob="**/*.md")
    assert [p.name for p in found] == ["a.md"]


def test_gather_qmd_files_explicit_list_filters_missing_and_invalid(tmp_path):
    keep = tmp_path / "keep.qmd"
    keep.write_text("x", encoding="utf-8")
    draft = tmp_path / "_draft.qmd"
    draft.write_text("x", encoding="utf-8")
    found = gather_qmd_files([str(keep), str(draft), str(tmp_path / "missing.qmd")])
    assert found == [keep]
