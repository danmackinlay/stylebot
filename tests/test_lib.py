"""Direct tests for stylebot.lib — the generic frontmatter/markdown layer.

Everything here feeds corpus extraction (Phase 1 diffs and Phase 2 targets both
come through these functions), so regressions corrupt training data silently.
`segment_for_edit` is a one-line call to `markdown.segment_spans`, pinned in
test_markdown.py; this file covers frontmatter, paragraphs and file gathering.
"""

from __future__ import annotations

import pytest
from ruamel.yaml import YAMLError

from stylebot.lib import (
    SourcePolicy,
    gather_qmd_files,
    is_valid_qmd_file,
    read_w_frontmatter_text,
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
# target (livingthing `_plans/retired/qmd-walker-2026-08.md`).
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


# --- SourcePolicy: the blog-shaped literals in one object (W4 step 5) --------


def test_source_policy_renders_its_authors_prompt():
    from stylebot.ai_core import DEFAULT_AUTHOR, STYLE_SYSTEM, author_of, style_system

    assert SourcePolicy().system_prompt == STYLE_SYSTEM
    ada = SourcePolicy(author="Ada Lovelace")
    assert ada.system_prompt == style_system("Ada Lovelace")
    assert author_of(ada.system_prompt) == "Ada Lovelace"
    assert author_of(STYLE_SYSTEM) == DEFAULT_AUTHOR
    assert author_of("not the template") is None


def test_source_policy_defaults_are_the_blogs_literals():
    policy = SourcePolicy()
    assert (policy.glob, policy.meta_field, policy.max_level) == ("**/*.qmd", "automation", 0)
    assert policy.selects({"automation": 0})
    assert not policy.selects({"automation": 1}) and not policy.selects({})
    assert policy.level_of({"automation": 2}) == 2
    assert policy.level_of({}) is None and policy.level_of(None) is None
    assert policy.level_of({"automation": True}) is None  # bool is not a level


def test_source_policy_retargets_the_glob_field_and_level():
    policy = SourcePolicy(glob="**/*.md", meta_field="ai_level", max_level=2)
    assert policy.selects({"ai_level": 2}) and not policy.selects({"ai_level": 3})
    assert not policy.selects({"automation": 0}), "another blog's field means nothing here"
    assert policy.level_of({"ai_level": 1, "automation": 9}) == 1
