"""Tests for the generic markdown segmenter ported into stylebot.lib.

These pin the editable/protected behaviour (a fork of livingthing's
qmd_core.segment_for_edit) so the two copies don't drift. The figure-`:::`-div
and code/math/blockquote cases are the leaks the old `_is_prose_chunk` missed.
"""

from __future__ import annotations

from stylebot.lib import editable_prose, segment_for_edit


def _protected(text):
    return [seg for seg, editable in segment_for_edit(text) if not editable]


def test_roundtrip_is_lossless():
    doc = "intro\n\n```py\ncode\n```\n\nmiddle\n\n$$x=1$$\n\nend\n"
    assert "".join(seg for seg, _ in segment_for_edit(doc)) == doc


def test_code_fence_is_protected():
    doc = "before\n\n```python\nprint('hi')\n```\n\nafter\n"
    assert any("print('hi')" in p for p in _protected(doc))
    assert "print('hi')" not in editable_prose(doc)
    assert "before" in editable_prose(doc)
    assert "after" in editable_prose(doc)


def test_math_block_is_protected():
    doc = "text\n\n$$\n\\int_0^1 x\\,dx\n$$\n\nmore\n"
    assert any("\\int_0^1" in p for p in _protected(doc))
    assert "\\int" not in editable_prose(doc)


def test_div_block_is_protected():
    # The exact leak: a Quarto figure div that the old filter let through.
    doc = (
        "real paragraph one.\n\n"
        ":::{#fig-x .figure .illustration}\n"
        "![](/images/x.png)\n"
        ":::\n\n"
        "real paragraph two.\n"
    )
    prot = _protected(doc)
    assert any("![](/images/x.png)" in p for p in prot)
    prose = editable_prose(doc)
    assert "fig-x" not in prose
    assert "real paragraph one." in prose
    assert "real paragraph two." in prose


def test_nested_div_blocks():
    doc = "p\n\n::: outer\n::: inner\nx\n:::\n:::\n\nq\n"
    prose = editable_prose(doc)
    assert "p" in prose and "q" in prose
    assert "outer" not in prose and "inner" not in prose


def test_blockquote_is_protected():
    doc = "lead in.\n\n> a quoted line\n> another\n\ntail.\n"
    assert any("quoted line" in p for p in _protected(doc))
    assert "quoted line" not in editable_prose(doc)


def test_plain_prose_is_all_editable():
    doc = "just one paragraph.\n\nand a second paragraph.\n"
    assert _protected(doc) == []
    assert editable_prose(doc) == doc
