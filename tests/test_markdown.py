"""Unit tests for the shared markdown primitives (`stylebot.markdown`).

`segment_for_edit`'s end-to-end protected-block behaviour is pinned in
test_segment.py and the train/serve contract in test_protection_parity.py; this
file pins the primitives those rest on.
"""

from __future__ import annotations

from stylebot import markdown as md

# -- headings ---------------------------------------------------------------


def test_atx_heading_accepts_up_to_three_spaces():
    assert md.is_heading_line("# One")
    assert md.is_heading_line("   ### Three spaces")
    assert not md.is_heading_line("    #### Four spaces is a code block")


def test_atx_heading_rejects_a_leading_tab():
    # CommonMark: a tab indents by 4, so it cannot open a heading.
    assert not md.is_heading_line("\t## Tabbed")


def test_atx_heading_rejects_hashtags_and_over_deep_levels():
    assert not md.is_heading_line("#hashtag")
    assert not md.is_heading_line("####### seven hashes")
    assert not md.is_heading_line("not a heading at all")


def test_bare_hashes_are_an_empty_heading():
    assert md.is_heading_line("##")
    assert md.heading_title("##") == ""


def test_heading_title_strips_hashes_and_trailing_space():
    assert md.heading_title("##  Some Title  ") == "Some Title"
    assert md.heading_title("plain text") is None


def test_heading_line_tolerates_a_trailing_newline():
    assert md.is_heading_line("## With newline\n")
    assert md.heading_title("## With newline\n") == "With newline"


# -- protected spans --------------------------------------------------------


def _kinds(text, **kw):
    return [k for _s, _e, k in md.protected_spans(text, **kw)]


def test_protected_spans_label_each_construct():
    doc = "p\n\n```py\nx\n```\n\n$$\na\n$$\n\n> quote\n\n::: {.note}\nn\n:::\n\ntail\n"
    assert _kinds(doc) == [md.FENCE, md.MATH, md.BLOCKQUOTE, md.DIV]


def test_unclosed_fence_yields_no_span():
    # CODE_FENCE_RE needs a closer; consumers that must swallow a runaway fence
    # keep their own state machine (infer._blocks_with_separators).
    assert md.protected_spans("p\n\n```py\nnever closed\n") == []


def test_unclosed_div_yields_no_span():
    assert md.protected_spans("p\n\n::: open\nnever closed\n") == []


def test_block_only_keeps_display_math():
    doc = "text\n\n$$\nx = 1\n$$ {#eq-a}\n\nmore\n"
    assert _kinds(doc, block_only=True) == [md.MATH]


def test_block_only_drops_midline_math():
    doc = "high-$$d$$ behaviour is strange.\n"
    assert _kinds(doc) == [md.MATH]
    assert _kinds(doc, block_only=True) == []


def test_block_only_keeps_indented_display_math():
    assert _kinds("   $$\nx\n$$\n", block_only=True) == [md.MATH]
    # Four spaces is a code block, not display math.
    assert _kinds("    $$\nx\n$$\n", block_only=True) == []


def test_segment_spans_roundtrip_is_lossless():
    doc = "a\n\n```\nc\n```\n\n$$m$$\n\n> q\n\n::: d\nx\n:::\n\nz"
    assert "".join(seg for seg, _ in md.segment_spans(doc)) == doc


# -- div policy -------------------------------------------------------------

CALLOUT = (
    "before\n\n"
    "::: {.callout-note}\n"
    "Callout prose one.\n\n"
    "Callout prose two.\n"
    "::: {#fig-x}\n"
    "![](x.png)\n"
    ":::\n"
    ":::\n\n"
    "after\n"
)


def test_div_default_protects_everything():
    prose = "".join(s for s, e in md.segment_spans(CALLOUT) if e)
    assert "Callout prose one." not in prose
    assert "fig-x" not in prose


def test_div_is_prose_frees_contents_but_not_fences_or_nested_figures():
    policy = lambda attrs: "callout" in attrs  # noqa: E731
    prose = "".join(s for s, e in md.segment_spans(CALLOUT, div_is_prose=policy) if e)
    assert "Callout prose one." in prose
    assert "Callout prose two." in prose
    assert "callout-note" not in prose  # the ::: fence lines stay protected
    assert "fig-x" not in prose  # a nested figure keeps its own protection
    assert "![](x.png)" not in prose


def test_div_is_prose_stays_lossless():
    policy = lambda attrs: "callout" in attrs  # noqa: E731
    assert "".join(s for s, _ in md.segment_spans(CALLOUT, div_is_prose=policy)) == CALLOUT


# -- line kinds -------------------------------------------------------------


def test_line_kinds_labels_one_per_line():
    doc = "## H\n\nprose\n\n```py\nx\n```\n"
    assert md.line_kinds(doc) == [
        md.HEADING,
        None,
        None,
        None,
        md.FENCE,
        md.FENCE,
        md.FENCE,
        None,  # the trailing "" after the final newline
    ]


def test_line_kinds_do_not_bleed_past_a_span():
    doc = "> quoted\n\nprose after\n"
    assert md.line_kinds(doc) == [md.BLOCKQUOTE, None, None, None]


def test_line_kinds_mark_blank_lines_inside_a_fence():
    doc = "```py\na\n\nb\n```\n"
    assert md.line_kinds(doc) == [md.FENCE] * 5 + [None]


def test_line_kinds_ignore_midline_math():
    assert md.line_kinds("high-$$d$$ behaviour\n") == [None, None]


def test_line_kinds_can_skip_headings():
    doc = "# H\nprose\n"
    assert md.line_kinds(doc, headings=False) == [None, None, None]


def test_line_kinds_see_callout_prose_under_policy():
    policy = lambda attrs: "callout" in attrs  # noqa: E731
    kinds = md.line_kinds(CALLOUT, div_is_prose=policy)
    lines = CALLOUT.split("\n")
    got = dict(zip(lines, kinds, strict=True))
    assert got["::: {.callout-note}"] == md.DIV
    assert got["Callout prose one."] is None
    assert got["::: {#fig-x}"] == md.DIV
    assert got["![](x.png)"] == md.DIV


def test_line_kinds_pin_a_heading_inside_a_callout():
    doc = "::: {.callout-note}\n## Inner heading\ntext\n:::\n"
    policy = lambda attrs: "callout" in attrs  # noqa: E731
    assert md.line_kinds(doc, div_is_prose=policy) == [md.DIV, md.HEADING, None, md.DIV, None]
