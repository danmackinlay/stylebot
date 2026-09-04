"""Unit tests for the shared markdown primitives (`stylebot.markdown`).

The segmenter's unit spec. `lib.segment_for_edit` is a one-line call to
`segment_spans`, so its behaviour is pinned here and nowhere else; the
train/serve contract over a full document is test_infer_segment.py, and the two
named policies over real posts are test_segment_hard.py.
"""

from __future__ import annotations

import pytest

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


def test_unclosed_fence_runs_to_end_of_document():
    # CommonMark: an unclosed fence's block "contains all of the lines after
    # the opening fence". This lived in `infer._blocks_with_separators` as a
    # separate state machine until 2026-09-03, so inference swallowed a runaway
    # fence and `editable_prose` — every training target — did not.
    doc = "p\n\n```py\nnever closed\n"
    assert md.protected_spans(doc) == [(3, len(doc), md.FENCE)]


def test_a_closed_fence_still_stops_at_its_closer():
    # The run-to-EOF branch is an alternative of last resort: the closed form is
    # tried first and its search is lazy, so nothing about a normal fence moves.
    doc = "p\n\n```py\nx\n```\n\ntail\n"
    assert md.protected_spans(doc) == [(3, 15, md.FENCE)]


def test_a_four_backtick_fence_can_close_itself():
    # The opener takes a bare three then `.*?$`, so ```` opens; the closer takes
    # `{3,}` so it also CLOSES. With a bare-three closer it could not, and the
    # fence ran to the end of the document instead — measured live in
    # notebook/quarto.qmd, where one span overran by 999 chars and swallowed
    # three callout divs.
    doc = "p\n\n````markdown\n{{< include x.md >}}\n````\n\ntail\n"
    spans = md.protected_spans(doc)
    assert [k for _s, _e, k in spans] == [md.FENCE]
    assert doc[spans[0][0] : spans[0][1]] == "````markdown\n{{< include x.md >}}\n````\n"


def test_a_fence_swallows_the_markers_inside_it():
    # ONE fence span, not fence + div + math.
    doc = "p1\n\n```txt\n::: not a div\n$$ not math $$\n```\n\np2\n"
    spans = md.protected_spans(doc)
    assert [k for _s, _e, k in spans] == [md.FENCE]
    assert "::: not a div" in doc[spans[0][0] : spans[0][1]]


def test_unclosed_div_yields_no_span():
    assert md.protected_spans("p\n\n::: open\nnever closed\n") == []


def test_block_only_keeps_display_math():
    doc = "text\n\n$$\nx = 1\n$$ {#eq-a}\n\nmore\n"
    assert _kinds(doc, block_only=True) == [md.MATH]


def test_one_line_display_math_with_attrs_is_protected():
    assert _kinds("text\n\n$$x = 1$$ {#eq-a}\n\nmore\n", block_only=True) == [md.MATH]


def test_block_only_drops_midline_math():
    # The corruption `block_only` fixed: excising a mid-sentence `$$d$$` joined
    # the remnants, so a TRAINING TARGET read "high- behaviour".
    doc = "high-$$d$$ behaviour is strange.\n"
    assert _kinds(doc) == [md.MATH]
    assert _kinds(doc, block_only=True) == []


def test_block_only_keeps_indented_display_math():
    assert _kinds("   $$\nx\n$$\n", block_only=True) == [md.MATH]
    # Four spaces is a code block, not display math.
    assert _kinds("    $$\nx\n$$\n", block_only=True) == []


@pytest.mark.parametrize(
    "doc",
    [
        "a\n\n```\nc\n```\n\n$$m$$\n\n> q\n\n::: d\nx\n:::\n\nz",
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
        "",
    ],
)
def test_segment_spans_roundtrip_is_lossless(doc):
    assert "".join(seg for seg, _ in md.segment_spans(doc)) == doc


# -- the opt-in kinds: raw HTML and HTML comments ---------------------------

SCRIPT = 'before\n\n<script src="x.js">\nconst a = 1;\n\nconst b = 2;\n</script>\n\nafter\n'


def test_raw_html_is_off_by_default():
    assert _kinds(SCRIPT) == []
    assert "const a = 1;" in "".join(s for s, e in md.segment_spans(SCRIPT) if e)


def test_raw_html_protects_script_blocks_when_enabled():
    assert _kinds(SCRIPT, raw_html=True) == [md.RAW_HTML]
    prose = "".join(s for s, e in md.segment_spans(SCRIPT, raw_html=True) if e)
    assert "const a = 1;" not in prose
    assert "const b = 2;" not in prose  # the blank line inside must not end it
    assert "before" in prose and "after" in prose


def test_raw_html_protects_style_blocks_and_ignores_case():
    doc = "p\n\n<STYLE>\nbody { color: red }\n</Style>\n\nq\n"
    assert _kinds(doc, raw_html=True) == [md.RAW_HTML]


def test_raw_html_leaves_other_elements_alone():
    # Every other element's content IS markup, so prose inside it stays prose.
    doc = "p\n\n<div class='x'>\nOrdinary prose inside a raw div.\n</div>\n\nq\n"
    assert _kinds(doc, raw_html=True) == []


def test_unclosed_script_yields_no_span():
    assert md.protected_spans("p\n\n<script>\nnever closed\n", raw_html=True) == []


def test_raw_html_requires_the_opener_to_start_a_line():
    doc = "Prose mentioning <script>inline()</script> mid-sentence.\n"
    assert _kinds(doc, raw_html=True) == []


def test_raw_html_does_not_span_two_separate_blocks():
    doc = "<script>\na\n</script>\n\ntext between\n\n<script>\nb\n</script>\n"
    assert _kinds(doc, raw_html=True) == [md.RAW_HTML, md.RAW_HTML]
    assert "text between" in "".join(s for s, e in md.segment_spans(doc, raw_html=True) if e)


def test_raw_html_stays_lossless_and_reaches_line_kinds():
    assert "".join(s for s, _ in md.segment_spans(SCRIPT, raw_html=True)) == SCRIPT
    kinds = md.line_kinds(SCRIPT, raw_html=True)
    got = dict(zip(SCRIPT.split("\n"), kinds, strict=True))
    assert got["const a = 1;"] == md.RAW_HTML
    assert got["before"] is None
    assert md.line_kinds(SCRIPT)[3] is None  # default off, all the way down


def test_html_comments_are_off_by_default_and_span_lines_when_enabled():
    doc = "p\n\n<!-- one line -->\n\nq <!-- opened\nmid-line, closed later -->\n\nr\n"
    assert _kinds(doc) == []
    spans = md.protected_spans(doc, html_comments=True)
    assert [k for _s, _e, k in spans] == [md.HTML_COMMENT, md.HTML_COMMENT]
    # A comment opened mid-line takes only the comment, not the prose before it.
    assert doc[spans[1][0] : spans[1][1]] == "<!-- opened\nmid-line, closed later -->"


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
