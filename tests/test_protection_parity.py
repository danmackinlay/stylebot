"""The train/serve contract this whole refactor exists to enforce.

What `lib.segment_for_edit` marks non-editable — and so never becomes a
training target — must be exactly what `infer._segment` refuses to send to the
styler. When the two disagree, the adapter meets constructs at inference that
it never saw in training, and the failure is silent: a styler quietly rewording
somebody else's quoted words looks like a successful rewrite.

Both sides are reduced to a set of protected LINE indices over one fixture that
contains every construct, and compared. Headings are excluded from the
comparison and asserted separately: they are protected on both sides, but by
different machinery (`targets._split_sections` upstream of the chunker, versus
`_segment`'s own kind), so a line-set comparison is the wrong instrument.

Following the `tests/test_marker_segmentation.py` idiom: one shared fixture,
asserted from both directions, with a test that the fixture actually exercises
every case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stylebot import infer
from stylebot import markdown as md
from stylebot.lib import segment_for_edit

FIXTURE = Path(__file__).parent / "fixtures" / "protection_parity.qmd"
BIG = 1_000_000  # never let the chunk budget split anything in these tests


def _line_starts(lines: list[str]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
    return starts


def _lines_touching(content: str, spans: list[tuple[int, int]]) -> set[int]:
    """Indices of `content.split("\\n")` lines any of `spans` overlaps."""
    lines = content.split("\n")
    starts = _line_starts(lines)
    hit: set[int] = set()
    for span_start, span_end in spans:
        for i, line in enumerate(lines):
            if starts[i] >= span_end:
                break
            if span_start < starts[i] + len(line) + 1:
                hit.add(i)
    return hit


def training_protected_lines(content: str, **kw) -> set[int]:
    """Lines `lib.segment_for_edit` refuses to hand the target cutter.

    Derived from `segment_for_edit`'s own char spans rather than from
    `markdown.line_kinds`, so the two sides of the comparison stay independent
    paths through the code.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for seg, editable in segment_for_edit(content, **kw):
        if not editable:
            spans.append((pos, pos + len(seg)))
        pos += len(seg)
    return _lines_touching(content, spans)


def _segment_spans(content: str, **kw) -> list[tuple[str, int, int]]:
    """`infer._segment`'s items as `(kind, start, end)` over the whole document."""
    frontmatter, body = infer.split_frontmatter(content)
    items, seps = infer._segment(body, max_chars=BIG, **kw)
    out: list[tuple[str, int, int]] = []
    pos = len(frontmatter)
    for i, (kind, text) in enumerate(items):
        if i:
            pos += len(seps[i - 1])
        out.append((kind, pos, pos + len(text)))
        pos += len(text)
    return out


def inference_protected_lines(content: str, **kw) -> set[int]:
    """Lines `infer._segment` refuses to send to the styler."""
    spans = [(s, e) for kind, s, e in _segment_spans(content, **kw) if kind in md.PROTECTED_KINDS]
    return _lines_touching(content, spans)


@pytest.fixture(scope="module")
def doc() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --- the contract ----------------------------------------------------------


def test_training_and_inference_protect_the_same_lines(doc: str):
    train = training_protected_lines(doc)
    serve = inference_protected_lines(doc)
    lines = doc.split("\n")
    assert sorted(train - serve) == [], (
        "training drops these lines but inference would send them: "
        + repr([lines[i] for i in sorted(train - serve)])
    )
    assert sorted(serve - train) == [], (
        "inference refuses these lines but they can become training targets: "
        + repr([lines[i] for i in sorted(serve - train)])
    )


def test_headings_are_protected_at_inference(doc: str):
    heading_lines = {
        i for i, line in enumerate(doc.split("\n")) if md.is_heading_line(line)
    } - training_protected_lines(doc)
    sent = {
        i
        for kind, s, e in _segment_spans(doc)
        if kind not in md.PROTECTED_KINDS and kind != "heading"
        for i in _lines_touching(doc, [(s, e)])
    }
    assert not heading_lines & sent


def test_segment_reassembly_is_byte_exact(doc: str):
    frontmatter, body = infer.split_frontmatter(doc)
    for inline_headings in (False, True):
        items, seps = infer._segment(body, max_chars=BIG, inline_headings=inline_headings)
        rebuilt = "".join(
            text + (seps[i] if i < len(seps) else "") for i, (_kind, text) in enumerate(items)
        )
        assert frontmatter + rebuilt == doc


def test_segment_reassembly_survives_a_tight_budget(doc: str):
    frontmatter, body = infer.split_frontmatter(doc)
    items, seps = infer._segment(body, max_chars=40, inline_headings=True)
    rebuilt = "".join(
        text + (seps[i] if i < len(seps) else "") for i, (_kind, text) in enumerate(items)
    )
    assert frontmatter + rebuilt == doc


def test_segment_for_edit_is_lossless(doc: str):
    assert "".join(seg for seg, _ in segment_for_edit(doc)) == doc


# --- the fixture is only a guard if it contains the cases ------------------


def test_fixture_exercises_every_protected_kind(doc: str):
    kinds = {kind for kind, _s, _e in _segment_spans(doc)}
    assert kinds >= md.PROTECTED_KINDS, f"fixture misses {md.PROTECTED_KINDS - kinds}"
    assert "heading" in kinds


def test_fixture_contains_every_named_case(doc: str):
    protected = "".join(seg for seg, editable in segment_for_edit(doc) if not editable)
    assert "a backtick fence" in protected
    assert "a tilde fence" in protected
    assert "\\int_0^1" in protected  # block $$…$$ carrying {#eq-…} attrs
    assert "somebody else's voice" in protected  # blockquote
    assert "no blank line after it" in protected  # blockquote glued to prose
    assert "{#refs}" in protected
    assert "fig-parity" in protected
    assert "callout-note" in protected
    assert "fig-nested" in protected  # a figure nested inside a callout

    lines = doc.split("\n")
    assert "  ## An indented heading" in lines
    assert "## A heading glued to prose" in lines
    assert "## A heading inside a callout" in lines
    assert "high-$$d$$ behaviour" in doc  # the mid-line math case
    assert "::: an-opener-that-is-never-closed" in lines


def test_an_unclosed_div_protects_nothing(doc: str):
    # An opener with no closer yields no span, on either side: over-protecting
    # to end-of-document would swallow the rest of the post.
    assert "never-closed" not in "".join(
        seg for seg, editable in segment_for_edit(doc) if not editable
    )
    last = _segment_spans(doc)[-1]
    assert last[0] not in md.PROTECTED_KINDS


# --- the prose-bearing div policy (A-7) ------------------------------------

CALLOUT_POLICY = "callout"


def _is_callout(attrs: str) -> bool:
    return CALLOUT_POLICY in attrs


def test_callout_policy_frees_the_same_contents_on_both_sides(doc: str):
    train = training_protected_lines(doc, div_is_prose=_is_callout)
    serve = inference_protected_lines(doc, div_is_prose=_is_callout)
    assert train == serve


def test_callout_policy_admits_a_heading_inside_a_callout(doc: str):
    # A `##` inside a callout is common and becomes a section boundary in
    # `_split_sections` once the callout's contents are prose — correct, and
    # pinned here because it is easy to break silently.
    kinds = md.line_kinds(doc, div_is_prose=_is_callout)
    idx = doc.split("\n").index("## A heading inside a callout")
    assert kinds[idx] == md.HEADING

    at_inference = [
        kind
        for kind, s, e in _segment_spans(doc, div_is_prose=_is_callout)
        if doc[s:e].strip() == "## A heading inside a callout"
    ]
    assert at_inference == ["heading"]


def test_callout_policy_keeps_fences_and_nested_figures_protected(doc: str):
    prose = "".join(
        seg for seg, editable in segment_for_edit(doc, div_is_prose=_is_callout) if editable
    )
    assert "Callout prose paragraph one" in prose
    assert "Callout prose paragraph two" in prose
    assert "Trailing callout prose" in prose
    assert "callout-note" not in prose  # the ::: lines carry class semantics
    assert "callout-warning" not in prose
    assert "fig-nested" not in prose  # a nested figure keeps its own protection
    assert "nested.png" not in prose
    assert "fig-parity" not in prose  # an unrelated figure div is untouched


def test_callout_policy_stays_lossless(doc: str):
    segs = segment_for_edit(doc, div_is_prose=_is_callout)
    assert "".join(seg for seg, _ in segs) == doc
