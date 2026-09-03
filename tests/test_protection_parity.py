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

One shared fixture, asserted from both directions, with a test that the fixture
actually exercises every case. `tests/test_serve.py` re-uses the same fixture for
the editor seam's third reading of it.
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
    # to end-of-document would swallow the rest of the post. (A `:::` opener,
    # note — an unclosed *fence* does exactly the opposite; see below. The two
    # differ because a div's contents are markdown that renders either way,
    # while a fence's are code that a rewrite would destroy.)
    assert "never-closed" not in "".join(
        seg for seg, editable in segment_for_edit(doc) if not editable
    )
    opener = "::: an-opener-that-is-never-closed"
    idx = doc.split("\n").index(opener)
    assert idx not in training_protected_lines(doc)
    assert idx not in inference_protected_lines(doc)


def test_an_unclosed_fence_runs_to_eof_on_both_sides(doc: str):
    """The gap `markdown.py` used to document, closed 2026-09-03 (W2).

    `CODE_FENCE_RE` needed a closer, so a runaway fence yielded no span: it was
    protected at inference by a state machine in `infer._blocks_with_separators`
    and was a legitimate training target in `editable_prose`. The state machine
    moved into `CODE_FENCE_RE` as a run-to-EOF alternative, so both sides now
    read the same answer off one pattern — which is what the line-set contract
    above asserts generally, and this pins by name.
    """
    runaway = 'return "an unclosed fence runs to end of document"'
    protected = "".join(seg for seg, editable in segment_for_edit(doc) if not editable)
    assert runaway in protected
    kinds = [kind for kind, _s, _e in _segment_spans(doc) if runaway in doc[_s:_e]]
    assert kinds == [md.FENCE]


# --- packing parity (D27, amended 2026-09-03) ------------------------------
#
# `targets._post_targets` deletes protected blocks and THEN packs paragraphs, so
# a passage either side of a figure becomes one target — a passage the document
# does not contain and `infer._segment` can never present, because it breaks a
# chunk at every protected item. `ChunkPolicy.pack_across_protected=False` is the
# parity setting; True is what the live corpus was cut under. Both are pinned
# here: the first as the contract, the second as the declared, measured
# exception, so the difference cannot be mistaken for a bug again.

PARITY_POLICY = dict(
    merge=True,
    cross_heading=True,
    # Hygiene off: this test is about BOUNDARIES. `min_chars`/`drop_link_dumps`
    # are a filter over chunks that inference does not apply at all, so leaving
    # them on would make a dropped junk chunk look like a boundary disagreement.
    min_chars=1,
    drop_link_dumps=False,
)


def _is_callout(attrs: str) -> bool:
    return "callout" in attrs


def _norm(chunks) -> list[str]:
    """Chunk texts, whitespace-normalised.

    `_split_sections` strips a heading line (so an indented `  ##` loses its
    indent) and `_pack_paragraphs` rejoins with a fixed blank line, while
    `_segment` reassembles the document's own separators byte-for-byte. Neither
    is wrong; both are the same boundaries.
    """
    return [" ".join(c.split()) for c in chunks if c.strip()]


def _training_chunks(doc: str, *, pack_across_protected: bool) -> list[str]:
    from stylebot import targets

    policy = targets.ChunkPolicy(
        merge_max_chars=BIG,
        div_is_prose=_is_callout,
        pack_across_protected=pack_across_protected,
        **PARITY_POLICY,
    )
    return _norm(t.text for t in targets._post_targets(doc, "parity.qmd", policy))


def _serving_chunks(doc: str) -> list[str]:
    _frontmatter, body = infer.split_frontmatter(doc)
    items, _seps = infer._segment(
        body, max_chars=BIG, inline_headings=True, div_is_prose=_is_callout
    )
    # "heading" is content under the cross-heading diet — a heading that failed
    # to group with anything keeps the label, and training cuts it as a target.
    return _norm(text for kind, text in items if kind in ("prose", "heading"))


def test_packing_parity_holds_when_the_policy_asks_for_it(doc: str):
    assert _training_chunks(doc, pack_across_protected=False) == _serving_chunks(doc)


def test_the_default_policy_packs_across_a_deleted_block(doc: str):
    """The declared exception, with its price on the record.

    Measured 2026-09-03 against the two corpus-stability arms: flipping the
    default takes section-bounded reproduction from 3,023/3,028 to 2,116 and
    cross-heading from 4,181/4,194 to 964 — 4,124 cells orphaned, 57% of the
    synthetic corpus, refillable only by a paid synth run and, for the frozen
    section-bounded arm, not at all. So the default stays, and this asserts the
    shape of what it costs rather than pretending the two agree.
    """
    packed = _training_chunks(doc, pack_across_protected=True)
    split = _training_chunks(doc, pack_across_protected=False)
    assert len(packed) < len(split)
    # Every parity chunk is a substring of some packed one: the difference is
    # only that packing joined them, never that it cut somewhere else.
    for chunk in split:
        assert any(chunk in whole for whole in packed), chunk
    # And the join really does span a protected block that inference will send
    # through untouched.
    before = "A paragraph before a protected block, which training packs across."
    after = "A paragraph after it:"
    assert any(before in c and after in c for c in packed)
    assert not any(before in c and after in c for c in split)
    assert not any(before in c and after in c for c in _serving_chunks(doc))


# --- the prose-bearing div policy (A-7) ------------------------------------

# `_is_callout` is declared once, above with the packing-parity helpers: the
# blog's real predicate matches declared classes, and a stub that says "callout"
# is anywhere in the attrs is enough to exercise the hook.


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
