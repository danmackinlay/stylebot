"""`infer._segment` against the segmenter it reads, over one full document.

`markdown.segment_spans` (what `lib.segment_for_edit` and the target cutter
see) and `markdown.line_kinds` (what `infer._segment` applies) are two
derivations inside one module; a construct one protects and the other sends to
the styler fails silently — a styler rewording somebody else's quoted words
looks like a successful rewrite. Both are reduced to protected LINE indices
over `fixtures/hard/segmenter_conformance.qmd` and compared, under the default
div policy and the callout one. Headings are asserted separately: protected on
both sides, by different machinery. Then `_segment`'s own contract — byte-exact
reassembly under any budget — and the packing-parity pair against
`targets._post_targets` (D27), which nothing else covers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stylebot import infer
from stylebot import markdown as md
from stylebot.lib import segment_for_edit

FIXTURE = Path(__file__).parent / "fixtures" / "hard" / "segmenter_conformance.qmd"
BIG = 1_000_000  # never let the chunk budget split anything in these tests


def _is_callout(attrs: str) -> bool:
    # The blog's real predicate matches declared classes; "callout" anywhere in
    # the attrs is enough to exercise the hook.
    return "callout" in attrs


POLICIES = [pytest.param(None, id="default"), pytest.param(_is_callout, id="callout")]


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


@pytest.mark.parametrize("policy", POLICIES)
def test_training_and_inference_protect_the_same_lines(doc: str, policy):
    train = training_protected_lines(doc, div_is_prose=policy)
    serve = inference_protected_lines(doc, div_is_prose=policy)
    lines = doc.split("\n")
    assert sorted(train - serve) == [], (
        "training drops these lines but inference would send them: "
        + repr([lines[i] for i in sorted(train - serve)])
    )
    assert sorted(serve - train) == [], (
        "inference refuses these lines but they can become training targets: "
        + repr([lines[i] for i in sorted(serve - train)])
    )


@pytest.mark.parametrize("policy", POLICIES)
def test_headings_are_protected_at_inference(doc: str, policy):
    # Under the callout policy this includes the `##` inside a callout, which
    # becomes a section boundary once the callout's contents are prose.
    heading_lines = {
        i for i, line in enumerate(doc.split("\n")) if md.is_heading_line(line)
    } - training_protected_lines(doc, div_is_prose=policy)
    assert heading_lines
    sent = {
        i
        for kind, s, e in _segment_spans(doc, div_is_prose=policy)
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


def test_fixture_exercises_every_protected_kind(doc: str):
    kinds = {kind for kind, _s, _e in _segment_spans(doc)}
    assert kinds >= md.PROTECTED_KINDS, f"fixture misses {md.PROTECTED_KINDS - kinds}"
    assert "heading" in kinds


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

    Flipping the default orphans most of the synthetic corpus, refillable only
    by a paid synth run (the table under D27's alternative B,
    `decisions-archive.md`). So the default stays, and this asserts the shape
    of what it costs rather than pretending the two agree.
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


def test_segment_protects_a_whole_line_comment_and_reassembles():
    """D93: with `html_comments`, a comment on its own line is never handed to
    the styler; an inline one rides inside its prose paragraph; off by default."""
    body = "First paragraph of prose.\n\n<!-- vale off -->\n\nSecond, with <!-- sem --> inline."
    assert md.HTML_COMMENT not in [k for k, _ in infer._segment(body, max_chars=10_000)[0]]
    items, seps = infer._segment(body, max_chars=10_000, html_comments=True)
    kinds = [k for k, _ in items]
    assert md.HTML_COMMENT in kinds
    assert [t for k, t in items if k == md.HTML_COMMENT] == ["<!-- vale off -->"]
    rebuilt = items[0][1] + "".join(s + t for s, (_, t) in zip(seps, items[1:], strict=True))
    assert rebuilt == body
