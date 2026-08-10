"""Markdown block primitives — the ONE segmentation every phase shares.

Before this module, segmentation was implemented five times across the package
and the copies had silently diverged: what Phase 2 excluded from *training*
targets was not what Phase 4 refused to send at *inference*. Everything that
needs to know "is this line prose?" now asks here.

**The protection principle.** A construct is protected — excluded from training
targets *and* from inference chunks — when rewriting it would change meaning
rather than voice, or when there is no voice to apply. Three grounds:

1. **Not the author's words.** Blockquotes. A style rewrite of quoted text is
   misattribution, however Dan it sounds.
2. **Not prose.** Fenced code, display math, figure and reference divs.
   Rewriting is corruption, not restyling.
3. **Structural markers.** Heading lines and ``:::`` fence lines carry
   level/class semantics the rewrite must not touch — *though what they delimit
   may be prose.*

**Corollary: a container is not protected merely for being a container.** Ground
3 protects the ``:::`` lines, not automatically their contents. Which div
classes bear prose is *policy* (``.callout-note`` is a Quarto convention, and
stylebot's blog/generic boundary keeps Quarto policy out of here), so callers
opt in through the ``div_is_prose`` hook. Its default — ``None``, every div
protected wholesale — is the conservative behaviour this package shipped with.

Generic markdown tooling only: no frontmatter, no blog knowledge.
SIBLING: ``livingthing/src/livingthing/qmd_core.py::segment_for_edit`` — keep
behaviour in sync; the segmenter tests in ``tests/test_segment.py`` and the
cross-language fixture in ``tests/test_marker_segmentation.py`` pin it.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# A span of `text`: (start, end, kind). `end` is exclusive.
Span = tuple[int, int, str]

PROSE = "prose"
HEADING = "heading"
FENCE = "fence"
MATH = "math"
DIV = "div"
BLOCKQUOTE = "blockquote"

#: Kinds a rewrite must never be handed. `HEADING` is deliberately absent: a
#: heading is protected *as a line* but rides along as chunk context (or, under
#: the cross-heading diet, as chunk content), so callers treat it separately.
PROTECTED_KINDS = frozenset({FENCE, MATH, DIV, BLOCKQUOTE})

#: CommonMark ATX heading: up to 3 *spaces* of indent, 1-6 hashes, then either
#: end-of-line (a bare `##` is a legal empty heading) or a space/tab and the
#: title. Anchored at both ends so `#hashtag` and `####### seven` are not
#: headings. This is the ONE heading spelling — `targets`, `capture` and `infer`
#: all route through `is_heading_line`.
ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(?P<title>.*?))?[ \t]*$")

#: An opening code fence: 3+ backticks or 3+ tildes at ≤3 spaces of indent.
FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")

# Protected-block patterns. `CODE_FENCE_RE` requires a closer, so an unclosed
# fence yields NO span — consumers that must swallow a runaway fence keep their
# own state machine (see `infer._blocks_with_separators`).
#
# The opener/closer alternate independently rather than back-referencing one
# another. A backreference form (``(```+|~~~+)…\1``) is NOT equivalent: it lets
# a longer outer fence swallow the inner fences of a post that *documents*
# nested fences (notebook/blogdown.qmd), which measurably moves the corpus:
# 7 section-bounded and 19 cross-heading target bodies change. Mixing ``` and
# ~~~ across a
# block is malformed markdown either way; matching it loosely is the safer
# failure, since over-protection only costs a target.
CODE_FENCE_RE = re.compile(
    r"^[ \t]*(?:```|~~~).*?$.*?^[ \t]*(?:```|~~~)[ \t]*$(?:\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$[ \t]*(?:\{[^}\n]*\})?", re.DOTALL)
BLOCKQUOTE_RE = re.compile(r"^(?:[ \t]{0,3}>[^\n]*\n?)+", re.MULTILINE)

#: Receives a div opener's attribute text (e.g. ``{.callout-note}``) and says
#: whether the div's *contents* are prose. Policy, supplied by the caller.
DivIsProse = Callable[[str], bool]


# ---------------------------------------------------------------------------
# Headings
# ---------------------------------------------------------------------------


def is_heading_line(line: str) -> bool:
    """True if `line` is an ATX heading line (a trailing newline is fine)."""
    return ATX_HEADING_RE.match(line) is not None


def heading_title(line: str) -> str | None:
    """The heading's title text, `""` for a bare `##`, `None` if not a heading."""
    m = ATX_HEADING_RE.match(line)
    if m is None:
        return None
    return m.group("title") or ""


# ---------------------------------------------------------------------------
# Protected spans
# ---------------------------------------------------------------------------


def _div_attrs(line: str) -> str:
    """The attribute text of a ``:::`` opener — everything past the colons."""
    return line.strip().lstrip(":").strip()


def _div_spans(
    lines: list[str],
    offsets: list[int],
    lo: int,
    hi: int,
    div_is_prose: DivIsProse | None,
) -> list[Span]:
    """Div spans within `lines[lo:hi]`, nesting-aware.

    Pandoc rule: a closing fence is a line that is exactly ``:::`` (whitespace
    allowed); a ``:::`` line with arguments opens a *nested* div. An unclosed
    opener produces no span at all — scanning simply continues past it.

    With `div_is_prose` returning true for an opener, only the opener and closer
    lines are protected (ground 3) and the interior is rescanned for nested
    divs, which stay protected on their own merits.
    """
    spans: list[Span] = []
    i = lo
    while i < hi:
        if lines[i].strip().startswith(":::"):
            depth = 1
            j = i + 1
            while j < hi and depth > 0:
                inner = lines[j].strip()
                if inner.startswith(":::"):
                    if inner == ":::":
                        depth -= 1
                    else:
                        depth += 1
                j += 1
            if depth == 0:
                if div_is_prose is not None and div_is_prose(_div_attrs(lines[i])):
                    spans.append((offsets[i], offsets[i + 1], DIV))
                    spans.extend(_div_spans(lines, offsets, i + 1, j - 1, div_is_prose))
                    spans.append((offsets[j - 1], offsets[j], DIV))
                else:
                    spans.append((offsets[i], offsets[j], DIV))
                i = j
                continue
        i += 1
    return spans


def _find_div_blocks(content: str, *, div_is_prose: DivIsProse | None = None) -> list[Span]:
    """All ``:::`` div/callout spans, nesting-aware (end includes the newline)."""
    lines = content.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)
    return _div_spans(lines, offsets, 0, len(lines), div_is_prose)


def _is_block_anchored(text: str, start: int, end: int) -> bool:
    """True if `text[start:end]` occupies whole lines (≤3 spaces in, blank out).

    The line-anchoring test that keeps a mid-sentence ``$$k$$`` out of the
    protected set: excising one mid-sentence joins the remnants
    (``high-$$d$$ behaviour`` → ``high-behaviour``) inside a *training target*.
    """
    line_start = text.rfind("\n", 0, start) + 1
    indent = text[line_start:start]
    if indent.strip(" ") or len(indent) > 3:
        return False
    line_end = text.find("\n", end)
    tail = text[end:] if line_end < 0 else text[end:line_end]
    return not tail.strip()


def protected_spans(
    text: str,
    *,
    block_only: bool = False,
    div_is_prose: DivIsProse | None = None,
) -> list[Span]:
    """Merged, sorted, non-overlapping protected spans of `text`.

    `block_only` restricts ``$$…$$`` to line-anchored display math; an inline
    ``$$k$$`` then stays in the prose the styler rewrites. Overlapping spans of
    different kinds merge under the kind that starts first.

    Public on purpose, though every caller today is in this module: this IS the
    module's central question — what a rewrite must never be handed — and it is
    the entry point a second implementation should consume rather than re-port.
    `livingthing.qmd_core` still carries its own copy of this logic and has
    already drifted from it (`~~~` fences, inline ``$$``); harmonising it means
    calling this. Contrast `_find_div_blocks`, private because nothing outside
    wants divs alone.
    """
    spans: list[Span] = []
    for m in CODE_FENCE_RE.finditer(text):
        spans.append((m.start(), m.end(), FENCE))
    for m in MATH_BLOCK_RE.finditer(text):
        if block_only and not _is_block_anchored(text, m.start(), m.end()):
            continue
        spans.append((m.start(), m.end(), MATH))
    for m in BLOCKQUOTE_RE.finditer(text):
        spans.append((m.start(), m.end(), BLOCKQUOTE))
    spans.extend(_find_div_blocks(text, div_is_prose=div_is_prose))
    spans.sort(key=lambda sp: (sp[0], sp[1]))

    merged: list[Span] = []
    for start, end, kind in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
        else:
            merged.append((start, end, kind))
    return merged


def segment_spans(text: str, *, div_is_prose: DivIsProse | None = None) -> list[tuple[str, bool]]:
    """Split `text` into ``(chunk, editable)`` spans, losslessly.

    Concatenating every chunk reproduces `text` exactly; `editable` is False for
    every protected span.

    Display math is **block-anchored** here (`block_only=True`), matching what
    `line_kinds` — and therefore inference — can see. Excising a mid-sentence
    ``$$k$$`` and joining the remnants (``high-$$d$$ behaviour`` ->
    ``high- behaviour``) produced truncated sentences inside real training
    targets; leaving inline display math in the prose the styler rewrites is the
    lesser evil, and consistent with raw inline markdown already being
    in-distribution at ~16% of pairs.
    """
    out: list[tuple[str, bool]] = []
    i = 0
    for start, end, _kind in protected_spans(text, block_only=True, div_is_prose=div_is_prose):
        if i < start:
            out.append((text[i:start], True))
        out.append((text[start:end], False))
        i = end
    if i < len(text):
        out.append((text[i:], True))
    return out


def line_kinds(
    text: str,
    *,
    headings: bool = True,
    div_is_prose: DivIsProse | None = None,
) -> list[str | None]:
    """One kind label per line of ``text.split("\\n")`` — `None` means prose.

    A per-line mask rather than char spans, so a consumer can gain protection
    without disturbing a byte-exact reassembly proof: `infer._segment` applies
    this exactly the way it already applies its fence state machine. Only
    line-anchored (``block_only``) protection is reported, since a partially
    protected line has no single answer.
    """
    lines = text.split("\n")
    kinds: list[str | None] = [None] * len(lines)
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1

    for span_start, span_end, kind in protected_spans(
        text, block_only=True, div_is_prose=div_is_prose
    ):
        for i, line in enumerate(lines):
            if starts[i] >= span_end:
                break
            # The line owns [start, start + len + 1) — its own newline included,
            # so a span that stops at a line boundary does not bleed onward.
            if span_start < starts[i] + len(line) + 1 and kinds[i] is None:
                kinds[i] = kind
    if headings:
        for i, line in enumerate(lines):
            if kinds[i] is None and is_heading_line(line):
                kinds[i] = HEADING
    return kinds
