"""Markdown block primitives — the ONE segmentation every phase shares.

Segmentation was implemented five times here and the copies diverged; everything
asking "is this line prose?" asks here. Generic only: no frontmatter, no blog.

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

**Corollary: a container is not protected merely for being a container.**
Ground 3 protects the ``:::`` lines, not automatically their contents. Which
div classes bear prose is *policy* (``.callout-note`` is a Quarto convention,
and stylebot's blog/generic boundary keeps Quarto policy out of here), so
callers opt in through the ``div_is_prose`` hook; its default ``None``
protects every div wholesale. ``raw_html`` and ``html_comments`` are the same
shape of policy, off by default because nothing here reflows whole documents —
their one production caller is `livingthing.qmd_core.segment_for_edit`, so
changing them moves the blog's editing path (D29).

Pinned by `tests/test_markdown.py`, `test_infer_segment.py`, `test_segment_hard.py`
and `tests/fixtures/segmenter_conformance.json` (``STYLEBOT_REGEN_SEGMENTS=1``).
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
RAW_HTML = "raw_html"
HTML_COMMENT = "html_comment"

#: Kinds a rewrite must never be handed, in the default configuration.
#: `HEADING` is absent because a heading is protected *as a line* but rides
#: along as chunk context or content, so callers treat it separately.
#: `RAW_HTML` and `HTML_COMMENT` are absent because they are opt-in — a
#: reflower wants both, a target-cutter loses prose to a mid-line comment, and
#: that split is `block_only`'s trade in a second place, with `block_only`'s
#: answer (D29). Use `protected_kinds` when the option is not fixed at the call
#: site.
PROTECTED_KINDS = frozenset({FENCE, MATH, DIV, BLOCKQUOTE})


def protected_kinds(*, raw_html: bool = False, html_comments: bool = False) -> frozenset[str]:
    """The kinds `protected_spans` can emit under a given configuration."""
    kinds = PROTECTED_KINDS
    if raw_html:
        kinds = kinds | {RAW_HTML}
    if html_comments:
        kinds = kinds | {HTML_COMMENT}
    return kinds


#: CommonMark ATX heading: up to 3 *spaces* of indent, 1-6 hashes, then either
#: end-of-line (a bare `##` is a legal empty heading) or a space/tab and the
#: title. Anchored at both ends so `#hashtag` and `####### seven` are not
#: headings. This is the ONE heading spelling — `targets`, `capture` and `infer`
#: all route through `is_heading_line`.
ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(?P<title>.*?))?[ \t]*$")

# Protected-block patterns. Three shapes here are load-bearing and were each
# arrived at against the live corpus (D27, D69, findings.md "the fence closer"):
# the opener captures its whole run and the closer back-references it — same
# character, at least as long, CommonMark's rule — so a five-backtick fence
# quoting a three-backtick one is not closed by the inner fence, while a longer
# line still closes a shorter opener (W2); and an **unclosed fence runs to end
# of document** (the second alternative, CommonMark's own reading) here rather
# than in a consumer, so inference and `editable_prose` cannot disagree about a
# runaway fence. That alternative always matches, so the opener's run never
# backtracks shorter and takes an inner closer instead.
CODE_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>(?P<fc>[`~])(?P=fc){2,}).*?$"
    r"(?:.*?^[ \t]*(?P=fence)(?P=fc)*[ \t]*$(?:\n|\Z)|.*\Z)",
    re.MULTILINE | re.DOTALL,
)
MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$[ \t]*(?:\{[^}\n]*\})?", re.DOTALL)
BLOCKQUOTE_RE = re.compile(r"^(?:[ \t]{0,3}>[^\n]*\n?)+", re.MULTILINE)

#: ``<script>``/``<style>`` blocks — opener at line start, closer ending a line.
#: These two are the elements whose content HTML defines as *raw text* rather
#: than markup, so reflowing them is never right (join the line after a ``//``
#: comment onto it and the line is commented out). Every other element's content
#: *is* markup, so prose inside a raw ``<div>`` stays rewritable. DOTALL so a
#: blank line cannot end the match; an unclosed opener yields no span, as an
#: unclosed `:::` does (D76) — unlike a code fence, which runs to end of document.
RAW_HTML_BLOCK_RE = re.compile(
    r"^[ \t]*<(?P<tag>script|style)\b[^>]*>.*?</(?P=tag)[ \t]*>[ \t]*$(?:\n|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

#: An HTML comment, at any indent, spanning any number of lines. Non-greedy, so
#: two comments are two spans rather than one that swallows the prose between
#: them. Deliberately not folded into `RAW_HTML_BLOCK_RE`: the line anchoring
#: differs, since a comment may open mid-line and a trailing ``<!-- vale on -->``
#: must still be protected.
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

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


def is_div_fence_line(line: str) -> bool:
    """True if `line` opens or closes a ``:::`` div (any indent, trailing newline fine)."""
    return line.strip().startswith(":::")


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
    opener produces no span at all — deliberately, and not pandoc's reading (D76).

    With `div_is_prose` returning true for an opener, only the opener and closer
    lines are protected (ground 3) and the interior is rescanned for nested
    divs, which stay protected on their own merits.
    """
    spans: list[Span] = []
    i = lo
    while i < hi:
        if is_div_fence_line(lines[i]):
            depth = 1
            j = i + 1
            while j < hi and depth > 0:
                inner = lines[j].strip()
                if is_div_fence_line(inner):
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
    raw_html: bool = False,
    html_comments: bool = False,
) -> list[Span]:
    """Merged, sorted, non-overlapping protected spans of `text`.

    `block_only` restricts ``$$…$$`` to line-anchored display math; `raw_html`
    adds ``<script>``/``<style>`` blocks and `html_comments` adds ``<!--…-->``,
    both on the terms recorded above their patterns. Overlapping spans of
    different kinds merge under the kind that starts first.

    Public on purpose: this IS the module's central question, and
    `livingthing.qmd_core.segment_for_edit` is a thin call to it rather than a
    second implementation — the keywords are what let one implementation serve
    two policies without forking (D29).
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
    if html_comments:
        for m in HTML_COMMENT_RE.finditer(text):
            spans.append((m.start(), m.end(), HTML_COMMENT))
    if raw_html:
        for m in RAW_HTML_BLOCK_RE.finditer(text):
            spans.append((m.start(), m.end(), RAW_HTML))
    spans.extend(_find_div_blocks(text, div_is_prose=div_is_prose))
    spans.sort(key=lambda sp: (sp[0], sp[1]))

    merged: list[Span] = []
    for start, end, kind in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
        else:
            merged.append((start, end, kind))
    return merged


def segment_spans(
    text: str,
    *,
    div_is_prose: DivIsProse | None = None,
    raw_html: bool = False,
) -> list[tuple[str, bool]]:
    """Split `text` into ``(chunk, editable)`` spans, losslessly.

    Concatenating every chunk reproduces `text` exactly; `editable` is False for
    every protected span. `raw_html` is passed to `protected_spans`.

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
    for start, end, _kind in protected_spans(
        text, block_only=True, div_is_prose=div_is_prose, raw_html=raw_html
    ):
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
    raw_html: bool = False,
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
        text, block_only=True, div_is_prose=div_is_prose, raw_html=raw_html
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
