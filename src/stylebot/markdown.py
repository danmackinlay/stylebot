"""Markdown block primitives — the one segmentation every phase shares (D26).

Everything asking "is this line prose?" asks here. Generic only: no
frontmatter, no blog.

The protection principle: a construct is protected — excluded from training
targets and from inference chunks — when rewriting it would change meaning
rather than voice, or when there is no voice to apply. Three grounds: not the
author's words (blockquotes); not prose (fenced code, display math, figure
and reference divs); structural markers (heading lines and ``:::`` fence
lines), though what those delimit may be prose. So a container is not
protected merely for being one: which div classes bear prose is policy,
through ``div_is_prose`` (default ``None`` protects every div); ``raw_html``
and ``html_comments`` are the same shape of policy, off by default (D29).

Pinned by `tests/test_markdown.py`, `test_infer_segment.py`,
`test_segment_hard.py` and `tests/fixtures/segmenter_conformance.json`.
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
#: `HEADING` is absent because a heading rides along as chunk context or
#: content; `RAW_HTML` and `HTML_COMMENT` because they are opt-in (D29). Use
#: `protected_kinds` when the option is not fixed at the call site.
PROTECTED_KINDS = frozenset({FENCE, MATH, DIV, BLOCKQUOTE})


def protected_kinds(*, raw_html: bool = False, html_comments: bool = False) -> frozenset[str]:
    """The kinds `protected_spans` can emit under a given configuration."""
    kinds = PROTECTED_KINDS
    if raw_html:
        kinds = kinds | {RAW_HTML}
    if html_comments:
        kinds = kinds | {HTML_COMMENT}
    return kinds


#: CommonMark ATX heading: up to 3 spaces of indent, 1-6 hashes, then
#: end-of-line or a space/tab and the title; anchored at both ends. The one
#: heading spelling — everything routes through `is_heading_line`.
ATX_HEADING_RE = re.compile(r"^ {0,3}#{1,6}(?:[ \t]+(?P<title>.*?))?[ \t]*$")

# Protected-block patterns (D27, D69). The fence closer back-references its
# opener's run — same character, at least as long — and an unclosed fence
# runs to end of document here rather than in a consumer, so inference and
# `editable_prose` cannot disagree. That alternative always matches, so the
# opener's run never backtracks shorter and takes an inner closer.
CODE_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>(?P<fc>[`~])(?P=fc){2,}).*?$"
    r"(?:.*?^[ \t]*(?P=fence)(?P=fc)*[ \t]*$(?:\n|\Z)|.*\Z)",
    re.MULTILINE | re.DOTALL,
)
MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$[ \t]*(?:\{[^}\n]*\})?", re.DOTALL)
BLOCKQUOTE_RE = re.compile(r"^(?:[ \t]{0,3}>[^\n]*\n?)+", re.MULTILINE)

#: ``<script>``/``<style>`` blocks, the two elements whose content HTML defines
#: as raw text; prose inside any other raw element stays rewritable. DOTALL so a
#: blank line cannot end the match; an unclosed opener yields no span (D76).
RAW_HTML_BLOCK_RE = re.compile(
    r"^[ \t]*<(?P<tag>script|style)\b[^>]*>.*?</(?P=tag)[ \t]*>[ \t]*$(?:\n|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

#: An HTML comment, at any indent, any number of lines, non-greedy. Apart from
#: `RAW_HTML_BLOCK_RE` because a comment may open mid-line.
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
    """Div spans within `lines[lo:hi]`, nesting-aware: a bare ``:::`` closes, a
    ``:::`` with arguments opens a nested div, an unclosed opener yields no span
    (D76). Where `div_is_prose` is true only the fence lines are protected and
    the interior is rescanned."""
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
    # Split as `line_kinds` does: on `\n` only, so a lone `\r` is not a line break.
    lines = [line + "\n" for line in content.split("\n")]
    lines[-1] = lines[-1][:-1]
    if not lines[-1]:
        lines.pop()
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)
    offsets.append(pos)
    return _div_spans(lines, offsets, 0, len(lines), div_is_prose)


def _is_block_anchored(text: str, start: int, end: int) -> bool:
    """True if `text[start:end]` occupies whole lines (≤3 spaces in, blank
    out): the test that keeps a mid-sentence ``$$k$$`` out of the protected set."""
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
    """Merged, sorted, non-overlapping protected spans of `text` (D29).

    `block_only` restricts ``$$…$$`` to line-anchored display math; `raw_html`
    adds ``<script>``/``<style>`` blocks and `html_comments` adds ``<!--…-->``.
    Overlapping spans of different kinds merge under the kind that starts first.
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
    """Split `text` into ``(chunk, editable)`` spans, losslessly; `editable` is
    False for every protected span. Display math is block-anchored here
    (`block_only=True`), matching what `line_kinds` and inference see (D27)."""
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
    A per-line mask, so a consumer keeps its byte-exact reassembly; only
    line-anchored (``block_only``) protection is reported."""
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
