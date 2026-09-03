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

**Raw HTML is the same shape of policy, and nothing in stylebot turns it on.**
``<script>``/``<style>`` contents are ground 2 (not prose — they are JavaScript
or CSS, and line structure is load-bearing), but only a caller that *reflows
whole documents* needs the protection. stylebot cuts targets and never touches
raw HTML, so ``raw_html`` defaults to False and no production call site here
passes True.

It is still not dead code, and the reason changed on 2026-09-03. It used to be
that ``raw_html`` let this implementation express a *sibling's* policy so a
conformance fixture could record both honestly. There is no sibling any more:
``livingthing.qmd_core.segment_for_edit`` is now a thin call to
`protected_spans` here, under the configuration ``block_only=False,
raw_html=True, html_comments=True`` (stylebot's base install was always
dependency-light, so the packaging line that forced a second implementation was
in the wrong place — stylebot `_plans/decisions.md` D28/D30, superseded). What
survives is D29, and it is the load-bearing half: the differences between the
two policies are real, they derive from what each caller *does with* a protected
span — livingthing reassembles them, stylebot deletes them — and they are
expressed as keyword arguments rather than as forked code or a forced
compromise. So these parameters have exactly one production caller each, in the
other repo, and changing what they do moves the blog's editing path.

Generic markdown tooling only: no frontmatter, no blog knowledge. THIS module is
now the single block-level segmenter across both repos, which makes its own
tests the only thing pinning it: ``tests/test_segment.py`` and
``tests/test_markdown.py`` for behaviour, ``tests/test_protection_parity.py``
for the train/serve contract, and ``tests/fixtures/segmenter_conformance.{qmd,json}``
for the two named configurations — plus, blog-side, ``livingthing``'s
``tests/test_corpus_stability.py``, which is the only thing that can see a change
here re-cutting the corpus (it needs the corpus, which never enters this repo).
The conformance fixture now asserts stylebot against its
own output rather than across an implementation boundary; it earns its place as
the written-down meaning of the configurations, and its regeneration hatch
(``STYLEBOT_REGEN_SEGMENTS=1``) is how a deliberate change is recorded next to
its reason.
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
#: `HEADING` is deliberately absent: a heading is protected *as a line* but
#: rides along as chunk context (or, under the cross-heading diet, as chunk
#: content), so callers treat it separately. `RAW_HTML` is absent because it is
#: opt-in — use `protected_kinds` when the option is not fixed at the call site.
#:
#: `HTML_COMMENT` is absent for the same reason as `RAW_HTML` and by the same
#: mechanism, but on a different argument. An HTML comment is nobody's prose — it
#: renders as nothing, and it routinely holds words that are not the author's at
#: all (tool directives like ``<!-- vale off -->``, notes-to-self, imported
#: reviewer comments) — so a *reflower* should always protect one. But a comment
#: may open mid-line, and protecting it there **splits the prose around it**: a
#: caller that deletes protected spans to cut training targets loses the target,
#: where a caller that reassembles them loses nothing. Measured against the blog
#: corpus at 1aa290e463, protecting comments unconditionally took section-bounded
#: reproducibility from 3023/3028 to 3020/3028, all three losses prose interrupted
#: by an inline ``<!-- vale … -->`` marker. So this is `block_only`'s trade in a
#: second place, and it gets `block_only`'s answer: a keyword, and two recorded
#: configurations, rather than one side's policy imposed on the other.
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

# Protected-block patterns.
#
# The opener/closer alternate independently rather than back-referencing one
# another. A backreference form (``(```+|~~~+)…\1``) is NOT equivalent: it lets
# a longer outer fence swallow the inner fences of a post that *documents*
# nested fences (notebook/blogdown.qmd), which measurably moves the corpus:
# 7 section-bounded and 19 cross-heading target bodies change. Mixing ``` and
# ~~~ across a
# block is malformed markdown either way; matching it loosely is the safer
# failure, since over-protection only costs a target.
#
# The CLOSER takes ``{3,}`` where the opener takes a bare three, and that
# asymmetry is not cosmetic: the opener runs ``.*?$`` after its three, so a
# ```` ```` ```` line already opens, but with a bare-three closer it could never
# *close*. The consequence was live and silent — in ``notebook/quarto.qmd`` one
# fence span ran 999 chars past its own closer and swallowed three whole
# ``.callout-tip`` divs, and in ``notebook/learning_with_ai.qmd`` a ```` ````
# ```` block was not protected at all. Still not a backreference, so the nested
# case above is unaffected: a ``` closer still ends a ```` opener, exactly as
# before.
#
# **An UNCLOSED fence runs to end of document** — the second alternative below,
# and the reason it is here rather than in a consumer. CommonMark says exactly
# this ("if the end of the containing block is reached ... the code block
# contains all of the lines after the opening fence"), so it is the correct
# reading and not a defensive hack. It used to live only in
# `infer._blocks_with_separators`, as a line-wise state machine running
# alongside this pattern: inference swallowed a runaway fence and
# `editable_prose` — hence every training target — did not. That is the same
# shape of train/serve split this module exists to end, so the state machine
# moved here and its consumer deleted it.
#
# The closed alternative is tried first and its `.*?` is lazy, so a document
# with a closer is matched exactly as before; only a document with none reaches
# the greedy run-to-EOF branch.
CODE_FENCE_RE = re.compile(
    r"^[ \t]*(?:```|~~~).*?$"
    r"(?:.*?^[ \t]*(?:`{3,}|~{3,})[ \t]*$(?:\n|\Z)|.*\Z)",
    re.MULTILINE | re.DOTALL,
)
MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$[ \t]*(?:\{[^}\n]*\})?", re.DOTALL)
BLOCKQUOTE_RE = re.compile(r"^(?:[ \t]{0,3}>[^\n]*\n?)+", re.MULTILINE)

#: ``<script>``/``<style>`` blocks — opener at line start, closer ending a line.
#: These two are the elements whose content HTML defines as *raw text* rather
#: than markup, so reflowing them is never right: join the line after a ``//``
#: comment onto it and that line is now commented out. Not hypothetical — the
#: blog shipped an ``Uncaught SyntaxError`` on its front page in 2026-05 when a
#: reflow collapsed a topic-map ``<script>``. Every other element's content *is*
#: markup, so prose inside a raw ``<div>`` should still be rewritable.
#:
#: DOTALL so a blank line inside the block cannot end the match; an unclosed
#: opener yields no span, exactly as `CODE_FENCE_RE` and `_div_spans` do. Ported
#: from ``livingthing.qmd_core`` with its numbered backreference made a named
#: one, so the pattern survives being composed into a larger alternation.
RAW_HTML_BLOCK_RE = re.compile(
    r"^[ \t]*<(?P<tag>script|style)\b[^>]*>.*?</(?P=tag)[ \t]*>[ \t]*$(?:\n|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)

#: An HTML comment, at any indent, spanning any number of lines. Non-greedy, so
#: two comments in one document are two spans rather than one that swallows the
#: prose between them.
#:
#: Deliberately *not* folded into `RAW_HTML_BLOCK_RE`: that one is opt-in and this
#: one is always on (see `PROTECTED_KINDS`), and a shared pattern could not express
#: the difference. The line anchoring differs too — a comment may legitimately open
#: mid-line, and a trailing ``<!-- vale on -->`` after a sentence must still be
#: protected.
#:
#: Measured 2026-08-22 in livingthing, which is what prompted this: a multi-line
#: comment handed to `format-sentences` had its opening line joined to its body and
#: the result split at sentence boundaries. The comment survived only because
#: nothing inside it happened to spell ``-->``.
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
    raw_html: bool = False,
    html_comments: bool = False,
) -> list[Span]:
    """Merged, sorted, non-overlapping protected spans of `text`.

    `block_only` restricts ``$$…$$`` to line-anchored display math; an inline
    ``$$k$$`` then stays in the prose the styler rewrites. `raw_html` adds
    ``<script>``/``<style>`` blocks (`RAW_HTML_BLOCK_RE`), which nothing in
    stylebot wants but a whole-document reflower does; `html_comments` adds
    ``<!--…-->`` (`HTML_COMMENT_RE`), on the same terms and for the reason
    recorded above `HTML_COMMENT_RE`. Overlapping spans of different kinds merge
    under the kind that starts first.

    Public on purpose, and since 2026-09-03 it has an out-of-package caller:
    `livingthing.qmd_core.segment_for_edit` is a thin call to this function
    rather than a second implementation of it. This IS the module's central
    question — what a rewrite must never be handed — so it is the entry point a
    consumer should call rather than re-port. The keyword arguments are what let
    one implementation serve two policies without forking, and the difference
    they express is genuine: livingthing **reassembles** protected spans, so
    protecting an inline ``$$k$$`` costs it nothing and buys safety, while
    stylebot **deletes** them, so protecting one joins the remnants and mangles
    the sentence (see `block_only`). Contrast `_find_div_blocks`, private
    because nothing outside wants divs alone.
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
