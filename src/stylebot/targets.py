"""Targets — human-authored prose chunks, the assistant side of each pair.

Everything about turning blog posts into synthesis targets lives here: the
frontmatter/selector walk, prose-only extraction, section-aware splitting and
merge-packing, and the chunk-hygiene gates. All generic (no blog knowledge —
policy arrives via the caller's `selector`, markers and header names).

`stylebot.synth` re-exports the public names (`Target`, `ChunkPolicy`,
`iter_targets`, the chunk constants) — external callers keep importing via
`stylebot.synth`; this module is the implementation home.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from stylebot.lib import (
    editable_prose,
    gather_qmd_files,
    is_human_authored,
    read_w_frontmatter_text,
    split_paragraphs,
)

logger = logging.getLogger(__name__)

# Chunk hygiene. There is no voice transform to learn from a bare heading, a
# fenced code block, a link list, or a stub paragraph — so trim them.
MIN_CHUNK_CHARS = 80  # drop chunks shorter than this
MAX_CHUNK_CHARS = 8000  # drop chunks longer than this (truncating would corrupt the target)
MERGE_MAX_CHUNK_CHARS = 1500  # soft budget per packed passage in merge mode

# Link-LIST detection by *prose residual* (not link-char density): strip the
# `[text](url)` spans and measure how much real prose is left, as a fraction of
# the chunk. Density mis-fires on link-dense prose because URLs are long; the
# residual *fraction* is URL- and size-immune — a pure link list leaves ~0%
# whatever its length, while link-dense prose keeps most of its words.
LINK_LIST_MIN_PROSE_FRACTION = 0.2

_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class Target:
    """One paragraph-chunk (or merged passage) of human-authored prose.

    `context` is the section heading this passage sits under (verbatim, possibly
    empty) — prepended identically to both sides of the pair at synthesis time so
    the styler restyles the body conditioned on the heading. See
    `_plans/heading-context.md`.
    """

    text: str
    source: str  # path of the post it came from (relative to blog-root if known)
    chunk_index: int
    chunk_total: int
    context: str = ""


@dataclass(frozen=True)
class ChunkPolicy:
    """The chunk-hygiene knobs, bundled.

    Internal shape: `iter_targets`' flat kwargs remain the public API (callers
    splat a CLI surface into it); this object is how they thread through
    `_post_targets` — the completion of what was an ad-hoc kwargs dict.
    """

    min_chars: int = MIN_CHUNK_CHARS
    max_chars: int | None = MAX_CHUNK_CHARS
    prose_only: bool = True
    ignore_markers: Sequence[str] = ()
    drop_link_dumps: bool = True
    drop_list_items: bool = False
    merge: bool = False
    merge_max_chars: int = MERGE_MAX_CHUNK_CHARS
    stop_at_headers: Sequence[str] = ()
    heading_context: str = "none"


_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")


def _norm_header(text: str) -> str:
    """Normalise a header title for matching: drop ``#``s, trailing ``{...}`` attrs, case."""
    text = text.lstrip("#").strip()
    text = re.sub(r"\s*\{[^}]*\}\s*$", "", text)  # strip a trailing pandoc attribute block
    return text.lower()


def _truncate_at_headers(body: str, headers: Sequence[str]) -> str:
    """Drop everything from the first matching section header to end of document.

    Generic "cut a trailing section" mechanism: each entry in `headers` is
    matched (level-agnostic, case-insensitive) against the body's header lines.
    The caller supplies the section name as policy (e.g. ``"## Incoming"`` — a
    trailing link/quote dump with no authored signal).
    """
    if not headers:
        return body
    wanted = {_norm_header(h) for h in headers}
    out: list[str] = []
    for line in body.splitlines(keepends=True):
        m = _HEADER_RE.match(line)
        if m and _norm_header(m.group(1)) in wanted:
            break
        out.append(line)
    return "".join(out)


_LIST_ITEM_RE = re.compile(r"^\s{0,3}(?:[-*+]\s|\d+[.)]\s)")
# Markup/punctuation noise stripped when measuring prose residual.
_MARKUP_NOISE_RE = re.compile(r"[\s*_#>|`/\\\-]+")


def _prose_residual_len(chunk: str) -> int:
    """Length of the actual prose left after removing links and list/markup noise.

    A "link list" — a run of bare ``[Title](url)`` lines, bulleted or not — has
    almost nothing left once the link spans, list markers, and markup are gone;
    real prose (even link-dense prose) keeps its words. URL-immune by design.
    """
    t = _LINK_RE.sub(" ", chunk)
    t = re.sub(r"(?m)^\s*(?:[-*+]|\d+[.)])\s", " ", t)  # leading list markers
    t = _MARKUP_NOISE_RE.sub(" ", t)
    return len(t.strip())


def _is_link_list(chunk: str) -> bool:
    """True if a chunk is mostly markdown links with little authored prose."""
    s = chunk.strip()
    if not s:
        return True
    return _prose_residual_len(s) / len(s) < LINK_LIST_MIN_PROSE_FRACTION


def _is_table_block(chunk: str) -> bool:
    """True if every non-blank line is a pipe-table row."""
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    return bool(lines) and all(ln.lstrip().startswith("|") for ln in lines)


def _is_list_item(chunk: str) -> bool:
    """True if every non-blank line is a markdown list item."""
    lines = [ln for ln in chunk.splitlines() if ln.strip()]
    return bool(lines) and all(_LIST_ITEM_RE.match(ln) for ln in lines)


def _keep_chunk(
    chunk: str,
    *,
    min_chars: int,
    max_chars: int | None,
    ignore_markers: Sequence[str],
    drop_link_dumps: bool,
    drop_list_items: bool = False,
) -> bool:
    """Generic chunk-hygiene gate (no blog knowledge)."""
    s = chunk.strip()
    if len(s) < min_chars:
        return False
    if max_chars is not None and len(s) > max_chars:
        return False
    if any(marker in chunk for marker in ignore_markers):
        return False
    if _is_table_block(s):
        return False
    if drop_list_items and _is_list_item(s):
        return False
    if drop_link_dumps and _is_link_list(s):
        return False
    return True


def _load_meta_body(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    try:
        meta, body = read_w_frontmatter_text(text)
    except Exception:
        # Garbage frontmatter raises out of the YAML parser (see test_lib).
        # Empty meta means the selector rejects the post — same outcome as
        # before, but no longer silent.
        logger.warning("unparseable frontmatter in %s — treated as unselected", path)
        return {}, text
    return (meta or {}), body


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a body into ``(heading, section_body)`` pairs.

    A header line (`_HEADER_RE`) starts a new section and is itself **excluded**
    from the body (headers carry no paragraph voice and must never pack with
    prose) — instead it becomes the `heading` of the section that *follows* it
    (the immediate heading the section sits under). The pre-first-header preamble
    has an empty heading. Run this *after* `editable_prose` so code-comment ``#``
    lines — already stripped as protected blocks — can't be mistaken for headers.
    """
    sections: list[tuple[str, str]] = []
    heading = ""
    cur: list[str] = []
    for line in text.splitlines(keepends=True):
        if _HEADER_RE.match(line):
            if cur:
                sections.append((heading, "".join(cur)))
                cur = []
            heading = line.strip()  # frames the next section
        else:
            cur.append(line)
    if cur:
        sections.append((heading, "".join(cur)))
    return sections


def _pack_paragraphs(paras: Sequence[str], *, merge_max_chars: int) -> list[str]:
    """Greedily pack consecutive paragraphs into blocks up to a soft budget.

    Paragraphs are joined with a blank line. A paragraph already larger than the
    budget is emitted alone (kept whole — splitting mid-paragraph would corrupt
    the target). NB: not `qmd_core.chunk_editable_preserve_newlines`, which is
    char-budget chunking of one span — neither paragraph- nor section-aware.
    """
    blocks: list[str] = []
    buf: list[str] = []
    blen = 0
    for p in paras:
        add = len(p) + (2 if buf else 0)  # "\n\n" join cost
        if buf and blen + add > merge_max_chars:
            blocks.append("\n\n".join(buf))
            buf, blen = [p], len(p)
        else:
            buf.append(p)
            blen += add
    if buf:
        blocks.append("\n\n".join(buf))
    return blocks


def _post_targets(path: Path, source: str, policy: ChunkPolicy) -> list[Target]:
    _, body = _load_meta_body(path)
    # Cut trailing dump sections (e.g. "## Incoming") before anything else.
    body = _truncate_at_headers(body, policy.stop_at_headers)
    # Prose-only: drop protected blocks (code/math/:::divs/blockquotes) before
    # splitting, so they never become targets and chunk boundaries match what
    # the blog's own edit pipeline sees.
    text = editable_prose(body) if policy.prose_only else body
    want_context = policy.heading_context == "immediate"
    gate_kw = dict(
        ignore_markers=policy.ignore_markers,
        drop_link_dumps=policy.drop_link_dumps,
        drop_list_items=policy.drop_list_items,
    )

    # Both modes iterate header-delimited sections (so the section heading is
    # available as context, and so a passage never packs across a header). The
    # heading is attached to each emitted chunk when heading_context is on.
    chunks: list[tuple[str, str]] = []  # (text, context)
    for heading, section in _split_sections(text):
        ctx = heading if want_context else ""
        if not policy.merge:
            for c in split_paragraphs(section):
                if _keep_chunk(c, min_chars=policy.min_chars, max_chars=policy.max_chars, **gate_kw):
                    chunks.append((c, ctx))
        else:
            # Keep prose paragraphs (no min/max floor yet — short prose survives
            # to be rescued; junk dropped pre-pack), greedily pack to the budget,
            # then gate the packed block on min/max/link-list.
            paras = [
                p
                for p in split_paragraphs(section)
                if _keep_chunk(p, min_chars=1, max_chars=None, **gate_kw)
            ]
            for block in _pack_paragraphs(paras, merge_max_chars=policy.merge_max_chars):
                if _keep_chunk(
                    block, min_chars=policy.min_chars, max_chars=policy.max_chars, **gate_kw
                ):
                    chunks.append((block, ctx))

    total = len(chunks)
    return [
        Target(text=t, source=source, chunk_index=i, chunk_total=total, context=c)
        for i, (t, c) in enumerate(chunks)
    ]


def _source_label(path: Path, blog_root: Path | None) -> str:
    if blog_root is not None:
        try:
            return str(path.resolve().relative_to(blog_root.resolve()))
        except ValueError:
            pass
    return str(path)


def iter_targets(
    *,
    files: Sequence[Path | str] | None = None,
    blog_root: Path | str | None = None,
    selector: Callable[[dict], bool] = is_human_authored,
    glob: str = "**/*.qmd",
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int | None = MAX_CHUNK_CHARS,
    prose_only: bool = True,
    ignore_markers: Sequence[str] = (),
    drop_link_dumps: bool = True,
    drop_list_items: bool = False,
    merge: bool = False,
    merge_max_chars: int = MERGE_MAX_CHUNK_CHARS,
    stop_at_headers: Sequence[str] = (),
    heading_context: str = "none",
    sort_key: Callable[[Target], object] | None = None,
) -> list[Target]:
    """Collect prose chunks to use as synthesis targets.

    Two mutually exclusive input modes (OVERVIEW "Selection is a user-supplied
    policy"):

    - **`files`** — a pre-selected list the caller already filtered/ordered. The
      `selector` is **not** applied (the caller owns selection); every readable
      file is chunked.
    - **`blog_root` + `selector`** — stylebot walks `blog_root` for `glob`,
      reads each post's frontmatter, and keeps those for which `selector(meta)`
      is true. `selector` defaults to the bundled `is_human_authored`
      (`automation: 0`); pass your own to retarget.

    Chunk hygiene (all generic — no blog knowledge; policy stays in `selector`
    and the caller's marker choice):

    - `prose_only` (default True): drop protected blocks — fenced code,
      ``$$math$$``, ``:::`` divs/callouts, blockquotes — via
      `stylebot.lib.editable_prose` before splitting, so they never become
      targets (STYLE_SYSTEM preserves them verbatim anyway).
    - `min_chars` / `max_chars`: drop chunks outside the range (oversized chunks
      are dropped, never truncated — truncating corrupts the target).
    - `ignore_markers`: drop any chunk containing one of these literal strings
      (e.g. a stub marker like ``"🚧TODO🚧"`` the caller maintains).
    - `drop_link_dumps`: drop chunks that are mostly markdown links with little
      authored prose (measured by prose residual, not link density).
    - `drop_list_items`: drop chunks that are entirely markdown list items.
    - `merge`: pack consecutive prose paragraphs *within a section* into
      multi-paragraph passages up to `merge_max_chars` (never crossing a header,
      which keeps passages topically tight and self-limits length). Rescues short
      paragraphs that the `min_chars` floor would otherwise discard; `min_chars`
      then floors the *packed* block. A single paragraph over the budget is kept
      whole; `max_chars` still hard-drops anything oversized.
    - `stop_at_headers`: truncate each post body at the first matching section
      header (level-agnostic, case-insensitive), dropping everything after it —
      e.g. a trailing ``"## Incoming"`` dump of quotes/links with no authored
      signal.
    - `heading_context` (``"none"`` | ``"immediate"``): when ``"immediate"``,
      populate ``Target.context`` with the section heading each chunk sits under,
      so `synthesize_pairs` can prepend it verbatim to both sides of the pair.
      ``"none"`` (default) leaves context empty (unchanged behaviour).

    `sort_key` orders the resulting chunks. Returns a flat list of `Target`s.
    """
    if files is not None and blog_root is not None:
        raise ValueError("pass either `files` (pre-selected) or `blog_root` (walk + selector), not both")
    if files is None and blog_root is None:
        raise ValueError("provide `files` (pre-selected list) or `blog_root` (walk)")

    targets: list[Target] = []
    root = Path(blog_root) if blog_root is not None else None
    policy = ChunkPolicy(
        min_chars=min_chars,
        max_chars=max_chars,
        prose_only=prose_only,
        ignore_markers=ignore_markers,
        drop_link_dumps=drop_link_dumps,
        drop_list_items=drop_list_items,
        merge=merge,
        merge_max_chars=merge_max_chars,
        stop_at_headers=stop_at_headers,
        heading_context=heading_context,
    )

    if files is not None:
        for f in files:
            path = Path(f)
            if not path.is_file():
                continue
            targets.extend(_post_targets(path, _source_label(path, root), policy))
    else:
        for path in gather_qmd_files([], base=root, default_glob=glob):
            meta, _ = _load_meta_body(path)
            if not selector(meta):
                continue
            targets.extend(_post_targets(path, _source_label(path, root), policy))

    if sort_key is not None:
        targets.sort(key=sort_key)
    return targets
