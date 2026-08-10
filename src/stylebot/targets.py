"""Targets — human-authored prose chunks, the assistant side of each pair.

Turning posts into synthesis targets: the frontmatter/selector walk, prose-only
extraction, section-aware splitting and merge-packing, and the chunk-hygiene
gates. All generic — policy arrives via the caller's `selector`, markers and
header names.

`stylebot.synth` re-exports the public names; this module is the implementation
home.
"""

from __future__ import annotations

import hashlib
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
from stylebot.splits import role_of, unit_from_digest

logger = logging.getLogger(__name__)

# Chunk hygiene. There is no voice transform to learn from a bare heading, a
# fenced code block, a link list, or a stub paragraph — so trim them.
MIN_CHUNK_CHARS = 80  # drop chunks shorter than this
MAX_CHUNK_CHARS = 8000  # drop chunks longer than this (truncating would corrupt the target)
MERGE_MAX_CHUNK_CHARS = 1500  # soft budget per packed passage in merge mode

# Link-LIST detection by *prose residual*, not link-char density: strip the
# `[text](url)` spans and measure the real prose left as a fraction of the
# chunk. Density mis-fires on link-dense prose because URLs are long; the
# residual fraction is URL- and size-immune — a pure link list leaves ~0%
# whatever its length, while link-dense prose keeps most of its words.
LINK_LIST_MIN_PROSE_FRACTION = 0.2

_LINK_RE = re.compile(r"\[[^\]]*\]\([^)]*\)")


@dataclass(frozen=True)
class Target:
    """One paragraph-chunk (or merged passage) of human-authored prose.

    `context` is the section heading this passage sits under (verbatim, possibly
    empty), prepended identically to both sides of the pair at synthesis time.
    See `_plans/decisions.md` D14.
    """

    text: str
    source: str  # path of the post it came from (relative to blog-root if known)
    chunk_index: int
    chunk_total: int
    context: str = ""
    # Chunk-scale facet, stamped into pair meta so a corpus mixing chunk diets
    # stays ablatable by scale (`_plans/v2-longer-rewrites.md`): the merge budget
    # (None = unmerged single paragraphs), and how the chunk was cut — None =
    # within one section with the heading riding along as `context`,
    # "cross-heading" = heading lines packed in as content, spanning sections.
    merge_max_chars: int | None = None
    chunk_mode: str | None = None


@dataclass(frozen=True)
class ChunkPolicy:
    """The chunk-hygiene knobs, bundled — internal only.

    `iter_targets`' flat kwargs remain the public API; this is how they thread
    through `_post_targets`.
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
    # Pack across section boundaries, with heading lines as chunk CONTENT.
    # Only meaningful with `merge`; forces `heading_context` off, since an
    # inline heading is text the styler rewrites rather than fixed context.
    cross_heading: bool = False


_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*$")


def _norm_header(text: str) -> str:
    """Normalise a header title for matching: drop ``#``s, trailing ``{...}`` attrs, case."""
    text = text.lstrip("#").strip()
    text = re.sub(r"\s*\{[^}]*\}\s*$", "", text)  # strip a trailing pandoc attribute block
    return text.lower()


def _truncate_at_headers(body: str, headers: Sequence[str]) -> str:
    """Drop everything from the first matching section header to end of document.

    Entries in `headers` match level-agnostically and case-insensitively; the
    caller supplies the section names as policy.
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

    A link list has almost nothing left once link spans, list markers and markup
    are gone, while real prose keeps its words even when link-dense.
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
        # Garbage frontmatter raises out of the YAML parser. Empty meta means
        # the selector rejects the post — warned about, never silent.
        logger.warning("unparseable frontmatter in %s — treated as unselected", path)
        return {}, text
    return (meta or {}), body


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a body into ``(heading, section_body)`` pairs.

    A header line starts a new section and is itself **excluded** from the body —
    headers carry no paragraph voice and must never pack with prose — becoming
    the `heading` of the section that follows. The preamble has an empty heading.
    Run this *after* `editable_prose`, so code-comment ``#`` lines (stripped as
    protected blocks) can't be mistaken for headers.
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

    Joined with a blank line. A paragraph already over budget is emitted alone,
    kept whole — splitting mid-paragraph would corrupt the target.
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
    # Drop protected blocks before splitting, so they never become targets and
    # chunk boundaries match what the blog's own edit pipeline sees.
    text = editable_prose(body) if policy.prose_only else body
    cross = policy.cross_heading and policy.merge
    # An inline heading is content the styler rewrites, so it cannot also be
    # fixed context — cross-heading mode wins over heading_context.
    want_context = policy.heading_context == "immediate" and not cross
    gate_kw = dict(
        ignore_markers=policy.ignore_markers,
        drop_link_dumps=policy.drop_link_dumps,
        drop_list_items=policy.drop_list_items,
    )
    # Junk gate per-paragraph BEFORE packing, with no min/max floor: short prose
    # survives to be rescued by its neighbours, while tables/lists/link dumps are
    # dropped so they never pollute a packed block.
    prepack_kw = dict(min_chars=1, max_chars=None, **gate_kw)

    chunks: list[tuple[str, str]] = []  # (text, context)
    if cross:
        # Flatten to one ordered unit stream — each heading line followed by its
        # surviving paragraphs — and pack across section boundaries. Headings
        # bypass the junk gate: alone a heading is too short to clear it, but as
        # a packed block's first line it is the structure the styler must carry.
        units: list[str] = []
        for heading, section in _split_sections(text):
            if heading:
                units.append(heading)
            units.extend(p for p in split_paragraphs(section) if _keep_chunk(p, **prepack_kw))
        for block in _pack_paragraphs(units, merge_max_chars=policy.merge_max_chars):
            if _keep_chunk(
                block, min_chars=policy.min_chars, max_chars=policy.max_chars, **gate_kw
            ):
                chunks.append((block, ""))
    else:
        # Section-bounded: iterate header-delimited sections so a passage never
        # packs across a header, and the heading is available as context.
        for heading, section in _split_sections(text):
            ctx = heading if want_context else ""
            if not policy.merge:
                for c in split_paragraphs(section):
                    if _keep_chunk(
                        c, min_chars=policy.min_chars, max_chars=policy.max_chars, **gate_kw
                    ):
                        chunks.append((c, ctx))
            else:
                paras = [p for p in split_paragraphs(section) if _keep_chunk(p, **prepack_kw)]
                for block in _pack_paragraphs(paras, merge_max_chars=policy.merge_max_chars):
                    if _keep_chunk(
                        block, min_chars=policy.min_chars, max_chars=policy.max_chars, **gate_kw
                    ):
                        chunks.append((block, ctx))

    total = len(chunks)
    scale = policy.merge_max_chars if policy.merge else None
    mode = "cross-heading" if cross else None
    return [
        Target(text=t, source=source, chunk_index=i, chunk_total=total, context=c,
               merge_max_chars=scale, chunk_mode=mode)
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
    cross_heading: bool = False,
    sort_key: Callable[[Target], object] | None = None,
) -> list[Target]:
    """Collect prose chunks to use as synthesis targets.

    Two mutually exclusive input modes: **`files`** is a pre-selected list, and
    the `selector` is **not** applied to it (the caller owns selection);
    **`blog_root`** walks for `glob` and keeps posts where `selector(meta)` is
    true. Chunk hygiene below is all generic — policy stays in `selector` and the
    caller's marker choice.

    - `prose_only` (default True): drop protected blocks — fenced code,
      ``$$math$$``, ``:::`` divs/callouts, blockquotes — before splitting, so
      they never become targets (STYLE_SYSTEM preserves them verbatim anyway).
    - `min_chars` / `max_chars`: oversized chunks are DROPPED, never truncated —
      truncating corrupts the target.
    - `ignore_markers`: drop chunks containing any of these literal strings.
    - `drop_link_dumps`: drop chunks that are mostly markdown links, measured by
      prose residual rather than link density.
    - `drop_list_items`: drop chunks that are entirely markdown list items.
    - `merge`: pack consecutive paragraphs *within a section* up to
      `merge_max_chars`, never crossing a header. Rescues short paragraphs the
      `min_chars` floor would discard, and `min_chars` then floors the *packed*
      block. A single paragraph over budget is kept whole.
    - `stop_at_headers`: truncate each body at the first matching section header
      (level-agnostic, case-insensitive) — e.g. a trailing link dump.
    - `heading_context` (``"none"`` | ``"immediate"``): populate
      ``Target.context`` with the heading each chunk sits under.
    - `cross_heading` (requires `merge`): pack across section boundaries with
      heading lines as chunk **content**. This is the knob that actually buys
      length: most sections are far shorter than `merge_max_chars`, so
      section-bounded packing hits the section wall before the budget bites and
      raising the budget alone barely moves the distribution. Forces
      `heading_context` off — an inline heading is text the styler rewrites, not
      fixed context. See `_plans/v2-longer-rewrites.md`.

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
        cross_heading=cross_heading,
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


def hash_unit(*parts: object, salt: str = "") -> float:
    """A stable coordinate in [0, 1) for a tuple of identifiers.

    Pure function of the identifiers — no RNG state, no ordering, no corpus size
    — which is what lets a subsample stay put while the thing it is sampled from
    grows underneath it. Shares the digest → [0, 1) conversion with the splits
    hash rule but NOT the hashed input: each side's byte format is pinned by data
    already on disk, so they must stay separate.
    """
    blob = "\x00".join([str(salt), *(str(p) for p in parts)]).encode("utf-8")
    return unit_from_digest(hashlib.sha256(blob).digest())


def select_targets(
    targets: Sequence[Target],
    *,
    splits: dict | None = None,
    role: str | None = None,
    sample_frac: float | None = None,
    sample_salt: str = "",
) -> list[Target]:
    """Narrow a target list to an experiment's substrate: by splits role, then subsample.

    Two independent filters, both deliberately *not* random:

    - **`role`** keeps only chunks whose post carries that role under the by-post
      splits contract. An experiment measured with the voice classifier must not
      draw passages from the classifier's own training pool, or the detector
      marks its own homework; `role="styler"` is the usual answer, being disjoint
      from both the detector pool and the frozen eval stratum.

    - **`sample_frac`** keeps a fraction by `hash_unit(source, text)`, NOT the
      first N and not randomly. Membership is a property of the chunk's
      *content*, so as the corpus grows a later run is a strict **superset** of
      an earlier one: old measurements stay valid on their subset and two runs
      are comparable on the intersection. `--limit` cannot do this — its first N
      shifts the moment a new post sorts in ahead of an old one.

    `sample_salt` re-draws an independent subsample of the same size.
    """
    out = list(targets)
    if role is not None:
        if splits is None:
            raise ValueError("select_targets: role= requires splits=")
        out = [t for t in out if role_of(t.source, splits) == role]
    if sample_frac is not None:
        if not 0.0 < sample_frac <= 1.0:
            raise ValueError(f"select_targets: sample_frac must be in (0, 1], got {sample_frac}")
        if sample_frac < 1.0:
            out = [
                t
                for t in out
                if hash_unit(t.source, t.text, salt=sample_salt) < sample_frac
            ]
    return out
