"""Targets — human-authored prose chunks, the assistant side of each pair.

Turning posts into synthesis targets: the frontmatter/selector walk, prose-only
extraction, section-aware splitting and merge-packing, and the chunk-hygiene
gates. All generic — policy arrives via the caller's `selector`, markers and
header names.

`stylebot.synth` re-exports the subset its CLI kit needs (`iter_targets`, the
chunk-size constants); this module is the implementation home and the import
site to prefer.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from stylebot.cli_options import opt
from stylebot.lib import (
    DEFAULT_SOURCE,
    SourcePolicy,
    editable_prose,
    gather_qmd_files,
    read_w_frontmatter_text,
    segment_for_edit,
    split_paragraphs,
)
from stylebot.markdown import DivIsProse, heading_title, is_heading_line
from stylebot.provenance import policy_digest
from stylebot.splits import ROLES, role_of, unit_from_digest

logger = logging.getLogger(__name__)

# Chunk hygiene. There is no voice transform to learn from a bare heading, a
# fenced code block, a link list, or a stub paragraph — so trim them.
MIN_CHUNK_CHARS = 80  # drop chunks shorter than this
MAX_CHUNK_CHARS = 8000  # drop chunks longer than this (truncating would corrupt the target)
MERGE_MAX_CHUNK_CHARS = 1500  # soft budget per packed passage in merge mode

# Link-LIST detection by *prose residual*, not link-char density: strip the
# `[text](url)` spans and measure the prose left as a fraction of the chunk.
# Density mis-fires on link-dense prose because URLs are long; the residual is
# URL- and size-immune.
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
    # Chunk-scale facet, stamped into pair meta so a corpus mixing diets stays
    # ablatable by scale: the merge budget (None = unmerged), and how the chunk
    # was cut — None = within one section (heading carried as `context`),
    # "cross-heading" = heading lines packed in as content.
    merge_max_chars: int | None = None
    chunk_mode: str | None = None
    # Provenance carried from the cut, stamped into `meta.provenance` (D36).
    # `automation` is the post's frontmatter level AS READ AT CUT TIME — the
    # corpus is a frozen record and the post is not — read under the cut's
    # `lib.SourcePolicy` (`level_of`), so another field's value never lands here.
    automation: int | None = None
    policy_digest: str | None = None


@dataclass(frozen=True)
class ChunkPolicy:
    """The chunk-hygiene knobs — WHAT counts as a target. One declaration.

    Each field carries its own click flag via `cli_options.opt()`, so `ai-style
    synth` and the blog's mirror declare the surface once and re-pin what differs
    with `synth_options(defaults=[ChunkPolicy(...)])`. A field WITHOUT `opt()` is
    library-only: no flag, set by a caller in code.
    """

    cli_section = "Chunking / hygiene (selection)"

    min_chars: int = opt(
        MIN_CHUNK_CHARS,
        "--min-chars",
        show_default=True,
        type=int,
        help="Skip prose chunks shorter than this.",
    )
    max_chars: int | None = opt(
        MAX_CHUNK_CHARS,
        "--max-chars",
        show_default=True,
        type=int,
        help="Drop prose chunks longer than this (0 disables the cap).",
    )
    prose_only: bool = opt(
        True,
        "--prose-only/--no-prose-only",
        show_default=True,
        help="Drop code/math/:::divs/blockquotes before chunking.",
    )
    ignore_markers: Sequence[str] = opt(
        (),
        "--ignore-marker",
        "ignore_markers",
        multiple=True,
        help="Drop any chunk containing this literal (repeatable), e.g. a stub marker.",
    )
    drop_link_dumps: bool = opt(
        True,
        "--drop-link-dumps/--keep-link-dumps",
        show_default=True,
        help="Drop chunks that are mostly markdown links (little authored prose).",
    )
    drop_list_items: bool = opt(
        False,
        "--drop-list-items/--keep-list-items",
        show_default=True,
        help="Drop chunks that are entirely markdown list items.",
    )
    merge: bool = opt(
        False,
        "--merge/--no-merge",
        show_default=True,
        help="Pack consecutive prose paragraphs within a section into multi-paragraph passages.",
    )
    merge_max_chars: int = opt(
        MERGE_MAX_CHUNK_CHARS,
        "--merge-max-chars",
        show_default=True,
        type=int,
        help="Soft char budget per packed passage (merge mode).",
    )
    stop_at_headers: Sequence[str] = opt(
        (),
        "--stop-at-header",
        "stop_at_headers",
        multiple=True,
        help="Truncate each post body at this section header (repeatable), e.g. '## Incoming'.",
    )
    heading_context: str = opt(
        "none",
        "--heading-context",
        type=click.Choice(["none", "immediate"]),
        show_default=True,
        help="Prepend the section heading (verbatim) to both sides of each pair as fixed context.",
    )
    # Pack across section boundaries, with heading lines as chunk CONTENT.
    # Only meaningful with `merge`; forces `heading_context` off, since an
    # inline heading is text the styler rewrites rather than fixed context.
    cross_heading: bool = opt(
        False,
        "--cross-heading/--no-cross-heading",
        show_default=True,
        help="Pack across section boundaries with headings as chunk content (needs --merge; "
        "overrides --heading-context). This, not --merge-max-chars, is what buys chunk length.",
    )
    # Library-only, and the one field here that is a *train/serve contract*
    # rather than a hygiene knob: `_prose_for_chunking` for what it does, D27 for
    # why it defaults to the behaviour that does NOT match inference.
    pack_across_protected: bool = True
    # Library-only for a different reason: a predicate is not expressible on a
    # command line, and *which* `:::` classes wrap prose is caller policy (D29).
    div_is_prose: DivIsProse | None = None

    def __post_init__(self) -> None:
        # `--max-chars 0` is the CLI's spelling of "uncapped", and the library's
        # is None. Reconciling them HERE makes the sentinel a property of the
        # type rather than of one translation helper every caller had to
        # remember: a 0 that slipped past would drop every chunk, since
        # `_keep_chunk` compares `len(s) > max_chars`.
        if self.max_chars == 0:
            object.__setattr__(self, "max_chars", None)


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
        title = heading_title(line)
        if title is not None and _norm_header(title) in wanted:
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
    if drop_link_dumps and _is_link_list(s):  # noqa: SIM103 — last of a guard chain
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
        if is_heading_line(line):
            if cur:
                sections.append((heading, "".join(cur)))
                cur = []
            heading = line.strip()  # frames the next section
        else:
            cur.append(line)
    if cur:
        sections.append((heading, "".join(cur)))
    return sections


#: Stands in for a deleted protected block while a post is being chunked, so
#: packing can see where one was. Never reaches a target: `_pack` flushes on it
#: and the unmerged path skips it. NUL-delimited because prose cannot contain it.
PROTECTED_MARK = "\x00stylebot:protected\x00"


def _prose_for_chunking(body: str, policy: ChunkPolicy) -> str:
    """The text `_post_targets` chunks — protected blocks gone, one way or another.

    Two spellings of "gone", and the difference is the train/serve packing
    contract (D27):

    - `pack_across_protected=True` (the default, and what the live corpus was cut
      under): `editable_prose` deletes every protected span and **joins** the
      remnants, so paragraphs either side of a figure pack into one target — a
      passage the document does not contain and `infer._segment` cannot present.
    - `pack_across_protected=False`: each span becomes `PROTECTED_MARK`, its own
      paragraph, which flushes the packer, giving inference's boundaries.

    A mark rather than a list of runs, because `_split_sections` and
    `Target.context` need the heading structure of the *whole* body.
    """
    if not policy.prose_only:
        return body
    if policy.pack_across_protected:
        return editable_prose(body, div_is_prose=policy.div_is_prose)
    return "".join(
        seg if editable else f"\n\n{PROTECTED_MARK}\n\n"
        for seg, editable in segment_for_edit(body, div_is_prose=policy.div_is_prose)
    )


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


def _pack(units: Sequence[str], *, merge_max_chars: int) -> list[str]:
    """`_pack_paragraphs`, flushing at every `PROTECTED_MARK`.

    Identical to `_pack_paragraphs` when no mark is present, which is every call
    under the default policy — `_prose_for_chunking` only emits marks when
    `pack_across_protected` is off.
    """
    blocks: list[str] = []
    group: list[str] = []
    for unit in units:
        if unit == PROTECTED_MARK:
            blocks.extend(_pack_paragraphs(group, merge_max_chars=merge_max_chars))
            group = []
        else:
            group.append(unit)
    blocks.extend(_pack_paragraphs(group, merge_max_chars=merge_max_chars))
    return blocks


def _post_targets(
    body: str,
    source: str,
    policy: ChunkPolicy,
    *,
    meta: dict | None = None,
    source_policy: SourcePolicy = DEFAULT_SOURCE,
) -> list[Target]:
    """Chunk one post's **already-loaded** body into targets.

    Takes the body rather than a path so every caller loads and YAML-parses a
    post exactly once: the `blog_root` walk needs the frontmatter for its
    `selector`, and re-reading here would parse the same file twice (and warn
    twice about an unparseable one).

    `meta` is that same frontmatter, read under `source_policy` and carried
    onto each `Target` (see `Target.automation`) so the pair record can say
    what the selector was looking at when the target was cut.
    """
    # Cut trailing dump sections (e.g. "## Incoming") before anything else.
    body = _truncate_at_headers(body, policy.stop_at_headers)
    # Drop protected blocks before splitting, so they never become targets and
    # chunk boundaries match what the blog's own edit pipeline sees.
    text = _prose_for_chunking(body, policy)
    cross = policy.cross_heading and policy.merge
    # An inline heading is content the styler rewrites, so it cannot also be
    # fixed context — cross-heading mode wins over heading_context.
    want_context = policy.heading_context == "immediate" and not cross
    gate_kw = dict(
        ignore_markers=policy.ignore_markers,
        drop_link_dumps=policy.drop_link_dumps,
        drop_list_items=policy.drop_list_items,
    )
    keep_kw = dict(min_chars=policy.min_chars, max_chars=policy.max_chars, **gate_kw)
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
            units.extend(
                p
                for p in split_paragraphs(section)
                if p == PROTECTED_MARK or _keep_chunk(p, **prepack_kw)
            )
        for block in _pack(units, merge_max_chars=policy.merge_max_chars):
            if _keep_chunk(block, **keep_kw):
                chunks.append((block, ""))
    else:
        # Section-bounded: iterate header-delimited sections so a passage never
        # packs across a header, and the heading is available as context.
        for heading, section in _split_sections(text):
            ctx = heading if want_context else ""
            if not policy.merge:
                for c in split_paragraphs(section):
                    if c != PROTECTED_MARK and _keep_chunk(c, **keep_kw):
                        chunks.append((c, ctx))
            else:
                paras = [
                    p
                    for p in split_paragraphs(section)
                    if p == PROTECTED_MARK or _keep_chunk(p, **prepack_kw)
                ]
                for block in _pack(paras, merge_max_chars=policy.merge_max_chars):
                    if _keep_chunk(block, **keep_kw):
                        chunks.append((block, ctx))

    total = len(chunks)
    scale = policy.merge_max_chars if policy.merge else None
    mode = "cross-heading" if cross else None
    digest = policy_digest(policy)
    return [
        Target(
            text=t,
            source=source,
            chunk_index=i,
            chunk_total=total,
            context=c,
            merge_max_chars=scale,
            chunk_mode=mode,
            automation=source_policy.level_of(meta),
            policy_digest=digest,
        )
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
    source_policy: SourcePolicy = DEFAULT_SOURCE,
    selector: Callable[[dict], bool] | None = None,
    glob: str | None = None,
    policy: ChunkPolicy | None = None,
    **chunk_kw,
) -> list[Target]:
    """Collect prose chunks to use as synthesis targets.

    `blog_root` supplies the label root and the selector; `files` replaces only
    the walk (D80). Alone, `blog_root` walks for `source_policy.glob` and keeps
    posts where `selector(meta)` is true (default `source_policy.selects`), and
    `files` is taken as-is, unselected; together the caller's list is gated by
    the selector and labelled blog-relative.

    The chunk knobs arrive as a ready `policy=ChunkPolicy(...)` or as loose
    keywords for its constructor. **`ChunkPolicy` owns every knob and default.**

    Returns a flat list of `Target`s in document order, post by post. There is
    no ordering knob — arms are assigned by content hash, so order cannot change
    the corpus (D43); a caller that wants one sorts the returned list.
    """
    if files is None and blog_root is None:
        raise ValueError("provide `files` (pre-selected list) or `blog_root` (walk)")
    if policy is not None and chunk_kw:
        raise ValueError(
            f"pass either `policy` or loose chunk keywords, not both "
            f"(got policy= plus {sorted(chunk_kw)})"
        )

    targets: list[Target] = []
    root = Path(blog_root) if blog_root is not None else None
    policy = policy if policy is not None else ChunkPolicy(**chunk_kw)
    keep = selector if selector is not None else source_policy.selects

    if files is not None:
        paths = (p for p in map(Path, files) if p.is_file())
    else:
        paths = gather_qmd_files([], base=root, default_glob=glob or source_policy.glob)
    for path in paths:
        meta, body = _load_meta_body(path)
        if root is not None and not keep(meta):
            continue
        label = _source_label(path, root)
        targets.extend(_post_targets(body, label, policy, meta=meta, source_policy=source_policy))

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
      splits contract, so an experiment measured with the voice classifier does
      not draw from the classifier's own training pool (D17, D34).
    - **`sample_frac`** keeps a fraction by `hash_unit(source, text)`, so
      membership is a property of the chunk's *content*: a later run is a strict
      **superset** of an earlier one, and two runs are comparable on the
      intersection. `--limit` cannot do this.

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
            out = [t for t in out if hash_unit(t.source, t.text, salt=sample_salt) < sample_frac]
    return out


def split_off_role(
    targets: Sequence[Target], *, splits: dict, role: str
) -> tuple[list[Target], list[Target]]:
    """Partition targets into (those whose post holds `role`, everything else).

    The two-sided form of `select_targets(role=...)`, for callers that must
    *report* what they removed rather than quietly keep a subset — the
    frozen-eval guard in `synth_cli` names the posts it refused, because silence
    is how eval prose got cut as training targets in the first place.
    """
    if role not in ROLES:
        raise ValueError(f"split_off_role: unknown role {role!r}")
    matched, rest = [], []
    for target in targets:
        (matched if role_of(target.source, splits) == role else rest).append(target)
    return matched, rest
