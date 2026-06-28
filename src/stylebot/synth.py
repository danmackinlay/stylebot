"""Phase 2 — synthetic pair generation.

Library-first: `synthesize_pairs` is a typed function over explicit paths/params;
`ai-style synth` (`stylebot.bin.ai_style`) is a thin CLI wrapper. The blog build
can import `synthesize_pairs` directly.

Method (see `_plans/phase-2-synthetic-pairs.md`): take Dan's own human-authored
prose as the **target** (`messages[2]`, the assistant turn), ask an LLM to
paraphrase it into clearer/"more polished" prose — the **slop source**
(`messages[1]`, the user turn). The styler later learns to undo that transform.

Output is the **same** `pairs.jsonl` schema as Phase 1
(`stylebot.pairs.validate_pairs_file`), chunked the same way
(`stylebot.lib.split_paragraphs`), so real and synthetic pairs are mixable. Each
synthetic record additionally carries `meta.synthetic: true`,
`meta.generator: "<model>"`, `meta.synth_key` (for idempotent resume), and
`meta.tags` provenance.

Selection is a user-supplied policy (OVERVIEW "Selection is a user-supplied
policy"): `iter_targets` takes a `selector` defaulting to
`stylebot.lib.is_human_authored` plus an optional `sort_key`; callers pass their
own, or hand in a pre-selected file list and skip the walk entirely.

The generators are injected, not hardcoded: tests pass plain callables; the
`anthropic_generator` / `openai_generator` / `local_generator` factories build
real provider-backed ones (multi-source by design — rotate ≥2 so the styler
learns to undo AI writing broadly, not one model's tics).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import re

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.lib import (
    editable_prose,
    gather_qmd_files,
    is_human_authored,
    read_w_frontmatter_text,
    split_paragraphs,
)
from stylebot.pairs import build_pair_content

# Instruction we hand a generic LLM to manufacture "slop" from Dan's prose.
# It mirrors STYLE_SYSTEM's structure-preservation clause so the synthetic
# source differs from the target in *style*, not markdown shape — we want the
# styler to learn the voice transform, not a reformatting.
SLOP_SYSTEM = (
    "You are a writing assistant that polishes prose. Rewrite the user's "
    "passage to be clearer, more professional, and more engaging. "
    "Preserve all markdown structure (code fences, math, links, headings, "
    "list markers, blank lines) verbatim. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten passage, nothing else."
)

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

# Generous output budget for slop generation. Slop is an *expansion* of the
# target (AI prose runs longer than the human source — often 1.5-3x), so the
# cap must comfortably exceed the target's own token count, not match it. With
# targets capped at MAX_CHUNK_CHARS (~2k tokens), ~8k output tokens leaves room
# for 3-4x expansion. Well under every provider's non-streaming ceiling.
DEFAULT_SLOP_MAX_TOKENS = 8192


# ---------------------------------------------------------------------------
# Targets — Dan's prose chunks, the assistant-side of each pair
# ---------------------------------------------------------------------------


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


def _post_targets(
    path: Path,
    source: str,
    *,
    min_chars: int,
    max_chars: int | None,
    prose_only: bool,
    ignore_markers: Sequence[str],
    drop_link_dumps: bool,
    drop_list_items: bool,
    stop_at_headers: Sequence[str],
    merge: bool,
    merge_max_chars: int,
    heading_context: str,
) -> list[Target]:
    _, body = _load_meta_body(path)
    # Cut trailing dump sections (e.g. "## Incoming") before anything else.
    body = _truncate_at_headers(body, stop_at_headers)
    # Prose-only: drop protected blocks (code/math/:::divs/blockquotes) before
    # splitting, so they never become targets and chunk boundaries match what
    # the blog's own edit pipeline sees.
    text = editable_prose(body) if prose_only else body
    want_context = heading_context == "immediate"

    # Both modes iterate header-delimited sections (so the section heading is
    # available as context, and so a passage never packs across a header). The
    # heading is attached to each emitted chunk when heading_context is on.
    chunks: list[tuple[str, str]] = []  # (text, context)
    for heading, section in _split_sections(text):
        ctx = heading if want_context else ""
        if not merge:
            for c in split_paragraphs(section):
                if _keep_chunk(
                    c,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    ignore_markers=ignore_markers,
                    drop_link_dumps=drop_link_dumps,
                    drop_list_items=drop_list_items,
                ):
                    chunks.append((c, ctx))
        else:
            # Keep prose paragraphs (no min/max floor yet — short prose survives
            # to be rescued; junk dropped pre-pack), greedily pack to the budget,
            # then gate the packed block on min/max/link-list.
            paras = [
                p
                for p in split_paragraphs(section)
                if _keep_chunk(
                    p,
                    min_chars=1,
                    max_chars=None,
                    ignore_markers=ignore_markers,
                    drop_link_dumps=drop_link_dumps,
                    drop_list_items=drop_list_items,
                )
            ]
            for block in _pack_paragraphs(paras, merge_max_chars=merge_max_chars):
                if _keep_chunk(
                    block,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    ignore_markers=ignore_markers,
                    drop_link_dumps=drop_link_dumps,
                    drop_list_items=drop_list_items,
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
    chunk_opts = dict(
        min_chars=min_chars,
        max_chars=max_chars,
        prose_only=prose_only,
        ignore_markers=ignore_markers,
        drop_link_dumps=drop_link_dumps,
        drop_list_items=drop_list_items,
        stop_at_headers=stop_at_headers,
        merge=merge,
        merge_max_chars=merge_max_chars,
        heading_context=heading_context,
    )

    if files is not None:
        for f in files:
            path = Path(f)
            if not path.is_file():
                continue
            targets.extend(_post_targets(path, _source_label(path, root), **chunk_opts))
    else:
        for path in gather_qmd_files([], base=root, default_glob=glob):
            meta, _ = _load_meta_body(path)
            if not selector(meta):
                continue
            targets.extend(_post_targets(path, _source_label(path, root), **chunk_opts))

    if sort_key is not None:
        targets.sort(key=sort_key)
    return targets


# ---------------------------------------------------------------------------
# Generators — produce slop from a target, multi-source by design
# ---------------------------------------------------------------------------


@dataclass
class Generator:
    """A named slop producer. `name` becomes `meta.generator` on each pair."""

    name: str
    generate: Callable[[str], str] | None = None

    def __call__(self, target_text: str) -> str:
        if self.generate is None:
            raise RuntimeError(f"generator {self.name!r} has no callable (dry-run/name-only stub)")
        return self.generate(target_text)


def anthropic_generator(
    *,
    model: str = "claude-opus-4-8",
    system: str = SLOP_SYSTEM,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    api_key: str | None = None,
) -> Generator:
    """Claude-backed slop generator (`anthropic` SDK; key `ANTHROPIC_API_KEY`)."""
    import anthropic

    from stylebot import config

    client = anthropic.Anthropic(api_key=api_key or config.require_key("ANTHROPIC_API_KEY"))

    def generate(text: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        # A truncated slop (hit max_tokens) doesn't cover the whole target — that
        # is a broken pair, so fail loudly and let synthesize_pairs skip it.
        if resp.stop_reason == "max_tokens":
            raise RuntimeError(f"slop truncated at max_tokens={max_tokens} (raise it)")
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    return Generator(name=model, generate=generate)


def openai_generator(
    *,
    model: str = "gpt-4o",
    system: str = SLOP_SYSTEM,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    api_key: str | None = None,
    base_url: str | None = None,
    name: str | None = None,
) -> Generator:
    """OpenAI-compatible slop generator (`openai` SDK; key `OPENAI_API_KEY`).

    `base_url` repoints at any OpenAI-compatible endpoint — `local_generator`
    uses that to drive a utility/base model.
    """
    import openai

    from stylebot import config

    client = openai.OpenAI(
        api_key=api_key or config.require_key("OPENAI_API_KEY"),
        base_url=base_url,
    )

    def generate(text: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        )
        choice = resp.choices[0]
        # A truncated slop (finish_reason "length") is a broken pair — fail loudly.
        if choice.finish_reason == "length":
            raise RuntimeError(f"slop truncated at max_tokens={max_tokens} (raise it)")
        return (choice.message.content or "").strip()

    return Generator(name=name or model, generate=generate)


def local_generator(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    system: str = SLOP_SYSTEM,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
) -> Generator:
    """Local/utility base-model generator via an OpenAI-compatible endpoint.

    Reads `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` / `LOCAL_LLM_API_KEY` from the
    environment when not passed explicitly. Tagged `local-<model>` so its pairs
    are distinguishable in `meta.generator`.
    """
    from stylebot import config

    base_url = base_url or config.get_key("LOCAL_LLM_BASE_URL") or "http://localhost:8080/v1"
    model = model or config.get_key("LOCAL_LLM_MODEL") or "local"
    api_key = api_key or config.get_key("LOCAL_LLM_API_KEY") or "not-needed"
    return openai_generator(
        model=model,
        system=system,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        name=f"local-{model}",
    )


# ---------------------------------------------------------------------------
# Synthesis — assign generators to targets, generate, append schema-valid pairs
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _synth_key(generator_name: str, target_text: str, context: str = "") -> str:
    """Stable id for a (generator, context, target) pair — the resume/dedup key.

    Context is part of the key so toggling/changing heading context regenerates
    (a context-less pair and a context-prefixed pair are different training data).
    """
    h = hashlib.sha256()
    h.update(generator_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(context.encode("utf-8"))
    h.update(b"\x00")
    h.update(target_text.encode("utf-8"))
    return h.hexdigest()[:16]


def _effective_context(target: Target, context_dropout: float) -> str:
    """The context to actually use for a target, applying deterministic dropout.

    Dropping a deterministic fraction (keyed on the body hash, so resume is
    stable) keeps some pairs heading-less, so the styler doesn't *require* a
    heading at inference.
    """
    if not target.context or context_dropout <= 0:
        return target.context
    bucket = int(hashlib.sha256(target.text.encode("utf-8")).hexdigest(), 16) % 1000
    return "" if bucket < context_dropout * 1000 else target.context


def _capture_id(source: str, generator_name: str) -> str:
    """Group a post's chunks from one generator under one capture id."""
    h = hashlib.sha256(f"{source}\x00{generator_name}".encode("utf-8"))
    return h.hexdigest()[:8]


def existing_synth_keys(pairs_path: Path | str) -> set[str]:
    """Read the `meta.synth_key`s already present in a `pairs.jsonl`."""
    pairs_path = Path(pairs_path)
    keys: set[str] = set()
    if not pairs_path.exists():
        return keys
    with pairs_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (rec.get("meta") or {}).get("synth_key") if isinstance(rec, dict) else None
            if key:
                keys.add(key)
    return keys


def _build_record(
    *,
    slop: str,
    target: Target,
    generator_name: str,
    synth_key: str,
    context: str = "",
    extra_tags: Sequence[str] = (),
) -> dict:
    meta = {
        "source": target.source,
        "captured_at": _now_iso(),
        "capture_id": _capture_id(target.source, generator_name),
        "chunk_index": target.chunk_index,
        "chunk_total": target.chunk_total,
        "before_chars": len(slop),  # body lengths (the transform), excluding the heading prefix
        "after_chars": len(target.text),
        "synthetic": True,
        "generator": generator_name,
        "synth_key": synth_key,
        "tags": ["synthetic", "paraphrase", *extra_tags],
    }
    if context:
        # Shared contract: identical heading prefix on both sides (see
        # stylebot.pairs.build_pair_content); the styler restyles the body
        # conditioned on, but never rewriting, the heading.
        meta["context"] = context
        meta["context_mode"] = "immediate"
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": build_pair_content(context, slop)},
            {"role": "assistant", "content": build_pair_content(context, target.text)},
        ],
        "meta": meta,
    }


@dataclass
class SynthResult:
    """Outcome of a `synthesize_pairs` run."""

    written: int = 0
    skipped_existing: int = 0
    planned: int = 0  # (target, generator) assignments before dedup
    errors: list[tuple[str, str]] = field(default_factory=list)  # (synth_key, message)
    per_generator: dict[str, int] = field(default_factory=dict)


def _assign(
    targets: Sequence[Target],
    generator_names: Sequence[str],
    *,
    per_generator: bool,
    context_dropout: float = 0.0,
) -> list[tuple[Target, str, str, str]]:
    """Pair targets with generators → ``(target, generator, synth_key, context)``.

    Default (rotate, `per_generator=False`): round-robin — target *i* goes to
    generator *i % n*. Cheap, and across the corpus yields ≥2 generators.
    `per_generator=True`: every target × every generator (n× the pairs/cost).
    `context` is the effective heading context after dropout; the `synth_key`
    incorporates it so toggling context regenerates.
    """
    if not generator_names:
        return []
    out: list[tuple[Target, str, str, str]] = []

    def assign_one(t: Target, name: str) -> None:
        ctx = _effective_context(t, context_dropout)
        out.append((t, name, _synth_key(name, t.text, ctx), ctx))

    if per_generator:
        for t in targets:
            for name in generator_names:
                assign_one(t, name)
    else:
        n = len(generator_names)
        for i, t in enumerate(targets):
            assign_one(t, generator_names[i % n])
    return out


def synthesize_pairs(
    targets: Sequence[Target],
    data_dir: Path | str,
    generators: Sequence[Generator],
    *,
    per_generator: bool = False,
    dry_run: bool = False,
    extra_tags: Sequence[str] = (),
    context_dropout: float = 0.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> SynthResult:
    """Generate synthetic pairs and append them to `data_dir/pairs.jsonl`.

    Idempotent and resumable: each pair carries a `meta.synth_key`
    (`hash(generator, context, target)`); assignments whose key is already in the
    file are skipped, so re-running never duplicates and a crashed run resumes
    where it stopped (records are appended one-per-line, flushed as they go).

    When targets carry heading `context` (`iter_targets(heading_context=...)`),
    the heading is prepended verbatim to both sides of the pair via
    `stylebot.pairs.build_pair_content`, and the slop is generated from the body
    only (so the heading is never paraphrased). `context_dropout` keeps a
    deterministic fraction heading-less.

    `dry_run` plans the assignment and reports counts without calling any
    generator or writing — use it to vet selection against the real blog with
    no API spend (generators may be name-only `Generator(name, generate=None)`
    stubs in that case).
    """
    data_dir = Path(data_dir)
    pairs_path = data_dir / "pairs.jsonl"
    names = [g.name for g in generators]
    by_name = {g.name: g for g in generators}

    assignments = _assign(targets, names, per_generator=per_generator, context_dropout=context_dropout)
    result = SynthResult(planned=len(assignments))

    seen = existing_synth_keys(pairs_path)
    todo = [a for a in assignments if a[2] not in seen]
    result.skipped_existing = len(assignments) - len(todo)

    if dry_run:
        for _, name, _, _ in todo:
            result.per_generator[name] = result.per_generator.get(name, 0) + 1
        return result

    data_dir.mkdir(parents=True, exist_ok=True)
    written_keys: set[str] = set()
    total = len(todo)
    with pairs_path.open("a", encoding="utf-8") as fp:
        for idx, (target, name, key, context) in enumerate(todo, start=1):
            if on_progress is not None:
                on_progress(idx, total)
            if key in written_keys:  # guard within-run dup (same target listed twice)
                continue
            try:
                slop = by_name[name](target.text)
            except Exception as exc:  # API error etc. — record and continue
                result.errors.append((key, f"{type(exc).__name__}: {exc}"))
                continue
            if not slop.strip():
                result.errors.append((key, "generator returned empty output"))
                continue
            record = _build_record(
                slop=slop,
                target=target,
                generator_name=name,
                synth_key=key,
                context=context,
                extra_tags=extra_tags,
            )
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
            written_keys.add(key)
            result.written += 1
            result.per_generator[name] = result.per_generator.get(name, 0) + 1

    return result
