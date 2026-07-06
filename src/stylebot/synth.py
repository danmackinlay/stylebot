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
`meta.generator: "<model>"`, `meta.slop_strategy: "<which slop prompt>"`,
`meta.synth_key` (for idempotent resume), and `meta.tags` provenance.

Selection is a user-supplied policy (OVERVIEW "Selection is a user-supplied
policy"): `iter_targets` takes a `selector` defaulting to
`stylebot.lib.is_human_authored` plus an optional `sort_key`; callers pass their
own, or hand in a pre-selected file list and skip the walk entirely.

The generators are injected, not hardcoded: tests pass plain callables; the
`openai_generator` / `local_generator` / `openrouter_generator` factories build
real provider-backed ones (multi-source by design — rotate ≥2 so the styler
learns to undo AI writing broadly, not one model's tics; OpenRouter reaches many
upstream models off a single key, so hosted models like Claude/GPT go through it).

The slop *prompt* is itself a knob: `STRATEGIES` maps a label → a system prompt
flavour, recorded as `meta.slop_strategy` and folded into `synth_key`, so you can
generate, eyeball, and ablate different flavours of slop without them colliding
on resume or blurring together in the corpus.
"""

from __future__ import annotations

import hashlib
import json
import time
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
from stylebot.pairs import build_pair_content, iter_pairs

# Instruction we hand a generic LLM to manufacture "slop" from Dan's prose.
# It mirrors STYLE_SYSTEM's structure-preservation clause so the synthetic
# source differs from the target in *style*, not markdown shape — we want the
# styler to learn the voice transform, not a reformatting.
# Shared tail for every slop strategy: the formatting contract (preserve
# structure, return only the passage). Identical across strategies so the only
# thing that varies between them is the *flavour* of slop requested.
_SLOP_PRESERVE = (
    "Preserve all markdown structure (code fences, math, links, headings, "
    "list markers, blank lines) verbatim. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten passage, nothing else."
)

# Named slop strategies: label -> the system prompt that produces that flavour of
# slop. The label is recorded as `meta.slop_strategy` and folded into `synth_key`,
# so pairs from different strategies neither collide on resume nor blur together —
# you can ablate "which flavour of slop teaches the styler best". These are
# GENERIC AI-prose flavours; an author's own slop catalogue is injected as a
# custom prompt (CLI `--slop-system-file` / library `system=`), keeping stylebot
# free of any one author's slop definition.
SLOP_SYSTEM = (  # "polish": the neutral baseline (clearer / more professional)
    "You are a writing assistant that polishes prose. Rewrite the user's "
    "passage to be clearer, more professional, and more engaging. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_ENGAGING = (  # "engaging": hooks, signposting, surfaced takeaways
    "You are an enthusiastic content editor. Rewrite the user's passage to be "
    "maximally engaging and accessible to a broad audience: open with a hook, "
    "add helpful signposting, surface the key takeaways, and keep the reader "
    "moving. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_CATALOGUE = (  # "catalogue": the stereotypical LLM register, on purpose
    "You are a typical AI writing assistant. Rewrite the user's passage in the "
    "default polished register of a large language model: smooth, measured, and "
    "explanatory. Lean into the characteristic moves — open with throat-clearing "
    "context ('In today's world', 'It's worth noting that'), add signposting and "
    "a tidy summary, prefer abstract Latinate vocabulary and rule-of-three "
    "phrasing, hedge claims ('can', 'may', 'often', 'arguably'), and even out the "
    "rhythm so sentences land at a similar measured length. " + _SLOP_PRESERVE
)

@dataclass(frozen=True)
class SlopStrategy:
    """A named slop-prompt flavour: a human label, the system prompt, a version.

    `version` is bumped by hand when the prompt text changes meaningfully; the
    stable `prompt_id` (a content hash, see `prompt_id_of`) is what actually
    identifies the prompt for faceting/dedup, so editing a prompt changes its id
    regardless of the version bump.
    """

    label: str
    system: str
    version: int = 1


STRATEGIES: dict[str, SlopStrategy] = {
    "polish": SlopStrategy("polish", SLOP_SYSTEM, version=1),
    "engaging": SlopStrategy("engaging", SLOP_SYSTEM_ENGAGING, version=1),
    "catalogue": SlopStrategy("catalogue", SLOP_SYSTEM_CATALOGUE, version=1),
}
DEFAULT_STRATEGY = "polish"

# Reasoning is a recorded *covariate*, not a silent default. Slop generation is a
# paraphrase, but real AI prose is often produced at high reasoning, so we default
# HIGH and let experiments sweep down (see `_reasoning_extra_body`).
DEFAULT_REASONING_EFFORT = "high"


def prompt_id_of(system_text: str) -> str:
    """Stable content id for ANY slop system prompt (registry or custom file).

    Hashing the actual prompt text means a custom `--slop-system-file` gets a
    stable id and is faceted/deduped exactly like a registry strategy, and editing
    a registry prompt changes its id (so old and new pairs stay distinguishable).
    """
    return hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:12]


def resolve_strategy(name: str, system: str | None = None) -> tuple[str, str, int, str]:
    """Resolve a strategy name to ``(label, system_prompt, version, prompt_id)``.

    An explicit ``system`` overrides the registry, so a caller can inject a custom
    (e.g. blog-specific) slop prompt under any label without stylebot needing to
    know that author's catalogue; such a prompt has version 0 and is identified by
    its content hash. A name absent from the registry is an error *unless* an
    explicit ``system`` is supplied.
    """
    if system is not None:
        return name, system, 0, prompt_id_of(system)
    try:
        strat = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(
            f"unknown slop strategy {name!r}; known: {known} "
            f"(or pass an explicit system prompt / --slop-system-file)"
        ) from None
    return strat.label, strat.system, strat.version, prompt_id_of(strat.system)

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
# Per-request HTTP timeout for slop generation. Without one, the openai SDK
# waits 600s per attempt (x its automatic retries) — a bad upstream stalls a
# sequential run for half an hour in silence. 300s clears even slow
# high-reasoning generations (~60-120s observed) with headroom; a timed-out
# pair is recorded in SynthResult.errors and the run continues.
DEFAULT_GEN_TIMEOUT = 300.0


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


@dataclass(frozen=True)
class GenOutput:
    """A generator's output: the slop text plus per-call generation covariates.

    A generator's `generate` may return a bare ``str`` (test fakes / simple
    callables) or a ``GenOutput`` whose ``meta`` carries the recorded generation
    covariates (model, reasoning_effort, temperature, top_p, max_tokens, token
    usage, finish_reason, prompt id/version). `synthesize_pairs` coerces either via
    `_normalize_gen_output`, so bare-string callables keep working unchanged.
    """

    text: str
    meta: dict = field(default_factory=dict)


def _normalize_gen_output(out: "str | GenOutput") -> tuple[str, dict]:
    """Coerce a generator return (``str`` or ``GenOutput``) to ``(text, gen_meta)``."""
    if isinstance(out, GenOutput):
        return out.text, dict(out.meta)
    return out, {}


@dataclass
class Generator:
    """A named slop producer.

    `name` becomes `meta.generator` (the model id); `strategy` becomes
    `meta.slop_strategy` (which slop *prompt* produced the pair). `reasoning_effort`
    and `prompt_id` also feed the `synth_key`, so the same model under two
    strategies / reasoning levels / prompts yields distinct, non-colliding pairs.
    `generate` may return a bare `str` or a `GenOutput` (text + recorded covariates).
    """

    name: str
    generate: Callable[[str], "str | GenOutput"] | None = None
    strategy: str = DEFAULT_STRATEGY
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    prompt_id: str = ""
    prompt_version: int = 0

    def __call__(self, target_text: str) -> "str | GenOutput":
        if self.generate is None:
            raise RuntimeError(f"generator {self.name!r} has no callable (dry-run/name-only stub)")
        return self.generate(target_text)


# Approximate per-family reasoning budgets for upstreams that take a token budget
# instead of an effort enum.
_REASONING_MAX_TOKENS = {"high": 8000, "medium": 4000, "low": 1500}
# OpenRouter model-id prefixes whose upstreams take a `max_tokens` reasoning budget
# rather than the OpenAI/Anthropic `effort` enum (best-effort; OpenRouter normalizes
# the rest, and the REQUESTED effort is recorded regardless of the wire shape).
_REASONING_BUDGET_FAMILIES = ("google/", "qwen/", "nvidia/", "deepseek/")


def _reasoning_extra_body(model: str, effort: str) -> dict | None:
    """Map a requested reasoning effort to OpenRouter's `reasoning` request field.

    `off` disables reasoning; budget-style families get a token budget; everyone
    else gets the effort enum. Best-effort across heterogeneous upstreams — the
    *requested* effort is recorded in `meta.gen` independent of what the provider
    honors, and `finish_reason`/`completion_tokens` let you detect a model that
    reasoned anyway.
    """
    if effort == "off":
        return {"reasoning": {"enabled": False}}
    if model.startswith(_REASONING_BUDGET_FAMILIES):
        return {"reasoning": {"max_tokens": _REASONING_MAX_TOKENS[effort]}}
    return {"reasoning": {"effort": effort}}


def openai_generator(
    *,
    model: str = "gpt-4o",
    strategy: str = DEFAULT_STRATEGY,
    system: str | None = None,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    name: str | None = None,
    extra_body: dict | None = None,
    timeout: float | None = DEFAULT_GEN_TIMEOUT,
) -> Generator:
    """OpenAI-compatible slop generator (`openai` SDK; key `OPENAI_API_KEY`).

    `base_url` repoints at any OpenAI-compatible endpoint — `local_generator` and
    `openrouter_generator` use that to drive a base model / OpenRouter. `extra_body`
    passes provider-specific knobs through (e.g. OpenRouter's `reasoning` field, set
    by `openrouter_generator`). `reasoning_effort` is recorded verbatim as the
    *requested* covariate regardless of whether/how the provider honors it; sampling
    params (`temperature`/`top_p`) are sent only when set, and recorded. `generate`
    returns a `GenOutput` carrying these covariates plus token usage.
    """
    import openai

    from stylebot import config

    label, system, prompt_version, prompt_id = resolve_strategy(strategy, system)
    client = openai.OpenAI(
        api_key=api_key or config.require_key("OPENAI_API_KEY"),
        base_url=base_url,
        timeout=timeout,
    )

    def generate(text: str) -> GenOutput:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        }
        # Send sampling/reasoning knobs only when set, so providers keep their
        # defaults (and so the recorded request mirrors what was actually sent).
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if extra_body:
            kwargs["extra_body"] = extra_body
        t0 = time.monotonic()
        resp = client.chat.completions.create(**kwargs)
        gen_seconds = time.monotonic() - t0
        # Some providers return choices=None/[] on an upstream error rather than
        # raising — surface a clear, catchable message, not an opaque TypeError.
        if not resp.choices:
            raise RuntimeError(f"{model}: provider returned no choices (upstream error?)")
        choice = resp.choices[0]
        # A truncated slop (finish_reason "length") is a broken pair — fail loudly.
        if choice.finish_reason == "length":
            raise RuntimeError(f"slop truncated at max_tokens={max_tokens} (raise --max-tokens)")
        usage = getattr(resp, "usage", None)
        # OpenRouter/OpenAI split reasoning tokens out of completion_tokens here
        # (None when the provider doesn't report it). Latency + this split let a
        # slow run be diagnosed from the corpus alone: reasoning blowout shows as
        # reasoning_tokens ~ its budget; a slow upstream shows as low
        # completion_tokens / gen_seconds.
        details = getattr(usage, "completion_tokens_details", None)
        gen_meta = {
            "model": model,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": getattr(details, "reasoning_tokens", None),
            "gen_seconds": round(gen_seconds, 2),
            "prompt_id": prompt_id,
            "prompt_version": prompt_version,
            "prompt_label": label,
        }
        return GenOutput((choice.message.content or "").strip(), gen_meta)

    return Generator(
        name=name or model,
        generate=generate,
        strategy=label,
        reasoning_effort=reasoning_effort,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
    )


def local_generator(
    *,
    model: str | None = None,
    strategy: str = DEFAULT_STRATEGY,
    base_url: str | None = None,
    api_key: str | None = None,
    system: str | None = None,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout: float | None = DEFAULT_GEN_TIMEOUT,
) -> Generator:
    """Local/utility base-model generator via an OpenAI-compatible endpoint.

    Reads `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` / `LOCAL_LLM_API_KEY` from the
    environment when not passed explicitly. Tagged `local-<model>` so its pairs
    are distinguishable in `meta.generator`. `reasoning_effort` is recorded but no
    reasoning wire-param is sent (local OpenAI-compatible servers vary).
    """
    from stylebot import config

    base_url = base_url or config.get_key("LOCAL_LLM_BASE_URL") or "http://localhost:8080/v1"
    model = model or config.get_key("LOCAL_LLM_MODEL") or "local"
    api_key = api_key or config.get_key("LOCAL_LLM_API_KEY") or "not-needed"
    return openai_generator(
        model=model,
        strategy=strategy,
        system=system,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url,
        name=f"local-{model}",
        timeout=timeout,
    )


def openrouter_generator(
    *,
    model: str,
    strategy: str = DEFAULT_STRATEGY,
    system: str | None = None,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = DEFAULT_GEN_TIMEOUT,
) -> Generator:
    """OpenRouter slop generator — one key, many upstream models.

    OpenRouter is OpenAI-compatible, so this is `openai_generator` pointed at the
    OpenRouter endpoint. `model` is an OpenRouter model id (e.g.
    ``anthropic/claude-opus-4.8``, ``qwen/qwen3-8b``), which makes multi-source slop
    rotation a single-credential affair. Tagged ``openrouter/<model>`` in
    `meta.generator` so its pairs stay distinguishable.

    Reads `OPENROUTER_API_KEY` (required) and optional `OPENROUTER_BASE_URL`
    (default ``https://openrouter.ai/api/v1``) from the environment / `.env`.

    `reasoning_effort` (high|medium|low|off) is a recorded covariate. Many models
    (Qwen3, Nemotron, …) reason by default, which on a paraphrase burns the token
    budget (≈14× completion tokens) and truncates the output; `_reasoning_extra_body`
    maps the requested effort to OpenRouter's `reasoning` field per model family.
    Default is HIGH (real AI prose is often produced at high reasoning); sweep down
    for experiments.
    """
    from stylebot import config

    base_url = base_url or config.get_key("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    return openai_generator(
        model=model,
        strategy=strategy,
        system=system,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key or config.require_key("OPENROUTER_API_KEY"),
        base_url=base_url,
        name=f"openrouter/{model}",
        extra_body=_reasoning_extra_body(model, reasoning_effort),
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Synthesis — assign generators to targets, generate, append schema-valid pairs
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _synth_key(
    generator_name: str,
    target_text: str,
    context: str = "",
    strategy: str = DEFAULT_STRATEGY,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    prompt_id: str = "",
) -> str:
    """Stable id for one synthetic pair — the resume/dedup key.

    The key spans every experimental axis whose variants we want to *coexist*
    rather than shadow each other on resume: generator, slop strategy, reasoning
    effort, prompt id (content hash of the system prompt), heading context, and the
    target text. Sampling params (temperature/top_p) are deliberately NOT in the key
    — they're recorded covariates, not dedup axes (continuous; would explode the key
    space). Promote them here only if swept as a primary arm.
    """
    h = hashlib.sha256()
    for part in (generator_name, strategy, reasoning_effort, prompt_id, context, target_text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
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


def _capture_id(source: str, generator_name: str, strategy: str = DEFAULT_STRATEGY) -> str:
    """Group a post's chunks from one generator+strategy under one capture id."""
    h = hashlib.sha256(f"{source}\x00{generator_name}\x00{strategy}".encode("utf-8"))
    return h.hexdigest()[:8]


def existing_synth_keys(pairs_path: Path | str) -> set[str]:
    """Read the `meta.synth_key`s already present in a `pairs.jsonl`.

    Uses the shared tolerant reader `stylebot.pairs.iter_pairs` (UTF-8, blank /
    undecodable lines skipped, missing file → empty), so resume and the schema
    contract stay on one JSONL reader.
    """
    return {
        key
        for rec in iter_pairs(pairs_path)
        if (key := (rec.get("meta") or {}).get("synth_key"))
    }


def _build_record(
    *,
    slop: str,
    target: Target,
    generator_name: str,
    synth_key: str,
    strategy: str = DEFAULT_STRATEGY,
    context: str = "",
    extra_tags: Sequence[str] = (),
    gen_meta: dict | None = None,
) -> dict:
    meta = {
        "source": target.source,
        "captured_at": _now_iso(),
        "capture_id": _capture_id(target.source, generator_name, strategy),
        "chunk_index": target.chunk_index,
        "chunk_total": target.chunk_total,
        "before_chars": len(slop),  # body lengths (the transform), excluding the heading prefix
        "after_chars": len(target.text),
        "synthetic": True,
        "generator": generator_name,
        "slop_strategy": strategy,
        "synth_key": synth_key,
        "tags": ["synthetic", "paraphrase", *extra_tags],
    }
    if context:
        # Shared contract: identical heading prefix on both sides (see
        # stylebot.pairs.build_pair_content); the styler restyles the body
        # conditioned on, but never rewriting, the heading.
        meta["context"] = context
        meta["context_mode"] = "immediate"
    if gen_meta:
        # The per-call generation covariates (model, reasoning_effort, sampling,
        # token usage, prompt id/version). Synthetic-only; Phase-1 real pairs have
        # no `gen` (see _plans plan: real pairs are the falsy-`synthetic` stratum).
        meta["gen"] = gen_meta
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
    generators: Sequence[Generator],
    *,
    per_generator: bool,
    context_dropout: float = 0.0,
) -> list[tuple[Target, Generator, str, str]]:
    """Pair targets with generators → ``(target, generator, synth_key, context)``.

    Default (rotate, `per_generator=False`): round-robin — target *i* goes to
    generator *i % n*. Cheap, and across the corpus yields ≥2 generators.
    `per_generator=True`: every target × every generator (n× the pairs/cost).
    `context` is the effective heading context after dropout; the `synth_key`
    incorporates the generator name, its slop strategy, and the context, so
    toggling any of them regenerates rather than colliding on resume.
    """
    if not generators:
        return []
    out: list[tuple[Target, Generator, str, str]] = []

    def assign_one(t: Target, gen: Generator) -> None:
        ctx = _effective_context(t, context_dropout)
        key = _synth_key(gen.name, t.text, ctx, gen.strategy, gen.reasoning_effort, gen.prompt_id)
        out.append((t, gen, key, ctx))

    if per_generator:
        for t in targets:
            for gen in generators:
                assign_one(t, gen)
    else:
        n = len(generators)
        for i, t in enumerate(targets):
            assign_one(t, generators[i % n])
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
    on_error: Callable[[str, str], None] | None = None,
) -> SynthResult:
    """Generate synthetic pairs and append them to `data_dir/pairs.jsonl`.

    Idempotent and resumable: each pair carries a `meta.synth_key`
    (`hash(generator, strategy, reasoning_effort, prompt_id, context, target)`);
    assignments whose key is already in the file are skipped, so re-running never
    duplicates and a crashed run resumes where it stopped (records are appended
    one-per-line, flushed as they go).

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

    assignments = _assign(targets, generators, per_generator=per_generator, context_dropout=context_dropout)
    result = SynthResult(planned=len(assignments))

    seen = existing_synth_keys(pairs_path)
    todo = [a for a in assignments if a[2] not in seen]
    result.skipped_existing = len(assignments) - len(todo)

    if dry_run:
        for _, gen, _, _ in todo:
            result.per_generator[gen.name] = result.per_generator.get(gen.name, 0) + 1
        return result

    data_dir.mkdir(parents=True, exist_ok=True)
    written_keys: set[str] = set()
    total = len(todo)
    with pairs_path.open("a", encoding="utf-8") as fp:
        for idx, (target, gen, key, context) in enumerate(todo, start=1):
            if on_progress is not None:
                on_progress(idx, total)
            if key in written_keys:  # guard within-run dup (same target listed twice)
                continue
            try:
                raw = gen(target.text)
            except Exception as exc:  # API error etc. — record and continue
                msg = f"{type(exc).__name__}: {exc}"
                result.errors.append((key, msg))
                if on_error is not None:
                    on_error(key, msg)
                continue
            slop, gen_meta = _normalize_gen_output(raw)
            if not slop.strip():
                result.errors.append((key, "generator returned empty output"))
                if on_error is not None:
                    on_error(key, "generator returned empty output")
                continue
            record = _build_record(
                slop=slop,
                target=target,
                generator_name=gen.name,
                synth_key=key,
                strategy=gen.strategy,
                context=context,
                extra_tags=extra_tags,
                gen_meta=gen_meta,
            )
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()
            written_keys.add(key)
            result.written += 1
            result.per_generator[gen.name] = result.per_generator.get(gen.name, 0) + 1

    return result
