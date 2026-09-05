"""Phase 2 — synthetic pair generation.

Library-first: `synthesize_pairs` is a typed function over explicit paths/params;
`ai-style synth` (`stylebot.bin.ai_style`) is a thin CLI wrapper.

Method (see `_plans/decisions.md` D9, D11-D13): human-authored prose is the
**target** (`messages[2]`); an LLM paraphrases it into "more polished" prose, the
**slop source** (`messages[1]`), and the styler learns to undo that transform.
Every generator call is appended to `draws_dir` as one draw (D84); a reader sees
the accepted draws' records, in the Phase-1 schema chunked the same way, so real
and synthetic pairs are mixable. Synthetic records add `meta.synthetic`,
`meta.generator`, `meta.slop_strategy`, `meta.synth_key` and `meta.tags`.
Generators and slop prompts are injected knobs, both folded into `synth_key`;
rotate ≥2 generators so the styler undoes AI writing broadly, not one model's
tics.
"""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import inspect
import json
import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from stylebot import config, corpus, draws
from stylebot.ai_core import style_system
from stylebot.generators import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_STRATEGY,
    Generator,
    _normalize_gen_output,
)
from stylebot.infer import anchor_delta
from stylebot.jsonl import append_jsonl, dumps_line, iter_jsonl
from stylebot.pairs import PAIR_SCHEMA_VERSION, build_pair_content
from stylebot.provenance import now_iso, pair_provenance, repo_provenance
from stylebot.targets import Target

logger = logging.getLogger(__name__)

# --- re-exports ---------------------------------------------------------------
# Names reached by ATTRIBUTE lookup (`synth.openrouter_generator`) by
# `synth_cli`, `bin/ai_style.py` and livingthing — the indirection that lets a
# test monkeypatch a factory where production resolves it. Anything not needing
# that imports from `stylebot.generators` / `.targets` directly. Keep this list
# minimal and check before adding: nothing prunes it on its own.
from stylebot.generators import (  # noqa: E402, F401
    DEFAULT_GEN_TIMEOUT,
    DEFAULT_SLOP_MAX_TOKENS,
    STRATEGIES,
    GenOutput,
    ResolvedStrategy,
    openrouter_context_windows,
    openrouter_generator,
    prompt_id_of,
    resolve_strategy,
)
from stylebot.targets import iter_targets  # noqa: E402, F401


def transform_similarity(a: str, b: str) -> float:
    """Character-level copying ratio between two prose bodies, in [0, 1].

    1.0 = verbatim copy, ~0 = no shared runs. A *copying* measure, not a style
    measure — sentence reordering counts as a transform. `autojunk=False`: on
    prose the default popularity heuristic junks the space character, which
    makes ratios erratic.
    """
    na = " ".join(a.split())
    nb = " ".join(b.split())
    if not na and not nb:
        return 1.0
    return round(difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio(), 3)


# Multi-turn sessions never push the prompt past this fraction of the model's
# context window: end-of-window truncation and degraded attention are failure
# modes, not covariates worth sampling.
SESSION_WINDOW_FILL_CAP = 0.8
# Extra draws after a TRANSIENT upstream failure (502/429/timeout), and the
# first backoff, doubling thereafter. Distinct from `gate_retries`, which
# redraws a draw that ARRIVED and was rejected on quality; the counts must not
# be pooled. 2, and a short backoff, because the fix that works is re-pinning
# the provider rather than waiting — the SDK already retried the same request.
DEFAULT_GEN_RETRIES = 2
_RETRY_BACKOFF_SECS = 2.0
# Default per-session prompt-token budget, same for every model (cost grows
# ~quadratically with it). Per-model overrides go on Generator.session_budget.
DEFAULT_SESSION_MAX_TOKENS = 32_000


# ---------------------------------------------------------------------------
# Synthesis — assign generators to targets, generate, append schema-valid pairs
# ---------------------------------------------------------------------------


def _synth_key(
    generator_name: str,
    target_text: str,
    context: str = "",
    strategy: str = DEFAULT_STRATEGY,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    prompt_id: str = "",
    replicate: str = "",
) -> str:
    """Stable id for one synthetic pair — the resume/dedup cell identity (D11).

    **Key on content, record circumstance.** Keyed: generator, strategy,
    reasoning effort, prompt id, heading context, target text, `replicate`. NOT
    keyed: sampling params, anything positional, and session membership above
    all (D13).

    **Also not keyed: the chunk budget.** Two budgets that cut a passage to
    byte-identical text are ONE cell, stamped `meta.merge_max_chars` by whichever
    batch reached it first — so that field records provenance, not a design cell.
    Keying on it would mint a duplicate pair per budget for identical text
    (D82).
    """
    h = hashlib.sha256()
    for part in (generator_name, strategy, reasoning_effort, prompt_id, context, target_text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    if replicate:
        h.update(replicate.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _effective_context(target: Target, context_dropout: float) -> str:
    """The context to use for a target, applying deterministic dropout.

    Keyed on the body hash, so resume is stable. Leaving some pairs heading-less
    stops the styler *requiring* a heading at inference.
    """
    if not target.context or context_dropout <= 0:
        return target.context
    bucket = int(hashlib.sha256(target.text.encode("utf-8")).hexdigest(), 16) % 1000
    return "" if bucket < context_dropout * 1000 else target.context


def _capture_id(source: str, generator_name: str, strategy: str = DEFAULT_STRATEGY) -> str:
    """Group a post's chunks from one generator+strategy under one capture id."""
    h = hashlib.sha256(f"{source}\x00{generator_name}\x00{strategy}".encode())
    return h.hexdigest()[:8]


# Markdown inline-link destination, one level of balanced parens deep:
# `](dest)` / `](dest "title")`. Reference definitions and autolinks are left
# alone — only the inline form drifted.
_LINK_DEST_RE = re.compile(r"\]\((?:[^()\\]|\\.|\([^()]*\))*\)")

# Curly punctuation the blog's quote-curling pass introduced, folded back to the
# ASCII the corpus was captured with.
_QUOTE_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def coverage_key(body: str) -> str:
    """The key `skip_covered` compares target bodies on, cosmetic drift removed.

    The corpus is a frozen record and does NOT track current prose, so a post
    edited after its pair was generated re-reads as uncovered and gets paid for
    twice; most of that drift is mechanical and not a difference the styler
    could learn (D53).

    Three folds, each a difference the style prompt makes invisible: inline link
    **destinations**, curly **quotes**, and **whitespace** runs. Nothing else —
    every extra rule risks collapsing two distinct passages onto one key, and
    that failure is silent. NOT `_synth_key`, which stays keyed on raw text.
    """
    return " ".join(_LINK_DEST_RE.sub("]()", body).translate(_QUOTE_FOLD).split())


def drawn_records(draws_dir: Path | str) -> Iterator[dict]:
    """The records a draws dir holds — the lines `DrawWriter` keys on (D84)."""
    for draw in draws.iter_draws(draws_dir):
        if draw.record is not None:
            yield draw.record


def covered_target_keys(records: Iterable[dict]) -> set[str]:
    """The `coverage_key`s that already have >=1 pair, over any record stream.

    Context-agnostic: the assistant side of every record is
    `build_pair_content(meta.context, body)`, so the body is recovered by
    stripping the recorded prefix. Backs `skip_covered`, and returns normalised
    keys rather than raw bodies — see `coverage_key`. A run passes
    `drawn_records`; `pairs.iter_pairs` reads a corpus file instead.
    """
    bodies: set[str] = set()
    for rec in records:
        msgs = rec.get("messages") or []
        if len(msgs) < 3 or not isinstance(msgs[2], dict):
            continue
        content = msgs[2].get("content")
        if not isinstance(content, str):
            continue
        context = ((rec.get("meta") or {}).get("context") or "").strip()
        prefix = f"{context}\n\n"
        if context and content.startswith(prefix):
            bodies.add(coverage_key(content[len(prefix) :]))
        else:
            bodies.add(coverage_key(content))
    return bodies


def existing_synth_keys(records: Iterable[dict]) -> set[str]:
    """The `meta.synth_key`s a record stream already holds — `drawn_records` for
    a run, `pairs.iter_pairs` for a corpus file; records without one are skipped."""
    return {key for rec in records if (key := (rec.get("meta") or {}).get("synth_key"))}


def _build_record(
    *,
    slop: str,
    target: Target,
    generator_name: str,
    synth_key: str,
    system: str,
    strategy: str = DEFAULT_STRATEGY,
    context: str = "",
    extra_tags: Sequence[str] = (),
    gen_meta: dict | None = None,
    repo_prov: Mapping | None = None,
) -> dict:
    meta = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "source": target.source,
        "captured_at": now_iso(),
        "capture_id": _capture_id(target.source, generator_name, strategy),
        "chunk_index": target.chunk_index,
        "chunk_total": target.chunk_total,
        "before_chars": len(slop),  # body lengths (the transform), excluding the heading prefix
        "after_chars": len(target.text),
        "transform_sim": transform_similarity(slop, target.text),
        # Chunk-scale facet: merge budget (None = unmerged) and cut (None = section).
        "merge_max_chars": target.merge_max_chars,
        "chunk_mode": target.chunk_mode,
        "synthetic": True,
        "generator": generator_name,
        "slop_strategy": strategy,
        "synth_key": synth_key,
        "tags": ["synthetic", "paraphrase", *extra_tags],
        # Which trees and which cut produced this pair. The chunk policy arrives
        # ON THE TARGET (`iter_targets` digests the `ChunkPolicy` it was handed),
        # so the stamp tracks the composed policy without being told about it.
        # `synth_key` folds none of this in: resume must survive a tree commit.
        "provenance": pair_provenance(
            repo_prov,
            policy_digest=target.policy_digest,
            source_automation=target.automation,
        ),
    }
    if context:
        # Identical heading prefix on both sides: conditioned on, never rewritten.
        meta["context"] = context
        meta["context_mode"] = "immediate"
    if gen_meta:
        meta["gen"] = gen_meta  # synthetic-only; real pairs have no `gen`
    return {
        "messages": [
            {"role": "system", "content": system},
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
    skipped_covered: int = 0  # targets dropped by skip_covered (any-config coverage)
    # Hygiene-gate drops: paid for, never written. A dropped cell stays un-keyed,
    # so a re-run retries it and a consistently failing model burns spend every
    # run until it leaves the roster — these counts are the signal to act on.
    skipped_degenerate: int = 0  # identity: transform_sim > max_transform_sim
    skipped_inflated: int = 0  # inflation: length ratio > max_length_ratio
    skipped_invented_anchors: int = 0  # anchor edits: the slop lost or un-escaped one
    # Redraw accounting (`gate_retries`): a gated draw is paid for either way, so
    # these are the cost of NOT letting a gate quietly bias coverage.
    gate_redraws: int = 0  # extra paid draws spent clearing a gate
    redrawn_cells: int = 0  # cells that needed at least one redraw
    redraws_recovered: int = 0  # ... of those, ones that went on to be written
    # Rejected draws per generator: against `per_generator`, an arm's rejection rate (D32).
    drops_by_generator: dict[str, int] = field(default_factory=dict)
    # Transient failures survived, and cells lost to one anyway: many retries that
    # still error is a sick provider, not a sick config (`lost_draws.jsonl`, kind="error").
    gen_retries_spent: int = 0
    errors_by_generator: dict[str, int] = field(default_factory=dict)
    planned: int = 0  # (target, generator) assignments before dedup
    planned_sessions: int = 0  # sessions holding >=1 not-yet-generated turn
    errors: list[tuple[str, str]] = field(default_factory=list)  # (synth_key, message)
    per_generator: dict[str, int] = field(default_factory=dict)
    # Reflow accounting: turns a session couldn't run are respun, never discarded.
    budget_bound_sessions: int = 0  # sessions ended by the token budget
    reflow_sessions: int = 0  # fresh sessions spawned to carry leftovers
    reflowed_turns: int = 0  # leftover turns respun (a turn respun twice counts twice)
    # Per-model economics keyed by gen.name. Cost is billed credits, not an
    # estimate; zeros for generators that report no usage.
    model_stats: dict[str, dict] = field(default_factory=dict)

    def add_gen_stats(self, name: str, gen_meta: Mapping) -> None:
        ms = self.model_stats.setdefault(
            name,
            {
                "pairs": 0,
                "cost": 0.0,
                "gen_seconds": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "cached_tokens": 0,
            },
        )
        ms["pairs"] += 1
        ms["cost"] += gen_meta.get("cost") or 0.0
        ms["gen_seconds"] += gen_meta.get("gen_seconds") or 0.0
        for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"):
            ms[k] += gen_meta.get(k) or 0

    @property
    def total_cost(self) -> float:
        return sum(ms["cost"] for ms in self.model_stats.values())


@dataclass
class _Turn:
    target: Target
    key: str
    context: str
    index: int  # 1-based position within the session


@dataclass
class _Session:
    generator: Generator
    session_id: str
    turns: list[_Turn]


def _arm_identity(gen: Generator) -> str:
    """The identity assignment hashes on — the ARM, not the model.

    The rotation is models × strategies × efforts, so many arms share one
    `Generator.name`; keying on `name` would score them identically and let one
    win every target while its siblings got no work at all.
    """
    return f"{gen.name}\x00{gen.strategy}\x00{gen.reasoning_effort}\x00{gen.prompt_id}"


def _assign(
    targets: Sequence[Target],
    generators: Sequence[Generator],
    *,
    per_generator: bool,
    context_dropout: float = 0.0,
    assign_seed: str = "",
) -> list[tuple[Target, Generator, str]]:
    """Pair targets with generators → ``(target, generator, effective_context)``.

    Default (rotate, `per_generator=False`): **rendezvous assignment** — score
    every arm as ``sha256(salt, arm_identity, text)``, take the argmax, break
    ties on arm identity (D12). Balance is multinomial, not exact;
    `per_generator=True` crosses every target x every generator instead, at n x
    the pairs and cost. `assign_seed` re-randomizes the whole design, which
    changes which arm generated each target, so those assignments regenerate.
    """
    if not generators:
        return []
    out: list[tuple[Target, Generator, str]] = []
    if per_generator:
        for t in targets:
            for gen in generators:
                out.append((t, gen, _effective_context(t, context_dropout)))
    else:
        # Hoisted: identities don't vary per target, and the inner loop runs
        # len(targets) × len(generators) sha256s.
        arms = [(_arm_identity(gen), gen) for gen in generators]
        for t in targets:
            best_score = b""
            best_ident = ""
            best_gen = arms[0][1]
            for ident, gen in arms:
                score = hashlib.sha256(f"{assign_seed}\x00{ident}\x00{t.text}".encode()).digest()
                # Tiebreak on identity, so a collision can't resolve on list
                # position. Arms sharing an identity are already one cell.
                if (score, ident) > (best_score, best_ident):
                    best_score, best_ident, best_gen = score, ident, gen
            out.append((t, best_gen, _effective_context(t, context_dropout)))
    return out


def _session_id(gen: Generator, items: Sequence[tuple[Target, str]]) -> str:
    """Grouping label for one session's actual composition — a covariate, NOT a key.

    Recomputed at reflow time, so it always names what the session contained.
    """
    sid_h = hashlib.sha256(f"{gen.name}\x00{gen.strategy}\x00{gen.prompt_id}".encode())
    for target, ctx in items:
        sid_h.update(b"\x00")
        sid_h.update(ctx.encode("utf-8"))
        sid_h.update(target.text.encode("utf-8"))
    return sid_h.hexdigest()[:8]


def _plan_sessions(
    assignments: Sequence[tuple[Target, Generator, str, str]],
    *,
    session_turns: int = 1,
) -> list[_Session]:
    """Chunk each generator's (target, ctx, key) work items into sessions.

    Takes only *missing* cells — the caller filters against the corpus first —
    so a session never contains already-generated turns; assignment order is
    preserved. Sessions do not touch keys, so a cell keys identically whether it
    lands in a session, reflows into a fresh one, or runs stateless.
    `session_turns` is only a backstop; the token budget is the depth control.
    """
    # Group by generator IDENTITY, not name: a rotation may carry the same model
    # under several strategies, and name-keying would silently merge them.
    per_gen: dict[int, tuple[Generator, list[tuple[Target, str, str]]]] = {}
    for target, gen, ctx, key in assignments:
        per_gen.setdefault(id(gen), (gen, []))[1].append((target, ctx, key))

    sessions: list[_Session] = []
    for gen, items in per_gen.values():
        for start in range(0, len(items), max(1, session_turns)):
            chunk = items[start : start + max(1, session_turns)]
            sid = _session_id(gen, [(t, ctx) for t, ctx, _ in chunk]) if session_turns > 1 else ""
            turns = [
                _Turn(target=target, key=key, context=ctx, index=idx)
                for idx, (target, ctx, key) in enumerate(chunk, start=1)
            ]
            sessions.append(_Session(generator=gen, session_id=sid, turns=turns))
    return sessions


def _respin_session(gen: Generator, leftover: Sequence[_Turn]) -> _Session:
    """A fresh session holding the turns an earlier one couldn't run.

    Fresh history (window position restarts), fresh session_id from the actual
    composition, turn indices renumbered from 1.
    """
    sid = _session_id(gen, [(t.target, t.context) for t in leftover])
    turns = [
        _Turn(target=t.target, key=t.key, context=t.context, index=i)
        for i, t in enumerate(leftover, start=1)
    ]
    return _Session(generator=gen, session_id=sid, turns=turns)


def _record_prompts(data_dir: Path, generators: Sequence[Generator]) -> None:
    """Archive each run's system prompts to `<data-dir>/prompts.jsonl` (id-deduped).

    `prompt_id` is a content hash, opaque on its own; this sidecar keeps the
    prompt text next to the corpus it produced, so resolving an id is a grep —
    including for custom `--slop-system-file` prompts and superseded versions.
    """
    path = data_dir / "prompts.jsonl"
    seen = {rec.get("prompt_id", "") for rec in iter_jsonl(path)}
    with path.open("a", encoding="utf-8") as fp:
        for g in generators:
            if g.prompt_id and g.prompt_system and g.prompt_id not in seen:
                entry = {
                    "prompt_id": g.prompt_id,
                    "label": g.strategy,
                    "version": g.prompt_version,
                    "system": g.prompt_system,
                }
                fp.write(dumps_line(entry) + "\n")
                seen.add(g.prompt_id)


def _exchange(user_text: str, assistant_text: str) -> list[dict]:
    """One (user → assistant) session exchange, as chat messages."""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


@dataclass
class _RunState:
    """Shared mutable state for one `synthesize_pairs` run.

    Every session's coroutine shares this object. Single-threaded asyncio —
    mutations interleave only between awaits, so no locks.
    """

    result: SynthResult
    windows: dict[str, int]  # generator name -> context window (live sessions)
    session_max_tokens: int | None
    data_dir: Path
    extra_tags: Sequence[str]
    total: int  # pairs this run will attempt (for progress)
    writer: DrawWriter  # the one handle on `draws/` for this run's duration
    system_prompt: str  # `messages[0]` of every draw this run lands (D85)
    max_transform_sim: float | None = None  # identity gate (None = off)
    max_length_ratio: float | None = None  # inflation gate (None = off)
    drop_invented_anchors: bool = True  # fabrication gate (see `_gate_verdict`)
    gate_retries: int = 0  # extra draws to spend clearing a gate before dropping
    gen_retries: int = DEFAULT_GEN_RETRIES  # extra draws after a TRANSIENT upstream failure
    replicate: str = ""  # deliberate-resample label, keyed AND recorded
    done: int = 0
    on_progress: Callable[[int, int, SynthResult], None] | None = None
    on_error: Callable[[str, str], None] | None = None
    repo_prov: Mapping | None = None  # tree provenance, resolved once per run


def synth_key_of(record: Mapping | None) -> str:
    """A draw's cell key, or "" for one that renders nothing and stays redrawable."""
    return ((record or {}).get("meta") or {}).get("synth_key") or ""


class DrawWriter:
    """The one append handle on a `draws/`, which will not duplicate a cell.

    Every draw a run makes goes through it, routed by `draws.draw_shard_of` so
    the run leaves every other shard byte-identical (D50); handles are opened
    lazily and kept, one per shard touched.

    Before each append the writer catches up on whatever another run appended
    since it last looked — a tail read from a remembered offset over *every*
    shard, since re-reading per record would be quadratic (D36). No lock and
    no inode check: nothing rewrites a corpus (D84), so a handle never goes
    stale and a run is the only writer of its shards.
    """

    def __init__(self, draws_dir: Path, seen: set[str]) -> None:
        self._path = Path(draws_dir)
        self._seen = seen
        self._offsets: dict[Path, int] = {}
        self._handles: dict[Path, object] = {}

    def __enter__(self) -> DrawWriter:
        self._path.mkdir(parents=True, exist_ok=True)
        for shard in corpus.shard_paths(self._path):
            self._offsets[shard] = shard.stat().st_size if shard.exists() else 0
        return self

    def __exit__(self, *exc) -> None:
        for fp in self._handles.values():
            fp.close()
        self._handles.clear()

    def _handle(self, shard: Path):
        """One shard's append handle, opened on first use and kept."""
        fp = self._handles.get(shard)
        if fp is None:
            # BINARY "ab+", not text mode, for two reasons that both bite
            # silently: a text handle raises on `tell()` after line iteration,
            # and its `seek` takes an opaque cookie rather than a byte offset.
            # Encoding one JSON line by hand is the cheaper half of that trade.
            fp = shard.open("ab+")
            self._handles[shard] = fp
            self._offsets.setdefault(shard, 0)
        return fp

    def _catch_up(self) -> None:
        """Fold every synth_key appended by anyone since our last look into `seen`.

        Across every shard, not just the one about to be written: a cell's key
        does not know which month it landed in, so a per-shard catch-up would
        answer "not seen" for a key another writer had just put in a neighbour.
        The read is a tail from a remembered byte offset, so the cost is what was
        appended since the last write, not the 38 MB corpus.
        """
        for shard in corpus.shard_paths(self._path):
            size = shard.stat().st_size if shard.exists() else 0
            start = self._offsets.get(shard, 0)
            if size <= start:
                self._offsets[shard] = size
                continue
            with shard.open("rb") as fp:
                fp.seek(start)
                tail = fp.read()
                self._offsets[shard] = fp.tell()
            for raw in tail.split(b"\n"):
                if not raw.strip():
                    continue
                try:
                    rec = json.loads(raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                key = synth_key_of(rec.get("record")) if isinstance(rec, dict) else ""
                if key:
                    self._seen.add(key)

    def write(self, draw: draws.Draw) -> bool:
        """Append `draw`, dropping its record if the corpus already has its cell:
        it was paid for either way, but only one draw per cell renders. True = ours."""
        self._catch_up()
        key = synth_key_of(draw.record)
        fresh = not (key and key in self._seen)
        if not fresh:
            draw.record = None
        shard = self._path / draws.draw_shard_of(draw)
        fp = self._handle(shard)
        fp.seek(0, 2)
        fp.write((dumps_line(draw.to_json()) + "\n").encode("utf-8"))
        fp.flush()
        self._offsets[shard] = fp.tell()
        if key:
            self._seen.add(key)
        return fresh


def _session_token_cap(
    gen: Generator, window: int | None, session_max_tokens: int | None
) -> int | None:
    """One session's prompt-token budget: the generator's own budget (else the
    global default), further capped at `SESSION_WINDOW_FILL_CAP` × window."""
    caps = [gen.session_budget or session_max_tokens]
    if window:
        caps.append(int(window * SESSION_WINDOW_FILL_CAP))
    return min((c for c in caps if c), default=None)


def _estimate_next_prompt(
    history: Sequence[dict], last_prompt: int, last_visible: int, next_text: str
) -> int:
    """Estimate the next turn's prompt tokens: last measured prompt + what the
    last reply added + the next passage (~4 chars/token), falling back to a pure
    char estimate before any live turn has reported usage."""
    if last_prompt:
        return last_prompt + last_visible + len(next_text) // 4
    return (sum(len(m["content"]) for m in history) + len(next_text)) // 4


def _gate_verdict(slop: str, target_text: str, st: _RunState) -> tuple[str, str] | None:
    """`(counter_name, reason)` if a hygiene gate rejects this draw, else None.

    Pure and record-free so it can run BEFORE the per-pair bookkeeping (reasoning
    sidecar, session covariates), which a redrawn sample must not repeat.

    **Every reason describes what the GENERATOR did**, never what the pair looks
    like afterwards: synthesis is a reverse edit, so a defect in the generated
    slop surfaces as an apparent property of Dan's prose. "target invents 3
    anchors" is backwards — the generator DELETED three links while producing
    slop, and reversing that edit is what would teach invention.
    """
    if st.max_transform_sim is not None:
        sim = transform_similarity(slop, target_text)
        if sim > st.max_transform_sim:
            # Identity output: the model returned (nearly) the input.
            return (
                "skipped_degenerate",
                f"generator returned (nearly) its input, transform_sim={sim:.3f} "
                f"> {st.max_transform_sim}",
            )
    if st.max_length_ratio is not None and target_text:
        ratio = len(slop) / len(target_text)
        if ratio > st.max_length_ratio:
            # Inflated output: slop several times the target length, which would
            # teach manic deletion.
            return (
                "skipped_inflated",
                f"generator wrote {ratio:.2f}x the target length, over {st.max_length_ratio}",
            )
    if st.drop_invented_anchors:
        lost, invented = anchor_delta(slop, target_text)
        # Reversed, a deleted link is a lesson in inventing one, and an
        # un-escaped `@key` (the target's `\\@key`) in breaking one (D32, C13).
        unescaped = [a for a in lost if "\\" + a in target_text]
        if invented or unescaped:
            uniq = sorted(set(invented or unescaped))
            shown = ", ".join(uniq[:2]) + ("…" if len(uniq) > 2 else "")
            what = "deleted" if invented else "un-escaped"
            return (
                "skipped_invented_anchors",
                f"generator {what} {len(uniq)} anchor(s) the target keeps ({shown}); "
                "reversed, this pair would teach the styler to invent or break them",
            )
    return None


class _EmptyDraw(RuntimeError):
    """A provider answered a valid request with no content.

    Its own type because it is a *transient upstream* failure wearing the
    clothes of a programming error: retrying against a different provider
    usually works, and `_is_transient` has to be able to tell it apart from a
    genuine `RuntimeError`.
    """


# Upstream statuses worth another draw. Everything else (400/401/403/404 — a
# malformed request, a bad key, a model that does not exist) is our bug or our
# config, and retrying it just burns the budget more slowly.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_NAMES = frozenset(
    {"APIConnectionError", "APITimeoutError", "ConnectionError", "TimeoutError"}
)


def _is_transient(exc: BaseException) -> bool:
    """Is this failure worth redrawing, or is it ours to fix?

    Duck-typed on `status_code` rather than catching `openai.APIStatusError`:
    the SDK is a lazy dependency here, and generators are an injected seam that
    any backend may implement.
    """
    if isinstance(exc, _EmptyDraw):
        return True
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUS
    return type(exc).__name__ in _TRANSIENT_NAMES


def _record_lost_draw(
    data_dir: Path,
    turn: _Turn,
    gen: Generator,
    gen_meta: dict,
    *,
    kind: str,
    gate: str | None = None,
    reason: str,
    attempt: int,
    redrawn: bool,
) -> None:
    """Append one draw that produced no pair to `<data-dir>/lost_draws.jsonl`.

    One file for `kind="gate"` and `kind="error"`: an arm's rejection rate and its
    discarded spend (D32). `attempt` is the draw's index within its cell's redraws.
    """
    append_jsonl(
        data_dir / "lost_draws.jsonl",
        {
            "at": now_iso(),
            "synth_key": turn.key,
            "source": turn.target.source,
            "generator": gen.name,
            "slop_strategy": gen.strategy,
            "reasoning_effort": gen.reasoning_effort,
            "kind": kind,
            "gate": gate,
            "reason": reason,
            "attempt": attempt,
            "redrawn": redrawn,
            "cost": gen_meta.get("cost"),
            "prompt_tokens": gen_meta.get("prompt_tokens"),
            "completion_tokens": gen_meta.get("completion_tokens"),
            "target_chars": len(turn.target.text),
        },
    )


def _pop_reasoning_to_sidecar(
    data_dir: Path, turn: _Turn, gen: Generator, gen_meta: dict, *, in_session: bool
) -> None:
    """Route a reasoning trace to the `reasoning.jsonl` sidecar (keyed to the pair).

    Traces are diagnostics, not corpus: `reasoning_text` is POPPED from
    `gen_meta`, so it never enters `pairs.jsonl`.
    """
    reasoning_text = gen_meta.pop("reasoning_text", None)
    if reasoning_text is None:
        return
    append_jsonl(
        data_dir / "reasoning.jsonl",
        {
            "synth_key": turn.key,
            "generator": gen.name,
            "slop_strategy": gen.strategy,
            "reasoning_effort": gen.reasoning_effort,
            "session_turn": turn.index if in_session else None,
            "reasoning_tokens": gen_meta.get("reasoning_tokens"),
            "reasoning": reasoning_text,
        },
    )


def _record_gate_drop(
    st: _RunState,
    turn: _Turn,
    gen: Generator,
    gen_meta: dict,
    slop: str,
    verdict: tuple[str, str],
    *,
    attempt: int,
    redrawn: bool,
) -> None:
    """Account for one gated draw — the draw itself, sidecar line, rejection count,
    spend — on BOTH gate paths, redraw and give-up: it is paid for either way."""
    gate, reason = verdict
    _record_draw(st, turn, gen, gen_meta, slop, f"gate:{gate}")
    _record_lost_draw(
        st.data_dir,
        turn,
        gen,
        gen_meta,
        kind="gate",
        gate=gate,
        reason=reason,
        attempt=attempt,
        redrawn=redrawn,
    )
    st.result.drops_by_generator[gen.name] = st.result.drops_by_generator.get(gen.name, 0) + 1
    st.result.add_gen_stats(gen.name, gen_meta)  # the discarded draw was paid for


def _record_draw(
    st: _RunState,
    turn: _Turn,
    gen: Generator,
    gen_meta: Mapping,
    raw_output: str | None,
    outcome: str,
    record: dict | None = None,
) -> bool:
    """Append one generator call's draw through the run's one writer, no lock;
    True when it kept its record, i.e. this run claimed the cell (D84)."""
    draw = draws.new_draw(
        gen,
        st.repo_prov,
        turn.target,
        turn.context,
        gen=gen_meta,
        raw_output=raw_output,
        outcome=outcome,
        record=record,
        replicate=st.replicate or 0,
    )
    return st.writer.write(draw)


async def _attempt(fn, text: str, history: Sequence[dict]) -> tuple[str, dict]:
    """One call to a generator callable, sync or async, normalized.

    `history` is a parameter, not a closed-over variable: a session mutates its
    history between turns, and passing it explicitly is what keeps the binding
    obvious rather than something the reader has to reason about.
    """
    out = fn(text, history=list(history) or None)
    if inspect.isawaitable(out):
        out = await out
    slop, gen_meta = _normalize_gen_output(out)
    if not slop.strip():
        raise _EmptyDraw("generator returned empty output")
    return slop, gen_meta


async def _draw(
    call, gen: Generator, turn: _Turn, history: Sequence[dict], st: _RunState
) -> tuple[str, dict, object]:
    """One paid draw, retried across providers. Raises once it gives up.

    Returns `(slop, gen_meta, call)`: a retry RE-PINS the provider (see
    `DEFAULT_GEN_RETRIES`), so the caller must keep using the callable that worked.
    """
    delay = _RETRY_BACKOFF_SECS
    for attempt in range(st.gen_retries + 1):
        try:
            slop, gen_meta = await _attempt(call, turn.target.text, history)
            return slop, gen_meta, call
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt == st.gen_retries or not _is_transient(exc):
                raise
            st.result.gen_retries_spent += 1
            logger.info(
                "transient %s from %s, redrawing (%d/%d): %s",
                type(exc).__name__,
                gen.name,
                attempt + 1,
                st.gen_retries,
                turn.key,
            )
            if gen.begin_session is not None:
                call = gen.begin_session()
            await asyncio.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover


async def _draw_past_gates(
    call, gen: Generator, turn: _Turn, history: Sequence[dict], st: _RunState
) -> tuple[str, dict, tuple[str, str] | None, int, object]:
    """Draw for one turn, redrawing while a hygiene gate rejects the sample (D32).

    Returns `(slop, gen_meta, verdict, redraws, call)`: `verdict` is None once a
    sample cleared, else the last rejection when the redraw budget ran out.
    Raises when `_draw` gives up, on a redraw as on the first draw: the caller's
    error path is the one place a lost draw is recorded (C14).
    """
    redraws = 0
    try:
        slop, gen_meta, call = await _draw(call, gen, turn, history, st)
        verdict = _gate_verdict(slop, turn.target.text, st)
        while verdict is not None and redraws < st.gate_retries:
            _record_gate_drop(st, turn, gen, gen_meta, slop, verdict, attempt=redraws, redrawn=True)
            redraws += 1
            logger.info(
                "gate redraw %d/%d: %s %s (%s)",
                redraws,
                st.gate_retries,
                gen.name,
                verdict[1],
                turn.key,
            )
            slop, gen_meta, call = await _draw(call, gen, turn, history, st)
            st.result.gate_redraws += 1  # a paid redraw: counted once it was drawn
            verdict = _gate_verdict(slop, turn.target.text, st)
    except Exception as exc:
        exc.redraws = redraws  # the error row's `attempt`, the index a gate row carries
        raise
    if redraws:
        st.result.redrawn_cells += 1
        if verdict is None:
            st.result.redraws_recovered += 1
    return slop, gen_meta, verdict, redraws, call


async def _run_session(sess: _Session, st: _RunState) -> list[_Turn]:
    """Run one session's turns in order, appending records as they complete.

    Returns the turns this session could NOT run, for the caller to reflow —
    never silently dropped: a binding token budget returns every remaining turn,
    an erroring turn returns the turns AFTER it (the failed cell retries next
    run). The budget check is skipped while history is empty, so every reflow
    writes a pair or sheds an errored turn, strictly shrinking the leftovers.
    """
    gen = sess.generator
    # Multi-turn sessions get a per-session callable when the generator offers
    # one (sticky provider routing state); stateless stays on the plain path.
    call = gen.begin_session() if (len(sess.turns) > 1 and gen.begin_session is not None) else gen
    window = st.windows.get(gen.name)
    cap = _session_token_cap(gen, window, st.session_max_tokens)
    history: list[dict] = []
    last_prompt = 0  # prompt_tokens reported for the last live turn
    last_visible = 0  # ~tokens the last reply added to history

    for pos, turn in enumerate(sess.turns):
        if (
            history
            and cap is not None
            and _estimate_next_prompt(history, last_prompt, last_visible, turn.target.text) > cap
        ):
            st.result.budget_bound_sessions += 1
            return list(sess.turns[pos:])

        try:
            slop, gen_meta, verdict, redraws, call = await _draw_past_gates(
                call, gen, turn, history, st
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A failed pair is never written, so its synth_key resolves to
            # nothing: the recorded error must carry its config to be attributable.
            who = f"{gen.name} strategy={gen.strategy} effort={gen.reasoning_effort}"
            if sess.session_id:
                who += f" turn={turn.index}"
            msg = f"[{who}] {type(exc).__name__}: {exc} (after {st.gen_retries + 1} attempt(s))"
            st.result.errors.append((turn.key, msg))
            if st.on_error is not None:
                st.on_error(turn.key, msg)
            _record_lost_draw(
                st.data_dir,
                turn,
                gen,
                {},
                kind="error",
                reason=f"{type(exc).__name__}: {exc}",
                attempt=getattr(exc, "redraws", 0),
                redrawn=False,
            )
            _record_draw(st, turn, gen, {}, None, f"error:{type(exc).__name__}")
            st.result.errors_by_generator[gen.name] = (
                st.result.errors_by_generator.get(gen.name, 0) + 1
            )
            # Only this cell is lost; the rest reflows: a failing sticky session keeps failing.
            return list(sess.turns[pos + 1 :])

        prompt_tokens = gen_meta.get("prompt_tokens")
        if sess.session_id:
            gen_meta.update(
                session_id=sess.session_id,
                session_turn=turn.index,
                context_window=window,
                window_fill=(
                    round(prompt_tokens / window, 4) if prompt_tokens and window else None
                ),
            )
        if st.replicate:
            gen_meta["replicate"] = st.replicate
        _pop_reasoning_to_sidecar(
            st.data_dir, turn, gen, gen_meta, in_session=bool(sess.session_id)
        )
        record = _build_record(
            slop=slop,
            target=turn.target,
            generator_name=gen.name,
            synth_key=turn.key,
            system=st.system_prompt,
            strategy=gen.strategy,
            context=turn.context,
            extra_tags=st.extra_tags,
            gen_meta=gen_meta,
            repo_prov=st.repo_prov,
        )
        if verdict is not None:
            # Count loudly, record the spend, write nothing — the cell stays
            # un-keyed, so a re-run retries it.
            counter, reason = verdict
            setattr(st.result, counter, getattr(st.result, counter) + 1)
            _record_gate_drop(
                st, turn, gen, gen_meta, slop, verdict, attempt=redraws, redrawn=False
            )
            # "gate dropped a draw", not "degenerate output": the anchor gate
            # catches a fluent draw that merely deleted a link (D32).
            logger.warning("gate dropped a draw: %s — %s (%s)", gen.name, reason, turn.key)
        elif _record_draw(st, turn, gen, gen_meta, slop, draws.ACCEPTED, record):
            st.result.written += 1
            st.result.per_generator[gen.name] = st.result.per_generator.get(gen.name, 0) + 1
            st.result.add_gen_stats(gen.name, gen_meta)
        else:
            # Another writer claimed this cell between plan and write (`DrawWriter`):
            # the draw is kept without its record, so a view renders theirs.
            st.result.skipped_existing += 1
            logger.info("cell already written by a concurrent run: %s", turn.key)
        st.done += 1
        if st.on_progress is not None:
            st.on_progress(st.done, st.total, st.result)
        history += _exchange(turn.target.text, slop)
        if prompt_tokens:
            last_prompt = prompt_tokens
        completion = gen_meta.get("completion_tokens") or 0
        reasoning = gen_meta.get("reasoning_tokens") or 0
        last_visible = max(completion - reasoning, len(slop) // 4)
    return []


async def _run_sessions(sessions: Sequence[_Session], st: _RunState, max_workers: int) -> None:
    """Run sessions concurrently, `max_workers` at a time, on one append handle.

    Leftover turns (budget bind, error) reflow into fresh sessions inside the
    same worker slot until the work list is empty, so the run finishes with every
    planned turn either written or individually errored.
    """
    sem = asyncio.Semaphore(max(1, max_workers))

    async def _gated(sess: _Session) -> None:
        async with sem:
            leftover = await _run_session(sess, st)
            while leftover:
                st.result.reflow_sessions += 1
                st.result.reflowed_turns += len(leftover)
                sess = _respin_session(sess.generator, leftover)
                leftover = await _run_session(sess, st)

    with st.writer:
        await asyncio.gather(*(_gated(s) for s in sessions))


def synthesize_pairs(
    targets: Sequence[Target],
    data_dir: Path | str,
    generators: Sequence[Generator],
    *,
    per_generator: bool = False,
    dry_run: bool = False,
    extra_tags: Sequence[str] = (),
    context_dropout: float = 0.0,
    on_progress: Callable[[int, int, SynthResult], None] | None = None,
    on_error: Callable[[str, str], None] | None = None,
    max_workers: int = 1,
    session_turns: int = 1,
    session_max_tokens: int | None = DEFAULT_SESSION_MAX_TOKENS,
    context_windows: Mapping[str, int] | None = None,
    assign_seed: str = "",
    replicate: str = "",
    skip_covered: bool = False,
    max_transform_sim: float | None = None,
    max_length_ratio: float | None = None,
    drop_invented_anchors: bool = True,
    gate_retries: int = 0,
    gen_retries: int = DEFAULT_GEN_RETRIES,
    blog_root: Path | str | None = None,
    draws_dir: Path | str | None = None,
    system_prompt: str | None = None,
) -> SynthResult:
    """Generate synthetic pairs and append them to `draws_dir` as draws.

    Idempotent and resumable: cells whose content-only `meta.synth_key` is
    already in the corpus are skipped, so re-running never duplicates and a
    crashed run resumes where it stopped (D11). Sessions run on one asyncio
    loop, `max_workers` at a time, no locks (D56); generators may be sync or
    async.

    **Sessions** (`session_turns > 1`): missing cells are chunked into live
    multi-turn sessions, fill is *measured* (`prompt_tokens`, `context_window`,
    `window_fill` in `meta.gen`), and **the token budget is the depth control**:
    a session ends at `min(session_budget or session_max_tokens,
    SESSION_WINDOW_FILL_CAP x window)` and the turns it could not run REFLOW
    into fresh sessions rather than being dropped.

    **Hygiene gates** count loudly, write nothing and leave the cell un-keyed so
    a re-run retries it (D32). Heading `context` is prepended verbatim to both
    sides, slop is generated from the body alone (D14). `dry_run` only plans.
    **Provenance** is resolved once per run (D36). Every call lands as one draw
    under `draws_dir` (default `<data-dir>/draws`) carrying `system_prompt` (the
    default author's when None) as `messages[0]`; no `pairs.jsonl` (D84).
    """
    data_dir = Path(data_dir)
    draws_dir = Path(draws_dir) if draws_dir is not None else config.draws_path(data_dir)

    skipped_covered = 0
    if skip_covered:
        # Coverage mode: a target with ANY existing pair is done. Compared on
        # `coverage_key`, so cosmetic prose drift since generation (rewritten
        # link destinations, curled quotes, a reflow) still reads as covered.
        # Applied before assignment; cell-level dedup below still governs the
        # rest, and it keys on RAW text.
        covered = covered_target_keys(drawn_records(draws_dir))
        kept = [t for t in targets if coverage_key(t.text) not in covered]
        skipped_covered = len(targets) - len(kept)
        targets = kept

    assignments = _assign(
        targets,
        generators,
        per_generator=per_generator,
        context_dropout=context_dropout,
        assign_seed=assign_seed,
    )
    result = SynthResult(planned=len(assignments), skipped_covered=skipped_covered)

    # Key everything, drop what the corpus already has plus in-run duplicates,
    # then plan sessions over only the missing cells.
    # `disk_keys` is what is ON DISK, and is NOT the planner's `seen` below: that
    # also holds every cell this run PLANS, which would refuse its own first write.
    disk_keys = existing_synth_keys(drawn_records(draws_dir))
    seen = set(disk_keys)
    missing: list[tuple[Target, Generator, str, str]] = []
    for target, gen, ctx in assignments:
        key = _synth_key(
            gen.name,
            target.text,
            ctx,
            gen.strategy,
            gen.reasoning_effort,
            gen.prompt_id,
            replicate=replicate,
        )
        if key in seen:
            continue
        seen.add(key)  # also dedupes identical cells within this run
        missing.append((target, gen, ctx, key))
    result.skipped_existing = len(assignments) - len(missing)

    sessions = _plan_sessions(missing, session_turns=session_turns)
    result.planned_sessions = len(sessions) if session_turns > 1 else 0

    if dry_run:
        for _, gen, _, _ in missing:
            result.per_generator[gen.name] = result.per_generator.get(gen.name, 0) + 1
        return result

    data_dir.mkdir(parents=True, exist_ok=True)
    _record_prompts(data_dir, generators)
    st = _RunState(
        result=result,
        windows=dict(context_windows or {}),
        session_max_tokens=session_max_tokens,
        data_dir=data_dir,
        extra_tags=extra_tags,
        total=len(missing),
        writer=DrawWriter(draws_dir, disk_keys),
        system_prompt=system_prompt or style_system(),
        max_transform_sim=max_transform_sim,
        max_length_ratio=max_length_ratio,
        drop_invented_anchors=drop_invented_anchors,
        gate_retries=gate_retries,
        gen_retries=gen_retries,
        replicate=replicate,
        on_progress=on_progress,
        on_error=on_error,
        repo_prov=repo_provenance(blog_root if blog_root is not None else data_dir),
    )
    asyncio.run(_run_sessions(sessions, st, max_workers))
    return result
