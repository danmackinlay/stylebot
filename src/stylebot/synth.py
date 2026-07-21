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

import asyncio
import difflib
import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.jsonl import iter_jsonl
from stylebot.pairs import build_pair_content, iter_pairs

# ---------------------------------------------------------------------------
# Strategies + generators — implemented in stylebot.generators; re-exported here
# ---------------------------------------------------------------------------

# Load-bearing re-exports, not courtesy: tests monkeypatch the factories ON
# THIS MODULE (the CLI kit calls synth.<factory> by attribute lookup), and
# livingthing + the CLI kit import the registry and defaults via stylebot.synth.
from stylebot.generators import (  # noqa: F401
    DEFAULT_GEN_TIMEOUT,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_SLOP_MAX_TOKENS,
    DEFAULT_STRATEGY,
    SLOP_SYSTEM,
    SLOP_SYSTEM_CASUAL,
    SLOP_SYSTEM_ENGAGING,
    SLOP_SYSTEM_MEASURED,
    STRATEGIES,
    GenOutput,
    Generator,
    ResolvedStrategy,
    SlopStrategy,
    _normalize_gen_output,
    local_generator,
    openai_generator,
    openrouter_context_windows,
    openrouter_generator,
    prompt_id_of,
    resolve_strategy,
)

def transform_similarity(a: str, b: str) -> float:
    """Character-level copying ratio between two prose bodies, in [0, 1].

    `difflib.SequenceMatcher` over whitespace-normalized text: 1.0 = verbatim
    copy, ~0 = no shared runs. Deliberately a *copying* measure, not a style
    measure — sentence reordering counts as a transform. Cheap, deterministic,
    stdlib-only, so it is baked into every synthetic pair as the frozen
    `meta.transform_sim` covariate; the *living* style-shift measure is the
    detector-score gap at eval time. Consumers filter by policy (e.g. the
    voice-classifier trainer drops near-identity pairs, which would be label
    noise). `autojunk=False`: on prose the default popularity heuristic junks
    the space character, which makes ratios erratic.
    """
    na = " ".join(a.split())
    nb = " ".join(b.split())
    if not na and not nb:
        return 1.0
    return round(difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio(), 3)


# Multi-turn sessions never push the prompt past this fraction of the model's
# context window — end-of-window behaviour (truncation, degraded attention) is
# a failure mode, not a covariate we want to sample.
SESSION_WINDOW_FILL_CAP = 0.8
# Global default per-session prompt-token budget (absolute, same for every
# model — cost grows ~quadratically with it). Per-model overrides go on
# Generator.session_budget.
DEFAULT_SESSION_MAX_TOKENS = 32_000


# ---------------------------------------------------------------------------
# Targets — implemented in stylebot.targets; re-exported here
# ---------------------------------------------------------------------------

# Re-exports are load-bearing, not courtesy: livingthing and the CLI kit
# import Target/iter_targets/the chunk constants via `stylebot.synth`.
from stylebot.targets import (  # noqa: E402, F401
    MAX_CHUNK_CHARS,
    MERGE_MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    ChunkPolicy,
    Target,
    iter_targets,
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
    session: str = "",
) -> str:
    """Stable id for one synthetic pair — the resume/dedup key.

    The key spans every experimental axis whose variants we want to *coexist*
    rather than shadow each other on resume: generator, slop strategy, reasoning
    effort, prompt id (content hash of the system prompt), heading context, and the
    target text. Sampling params (temperature/top_p) are deliberately NOT in the key
    — they're recorded covariates, not dedup axes (continuous; would explode the key
    space). Promote them here only if swept as a primary arm.

    `session` ("<session_id>:<turn>") is folded in ONLY for multi-turn session
    runs — session chunking depends on the whole target list, so folding it into
    stateless (session_turns=1) keys would churn every key whenever the blog
    gains a post; the corpus-building path must regenerate nothing on resume.
    """
    h = hashlib.sha256()
    for part in (generator_name, strategy, reasoning_effort, prompt_id, context, target_text):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    if session:
        h.update(session.encode("utf-8"))
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
        # Copying ratio slop<->target (1.0 = verbatim no-op). Frozen hygiene
        # covariate: near-identity pairs are label noise for the detector and
        # teach the styler to copy; consumers filter by threshold.
        "transform_sim": transform_similarity(slop, target.text),
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
    planned_sessions: int = 0  # sessions holding >=1 not-yet-generated turn
    errors: list[tuple[str, str]] = field(default_factory=list)  # (synth_key, message)
    per_generator: dict[str, int] = field(default_factory=dict)


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


def _assign(
    targets: Sequence[Target],
    generators: Sequence[Generator],
    *,
    per_generator: bool,
    context_dropout: float = 0.0,
    assign_seed: str = "",
) -> list[tuple[Target, Generator, str]]:
    """Pair targets with generators → ``(target, generator, effective_context)``.

    Default (rotate, `per_generator=False`): **content-hash assignment** —
    target → generator ``hash(salt, text) % n``. Statistically this makes the
    arm exchangeable with respect to document position (a round-robin cycle can
    phase-lock with chunk structure, aliasing arm with position-in-post and
    confounding any arm contrast); operationally it keeps each target's arm
    independent of the rest of the target set, so corpus resume stays stable as
    the blog grows. Balance across arms is multinomial, not exact (~√n wobble)
    — experiments wanting exact within-target crossing use `per_generator=True`
    (every target × every generator, n× the pairs/cost). `assign_seed`
    re-randomizes the whole assignment (a fresh replicate of the design; NOTE
    it changes which arm generated each target, so changed assignments
    regenerate on resume). `context` is the effective heading context after
    dropout. Keys are computed in `_plan_sessions` (they may fold in session
    membership).
    """
    if not generators:
        return []
    out: list[tuple[Target, Generator, str]] = []
    if per_generator:
        for t in targets:
            for gen in generators:
                out.append((t, gen, _effective_context(t, context_dropout)))
    else:
        n = len(generators)
        for t in targets:
            digest = hashlib.sha256(f"{assign_seed}\x00{t.text}".encode("utf-8")).digest()
            idx = int.from_bytes(digest[:8], "big") % n
            out.append((t, generators[idx], _effective_context(t, context_dropout)))
    return out


def _plan_sessions(
    assignments: Sequence[tuple[Target, Generator, str]],
    *,
    session_turns: int = 1,
) -> list[_Session]:
    """Chunk each generator's assigned targets (in order) into sessions.

    Assignment order is preserved — with the default walk order a session works
    through one post, then the next, like a real editing pass. `session_turns=1`
    yields one single-turn session per pair with NO session component in the
    key (stateless keys stay stable as the target set grows); multi-turn
    sessions fold `session_id:turn` into each key so every turn is a distinct,
    resumable pair. `session_id` is derived from the generator name and the
    ordered turn inputs — never from prior generations.
    """
    # Group by generator IDENTITY, not name: a rotation may carry the same
    # model under several slop strategies (same .name, different .strategy),
    # and name-keying would silently merge them.
    per_gen: dict[int, tuple[Generator, list[tuple[Target, str]]]] = {}
    for target, gen, ctx in assignments:
        per_gen.setdefault(id(gen), (gen, []))[1].append((target, ctx))

    sessions: list[_Session] = []
    for gen, items in per_gen.values():
        for start in range(0, len(items), max(1, session_turns)):
            chunk = items[start : start + max(1, session_turns)]
            if session_turns > 1:
                # Strategy/prompt distinguish sessions of the same model, so
                # session_id stays a unique grouping id across the rotation.
                sid_h = hashlib.sha256(
                    f"{gen.name}\x00{gen.strategy}\x00{gen.prompt_id}".encode("utf-8")
                )
                for target, ctx in chunk:
                    sid_h.update(b"\x00")
                    sid_h.update(ctx.encode("utf-8"))
                    sid_h.update(target.text.encode("utf-8"))
                sid = sid_h.hexdigest()[:8]
            else:
                sid = ""
            turns = [
                _Turn(
                    target=target,
                    key=_synth_key(
                        gen.name, target.text, ctx, gen.strategy, gen.reasoning_effort,
                        gen.prompt_id, session=f"{sid}:{idx}" if sid else "",
                    ),
                    context=ctx,
                    index=idx,
                )
                for idx, (target, ctx) in enumerate(chunk, start=1)
            ]
            sessions.append(_Session(generator=gen, session_id=sid, turns=turns))
    return sessions


def _record_prompts(data_dir: Path, generators: Sequence[Generator]) -> None:
    """Archive each run's system prompts to `<data-dir>/prompts.jsonl` (id-deduped).

    `meta.gen.prompt_id` is a content hash — opaque on its own. This sidecar
    keeps the exact prompt text next to the corpus it produced, so
    "what did prompt dc0f6c5c5de6 actually say?" is a grep, including for
    custom `--slop-system-file` prompts and superseded registry versions.
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
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
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

    Explicit where it used to be closure cells: `_run_session` coroutines for
    every session share this one object. Single-threaded asyncio — mutations
    interleave only between awaits, so no locks.
    """

    result: SynthResult
    seen: set[str]  # synth_keys already in the file (resume)
    claimed: set[str]  # in-run dup guard, claimed BEFORE the await
    recorded_slop: dict[str, str]  # already-recorded turns, replayed into history
    windows: dict[str, int]  # generator name -> context window (live sessions)
    session_max_tokens: int | None
    data_dir: Path
    extra_tags: Sequence[str]
    total: int  # pairs this run will attempt (for progress)
    done: int = 0
    on_progress: Callable[[int, int], None] | None = None
    on_error: Callable[[str, str], None] | None = None


def _session_token_cap(gen: Generator, window: int | None, session_max_tokens: int | None) -> int | None:
    """One session's prompt-token budget: the generator's own budget (else the
    global default), further capped at `SESSION_WINDOW_FILL_CAP` × window."""
    caps = [gen.session_budget or session_max_tokens]
    if window:
        caps.append(int(window * SESSION_WINDOW_FILL_CAP))
    return min((c for c in caps if c), default=None)


def _estimate_next_prompt(history: Sequence[dict], last_prompt: int, last_visible: int, next_text: str) -> int:
    """Estimate the next turn's prompt tokens.

    Last measured prompt + what the last reply added + the next passage
    (~4 chars/token) — or a pure char estimate before any live turn has
    reported usage (e.g. resume replayed history without generating yet).
    """
    if last_prompt:
        return last_prompt + last_visible + len(next_text) // 4
    return (sum(len(m["content"]) for m in history) + len(next_text)) // 4


def _load_recorded_slop(pairs_path: Path, sessions: Sequence[_Session], seen: set[str]) -> dict[str, str]:
    """Session resume: the recorded slop of already-done turns, for history replay.

    One extra pass over the file, restricted to keys that can actually matter
    (multi-turn sessions with an already-recorded turn).
    """
    from stylebot.eval import extract_slop  # lazy: eval pulls in scorer deps

    needed = {t.key for s in sessions if len(s.turns) > 1 for t in s.turns} & seen
    out: dict[str, str] = {}
    if needed:
        for rec in iter_pairs(pairs_path):
            k = (rec.get("meta") or {}).get("synth_key")
            if k in needed:
                out[k] = extract_slop(rec)
    return out


def _pop_reasoning_to_sidecar(data_dir: Path, turn: _Turn, gen: Generator, gen_meta: dict, *, in_session: bool) -> None:
    """Route a reasoning trace to the `reasoning.jsonl` sidecar (keyed to the pair).

    Traces are diagnostics, not corpus: `reasoning_text` is POPPED from
    `gen_meta` so it never enters `pairs.jsonl`. No-op when the generator
    captured none.
    """
    reasoning_text = gen_meta.pop("reasoning_text", None)
    if reasoning_text is None:
        return
    with (data_dir / "reasoning.jsonl").open("a", encoding="utf-8") as rf:
        rf.write(json.dumps({
            "synth_key": turn.key,
            "generator": gen.name,
            "slop_strategy": gen.strategy,
            "reasoning_effort": gen.reasoning_effort,
            "session_turn": turn.index if in_session else None,
            "reasoning_tokens": gen_meta.get("reasoning_tokens"),
            "reasoning": reasoning_text,
        }, ensure_ascii=False) + "\n")


async def _run_session(sess: _Session, fp, st: _RunState) -> None:
    """Run one session's turns in order, appending records as they complete.

    A turn error ends THIS session (later turns would see a hole in history);
    other sessions continue, and the failed turn retries next run under the
    same key. Already-recorded turns are replayed into history from
    `st.recorded_slop` instead of regenerating.
    """
    gen = sess.generator
    # Multi-turn sessions get a per-session callable when the generator
    # offers one (sticky provider routing state); stateless stays on the
    # plain path.
    call = gen.begin_session() if (len(sess.turns) > 1 and gen.begin_session is not None) else gen
    window = st.windows.get(gen.name)
    cap = _session_token_cap(gen, window, st.session_max_tokens)
    history: list[dict] = []
    last_prompt = 0  # prompt_tokens reported for the last live turn
    last_visible = 0  # ~tokens the last reply added to history

    for turn in sess.turns:
        if history and cap is not None:
            if _estimate_next_prompt(history, last_prompt, last_visible, turn.target.text) > cap:
                break
        if turn.key in st.seen or turn.key in st.claimed:
            if (slop := st.recorded_slop.get(turn.key)) is not None:
                history += _exchange(turn.target.text, slop)
            continue
        st.claimed.add(turn.key)

        # A failed pair is never written, so its synth_key resolves to
        # nothing — the recorded error must carry the config that produced
        # it or the failure is unattributable.
        who = f"{gen.name} strategy={gen.strategy} effort={gen.reasoning_effort}"
        if sess.session_id:
            who += f" turn={turn.index}"

        def _fail(message: str, *, _who: str = who, _key: str = turn.key) -> None:
            msg = f"[{_who}] {message}"
            st.result.errors.append((_key, msg))
            if st.on_error is not None:
                st.on_error(_key, msg)

        async def _attempt(fn, _turn: _Turn = turn) -> tuple[str, dict]:
            out = fn(_turn.target.text, history=history or None)
            if inspect.isawaitable(out):
                out = await out
            slop, gen_meta = _normalize_gen_output(out)
            if not slop.strip():
                raise RuntimeError("generator returned empty output")
            return slop, gen_meta

        try:
            slop, gen_meta = await _attempt(call)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Sticky routing pins a session to whoever served turn 1, and some
            # upstreams cannot serve the rest of it at all: OpenRouter replays
            # assistant turns with a `reasoning_content` some providers reject,
            # and others answer a valid request with empty content. Ending the
            # session on the first such error does not merely lose turns, it
            # loses the LATE ones — which silently truncates every session that
            # lands on a bad provider and biases any window-position analysis
            # toward low context fill. Re-pin onto a fresh provider once before
            # giving up; single-turn sessions have nothing to salvage.
            if sess.session_id and gen.begin_session is not None:
                try:
                    call = gen.begin_session()
                    slop, gen_meta = await _attempt(call)
                except asyncio.CancelledError:
                    raise
                except Exception as retry_exc:
                    _fail(f"{type(retry_exc).__name__}: {retry_exc} (after re-pinning provider)")
                    break
            else:
                _fail(f"{type(exc).__name__}: {exc}")
                break
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
        _pop_reasoning_to_sidecar(st.data_dir, turn, gen, gen_meta, in_session=bool(sess.session_id))
        record = _build_record(
            slop=slop,
            target=turn.target,
            generator_name=gen.name,
            synth_key=turn.key,
            strategy=gen.strategy,
            context=turn.context,
            extra_tags=st.extra_tags,
            gen_meta=gen_meta,
        )
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        fp.flush()
        st.result.written += 1
        st.result.per_generator[gen.name] = st.result.per_generator.get(gen.name, 0) + 1
        st.done += 1
        if st.on_progress is not None:
            st.on_progress(st.done, st.total)
        history += _exchange(turn.target.text, slop)
        if prompt_tokens:
            last_prompt = prompt_tokens
        completion = gen_meta.get("completion_tokens") or 0
        reasoning = gen_meta.get("reasoning_tokens") or 0
        last_visible = max(completion - reasoning, len(slop) // 4)


async def _run_sessions(sessions: Sequence[_Session], pairs_path: Path, st: _RunState, max_workers: int) -> None:
    """Run sessions concurrently, `max_workers` at a time, on one append handle."""
    sem = asyncio.Semaphore(max(1, max_workers))

    async def _gated(sess: _Session, fp) -> None:
        async with sem:
            await _run_session(sess, fp, st)

    with pairs_path.open("a", encoding="utf-8") as fp:
        await asyncio.gather(*(_gated(s, fp) for s in sessions))


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
    max_workers: int = 1,
    session_turns: int = 1,
    session_max_tokens: int | None = DEFAULT_SESSION_MAX_TOKENS,
    context_windows: Mapping[str, int] | None = None,
    assign_seed: str = "",
) -> SynthResult:
    """Generate synthetic pairs and append them to `data_dir/pairs.jsonl`.

    Idempotent and resumable: each pair carries a `meta.synth_key`
    (`hash(generator, strategy, reasoning_effort, prompt_id, context, target
    [, session])`); assignments whose key is already in the file are skipped, so
    re-running never duplicates and a crashed run resumes where it stopped
    (records are appended one-per-line, flushed as they go).

    **Concurrency**: sessions run concurrently on an asyncio event loop,
    `max_workers` at a time (single-threaded — writes and callbacks interleave
    only between awaits, so no locks). Generator callables may be sync or async.

    **Sessions** (`session_turns > 1`): each generator's targets are chunked
    into live multi-turn sessions — every turn sees the real prior
    (passage → slop) exchanges, so the window-position covariate is honest
    self-conditioned context. Fill is *measured*, not engineered: the recorded
    `prompt_tokens` (each model's own tokenizer) plus `context_window` /
    `window_fill` in `meta.gen` are the analysis covariates. A session ends
    early at `min(session_budget or session_max_tokens,
    SESSION_WINDOW_FILL_CAP × window)` estimated prompt tokens, or on a turn
    error (later turns would see a hole in history; the failed turn retries
    next run under the same key, and already-recorded turns are replayed into
    history from the file).

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

    assignments = _assign(
        targets, generators,
        per_generator=per_generator, context_dropout=context_dropout, assign_seed=assign_seed,
    )
    sessions = _plan_sessions(assignments, session_turns=session_turns)
    all_turns = [(s, t) for s in sessions for t in s.turns]
    result = SynthResult(planned=len(all_turns))

    seen = existing_synth_keys(pairs_path)
    total = sum(1 for _, t in all_turns if t.key not in seen)
    result.skipped_existing = len(all_turns) - total
    result.planned_sessions = sum(
        1 for s in sessions if any(t.key not in seen for t in s.turns)
    )

    if dry_run:
        for s, t in all_turns:
            if t.key not in seen:
                result.per_generator[s.generator.name] = result.per_generator.get(s.generator.name, 0) + 1
        return result

    recorded_slop: dict[str, str] = {}
    if session_turns > 1 and result.skipped_existing:
        recorded_slop = _load_recorded_slop(pairs_path, sessions, seen)

    data_dir.mkdir(parents=True, exist_ok=True)
    _record_prompts(data_dir, generators)
    st = _RunState(
        result=result,
        seen=seen,
        claimed=set(),
        recorded_slop=recorded_slop,
        windows=dict(context_windows or {}),
        session_max_tokens=session_max_tokens,
        data_dir=data_dir,
        extra_tags=extra_tags,
        total=total,
        on_progress=on_progress,
        on_error=on_error,
    )
    asyncio.run(_run_sessions(sessions, pairs_path, st, max_workers))
    return result
