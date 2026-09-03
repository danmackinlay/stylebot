"""Phase 2 — synthetic pair generation.

Library-first: `synthesize_pairs` is a typed function over explicit paths/params;
`ai-style synth` (`stylebot.bin.ai_style`) is a thin CLI wrapper.

Method (see `_plans/decisions.md` D9, D11-D13): human-authored prose is the
**target** (`messages[2]`); an LLM paraphrases it into "more polished" prose, the
**slop source** (`messages[1]`), and the styler learns to undo that transform.
Output is the Phase-1 `pairs.jsonl` schema chunked the same way, so real and
synthetic pairs are mixable; synthetic records add `meta.synthetic`,
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
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.generators import (
    DEFAULT_REASONING_EFFORT,
    DEFAULT_STRATEGY,
    Generator,
    _normalize_gen_output,
)
from stylebot.infer import invented_anchors
from stylebot.jsonl import append_jsonl, dumps_line, iter_jsonl
from stylebot.pairs import PAIR_SCHEMA_VERSION, build_pair_content, iter_pairs, system_sha
from stylebot.provenance import now_iso, pair_provenance, repo_provenance
from stylebot.targets import Target

logger = logging.getLogger(__name__)

# --- re-exports ---------------------------------------------------------------
# The surface this module publishes on behalf of its implementation homes
# (`stylebot.generators`, `stylebot.targets`). Every name here is reached by
# ATTRIBUTE lookup — `synth.openrouter_generator`, `synth.iter_targets` — by
# `stylebot.synth_cli`, `bin/ai_style.py` and livingthing, and that indirection
# is what lets a test monkeypatch a factory at the one place production resolves
# it. Direct `from stylebot.generators import …` is the better spelling for
# anything that does not need patching, and both consumers already do that where
# they can; this list is only what the attribute idiom actually requires.
#
# Keep it minimal, and check before adding: three names (`MAX_CHUNK_CHARS`,
# `MERGE_MAX_CHUNK_CHARS`, `MIN_CHUNK_CHARS`) sat here unreached once
# `ChunkPolicy` moved to its consumer, because nothing prunes this list on its
# own. Names used by this module's own code are imported normally above and are
# not part of it.
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
# first backoff between them — doubling thereafter. Distinct from
# `gate_retries`, which redraws a draw that ARRIVED and was rejected on quality:
# these two failures want different remedies and the counts must not be pooled.
#
# 2 (three attempts) because the fix that works is re-pinning the provider, not
# waiting: the openai SDK already retries the same request, and a `502 Provider
# returned error` means the upstream OpenRouter chose is sick, so attempts 2 and
# 3 ask for a different one. Backoff starts short for the same reason — a synth
# run is thousands of independent cells with other work to do.
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
    """Stable id for one synthetic pair — the resume/dedup cell identity.

    **Key on content, record circumstance.** Keyed: generator, strategy,
    reasoning effort, prompt id, heading context, target text, and the optional
    `replicate` label. NOT keyed: sampling params (continuous — they would
    explode the key space) and anything positional, session membership above all,
    being a runtime outcome that would make resume incoherent under drift.

    **Also not keyed: the chunk budget.** Two budgets that cut a passage to
    byte-identical text produce ONE cell, generated once and stamped
    (`meta.merge_max_chars`) by whichever batch reached it first — so that field
    records provenance, not the design cell a scale ablation would want, and an
    interrupted batch shifts the labelling. Keying on the budget instead would
    mint a duplicate pair per budget for identical text, multiplying the weight
    of exactly the short passages the section wall already over-represents.
    Measured overlap and what to do about it:
    `_plans/v2-longer-rewrites.md`, "What `meta.merge_max_chars` actually means".
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

    The corpus is a frozen record and deliberately does NOT track current prose,
    so a post edited after its pair was generated re-reads as an uncovered target
    and gets paid for twice. Most of that drift is *mechanical* — link-anchor
    repair scripts, a bulk quote-curling pass, a sentence-per-line reflow — and
    none of it is a difference the styler could learn, so it should not mint a
    second pair for the same passage.

    Three folds, each one a difference `STYLE_SYSTEM` makes invisible:

    1. **Inline link destinations.** The model is told to preserve links
       verbatim, so a rewritten URL or `#fragment` carries no voice; the link
       *text* is kept, being prose.
    2. **Curly quotes and apostrophes**, folded to ASCII.
    3. **Whitespace runs**, collapsed — a join is not an edit.

    Deliberately NOT folded: case, punctuation generally, dashes, ellipses,
    entity spellings, emphasis markers. Every extra rule risks collapsing two
    genuinely distinct passages onto one key, and that failure is silent — the
    second passage would never get a pair at all. Measured over the 3686 distinct
    corpus bodies, these three produce **zero** collisions.

    NOT `_synth_key`. That is the corpus's cell identity and stays keyed on raw
    text; this normalisation lives entirely inside the coverage comparison.
    """
    return " ".join(_LINK_DEST_RE.sub("]()", body).translate(_QUOTE_FOLD).split())


def covered_target_keys(pairs_path: Path | str) -> set[str]:
    """The `coverage_key`s that already have >=1 pair in a corpus, any config.

    Context-agnostic: the assistant side of every record is
    `build_pair_content(meta.context, body)`, so the body is recovered by
    stripping the recorded prefix. Backs `skip_covered`, and returns normalised
    keys rather than raw bodies — see `coverage_key`.
    """
    bodies: set[str] = set()
    for rec in iter_pairs(pairs_path):
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


def existing_synth_keys(pairs_path: Path | str) -> set[str]:
    """Read the `meta.synth_key`s already present in a `pairs.jsonl`.

    Tolerant: blank/undecodable lines are skipped, a missing file reads as empty.
    """
    return {
        key for rec in iter_pairs(pairs_path) if (key := (rec.get("meta") or {}).get("synth_key"))
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
    repo_prov: Mapping | None = None,
) -> dict:
    meta = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "system_sha": system_sha(),
        "source": target.source,
        "captured_at": now_iso(),
        "capture_id": _capture_id(target.source, generator_name, strategy),
        "chunk_index": target.chunk_index,
        "chunk_total": target.chunk_total,
        "before_chars": len(slop),  # body lengths (the transform), excluding the heading prefix
        "after_chars": len(target.text),
        "transform_sim": transform_similarity(slop, target.text),
        # Chunk-scale facet: the merge budget the target was packed under (None =
        # unmerged) and how it was cut (None = section-bounded).
        "merge_max_chars": target.merge_max_chars,
        "chunk_mode": target.chunk_mode,
        "synthetic": True,
        "generator": generator_name,
        "slop_strategy": strategy,
        "synth_key": synth_key,
        "tags": ["synthetic", "paraphrase", *extra_tags],
        # Which trees and which cut produced this pair. The chunk policy arrives
        # ON THE TARGET (`iter_targets` digests whatever `ChunkPolicy` it was
        # handed) rather than as a parameter here, so the stamp tracks the real
        # composed policy without `synthesize_pairs` having to be told about it
        # — and keeps tracking it once the blog composes that policy in one
        # place. `synth_key` deliberately does NOT fold any of this in: resume
        # must not restart because the tree moved a commit.
        "provenance": pair_provenance(
            repo_prov,
            policy_digest=target.policy_digest,
            source_automation=target.automation,
        ),
    }
    if context:
        # Identical heading prefix on both sides: the styler restyles the body
        # conditioned on, but never rewriting, the heading.
        meta["context"] = context
        meta["context_mode"] = "immediate"
    if gen_meta:
        meta["gen"] = gen_meta  # synthetic-only; real pairs have no `gen`
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
    skipped_covered: int = 0  # targets dropped by skip_covered (any-config coverage)
    # Hygiene-gate drops: paid for, never written. A dropped cell stays un-keyed,
    # so a re-run retries it and a consistently failing model burns spend every
    # run until it leaves the roster — these counts are the signal to act on.
    skipped_degenerate: int = 0  # identity: transform_sim > max_transform_sim
    skipped_inflated: int = 0  # inflation: length ratio > max_length_ratio
    skipped_invented_anchors: int = 0  # fabrication: target carries anchors the slop lacks
    # Redraw accounting (`gate_retries`): a gated draw is paid for either way, so
    # these are the cost of NOT letting a gate quietly bias coverage.
    gate_redraws: int = 0  # extra paid draws spent clearing a gate
    redrawn_cells: int = 0  # cells that needed at least one redraw
    redraws_recovered: int = 0  # ... of those, ones that went on to be written
    # Rejected draws per generator — the run-time face of `gate_drops.jsonl`.
    # Against `per_generator` (written) this is an arm's rejection rate, which is
    # what says whether an arm belongs on the roster: llama-3.3-70b left it on
    # exactly this number, and no other gate could see the defect.
    drops_by_generator: dict[str, int] = field(default_factory=dict)
    # Transient upstream failures survived, and cells lost to one anyway. A run
    # that spends many retries and still errors is looking at a sick provider,
    # not a sick config — the distinction the printed-and-discarded errors could
    # never make (`lost_draws.jsonl`, kind="error").
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

    Default (rotate, `per_generator=False`): **rendezvous (highest-random-weight)
    assignment** — score every arm as ``sha256(salt, arm_identity, text)``, take
    the argmax, break ties on arm identity. Scoring on *content* rather than
    position keeps the arm exchangeable with document position and each target's
    arm independent of the rest of the set, so adding posts or editing the roster
    does not re-key the corpus wholesale. Balance is multinomial, not exact;
    `per_generator=True` crosses every target × every generator instead, at n×
    the pairs and cost. `assign_seed` re-randomizes the whole design — NOTE that
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
    # The synth_keys actually ON DISK, handed to `PairsWriter`. Deliberately
    # NOT the planner's `seen`, which additionally holds every cell this run
    # PLANS to write — feeding that to the writer would make it refuse its own
    # first write of every cell.
    disk_keys: set[str] = field(default_factory=set)


class PairsWriter:
    """An append handle for `pairs.jsonl` that will not duplicate a cell.

    **The defect this exists for.** `synthesize_pairs` loads
    `existing_synth_keys` once, plans, and appends with a bare `fp.write`. Two
    runs against the same corpus each read the file before the other wrote to
    it, so every cell both planned got written twice: 21 duplicate `record_id`s
    reached the live corpus on 2026-08-12, from two budgets run concurrently.
    Nothing downstream tolerates that — eval and rewrite resume on `record_id`,
    score the first copy and skip the second, while training takes both, so the
    pair silently gets double weight and a score measured on the other draw.

    So every append is guarded by an exclusive `flock` on the file, and inside
    that lock the writer catches up on **whatever was appended since it last
    looked** before deciding. The catch-up is a tail read from a byte offset,
    not a re-read of the corpus: re-reading 38 MB per record would be
    quadratic, which is why "re-read `seen` before each write" is not on its own
    an implementable instruction.

    `flock` is advisory and per open-file-description, so it serialises two
    processes *and* two handles in one process, and costs nothing when there is
    only one writer. Platforms without `fcntl` (Windows) degrade to the
    unlocked-but-still-tail-checking path rather than failing to write.
    """

    def __init__(self, path: Path, seen: set[str]) -> None:
        self._path = Path(path)
        self._seen = seen
        self._offset = 0
        self._fp = None

    def __enter__(self) -> PairsWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # BINARY "ab+", not text mode, for two reasons that both bite silently:
        # a text handle raises on `tell()` after line iteration, and its `seek`
        # takes an opaque cookie rather than a byte offset. Encoding one JSON
        # line by hand is the cheaper half of that trade.
        self._fp = self._path.open("ab+")
        self._fp.seek(0, 2)
        self._offset = self._fp.tell()
        return self

    def __exit__(self, *exc) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _catch_up(self) -> None:
        """Fold every synth_key appended by anyone since our last look into `seen`."""
        fp = self._fp
        fp.seek(self._offset)
        tail = fp.read()
        self._offset = fp.tell()
        for raw in tail.split(b"\n"):
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            key = (rec.get("meta") or {}).get("synth_key") if isinstance(rec, dict) else None
            if key:
                self._seen.add(key)

    def write(self, key: str, record: dict) -> bool:
        """Append `record` unless `key` is already in the file. True if written."""
        fp = self._fp
        with _flock(fp):
            self._catch_up()
            if key and key in self._seen:
                return False
            fp.write((dumps_line(record) + "\n").encode("utf-8"))
            fp.flush()
            self._offset = fp.tell()
            if key:
                self._seen.add(key)
        return True


@contextmanager
def _flock(fp):
    """Exclusive advisory lock on `fp` for the block, a no-op without `fcntl`."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover — non-POSIX
        yield
        return
    fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)


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
    like afterwards. The distinction is not pedantry: synthesis is a reverse
    edit, so a defect in the generated slop surfaces as an apparent property of
    Dan's prose, and a message phrased the second way reads as an accusation
    against the one artifact here that is above suspicion. "target invents 3
    anchors" is exactly backwards — the target is a fixed human passage that
    invents nothing; the generator DELETED three links while producing slop, and
    reversing that edit is what would teach invention.
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
    if st.drop_invented_anchors and (invented := invented_anchors(slop, target_text)):
        # The generator dropped a link while "improving" the passage, and the
        # reverse-edit swap turns that into an example reading *given text
        # without links, produce text with these links*. The most-taught failure
        # in the corpus and the one nothing checked before 0.3.0
        # (`_plans/anchor-integrity.md`).
        uniq = sorted(set(invented))
        shown = ", ".join(uniq[:2]) + ("…" if len(uniq) > 2 else "")
        return (
            "skipped_invented_anchors",
            f"generator deleted {len(uniq)} anchor(s) the target keeps ({shown}); "
            "reversed, this pair would teach the styler to invent them",
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

    **One file for both ways a draw is lost**, `kind="gate"` and `kind="error"`,
    because the operational question is which of the two is eating a run — and
    that was unanswerable while both were printed to stderr and discarded.

    **A gated draw is paid for and never written**, so without this its cost and
    its very existence vanish when the run ends — the counters were computed,
    printed to stderr and discarded. Two questions that matters for, both of
    which came up the day the fabrication gate landed:

    - *What is this arm's rejection rate?* Once a gate is on, every pair in the
      corpus passed it, so the rate is no longer recoverable from `pairs.jsonl`
      at all. The denominator is written pairs (which carry `meta.gen.cost`)
      plus these lines; the numerator is these lines.
    - *What is this arm really costing?* `meta.gen.cost` prices only surviving
      draws, so an arm that fails half its draws looks half as expensive as it
      is. `cost` here is the discarded spend.

    Unconditional, like `prompts.jsonl` and unlike the reasoning sidecar: it is
    one short line per lost draw, carries no prose, and is the only record that
    a paid draw happened. `attempt` is 0 for the first draw; `redrawn` says
    whether a further draw followed, so `redrawn=false` marks the cell as
    actually lost rather than recovered.
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
    *,
    gate: str,
    reason: str,
    attempt: int,
    redrawn: bool,
) -> None:
    """Account for one gated draw: sidecar line, arm rejection count, spend.

    All three happen on BOTH gate paths — the redraw that will try again, and
    the final give-up — because a gated draw is paid for either way. They were
    written out twice, twenty lines apart, which is how an arm's rejection rate
    and its true cost get to disagree.
    """
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

    Returns `(slop, gen_meta, call)` — the callable comes back because a retry
    RE-PINS it, and the caller must keep using the one that worked.

    **Re-pinning matters more than waiting.** OpenRouter routes to an upstream
    and sticky routing keeps a session on whoever served turn 1, so when that
    upstream is sick (`502 Provider returned error` from Cloudflare is the
    common one) a plain retry re-hits the same broken box — which is why the
    openai SDK's own retries do not save this, and why every attempt after the
    first asks for a fresh provider.

    The backoff is there for the case re-pinning cannot fix, an upstream being
    rate-limited or restarting; it is short because a synth run is thousands of
    independent cells and there is other work to do.
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
    """Draw for one turn, redrawing while a hygiene gate rejects the sample.

    Returns `(slop, gen_meta, verdict, redraws, call)`. `verdict` is None once a
    sample cleared, or the last rejection when the redraw budget ran out — the
    caller decides what a surviving verdict costs. Every rejected draw is
    accounted for here, since it was paid for.

    A gated sample is paid for whether or not it is written, so a redraw costs
    no more than the drop already did — and it usually clears: rejection is
    mostly a per-sample accident, not a property of the passage (measured
    2026-08-12, anchor invention spans 1.3% to 51% BY ARM, and of the corpus
    targets that ever lost an anchor only 20% lost one under every arm that saw
    them). Without redraws a gate silently biases coverage toward the passages
    models find easy — link-dense prose drops 33% of the time against 16% for
    single-link — which is exactly the prose anchor fidelity matters most for.
    """
    slop, gen_meta, call = await _draw(call, gen, turn, history, st)
    verdict = _gate_verdict(slop, turn.target.text, st)
    redraws = 0
    while verdict is not None and redraws < st.gate_retries:
        _record_gate_drop(
            st,
            turn,
            gen,
            gen_meta,
            gate=verdict[0],
            reason=verdict[1],
            attempt=redraws,
            redrawn=True,
        )
        redraws += 1
        st.result.gate_redraws += 1
        logger.info(
            "gate redraw %d/%d: %s %s (%s)",
            redraws,
            st.gate_retries,
            gen.name,
            verdict[1],
            turn.key,
        )
        try:
            slop, gen_meta, call = await _draw(call, gen, turn, history, st)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A provider failing DURING a redraw must not cost the session. We
            # already hold a gated draw, so the honest outcome is the one we
            # would have had without retries: drop this cell, keep going.
            # Abandoning the remaining turns instead would turn one transient
            # 502 into a session-wide loss.
            logger.warning(
                "gate redraw %d failed (%s); dropping the cell and continuing: %s",
                redraws,
                type(exc).__name__,
                turn.key,
            )
            break
        verdict = _gate_verdict(slop, turn.target.text, st)
    if redraws:
        st.result.redrawn_cells += 1
        if verdict is None:
            st.result.redraws_recovered += 1
    return slop, gen_meta, verdict, redraws, call


async def _run_session(sess: _Session, writer: PairsWriter, st: _RunState) -> list[_Turn]:
    """Run one session's turns in order, appending records as they complete.

    Returns the turns this session could NOT run, for the caller to reflow into a
    fresh session — never silently dropped. A binding token budget returns every
    remaining turn; a turn erroring (after the one re-pin retry) returns the
    turns AFTER it, while the failed turn retries next run under the same key.

    Termination: the budget check is skipped while history is empty, so a fresh
    session always attempts its first turn — every reflow either writes a pair or
    sheds an errored turn, strictly shrinking the leftover list.
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
            # nothing — the recorded error must carry its config or be
            # unattributable.
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
                attempt=st.gen_retries,
                redrawn=False,
            )
            st.result.errors_by_generator[gen.name] = (
                st.result.errors_by_generator.get(gen.name, 0) + 1
            )
            # The REST of the session still reflows into fresh sessions rather
            # than being dropped — sticky routing means a session that has
            # started failing usually keeps failing, and the leftover turns
            # deserve a healthy provider. Only this cell is lost.
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
                st, turn, gen, gen_meta, gate=counter, reason=reason, attempt=redraws, redrawn=False
            )
            # "gate dropped a draw", not "degenerate output": two of the three
            # gates catch a degenerate generator, but the anchor gate catches a
            # perfectly fluent draw that merely deleted a link, and calling that
            # degenerate sent readers looking for the wrong defect.
            logger.warning("gate dropped a draw: %s — %s (%s)", gen.name, reason, turn.key)
        elif writer.write(turn.key, record):
            st.result.written += 1
            st.result.per_generator[gen.name] = st.result.per_generator.get(gen.name, 0) + 1
            st.result.add_gen_stats(gen.name, gen_meta)
        else:
            # Another writer claimed this cell between our plan and our write.
            # The draw is paid for and discarded rather than appended: one cell,
            # one record, whoever got there first (`PairsWriter`).
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


async def _run_sessions(
    sessions: Sequence[_Session], pairs_path: Path, st: _RunState, max_workers: int
) -> None:
    """Run sessions concurrently, `max_workers` at a time, on one append handle.

    Leftover turns (budget bind, error) reflow into fresh sessions inside the
    same worker slot until the work list is empty, so the run finishes with every
    planned turn either written or individually errored.
    """
    sem = asyncio.Semaphore(max(1, max_workers))

    async def _gated(sess: _Session, writer: PairsWriter) -> None:
        async with sem:
            leftover = await _run_session(sess, writer, st)
            while leftover:
                st.result.reflow_sessions += 1
                st.result.reflowed_turns += len(leftover)
                sess = _respin_session(sess.generator, leftover)
                leftover = await _run_session(sess, writer, st)

    with PairsWriter(pairs_path, st.disk_keys) as writer:
        await asyncio.gather(*(_gated(s, writer) for s in sessions))


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
) -> SynthResult:
    """Generate synthetic pairs and append them to `data_dir/pairs.jsonl`.

    Idempotent and resumable: assignments whose content-only `meta.synth_key` is
    already in the file are skipped, so re-running never duplicates and a crashed
    run resumes where it stopped (records are appended one-per-line, flushed as
    they go). Missing cells are freshly packed into sessions each run.

    **Concurrency**: sessions run on an asyncio event loop, `max_workers` at a
    time (single-threaded — writes and callbacks interleave only between awaits,
    so no locks). Generator callables may be sync or async.

    **Sessions** (`session_turns > 1`): missing cells are chunked into live
    multi-turn sessions, so every turn sees the real prior (passage → slop)
    exchanges and the window-position covariate is honest self-conditioned
    context. Fill is *measured*, not engineered — `prompt_tokens`,
    `context_window` and `window_fill` in `meta.gen` are the analysis covariates.
    **The token budget is the depth control**: a session ends at
    `min(session_budget or session_max_tokens, SESSION_WINDOW_FILL_CAP × window)`
    estimated prompt tokens, and turns it couldn't run REFLOW into fresh sessions
    rather than being dropped.

    **Hygiene gates** all share one shape: count loudly, write nothing, leave
    the cell un-keyed so a re-run retries it and a consistently failing model
    burns visible spend until it leaves the roster. `max_transform_sim` catches
    identity output, `max_length_ratio` catches inflation, and
    `drop_invented_anchors` catches fabrication — a target carrying a link, URL
    or citation its slop never had, which after the reverse-edit swap teaches
    the styler to conjure destinations it cannot know.

    The fabrication gate is the one that defaults **on**, unlike its two
    threshold-valued siblings: there is no number to tune and no defensible
    setting on the other side, since a pair teaching link invention has no
    legitimate use. `infer.is_anchor_faithful` is the matching `selector` for
    pairs written before it existed (`_plans/anchor-integrity.md`, D32).

    **`gate_retries`** spends up to N extra draws clearing a gate before giving
    up on a cell. Worth setting whenever a gate is on: a rejected draw is paid
    for either way, so a redraw costs no more than the drop, and without it a
    gate quietly biases coverage toward whatever the roster finds easy.

    Heading `context` is prepended verbatim to both sides of the pair while slop
    is generated from the body only, so the heading is never paraphrased;
    `context_dropout` keeps a deterministic fraction heading-less. `dry_run`
    plans and counts without calling any generator or writing, so generators may
    be name-only `Generator(name, generate=None)` stubs.

    **Provenance** (`meta.provenance`) is resolved ONCE per run and stamped on
    every record: `blog_root` names the prose repo whose SHA and dirty flag go
    on the pair, defaulting to the git work-tree containing `data_dir` — which
    is the blog, in the only deployment there is, since that is where the corpus
    is committed. The chunk policy and the source post's `automation` level ride
    in on the `Target`s (`targets._post_targets` stamps them at cut time), so
    the record says what the frontmatter *said then*, not what it says now.
    """
    data_dir = Path(data_dir)
    pairs_path = data_dir / "pairs.jsonl"

    skipped_covered = 0
    if skip_covered:
        # Coverage mode: a target with ANY existing pair is done. Compared on
        # `coverage_key`, so cosmetic prose drift since generation (rewritten
        # link destinations, curled quotes, a reflow) still reads as covered.
        # Applied before assignment; cell-level dedup below still governs the
        # rest, and it keys on RAW text.
        covered = covered_target_keys(pairs_path)
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
    disk_keys = existing_synth_keys(pairs_path)
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
        max_transform_sim=max_transform_sim,
        max_length_ratio=max_length_ratio,
        drop_invented_anchors=drop_invented_anchors,
        gate_retries=gate_retries,
        gen_retries=gen_retries,
        replicate=replicate,
        on_progress=on_progress,
        on_error=on_error,
        repo_prov=repo_provenance(blog_root if blog_root is not None else data_dir),
        disk_keys=disk_keys,
    )
    asyncio.run(_run_sessions(sessions, pairs_path, st, max_workers))
    return result
