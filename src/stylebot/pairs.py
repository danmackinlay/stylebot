"""Validation for the shared `pairs.jsonl` contract.

The corpus schema is the seam between Phase 1 (real edit pairs), Phase 2
(synthetic pairs), and Phase 3 (training). This module makes that contract
*enforceable* rather than just documented: Phase 2 should validate its output
here as a done-criteria gate, and Phase 3 should validate its input before
spending money on a training run.

The schema is defined in `_plans/phase-1-pair-capture.md` and produced by
`stylebot.bin.ai_style_log`. Keep this in sync with that producer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.jsonl import iter_jsonl

# meta keys every well-formed pair record carries (chunk 0 additionally may
# carry before_frontmatter/after_frontmatter; tags/synthetic/generator are
# optional provenance). These are the load-bearing ones downstream relies on.
#
# Chunk-scale facets (optional, synthetic pairs): `merge_max_chars` is the pack
# budget, `chunk_mode` is how the chunk was cut — None/"section" for
# section-bounded (heading excluded from the text, carried as `context`) or
# "cross-heading" for headings packed in as CONTENT. Both are needed to ablate
# a mixed-scale corpus, and mode dominates: section-bounded packing hits the
# section wall long before the budget, so records differing only in
# `merge_max_chars` are near-identical diets. Cross-heading records carry no
# `context` (an inline heading is rewritable text, not fixed context), so the
# verbatim-prefix invariant below simply does not apply to them.
REQUIRED_META_KEYS = (
    "source",
    "captured_at",
    "capture_id",
    "chunk_index",
    "chunk_total",
)

# --- meta.gen schema (synthetic pairs only) ----------------------------------
# The generation covariates recorded under `meta.gen`, assembled by
# stylebot.synth's generator factories and consumed by stylebot.eval (flattened
# onto score records) and stylebot.report (row columns). One list, so a key
# rename fails a test instead of silently producing empty facet columns
# downstream. Callers may extend the dict via a factory's `extra_meta`; the
# stock schema is what's listed here.
GEN_META_KEYS = (
    "model",
    "reasoning_effort",
    "temperature",
    "top_p",
    "max_tokens",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cost",
    "gen_seconds",
    "provider",
    "provider_sort",
    "prompt_id",
    "prompt_version",
    "prompt_label",
    "replicate",  # deliberate-resample label (folded into synth_key; absent = base corpus)
)
# Added by the session loop only (`session_turns > 1`).
GEN_SESSION_KEYS = ("session_id", "session_turn", "context_window", "window_fill")
# The experiment-relevant subset eval flattens onto each score record so
# `summarize_scores(by=...)` and the report can facet by them directly.
#
# This is also the *durable* record of an experiment: a scores.jsonl outlives
# the pairs.jsonl it was scored from (the corpus is regenerable slop; the
# analysis is not). So it carries the covariates that explain an outcome —
# cost, latency and context position — not just the knobs that were set. A
# sweep whose scores omit `gen_seconds` cannot answer "what did that effort
# level buy", and no amount of re-aggregation gets it back.
GEN_FACET_KEYS = (
    "model",
    "reasoning_effort",
    "temperature",
    "top_p",
    "prompt_id",
    "prompt_version",
    "prompt_label",
    "replicate",
    "session_id",
    "session_turn",
    "context_window",
    "window_fill",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost",
    "gen_seconds",
    "provider",
    "finish_reason",
)


def build_pair_content(context: str, body: str) -> str:
    """Assemble a pair message body: heading context (verbatim) + the prose body.

    The **shared contract** between Phase 1 (`ai-style log`) and Phase 2
    (`stylebot.synth`): `context` — a section heading the body sits under — is
    prepended *identically* to both the user (slop) and assistant (target) sides
    so the styler learns to preserve the heading and restyle only the body
    beneath it (matching `STYLE_SYSTEM`'s preserve-structure clause). Empty
    context returns the body unchanged, so heading-less pairs are unaffected.

    Both producers MUST build message content through this one function, and
    record the same `context` under `meta.context`, so real and synthetic pairs
    stay shape-compatible. See `_plans/heading-context.md`.
    """
    context = (context or "").strip()
    body = body or ""
    return f"{context}\n\n{body}" if context else body


def pair_body(content: str, context: str | None) -> str:
    """The prose body of a pair message, with the heading-context prefix removed.

    `build_pair_content(context, body)` prepends ``f"{context}\\n\\n"``
    *identically to both sides* of a pair when a heading context is present; this
    is its inverse, so a consumer sees the transform region (the body), not the
    shared heading. No context (or content lacking the prefix) returns unchanged.
    """
    if not context:
        return content
    prefix = f"{context}\n\n"
    return content[len(prefix):] if content.startswith(prefix) else content


def extract_slop(record: dict) -> str:
    """The slop-side body of a pair (messages[1], heading-context stripped).

    Public API: the voice-classifier trainer (livingthing) assembles its training
    texts through these extractors so eval and training read a pair identically.
    """
    return pair_body(record["messages"][1]["content"], (record.get("meta") or {}).get("context"))


def extract_target(record: dict) -> str:
    """The Dan-side body of a pair (messages[2], heading-context stripped)."""
    return pair_body(record["messages"][2]["content"], (record.get("meta") or {}).get("context"))


def extract_output(record: dict) -> str:
    """The styler-output body (top-level `"output"` of a Phase-4 rewrite
    JSONL — `stylebot.infer.rewrite_pairs_file` — heading-context stripped)."""
    return pair_body(record.get("output") or "", (record.get("meta") or {}).get("context"))


# Field name -> (record -> body text). Defaults cover the pairs schema
# (messages[1]=slop/user, messages[2]=Dan/assistant); "output" is the Phase-4
# styler rewrite, present only on infer.rewrite_pairs_file output JSONLs.
FIELD_EXTRACTORS: dict[str, Callable[[dict], str]] = {
    "slop": extract_slop,
    "target": extract_target,
    "output": extract_output,
}


def _recut_digest(recut: object) -> str:
    """A short, stable digest of one re-cut's gap policy (`meta.recut`).

    Canonical JSON (sorted keys, no whitespace) so the digest keys off the
    policy's VALUES, not off dict insertion order — and so any key a future
    `recut_archives` adds to the stamp participates automatically. Eight hex
    chars: this discriminates a handful of hand-chosen cut policies, not a
    corpus, so collision pressure is nil and the id stays readable.
    """
    try:
        canon = json.dumps(recut, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:  # a stamp JSON can't encode — degrade to repr, never raise
        canon = repr(recut)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def record_id(record: dict) -> str:
    """A stable join key linking a pair's score back to the corpus.

    Prefers `meta.synth_key` (synthetic pairs); else `capture_id:chunk_index`
    (real captures group by `capture_id`); else a hash of `source` + chunk index.

    **Re-cut records additionally carry the cut policy in the key.**
    `capture.recut_archives` deliberately REUSES the archive's `capture_id` and
    renumbers `chunk_index` from 0 — that is exactly what makes a re-cut
    idempotent *in `pairs.jsonl`* (the re-cut replaces the old cut for that
    capture instead of stacking duplicates). The cost is that
    `capture_id:chunk_index` alone is NOT stable across a re-cut: after
    coalescing, `<cap>:0` names a merged multi-paragraph pair where it used to
    name a single paragraph. That is a correctness bug downstream, not
    cosmetics — everything that joins or resumes on this key would silently
    mismatch:

    - `eval.score_pairs_file` resumes by skipping ids already in `scores.jsonl`,
      so a re-cut pair reads as "already scored" and STALE scores stay attached
      to materially changed prose;
    - `report.render_scores_report` joins `pairs_by_id`, so the pair browser
      renders new text under old judge/detector numbers;
    - `infer.rewrite_pairs_file` skips it on resume for the same reason.

    So fold `meta.recut` — the gap policy `recut_archives` stamps on every
    record it emits, and the only discriminator available — into the key. A
    re-cut id then differs from the original cut's and from another policy's,
    while staying byte-identical when the SAME policy is re-run: a re-cut has
    to remain idempotent end to end, not just inside `pairs.jsonl`.

    Records with no `meta.recut` keep their key byte-for-byte. That is
    load-bearing: a committed `scores.jsonl` lives blog-side and must keep
    joining for the never-re-cut captures that are most of the corpus.
    """
    meta = record.get("meta") or {}
    if meta.get("synth_key"):
        return str(meta["synth_key"])
    recut = meta.get("recut")
    # Empty for the overwhelmingly common non-re-cut record — which is what
    # preserves the historical id shape exactly.
    policy = f":r{_recut_digest(recut)}" if recut else ""
    capture_id = meta.get("capture_id")
    chunk_index = meta.get("chunk_index")
    if capture_id is not None:
        return f"{capture_id}{policy}:{chunk_index}"
    # No capture_id (hand-built / pre-capture_id records). Same reasoning, so
    # the policy joins the hashed material rather than the visible key; with no
    # re-cut stamp the hashed string is unchanged and so is the digest.
    h = hashlib.sha256(f"{meta.get('source')}\x00{chunk_index}{policy}".encode("utf-8"))
    return h.hexdigest()[:16]


def validate_pair_record(rec: object) -> list[str]:
    """Return a list of human-readable problems with one pair record.

    Empty list == valid. Checks the message triple (system/user/assistant),
    that the system prompt is the frozen `STYLE_SYSTEM`, that user/assistant
    content is non-empty, and that the required `meta` keys are present.
    """
    errors: list[str] = []

    if not isinstance(rec, dict):
        return ["record is not a JSON object"]

    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        errors.append("messages must be a list of exactly 3 items")
    else:
        roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"message roles must be [system, user, assistant], got {roles}")
        else:
            if msgs[0].get("content") != STYLE_SYSTEM:
                errors.append(
                    "system message content is not the frozen STYLE_SYSTEM "
                    "(changing it invalidates the corpus)"
                )
            for role_idx, label in ((1, "user"), (2, "assistant")):
                content = msgs[role_idx].get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"{label} message content is empty or non-string")

    meta = rec.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be a JSON object")
    else:
        for key in REQUIRED_META_KEYS:
            if key not in meta:
                errors.append(f"meta missing required key {key!r}")

        # Heading-context invariant (shared contract): if a pair declares
        # meta.context, that heading must be the verbatim prefix of BOTH the
        # user and assistant content — i.e. it was added as fixed context via
        # build_pair_content, not paraphrased into the slop. Catches accidental
        # heading-rewriting. Only enforced when context is present and the
        # message triple parsed cleanly above.
        context = meta.get("context")
        if context and isinstance(msgs, list) and len(msgs) == 3:
            for role_idx, label in ((1, "user"), (2, "assistant")):
                content = msgs[role_idx].get("content") if isinstance(msgs[role_idx], dict) else None
                if not (isinstance(content, str) and content.startswith(context)):
                    errors.append(f"{label} content does not start with meta.context (heading must be a verbatim prefix)")

    return errors


def validate_pairs_file(path: str | Path) -> list[tuple[int, list[str]]]:
    """Validate every line of a `pairs.jsonl`.

    Returns a list of `(line_number, errors)` for each invalid line (1-based,
    skipping blank lines). An empty result means the whole file is valid.
    Unparseable JSON is reported as a single error for that line.
    """
    path = Path(path)
    problems: list[tuple[int, list[str]]] = []
    with path.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append((lineno, [f"invalid JSON: {exc}"]))
                continue
            errs = validate_pair_record(rec)
            if errs:
                problems.append((lineno, errs))
    return problems


def iter_pairs(path: str | Path) -> Iterator[dict]:
    """Yield each JSON object from a `pairs.jsonl`, one per non-blank line.

    The tolerant reader for the corpus schema (`stylebot.jsonl` under the hood:
    UTF-8, blank / undecodable / non-object lines skipped rather than raising).
    Yields the raw parsed dict; callers validate with `validate_pair_record` if
    they need the contract enforced. A missing file yields nothing.
    """
    yield from iter_jsonl(path)
