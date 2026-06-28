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

import json
from collections.abc import Iterator
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM

# meta keys every well-formed pair record carries (chunk 0 additionally may
# carry before_frontmatter/after_frontmatter; tags/synthetic/generator are
# optional provenance). These are the load-bearing ones downstream relies on.
REQUIRED_META_KEYS = (
    "source",
    "captured_at",
    "capture_id",
    "chunk_index",
    "chunk_total",
)


def build_pair_content(context: str, body: str) -> str:
    """Assemble a pair message body: heading context (verbatim) + the prose body.

    The **shared contract** between Phase 1 (`ai-style-log`) and Phase 2
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

    The shared, tolerant line-by-line reader for the corpus schema (UTF-8, blank
    lines skipped, undecodable / non-object lines skipped rather than raising) —
    the same idiom as `synth.existing_synth_keys`. Yields the raw parsed dict;
    callers validate with `validate_pair_record` if they need the contract
    enforced. A missing file yields nothing.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                yield rec
