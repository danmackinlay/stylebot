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
