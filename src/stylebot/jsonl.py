"""Tolerant JSONL reading, shared by every corpus artifact.

`pairs.jsonl`, `scores.jsonl`, `prompts.jsonl` and `reasoning.jsonl` are all
append-only JSONL; this is their one line-level reader (writers append
locally — the formats differ, the read idiom doesn't). Stdlib-only with zero
stylebot imports, so any module — including the dep-free `classify` runtime —
could use it without an import cycle.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path


def iter_jsonl(path: str | Path, *, keep_undecodable: bool = False) -> Iterator[dict]:
    """Yield each JSON object from a JSONL file, one per non-blank line.

    UTF-8; blank lines are skipped; undecodable and non-object lines are
    skipped — unless ``keep_undecodable``, which yields them as
    ``{"_raw": line}`` (trailing newline stripped) so read-rewrite paths can
    round-trip a corrupt line instead of silently dropping it. A missing file
    yields nothing.
    """
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                if keep_undecodable:
                    yield {"_raw": line}
                continue
            if isinstance(rec, dict):
                yield rec
            elif keep_undecodable:
                yield {"_raw": line}


def read_jsonl(path: str | Path, **kw) -> list[dict]:
    """`iter_jsonl` collected into a list."""
    return list(iter_jsonl(path, **kw))
