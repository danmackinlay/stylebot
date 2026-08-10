"""Tolerant JSONL reading and writing, shared by every corpus artifact.

`pairs.jsonl`, `scores.jsonl`, `prompts.jsonl` and `reasoning.jsonl` are all
append-only JSONL; this is their one line-level reader and writer — the record
schemas differ, the line idiom doesn't. Stdlib-only with zero stylebot imports,
so any module — including the dep-free `classify` runtime — could use it
without an import cycle.
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


def dumps_line(record: dict) -> str:
    """One JSONL line (no trailing newline), UTF-8 literal rather than escaped.

    `ensure_ascii=False` is the house style across every artifact: the corpus is
    prose full of dashes and curly quotes, and escaping them triples the byte
    cost of a line for no reader's benefit.
    """
    return json.dumps(record, ensure_ascii=False)


def append_jsonl(path: str | Path, record: dict) -> None:
    """Append one record to a JSONL file, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(dumps_line(record) + "\n")


def write_jsonl_atomic(path: str | Path, records: list[dict]) -> None:
    """Atomic rewrite of a JSONL file from the given record list.

    Writes a sibling `.tmp` and renames, so a crash mid-write leaves the
    original intact. Records that are pure ``{"_raw": line}`` wrappers (the
    unparseable lines `iter_jsonl(keep_undecodable=True)` preserves) are written
    back verbatim, so a read-rewrite round trip never silently drops a corrupt
    line.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for rec in records:
            if "_raw" in rec and set(rec.keys()) == {"_raw"}:
                fp.write(rec["_raw"] + "\n")
            else:
                fp.write(dumps_line(rec) + "\n")
    tmp.replace(path)
