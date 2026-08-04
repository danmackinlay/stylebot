#!/usr/bin/env python3
"""Re-stamp `messages[0].content` in a pairs.jsonl to the current STYLE_SYSTEM.

`stylebot.pairs.validate_pair_record` pins the system message to the frozen
`STYLE_SYSTEM`, so amending that string fails the entire existing corpus until
every record is updated. This is that migration.

It is safe *because* it touches only the system message. The user and assistant
sides — the actual (slop, Dan) transform the styler learns — are never read or
written. What changes is the instruction the pair is conditioned on, which is
exactly what should change when the instruction's meaning changes.

The v2 amendment (2026-08-04) dropped "headings" from the preserve list so that
cross-heading chunks, which carry heading lines as rewritable content, do not
contradict their own system prompt. Pre-v2 records never contained a heading as
content, so re-stamping them is harmless: they simply never exercise the clause.

Usage:
    uv run python scripts/restamp_style_system.py PAIRS.jsonl [--dry-run]

Writes atomically via a sibling temp file, so an interrupted run cannot leave a
half-written corpus.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM


def restamp(path: Path, *, dry_run: bool = False) -> tuple[int, int]:
    """Return (n_records, n_changed). Writes nothing under `dry_run`."""
    records: list[dict] = []
    changed = 0
    with path.open(encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: unparseable JSON, refusing to rewrite: {exc}")
            msgs = rec.get("messages")
            if not (isinstance(msgs, list) and msgs and isinstance(msgs[0], dict)):
                sys.exit(f"{path}:{lineno}: record has no messages[0], refusing to rewrite")
            if msgs[0].get("role") != "system":
                sys.exit(f"{path}:{lineno}: messages[0] is not the system message")
            if msgs[0].get("content") != STYLE_SYSTEM:
                msgs[0]["content"] = STYLE_SYSTEM
                changed += 1
            records.append(rec)

    if not dry_run and changed:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                for rec in records:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
    return len(records), changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total, changed = restamp(args.pairs, dry_run=args.dry_run)
    verb = "would re-stamp" if args.dry_run else "re-stamped"
    print(f"{verb} {changed} of {total} record(s) in {args.pairs}")
    if not args.dry_run and changed:
        from stylebot.pairs import validate_pairs_file

        problems = validate_pairs_file(args.pairs)
        if problems:
            sys.exit(f"corpus still invalid after re-stamp: {len(problems)} bad record(s)")
        print("corpus validates clean")


if __name__ == "__main__":
    main()
