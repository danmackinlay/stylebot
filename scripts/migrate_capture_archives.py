#!/usr/bin/env python3
"""Migrate legacy JSON capture archives and session snapshots to directories.

The old format put whole documents inside a one-line JSON file as escaped
strings:

    captures/<capture_id>.json      {capture_id, source, snapped_at, whole,
                                     before_text, after_text}
    snapshots/<path>.qmd.json       {source, snapped_at, content}

which is unreadable, unopenable and undiffable — perverse, given the
before->after diff IS the training signal. The new format puts the documents on
disk as ordinary files:

    captures/<capture_id>/{before.md, after.md, meta.json}
    snapshots/<path>.qmd/{content.md, meta.json}

`.md` and not `.qmd` deliberately: the blog's pre-commit hook reformats staged
`*.qmd` (which would destroy the verbatim record), `qmd` search indexes
`**/*.qmd` (which would return captured slop as if Dan wrote it), and
`iter_targets` samples `**/*.qmd` as synthesis targets. `.md` is invisible to
all three.

Usage:
    uv run python scripts/migrate_capture_archives.py _training_pairs [--apply]

Dry-run by default. Each conversion writes the new directory first and unlinks
the legacy file only once the directory reads back with byte-identical text, so
an interrupted run leaves both forms rather than neither — nothing is lost, but
re-run it until `captures/` holds no `*.json`: since 0.3 `iter_archives` reads
only directories and REFUSES (rather than silently skipping) a leftover blob.

This script is the only remaining reader of the old format. It parses the blobs
itself, so it keeps working after the library branch is gone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stylebot.capture import read_archive, write_archive


def migrate_captures(cap_dir: Path, *, apply: bool) -> tuple[int, int]:
    """Convert `captures/*.json`. Returns (converted, skipped)."""
    converted = skipped = 0
    if not cap_dir.is_dir():
        return 0, 0
    for path in sorted(cap_dir.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  SKIP {path.name}: unreadable ({exc})")
            skipped += 1
            continue
        cap_id = blob.get("capture_id") or path.stem
        target = cap_dir / cap_id
        if target.exists():
            print(f"  SKIP {path.name}: {target.name}/ already exists")
            skipped += 1
            continue
        print(
            f"  {path.name} -> {cap_id}/  ({len(blob.get('before_text') or '')} -> "
            f"{len(blob.get('after_text') or '')} chars)"
        )
        if not apply:
            converted += 1
            continue
        write_archive(
            target,
            capture_id=cap_id,
            source=blob.get("source"),
            snapped_at=blob.get("snapped_at"),
            whole=bool(blob.get("whole")),
            before_text=blob.get("before_text") or "",
            after_text=blob.get("after_text") or "",
        )
        back = read_archive(target)
        if (
            back is None
            or back["before_text"] != (blob.get("before_text") or "")
            or back["after_text"] != (blob.get("after_text") or "")
        ):
            sys.exit(f"FATAL: round-trip mismatch for {path.name}; legacy file kept")
        path.unlink()  # only after the new form verifies
        converted += 1
    return converted, skipped


def migrate_snapshots(snap_root: Path, *, apply: bool) -> tuple[int, int]:
    """Convert `snapshots{,-closed}/**/<name>.json`. Returns (converted, skipped)."""
    converted = skipped = 0
    if not snap_root.is_dir():
        return 0, 0
    for path in sorted(snap_root.rglob("*.json")):
        if path.name == "meta.json":
            continue  # already migrated
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  SKIP {path}: unreadable ({exc})")
            skipped += 1
            continue
        # snapshots/post/foo.qmd.json -> snapshots/post/foo.qmd/
        target = path.with_suffix("")
        if target.exists():
            print(f"  SKIP {path}: {target.name}/ already exists")
            skipped += 1
            continue
        content = blob.get("content") or ""
        print(f"  {path.relative_to(snap_root)} -> {target.name}/  ({len(content)} chars)")
        if not apply:
            converted += 1
            continue
        target.mkdir(parents=True)
        (target / "content.md").write_text(content, encoding="utf-8")
        (target / "meta.json").write_text(
            json.dumps(
                {"source": blob.get("source"), "snapped_at": blob.get("snapped_at")},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if (target / "content.md").read_text(encoding="utf-8") != content:
            sys.exit(f"FATAL: round-trip mismatch for {path}; legacy file kept")
        path.unlink()
        converted += 1
    return converted, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_dir", type=Path, help="the _training_pairs directory")
    ap.add_argument("--apply", action="store_true", help="actually convert (default: dry run)")
    args = ap.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"no such directory: {args.data_dir}")

    total = 0
    print(f"captures/  ({'APPLY' if args.apply else 'dry run'})")
    c, s = migrate_captures(args.data_dir / "captures", apply=args.apply)
    print(f"  -> {c} converted, {s} skipped")
    total += c

    for name in ("snapshots", "snapshots-closed"):
        print(f"{name}/")
        c, s = migrate_snapshots(args.data_dir / name, apply=args.apply)
        print(f"  -> {c} converted, {s} skipped")
        total += c

    if not args.apply and total:
        print(f"\ndry run: {total} item(s) would convert. Re-run with --apply.")


if __name__ == "__main__":
    main()
