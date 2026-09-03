"""One-off data migrations: rewrite corpus state already on disk to a new schema.

These used to be loose files under `scripts/`, the single carve-out from the
"one entry point, subcommands" rule in CLAUDE.md. The carve-out cost more than
it saved, and W5 (2026-09-03) closed it:

* `scripts/` is not in the wheel, so `pip install stylebot` shipped a package
  whose own error message (`capture.iter_archives`) named a file the user did
  not have. The blog had to reach across the seam — `uv run python
  ../stylebot/scripts/migrate_capture_archives.py` in two places in its
  RUNBOOK — which is a hardcoded blog→stylebot *path*, the arrow W5 exists to
  straighten.
* "Typed once per schema change" was aspirational: only the prune script ever
  grew a test, though CLAUDE.md claimed both had one.

So the mechanism lives here, as ordinary typed library functions, and
`ai-style data` (a **hidden** click group — see `bin/ai_style_data.py`) is the
thin shell over it. Hidden keeps the promise the carve-out was really making:
the `--help` tree stays the pipeline, uncluttered by commands nobody should be
migrating with twice.

Everything here is *deletable*. A migration is spent once the only corpus that
exists has been through it (`restamp_style_system` and `prune_stale_synth_pairs`
went that way in 0.3); users migrate on version increment rather than carrying
dead code. Delete the function, its command and its test together.

Two live migrations:

`migrate_captures` / `migrate_snapshots`
    The superseded `captures/<id>.json` blobs -> `captures/<id>/` directories.
    The only remaining reader of the old format, which is why it parses the
    blobs itself rather than calling `capture.read_archive` on them.

`find_inventing`
    The anchor-invention cull (D32), for the corpus that predates
    `--drop-invented-anchors`.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from stylebot.capture import read_archive, write_archive
from stylebot.infer import invented_anchors
from stylebot.jsonl import write_jsonl_atomic
from stylebot.pairs import extract_slop, extract_target, iter_pairs

#: Where a migration reports progress. Injectable so the CLI can route it
#: through `click.echo` and a test can capture it without touching stdout.
Log = Callable[[str], None]


# ---------------------------------------------------------------------------
# Legacy capture archives -> directories
#
# The old format put whole documents inside a one-line JSON file as escaped
# strings:
#
#     captures/<capture_id>.json      {capture_id, source, snapped_at, whole,
#                                      before_text, after_text}
#     snapshots/<path>.qmd.json       {source, snapped_at, content}
#
# which is unreadable, unopenable and undiffable — perverse, given the
# before->after diff IS the training signal. The new format puts the documents
# on disk as ordinary files:
#
#     captures/<capture_id>/{before.md, after.md, meta.json}
#     snapshots/<path>.qmd/{content.md, meta.json}
#
# `.md` and not `.qmd` deliberately: the blog's pre-commit hook reformats staged
# `*.qmd` (which would destroy the verbatim record), `qmd` search indexes
# `**/*.qmd` (which would return captured slop as if Dan wrote it), and
# `iter_targets` samples `**/*.qmd` as synthesis targets. `.md` is invisible to
# all three.
# ---------------------------------------------------------------------------


class MigrationError(RuntimeError):
    """A migration refused to continue; nothing further was written."""


def migrate_captures(cap_dir: Path, *, apply: bool, log: Log = print) -> tuple[int, int]:
    """Convert `captures/*.json`. Returns `(converted, skipped)`.

    Each conversion writes the new directory first and unlinks the legacy file
    only once the directory reads back with byte-identical text, so an
    interrupted run leaves both forms rather than neither — nothing is lost, but
    re-run it until `captures/` holds no `*.json`: since 0.3 `iter_archives`
    reads only directories and REFUSES (rather than silently skipping) a
    leftover blob.
    """
    converted = skipped = 0
    if not cap_dir.is_dir():
        return 0, 0
    for path in sorted(cap_dir.glob("*.json")):
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log(f"  SKIP {path.name}: unreadable ({exc})")
            skipped += 1
            continue
        cap_id = blob.get("capture_id") or path.stem
        target = cap_dir / cap_id
        if target.exists():
            log(f"  SKIP {path.name}: {target.name}/ already exists")
            skipped += 1
            continue
        log(
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
            raise MigrationError(f"round-trip mismatch for {path.name}; legacy file kept")
        path.unlink()  # only after the new form verifies
        converted += 1
    return converted, skipped


def migrate_snapshots(snap_root: Path, *, apply: bool, log: Log = print) -> tuple[int, int]:
    """Convert `snapshots{,-closed}/**/<name>.json`. Returns `(converted, skipped)`."""
    converted = skipped = 0
    if not snap_root.is_dir():
        return 0, 0
    for path in sorted(snap_root.rglob("*.json")):
        if path.name == "meta.json":
            continue  # already migrated
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log(f"  SKIP {path}: unreadable ({exc})")
            skipped += 1
            continue
        # snapshots/post/foo.qmd.json -> snapshots/post/foo.qmd/
        target = path.with_suffix("")
        if target.exists():
            log(f"  SKIP {path}: {target.name}/ already exists")
            skipped += 1
            continue
        content = blob.get("content") or ""
        log(f"  {path.relative_to(snap_root)} -> {target.name}/  ({len(content)} chars)")
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
            raise MigrationError(f"round-trip mismatch for {path}; legacy file kept")
        path.unlink()
        converted += 1
    return converted, skipped


def migrate_capture_archives(data_dir: Path, *, apply: bool, log: Log = print) -> int:
    """Run both archive conversions over a data dir. Returns items converted.

    Dry-run unless `apply`, in which case every conversion is verified by
    reading it back before the legacy file is removed.
    """
    if not data_dir.is_dir():
        raise MigrationError(f"no such directory: {data_dir}")
    total = 0
    log(f"captures/  ({'APPLY' if apply else 'dry run'})")
    c, s = migrate_captures(data_dir / "captures", apply=apply, log=log)
    log(f"  -> {c} converted, {s} skipped")
    total += c
    for name in ("snapshots", "snapshots-closed"):
        log(f"{name}/")
        c, s = migrate_snapshots(data_dir / name, apply=apply, log=log)
        log(f"  -> {c} converted, {s} skipped")
        total += c
    if not apply and total:
        log(f"\ndry run: {total} item(s) would convert. Re-run with --apply.")
    return total


# ---------------------------------------------------------------------------
# Anchor-invention cull (D32)
#
# Reverse-edit synthesis produces these directly: a generator that drops a link
# while "improving" a paragraph becomes, after the swap, a training example
# reading *given text without links, produce text with these links*. Measured
# over the 4,595-pair corpus, 855 pairs (18.6%) did exactly that — half of the
# invented anchors being external URLs whose destination no model can know.
# Fabrication outnumbers the loss case the old guard checked by roughly 7:1
# (stylebot#1, D32).
#
# `--drop-invented-anchors` now gates this at synthesis time and defaults on, so
# no run after 0.3.0 writes such a pair. This is the one-off for the corpus that
# predates the gate. `infer.is_anchor_faithful` is the same predicate as a
# training-assembly `selector`, for anyone who would rather filter than cull.
# ---------------------------------------------------------------------------


def anchor_kind(key: str) -> str:
    """Coarse taxonomy of an invented anchor, for the report."""
    if key.startswith("http"):
        return "external URL"
    if key.startswith("@"):
        return "citation key"
    if key.startswith("#"):
        return "same-page anchor"
    return "internal link"


def find_inventing(pairs_path: Path, *, include_real: bool) -> tuple[list[dict], dict]:
    """`(records_to_keep, report)`. Reads only; decides nothing about writing.

    **Synthetic and real pairs are different propositions, so `include_real`
    defaults off wherever this is called.** A synthetic cell that goes is
    *regenerable*: it leaves its `synth_key` missing, so the next `ai-style
    synth` refills it — under the gate this time, and with `--gate-retries` most
    cells clear on a redraw. A real pair is a captured record of an edit Dan
    actually made and nothing can bring it back but git. They also fail
    differently: 51.4% of what the synthetic stratum invents is an external URL
    (a generator artifact), against 11.5% for the real stratum, which is
    dominated by internal cross-references (53.8%) and citation keys (26.9%) —
    Dan wiring a note into his own graph while editing. That is genuine
    practice, even though the styler cannot learn to reproduce it.
    """
    keep: list[dict] = []
    kinds: Counter[str] = Counter()
    dropped = Counter({"synthetic": 0, "real": 0})
    spared_real = 0
    samples: list[tuple[str, list[str]]] = []
    # Coverage accounting: a target losing EVERY pair goes back to uncovered, so
    # a plain re-run regenerates it. That is the intended repair, not a loss —
    # but it is the number that says how much regeneration the cull implies.
    per_target: dict[str, list[int]] = {}
    total = 0

    for rec in iter_pairs(pairs_path):
        total += 1
        meta = rec.get("meta") or {}
        synthetic = bool(meta.get("synthetic"))
        invented = invented_anchors(extract_slop(rec), extract_target(rec))
        if synthetic:
            seen = per_target.setdefault(extract_target(rec), [0, 0])
            seen[0] += 1
            seen[1] += bool(invented)
        if not invented:
            keep.append(rec)
            continue
        if not synthetic and not include_real:
            spared_real += 1
            keep.append(rec)
            continue
        dropped["synthetic" if synthetic else "real"] += 1
        kinds.update(anchor_kind(k) for k in invented)
        if len(samples) < 5:
            samples.append((meta.get("source") or "?", sorted(set(invented))[:3]))

    wiped = sum(1 for n, bad in per_target.values() if n and n == bad)
    return keep, {
        "records": total,
        "dropped": dict(dropped),
        "spared_real": spared_real,
        "kinds": kinds.most_common(),
        "targets_wiped": wiped,
        "targets": len(per_target),
        "samples": samples,
    }


def write_pruned(pairs_path: Path, keep: list[dict]) -> Path:
    """Rewrite `pairs_path` with `keep`, returning the path of the `.bak`.

    Atomic (a sibling temp file) and backed up, so an interrupted run cannot
    leave a half-written corpus. Sidecars are NOT touched: `scores.jsonl` has
    its own pruner (`eval.prune_orphan_scores`), and `reasoning.jsonl` is a
    diagnostic whose orphans cost nothing.
    """
    backup = pairs_path.with_suffix(pairs_path.suffix + ".bak")
    shutil.copy2(pairs_path, backup)
    write_jsonl_atomic(pairs_path, keep)
    return backup
