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

Three live migrations:

`plan_stale_prune`
    The corpus consequence of a policy change (W12): synthetic cells no current
    cut reproduces, and synthetic cells sitting on frozen-eval prose.

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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from stylebot.capture import read_archive, write_archive
from stylebot.infer import invented_anchors
from stylebot.jsonl import write_jsonl_atomic
from stylebot.pairs import extract_slop, extract_target, iter_pairs
from stylebot.splits import canonical_source, role_of
from stylebot.targets import ChunkPolicy, iter_targets

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


# ---------------------------------------------------------------------------
# Stale-cell prune (W12): orphans, and eval contamination
#
# Two independent reasons a synthetic cell on disk no longer matches the policy
# that would produce it, both introduced by fixes rather than by rot:
#
# 1. ORPHANS. `synth_key` hashes the target TEXT, so a segmentation or policy
#    change orphans a cell rather than colliding with it: the pair stays, and no
#    future `synth` run regenerates it, because no cut asks for that text any
#    more. This is the situation the 2026-08-10 `prune_stale_synth_pairs.py`
#    handled for the unanchored-`$$` defect (A5) and was deleted after use.
#
# 2. EVAL CONTAMINATION. Until W0 gave `role_of` an alias map and taught `synth`
#    to refuse the eval stratum, flagless runs cut frozen-eval prose as training
#    targets. The guard stops it recurring; it cannot remove what is there.
#
# **Real captures are never orphans, and the asymmetry is structural rather than
# a kindness.** A capture's target comes from a paragraph diff of an edit
# archive (`capture.recut_archives`), never from `iter_targets`, so asking
# whether the chunker reproduces it is a category error — it was not chunked.
# `find_orphans` therefore reports real pairs as UNCHECKED, not as clean, and
# refuses to drop one whatever the flags say. The same asymmetry, for the
# different reason that a captured edit is irreplaceable, spares real pairs on
# eval posts: `ai-style log save` warns on an eval post rather than refusing,
# because capturing yourself de-slopping held-out prose is real editing work.
# ---------------------------------------------------------------------------


#: A corpus cell's chunk diet: `(chunk_mode, merge_max_chars)`. Records written
#: before those stamps existed (every pair from the 2026-07 batch) carry
#: neither; they are re-cut as section-bounded at the policy's own budget, which
#: re-derives 99.8% of them — an inference from agreement, not provenance.
Diet = tuple[str, int]

SECTION_BOUNDED = "section-bounded"
CROSS_HEADING = "cross-heading"


def cell_diet(meta: dict, policy: ChunkPolicy) -> Diet:
    """The `(mode, budget)` a record was cut under, defaulting the legacy stamps."""
    mode = meta.get("chunk_mode") or SECTION_BOUNDED
    return mode, int(meta.get("merge_max_chars") or policy.merge_max_chars)


def diet_policy(policy: ChunkPolicy, diet: Diet) -> ChunkPolicy:
    """`policy` re-pinned to one cell's diet.

    Asking every cell for the live budget is what made the corpus-stability
    harness read a third of the corpus as broken: a 4500-char merge is not
    reproducible under a 1500-char budget, and a cross-heading target is not
    reproducible under section-bounded packing. Each regime is asked only for
    the cells it cut.
    """
    mode, budget = diet
    return replace(policy, cross_heading=mode == CROSS_HEADING, merge_max_chars=budget)


@dataclass
class OrphanReport:
    """What a re-cut says about the corpus, before anything is deleted."""

    records: int = 0
    synthetic: int = 0
    #: Real captures, which `iter_targets` never produced and this cannot judge.
    real_unchecked: int = 0
    orphans: int = 0
    #: Orphans whose source post is missing under `--blog-root` (renamed without
    #: an alias, or deleted). Distinguished because the remedy is a splits alias
    #: or a `git mv`, not a prune.
    orphans_missing_source: int = 0
    #: `{source: orphaned pairs}` and `{(mode, budget): orphaned pairs}`.
    per_post: Counter = field(default_factory=Counter)
    per_diet: Counter = field(default_factory=Counter)
    #: Cells (distinct target bodies) rather than pairs, for comparison with the
    #: corpus-stability floors, which count bodies.
    cells: int = 0
    orphan_cells: int = 0
    #: A target losing every pair goes back to uncovered, so a re-run refills it
    #: — but only where the diet is still live policy.
    sources_touched: int = 0


def _corpus_cells(
    pairs_path: Path, policy: ChunkPolicy, splits: dict | None
) -> dict[tuple[str, Diet], set[str]]:
    """`{(canonical source, diet): {target body, ...}}` over the synthetic pairs."""
    cells: dict[tuple[str, Diet], set[str]] = {}
    for rec in iter_pairs(pairs_path):
        meta = rec.get("meta") or {}
        if not meta.get("synthetic") or not meta.get("source"):
            continue
        source = canonical_source(meta["source"], splits) if splits else meta["source"]
        cells.setdefault((source, cell_diet(meta, policy)), set()).add(extract_target(rec))
    return cells


def find_orphans(
    pairs_path: Path,
    *,
    blog_root: Path,
    policy: ChunkPolicy,
    splits: dict | None = None,
) -> tuple[set[tuple[str, str]], OrphanReport]:
    """Cells the current cut no longer produces: `({(source, target body)}, report)`.

    Re-cuts each source ONCE per diet present in the corpus for it, and calls a
    pair orphaned when its target body is absent from the derived set. The
    identity returned is `(canonical source, target body)` rather than a
    `synth_key`, because several generators' pairs share one target and a cell
    that stopped existing takes all of them.

    Reads the blog's WORKING TREE, deliberately: the question a prune asks is
    "would the next synth run regenerate this?", and the next synth run reads
    the tree. That conflates a policy change with Dan having edited the post
    since — which is why the caller should also read the same measurement
    against a pinned revision (the blog's `tests/test_corpus_stability.py`) to
    separate the two. Prose drift is not a reason to delete a pair.

    `splits` is optional and used only for `canonical_source`, so a post renamed
    since generation is re-cut at its new path rather than counted as missing.
    """
    cells = _corpus_cells(pairs_path, policy, splits)
    derived: dict[tuple[str, Diet], set[str] | None] = {}
    orphan_cells: set[tuple[str, str]] = set()
    report = OrphanReport()

    for (source, diet), bodies in cells.items():
        report.cells += len(bodies)
        path = Path(blog_root) / source
        if not path.is_file():
            derived[(source, diet)] = None
            orphan_cells |= {(source, body) for body in bodies}
            report.orphan_cells += len(bodies)
            continue
        found = {t.text for t in iter_targets(files=[path], policy=diet_policy(policy, diet))}
        derived[(source, diet)] = found
        missing = bodies - found
        orphan_cells |= {(source, body) for body in missing}
        report.orphan_cells += len(missing)

    for rec in iter_pairs(pairs_path):
        report.records += 1
        meta = rec.get("meta") or {}
        if not meta.get("synthetic"):
            report.real_unchecked += 1
            continue
        report.synthetic += 1
        source = (
            canonical_source(meta.get("source") or "", splits)
            if splits
            else (meta.get("source") or "")
        )
        if (source, extract_target(rec)) not in orphan_cells:
            continue
        report.orphans += 1
        report.per_post[source] += 1
        report.per_diet[cell_diet(meta, policy)] += 1
        if derived.get((source, cell_diet(meta, policy))) is None:
            report.orphans_missing_source += 1
    report.sources_touched = len(report.per_post)
    return orphan_cells, report


@dataclass
class EvalReport:
    """Synthetic pairs our own generators cut from frozen-eval prose."""

    synthetic_dropped: int = 0
    real_spared: int = 0
    per_post: Counter = field(default_factory=Counter)
    per_generator: Counter = field(default_factory=Counter)


def find_eval_contamination(pairs_path: Path, splits: dict) -> tuple[set[int], EvalReport]:
    """Line indices of synthetic pairs whose source resolves to role `eval`.

    Indices rather than keys because 22 duplicate `record_id`s lived in this
    corpus until W8, and a prune that keys on a non-unique id is how you delete
    the wrong record. The corpus is read in file order twice, so an index is
    stable within one run and meaningless outside it — which is the point.
    """
    drop: set[int] = set()
    report = EvalReport()
    for i, rec in enumerate(iter_pairs(pairs_path)):
        meta = rec.get("meta") or {}
        if role_of(meta.get("source") or "", splits) != "eval":
            continue
        if not meta.get("synthetic"):
            report.real_spared += 1
            continue
        drop.add(i)
        report.synthetic_dropped += 1
        report.per_post[canonical_source(meta.get("source") or "", splits)] += 1
        report.per_generator[meta.get("generator") or "?"] += 1
    return drop, report


@dataclass
class PrunePlan:
    """What a prune would remove and what it would leave. Decides nothing."""

    keep: list[dict] = field(default_factory=list)
    orphan_report: OrphanReport | None = None
    eval_report: EvalReport | None = None
    n_orphans: int = 0
    n_eval: int = 0
    n_both: int = 0  # pairs in both sets
    n_dropped: int = 0
    #: `{"total": n, "real": n, "synthetic": n, "posts": n}` for the corpus as it
    #: is, after each prune alone, and after both.
    composition: dict[str, dict[str, int]] = field(default_factory=dict)
    #: The same four corpora as record lists, so a caller can measure anything
    #: else about them — `styler_passage_scale` above all, since "does this help
    #: the dilution the budget sweep caused" is the question that decides whether
    #: the prune is worth doing at all, and it is not answerable from counts.
    variants: dict[str, list[dict]] = field(default_factory=dict)


def _composition(records: Iterable[dict]) -> dict[str, int]:
    total = real = 0
    posts: set[str] = set()
    for rec in records:
        meta = rec.get("meta") or {}
        total += 1
        real += not meta.get("synthetic")
        posts.add(meta.get("source") or "?")
    return {"total": total, "real": real, "synthetic": total - real, "posts": len(posts)}


def plan_stale_prune(
    pairs_path: Path,
    *,
    blog_root: Path | None,
    policy: ChunkPolicy | None,
    splits: dict | None,
    drop_orphans: bool = True,
    drop_eval: bool = True,
) -> PrunePlan:
    """Plan the W12 prune: read-only, and the ONLY place the two sets combine.

    Returns the records to keep plus the accounting the dry run prints, so
    `--apply` writes exactly what the dry run reported — the two paths differ by
    one `write_pruned` call and nothing else.

    Neither prune ever removes a real capture. The orphan set cannot contain one
    (see the module comment); the eval set spares them by rule.
    """
    plan = PrunePlan()
    orphan_cells: set[tuple[str, str]] = set()
    eval_rows: set[int] = set()

    if drop_orphans:
        if blog_root is None:
            raise MigrationError("orphan detection needs a --blog-root to re-cut against")
        orphan_cells, plan.orphan_report = find_orphans(
            pairs_path, blog_root=blog_root, policy=policy, splits=splits
        )
    if drop_eval:
        if splits is None:
            raise MigrationError("eval contamination needs a --splits to resolve roles")
        eval_rows, plan.eval_report = find_eval_contamination(pairs_path, splits)

    records = list(iter_pairs(pairs_path))
    is_orphan: list[bool] = []
    for i, rec in enumerate(records):
        meta = rec.get("meta") or {}
        source = (
            canonical_source(meta.get("source") or "", splits)
            if splits
            else (meta.get("source") or "")
        )
        orphan = bool(meta.get("synthetic")) and (source, extract_target(rec)) in orphan_cells
        is_orphan.append(orphan)
        plan.n_orphans += orphan
        plan.n_eval += i in eval_rows
        plan.n_both += orphan and i in eval_rows

    keep = [r for i, r in enumerate(records) if not is_orphan[i] and i not in eval_rows]
    plan.keep = keep
    plan.n_dropped = len(records) - len(keep)
    # Only the variants actually computed. A "minus orphans" row that reads
    # identical to "now" because orphan detection was switched off is a number
    # saying "there are none", which is the opposite of what happened.
    plan.variants = {"now": records}
    if drop_orphans:
        plan.variants["minus orphans"] = [r for i, r in enumerate(records) if not is_orphan[i]]
    if drop_eval:
        plan.variants["minus eval synthetic"] = [
            r for i, r in enumerate(records) if i not in eval_rows
        ]
    if drop_orphans and drop_eval:
        plan.variants["minus both"] = keep
    plan.composition = {label: _composition(recs) for label, recs in plan.variants.items()}
    return plan


def styler_passage_scale(
    records: Sequence[dict], *, splits: dict, selector: Callable[[dict], bool] | None = None
) -> dict:
    """Passage accounting for the styler train set a corpus would produce.

    The number W9 made the point of: real pairs are a far larger share of train
    *passages* than of train *pairs*, because the budget sweep walked the
    synthetic side several times and the captures once. A prune is worth doing
    only if it moves that, so a prune report has to be able to say so.

    Deliberately NOT a call into `train.assemble_training_corpus`: that
    validates and hashes a FILE, and this must answer for a record list that
    does not exist on disk yet. It reproduces the same role filter and the same
    passage grouping, and nothing else — no val split, since the dilution
    question is about the train side.
    """
    from stylebot.train import passage_groups

    kept = [
        rec
        for rec in records
        if role_of((rec.get("meta") or {}).get("source") or "?", splits) == "styler"
        and (selector is None or selector(rec))
    ]
    groups = passage_groups(kept)
    real_pairs = sum(1 for rec in kept if not (rec.get("meta") or {}).get("synthetic"))
    real_passages = sum(
        1 for g in groups if any(not (r.get("meta") or {}).get("synthetic") for r in g)
    )
    return {
        "pairs": len(kept),
        "real_pairs": real_pairs,
        "passages": len(groups),
        "real_passages": real_passages,
        "mean_pairs_per_passage": round(len(kept) / len(groups), 2) if groups else 0.0,
        "max_pairs_per_passage": max((len(g) for g in groups), default=0),
        "real_pair_share": round(100 * real_pairs / len(kept), 2) if kept else 0.0,
        "real_passage_share": round(100 * real_passages / len(groups), 2) if groups else 0.0,
    }
