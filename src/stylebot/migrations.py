"""One-off data migrations: rewrite corpus state already on disk to a new schema.

Ordinary typed library functions, with `ai-style data` (a **hidden** click
group, `bin/ai_style_data.py`) as the thin shell over them. Why they are not
loose files under `scripts/`, and why the group is hidden: D38.

Everything here is *deletable*. A migration is spent once the only corpus that
exists has been through it (`restamp_style_system` and `prune_stale_synth_pairs`
went that way in 0.3, the archive and shard conversions in X1); users migrate
on version increment rather than carrying dead code. Delete the function, its
command and its test together (D38).

Two live migrations:

`plan_stale_prune`
    The corpus consequence of a policy change (D53): synthetic cells no current
    cut reproduces, and synthetic cells sitting on frozen-eval prose.
`backfill_draws`
    The one-off lift of today's pairs and reject side files into draws (D84).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from stylebot import corpus, draws
from stylebot.draws import Draw
from stylebot.errors import RunError
from stylebot.jsonl import iter_jsonl
from stylebot.pairs import backup_corpus, extract_target, iter_pairs
from stylebot.splits import canonical_source, role_of
from stylebot.targets import ChunkPolicy, iter_targets


class MigrationError(RunError):
    """A migration refused to continue; nothing further was written."""


# ---------------------------------------------------------------------------
# The corpus rewriter the prunes share
# ---------------------------------------------------------------------------


def write_pruned(pairs_path: Path, dropped: list[dict]) -> Path:
    """Remove `dropped` from the corpus at `pairs_path`. Returns the backup path.

    What the plan REMOVES, not what it keeps: the corpus is re-read under the
    lock, where "keep exactly this list" would also delete whatever was appended
    while the user read the dry run (D60). Records match by serialised line, one
    occurrence per planned drop. Sidecars are NOT touched — `scores.jsonl` has
    its own pruner (`eval.prune_orphan_scores`).
    """
    backup = backup_corpus(pairs_path)
    budget = Counter(corpus.line_of(rec) for rec in dropped)

    def prune(records: list[dict]) -> list[dict]:
        kept: list[dict] = []
        for rec in records:
            line = corpus.line_of(rec)
            if budget[line]:
                budget[line] -= 1
                continue
            kept.append(rec)
        return kept

    corpus.rewrite_with(pairs_path, prune)
    return backup


# ---------------------------------------------------------------------------
# Stale-cell prune: orphans, and eval contamination (D53)
#
# Two independent reasons a synthetic cell on disk no longer matches the policy
# that would produce it: an ORPHAN (`synth_key` hashes the target TEXT, so a
# policy change orphans a cell rather than colliding with it, and no future run
# regenerates it), and EVAL CONTAMINATION (flagless runs cut frozen-eval prose
# before `synth` learned to refuse the stratum — D34 stops it recurring but
# cannot remove what is there).
#
# **Real captures are never orphans**, structurally: a capture's target comes
# from a paragraph diff of an edit archive, never from `iter_targets`, so asking
# whether the chunker reproduces it is a category error. `find_orphans` reports
# real pairs as UNCHECKED, not clean, and refuses to drop one whatever the flags
# say.
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

    Asking every cell for the live budget made the corpus-stability harness read
    a third of the corpus as broken: a 4500-char merge is not reproducible under
    a 1500-char budget. Each regime is asked only for the cells it cut.
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

    Re-cuts each source ONCE per diet present in the corpus for it, calling a
    pair orphaned when its target body is absent from the derived set. The
    identity is `(canonical source, target body)` rather than a `synth_key`,
    since several generators' pairs share one target.

    Reads the blog's WORKING TREE deliberately — a prune asks "would the next
    synth run regenerate this?" — which conflates a policy change with prose
    drift; separate the two against a pinned revision (D53). `splits` is used
    only for `canonical_source`, so a renamed post is re-cut at its new path.
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
    #: What the prune removes — the half `write_pruned` applies; see it for why.
    dropped: list[dict] = field(default_factory=list)
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
    plan.dropped = [r for i, r in enumerate(records) if is_orphan[i] or i in eval_rows]
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

    Real pairs are a far larger share of train *passages* than of train *pairs*,
    because the budget sweep walked the synthetic side several times and the
    captures once (D43); a prune is worth doing only if it moves that.

    Deliberately NOT a call into `train.assemble_training_corpus`, which
    validates and hashes a FILE: this answers for a record list not yet on disk.
    Same role filter and passage grouping, no val split.
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


# --- Backfill: pairs and the reject side files -> draws (W4 step 3, D84) ---

#: `record -> (blog_sha, blog_dirty, span)`; the blog's git search, supplied (D45).
Resolve = Callable[[dict], tuple[str | None, bool, tuple[int, int] | None]]

#: The reject logs a synth run kept beside `pairs/` before draws existed.
SIDE_FILE_NAMES = ("lost_draws.jsonl", "gate_drops.jsonl")

#: Row keys a tombstone lifts into `Draw` fields; the rest stay in `gen`.
_ROW_FIELDS = frozenset({"at", "source", "kind", "gate"})


@dataclass
class BackfillReport:
    """What a backfill read, and would write or wrote."""

    records: int = 0
    resolved: int = 0
    unresolved: int = 0
    superseded: int = 0
    tombstones: Counter = field(default_factory=Counter)  # by outcome
    #: Draws re-keyed by `resolve` onto a cell an earlier draw already holds.
    duplicate_ids: int = 0
    shards: list[Path] = field(default_factory=list)
    applied: bool = False


def tombstone_from_row(row: dict) -> Draw:
    """A side-file row as a rejected draw, text absent (D84).

    A row has no prompt id and no span, so `draw_id` hashes the whole row.
    """
    if row.get("kind") == "error":
        cls = str(row.get("reason") or "").split(":", 1)[0].strip()
        outcome = f"error:{cls or 'Exception'}"
    else:
        outcome = f"gate:{row.get('gate') or 'unknown'}"
    blob = json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return Draw(
        draw_id=hashlib.sha256(blob).hexdigest()[:16],
        source=row.get("source"),
        gen={k: v for k, v in row.items() if k not in _ROW_FIELDS} or None,
        outcome=outcome,
        drawn_at=row.get("at"),
    )


def _lift_pair(record: dict, resolve: Resolve | None, superseded: Mapping[str, str]) -> Draw:
    """`draw_from_pair`, re-keyed on its cell once `resolve` finds a sha or span."""
    draw = draws.draw_from_pair(record)
    meta = record.get("meta") or {}
    if resolve is not None:
        draw.blog_sha, draw.blog_dirty, draw.span = resolve(record)
        if draw.blog_sha is not None or draw.span is not None:
            gen = meta.get("gen") or {}
            draw.draw_id = draws.draw_id(
                draw.source,
                draw.blog_sha,
                draw.span,
                meta.get("generator"),
                meta.get("slop_strategy"),
                gen.get("prompt_id"),
                gen.get("reasoning_effort"),
                int(gen.get("replicate") or 0),
            )
    capture_id = meta.get("capture_id")
    if not meta.get("synthetic") and capture_id in superseded:
        draw.superseded_by = superseded[capture_id]
    return draw


def _report_lines(report: BackfillReport, draws_dir: Path) -> Iterable[str]:
    yield f"pair records            : {report.records}"
    yield f"  with a blog sha       : {report.resolved}"
    yield f"  unresolved            : {report.unresolved}"
    yield f"  superseded            : {report.superseded}"
    yield f"tombstones              : {sum(report.tombstones.values())}"
    for outcome, n in sorted(report.tombstones.items()):
        yield f"  {outcome:<22}: {n}"
    yield f"duplicate draw ids      : {report.duplicate_ids}"
    if report.applied:
        yield f"wrote {len(report.shards)} shard(s) -> {draws_dir}"
    else:
        yield "dry run — nothing written. Re-run with --apply to write draws."


def backfill_draws(
    corpus_path: str | Path,
    draws_dir: str | Path,
    *,
    side_files: Sequence[Path] = (),
    resolve: Resolve | None = None,
    superseded: Mapping[str, str] | None = None,
    apply: bool,
    log: Callable[[str], object] = print,
) -> BackfillReport:
    """Every pair record and side-file row of `corpus_path`, as draws; never writes it.

    `superseded` is `{capture_id: successor capture_id}`. `apply` appends month
    shards under an empty `draws_dir`; a dry run only counts.
    """
    draws_dir = Path(draws_dir)
    if apply and draws_dir.is_dir() and any(draws_dir.iterdir()):
        raise MigrationError(f"{draws_dir} is not empty; a backfill writes a fresh directory")
    report = BackfillReport(applied=apply)
    seen: Counter = Counter()
    shards: dict[Path, None] = {}

    def emit(draw: Draw) -> None:
        seen[draw.draw_id] += 1
        if apply:
            shards[draws.append(draws_dir, draw)] = None

    for record in corpus.iter_records(corpus_path):
        draw = _lift_pair(record, resolve, superseded or {})
        report.records += 1
        report.resolved += draw.blog_sha is not None
        report.unresolved += draw.blog_sha is None
        report.superseded += draw.superseded_by is not None
        emit(draw)
    for path in side_files:
        for row in iter_jsonl(path):
            draw = tombstone_from_row(row)
            report.tombstones[draw.outcome] += 1
            emit(draw)
    report.duplicate_ids = sum(n - 1 for n in seen.values() if n > 1)
    report.shards = list(shards)
    for line in _report_lines(report, draws_dir):
        log(line)
    return report
