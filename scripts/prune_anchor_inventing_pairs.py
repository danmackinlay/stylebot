#!/usr/bin/env python3
r"""Drop pairs whose target carries a link/URL/citation its slop side never had.

Reverse-edit synthesis produces these directly: a generator that drops a link
while "improving" a paragraph becomes, after the swap, a training example
reading *given text without links, produce text with these links*. Measured over
the 4,595-pair corpus, 855 pairs (18.6%) do exactly that — half of the invented
anchors being external URLs whose destination no model can know. Fabrication
outnumbers the loss case the old guard checked by roughly 7:1 (stylebot#1, D32).

`--drop-invented-anchors` now gates this at synthesis time and defaults on, so
no run after 0.3.0 writes such a pair. This is the one-off for the corpus that
predates the gate. `infer.is_anchor_faithful` is the same predicate as a
training-assembly `selector`, for anyone who would rather filter than cull.

**Synthetic and real pairs are different propositions, so the default culls
only the synthetic ones.** A synthetic cell that goes is *regenerable*: it
leaves its `synth_key` missing, so the next `ai-style synth` refills it — under
the gate this time, and with `--gate-retries` most cells clear on a redraw. A
real pair is a captured record of an edit Dan actually made and nothing can
bring it back but git. They also fail differently: 51.4% of what the synthetic
stratum invents is an external URL (a generator artifact), against 11.5% for
the real stratum, which is dominated by internal cross-references (53.8%) and
citation keys (26.9%) — Dan wiring a note into his own graph while editing.
That is genuine practice, even though the styler cannot learn to reproduce it.
Pass `--include-real` to cull those too; they are excludable from training
without culling via the selector above.

Usage (dry run by default; nothing is written without --apply):

    uv run python scripts/prune_anchor_inventing_pairs.py \
        ~/Source/livingthing/_training_pairs/pairs.jsonl

    uv run python scripts/prune_anchor_inventing_pairs.py \
        ~/Source/livingthing/_training_pairs/pairs.jsonl --apply

Writes atomically via a sibling temp file and keeps a `.bak` of the original,
so an interrupted run cannot leave a half-written corpus. Sidecars are NOT
touched: `scores.jsonl` has its own pruner (`eval.prune_orphan_scores`), and
`reasoning.jsonl` is a diagnostic whose orphans cost nothing. Both counts are
reported so the decision is yours.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

from stylebot.infer import invented_anchors
from stylebot.jsonl import write_jsonl_atomic
from stylebot.pairs import extract_slop, extract_target, iter_pairs


def _anchor_kind(key: str) -> str:
    if key.startswith("http"):
        return "external URL"
    if key.startswith("@"):
        return "citation key"
    if key.startswith("#"):
        return "same-page anchor"
    return "internal link"


def find_inventing(pairs_path: Path, *, include_real: bool) -> tuple[list[dict], dict]:
    """`(records_to_keep, report)`. Reads only; decides nothing about writing."""
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
        kinds.update(_anchor_kind(k) for k in invented)
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("pairs", type=Path, help="path to pairs.jsonl")
    p.add_argument("--apply", action="store_true", help="rewrite pairs.jsonl (default: dry run)")
    p.add_argument(
        "--include-real",
        action="store_true",
        help="also cull captured (non-synthetic) pairs. They cannot be regenerated.",
    )
    args = p.parse_args(argv)

    if not args.pairs.is_file():
        print(f"no such file: {args.pairs}", file=sys.stderr)
        return 1

    keep, report = find_inventing(args.pairs, include_real=args.include_real)
    n_dropped = sum(report["dropped"].values())

    print(f"records          : {report['records']}")
    print(
        f"dropped          : {n_dropped} "
        f"({report['dropped']['synthetic']} synthetic, {report['dropped']['real']} real)"
    )
    if report["spared_real"]:
        print(
            f"real, left alone : {report['spared_real']} "
            "(captured records — pass --include-real to cull, or exclude them at "
            "training assembly with infer.is_anchor_faithful)"
        )
    print(f"remaining        : {len(keep)}")
    for kind, n in report["kinds"]:
        print(f"    {kind:18} {n}")
    print(
        f"targets emptied  : {report['targets_wiped']} of {report['targets']} "
        "— these go back to uncovered, so the next synth run refills them"
    )
    for source, anchors in report["samples"]:
        print(f"  e.g. {source}: invented {', '.join(anchors)}")

    if not n_dropped:
        print("\nnothing to prune.")
        return 0
    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply to prune.")
        return 0

    backup = args.pairs.with_suffix(args.pairs.suffix + ".bak")
    shutil.copy2(args.pairs, backup)
    write_jsonl_atomic(args.pairs, keep)
    print(f"\npruned {n_dropped} pair(s) -> {args.pairs} (backup at {backup})")
    print("next: `ai-style synth --gate-retries 2` to refill the emptied cells,")
    print("      then `ai-style eval --prune-orphans` if you keep scores.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
