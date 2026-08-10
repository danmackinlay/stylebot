#!/usr/bin/env python3
r"""Drop synthetic pairs whose target text the segmenter can no longer produce.

`pairs.jsonl` is append-only and `synthesize_pairs` resumes on `synth_key`, a
function of the target text. So when the segmenter changes what a target *is*,
re-running synth ADDS the corrected pair and leaves the superseded one in place
forever: the corpus keeps teaching the old, wrong cut. No existing tool removes
those — `eval.prune_orphan_scores` prunes scores, and `log tidy` keys on
`captured_at` and does not apply to synth records. This is that tool.

**The 2026-08-10 migration it exists for.** `lib.MATH_BLOCK_RE` was not
line-anchored, so a mid-sentence `$$k$$` was excised as if it were a display
block and the remnants joined: `high-$$d$$ behaviour` became `high- behaviour`
*inside a training target*. Those pairs teach the styler to emit truncated
sentences. `stylebot.markdown` now anchors display math to line boundaries,
which orphans exactly the damaged cells.

**The safety property.** A record is dropped only when its target is
unproducible NOW but WAS producible under the superseded segmentation. Targets
that are unproducible under both — ordinary blog prose drift since generation,
which is most of the gap — are reported and left alone: they are not this
migration's business, and a blanket "prune everything unproducible" would eat
thousands of good pairs the first time it met a policy it was not run under.

Chunk policy is yours to supply, because it is yours: stylebot owns the
mechanism and the blog owns the numbers (see `livingthing.training_targets`).
Pass the SAME policy the corpus was generated under, or you will measure a
different corpus than the one on disk.

Usage (dry run by default; nothing is written without --apply):

    uv run python scripts/prune_stale_synth_pairs.py \
        ~/Source/livingthing/_training_pairs/pairs.jsonl \
        --blog-root ~/Source/livingthing \
        --merge --merge-max-chars 1500 --min-chars 200 --drop-list-items \
        --heading-context immediate --no-cross-heading \
        --ignore-marker 'STUB' --stop-at-header '## Incoming'

Writes atomically via a sibling temp file and keeps a `.bak` of the original,
so an interrupted run cannot leave a half-written corpus.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from stylebot import markdown, targets
from stylebot.jsonl import write_jsonl_atomic
from stylebot.pairs import extract_target, iter_pairs


@contextmanager
def _legacy_segmentation() -> Iterator[None]:
    """Run the target cutter with the pre-2026-08-10 (unanchored) math rule.

    A deliberate module-attribute swap rather than a permanent `block_only`
    knob threaded through `iter_targets` / `ChunkPolicy`: reproducing a
    superseded cut is a one-off migration's need, and the public target API
    should not carry a parameter whose only caller is this file.
    """
    original = targets.editable_prose

    def legacy(content: str, **kw):
        spans = markdown.protected_spans(content, block_only=False, **kw)
        out = []
        i = 0
        for start, end, _kind in spans:
            out.append(content[i:start])
            i = end
        out.append(content[i:])
        return "".join(out)

    targets.editable_prose = legacy
    try:
        yield
    finally:
        targets.editable_prose = original


def _producible(blog_root: Path, sources: set[str], policy: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for source in sorted(sources):
        path = blog_root / source
        if not path.is_file():
            out[source] = set()
            continue
        out[source] = {t.text for t in targets.iter_targets(files=[path], **policy)}
    return out


def find_stale(pairs_path: Path, blog_root: Path, policy: dict) -> tuple[list[dict], dict]:
    """`(records_to_keep, report)`. Reads only; decides nothing about writing."""
    records = list(iter_pairs(pairs_path))
    sources = {
        (r.get("meta") or {}).get("source")
        for r in records
        if (r.get("meta") or {}).get("synth_key") and (r.get("meta") or {}).get("source")
    }
    sources.discard(None)

    now = _producible(blog_root, sources, policy)
    with _legacy_segmentation():
        before = _producible(blog_root, sources, policy)

    keep: list[dict] = []
    stale: list[dict] = []
    already_gone = 0
    stale_texts: set[tuple[str, str]] = set()
    for rec in records:
        meta = rec.get("meta") or {}
        source = meta.get("source")
        if not meta.get("synth_key") or not source:
            keep.append(rec)
            continue
        body = extract_target(rec)
        if body in now.get(source, set()):
            keep.append(rec)
        elif body in before.get(source, set()):
            stale.append(rec)
            stale_texts.add((source, body))
        else:
            already_gone += 1
            keep.append(rec)

    report = {
        "records": len(records),
        "synth_records": sum(1 for r in records if (r.get("meta") or {}).get("synth_key")),
        "stale_pairs": len(stale),
        "stale_texts": len(stale_texts),
        "stale_sources": sorted({s for s, _ in stale_texts}),
        "unproducible_left_alone": already_gone,
        "samples": sorted(stale_texts)[:5],
    }
    return keep, report


def _policy_from_args(args: argparse.Namespace) -> dict:
    return dict(
        min_chars=args.min_chars,
        max_chars=args.max_chars or None,
        prose_only=True,
        ignore_markers=tuple(args.ignore_marker),
        drop_link_dumps=args.drop_link_dumps,
        drop_list_items=args.drop_list_items,
        merge=args.merge,
        merge_max_chars=args.merge_max_chars,
        stop_at_headers=tuple(args.stop_at_header),
        heading_context=args.heading_context,
        cross_heading=args.cross_heading,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("pairs", type=Path, help="path to pairs.jsonl")
    p.add_argument("--blog-root", type=Path, required=True, help="checkout the targets came from")
    p.add_argument("--apply", action="store_true", help="rewrite pairs.jsonl (default: dry run)")
    p.add_argument("--min-chars", type=int, default=targets.MIN_CHUNK_CHARS)
    p.add_argument("--max-chars", type=int, default=targets.MAX_CHUNK_CHARS, help="0 = uncapped")
    p.add_argument("--merge", action="store_true", default=False)
    p.add_argument("--no-merge", dest="merge", action="store_false")
    p.add_argument("--merge-max-chars", type=int, default=targets.MERGE_MAX_CHUNK_CHARS)
    p.add_argument("--drop-list-items", action="store_true", default=False)
    p.add_argument("--keep-list-items", dest="drop_list_items", action="store_false")
    p.add_argument("--drop-link-dumps", action="store_true", default=True)
    p.add_argument("--keep-link-dumps", dest="drop_link_dumps", action="store_false")
    p.add_argument("--heading-context", choices=("none", "immediate"), default="none")
    p.add_argument("--cross-heading", action="store_true", default=False)
    p.add_argument("--no-cross-heading", dest="cross_heading", action="store_false")
    p.add_argument("--ignore-marker", action="append", default=[])
    p.add_argument("--stop-at-header", action="append", default=[])
    args = p.parse_args(argv)

    if not args.pairs.is_file():
        print(f"no such file: {args.pairs}", file=sys.stderr)
        return 1

    keep, report = find_stale(args.pairs, args.blog_root, _policy_from_args(args))

    print(f"records            : {report['records']} ({report['synth_records']} synthetic)")
    print(f"stale target texts : {report['stale_texts']}")
    print(f"stale pairs        : {report['stale_pairs']}")
    left = report["unproducible_left_alone"]
    print(f"unproducible, kept : {left} (blog drift, not this migration)")
    print(f"sources affected   : {len(report['stale_sources'])}")
    for source in report["stale_sources"]:
        print(f"  - {source}")
    for source, body in report["samples"]:
        head = body.strip().replace("\n", " ")[:110]
        print(f"  e.g. {source}: {head}…")

    if not args.apply:
        print("\ndry run — nothing written. Re-run with --apply to prune.")
        return 0
    if not report["stale_pairs"]:
        print("\nnothing to prune.")
        return 0

    backup = args.pairs.with_suffix(args.pairs.suffix + ".bak")
    shutil.copy2(args.pairs, backup)
    write_jsonl_atomic(args.pairs, keep)
    print(f"\npruned {report['stale_pairs']} pair(s); original saved to {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
