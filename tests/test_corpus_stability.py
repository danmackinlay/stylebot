"""Does the segmenter still produce the corpus we already paid to generate?

Every refactor in `stylebot.markdown` risks silently re-cutting the blog into
different chunks. A pair whose target text is no longer producible is *orphaned*:
`synthesize_pairs` resumes on `synth_key` (a function of the target text), so the
next run will not regenerate it, and the corpus quietly loses that cell.

So: re-derive targets for every `meta.source` in the live corpus and count how
many distinct corpus target bodies come back. The number is a **floor**, asserted
not to fall. It is deliberately not an equality — the blog is edited daily, and
prose drift since generation accounts for the standing ~5% gap.

Skipped unless ``LIVINGTHING_ROOT`` names the blog checkout: the corpus must
never enter this repo (see CLAUDE.md, "Never commit"). Nothing here writes.

    LIVINGTHING_ROOT=~/Source/livingthing uv run pytest tests/test_corpus_stability.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from stylebot.pairs import extract_target, iter_pairs
from stylebot.targets import iter_targets

# Floors measured 2026-08-10 against the then-4624-pair corpus, 3710 distinct
# (source, target-body) cells from the 4169 synthetic pairs. Both floors were
# lowered by the block-anchored-math fix, which orphans 24 target texts (29
# pairs) BY DESIGN: those are the ones a non-line-anchored `$$` had mangled
# into truncated sentences. `scripts/prune_stale_synth_pairs.py` removes them.
# Before that fix the floors were 3527 / 401.
#
# The prune has since run, so the live corpus is 4595 pairs / 4140 synthetic /
# 3686 distinct cells. The floors below are unchanged by that — they count
# cells still reproducible, and pruning removed only cells that no longer were.
#
# Section-bounded is the diet the corpus was GENERATED under: `CROSS_HEADING`
# was promoted after it settled, and re-deriving under cross-heading reproduces
# only 401/3710, which would make this harness a null instrument. So the live
# policy is read from the blog but the chunk *mode* is pinned to the corpus's
# own — that is what a stability floor has to compare against.
FLOOR_SECTION_BOUNDED = 3503
# The same corpus under today's live cross-heading policy: a much weaker signal
# (the cells barely overlap), but it exercises the cross-heading path over real
# prose, which no fixture does.
FLOOR_CROSS_HEADING = 396


def _blog_root() -> Path:
    raw = os.environ.get("LIVINGTHING_ROOT")
    if not raw:
        pytest.skip("set LIVINGTHING_ROOT to the blog checkout to run corpus-stability checks")
    root = Path(raw).expanduser()
    if not (root / "_training_pairs" / "pairs.jsonl").is_file():
        pytest.skip(f"no _training_pairs/pairs.jsonl under {root}")
    return root


def _live_policy(root: Path) -> dict:
    """The blog's chunk policy, read from its source rather than duplicated here."""
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import livingthing.training_targets as tt
    except Exception as exc:  # pragma: no cover - depends on the sibling checkout
        # FAIL, do not skip. Setting LIVINGTHING_ROOT is an explicit request to
        # measure the corpus; skipping turns that into two green skips that read
        # as "measured, fine". That is exactly what happened on 2026-08-10: the
        # (correct) removal of stylebot's unused python-dateutil dependency also
        # removed it from the env, `livingthing.training_targets` imports it
        # transitively, and this harness quietly stopped running for hours.
        pytest.fail(
            f"LIVINGTHING_ROOT is set but livingthing.training_targets will not "
            f"import, so nothing was measured: {exc}. Install the sibling's "
            f"import deps (`uv run --with python-dateutil pytest …`) or unset "
            f"LIVINGTHING_ROOT to opt out deliberately."
        )
    return dict(
        min_chars=tt.MIN_CHARS,
        merge=tt.MERGE,
        merge_max_chars=tt.MERGE_MAX_CHARS,
        drop_list_items=tt.DROP_LIST_ITEMS,
        heading_context=tt.HEADING_CONTEXT,
        cross_heading=tt.CROSS_HEADING,
        ignore_markers=[tt.STUB_MARKER],
        stop_at_headers=tt.STOP_HEADERS,
    )


def _synth_bodies(root: Path) -> dict[str, set[str]]:
    """`{source: {target body, ...}}` over the synthetic pairs.

    Real captures are excluded: they come from paragraph diffs of edit archives,
    not from `iter_targets`, so they were never reproducible by construction.
    """
    by_source: dict[str, set[str]] = {}
    for rec in iter_pairs(root / "_training_pairs" / "pairs.jsonl"):
        meta = rec.get("meta") or {}
        if not meta.get("synth_key") or not meta.get("source"):
            continue
        by_source.setdefault(meta["source"], set()).add(extract_target(rec))
    return by_source


def reproducible(root: Path, policy: dict) -> tuple[int, int]:
    """`(reproducible, total)` corpus target bodies under `policy`."""
    by_source = _synth_bodies(root)
    total = sum(len(v) for v in by_source.values())
    ok = 0
    for source, bodies in by_source.items():
        path = root / source
        if not path.is_file():  # post deleted or renamed since generation
            continue
        derived = {t.text for t in iter_targets(files=[path], **policy)}
        ok += len(bodies & derived)
    return ok, total


@pytest.fixture(scope="module")
def blog_root() -> Path:
    return _blog_root()


def test_section_bounded_targets_still_reproduce(blog_root: Path):
    policy = {**_live_policy(blog_root), "cross_heading": False}
    ok, total = reproducible(blog_root, policy)
    print(f"\ncorpus stability (section-bounded): {ok}/{total} reproducible")
    assert ok >= FLOOR_SECTION_BOUNDED, (
        f"segmentation regressed: {ok} of {total} corpus targets reproducible, "
        f"floor is {FLOOR_SECTION_BOUNDED}. Every lost cell is a pair "
        f"`synthesize_pairs` will never regenerate."
    )


def test_cross_heading_targets_still_reproduce(blog_root: Path):
    policy = _live_policy(blog_root)
    assert policy["cross_heading"] is True, "policy changed; re-measure FLOOR_CROSS_HEADING"
    ok, total = reproducible(blog_root, policy)
    print(f"\ncorpus stability (cross-heading): {ok}/{total} reproducible")
    assert ok >= FLOOR_CROSS_HEADING
