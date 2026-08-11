"""Does the segmenter still produce the corpus we already paid to generate?

Every refactor in `stylebot.markdown` risks silently re-cutting the blog into
different chunks. A pair whose target text is no longer producible is *orphaned*:
`synthesize_pairs` resumes on `synth_key` (a function of the target text), so the
next run will not regenerate it, and the corpus quietly loses that cell.

So: re-derive targets for every `meta.source` in the live corpus and count how
many distinct corpus target bodies come back. The number is a **floor**, asserted
not to fall. It is deliberately not an equality — the blog is edited daily, and
prose drift since generation accounts for the standing ~5% gap.

The corpus must never enter this repo (see CLAUDE.md, "Never commit"), so it is
read from the blog checkout in place; nothing here writes. Finding that checkout
is the whole of ``_blog_root`` below, and its rule is: **run wherever we can,
skip only where we genuinely can't.** ``LIVINGTHING_ROOT`` overrides; unset, the
conventional sibling locations are probed; a skip happens only when no candidate
exists on disk at all, and says so out loud. A checkout that is present but
unusable *fails* — a green skip here reads as "measured, fine".

    LIVINGTHING_ROOT=~/Source/livingthing uv run pytest tests/test_corpus_stability.py -q
    LIVINGTHING_ROOT= uv run pytest tests/test_corpus_stability.py -q  # opt out
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
# into truncated sentences. A one-off prune script removed them (deleted in 0.3,
# once it had run — see the `scripts/` carve-out in CLAUDE.md).
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


CORPUS = Path("_training_pairs") / "pairs.jsonl"

# How to say "I know, I don't have it, stop telling me" — because *unsetting*
# LIVINGTHING_ROOT is no longer that statement: unset now means "go and look".
OPT_OUT = "set LIVINGTHING_ROOT= (to the empty string) to opt out deliberately"


def _candidate_roots() -> list[Path]:
    """Where the blog checkout conventionally sits, when nobody said.

    Consulted only when `LIVINGTHING_ROOT` is unset. Two probes: a sibling of
    this checkout, which is the layout CLAUDE.md documents (`…/stylebot` next to
    `…/livingthing`); and `~/Source/livingthing`, which still finds the blog when
    stylebot is being run from a git worktree nested somewhere else entirely.
    """
    return list(
        dict.fromkeys(
            [
                Path(__file__).resolve().parents[2] / "livingthing",
                Path.home() / "Source" / "livingthing",
            ]
        )
    )


def _blog_root() -> Path:
    """The blog checkout, or a skip/fail that explains itself.

    Skipping is reserved for the one case that deserves it — a machine with no
    blog checkout anywhere, which is every contributor's machine, since the
    corpus is private. Anywhere a checkout *is* named or found, an unusable one
    is a failure: this file's whole job is to notice segmentation moving the
    corpus, and it cannot do that from a green skip.
    """
    raw = os.environ.get("LIVINGTHING_ROOT")
    if raw is not None and not raw.strip():
        pytest.skip("LIVINGTHING_ROOT is empty: corpus-stability checks opted out deliberately")
    if raw:
        root = Path(raw).expanduser()
        if not (root / CORPUS).is_file():
            # FAIL, not skip: naming a root is an explicit request to measure.
            pytest.fail(
                f"LIVINGTHING_ROOT={raw} names no blog checkout — {root / CORPUS} "
                f"is not a file — so nothing was measured. Point it at the "
                f"livingthing repo, or {OPT_OUT}."
            )
        return root

    found = [c for c in _candidate_roots() if (c / CORPUS).is_file()]
    if found:
        return found[0]
    present = [c for c in _candidate_roots() if c.is_dir()]
    if present:
        # A checkout is there but its corpus isn't: in the livingthing repo
        # pairs.jsonl is committed, so this is a broken checkout, not an absent
        # one. Same reasoning as above — do not let it pass for "measured".
        pytest.fail(
            f"found {present[0]} but no {CORPUS} inside it, so nothing was "
            f"measured. Restore the corpus, point LIVINGTHING_ROOT elsewhere, "
            f"or {OPT_OUT}."
        )
    looked = ", ".join(str(c) for c in _candidate_roots())
    pytest.skip(
        f"no blog checkout found (looked for {CORPUS} under {looked}). The "
        f"corpus is private and lives in the livingthing repo, so this guard "
        f"cannot run here; set LIVINGTHING_ROOT=/path/to/livingthing if you do "
        f"have a checkout."
    )


def _live_policy(root: Path) -> dict:
    """The blog's chunk policy, read from its source rather than duplicated here."""
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import livingthing.training_targets as tt
    except Exception as exc:  # pragma: no cover - depends on the sibling checkout
        # FAIL, do not skip. Having found the blog is an implicit request to
        # measure the corpus; skipping turns that into two green skips that read
        # as "measured, fine". That is exactly what happened on 2026-08-10: the
        # (correct) removal of stylebot's unused python-dateutil dependency also
        # removed it from the env, `livingthing.training_targets` imports it
        # transitively, and this harness quietly stopped running for hours.
        pytest.fail(
            f"the blog checkout at {root} was found but its "
            f"livingthing.training_targets will not import, so nothing was "
            f"measured: {exc}. Install the sibling's import deps "
            f"(`uv run --with python-dateutil pytest …`) or {OPT_OUT}."
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
    for rec in iter_pairs(root / CORPUS):
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


def test_section_bounded_targets_still_reproduce():
    # Resolved in the test body, not a fixture: `_blog_root` can fail, and a
    # failure raised in fixture setup is reported as an error, which is a
    # weaker-looking signal than the FAILED this deserves.
    blog_root = _blog_root()
    policy = {**_live_policy(blog_root), "cross_heading": False}
    ok, total = reproducible(blog_root, policy)
    print(f"\ncorpus stability (section-bounded): {ok}/{total} reproducible")
    assert ok >= FLOOR_SECTION_BOUNDED, (
        f"segmentation regressed: {ok} of {total} corpus targets reproducible, "
        f"floor is {FLOOR_SECTION_BOUNDED}. Every lost cell is a pair "
        f"`synthesize_pairs` will never regenerate."
    )


def test_cross_heading_targets_still_reproduce():
    blog_root = _blog_root()
    policy = _live_policy(blog_root)
    assert policy["cross_heading"] is True, "policy changed; re-measure FLOOR_CROSS_HEADING"
    ok, total = reproducible(blog_root, policy)
    print(f"\ncorpus stability (cross-heading): {ok}/{total} reproducible")
    assert ok >= FLOOR_CROSS_HEADING
