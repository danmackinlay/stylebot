"""Does the segmenter still produce the corpus we already paid to generate?

Every refactor in `stylebot.markdown` risks silently re-cutting the blog into
different chunks. A pair whose target text is no longer producible is *orphaned*:
`synthesize_pairs` resumes on `synth_key` (a function of the target text), so the
next run will not regenerate it, and the corpus quietly loses that cell.

So: re-derive targets for every `meta.source` in the live corpus and count how
many distinct corpus target bodies come back.

**The prose is read at a pinned blog revision, not at HEAD.** Reading HEAD
conflates two unrelated events — segmentation regressing, which is the bug this
file exists to catch, and Dan editing a post, which is expected and constant.
Against HEAD the number therefore sags on a schedule set by Dan's writing (a
single bulk `ai-preen` pass on 2026-08-11 cost 24 cells), the floor has to be
hand-lowered to absorb it, and each lowering buys the guard less teeth. Pinning
removes prose drift from the measurement entirely: the input is a frozen git
blob, so the only things that can move the number are the segmenter and the
corpus file itself. That is what makes the floors below near-equalities rather
than a shrug.

The corpus must never enter this repo (see CLAUDE.md, "Never commit"), so it is
read from the blog checkout in place; nothing here writes to the blog, and the
pinned revision is materialised into a temp dir with `git cat-file` — no
checkout, no stash, no worktree. Treat the blog as strictly read-only: it has
uncommitted work of its own.

Finding that checkout is the whole of ``_blog_root`` below, and its rule is:
**run wherever we can, skip only where we genuinely can't.** ``LIVINGTHING_ROOT``
overrides; unset, the conventional sibling locations are probed; a skip happens
only when no candidate exists on disk at all, and says so out loud. A checkout
that is present but unusable *fails* — a green skip here reads as "measured,
fine".

    LIVINGTHING_ROOT=~/Source/livingthing uv run pytest tests/test_corpus_stability.py -q
    LIVINGTHING_ROOT= uv run pytest tests/test_corpus_stability.py -q  # opt out
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from stylebot.pairs import extract_target, iter_pairs
from stylebot.targets import iter_targets

# The blog revision the corpus was generated against — livingthing
# `1aa290e463 2026-07-22T21:47 "dan-style serve: editor sidecar mirror"`.
#
# Chosen empirically, not guessed. Every synthetic pair is stamped `captured_at`
# 2026-07-21 or 2026-07-22; re-deriving the corpus against each of the 244
# first-parent blog commits since 2026-07-19 draws a curve with an unmistakable
# knee in exactly that window:
#
#     2026-07-20T11:29  3625/3686     2026-07-26T21:13  3558/3686
#     2026-07-20T17:20  3636          2026-07-28T10:30  3543
#     2026-07-21T20:30  3681  <- peak 2026-08-03T17:02  3513
#     2026-07-22T21:47  3681  <- pin  2026-08-10T16:21  3503
#     2026-07-22T21:51  3679          2026-08-11T20:37  3479  (HEAD)
#
# It rises to a flat 3681 plateau spanning the generation window and decays
# monotonically after it — that is prose drift, one edit at a time. The pin is
# the LAST commit on the plateau, so the whole generation window is behind it.
#
# **To re-pin** (needed only when the corpus is re-generated against newer
# prose, never merely because the blog moved on): sweep candidate revisions,
# take the argmax, and re-measure both floors. The sweep is ~15 lines around
# `reproducible()` — feed it `git -C <blog> cat-file blob <rev>:<source>`
# instead of the pin — and runs in well under a minute over 200 commits,
# because most sources are unchanged between adjacent revisions.
PINNED_BLOG_REV = "1aa290e463fd5fc184e4be693ba149f5182659c7"

# Floors measured 2026-08-11 against the 4595-pair corpus (4140 synthetic, 3686
# distinct (source, target-body) cells), re-derived at PINNED_BLOG_REV.
#
# These are equalities, not the ~5%-under fudge the HEAD-relative version
# needed: 3681 of 3686 cells reproduce, i.e. 99.86%. The five that do not are
# unreachable from *any* committed revision of their files (3 in
# `notebook/fish.qmd`, 1 each in `neo_friendly_societies.qmd` and
# `python_debug.qmd`) — synthesis ran against Dan's working tree, which had
# uncommitted edits. That is the irreducible residue, so zero headroom is left
# deliberately: with the prose frozen, `ok` is monotone non-decreasing in the
# corpus (adding pairs can only enlarge the intersection), so nothing benign
# moves it. The two ways it can fall are a segmentation regression — the alarm
# we want — and someone pruning cells that *were* reproducible, which is a
# deliberate corpus edit and deserves this failure plus a re-measure.
#
# Section-bounded is the diet the corpus was GENERATED under: `CROSS_HEADING`
# was promoted after it settled, and re-deriving under cross-heading reproduces
# only 417/3686, which would make this harness a null instrument. So the live
# policy is read from the blog but the chunk *mode* is pinned to the corpus's
# own — that is what a stability floor has to compare against. (`div_is_prose`
# is likewise not passed: the blog's callout-divs-as-prose policy postdates the
# corpus by three weeks, and applying it here costs 85 cells.)
FLOOR_SECTION_BOUNDED = 3681
# The same corpus under today's live cross-heading policy: a much weaker signal
# (the cells barely overlap), but it exercises the cross-heading path over real
# prose, which no fixture does. Peaks at the same pinned revision.
FLOOR_CROSS_HEADING = 417


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

    Read from the *live* corpus, not from the pinned revision: the question is
    whether today's segmenter still cuts today's corpus out of the prose that
    corpus was cut from. Real captures are excluded — they come from paragraph
    diffs of edit archives, not from `iter_targets`, so they were never
    reproducible by construction.
    """
    by_source: dict[str, set[str]] = {}
    for rec in iter_pairs(root / CORPUS):
        meta = rec.get("meta") or {}
        if not meta.get("synth_key") or not meta.get("source"):
            continue
        by_source.setdefault(meta["source"], set()).add(extract_target(rec))
    return by_source


def _materialise(root: Path, sources: list[str], dest: Path) -> None:
    """Write `PINNED_BLOG_REV:<source>` for each source under `dest`.

    `iter_targets(files=…)` wants real paths on disk and the blog checkout is
    strictly read-only, so the pinned blobs are copied out: one `git cat-file
    --batch` for the lot (456 files, well under a second) rather than a process
    per file. A source absent at the pin is simply not written, and
    `reproducible` then skips it exactly as it skips a post deleted since.
    """
    if not (root / ".git").exists():
        pytest.fail(
            f"{root} holds a corpus but is not a git checkout, so the pinned "
            f"blog revision {PINNED_BLOG_REV[:10]} cannot be read and nothing "
            f"was measured. Use a real livingthing clone, or {OPT_OUT}."
        )
    rev = f"{PINNED_BLOG_REV}^{{commit}}"
    if subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "--quiet", rev],
        capture_output=True,
    ).returncode:
        # Never fall back to HEAD: that silently restores the drift-sensitivity
        # the pin exists to remove, and the floors would then be nonsense.
        pytest.fail(
            f"the blog checkout at {root} does not contain the pinned revision "
            f"{PINNED_BLOG_REV} (shallow clone, or history rewritten), so "
            f"nothing was measured. Fetch it (`git -C {root} fetch --unshallow`) "
            f"or re-pin PINNED_BLOG_REV and re-measure both floors, as described "
            f"beside it in {Path(__file__).name}. Do not point it at HEAD."
        )
    proc = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="\n".join(f"{PINNED_BLOG_REV}:{s}" for s in sources).encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    out, pos = proc.stdout, 0
    for source in sources:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].decode().split()
        if len(header) != 3 or header[1] != "blob":  # absent at the pin
            pos = nl + 1
            continue
        size = int(header[2])
        path = dest / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(out[nl + 1 : nl + 1 + size])
        pos = nl + 1 + size + 1  # the blob, then git's trailing newline


@pytest.fixture(scope="session")
def pinned_tree() -> Iterator[Callable[[Path], Path]]:
    """Materialise the pinned blog revision once per session, and clean it up.

    A fixture for the *lifetime*, not for the *work*: nothing in setup can fail,
    so nothing here can downgrade a real problem to a pytest ERROR. Locating the
    blog and reading git both happen inside the returned callable, which each
    test calls from its own body — see the note in the first test.
    """
    tmp = Path(tempfile.mkdtemp(prefix="stylebot-pinned-blog-"))
    done: dict[Path, Path] = {}

    def materialise(root: Path) -> Path:
        if root not in done:
            dest = tmp / str(len(done))
            _materialise(root, sorted(_synth_bodies(root)), dest)
            done[root] = dest
        return done[root]

    try:
        yield materialise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def reproducible(root: Path, tree: Path, policy: dict) -> tuple[int, int]:
    """`(reproducible, total)` corpus target bodies under `policy`.

    Corpus from the live checkout at `root`; prose from `tree`, the pinned
    revision materialised.
    """
    by_source = _synth_bodies(root)
    total = sum(len(v) for v in by_source.values())
    ok = 0
    for source, bodies in by_source.items():
        path = tree / source
        if not path.is_file():  # post did not exist at the pinned revision
            continue
        derived = {t.text for t in iter_targets(files=[path], **policy)}
        ok += len(bodies & derived)
    return ok, total


def test_section_bounded_targets_still_reproduce(pinned_tree):
    # Resolved in the test body, not a fixture: `_blog_root` can fail, and a
    # failure raised in fixture setup is reported as an error, which is a
    # weaker-looking signal than the FAILED this deserves. The git reads in
    # `_materialise` can fail for the same kind of reason, which is why
    # `pinned_tree` hands back a callable rather than a ready-made path.
    blog_root = _blog_root()
    tree = pinned_tree(blog_root)
    policy = {**_live_policy(blog_root), "cross_heading": False}
    ok, total = reproducible(blog_root, tree, policy)
    print(f"\ncorpus stability (section-bounded): {ok}/{total} reproducible")
    assert ok >= FLOOR_SECTION_BOUNDED, (
        f"segmentation regressed: {ok} of {total} corpus targets reproducible "
        f"at blog {PINNED_BLOG_REV[:10]}, floor is {FLOOR_SECTION_BOUNDED}. The "
        f"prose is a frozen git blob, so edits to the blog cannot be the cause. "
        f"Every lost cell is a pair `synthesize_pairs` will never regenerate."
    )


def test_cross_heading_targets_still_reproduce(pinned_tree):
    blog_root = _blog_root()
    tree = pinned_tree(blog_root)
    policy = _live_policy(blog_root)
    assert policy["cross_heading"] is True, "policy changed; re-measure FLOOR_CROSS_HEADING"
    ok, total = reproducible(blog_root, tree, policy)
    print(f"\ncorpus stability (cross-heading): {ok}/{total} reproducible")
    assert ok >= FLOOR_CROSS_HEADING, (
        f"segmentation regressed: {ok} of {total} corpus targets reproducible "
        f"under cross-heading at blog {PINNED_BLOG_REV[:10]}, floor is "
        f"{FLOOR_CROSS_HEADING}."
    )
