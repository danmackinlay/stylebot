"""The hard-case corpus: eleven real posts, and the whole-blog sweep behind them.

Default path: `fixtures/hard/`, eleven blog posts picked by the difficulty
ranking in `fixtures/hard/manifest.json`, pinned span-by-span in `spans.json`.

Opt-in path, the pre-release check:

    STYLEBOT_SWEEP_BLOG_ROOT=~/Source/livingthing uv run pytest tests/test_segment_hard.py

Both paths run `violations()` over the same documents, and
`test_the_two_paths_agree_on_the_fixture_corpus` asserts they agree, so the fast
one is a proxy for the slow one rather than a second opinion (D68). The sweep
has no stored baseline — the corpus is not in this repo and moves daily — so it
checks the invariants; the snapshot is what catches an invariant-preserving
change, and it needs the fixtures.

`STYLEBOT_REGEN_SEGMENTS=1` rewrites `spans.json` when a policy change is
deliberate; it belongs in the same commit as the change, and is never hand-edited.
Which post is the hard case for what: `fixtures/hard/manifest.json`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest
from test_segmenter_conformance import CONFIGURATIONS, covered_chars, spans_for

from stylebot import markdown as md
from stylebot.lib import gather_qmd_files

HARD = Path(__file__).parent / "fixtures" / "hard"
MANIFEST = HARD / "manifest.json"
SPANS = HARD / "spans.json"
SWEEP_ENV = "STYLEBOT_SWEEP_BLOG_ROOT"

#: A fence opener's run of characters, at the start of the span.
OPENER_RE = re.compile(r"^[ \t]*(?P<run>`{3,}|~{3,})")

#: Posts whose invariants break TODAY — live corpus defects, not regressions.
#: A ratchet in `test_budget.py`'s idiom: an entry is deleted when the defect is
#: fixed, and none is added without a `decisions.md` entry. Empty since D69
#: matched the fence closer to its opener (D68 recorded the two it held).
KNOWN_DEFECTS: frozenset[str] = frozenset()

#: Constructs the fixtures were chosen to cover, as a regex that must match
#: somewhere in the corpus. Structural ones (nested divs, mid-line math) are
#: asserted from the spans instead, below.
LANDMARKS: dict[str, str] = {
    "four_backtick_fence": r"(?m)^[ \t]*````",
    "indented_fence": r"(?m)^[ \t]+(?:```|~~~)",
    "indented_div": r"(?m)^[ \t]+:::",
    "raw_html_block": r"(?mi)^[ \t]*<script\b",
    "html_comment": r"<!--",
    "math_with_attrs": r"\$\$[ \t]*\{",
    "callout_div": r"(?m)^[ \t]*:::.*\.callout-",
    "fig_div": r"(?m)^[ \t]*:::.*#fig-",
    "refs_div": r"(?m)^[ \t]*:::.*#refs",
    "layout_div": r"(?m)^[ \t]*:::.*#tbl-",
    "heading_line": r"(?m)^#{1,6} \S",
}


# --- the invariants both paths run -----------------------------------------


def _span_shape(doc: str, config: str) -> list[str]:
    """Sorted, disjoint, in-bounds, correctly kinded, and losslessly rebuildable."""
    out: list[str] = []
    spans = spans_for(doc, config)
    allowed = md.protected_kinds(
        raw_html=CONFIGURATIONS[config]["raw_html"],
        html_comments=CONFIGURATIONS[config]["html_comments"],
    )
    prev = 0
    rebuilt: list[str] = []
    for start, end, kind in spans:
        if not 0 <= start < end <= len(doc):
            out.append(f"{config}: span ({start}, {end}) out of bounds")
        if start < prev:
            out.append(f"{config}: span at {start} overlaps or unsorts the one before")
        if kind not in allowed:
            out.append(f"{config}: kind {kind!r} is not in {sorted(allowed)}")
        rebuilt.append(doc[prev:start])
        rebuilt.append(doc[start:end])
        prev = end
    rebuilt.append(doc[prev:])
    if "".join(rebuilt) != doc:
        out.append(f"{config}: spans are not lossless")
    return out


def _fence_stops_at_its_closer(doc: str, config: str) -> list[str]:
    """W2's bug: a fence span ran 999 chars past its own closer, eating three divs.

    CommonMark's closer, spelled here independently of the module's: the same
    character as the opener, at least as long, alone on its line. Spelling it as
    ``{3,}`` is what let a long outer fence be closed by the short one it quotes
    (D69), so this rule is the invariant, not an implementation detail.
    """
    out: list[str] = []
    for start, end, kind in spans_for(doc, config):
        if kind != md.FENCE:
            continue
        lines = doc[start:end].split("\n")
        run = OPENER_RE.match(lines[0]).group("run")  # type: ignore[union-attr]
        closer_re = re.compile(rf"^[ \t]*{run[0]}{{{len(run)},}}[ \t]*$")
        closer = next((i for i, ln in enumerate(lines) if i and closer_re.match(ln)), None)
        if closer is None:
            if end != len(doc):
                out.append(f"{config}: unclosed fence at {start} stops before EOF")
        elif lines[closer + 1 :] not in ([], [""]):
            tail = "\n".join(lines[closer + 1 :])
            out.append(f"{config}: fence at {start} runs {len(tail)} chars past its closer")
    return out


def _headings_survive(doc: str, config: str) -> list[str]:
    """A runaway `$$` or `>` span that swallows a heading has lost the document."""
    out: list[str] = []
    for start, end, kind in spans_for(doc, config):
        if kind not in (md.MATH, md.BLOCKQUOTE):
            continue
        for line in doc[start:end].split("\n"):
            if md.is_heading_line(line) and line.strip():
                out.append(f"{config}: {kind} at {start} swallowed heading {line.strip()!r}")
    return out


def _divs_open_on_a_div_line(doc: str, config: str) -> list[str]:
    out: list[str] = []
    for start, _end, kind in spans_for(doc, config):
        if kind == md.DIV and not doc[start:].lstrip(" \t").startswith(":::"):
            out.append(f"{config}: div span at {start} does not start on a ':::' line")
    return out


def violations(doc: str) -> list[str]:
    """Every invariant break in `doc`, one message each. Empty means clean."""
    out: list[str] = []
    for config in CONFIGURATIONS:
        out += _span_shape(doc, config)
        out += _fence_stops_at_its_closer(doc, config)
        out += _headings_survive(doc, config)
        out += _divs_open_on_a_div_line(doc, config)
    leaked = covered_chars(doc, "stylebot") - covered_chars(doc, "livingthing")
    if leaked:
        out.append(f"stylebot protects {len(leaked)} chars livingthing does not")
    return out


def sweep(root: Path) -> dict[str, list[str]]:
    """`violations()` for every `.qmd` under `root`, keyed by relative path."""
    return {
        str(p.relative_to(root)): violations(p.read_text(encoding="utf-8"))
        for p in sorted(gather_qmd_files([], base=root))
    }


# --- the fixtures ----------------------------------------------------------


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(HARD.glob("*.qmd"))}


def build_record(corpus: dict[str, str]) -> dict:
    """The whole expectation, derived from the fixtures — `spans.json`'s schema."""
    return {
        "configurations": CONFIGURATIONS,
        "posts": {
            name: {
                "sha256": hashlib.sha256(doc.encode("utf-8")).hexdigest(),
                "spans": {
                    config: [
                        {"start": s, "end": e, "kind": k} for s, e, k in spans_for(doc, config)
                    ]
                    for config in CONFIGURATIONS
                },
            }
            for name, doc in corpus.items()
        },
    }


# --- the fast path ---------------------------------------------------------


def test_each_hard_post_matches_its_recorded_spans(corpus: dict[str, str]):
    record = build_record(corpus)
    if os.environ.get("STYLEBOT_REGEN_SEGMENTS"):
        SPANS.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    expected = json.loads(SPANS.read_text(encoding="utf-8"))
    assert record["configurations"] == expected["configurations"]
    assert sorted(record["posts"]) == sorted(expected["posts"])
    for name in record["posts"]:
        assert record["posts"][name]["sha256"] == expected["posts"][name]["sha256"], (
            f"{name} changed on disk; regenerate spans.json in the same commit"
        )
        assert record["posts"][name]["spans"] == expected["posts"][name]["spans"], name


def test_the_hard_corpus_breaks_exactly_where_the_corpus_does(corpus: dict[str, str]):
    found = {name: v for name, v in ((n, violations(d)) for n, d in corpus.items()) if v}
    assert set(found) == KNOWN_DEFECTS & set(corpus), found


def test_the_two_paths_agree_on_the_fixture_corpus(corpus: dict[str, str]):
    """The sweep, pointed at the fixtures, is the fast path — not a second opinion."""
    assert sweep(HARD) == {name: violations(doc) for name, doc in corpus.items()}


# --- the corpus is only a proxy if it still contains the constructs --------


def test_the_manifest_matches_the_fixtures_on_disk(corpus: dict[str, str], manifest: dict):
    listed = {p["fixture"]: p for p in manifest["posts"]}
    assert sorted(listed) == sorted(corpus)
    for name, doc in corpus.items():
        assert hashlib.sha256(doc.encode("utf-8")).hexdigest() == listed[name]["sha256"], name


def test_every_landmark_construct_survives_in_the_corpus(corpus: dict[str, str]):
    joined = "\n".join(corpus.values())
    missing = [name for name, pattern in LANDMARKS.items() if not re.search(pattern, joined)]
    assert not missing, f"the fixtures no longer contain: {missing}"


def test_the_corpus_keeps_its_structural_edge_cases(corpus: dict[str, str]):
    """Two cases no regex sees: a nested `:::`, and mid-line `$$`.

    An unclosed `:::` opener was the third until the blog fixed the five
    asymmetric fences that produced it: zero corpus-wide, so it lives in
    `segmenter_conformance.qmd` on D68's rule for a construct no post has.
    """
    nested = midline_math = False
    for doc in corpus.values():
        depth = 0
        for line in doc.split("\n"):
            stripped = line.strip()
            if not stripped.startswith(":::"):
                continue
            depth += -1 if stripped == ":::" else 1
            nested = nested or depth > 1
        block = {(s, e) for s, e, k in spans_for(doc, "stylebot") if k == md.MATH}
        inline = {(s, e) for s, e, k in spans_for(doc, "livingthing") if k == md.MATH}
        midline_math = midline_math or bool(inline - block)
    assert nested, "no fixture nests ':::' divs"
    assert midline_math, "no fixture has mid-line `$$`, so the two policies cannot differ"


def test_every_protected_kind_appears_somewhere(corpus: dict[str, str]):
    for config in CONFIGURATIONS:
        seen = {k for doc in corpus.values() for _s, _e, k in spans_for(doc, config)}
        want = md.protected_kinds(
            raw_html=CONFIGURATIONS[config]["raw_html"],
            html_comments=CONFIGURATIONS[config]["html_comments"],
        )
        assert seen == want, f"{config} corpus misses {want - seen}"


# --- the opt-in sweep ------------------------------------------------------


@pytest.mark.skipif(not os.environ.get(SWEEP_ENV), reason=f"set {SWEEP_ENV} to sweep a blog")
def test_the_whole_blog_corpus_breaks_only_where_it_is_known_to():
    root = Path(os.environ[SWEEP_ENV]).expanduser()
    assert root.is_dir(), f"{SWEEP_ENV}={root} is not a directory"
    found = {k: v for k, v in sweep(root).items() if v}
    assert {Path(k).name for k in found} == KNOWN_DEFECTS, found
