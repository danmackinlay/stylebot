"""The hard-case corpus: eleven real posts, one synthetic document, the sweep.

Default path: `fixtures/hard/`, eleven blog-post excerpts picked by the
difficulty ranking in `fixtures/hard/manifest.json` plus the synthetic
`segmenter_conformance.qmd` (the constructs no post has, D69), pinned
span-by-span in `spans.json`. Opt-in path, the pre-release check:

    STYLEBOT_SWEEP_BLOG_ROOT=~/Source/livingthing uv run pytest tests/test_segment_hard.py

Both run `violations()`; the sweep has no stored baseline — the corpus is not
in this repo and moves daily — so it checks the invariants, and the snapshot is
what catches an invariant-preserving change (D68).

Two named configurations, because the two callers legitimately differ:

    stylebot     block_only=True,  raw_html=False, html_comments=False
    livingthing  block_only=False, raw_html=True,  html_comments=True

stylebot cuts *training targets* and deletes what it protects, so a mid-line
``$$k$$`` stays in the prose (defect A5); livingthing reflows whole documents and
reassembles, so inline math, raw ``<script>`` and comments are protected. Both
come from `markdown.protected_spans` alone — D29's "two policies, one
implementation" — and `segmenter_conformance.json` records which construct each
protects and where they diverge, so a change to either arrives as a reviewable
diff. Do not regenerate the JSON to make a red test green: `STYLEBOT_REGEN_SEGMENTS=1`
rewrites `spans.json` and the conformance JSON when a policy change is
deliberate, in the same commit as the change; neither is ever hand-edited.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import conftest
import pytest

from stylebot import markdown as md
from stylebot.lib import gather_qmd_files

HARD = Path(__file__).parent / "fixtures" / "hard"
SPANS = HARD / "spans.json"
SWEEP_ENV = conftest.SWEEP_ENV
CONFORMANCE = HARD / "segmenter_conformance.qmd"
CONFORMANCE_EXPECTED = HARD.parent / "segmenter_conformance.json"

#: The named configurations, as keyword arguments to `md.protected_spans`.
#: ``div_is_prose: null`` records that neither side frees div contents — that is
#: the caller's policy hook, out of scope for a segmenter conformance spec.
CONFIGURATIONS: dict[str, dict] = {
    "stylebot": {
        "block_only": True,
        "raw_html": False,
        "html_comments": False,
        "div_is_prose": None,
    },
    "livingthing": {
        "block_only": False,
        "raw_html": True,
        "html_comments": True,
        "div_is_prose": None,
    },
}

#: ``(case name, needle)``. Each needle must occur exactly once in the synthetic
#: document; the JSON records which configurations protect it. Cases whose
#: needle is protected NOWHERE are still cases — an indented or glued heading is
#: here to pin that no span swallows it.
CASES: list[tuple[str, str]] = [
    ("backtick_fence", 'return "a backtick fence"'),
    ("tilde_fence", "key: a tilde fence"),
    ("indented_heading", "  ## An indented heading"),
    ("glued_heading", "## A heading glued to prose"),
    ("display_math_with_attrs", r"\int_0^1"),
    ("midline_math", "$$d$$"),  # the whole construct: `high-$$d$$ behaviour`
    ("blockquote", "somebody else's voice"),
    ("blockquote_glued_to_prose", "no blank line after it"),
    ("refs_div", "{#refs}"),
    ("figure_div", "fig-conformance"),
    ("callout_div", "callout-note"),
    ("nested_div", "fig-nested"),
    ("raw_html_script", "const conformance = 1;"),
    ("html_comment_single_line", "never prose, in any configuration"),
    ("html_comment_multi_line", "Its second sentence sits on its own line."),
    ("html_comment_midline", "A trailing comment, opened mid-line."),
    ("prose_between_comments", "which must stay prose"),
    ("unclosed_div", "an-opener-that-is-never-closed"),
    ("unclosed_fence", "an unclosed fence runs to end of document"),
]

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


# --- the two configurations ------------------------------------------------


def spans_for(doc: str, config: str) -> list[md.Span]:
    """The protected spans of `doc` under the named configuration."""
    return md.protected_spans(doc, **CONFIGURATIONS[config])


def covered_chars(doc: str, config: str) -> set[int]:
    """Every character index the named configuration protects.

    Coverage rather than span identity, because merging makes the two
    incomparable as *tuples* while leaving the real invariant intact: a `>` line
    inside a multi-line comment is a blockquote span under one configuration and
    part of one larger comment span under the other. Both protect the same
    characters, which is the only thing a consumer can rely on.
    """
    return {i for s, e, _k in spans_for(doc, config) for i in range(s, e)}


def _covered(spans: list[md.Span], start: int, end: int) -> bool:
    """True if some single span wholly contains ``[start, end)`` (spans merge)."""
    return any(s <= start and end <= e for s, e, _kind in spans)


def conformance_record(doc: str) -> dict:
    """The whole expectation, derived from the document — the JSON's schema."""
    by_config = {name: spans_for(doc, name) for name in CONFIGURATIONS}
    cases = []
    for name, needle in CASES:
        start = doc.index(needle)
        end = start + len(needle)
        cases.append(
            {
                "name": name,
                "needle": needle,
                "start": start,
                "end": end,
                "protected_in": [c for c, sp in by_config.items() if _covered(sp, start, end)],
            }
        )
    return {
        "fixture": CONFORMANCE.name,
        "fixture_sha256": hashlib.sha256(doc.encode("utf-8")).hexdigest(),
        "configurations": CONFIGURATIONS,
        "cases": cases,
        "spans": {
            name: [{"start": s, "end": e, "kind": k, "text": doc[s:e]} for s, e, k in spans]
            for name, spans in by_config.items()
        },
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


# --- the corpus is only a proxy if it still contains the constructs --------


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


# --- the policy rows: what the two configurations mean, written down -------


@pytest.fixture(scope="module")
def conformance() -> str:
    return CONFORMANCE.read_text(encoding="utf-8")


def test_stylebot_matches_the_conformance_fixture(conformance: str):
    record = conformance_record(conformance)
    if os.environ.get("STYLEBOT_REGEN_SEGMENTS"):
        CONFORMANCE_EXPECTED.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    expected = json.loads(CONFORMANCE_EXPECTED.read_text(encoding="utf-8"))
    assert record["fixture_sha256"] == expected["fixture_sha256"], (
        "the fixture document changed; regenerate the JSON in the same commit"
    )
    assert record["configurations"] == expected["configurations"]
    assert record["cases"] == expected["cases"]
    for name in CONFIGURATIONS:
        assert record["spans"][name] == expected["spans"][name], f"drift in the {name} config"


def test_the_configurations_disagree_only_where_intended(conformance: str):
    """Three declared policy differences, and nothing else.

    Each is a case of the same asymmetry: livingthing reassembles protected spans
    and stylebot deletes them, so livingthing can afford to protect a construct
    that sits *inside* a sentence and stylebot cannot.
    """
    doc = conformance
    lt = {(s, e) for s, e, _k in spans_for(doc, "livingthing")}
    sb = {(s, e) for s, e, _k in spans_for(doc, "stylebot")}
    extra = sorted(doc[s:e] for s, e in lt - sb)
    assert any("$$d$$" in x for x in extra), extra  # inline display math
    assert any(x.lstrip().startswith("<script") for x in extra), extra
    comments = [x for x in extra if x.lstrip().startswith("<!--")]
    assert len(comments) == 3, comments  # single-line, multi-line, mid-line
    assert len(extra) == 2 + len(comments), extra
    leaked = covered_chars(doc, "stylebot") - covered_chars(doc, "livingthing")
    assert not leaked, (
        "the stylebot config must protect a subset of livingthing's; "
        f"unprotected for livingthing: {doc[min(leaked) : max(leaked) + 1]!r}"
    )


def test_html_comments_are_protected_for_livingthing_only(conformance: str):
    cases = {c["name"]: c["protected_in"] for c in conformance_record(conformance)["cases"]}
    for name in (
        "html_comment_single_line",
        "html_comment_multi_line",
        "html_comment_midline",
    ):
        assert cases[name] == ["livingthing"], name
    # The corollary that makes the mid-line case worth having: protecting a
    # comment must not drag the prose on either side of it out of reach.
    assert cases["prose_between_comments"] == []


def test_every_named_case_occurs_exactly_once(conformance: str):
    for name, needle in CASES:
        assert conformance.count(needle) == 1, f"{name}: needle {needle!r} is not unique"


# --- the opt-in sweep ------------------------------------------------------


@pytest.mark.sweep  # conftest deselects it unless SWEEP_ENV is set
def test_the_whole_blog_corpus_breaks_only_where_it_is_known_to():
    root = Path(os.environ[SWEEP_ENV]).expanduser()
    assert root.is_dir(), f"{SWEEP_ENV}={root} is not a directory"
    found = {k: v for k, v in sweep(root).items() if v}
    assert {Path(k).name for k in found} == KNOWN_DEFECTS, found
