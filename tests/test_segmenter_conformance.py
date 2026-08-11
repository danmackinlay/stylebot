"""The cross-implementation segmenter spec — stylebot's side of it.

`livingthing.qmd_core.segment_for_edit` is a **permanent** second implementation
of `stylebot.markdown`: livingthing is making stylebot an optional dependency,
so its blog-core editing path has to segment markdown with stylebot uninstalled.
The duplication cannot be deleted, so it is pinned instead. This is the same
idiom as `tests/test_marker_segmentation.py`, which pins the TypeScript port in
the VS Code marker: one shared fixture, one checked-in expectation, asserted
from both languages.

**The artefacts.** `fixtures/segmenter_conformance.qmd` is the document;
`fixtures/segmenter_conformance.json` is the expectation. Both are copied into
the consuming repo, which asserts its own segmenter against the same JSON —
hence `fixture_sha256`, so a stale copy of the document fails loudly instead of
disagreeing subtly.

**Two named configurations, because the two sides legitimately differ.**

    stylebot     block_only=True,  raw_html=False
    livingthing  block_only=False, raw_html=True

`block_only` keeps a mid-line ``$$k$$`` in the prose the styler rewrites: this
package cuts *training targets*, and excising mid-sentence math joins the
remnants into a truncated sentence (defect A5). livingthing reflows whole
documents, where inline ``$$`` is a real construct to leave alone and raw
``<script>``/``<style>`` must survive verbatim. Both configurations are produced
here by `stylebot.markdown.protected_spans` alone — that is the point of the
`raw_html` parameter, and it is what makes this file a spec rather than a
transcript of one implementation's habits.

**What a mismatch means.** A failure here is *drift*: the two implementations no
longer agree about what a rewrite must never be handed. The consequence is
silent — an editor quietly rewording somebody else's quoted words, or reflowing
a `<script>` into a syntax error, looks like a successful edit. Do not
regenerate the JSON to make a red test green.

**When the change is deliberate**, `STYLEBOT_REGEN_SEGMENTS=1 uv run pytest
tests/test_segmenter_conformance.py` rewrites the JSON — and the regenerated
fixture belongs in the *same commit* as the policy change, so review sees the
segmentation move next to the reason for it. Then re-run the other side.

Consumers with no per-kind labels (livingthing's segmenter returns only
``(text, editable)``) assert on ``start``/``end`` and ignore ``kind``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from stylebot import markdown as md

FIXTURE = Path(__file__).parent / "fixtures" / "segmenter_conformance.qmd"
EXPECTED = Path(__file__).parent / "fixtures" / "segmenter_conformance.json"

#: The named configurations, as keyword arguments to `md.protected_spans`.
#: ``div_is_prose: null`` records that neither side frees div contents — that is
#: the caller's policy hook, out of scope for a segmenter conformance spec.
CONFIGURATIONS: dict[str, dict] = {
    "stylebot": {"block_only": True, "raw_html": False, "div_is_prose": None},
    "livingthing": {"block_only": False, "raw_html": True, "div_is_prose": None},
}

#: ``(case name, needle)``. Each needle must occur exactly once in the fixture;
#: the JSON records which configurations protect it. Cases whose needle is
#: protected NOWHERE are still cases — an indented or glued heading is here to
#: pin that no span swallows it.
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
    ("unclosed_div", "an-opener-that-is-never-closed"),
]


def spans_for(doc: str, config: str) -> list[md.Span]:
    """The protected spans of `doc` under the named configuration."""
    return md.protected_spans(doc, **CONFIGURATIONS[config])


def _covered(spans: list[md.Span], start: int, end: int) -> bool:
    """True if some single span wholly contains ``[start, end)`` (spans merge)."""
    return any(s <= start and end <= e for s, e, _kind in spans)


def build_record(doc: str) -> dict:
    """The whole expectation, derived from the fixture — the JSON's schema."""
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
        "fixture": FIXTURE.name,
        "fixture_sha256": hashlib.sha256(doc.encode("utf-8")).hexdigest(),
        "configurations": CONFIGURATIONS,
        "cases": cases,
        "spans": {
            name: [{"start": s, "end": e, "kind": k, "text": doc[s:e]} for s, e, k in spans]
            for name, spans in by_config.items()
        },
    }


@pytest.fixture(scope="module")
def doc() -> str:
    return FIXTURE.read_text(encoding="utf-8")


# --- the spec --------------------------------------------------------------


def test_stylebot_matches_the_conformance_fixture(doc: str):
    record = build_record(doc)

    if os.environ.get("STYLEBOT_REGEN_SEGMENTS"):
        EXPECTED.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    assert record["fixture_sha256"] == expected["fixture_sha256"], (
        "the fixture document changed; regenerate the JSON in the same commit "
        "and re-run the other implementation's conformance test"
    )
    assert record["configurations"] == expected["configurations"]
    assert record["cases"] == expected["cases"]
    for name in CONFIGURATIONS:
        assert record["spans"][name] == expected["spans"][name], f"drift in the {name} config"


# --- the two configurations really are two ---------------------------------


def test_the_configurations_disagree_only_where_intended(doc: str):
    lt = {(s, e) for s, e, _k in spans_for(doc, "livingthing")}
    sb = {(s, e) for s, e, _k in spans_for(doc, "stylebot")}
    extra = sorted(doc[s:e] for s, e in lt - sb)
    assert len(extra) == 2, extra
    assert any("$$d$$" in x for x in extra), extra  # inline display math
    assert any(x.lstrip().startswith("<script") for x in extra), extra
    assert not sb - lt, "the stylebot config must protect a subset of livingthing's"


def test_midline_math_is_prose_for_stylebot_only(doc: str):
    case = next(c for c in build_record(doc)["cases"] if c["name"] == "midline_math")
    assert case["protected_in"] == ["livingthing"]


def test_the_script_block_is_protected_for_livingthing_only(doc: str):
    case = next(c for c in build_record(doc)["cases"] if c["name"] == "raw_html_script")
    assert case["protected_in"] == ["livingthing"]


# --- invariants a consumer is entitled to rely on --------------------------


def test_spans_are_sorted_disjoint_and_in_bounds(doc: str):
    for name in CONFIGURATIONS:
        prev_end = 0
        for start, end, kind in spans_for(doc, name):
            assert 0 <= start < end <= len(doc), name
            assert start >= prev_end, f"{name} spans overlap or are unsorted"
            assert kind in md.protected_kinds(raw_html=CONFIGURATIONS[name]["raw_html"])
            prev_end = end


def test_both_configurations_stay_lossless(doc: str):
    for name in CONFIGURATIONS:
        rebuilt = []
        i = 0
        for start, end, _kind in spans_for(doc, name):
            rebuilt.append(doc[i:start])
            rebuilt.append(doc[start:end])
            i = end
        rebuilt.append(doc[i:])
        assert "".join(rebuilt) == doc, name


# --- the fixture is only a spec if it contains the cases -------------------


def test_every_named_case_occurs_exactly_once(doc: str):
    for name, needle in CASES:
        assert doc.count(needle) == 1, f"{name}: needle {needle!r} is not unique in the fixture"


def test_the_fixture_exercises_every_protected_kind(doc: str):
    for name in CONFIGURATIONS:
        kinds = {k for _s, _e, k in spans_for(doc, name)}
        want = md.protected_kinds(raw_html=CONFIGURATIONS[name]["raw_html"])
        assert kinds == want, f"{name} fixture misses {want - kinds}"


def test_the_headings_are_not_swallowed_by_any_span(doc: str):
    # An indented heading and a heading glued to the prose under it are the two
    # ways a heading gets mis-parsed; neither may end up inside a protected span.
    for case in build_record(doc)["cases"]:
        if case["name"].endswith("heading"):
            assert case["protected_in"] == [], case
            assert md.is_heading_line(doc[case["start"] : case["end"]])


def test_the_unclosed_div_protects_nothing(doc: str):
    for name in CONFIGURATIONS:
        spans = spans_for(doc, name)
        assert not _covered(spans, doc.index("an-opener"), len(doc))
        assert spans[-1][1] < doc.index("::: an-opener-that-is-never-closed")
