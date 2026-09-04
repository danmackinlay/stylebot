"""What the two named segmenter configurations mean, written down.

This began as a *cross-implementation* spec: `livingthing.qmd_core` carried a
second implementation, because stylebot was reachable only through an optional
extra, and both repos asserted their own segmenter against one copied fixture
with a `fixture_sha256` gate. That ended on 2026-09-03 (W3) — bare stylebot
became a hard blog dependency, `segment_for_edit` became a thin call to
`markdown.protected_spans`, and the blog's copy of these two files went with the
duplication (stylebot `_plans/decisions.md` D28, superseded).

**What is left is not a transcript, and the distinction is the point.** The
`stylebot` and `livingthing` rows are not two runs of the same code; they are the
two *policies* D29 says must stay expressible as keyword arguments rather than as
forked code. Only one of them is exercised by anything else in this repo, so
without this file the blog's configuration would be pinned by nothing at all —
and `qmd_core` is the blog's editing path, `format-sentences` included. What the
JSON records is which constructs each policy protects and where the two diverge;
what it is for is making a change to either one arrive as a reviewable diff.

**The artefacts.** `fixtures/segmenter_conformance.qmd` is the document;
`fixtures/segmenter_conformance.json` is the expectation. The `fixture_sha256`
survives the deletion of the second copy for a smaller reason than it was added
for: the document is a `.qmd` under `tests/`, and a blog-side frontmatter
stamper walked into it twice (`128308fa55`, `2815a291fc`).

**Two named configurations, because the two callers legitimately differ.**

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

**What a mismatch means.** A failure here says what a rewrite may be handed has
moved, in one repo or both. The consequence is silent — an editor quietly
rewording somebody else's quoted words, or reflowing a `<script>` into a syntax
error, looks like a successful edit — and it now moves the *training corpus* as
well (`livingthing/tests/test_corpus_stability.py` is the other half of that
answer). Do not regenerate the JSON to make a red test green.

**When the change is deliberate**, `STYLEBOT_REGEN_SEGMENTS=1 uv run pytest
tests/test_segmenter_conformance.py` rewrites the JSON — and the regenerated
fixture belongs in the *same commit* as the policy change, so review sees the
segmentation move next to the reason for it. Never edit the JSON by hand.
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
    ("html_comment_single_line", "never prose, in any configuration"),
    ("html_comment_multi_line", "Its second sentence sits on its own line."),
    ("html_comment_midline", "A trailing comment, opened mid-line."),
    ("prose_between_comments", "which must stay prose"),
    ("unclosed_div", "an-opener-that-is-never-closed"),
    ("unclosed_fence", "an unclosed fence runs to end of document"),
]


#: The `CONFIGURATIONS` keys that select *kinds* rather than tune one kind's
#: shape. Derived rather than restated, so a third opt-in kind cannot be added to
#: `protected_spans` and silently left out of the invariant tests below.
_KIND_SWITCHES = ("raw_html", "html_comments")


def spans_for(doc: str, config: str) -> list[md.Span]:
    """The protected spans of `doc` under the named configuration."""
    return md.protected_spans(doc, **CONFIGURATIONS[config])


def covered_chars(doc: str, config: str) -> set[int]:
    """Every character index the named configuration protects.

    Coverage rather than span identity, because merging makes the two
    incomparable as *tuples* while leaving the real invariant intact. A `>` line
    inside a multi-line comment -- which is what a harvested reviewer reply looks
    like -- is a standalone blockquote span under a configuration that does not
    protect comments, and part of one larger comment span under one that does.
    Both protect the same characters, which is the only thing a consumer can rely
    on; asserting on the tuples made the fixture's richness look like a violation.
    """
    return {i for s, e, _k in spans_for(doc, config) for i in range(s, e)}


def kinds_for(config: str) -> frozenset[str]:
    """The kinds the named configuration is allowed to emit."""
    cfg = CONFIGURATIONS[config]
    return md.protected_kinds(**{k: cfg[k] for k in _KIND_SWITCHES})


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
    """Three declared policy differences, and nothing else.

    Each is a case of the same asymmetry: livingthing reassembles protected spans
    and stylebot deletes them, so livingthing can afford to protect a construct
    that sits *inside* a sentence and stylebot cannot.
    """
    lt = {(s, e) for s, e, _k in spans_for(doc, "livingthing")}
    sb = {(s, e) for s, e, _k in spans_for(doc, "stylebot")}
    extra = sorted(doc[s:e] for s, e in lt - sb)
    assert any("$$d$$" in x for x in extra), extra  # inline display math
    assert any(x.lstrip().startswith("<script") for x in extra), extra
    comments = [x for x in extra if x.lstrip().startswith("<!--")]
    assert len(comments) == 3, comments  # single-line, multi-line, mid-line
    assert len(extra) == 2 + len(comments), extra
    stylebot_chars = covered_chars(doc, "stylebot")
    livingthing_chars = covered_chars(doc, "livingthing")
    leaked = stylebot_chars - livingthing_chars
    assert not leaked, (
        "the stylebot config must protect a subset of livingthing's; "
        f"unprotected for livingthing: {doc[min(leaked):max(leaked) + 1]!r}"
    )


def test_html_comments_are_protected_for_livingthing_only(doc: str):
    cases = {c["name"]: c["protected_in"] for c in build_record(doc)["cases"]}
    for name in (
        "html_comment_single_line",
        "html_comment_multi_line",
        "html_comment_midline",
    ):
        assert cases[name] == ["livingthing"], name
    # The corollary that makes the mid-line case worth having: protecting a
    # comment must not drag the prose on either side of it out of reach.
    assert cases["prose_between_comments"] == []


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
            assert kind in kinds_for(name), name
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
        want = kinds_for(name)
        assert kinds == want, f"{name} fixture misses {want - kinds}"


def test_the_headings_are_not_swallowed_by_any_span(doc: str):
    # An indented heading and a heading glued to the prose under it are the two
    # ways a heading gets mis-parsed; neither may end up inside a protected span.
    for case in build_record(doc)["cases"]:
        if case["name"].endswith("heading"):
            assert case["protected_in"] == [], case
            assert md.is_heading_line(doc[case["start"] : case["end"]])


def test_the_unclosed_div_protects_nothing(doc: str):
    """A `:::` opener with no closer yields no span — the opposite of a fence.

    Deliberately opposite, and not pandoc's reading of the div half: pandoc
    closes it implicitly at EOF and warns. A div's contents render the same
    either way, so over-protecting to EOF would cost the rest of the post for
    nothing; an unclosed fence's contents ARE code and a rewrite destroys them.
    D76 argues it, and the count that decided it.
    """
    opener = doc.index("::: an-opener-that-is-never-closed")
    for name in CONFIGURATIONS:
        spans = spans_for(doc, name)
        assert not _covered(spans, doc.index("an-opener"), len(doc))
        # Nothing covers the opener line itself. (Something does start *after*
        # it — the unclosed fence below, which runs to EOF on its own merits.)
        assert not any(s <= opener < e for s, e, _k in spans)


def test_the_unclosed_fence_runs_to_end_of_document(doc: str):
    runaway = doc.index("an unclosed fence runs to end of document")
    for name in CONFIGURATIONS:
        span = next(sp for sp in spans_for(doc, name) if sp[0] <= runaway < sp[1])
        assert span[2] == md.FENCE, name
        assert span[1] == len(doc), name
        assert doc[span[0] :].startswith("```python\n"), name
