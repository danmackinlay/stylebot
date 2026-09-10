"""Tests for the fit judge (D97) — no network, no API key.

The judge is injected as a plain fake. These gate what the signal is worthless
without: neighbours cut the way the design says (protected blocks skipped,
headings breaking adjacency), presentation order randomised *and* recorded, and
ties excluded from the interval rather than split.
"""

from __future__ import annotations

import json

import pytest

from stylebot import fit, fit_judge
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.draws import ViewOpts

CUT = fit.Cut(max_chars=200)
WIDE = fit.Cut(max_chars=4500)

DOC = """---
title: Badgers
---

# Setting out

The badger arrives at dusk, which is when the fence matters.

```python
fence = "not prose"
```

$$
x = y
$$

> a quoted thing

It leaves by the same gap it came in through.

## A second section

Nothing here refers back.
"""


def pair(rid: str, source: str, target: str, output: str, *, ctx: str = "") -> dict:
    """A minimal Phase-4 rewrite record: pairs schema + top-level `output`."""
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": "sloppy version"},
            {"role": "assistant", "content": target},
        ],
        "output": output,
        "meta": {
            "source": source,
            "captured_at": "2026-08-21T00:00:00Z",
            "capture_id": rid,
            "chunk_index": 0,
            "chunk_total": 1,
            "context": ctx,
        },
    }


def write_jsonl(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def synth_pair(source, i, total, target, output, *, ctx=""):
    r = pair(f"cap{i}", source, target, output, ctx=ctx)
    r["meta"].update(
        {"synthetic": True, "chunk_index": i, "chunk_total": total, "synth_key": f"{source}#{i}"}
    )
    return r


# --- the stratum -----------------------------------------------------------


def test_protected_blocks_are_skipped_not_used_as_neighbours():
    got = fit.fits_from_document(DOC, "post/b.qmd", cut=CUT)
    first = next(s for s in got if s.span.startswith("The badger arrives"))
    # A fence, display math and a blockquote sit between the two paragraphs.
    # None of them is prose the span must agree with, so none is the neighbour.
    assert first.after.startswith("It leaves by the same gap")
    assert "fence" not in first.after and ">" not in first.after


def test_a_heading_breaks_adjacency():
    got = fit.fits_from_document(DOC, "post/b.qmd", cut=CUT)
    last = next(s for s in got if s.span.startswith("Nothing here"))
    assert last.before == ""  # the paragraph before it is in another section
    closing = next(s for s in got if s.span.startswith("It leaves"))
    assert closing.after == ""


def test_heading_context_matches_the_pair_contract():
    got = fit.fits_from_document(DOC, "post/b.qmd", cut=CUT)
    assert got[0].context == "# Setting out"
    assert got[-1].context == "## A second section"


def test_interior_flags_both_sided_cases():
    # One paragraph per chunk needs a budget under the paragraph length:
    # `_segment` groups adjacent prose up to `max_chars`, exactly as training
    # cut it, so a generous budget legitimately yields ONE span and no case.
    doc = "one paragraph.\n\ntwo paragraph.\n\nthree paragraph.\n"
    got = fit.fits_from_document(doc, "x.qmd", cut=fit.Cut(max_chars=15))
    assert [s.is_interior for s in got] == [False, True, False]


def test_a_generous_budget_swallows_the_boundaries_it_would_have_judged():
    """Not a bug — the honest consequence of cutting at the SERVING diet.

    At 4500 chars a short post is one chunk, so there is no boundary the styler
    could have broken and nothing for this signal to judge. The stratum shrinks
    visibly rather than inventing sub-chunk boundaries the adapter never saw.
    """
    doc = "one paragraph.\n\ntwo paragraph.\n\nthree paragraph.\n"
    got = fit.fits_from_document(doc, "x.qmd", cut=WIDE)
    assert len(got) == 1 and got[0].is_interior is False


def test_locate_neighbours_matches_by_content():
    found = fit.locate_neighbours(DOC, "It leaves by the same gap it came in through.", cut=CUT)
    assert found is not None
    before, after = found
    assert before.startswith("The badger arrives")
    assert after == ""  # a heading follows


def test_locate_neighbours_misses_loudly():
    assert fit.locate_neighbours(DOC, "text that never shipped", cut=CUT) is None
    assert fit.locate_neighbours(DOC, "   ", cut=CUT) is None


def test_locate_neighbours_spans_several_blocks():
    doc = "alpha para.\n\nbeta para.\n\ngamma para.\n\ndelta para.\n"
    # A merged chunk covering beta+gamma must be flanked by alpha and delta,
    # not by its own interior.
    found = fit.locate_neighbours(doc, "beta para.\n\ngamma para.", cut=fit.Cut(max_chars=12))
    assert found == ("alpha para.", "delta para.")


# --- presentation order ----------------------------------------------------


def test_a_first_is_deterministic_and_salted():
    assert fit_judge.a_first("x#1") is fit_judge.a_first("x#1")
    ids = [f"post#{i}" for i in range(400)]
    share = sum(fit_judge.a_first(i) for i in ids) / len(ids)
    assert 0.4 < share < 0.6
    # A salt re-randomises the whole comparison without touching the ids.
    assert [fit_judge.a_first(i) for i in ids] != [fit_judge.a_first(i, salt="rep2") for i in ids]


def _triples(n, span_a="A", span_b="B", ids=None):
    return [
        (
            fit.FitCase(
                id=(ids[i] if ids else f"s#{i}"),
                source="p.qmd",
                index=i,
                before="b",
                span=span_a,
                after="c",
            ),
            span_a,
            span_b,
        )
        for i in range(n)
    ]


def test_choice_is_normalised_to_the_arm_not_the_slot(tmp_path):
    """The judge always picks slot 1; the winner must follow the shuffle."""

    def always_first(before, first, second, after):
        return {"choice": 1, "rationale": "first"}

    out = tmp_path / "fits.jsonl"
    fit_judge.judge_fits(_triples(40), out, judge=always_first, arm_a="run4a", arm_b="run2")
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    for r in recs:
        assert r["choice"] == r["presented"][0]  # slot 1 == whoever was shown first
    assert {r["choice"] for r in recs} == {"run4a", "run2"}  # order really varies


def test_judge_fits_is_resumable(tmp_path):
    calls = []

    def counting(before, first, second, after):
        calls.append(first)
        return {"choice": 0, "rationale": "tie"}

    triples = _triples(3)
    out = tmp_path / "fits.jsonl"
    fit_judge.judge_fits(triples[:2], out, judge=counting)
    again = fit_judge.judge_fits(triples, out, judge=counting)
    assert again.skipped_existing == 2
    assert again.n_judged == 1
    assert len(calls) == 3


def test_an_async_judge_is_awaited(tmp_path):
    async def judge(before, first, second, after):
        return {"choice": 2, "rationale": "second"}

    out = tmp_path / "fits.jsonl"
    res = fit_judge.judge_fits(_triples(6), out, judge=judge, max_workers=3)
    assert res.n_judged == 6
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(r["choice"] == r["presented"][1] for r in recs)


def test_a_judge_error_costs_one_case_not_the_run(tmp_path):
    def flaky(before, first, second, after):
        if "boom" in (first, second):
            raise RuntimeError("upstream 500")
        return {"choice": 1, "rationale": "ok"}

    triples = _triples(2) + [
        (
            fit.FitCase(id="bad#1", source="p.qmd", index=9, before="b", span="x", after="c"),
            "boom",
            "B",
        )
    ]
    out = tmp_path / "fits.jsonl"
    res = fit_judge.judge_fits(triples, out, judge=flaky)
    assert res.n_judged == 2
    assert len(res.errors) == 1 and "bad#1" in res.errors[0]


# --- parsing and summary ---------------------------------------------------


def test_parse_fit_reply():
    assert fit_judge.parse_fit_reply('{"choice": 2, "rationale": "r"}')["choice"] == 2
    assert fit_judge.parse_fit_reply('junk {"choice": 0, "rationale": ""} junk')["choice"] == 0
    assert fit_judge.parse_fit_reply("I pick 1.")["choice"] == 1
    # Unparseable falls back to TIE, never to a side: a coin-flip fallback would
    # inject noise straight into the win counts.
    fell_back = fit_judge.parse_fit_reply("the model refused")
    assert fell_back["choice"] == 0 and fell_back["unparsed"] is True


def test_summary_excludes_ties_from_the_interval():
    recs = [{"choice": "a", "interior": True}] * 8 + [{"choice": "b", "interior": True}] * 2
    recs += [{"choice": "tie", "interior": False}] * 90
    got = fit_judge.summarize_fits(recs)
    assert got["n"] == 100 and got["n_decided"] == 10 and got["ties"] == 90
    assert got["a_win_rate"] == 0.8  # 8/10, not 8/100
    assert got["n_interior"] == 10


def test_summary_calls_a_wash_a_wash():
    even = [{"choice": "a"}] * 30 + [{"choice": "b"}] * 30
    assert fit_judge.summarize_fits(even)["decisive"] is False
    lopsided = [{"choice": "a"}] * 55 + [{"choice": "b"}] * 5
    got = fit_judge.summarize_fits(lopsided)
    assert got["decisive"] is True
    assert got["a_win_rate_ci95"][0] > 0.5


def test_summary_of_nothing_is_not_a_verdict():
    got = fit_judge.summarize_fits([])
    assert got["n"] == 0 and got["a_win_rate"] is None and got["decisive"] is False


# --- the rewrite join ------------------------------------------------------


def test_neighbours_come_from_corpus_siblings(tmp_path):
    """The exact path: adjacent chunk_index in the same cut IS adjacent prose."""
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(
        corpus,
        [
            synth_pair("post/b.qmd", 0, 3, "First chunk of Dan.", "x"),
            synth_pair("post/b.qmd", 1, 3, "Middle chunk of Dan.", "x"),
            synth_pair("post/b.qmd", 2, 3, "Last chunk of Dan.", "x"),
        ],
    )
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [synth_pair("post/b.qmd", 1, 3, "Middle chunk of Dan.", "Arm A middle.")])
    write_jsonl(b, [synth_pair("post/b.qmd", 1, 3, "Middle chunk of Dan.", "Arm B middle.")])
    triples, rep = fit.fits_from_rewrites(a, b, corpus_path=corpus, cut=WIDE)
    assert rep.n_built == 1 and rep.by_neighbour_source == {"sibling": 1}
    case, span_a, span_b = triples[0]
    assert case.before == "First chunk of Dan." and case.after == "Last chunk of Dan."
    assert case.is_interior and case.neighbour_source == "sibling"
    assert (span_a, span_b) == ("Arm A middle.", "Arm B middle.")


def test_a_section_break_walls_off_a_sibling(tmp_path):
    """Different heading context = another section = not a neighbour."""
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(
        corpus,
        [
            synth_pair("post/b.qmd", 0, 3, "Under one.", "x", ctx="# One"),
            synth_pair("post/b.qmd", 1, 3, "Also under one.", "x", ctx="# One"),
            synth_pair("post/b.qmd", 2, 3, "Under two.", "x", ctx="# Two"),
        ],
    )
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, out in ((a, "A"), (b, "B")):
        write_jsonl(path, [synth_pair("post/b.qmd", 1, 3, "Also under one.", out, ctx="# One")])
    triples, _ = fit.fits_from_rewrites(a, b, corpus_path=corpus, cut=WIDE)
    case = triples[0][0]
    assert case.before == "Under one."
    assert case.after == ""  # index 2 is under a different heading


def test_a_different_cut_is_not_a_neighbour(tmp_path):
    """Index 0 of a 3-chunk cut is not index 0 of a 12-chunk cut of the same post."""
    corpus = tmp_path / "pairs.jsonl"
    other = synth_pair("post/b.qmd", 0, 12, "From a finer cut.", "x")
    write_jsonl(corpus, [other, synth_pair("post/b.qmd", 1, 3, "Middle.", "x")])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, out in ((a, "A"), (b, "B")):
        write_jsonl(path, [synth_pair("post/b.qmd", 1, 3, "Middle.", out)])
    triples, rep = fit.fits_from_rewrites(a, b, corpus_path=corpus, cut=WIDE)
    assert triples == [] and rep.unplaceable == ["post/b.qmd#1"]


def test_real_records_never_use_sibling_adjacency(tmp_path):
    """`diff_chunks` emits only CHANGED regions, so index+1 can be pages away.

    A real record must fall through to content matching (archive, then post) or
    be reported unplaceable — fabricated adjacency is worse than none.
    """
    corpus = tmp_path / "pairs.jsonl"
    recs = []
    for i, t in enumerate(["First region.", "Second region.", "Third region."]):
        r = pair("cap", "post/b.qmd", t, "x")  # no meta.synthetic
        r["meta"].update({"chunk_index": i, "chunk_total": 3})
        recs.append(r)
    write_jsonl(corpus, recs)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for path, out in ((a, "A"), (b, "B")):
        rec = pair("cap", "post/b.qmd", "Second region.", out)
        rec["meta"].update({"chunk_index": 1, "chunk_total": 3})
        write_jsonl(path, [rec])
    triples, rep = fit.fits_from_rewrites(a, b, corpus_path=corpus, cut=WIDE)
    assert triples == [] and rep.unplaceable == ["cap:1"]


def _archive_arms(tmp_path, *, archive_id: str, meta_extra: dict | None = None):
    """Two one-record arms whose target lives in `captures/<archive_id>/after.md`."""
    write_jsonl(tmp_path / "pairs.jsonl", [])
    (tmp_path / "captures" / archive_id).mkdir(parents=True)
    (tmp_path / "captures" / archive_id / "after.md").write_text(DOC, encoding="utf-8")
    target = "It leaves by the same gap it came in through."
    paths = []
    for name, out in (("a.jsonl", "It slips out."), ("b.jsonl", "It exits.")):
        rec = pair("cap", "post/b.qmd", target, out)
        rec["meta"].update(meta_extra or {})
        write_jsonl(tmp_path / name, [rec])
        paths.append(tmp_path / name)
    return paths


def test_capture_archive_places_what_siblings_cannot(tmp_path):
    a, b = _archive_arms(tmp_path, archive_id="cap")
    triples, rep = fit.fits_from_rewrites(
        a, b, corpus_path=tmp_path / "pairs.jsonl", captures_dir=tmp_path / "captures", cut=CUT
    )
    assert rep.by_neighbour_source == {"archive": 1}
    assert triples[0][0].before.startswith("The badger arrives")


def test_a_recut_record_follows_archive_id_not_capture_id(tmp_path):
    """`log recut` mints a fresh capture_id; the archive is `meta.archive_id` (D96)."""
    a, b = _archive_arms(tmp_path, archive_id="arch1", meta_extra={"archive_id": "arch1"})
    triples, rep = fit.fits_from_rewrites(
        a, b, corpus_path=tmp_path / "pairs.jsonl", captures_dir=tmp_path / "captures", cut=CUT
    )
    assert rep.by_neighbour_source == {"archive": 1}
    assert triples[0][0].neighbour_source == "archive"


def test_the_live_post_is_the_last_resort(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(corpus, [])
    blog = tmp_path / "blog"
    (blog / "post").mkdir(parents=True)
    (blog / "post" / "b.qmd").write_text(DOC, encoding="utf-8")
    target = "It leaves by the same gap it came in through."
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [pair("cap", "post/b.qmd", target, "It slips out.")])
    write_jsonl(b, [pair("cap", "post/b.qmd", target, "It exits.")])
    triples, rep = fit.fits_from_rewrites(a, b, corpus_path=corpus, blog_root=blog, cut=CUT)
    assert rep.by_neighbour_source == {"post": 1}
    assert triples[0][0].neighbour_source == "post"


def test_an_empty_arm_output_is_not_a_case(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(corpus, [synth_pair("post/b.qmd", 0, 2, "A.", "x")])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [synth_pair("post/b.qmd", 1, 2, "B.", "")])
    write_jsonl(b, [synth_pair("post/b.qmd", 1, 2, "B.", "something")])
    triples, rep = fit.fits_from_rewrites(a, b, corpus_path=corpus, cut=WIDE)
    assert triples == [] and rep.empty_output == ["post/b.qmd#1"]


def test_only_in_one_arm_is_counted_not_judged(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(corpus, [])
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, [synth_pair("post/b.qmd", 0, 1, "A.", "out")])
    write_jsonl(b, [])
    triples, rep = fit.fits_from_rewrites(a, b, corpus_path=corpus, cut=WIDE)
    assert triples == [] and rep.only_in_a == 1 and rep.n_joined == 0


def _control_corpus(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(
        corpus,
        [
            synth_pair("post/b.qmd", 0, 3, "First chunk.", "x"),
            synth_pair("post/b.qmd", 1, 3, "Dan's own middle.", "x"),
            synth_pair("post/b.qmd", 2, 3, "Last chunk.", "x"),
        ],
    )
    one = tmp_path / "one.jsonl"
    write_jsonl(one, [synth_pair("post/b.qmd", 1, 3, "Dan's own middle.", "the rewrite")])
    return corpus, one


def test_field_selectors_drive_the_positive_control(tmp_path):
    """`--field-a target --field-b slop` on ONE file is the resolution check."""
    corpus, one = _control_corpus(tmp_path)
    triples, _ = fit.fits_from_rewrites(
        one, one, corpus_path=corpus, field_a="target", field_b="slop", cut=WIDE
    )
    _, span_a, span_b = triples[0]
    assert span_a == "Dan's own middle." and span_b == "sloppy version"


def test_an_unknown_field_is_refused(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    write_jsonl(corpus, [])
    one = tmp_path / "one.jsonl"
    write_jsonl(one, [synth_pair("post/b.qmd", 0, 1, "t", "o")])
    with pytest.raises(ValueError, match="unknown field"):
        fit.fits_from_rewrites(one, one, corpus_path=corpus, field_a="nope", cut=WIDE)


# --- the pass (D78) --------------------------------------------------------


def test_judge_fit_pass_runs_keyless_with_an_injected_judge(tmp_path):
    corpus, one = _control_corpus(tmp_path)
    lines: list[str] = []
    opts = fit_judge.FitOpts(
        a_path=one,
        b_path=one,
        field_a="target",
        field_b="slop",
        arm_a="dan",
        arm_b="slop",
        pairs_path=corpus,
        max_chunk_chars=4500,
        out_path=tmp_path / "fits.jsonl",
        summary_path=tmp_path / "summary.json",
        judge=lambda before, first, second, after: {"choice": 1, "rationale": "r"},
    )
    text = fit_judge.judge_fit_pass(
        opts, ViewOpts(), emit=lambda m, *, level="info": lines.append(m)
    )
    summary = json.loads(text)
    assert summary["n"] == 1 and summary["arms"] == ["dan", "slop"]
    assert json.loads((tmp_path / "summary.json").read_text())["n"] == 1
    assert any("stratum: 1 cases" in line for line in lines)
    assert any("neighbours: sibling 1" in line for line in lines)


def test_dry_run_builds_the_stratum_and_calls_no_judge(tmp_path):
    corpus, one = _control_corpus(tmp_path)
    opts = fit_judge.FitOpts(
        a_path=one,
        b_path=one,
        pairs_path=corpus,
        max_chunk_chars=4500,
        dry_run=True,
        out_path=tmp_path / "fits.jsonl",
    )
    assert fit_judge.judge_fit_pass(opts, ViewOpts()) == ""
    assert not (tmp_path / "fits.jsonl").exists()
