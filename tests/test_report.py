"""Tests for the read-only HTML reports (stylebot.report)."""

from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from stylebot import eval as ev
from stylebot import report, synth
from stylebot.pairs import iter_pairs, record_id
from stylebot.synth import Target


def _targets(n, base_len=120):
    return [
        Target(
            text="word " * (base_len // 5 + i),
            source=f"post/p{i % 3}.qmd",
            chunk_index=i,
            chunk_total=n,
        )
        for i in range(n)
    ]


def test_render_writes_self_contained_html(tmp_path):
    p = tmp_path / "r.html"
    report.render_targets_report(_targets(12), p)
    h = p.read_text()
    assert "<svg" in h
    assert "http://" not in h and "https://" not in h  # no network assets
    assert "<script src=" not in h  # no external scripts


def test_report_escapes_target_text(tmp_path):
    p = tmp_path / "r.html"
    evil = Target(
        text="<script>alert(1)</script> & <b>bold</b>",
        source="post/e.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    report.render_targets_report([evil], p, max_rows=None)
    h = p.read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in h
    assert "<script>alert(1)" not in h
    assert h.count("<script>") == 1  # only the report's own JS block


def test_report_max_rows_caps_table_not_stats(tmp_path):
    p = tmp_path / "r.html"
    report.render_targets_report(_targets(50), p, max_rows=10)
    h = p.read_text()
    assert h.count("<tr data-") == 10  # table capped
    assert "of 50" in h  # banner notes the full count
    assert report._summary_stats(_targets(50))["count"] == 50  # stats cover all


def test_summary_stats():
    stats = report._summary_stats(_targets(5))
    assert stats["count"] == 5
    assert stats["n_sources"] == 3
    assert stats["max"] >= stats["median"]


def test_sample_targets_seeded_deterministic():
    ts = _targets(20)
    a = report.sample_targets(ts, 5, seed=7)
    b = report.sample_targets(ts, 5, seed=7)
    assert [t.chunk_index for t in a] == [t.chunk_index for t in b]
    assert len(a) == 5


def test_sample_targets_caps_at_population():
    ts = _targets(3)
    assert len(report.sample_targets(ts, 10)) == 3


# --- scores report (joins pairs.jsonl + scores.jsonl) -----------------------


def _judge(text):
    # slop side (the fake generator prefixes "slop: ") scores low; Dan scores high.
    return {"score": 1 if text.startswith("slop:") else 4, "rationale": "ok <b>tag</b>"}


def _make_scored(tmp_path, *, judge=None):
    """Build a 2-pair corpus (2 strategies, one heading, one with <script>) and score it."""
    d = tmp_path / "corpus"
    specs = [
        (
            "Body prose under a heading, terse and idiosyncratic, the Dan target here.",
            "## A Heading",
            "catalogue",
        ),
        (
            "<script>alert(1)</script> and then a real authored paragraph after the tag.",
            None,
            "polish",
        ),
    ]
    for i, (text, context, strategy) in enumerate(specs):
        t = synth.Target(
            text=text,
            source=f"post/p{i}.qmd",
            chunk_index=0,
            chunk_total=1,
            context=context or "",
        )
        g = synth.Generator(name="m", generate=lambda s: "slop: " + s, strategy=strategy)
        synth.synthesize_pairs([t], d, [g])
    pairs, scores = d / "pairs.jsonl", tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, scores, judge=judge)
    return pairs, scores


def test_scores_report_self_contained(tmp_path):
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    assert "<svg" in h
    assert "http://" not in h and "https://" not in h  # no network assets
    assert "<script src=" not in h
    assert h.count("<script>") == 1  # only the report's own JS block


def test_scores_report_escapes_text(tmp_path):
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p, max_rows=None)
    h = p.read_text()
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in h  # Dan-side text escaped
    assert "<script>alert(1)" not in h
    assert "ok &lt;b&gt;tag&lt;/b&gt;" in h  # judge rationale escaped too


def test_scores_report_joins_and_headlines(tmp_path):
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    assert "terse and idiosyncratic" in h  # joined Dan body text (heading stripped)
    assert "catalogue" in h and "polish" in h  # per-strategy headline
    assert 'data-delta="3"' in h  # Dan 4 - slop 1


def test_scores_report_sort_attributes(tmp_path):
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    for attr in ("data-strategy=", "data-delta=", "data-slop=", "data-dan="):
        assert attr in h


def test_scores_report_keyless_no_crash(tmp_path):
    pairs, scores = _make_scored(tmp_path, judge=None)  # no judge -> scores None
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    assert 'data-delta=""' in h  # delta unavailable -> empty (sorts last)
    assert "—" in h  # placeholder badge / delta


def test_scores_report_max_rows_caps(tmp_path):
    d = tmp_path / "c"
    for i in range(6):
        t = synth.Target(
            text=f"Paragraph number {i} of authored prose to be scored now.",
            source=f"post/p{i}.qmd",
            chunk_index=0,
            chunk_total=1,
        )
        synth.synthesize_pairs(
            [t],
            d,
            [synth.Generator(name="m", generate=lambda s: "slop: " + s, strategy="polish")],
        )
    pairs, scores = d / "pairs.jsonl", tmp_path / "s.jsonl"
    ev.score_pairs_file(pairs, scores, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p, max_rows=3)
    h = p.read_text()
    assert h.count("<tr data-strategy") == 3  # table capped (headline rows have no data-strategy)
    assert "of 6" in h  # note states the full count


def test_format_scores_sample_seeded(tmp_path):
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    a = report.format_scores_sample(scores, pairs, 2, seed=3)
    b = report.format_scores_sample(scores, pairs, 2, seed=3)
    assert a == b and "slop" in a and "target" in a


def _make_scored_with_gen(tmp_path, *, judge=None):
    """Build a 2-pair corpus whose generators record gen covariates (meta.gen)."""
    d = tmp_path / "corpus"
    specs = [
        ("A terse idiosyncratic Dan target paragraph, long enough to score here.", "high"),
        ("Another authored paragraph, comfortably long enough to be scored too.", "low"),
    ]
    for i, (text, effort) in enumerate(specs):
        t = synth.Target(text=text, source=f"post/p{i}.qmd", chunk_index=0, chunk_total=1)
        g = synth.Generator(
            name="m",
            reasoning_effort=effort,
            generate=lambda s, e=effort: synth.GenOutput(
                "slop: " + s, {"model": "anthropic/claude-opus-4.8", "reasoning_effort": e}
            ),
        )
        synth.synthesize_pairs([t], d, [g])
    pairs, scores = d / "pairs.jsonl", tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, scores, judge=judge)
    return pairs, scores


def test_scores_report_shows_gen_params(tmp_path):
    pairs, scores = _make_scored_with_gen(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    # Facets are plain-text COLUMNS, not a muted subline.
    assert "<td class=facet>high</td>" in h and "<td class=facet>low</td>" in h
    for col in ("generator", "reasoning", "prompt", "turn", "sim"):
        assert f">{col}</th>" in h


def test_scores_report_prompt_column_is_human_readable(tmp_path):
    # A pair recording prompt_label/version renders "catalogue v1" (hash id in
    # the tooltip) — never a bare content hash in the visible cell.
    d = tmp_path / "corpus"
    t = synth.Target(
        text="A long-enough authored paragraph for scoring purposes.",
        source="p.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    g = synth.Generator(
        name="m",
        generate=lambda s: synth.GenOutput(
            "slop: " + s,
            {"prompt_id": "dc0f6c5c5de6", "prompt_label": "catalogue", "prompt_version": 1},
        ),
    )
    synth.synthesize_pairs([t], d, [g])
    pairs, scores = d / "pairs.jsonl", tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, scores)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    assert '<td class=facet title="dc0f6c5c5de6">catalogue v1</td>' in h


def test_scores_report_facet_by_covariate(tmp_path):
    pairs, scores = _make_scored_with_gen(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p, facet_by="reasoning_effort")
    h = p.read_text()
    assert "<th>reasoning_effort</th>" in h  # headline grouped by the covariate
    assert "<th>slop_strategy</th>" not in h  # not the default facet


def test_scores_report_row_facet_attrs_and_dropdowns(tmp_path):
    pairs, scores = _make_scored_with_gen(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p)
    h = p.read_text()
    # Row-level covariate attributes feed the facet filters.
    assert 'data-reasoning="high"' in h and 'data-reasoning="low"' in h
    assert 'data-generator="m"' in h
    # A dropdown appears only for covariates that discriminate: two reasoning
    # levels -> dropdown; a single generator/strategy -> no dropdown.
    assert 'data-key="reasoning"' in h
    assert 'data-key="generator"' not in h
    assert 'data-key="strategy"' not in h


def test_scores_report_vale_chip_keyless(tmp_path):
    # Hand-built records so the Vale signal is present regardless of whether a
    # local vale binary exists; judge=None = the keyless pair-browser path.
    pairs, _ = _make_scored_with_gen(tmp_path, judge=None)
    from stylebot.pairs import iter_pairs

    recs = [
        {
            "id": record_id(pr),
            "meta": {"slop_strategy": "polish", "source": pr["meta"]["source"], "generator": "m"},
            "scores": {
                f: {
                    "vale": {"available": True, "alerts": 3},
                    "judge": None,
                    "detector": {"score": None},
                }
                for f in ("slop", "target")
            },
        }
        for pr in iter_pairs(pairs)
    ]
    p = tmp_path / "r.html"
    report.render_scores_report(recs, pairs, p)
    h = p.read_text()
    assert "vale 3" in h  # the third signal is now visible per cell
    assert "—" in h  # judge badges render the keyless placeholder


# --- the shared scaffolding (one CSS, one JS, one page) ---------------------
#
# Both reports are rendered from the same `_page` / `_CSS` / `_JS`, so the way
# they can regress is by drifting apart again, or by the markup and the script
# disagreeing about a name. These pin the joins between them.


def _assets(h):
    """The `<style>` and `<script>` bodies of a rendered report."""
    return (
        re.search(r"<style>(.*?)</style>", h, re.S).group(1),
        re.search(r"<script>(.*?)</script>", h, re.S).group(1),
    )


class _Markup(HTMLParser):
    """Every tag and attribute name the browser would actually see."""

    def __init__(self, h):
        super().__init__()
        self.tags: list[str] = []
        self.attrs: set[str] = set()
        self.feed(h)

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attrs.update(k for k, _ in attrs)


def _both(tmp_path):
    """One targets report and one scores report over the same fixture corpus."""
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    t, s = tmp_path / "t.html", tmp_path / "s.html"
    report.render_targets_report(_targets(4), t)
    report.render_scores_report(scores, pairs, s)
    return t.read_text(), s.read_text()


def test_both_reports_share_one_css_and_one_js(tmp_path):
    th, sh = _both(tmp_path)
    assert _assets(th) == _assets(sh)  # the whole point of the consolidation


def test_controls_bind_to_the_data_table_not_the_headline(tmp_path):
    # The scores page renders the headline table FIRST, so a bare
    # querySelector('tbody') would wire every control to the wrong table.
    th, sh = _both(tmp_path)
    for h in (th, sh):
        assert "document.querySelector('#rows tbody')" in h
        assert h.count("<table id=rows>") == 1
    assert sh.index("<table class=headline>") < sh.index("<table id=rows>")
    assert "id=rows" not in sh[: sh.index("<table id=rows>")]


def test_sort_handlers_match_the_shared_js_signature(tmp_path):
    # Every onclick in the markup must call the two-argument sortBy(key, numeric).
    for h in _both(tmp_path):
        markup = h[: h.index("<script>")]
        calls = re.findall(r"sortBy\(([^)]*)\)", markup)
        assert calls
        assert all(re.fullmatch(r"'\w+',(true|false)", c) for c in calls), calls


def test_reports_carry_a_dark_palette(tmp_path):
    for h in _both(tmp_path):
        assert "@media (prefers-color-scheme:dark)" in h


def test_empty_histogram_is_an_empty_svg():
    # _score_histos leans on this rather than guarding each series itself.
    assert report._histogram_svg([], bins=5) == "<svg></svg>"


# --- escaping (a security property, so test the parsed DOM, not substrings) --


def _skeleton(h):
    """The document's shape: its tag sequence and the attribute names used.

    Hostile corpus text may change what a report *says*; it must never change
    this. Comparing a hostile render against a benign twin catches an injected
    tag or attribute without having to enumerate the payloads to look for.
    """
    m = _Markup(h)
    return m.tags, sorted(m.attrs)


def test_hostile_target_cannot_inject_tags_or_attributes(tmp_path):
    def render(text, source, context):
        p = tmp_path / f"{abs(hash(text))}.html"
        t = Target(text=text, source=source, chunk_index=0, chunk_total=1, context=context)
        report.render_targets_report([t], p, max_rows=None)
        return p.read_text()

    evil = render(
        '<img src=x onerror="alert(1)"> & <b>bold</b>',
        'post/e" onmouseover="alert(2)".qmd',
        '## H "quoted" <script>alert(3)</script>',
    )
    assert _skeleton(evil) == _skeleton(render("plain text", "post/e.qmd", "## H"))
    assert _Markup(evil).tags.count("script") == 1  # only the report's own script
    # …and the hostile strings survive as readable, inert text.
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in evil
    assert "post/e&quot; onmouseover=&quot;alert(2)&quot;.qmd" in evil


def test_scores_report_escapes_hostile_facet_values(tmp_path):
    pairs, _ = _make_scored_with_gen(tmp_path, judge=None)

    def render(strategy):
        recs = [
            {
                "id": record_id(pr),
                "meta": {"slop_strategy": strategy, "source": pr["meta"]["source"]},
                "scores": {
                    f: {"judge": {"score": 2, "rationale": "<i>r</i>"}} for f in ("slop", "target")
                },
            }
            for pr in iter_pairs(pairs)
        ]
        p = tmp_path / f"{abs(hash(strategy))}.html"
        report.render_scores_report(recs, pairs, p)
        return p.read_text()

    # A strategy name reaches a data- attribute, a <td>, a <select> option and
    # the headline table — four escape sites for one value.
    evil = render('<b>x</b>" onclick="y')
    assert _skeleton(evil) == _skeleton(render("plain"))
    assert "&lt;b&gt;x&lt;/b&gt;&quot; onclick=&quot;y" in evil
    assert "&lt;i&gt;r&lt;/i&gt;" in evil  # the judge rationale too


# --- edges: empty input, the row cap, clipping, malformed signals -----------


def test_empty_targets_report_renders(tmp_path):
    p = tmp_path / "r.html"
    report.render_targets_report([], p)
    h = p.read_text()
    assert "showing all 0 targets" in h
    assert "<svg></svg>" in h and "<tbody></tbody>" in h
    assert "<b>0</b><span>targets</span>" in h


def test_empty_scores_report_renders(tmp_path):
    pairs, _ = _make_scored(tmp_path, judge=None)
    p = tmp_path / "r.html"
    report.render_scores_report([], pairs, p)
    h = p.read_text()
    assert "showing all 0 pairs" in h
    assert "<table class=headline>" not in h  # nothing to facet, no headline
    assert "<b>0</b><span>pairs</span>" in h
    assert "<tbody></tbody>" in h


def test_cap_note_states_what_the_stats_still_cover(tmp_path):
    p = tmp_path / "t.html"
    report.render_targets_report(_targets(50), p, max_rows=10)
    assert "showing 10 of 50 rows (stats + histogram cover all 50)" in p.read_text()
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    q = tmp_path / "s.html"
    report.render_scores_report(scores, pairs, q, max_rows=1)
    assert "showing 1 of 2 rows (stats + summary cover all 2)" in q.read_text()


def test_long_target_is_clipped_with_the_full_text_in_the_tooltip(tmp_path):
    body = "z" * 900
    t = Target(text=body, source="p.qmd", chunk_index=0, chunk_total=1)
    p = tmp_path / "r.html"
    report.render_targets_report([t], p, max_rows=None)
    h = p.read_text()
    assert f'title="{body}"' in h  # tooltip keeps everything
    assert "z" * 500 + "…</td>" in h  # the cell is clipped
    assert "z" * 501 + "…" not in h


def test_signal_reader_degrades_instead_of_raising():
    # One accessor now serves judge/detector/vale, so its tolerance of junk is
    # the tolerance of all three. Anything not shaped like a signal reads None.
    for junk in (None, "nonsense", 7, [], {"judge": None}, {"judge": 7}, {}):
        assert report._signal(junk, "judge") is None
        assert report._signal(junk, "judge", "rationale") is None
        assert report._vale_alerts(junk) is None
    assert report._signal({"judge": {"score": 3}}, "judge") == 3
    assert report._vale_alerts({"vale": {"available": True}}) == 0  # available, no alerts
    assert report._vale_alerts({"vale": {"available": False, "alerts": 9}}) is None


def test_scores_report_tolerates_malformed_signal_payloads(tmp_path):
    # A judge payload that isn't a dict degrades to a placeholder rather than
    # failing the whole report. (Only shapes `eval.summarize_scores` itself
    # survives are exercised end to end; `_signal`'s full tolerance is unit
    # tested above, since the headline goes through eval's aggregator.)
    pairs, _ = _make_scored_with_gen(tmp_path, judge=None)
    recs = [
        {
            "id": record_id(pr),
            "meta": {"slop_strategy": "polish", "source": pr["meta"]["source"]},
            "scores": {
                f: {"judge": 7, "vale": {"available": True, "alerts": 0}}
                for f in ("slop", "target")
            },
        }
        for pr in iter_pairs(pairs)
    ]
    p = tmp_path / "r.html"
    report.render_scores_report(recs, pairs, p)
    h = p.read_text()
    assert 'data-slop=""' in h and 'data-delta=""' in h
    assert "vale 0" in h  # available with zero alerts is a chip, not an omission


def test_detector_cards_appear_only_when_a_detector_ran(tmp_path):
    pairs, _ = _make_scored_with_gen(tmp_path, judge=None)
    base = [
        {"id": record_id(pr), "meta": {"slop_strategy": "polish"}, "scores": {}}
        for pr in iter_pairs(pairs)
    ]
    off = [
        dict(r, scores={f: {"detector": {"score": None}} for f in ("slop", "target")}) for r in base
    ]
    on = [
        dict(r, scores={f: {"detector": {"score": 0.5}} for f in ("slop", "target")}) for r in base
    ]
    p, q = tmp_path / "off.html", tmp_path / "on.html"
    report.render_scores_report(off, pairs, p)
    report.render_scores_report(on, pairs, q)
    assert "P(slop)" not in p.read_text()
    assert "mean slop P(slop)" in q.read_text() and "P(slop) 0.50" in q.read_text()


def test_middle_field_gets_the_neutral_badge(tmp_path):
    # Three fields: only the first and last are the compared ends.
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    recs = [
        dict(r, scores={**r["scores"], "output": r["scores"]["target"]})
        for r in ev.load_scores(scores)
    ]
    p = tmp_path / "r.html"
    report.render_scores_report(recs, pairs, p, fields=("slop", "output", "target"))
    h = p.read_text()
    assert "s-slop" in h and "s-dan" in h and "s-mid" in h


# --- the N-arm comparison report -------------------------------------------


def test_comparison_report_puts_every_arm_in_one_table(tmp_path):
    """The whole point: one table, so a read-through is not two browser tabs
    scrolled in sync with no shared row anchor."""
    pairs, scores = _make_scored(tmp_path, judge=_judge)
    p = tmp_path / "cmp.html"
    report.render_comparison_report(
        [("slop in", pairs, scores, "slop"), ("Dan", pairs, scores, "target")], p
    )
    h = p.read_text()
    assert h.count("<table") == 1
    assert "slop in</th>" in h and "Dan</th>" in h
    assert "http://" not in h and "https://" not in h  # self-contained


def test_comparison_report_drops_the_empty_covariate_columns(tmp_path):
    """strategy/generator/reasoning/prompt are the widest columns in the per-arm
    report and read '—' for every real captured pair. A comparison view carries
    source and the scores, nothing else."""
    pairs, scores = _make_scored(tmp_path)
    p = tmp_path / "cmp.html"
    report.render_comparison_report(
        [("a", pairs, scores, "slop"), ("b", pairs, scores, "target")], p
    )
    h = p.read_text()
    for col in ("generator</th>", "reasoning</th>", "prompt</th>", "turn</th>"):
        assert col not in h


def test_comparison_report_does_not_truncate_the_text(tmp_path):
    """Clipping is visual (CSS max-height + an expander), never a substring —
    a 400-char cut hides exactly the long rewrites whose coherence is under
    review."""
    long_text = "Sentence about the thing. " * 60
    d = tmp_path / "corpus"
    t = synth.Target(text=long_text, source="post/long.qmd", chunk_index=0, chunk_total=1)
    synth.synthesize_pairs([t], d, [synth.Generator(name="m", generate=lambda s: "slop: " + s)])
    pairs, scores = d / "pairs.jsonl", tmp_path / "s.jsonl"
    ev.score_pairs_file(pairs, scores)
    p = tmp_path / "cmp.html"
    report.render_comparison_report(
        [("a", pairs, scores, "slop"), ("b", pairs, scores, "target")], p
    )
    h = p.read_text()
    assert long_text.strip() in h
    assert "max-height" in h and "expandAll" in h


def test_comparison_report_needs_two_arms(tmp_path):
    pairs, scores = _make_scored(tmp_path)
    with pytest.raises(ValueError, match="at least two arms"):
        report.render_comparison_report([("only", pairs, scores, "slop")], tmp_path / "x.html")


def test_comparison_columns_are_equal_width_and_not_a_flex_row(tmp_path):
    """Two bugs in one test. `.cmp` is already claimed by the per-arm report's
    `display:flex` rule, which silently stopped this table laying out as a
    table; and arm widths must be declared equal, because the eye reads column
    width as significance and one arm holding longer text is not significance."""
    pairs, scores = _make_scored(tmp_path)
    p = tmp_path / "cmp.html"
    report.render_comparison_report(
        [("a", pairs, scores, "slop"), ("b", pairs, scores, "target")], p
    )
    h = p.read_text()
    assert "class=armtab" in h and "<table class=cmp" not in h
    widths = re.findall(r"<col style='width:([\d.]+)%'>", h)
    assert len(widths) == 3  # source + two arms
    assert widths[1] == widths[2]
