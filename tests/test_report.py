"""Tests for the read-only target visualiser (stylebot.report)."""

from __future__ import annotations

import json

from stylebot import eval as ev
from stylebot import report, synth
from stylebot.synth import Target


def _targets(n, base_len=120):
    return [
        Target(text="word " * (base_len // 5 + i), source=f"post/p{i % 3}.qmd", chunk_index=i, chunk_total=n)
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
    evil = Target(text="<script>alert(1)</script> & <b>bold</b>", source="post/e.qmd", chunk_index=0, chunk_total=1)
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
    stats = json.loads(report.stats_json(_targets(50)))
    assert stats["count"] == 50  # stats cover all


def test_summary_stats():
    stats = json.loads(report.stats_json(_targets(5)))
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
        ("Body prose under a heading, terse and idiosyncratic, the Dan target here.", "## A Heading", "catalogue"),
        ("<script>alert(1)</script> and then a real authored paragraph after the tag.", None, "polish"),
    ]
    for i, (text, context, strategy) in enumerate(specs):
        t = synth.Target(text=text, source=f"post/p{i}.qmd", chunk_index=0, chunk_total=1, context=context or "")
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
        t = synth.Target(text=f"Paragraph number {i} of authored prose to be scored now.", source=f"post/p{i}.qmd", chunk_index=0, chunk_total=1)
        synth.synthesize_pairs([t], d, [synth.Generator(name="m", generate=lambda s: "slop: " + s, strategy="polish")])
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
            generate=lambda s, e=effort: synth.GenOutput("slop: " + s, {"model": "anthropic/claude-opus-4.8", "reasoning_effort": e}),
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
    assert "reasoning=high" in h  # per-pair generation-covariate sub-line
    assert "anthropic/claude-opus-4.8" in h


def test_scores_report_facet_by_covariate(tmp_path):
    pairs, scores = _make_scored_with_gen(tmp_path, judge=_judge)
    p = tmp_path / "r.html"
    report.render_scores_report(scores, pairs, p, facet_by="reasoning_effort")
    h = p.read_text()
    assert "<th>reasoning_effort</th>" in h  # headline grouped by the covariate
    assert "<th>slop_strategy</th>" not in h  # not the default facet
