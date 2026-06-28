"""Tests for the read-only target visualiser (stylebot.report)."""

from __future__ import annotations

import json

from stylebot import report
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
