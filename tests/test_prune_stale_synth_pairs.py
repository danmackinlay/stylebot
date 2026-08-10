"""Tests for the one-off `scripts/prune_stale_synth_pairs.py` migration.

The script deletes corpus records, so its safety property gets a test even
though `scripts/` is outside the package: a pair is dropped ONLY when its
target became unproducible through the block-anchored-math change, never
because the blog drifted underneath it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from stylebot.ai_core import STYLE_SYSTEM

SCRIPT = Path(__file__).parent.parent / "scripts" / "prune_stale_synth_pairs.py"


@pytest.fixture(scope="module")
def prune():
    spec = importlib.util.spec_from_file_location("prune_stale_synth_pairs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY = dict(
    min_chars=40,
    max_chars=8000,
    prose_only=True,
    ignore_markers=(),
    drop_link_dumps=True,
    drop_list_items=False,
    merge=False,
    merge_max_chars=1500,
    stop_at_headers=(),
    heading_context="none",
    cross_heading=False,
)

# One paragraph carrying a mid-line `$$k$$`. Under the superseded segmentation
# the `$$k$$` was excised and the remnants joined; under the current one it
# stays put, so the two cuts of this post differ by exactly one target.
POST = """---
title: A post
automation: 0
---

The high-$$d$$ behaviour of this estimator is strange, and it stays strange
however long you look at it, which is the whole difficulty here.

A second paragraph that no segmentation change touches at all, long enough to
clear the minimum-character floor comfortably.
"""


def _pair(source: str, body: str, *, synth: bool = True) -> dict:
    meta = {
        "source": source,
        "captured_at": "2026-01-01T00:00:00Z",
        "capture_id": "abcd1234",
        "chunk_index": 0,
        "chunk_total": 1,
        "before_chars": len(body),
        "after_chars": len(body),
    }
    if synth:
        meta["synth_key"] = f"k:{hash(body)}"
        meta["synthetic"] = True
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": "slop " + body},
            {"role": "assistant", "content": body},
        ],
        "meta": meta,
    }


@pytest.fixture
def corpus(tmp_path: Path, prune):
    blog = tmp_path / "blog"
    blog.mkdir()
    (blog / "post.qmd").write_text(POST, encoding="utf-8")

    from stylebot import targets

    now = {t.text for t in targets.iter_targets(files=[blog / "post.qmd"], **POLICY)}
    with prune._legacy_segmentation():
        before = {t.text for t in targets.iter_targets(files=[blog / "post.qmd"], **POLICY)}

    stale_text = next(iter(before - now))  # the mangled cut, now unproducible
    live_text = next(iter(now - before))  # its corrected replacement

    records = [
        _pair("post.qmd", stale_text),
        _pair("post.qmd", live_text),
        _pair("post.qmd", "A target that matches nothing in the post any more, blog drift."),
        _pair("post.qmd", stale_text, synth=False),  # a real capture, never touched
    ]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return pairs, blog, stale_text, live_text


def test_the_fixture_really_has_a_cut_that_changed(corpus):
    _pairs, _blog, stale_text, live_text = corpus
    assert "high-" in stale_text and "$$d$$" not in stale_text
    assert "$$d$$" in live_text


def test_drops_only_the_newly_unproducible_synth_pair(corpus, prune):
    pairs, blog, stale_text, _live = corpus
    keep, report = prune.find_stale(pairs, blog, POLICY)
    assert report["stale_pairs"] == 1
    assert report["stale_texts"] == 1
    assert report["stale_sources"] == ["post.qmd"]
    kept_bodies = [r["messages"][2]["content"] for r in keep]
    assert kept_bodies.count(stale_text) == 1  # the real capture survives
    assert len(keep) == 3


def test_blog_drift_is_reported_but_left_alone(corpus, prune):
    pairs, blog, _stale, _live = corpus
    _keep, report = prune.find_stale(pairs, blog, POLICY)
    assert report["unproducible_left_alone"] == 1


def test_dry_run_writes_nothing(corpus, prune):
    pairs, blog, _stale, _live = corpus
    original = pairs.read_bytes()
    rc = prune.main([str(pairs), "--blog-root", str(blog), "--min-chars", "40"])
    assert rc == 0
    assert pairs.read_bytes() == original


def test_apply_prunes_and_keeps_a_backup(corpus, prune):
    pairs, blog, _stale, _live = corpus
    rc = prune.main([str(pairs), "--blog-root", str(blog), "--min-chars", "40", "--apply"])
    assert rc == 0
    assert len(pairs.read_text(encoding="utf-8").strip().splitlines()) == 3
    backup = pairs.with_suffix(pairs.suffix + ".bak")
    assert len(backup.read_text(encoding="utf-8").strip().splitlines()) == 4


def test_legacy_segmentation_is_restored_after_use(prune):
    from stylebot import targets
    from stylebot.lib import editable_prose

    with prune._legacy_segmentation():
        assert targets.editable_prose is not editable_prose
    assert targets.editable_prose is editable_prose
