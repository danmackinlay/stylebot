"""`stylebot.draws`: a view built from today's pairs reproduces them (D84).

The fast tests run on a fixture corpus written by `corpus.rewrite`, the real
writer, so byte-identity is against stylebot's own bytes and not against a
copy this module made. The `sweep` test does the same over the blog corpus and
prints the `content_sha256` `dan-style status` reports.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import conftest
import pytest

from stylebot import corpus, draws, pairs
from stylebot.errors import ConfigError

SWEEP_ENV = conftest.SWEEP_ENV


@pytest.fixture
def pairs_dir(tmp_path, write_pairs_corpus) -> Path:
    """A sharded `pairs/` of 8 records — two real posts, two synthetic."""
    single = write_pairs_corpus(tmp_path / "flat.jsonl", n_posts=4, chunks=2, n_synth_posts=2)
    out = tmp_path / "pairs"
    out.mkdir()  # `rewrite` shards only into a directory that already exists
    corpus.rewrite(out, corpus.read_records(single))
    assert len(corpus.shard_paths(out)) == 2, "captures + one synth month"
    return out


def _ids(path: Path) -> set[str]:
    return {pairs.record_id(r) for r in corpus.iter_records(path)}


# --- lifting pairs, building the identity view --------------------------------


def test_identity_view_is_byte_identical_to_its_pairs(pairs_dir, tmp_path):
    view = tmp_path / "view"
    report = draws.build_view(draws.draws_from_pairs(pairs_dir), draws.ViewPolicy(), out=view)

    assert corpus.content_sha256(view) == corpus.content_sha256(pairs_dir)
    assert report.content_sha256 == corpus.content_sha256(pairs_dir)
    assert _ids(view) == _ids(pairs_dir) and len(_ids(view)) == 8
    assert report.records == 8 and report.by_outcome == {"accepted": 8}
    assert report.excluded == {}
    assert [p.name for p in report.shards] == [p.name for p in corpus.shard_paths(pairs_dir)]


def test_a_lifted_draw_keeps_the_record_and_its_join_key(pairs_dir):
    lifted = list(draws.draws_from_pairs(pairs_dir))
    records = corpus.read_records(pairs_dir)
    assert [d.record for d in lifted] == records
    assert [d.draw_id for d in lifted] == [pairs.record_id(r) for r in records]
    real, synth = lifted[0], lifted[-1]
    assert real.outcome == draws.ACCEPTED and real.span is None and real.blog_sha is None
    assert real.superseded_by is None and real.raw_output is None and real.gen is None
    assert synth.raw_output == pairs.extract_slop(synth.record)
    assert synth.target_text == pairs.extract_target(synth.record)
    assert synth.drawn_at == synth.record["meta"]["captured_at"]


def test_build_view_refuses_a_directory_that_already_holds_one(pairs_dir, tmp_path):
    view = tmp_path / "view"
    draws.build_view(draws.draws_from_pairs(pairs_dir), draws.ViewPolicy(), out=view)
    with pytest.raises(ConfigError, match="fresh directory"):
        draws.build_view(draws.draws_from_pairs(pairs_dir), draws.ViewPolicy(), out=view)


# --- the clauses -------------------------------------------------------------


def test_a_superseded_draw_is_excluded_unless_the_policy_keeps_it(pairs_dir, tmp_path):
    lifted = list(draws.draws_from_pairs(pairs_dir))
    lifted[1] = dataclasses.replace(lifted[1], superseded_by=lifted[2].draw_id)

    report = draws.build_view(lifted, draws.ViewPolicy(), out=tmp_path / "v1")
    assert report.records == 7 and report.excluded == {"superseded": 1}
    assert lifted[1].draw_id not in _ids(tmp_path / "v1")

    kept = draws.build_view(lifted, draws.ViewPolicy(keep_superseded=True), out=tmp_path / "v2")
    assert kept.records == 8 and kept.excluded == {}
    assert corpus.content_sha256(tmp_path / "v2") == corpus.content_sha256(pairs_dir)


def test_rejects_and_named_exclusions_never_reach_the_view(pairs_dir, tmp_path):
    lifted = list(draws.draws_from_pairs(pairs_dir))
    lifted[0] = dataclasses.replace(lifted[0], outcome="gate:anchor")
    lifted[3] = dataclasses.replace(lifted[3], outcome="error:TimeoutError", record=None)
    policy = draws.ViewPolicy(excluded_draw_ids=(lifted[5].draw_id,))

    report = draws.build_view(lifted, policy, out=tmp_path / "view")
    assert report.by_outcome == {"accepted": 6, "gate:anchor": 1, "error:TimeoutError": 1}
    assert report.excluded == {"outcome": 2, "excluded_draw_ids": 1}
    assert report.records == 5
    assert _ids(tmp_path / "view") == {d.draw_id for d in lifted[1:3] + lifted[4:5] + lifted[6:]}


# --- draws on disk -------------------------------------------------------------


def test_append_then_iter_draws_round_trips(tmp_path):
    draw = draws.Draw(
        draw_id=draws.draw_id("post/a.qmd", "abc123", (10, 20), "gpt", "polish", "p1", "off"),
        source="post/a.qmd",
        blog_sha="abc123",
        blog_dirty=False,
        span=(10, 20),
        target_text="Dan's version.",
        context="## Heading",
        gen={"model": "gpt", "cost": 0.01},
        raw_output="It is worth noting.",
        outcome="gate:length",
        drawn_at="2026-09-04T10:00:00+00:00",
        record={"messages": [], "meta": {"source": "post/a.qmd"}},
    )
    shard = draws.append(tmp_path / "draws", draw)
    assert shard.name == "synth-2026-09.jsonl"
    undated = draws.append(tmp_path / "draws", dataclasses.replace(draw, drawn_at=None))
    assert undated.name == "synth-undated.jsonl"

    back = list(draws.iter_draws(tmp_path / "draws"))
    assert back == [draw, dataclasses.replace(draw, drawn_at=None)]
    assert isinstance(back[0].span, tuple)


def test_draw_id_is_a_function_of_the_design_cell():
    base = ("post/a.qmd", "abc123", (10, 20), "gpt", "polish", "p1", "off")
    assert draws.draw_id(*base) == draws.draw_id(*base)
    assert draws.draw_id(*base, replicate=1) != draws.draw_id(*base)
    assert draws.draw_id("post/b.qmd", *base[1:]) != draws.draw_id(*base)
    assert len(draws.draw_id(*base)) == 16


# --- the policy's name -----------------------------------------------------


def test_policy_sha_ignores_order_and_notices_every_field():
    a = draws.ViewPolicy(retired_arms=("x", "y"), system_sha="s", excluded_draw_ids=("1", "2"))
    b = draws.ViewPolicy(excluded_draw_ids=("2", "1"), system_sha="s", retired_arms=("y", "x"))
    assert draws.policy_sha(a) == draws.policy_sha(b)

    identity = draws.policy_sha(draws.ViewPolicy())
    changed = {
        "chunk_policy_digest": "d",
        "gate_thresholds": {"max_transform_sim": 0.95},
        "automation_cap": 1,
        "retired_arms": ("qwen/qwen3-8b",),
        "anchor_rule": "drop-invented",
        "role_rule": "splits:abcd",
        "system_sha": "ffff",
        "keep_superseded": True,
        "excluded_draw_ids": ("cap0000:0",),
    }
    assert set(changed) == {f.name for f in dataclasses.fields(draws.ViewPolicy)}
    shas = {name: draws.policy_sha(draws.ViewPolicy(**{name: v})) for name, v in changed.items()}
    assert identity not in shas.values()
    assert len(set(shas.values())) == len(shas)


# --- the opt-in sweep over the blog corpus ------------------------------------


@pytest.mark.sweep  # conftest deselects it unless SWEEP_ENV is set
def test_the_view_reproduces_the_blog_corpus(tmp_path):
    root = Path(os.environ[SWEEP_ENV]).expanduser()
    src = root / "_training_pairs" / "pairs"
    assert src.is_dir(), f"{SWEEP_ENV}={root} has no _training_pairs/pairs"

    report = draws.build_view(
        draws.draws_from_pairs(src), draws.ViewPolicy(), out=tmp_path / "view"
    )
    src_sha = corpus.content_sha256(src)
    print(f"\ncontent_sha256={report.content_sha256} records={report.records}")
    print(f"policy_sha={report.policy_sha} source_sha={src_sha}")
    assert report.content_sha256 == src_sha
    assert report.excluded == {} and set(report.by_outcome) == {draws.ACCEPTED}
    assert _ids(tmp_path / "view") == _ids(src)
    assert len(_ids(src)) == report.records
