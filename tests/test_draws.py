"""`stylebot.draws`: a view built from today's pairs reproduces them (D84).

The fast tests run on a fixture corpus written by `corpus.rewrite`, the real
writer, so byte-identity is against stylebot's own bytes and not against a
copy this module made. The `sweep` test does the same over the blog corpus and
prints the `content_sha256` `dan-style status` reports.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import conftest
import pytest

from stylebot import config, corpus, draws, migrations, pairs
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


@pytest.fixture
def interleaved_dir(tmp_path, write_pairs_corpus) -> Path:
    """A `pairs/` whose real records span three months out of order, interleaved
    with synthetic ones — the live `captures.jsonl` shape (W4 step 3)."""
    flat = write_pairs_corpus(tmp_path / "flat.jsonl", n_posts=6, chunks=1, n_synth_posts=3)
    records = corpus.read_records(flat)
    ordered: list[dict] = []
    months = ("2026-07", "2026-05", "2026-09")
    for real, synth, month in zip(records[:3], records[3:], months, strict=True):
        real["meta"]["captured_at"] = f"{month}-15T00:00:00+00:00"
        ordered += [real, synth]
    out = tmp_path / "pairs"
    out.mkdir()
    corpus.rewrite(out, ordered)
    assert [p.name for p in corpus.shard_paths(out)] == ["captures.jsonl", "synth-2026-07.jsonl"]
    return out


def test_draws_on_disk_keep_corpus_order_for_real_records(interleaved_dir, tmp_path):
    for draw in draws.draws_from_pairs(interleaved_dir):
        draws.append(tmp_path / "draws", draw)
    names = sorted(p.name for p in (tmp_path / "draws").glob("*.jsonl"))
    assert names == ["captures.jsonl", "synth-2026-07.jsonl"], "shards mirror pairs/"

    back = list(draws.iter_draws(tmp_path / "draws"))
    assert [d.record for d in back] == corpus.read_records(interleaved_dir)
    report = draws.build_view(back, draws.ViewPolicy(), out=tmp_path / "view")
    assert report.content_sha256 == corpus.content_sha256(interleaved_dir)


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
        record={"messages": [], "meta": {"source": "post/a.qmd", "synthetic": True}},
    )
    draw.record["meta"]["captured_at"] = draw.drawn_at
    shard = draws.append(tmp_path / "draws", draw)
    assert shard.name == "synth-2026-09.jsonl"
    stone = dataclasses.replace(draw, drawn_at=None, record=None)
    assert draws.append(tmp_path / "draws", stone).name == "synth-undated.jsonl"

    back = list(draws.iter_draws(tmp_path / "draws"))
    assert back == [draw, stone]
    assert isinstance(back[0].span, tuple)


def test_draw_id_is_a_function_of_the_design_cell():
    base = ("post/a.qmd", "abc123", (10, 20), "gpt", "polish", "p1", "off")
    assert draws.draw_id(*base) == draws.draw_id(*base)
    assert draws.draw_id(*base, replicate=1) != draws.draw_id(*base)
    assert draws.draw_id("post/b.qmd", *base[1:]) != draws.draw_id(*base)
    assert len(draws.draw_id(*base)) == 16
    # A span names the cell; the chunk index only stands in while there is none.
    assert draws.draw_id(*base, chunk_index=3) == draws.draw_id(*base)
    unspanned = ("post/a.qmd", "abc123", None, "gpt", "polish", "p1", "off")
    assert draws.draw_id(*unspanned, chunk_index=0) != draws.draw_id(*unspanned, chunk_index=1)
    assert draws.draw_id(*unspanned, chunk_index=None) == draws.draw_id(*unspanned)


def test_new_draws_of_one_source_are_keyed_by_chunk_until_spans_exist():
    """W4 step 4: two chunks of one post under one arm were one `draw_id` (step 2)."""
    from stylebot.generators import Generator
    from stylebot.targets import Target

    arm = Generator(name="g", strategy="polish", prompt_id="p1", reasoning_effort="off")
    cut = [Target(text=f"c{i}", source="post/a.qmd", chunk_index=i, chunk_total=2) for i in (0, 1)]
    prov = {"blog_sha": "abc"}
    made = [
        draws.new_draw(arm, prov, t, None, gen=None, raw_output="x", outcome="gate:x") for t in cut
    ]
    assert made[0].draw_id != made[1].draw_id
    assert [d.chunk_index for d in made] == [0, 1] and all(d.span is None for d in made)
    cell = ("post/a.qmd", "abc", None, "g", "polish", "p1", "off")
    assert made[1].draw_id == draws.draw_id(*cell, 0, chunk_index=1)
    assert draws.Draw.from_json(made[1].to_json()) == made[1]


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


# --- views as a path (W4 step 4) ---------------------------------------------


@pytest.fixture
def draws_dir(pairs_dir, tmp_path) -> Path:
    """`pairs_dir` lifted into a `draws/` beside a `views/` root."""
    out = tmp_path / "draws"
    for draw in draws.draws_from_pairs(pairs_dir):
        draws.append(out, draw)
    return out


def test_ensure_view_builds_once_and_reuses_until_draws_or_policy_move(draws_dir, pairs_dir):
    data_dir = draws_dir.parent
    policy = draws.ViewPolicy()
    sha = draws.policy_sha(policy)
    assert config.draws_path(data_dir) == draws_dir
    assert config.view_path(data_dir, sha) == data_dir / "views" / f"pairs-{sha}"

    view = draws.ensure_view(draws_dir, policy, views_root=data_dir / "views")
    assert view == config.view_path(data_dir, sha)
    assert corpus.content_sha256(view) == corpus.content_sha256(pairs_dir)
    side = json.loads((view / draws.VIEW_SIDECAR).read_text(encoding="utf-8"))
    assert side["policy_sha"] == sha and side["records"] == 8 and side["built_at"]
    assert side["draws_sha256"] == draws.content_sha256(draws_dir)
    assert draws.content_sha256(draws_dir) == corpus.content_sha256(draws_dir)

    marker = view / "reused"
    marker.touch()
    assert draws.ensure_view(draws_dir, policy, views_root=data_dir / "views") == view
    assert marker.exists(), "an up-to-date view is reused, not rebuilt"

    # The draws grow: same name, rebuilt (the reject changes nothing the view renders).
    first = next(draws.draws_from_pairs(pairs_dir))
    draws.append(draws_dir, dataclasses.replace(first, draw_id="x", outcome="gate:x", record=None))
    assert draws.ensure_view(draws_dir, policy, views_root=data_dir / "views") == view
    assert not marker.exists()
    rebuilt = json.loads((view / draws.VIEW_SIDECAR).read_text(encoding="utf-8"))
    assert rebuilt["draws_sha256"] != side["draws_sha256"]
    assert corpus.content_sha256(view) == corpus.content_sha256(pairs_dir)

    # A sidecar that does not name this policy is stale, whatever the shards hold.
    (view / draws.VIEW_SIDECAR).write_text(json.dumps(rebuilt | {"policy_sha": "stale"}))
    marker.touch()
    draws.ensure_view(draws_dir, policy, views_root=data_dir / "views")
    assert not marker.exists()

    other = draws.ensure_view(
        draws_dir, draws.ViewPolicy(keep_superseded=True), views_root=data_dir / "views"
    )
    assert other != view and other.parent == view.parent


def test_build_view_without_out_reports_the_sha_the_view_would_have(draws_dir, pairs_dir):
    policy = draws.ViewPolicy()
    report = draws.build_view(draws.iter_draws(draws_dir), policy)
    assert report.content_sha256 == corpus.content_sha256(pairs_dir)
    assert report.records == 8 and report.shards == []
    assert not any(draws_dir.parent.glob("views/**/*.jsonl")), "a preview writes nothing"


def test_ensure_view_refuses_a_missing_draws_dir(tmp_path):
    with pytest.raises(ConfigError, match="no draws"):
        draws.ensure_view(tmp_path / "nope", draws.ViewPolicy(), views_root=tmp_path / "views")


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


@pytest.mark.sweep  # conftest deselects it unless SWEEP_ENV is set
def test_the_backfilled_draws_reproduce_the_blog_corpus(tmp_path):
    root = Path(os.environ[SWEEP_ENV]).expanduser()
    src = root / "_training_pairs" / "pairs"
    assert src.is_dir(), f"{SWEEP_ENV}={root} has no _training_pairs/pairs"

    migrations.backfill_draws(src, tmp_path / "draws", apply=True, log=lambda _: None)
    report = draws.build_view(
        draws.iter_draws(tmp_path / "draws"),
        draws.ViewPolicy(keep_superseded=True),
        out=tmp_path / "view",
    )
    print(f"\ncontent_sha256={report.content_sha256} records={report.records}")
    assert report.content_sha256 == corpus.content_sha256(src)
    assert [p.name for p in report.shards] == [p.name for p in corpus.shard_paths(src)]
