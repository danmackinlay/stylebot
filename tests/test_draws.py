"""`stylebot.draws`: a view built from today's pairs reproduces them (D84).

The fast tests run on a fixture corpus laid out by `build_view` itself, so
byte-identity is against stylebot's own bytes and not against a copy this
module made. The `sweep` test does the same over the blog corpus and
prints the `content_sha256` `dan-style status` reports.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import conftest
import pytest

from stylebot import ai_core, config, corpus, draws, pairs
from stylebot.errors import ConfigError

SWEEP_ENV = conftest.SWEEP_ENV


@pytest.fixture
def pairs_dir(tmp_path, write_pairs_corpus) -> Path:
    """A sharded `pairs/` of 8 records — two real posts, two synthetic."""
    single = write_pairs_corpus(tmp_path / "flat.jsonl", n_posts=4, chunks=2, n_synth_posts=2)
    out = tmp_path / "pairs"
    draws.build_view(draws.draws_from_pairs(single), draws.ViewPolicy(), out=out)
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


def test_a_policy_supersedes_a_capture_unless_the_policy_keeps_it(pairs_dir, tmp_path):
    """W4 step 5: the supersession map is authoritative at view time (D84)."""
    lifted = list(draws.draws_from_pairs(pairs_dir))
    policy = draws.ViewPolicy(superseded_captures=("cap0000",))
    assert [d.superseded_by for d in lifted] == [None] * 8, "nothing says so itself"

    report = draws.build_view(lifted, policy, out=tmp_path / "v1")
    assert report.records == 6 and report.excluded == {"superseded": 2}
    assert _ids(tmp_path / "v1") == {d.draw_id for d in lifted if d.draw_id[:7] != "cap0000"}
    seen = [(r.get("meta") or {}).get("capture_id") for r in draws.view_records(lifted, policy)]
    assert len(seen) == 6 and "cap0000" not in seen

    kept = draws.ViewPolicy(superseded_captures=("cap0000",), keep_superseded=True)
    report = draws.build_view(lifted, kept, out=tmp_path / "v2")
    assert report.records == 8 and report.excluded == {}
    assert corpus.content_sha256(tmp_path / "v2") == corpus.content_sha256(pairs_dir)


def test_superseded_captures_cannot_catch_a_synthetic_draw(pairs_dir, tmp_path):
    lifted = list(draws.draws_from_pairs(pairs_dir))
    synthetic = [d for d in lifted if (d.record["meta"] or {}).get("synthetic")]
    assert len(synthetic) == 4 and all(draws.capture_id_of(d) is None for d in synthetic)

    # A policy is user-supplied JSON: the synthetic records' own `capture_id`s,
    # an id no draw carries and an empty one all leave every draw in.
    policy = draws.ViewPolicy(superseded_captures=("cap0002", "cap0003", "nosuch", ""))
    report = draws.build_view(lifted, policy, out=tmp_path / "view")
    assert report.records == 8 and report.excluded == {}
    assert corpus.content_sha256(tmp_path / "view") == corpus.content_sha256(pairs_dir)

    # `capture_id_of` is None for a synthetic draw, so a null entry matches none
    # of them either (`policy_sha` sorts, so this policy only reaches the clause).
    null = draws.ViewPolicy(superseded_captures=(None,))
    assert len(list(draws.view_records(lifted, null))) == 8


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


def test_a_policy_with_none_for_a_collection_reads_it_as_empty(pairs_dir, tmp_path):
    """`None` where the blog passes a tuple hashes, and applies as empty (S5a)."""
    lifted = list(draws.draws_from_pairs(pairs_dir))
    policy = draws.ViewPolicy(retired_arms=None, excluded_draw_ids=None, superseded_captures=None)
    assert draws.policy_sha(policy)
    report = draws.build_view(lifted, policy, out=tmp_path / "view")
    assert report.records == 8 and report.excluded == {}


def _sources(view: Path) -> set[tuple[str, bool]]:
    recs = corpus.iter_records(view)
    return {(r["meta"]["source"], bool(r["meta"].get("synthetic"))) for r in recs}


def test_a_synthetic_draw_on_an_eval_source_never_reaches_the_view(pairs_dir, tmp_path):
    """W4 step 5 item 6: the retired eval-role prune, as a clause (D84)."""
    lifted = list(draws.draws_from_pairs(pairs_dir))  # p0-p1 real, p2-p3 synthetic
    policy = draws.ViewPolicy(eval_sources=("post/p0.qmd", "post/p3.qmd"))

    report = draws.build_view(lifted, policy, out=tmp_path / "view")
    assert report.records == 6 and report.excluded == {"eval_sources": 2}
    kept = _sources(tmp_path / "view")
    assert ("post/p0.qmd", False) in kept, "the eval stratum is real draws on eval posts"
    assert ("post/p2.qmd", True) in kept, "a styler-role post keeps its synthetic draws"
    assert ("post/p3.qmd", True) not in kept


def test_the_eval_clause_reads_the_capture_id_and_not_the_gen_block(pairs_dir, tmp_path):
    """A lifted synthetic pair carries no `meta.gen`, so `gen` is not the test."""
    lifted = list(draws.draws_from_pairs(pairs_dir))
    synth = [d for d in lifted if d.source == "post/p3.qmd"]
    assert len(synth) == 2 and all(d.gen is None for d in synth)
    assert all(draws.capture_id_of(d) is None for d in synth)

    report = draws.build_view(lifted, draws.ViewPolicy(eval_sources=("post/p3.qmd",)))
    assert report.excluded == {"eval_sources": 2}
    assert draws.build_view(lifted, draws.ViewPolicy()).excluded == {}


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
    draws.build_view(map(draws.draw_from_pair, ordered), draws.ViewPolicy(), out=out)
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


def test_two_saves_of_one_chunk_are_two_draws(tmp_path, write_pairs_corpus):
    """W4 step 5: the retired backfill keyed a real draw on (source, sha, span),
    so an interim save and its successor collided; `pairs.record_id` cannot."""
    flat = write_pairs_corpus(tmp_path / "flat.jsonl", n_posts=1, chunks=1)
    first = corpus.read_records(flat)[0]
    second = json.loads(json.dumps(first))
    second["meta"]["capture_id"] = "cap9999"

    a, b = draws.draw_from_pair(first), draws.draw_from_pair(second)
    assert a.source == b.source
    assert a.record["meta"]["chunk_index"] == b.record["meta"]["chunk_index"] == 0
    assert {a.draw_id, b.draw_id} == {"cap0000:0", "cap9999:0"}
    assert draws.build_view([a, b], draws.ViewPolicy()).records == 2


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
    a = draws.ViewPolicy(
        retired_arms=("x", "y"),
        system_prompt="p",
        excluded_draw_ids=("1", "2"),
        superseded_captures=("c1", "c2"),
        eval_sources=("notebook/a.qmd", "post/b.qmd"),
    )
    b = draws.ViewPolicy(
        excluded_draw_ids=("2", "1"),
        system_prompt="p",
        superseded_captures=("c2", "c1"),
        retired_arms=("y", "x"),
        eval_sources=("post/b.qmd", "notebook/a.qmd"),
    )
    assert draws.policy_sha(a) == draws.policy_sha(b)

    identity = draws.policy_sha(draws.ViewPolicy())
    changed = {
        "chunk_policy_digest": "d",
        "gate_thresholds": {"max_transform_sim": 0.95},
        "automation_cap": 1,
        "retired_arms": ("qwen/qwen3-8b",),
        "anchor_rule": "drop-invented",
        "role_rule": "splits:abcd",
        "system_prompt": "you rewrite prose",
        "keep_superseded": True,
        "excluded_draw_ids": ("cap0000:0",),
        "superseded_captures": ("cap0000",),
        "eval_sources": ("notebook/adhd.qmd",),
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


# --- the policy file and the view bundle ---------------------------------------


def test_view_policy_dumps_and_loads_as_the_same_object(tmp_path):
    policy = draws.ViewPolicy(
        retired_arms=("b", "a"), gate_thresholds={"k": 1}, eval_sources=("post/x.qmd",)
    )
    doc = json.loads(json.dumps(policy.dump()))
    assert doc["retired_arms"] == ["b", "a"] and doc["keep_superseded"] is False
    assert draws.ViewPolicy.load(doc) == policy
    assert draws.policy_sha(draws.ViewPolicy.load(doc)) == draws.policy_sha(policy)
    with pytest.raises(ConfigError, match="system_sha"):
        draws.ViewPolicy.load({"system_sha": "deadbeef"})
    with pytest.raises(ConfigError, match="not a ViewPolicy"):
        draws.ViewPolicy.load(["not", "an", "object"])
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.dump()), encoding="utf-8")
    assert draws.load_policy(path) == policy


def test_view_opts_names_one_corpus_and_builds_where_it_is_told(draws_dir, tmp_path):
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({"keep_superseded": True}), encoding="utf-8")
    sha = draws.policy_sha(draws.ViewPolicy(keep_superseded=True))

    opts = draws.ViewOpts(draws_dir=draws_dir, policy_path=policy_file)
    assert opts.view_policy() == draws.ViewPolicy(keep_superseded=True)
    assert opts.corpus_or_view(None) == draws_dir.parent / "views" / f"pairs-{sha}"
    elsewhere = draws.ViewOpts(draws_dir=draws_dir, views_root=tmp_path / "elsewhere")
    assert elsewhere.view() == tmp_path / "elsewhere" / config.view_name(
        draws.policy_sha(draws.ViewPolicy())
    )

    # A policy object stands in for --policy; naming both is incoherent.
    in_code = draws.ViewOpts(draws_dir=draws_dir, policy=draws.ViewPolicy(keep_superseded=True))
    assert in_code.view() == opts.view()
    with pytest.raises(ConfigError, match="not both"):
        dataclasses.replace(in_code, policy_path=policy_file).view_policy()

    # --pairs is a scratch file: it passes through, never beside --draws.
    scratch = tmp_path / "scratch.jsonl"
    assert draws.ViewOpts().corpus_or_view(scratch) == scratch
    assert draws.ViewOpts().corpus_or_view(None) is None
    with pytest.raises(ConfigError, match="not both"):
        opts.corpus_or_view(scratch)
    with pytest.raises(ConfigError, match="--draws"):
        draws.ViewOpts().corpus_or_view(None, required=True)
    with pytest.raises(ConfigError, match="--draws"):
        draws.ViewOpts().view()


def test_view_options_repins_a_default_and_drops_a_flag():
    import click

    @click.command()
    @draws.view_options(
        defaults=[draws.ViewOpts(draws_dir=Path("elsewhere"))], exclude=["policy_path"]
    )
    def cmd(**kw):
        pass

    params = {p.name: p for p in cmd.params}
    assert params["draws_dir"].default == Path("elsewhere")
    assert "policy_path" not in params and "views_root" in params
    assert all(
        getattr(params[n], "help_section", None) == "View" for n in ("draws_dir", "views_root")
    )


def test_ai_style_build_view_prints_the_view_it_built_or_reused(draws_dir, tmp_path):
    from click.testing import CliRunner

    from stylebot.bin.ai_style import main as ai_style

    args = ["build-view", "--draws", str(draws_dir), "--views-root", str(tmp_path / "v")]
    result = CliRunner().invoke(ai_style, args)
    assert result.exit_code == 0, result.output
    view = Path(result.output.strip().splitlines()[-1])
    assert view.parent == tmp_path / "v" and (view / draws.VIEW_SIDECAR).exists()
    assert "8 record(s)" in result.output
    assert CliRunner().invoke(ai_style, args).output.strip().splitlines()[-1] == str(view)
    bare = CliRunner().invoke(ai_style, ["build-view"])
    assert bare.exit_code == 2 and "--draws" in bare.output, bare.output


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


# --- the view carries the system prompt (W4 step 5) ---------------------------


def test_the_view_stamps_the_policys_system_prompt(pairs_dir, tmp_path):
    prompt = ai_core.style_system("Ada Lovelace")
    policy = draws.ViewPolicy(system_prompt=prompt)
    lifted = list(draws.draws_from_pairs(pairs_dir))
    view = tmp_path / "view"
    draws.build_view(lifted, policy, out=view)

    stamped = corpus.read_records(view)
    assert {r["messages"][0]["content"] for r in stamped} == {prompt}
    assert pairs.validate_pairs_file(view, system_prompt=prompt) == []
    # Nothing edits a draw: its stored record keeps the turn it was drawn under.
    assert {d.record["messages"][0]["content"] for d in lifted} == {ai_core.STYLE_SYSTEM}
    assert [r["meta"] for r in stamped] == [d.record["meta"] for d in lifted]
    assert [r["messages"][1:] for r in stamped] == [d.record["messages"][1:] for d in lifted]


def test_stamping_the_prompt_the_draws_already_carry_moves_no_bytes(pairs_dir, tmp_path):
    policy = draws.ViewPolicy(system_prompt=ai_core.STYLE_SYSTEM)
    report = draws.build_view(draws.draws_from_pairs(pairs_dir), policy, out=tmp_path / "view")
    assert report.content_sha256 == corpus.content_sha256(pairs_dir)
    assert corpus.read_records(tmp_path / "view") == corpus.read_records(pairs_dir)
    unstamped = list(draws.view_records(draws.draws_from_pairs(pairs_dir), draws.ViewPolicy()))
    assert list(draws.view_records(draws.draws_from_pairs(pairs_dir), policy)) == unstamped


def test_policy_system_sha_is_derived_from_the_prompt():
    assert draws.ViewPolicy().system_sha is None
    policy = draws.ViewPolicy(system_prompt=ai_core.STYLE_SYSTEM)
    assert policy.system_sha == pairs.system_sha() == pairs.system_sha(ai_core.STYLE_SYSTEM)
    other = draws.ViewPolicy(system_prompt=ai_core.style_system("Ada Lovelace"))
    assert other.system_sha == pairs.system_sha(other.system_prompt) != policy.system_sha


def test_load_policy_takes_a_prompt_and_refuses_a_supplied_sha(tmp_path):
    doc = tmp_path / "policy.json"
    doc.write_text(json.dumps({"system_prompt": "you rewrite prose"}), encoding="utf-8")
    assert draws.load_policy(doc).system_prompt == "you rewrite prose"
    doc.write_text(json.dumps({"system_sha": "deadbeefdeadbeef"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        draws.load_policy(doc)


def test_load_policy_round_trips_the_eval_sources(tmp_path):
    doc = tmp_path / "policy.json"
    sources = ["notebook/adhd.qmd", "post/should_we_allow_ai_articles.qmd"]
    doc.write_text(json.dumps({"eval_sources": sources}), encoding="utf-8")
    assert draws.load_policy(doc).eval_sources == tuple(sources)
    assert draws.load_policy(None).eval_sources is None


def test_training_view_pins_the_prompt_the_view_was_stamped_with(draws_dir):
    prompt = ai_core.style_system("Ada Lovelace")
    _view, pins = draws.training_view(draws_dir, draws.ViewPolicy(system_prompt=prompt))
    assert pins["style_system_sha"] == pairs.system_sha(prompt)
    _view, pins = draws.training_view(draws_dir, draws.ViewPolicy())
    assert pins["style_system_sha"] == pairs.system_sha()
