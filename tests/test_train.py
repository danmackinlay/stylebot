"""Tests for the Phase-3 styler trainer (`stylebot.train`).

Assembly, the data-policy hooks, the deterministic by-POST val split, and the
manifest round-trip — all keyless, no network, no tinker. The paid path is
exercised through the `runner` seam with a fake.
"""

from __future__ import annotations

import json

import pytest

from stylebot import splits as sp
from stylebot import train as tr
from stylebot.errors import ConfigError, RunError

# The styler-corpus shape: contract-complete (assemble_training_corpus validates
# by default), two chunks per post, the last two posts synthetic and carrying the
# identity covariate. Passed explicitly so the counts asserted below are visible
# at the call site; `n_posts` is the only knob that varies.
CORPUS = {"chunks": 2, "n_synth_posts": 2, "transform_sim": True}


def test_assemble_keeps_all_without_splits(write_pairs_corpus):
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    corpus = tr.assemble_training_corpus(corpus_path, val_frac=0)
    assert corpus.n_train == 12 and corpus.n_val == 0
    assert corpus.n_source_records == 12
    assert corpus.n_synthetic == 4 and corpus.n_real == 8
    assert corpus.pairs_sha256 and len(corpus.pairs_sha256) == 16


def test_assemble_keeps_near_copy_pairs(write_pairs_corpus):
    """Pinned data policy: transform_sim > 0.85 pairs are styler data, not noise."""
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    corpus = tr.assemble_training_corpus(corpus_path, val_frac=0)
    sims = [(rec.get("meta") or {}).get("transform_sim") for rec in corpus.train]
    assert 0.95 in sims  # the near-copy survived


def test_assemble_filters_to_styler_role(write_pairs_corpus):
    corpus_path = write_pairs_corpus(n_posts=8, **CORPUS)
    posts = [f"post/p{p}.qmd" for p in range(8)]
    splits_doc = sp.make_splits(posts, eval_frac=0.25, detector_frac=0.5, seed=0)
    corpus = tr.assemble_training_corpus(corpus_path, splits=splits_doc, val_frac=0)
    kept_posts = {(rec.get("meta") or {}).get("source") for rec in corpus.train}
    assert kept_posts  # the hash rule leaves at least one styler post at these sizes
    for post in kept_posts:
        assert sp.role_of(post, splits_doc) == "styler"
    assert corpus.dropped["role"] == corpus.n_source_records - corpus.n_train


def test_selector_and_per_passage_hooks(write_pairs_corpus):
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)

    def drop_synth(rec):
        return not (rec.get("meta") or {}).get("synthetic")

    corpus = tr.assemble_training_corpus(corpus_path, selector=drop_synth, val_frac=0)
    assert corpus.n_synthetic == 0 and corpus.dropped["selector"] == 4

    def cap_one(recs):
        return recs[:1]

    # No chunk of the fixture nests in another, so every pair is its own passage
    # and cap-one keeps everything…
    corpus = tr.assemble_training_corpus(corpus_path, per_passage=cap_one, val_frac=0)
    assert corpus.dropped["per_passage"] == 0 and corpus.n_train == 12
    # …and dropping everything via the hook zeroes the corpus.
    corpus = tr.assemble_training_corpus(corpus_path, per_passage=lambda recs: [], val_frac=0)
    assert corpus.n_train == 0 and corpus.dropped["per_passage"] == 12


# -- W9: a passage is a stretch of prose, not a string ----------------------


def _pair(capture_id: str, source: str, target: str, **meta) -> dict:
    from stylebot.ai_core import STYLE_SYSTEM

    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": f"It is worth noting that {target}"},
            {"role": "assistant", "content": target},
        ],
        "meta": {
            "source": source,
            "captured_at": "2026-09-03T00:00:00+00:00",
            "capture_id": capture_id,
            "chunk_index": 0,
            "chunk_total": 1,
            **meta,
        },
    }


def _write(records: list[dict], path):
    from stylebot.jsonl import dumps_line

    path.write_text("".join(dumps_line(rec) + "\n" for rec in records), encoding="utf-8")
    return path


# One post cut at three budgets: paragraph A alone, A+B, and A+B+C. This is the
# live corpus' shape — 3,981 styler-role train pairs on 3,533 distinct target
# bodies but only 982 passages, measured 2026-09-03.
A = "Alpha paragraph, long enough to be a real target chunk on its own."
B = "Beta paragraph, the one that only the wider budgets ever reached."
C = "Gamma paragraph, which only the widest budget packed in with the rest."
NESTED = [
    _pair("cap-a", "post/p.qmd", A, merge_max_chars=None),
    _pair("cap-ab", "post/p.qmd", f"{A}\n\n{B}", merge_max_chars=1500),
    _pair("cap-abc", "post/p.qmd", f"{A}\n\n{B}\n\n{C}", merge_max_chars=3000),
    # Same text, another post: identical strings never merge across sources.
    _pair("cap-other", "post/other.qmd", A, merge_max_chars=None),
]


def test_passage_groups_fold_nested_cuts_of_one_stretch_of_prose():
    groups = tr.passage_groups(NESTED)
    # Three budgets over one stretch = ONE passage; the other post never merges
    # in, however identical its text.
    assert [len(g) for g in groups] == [3, 1]
    assert {(g["meta"]["source"]) for g in groups[0]} == {"post/p.qmd"}
    assert groups[1][0]["meta"]["source"] == "post/other.qmd"


def test_per_passage_cap_counts_a_nested_target_once(tmp_path):
    """The done-criterion for W9: a nested target does not count twice.

    Under the old exact-body grouping these were three unrelated targets and a
    cap-one hook was a no-op, which is why every manifest to date records
    `per_target: null` — using it would not have helped.
    """
    corpus_path = _write(NESTED, tmp_path / "nested.jsonl")
    corpus = tr.assemble_training_corpus(corpus_path, per_passage=lambda recs: recs[:1], val_frac=0)
    assert corpus.n_train == 2  # one per passage, not one per distinct string
    assert corpus.dropped["per_passage"] == 2
    kept = sorted(rec["messages"][2]["content"] for rec in corpus.train)
    assert kept == [A, A]  # corpus order within a passage, so the first cut wins


def test_chunk_scale_reports_the_per_passage_weight_histogram(tmp_path):
    corpus_path = _write(NESTED, tmp_path / "nested.jsonl")
    corpus = tr.assemble_training_corpus(corpus_path, val_frac=0)
    scale = corpus.chunk_scale
    assert scale["passages"]["train"] == {
        "n_passages": 2,
        "pairs_per_passage": {"1": 1, "3": 1},
        "max_pairs_per_passage": 3,
        "mean_pairs_per_passage": 2.0,
    }
    assert scale["passages"]["val"]["n_passages"] == 0
    # Unweighted by default: every pair is one training example.
    assert scale["pair_weights"] == {"1": 4} and scale["n_train_examples"] == 4


def test_pair_weight_replicates_rows_in_the_train_file_only(tmp_path):
    """`meta.weight` was promised from the first draft and stripped one line
    before the trainer could see it (`_write_conversations`). It is integer
    replication now, and val is never replicated."""
    corpus_path = _write(NESTED, tmp_path / "nested.jsonl")

    def triple_the_unmerged(rec):
        return 3 if (rec.get("meta") or {}).get("merge_max_chars") is None else 1

    corpus = tr.assemble_training_corpus(corpus_path, pair_weight=triple_the_unmerged, val_frac=0)
    assert [tr.pair_weight_of(r) for r in corpus.train] == [3, 1, 1, 3]
    assert corpus.n_train == 4 and corpus.n_train_examples == 8
    assert corpus.chunk_scale["pair_weights"] == {"1": 2, "3": 2}

    train_file = tmp_path / "train.jsonl"
    assert tr._write_conversations(corpus.train, train_file) == 8
    assert len(train_file.read_text(encoding="utf-8").splitlines()) == 8
    # The val file takes the same records unreplicated: a weighted held-out loss
    # is an average over a distribution nobody chose.
    val_file = tmp_path / "val.jsonl"
    assert tr._write_conversations(corpus.train, val_file, replicate=False) == 4

    # And the cost estimate follows the replication, because that is what gets
    # paid for.
    assert tr.estimate_tokens(corpus.train) > tr.estimate_tokens(
        [{"messages": r["messages"]} for r in corpus.train]
    )


def test_pair_weight_of_is_defensive_about_junk():
    assert tr.pair_weight_of({"meta": {}}) == 1
    assert tr.pair_weight_of({}) == 1
    assert tr.pair_weight_of({"meta": {"weight": 0}}) == 1
    assert tr.pair_weight_of({"meta": {"weight": -2}}) == 1
    assert tr.pair_weight_of({"meta": {"weight": "nope"}}) == 1
    assert tr.pair_weight_of({"meta": {"weight": 4}}) == 4


def test_val_split_deterministic_and_by_post(write_pairs_corpus):
    corpus_path = write_pairs_corpus(n_posts=10, **CORPUS)
    c1 = tr.assemble_training_corpus(corpus_path, val_frac=0.2, seed=3)
    c2 = tr.assemble_training_corpus(corpus_path, val_frac=0.2, seed=3)
    assert c1.val_posts == c2.val_posts and len(c1.val_posts) == 2
    assert not set(c1.val_posts) & set(c1.train_posts)
    for rec in c1.val:
        assert (rec.get("meta") or {}).get("source") in set(c1.val_posts)
    # A different seed moves the split.
    c3 = tr.assemble_training_corpus(corpus_path, val_frac=0.2, seed=4)
    assert c3.val_posts != c1.val_posts


def test_validation_gate_refuses_malformed_corpus(tmp_path):
    bad = tmp_path / "pairs.jsonl"
    bad.write_text(
        json.dumps({"messages": [{"role": "user", "content": "no system"}], "meta": {}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RunError, match="pairs contract"):
        tr.assemble_training_corpus(bad)


def test_dry_run_reports_the_manifest_and_writes_nothing_under_runs(tmp_path, write_pairs_corpus):
    """The dry-run report, and the rule that it is a report: `runs/` stays untouched."""
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    manifest_out = tmp_path / "runs" / "test.json"
    calls = []

    result = tr.run_training(
        corpus_path,
        tmp_path / "work",
        manifest_out,
        dry_run=True,
        run_id="testrun",
        train_price_per_mtok=1.463,
        runner=lambda *a, **k: calls.append(1),
    )
    assert not calls  # the paid seam is never touched on dry-run
    assert not manifest_out.parent.exists(), "a dry run writes nothing under runs/"
    assert result.manifest_path == manifest_out  # where the paid run would write
    manifest = result.manifest
    assert manifest["dry_run"] is True and manifest["result"] is None
    assert manifest["run_id"] == "testrun"
    assert manifest["data"]["pairs_sha256"] == result.corpus.pairs_sha256
    assert manifest["data"]["n_train"] + manifest["data"]["n_val"] == 12
    assert manifest["data"]["policy_sha"] is None and manifest["data"]["draws_sha256"] is None
    assert manifest["estimate"]["cost_usd"] > 0
    assert manifest["hyperparameters"]["base_model"] == tr.DEFAULT_BASE_MODEL


def test_a_run_from_draws_trains_on_the_view_and_pins_it(tmp_path, write_pairs_corpus):
    """`draws_dir` + `policy` replace `pairs_path`: the manifest names the view (W4 step 4)."""
    from stylebot import config, corpus, draws

    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    draws_dir = tmp_path / "data" / "draws"
    for draw in draws.draws_from_pairs(corpus_path):
        draws.append(draws_dir, draw)
    policy = draws.ViewPolicy(keep_superseded=True)

    result = tr.run_training(
        None,
        tmp_path / "work",
        tmp_path / "runs" / "r.json",
        dry_run=True,
        run_id="v",
        draws_dir=draws_dir,
        policy=policy,
    )
    data = result.manifest["data"]
    view = config.view_path(tmp_path / "data", draws.policy_sha(policy))
    assert data["pairs_path"] == str(view) and view.is_dir()
    assert data["policy_sha"] == draws.policy_sha(policy)
    assert data["draws_sha256"] == draws.content_sha256(draws_dir)
    assert data["pairs_sha256"] == corpus.content_sha256(view) == result.corpus.pairs_sha256
    assert data["n_train"] + data["n_val"] == 12
    with pytest.raises(ConfigError, match="pairs_path or draws_dir"):
        tr.run_training(None, tmp_path / "work", tmp_path / "runs" / "r.json", dry_run=True)


def test_run_training_records_result_via_runner_seam(tmp_path, write_pairs_corpus):
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    manifest_out = tmp_path / "runs" / "run.json"
    seen = {}

    def fake_runner(corpus, manifest, work_dir, hp, **cfg):
        seen["n_train"] = corpus.n_train
        seen["hp"] = hp
        seen["cfg"] = cfg
        return {
            "tinker_run_id": "run-123",
            "checkpoints": {"state": "tinker://run-123/state/final"},
            "train_tokens": 42_000,
            "final_train_loss": 1.5,
            "final_val_loss": 1.7,
            "adapter_dir": str(work_dir / "peft_adapter"),
        }

    result = tr.run_training(
        corpus_path,
        tmp_path / "work",
        manifest_out,
        val_frac=0.2,
        hp=tr.Hyperparameters(seed=1),
        run_id="r1",
        runner=fake_runner,
    )
    assert seen["n_train"] == result.corpus.n_train
    assert seen["hp"].base_model == tr.DEFAULT_BASE_MODEL
    assert seen["hp"].seed == 1  # the seed the epoch shuffle is built from
    # The bundle handed to the runner is the one recorded in the manifest — the
    # relay that used to copy these nine names by hand at five layers.
    assert json.loads(manifest_out.read_text())["hyperparameters"] == seen["hp"].as_manifest()
    manifest = json.loads(manifest_out.read_text())
    assert manifest["result"]["tinker_run_id"] == "run-123"
    assert "adapter_dir" not in manifest["result"]  # local path, not reproducibility
    assert result.adapter_dir == tmp_path / "work" / "peft_adapter"
    assert manifest["dry_run"] is False


def test_manifest_pins_the_partition_the_selection_and_the_stack(tmp_path, write_pairs_corpus):
    """W10 item 5: what actually differed between two runs of "the same" recipe.

    `pairs_sha256` covers the FILE; the selection out of it, the partition that
    held posts back, the caller's policy literals and the installed stack were
    all unrecorded, so two manifests could agree on every field and describe
    different training data.
    """
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    splits_path = tmp_path / "splits.json"
    sp.save_splits(sp.make_splits([f"post/p{i}.qmd" for i in range(6)], seed=0), splits_path)

    def keep_even_chunks(rec):
        return (rec.get("meta") or {}).get("chunk_index", 0) % 2 == 0

    def run(**kw):
        return tr.run_training(
            corpus_path,
            tmp_path / "work",
            tmp_path / "runs" / "r.json",
            dry_run=True,
            splits_path=splits_path,
            **kw,
        ).manifest

    full = run()
    data = full["data"]
    assert data["splits_sha256"] == sp.splits_sha256(splits_path)
    assert data["eval_version"] == 1
    assert len(data["record_ids_sha256"]) == 16
    assert full["dependencies"]["stylebot"]  # the installed stack, by name
    assert "tinker" in full["dependencies"]  # absent -> None, never omitted
    assert full["base_model_revision"] is None

    # Two selections over one corpus file: same pairs_sha256, different records.
    filtered = run(selector=keep_even_chunks, selector_constants={"RETIRED_ARMS": ["m1"]})
    assert filtered["data"]["pairs_sha256"] == data["pairs_sha256"]
    assert filtered["data"]["record_ids_sha256"] != data["record_ids_sha256"]
    assert filtered["data"]["selector_constants"] == {"RETIRED_ARMS": ["m1"]}


def test_blog_root_refuses_a_run_whose_eval_pins_left_the_tree(tmp_path, write_pairs_corpus):
    """D34's missing-pin check, moved in front of the spend.

    `check_splits(blog_root=…)` has always reported it, but only `check-splits`
    calls that and nothing forces `check-splits` before a paid run (F3).
    """
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)
    splits_path = tmp_path / "splits.json"
    sp.save_splits(sp.make_splits([f"post/p{i}.qmd" for i in range(6)], seed=0), splits_path)
    pinned = sp.load_splits(splits_path)["eval_posts"]
    blog_root = tmp_path / "blog"
    (blog_root / "post").mkdir(parents=True)

    def run(**kw):
        return tr.run_training(
            corpus_path,
            tmp_path / "work",
            tmp_path / "runs" / "r.json",
            dry_run=True,
            splits_path=splits_path,
            **kw,
        )

    with pytest.raises(ConfigError, match=pinned[0]):
        run(blog_root=blog_root)
    # Without --blog-root the run is unchanged: the check is opt-in, so the blog
    # can adopt it in its own commit.
    assert run().manifest["data"]["splits_sha256"] == sp.splits_sha256(splits_path)
    # With the pins on disk it still refuses: every corpus source is gated (C12).
    for post in pinned:
        (blog_root / post).write_text("x", encoding="utf-8")
    with pytest.raises(ConfigError, match="corpus source"):
        run(blog_root=blog_root)
    for i in range(6):
        (blog_root / f"post/p{i}.qmd").write_text("x", encoding="utf-8")
    assert run(blog_root=blog_root).manifest["data"]["eval_version"] == 1


def test_train_cli_exits_2_on_a_missing_pin(tmp_path, write_pairs_corpus):
    """The CLI half: a broken contract is a ConfigError, so exit 2, not a warning."""
    from click.testing import CliRunner

    from stylebot import config
    from stylebot.bin import ai_style

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    write_pairs_corpus(path=config.pairs_path(data_dir), n_posts=6, **CORPUS)
    splits_path = tmp_path / "splits.json"
    sp.save_splits(sp.make_splits([f"post/p{i}.qmd" for i in range(6)], seed=0), splits_path)
    blog_root = tmp_path / "blog"
    blog_root.mkdir()

    result = CliRunner().invoke(
        ai_style.main,
        ["train", "--data-dir", str(data_dir), "--dry-run"]
        + ["--splits", str(splits_path), "--blog-root", str(blog_root)]
        + ["--work-dir", str(tmp_path / "w"), "--manifest-out", str(tmp_path / "m.json")],
    )
    assert result.exit_code == 2, result.output
    assert "missing source(s) under" in result.output


def test_train_cli_dry_run_prints_the_manifest_and_leaves_runs_untouched(
    tmp_path, write_pairs_corpus
):
    from click.testing import CliRunner

    from stylebot import config, draws
    from stylebot.bin import ai_style

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    write_pairs_corpus(path=config.pairs_path(data_dir), n_posts=6, **CORPUS)
    manifest_out = tmp_path / "runs" / "m.json"
    common = ["--work-dir", str(tmp_path / "w"), "--manifest-out", str(manifest_out), "--dry-run"]

    result = CliRunner().invoke(ai_style.main, ["train", "--data-dir", str(data_dir)] + common)
    assert result.exit_code == 0, result.output
    assert not manifest_out.parent.exists()
    assert '"dry_run": true' in result.output and '"policy_sha": null' in result.output

    draws_dir = config.draws_path(data_dir)
    for draw in draws.draws_from_pairs(config.pairs_path(data_dir)):
        draws.append(draws_dir, draw)
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(json.dumps({"keep_superseded": True, "retired_arms": ["x"]}))
    args = ["train", "--data-dir", str(data_dir), "--draws", str(draws_dir)]
    result = CliRunner().invoke(ai_style.main, args + ["--policy", str(policy_file)] + common)
    assert result.exit_code == 0, result.output
    sha = draws.policy_sha(draws.ViewPolicy(keep_superseded=True, retired_arms=("x",)))
    assert f'"policy_sha": "{sha}"' in result.output
    assert config.view_path(data_dir, sha).is_dir() and not manifest_out.parent.exists()


def test_run_training_pass_reports_then_asks_before_spending(
    tmp_path, write_pairs_corpus, monkeypatch
):
    """`run_training_pass` is the one `train` body both CLIs call (D78): the
    preview lines, the dry-run exit, the key gate, then `confirm` before the
    paid seam — which a refusal never reaches."""
    from stylebot import config

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    write_pairs_corpus(path=config.pairs_path(data_dir), n_posts=6, **CORPUS)
    opts = tr.TrainOpts(
        data_dir=data_dir, manifest_out=tmp_path / "m.json", work_dir=tmp_path / "w", dry_run=True
    )
    # Two passes below each mint a run id from the clock; pin it so a second
    # boundary between them cannot make the preview and the paid call disagree.
    monkeypatch.setattr(tr, "default_run_id", lambda: "20260101-000000")
    lines: list[str] = []
    preview = tr.run_training_pass(opts, emit=lambda m, level="info": lines.append(m))
    assert preview.manifest["dry_run"] and not (tmp_path / "m.json").exists()
    assert lines[:2] == tr.format_preview(preview.manifest)[:2]
    assert lines[0].startswith(f"{preview.corpus.n_train} train / {preview.corpus.n_val} val")
    assert any(line.startswith("[dry-run]") for line in lines)

    paid = tr.TrainOpts(**{**opts.__dict__, "dry_run": False})
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(RunError, match="TINKER_API_KEY"):
        tr.run_training_pass(paid)
    monkeypatch.setenv("TINKER_API_KEY", "not-a-key")
    monkeypatch.setattr(tr, "_tinker_runner", lambda *a, **k: pytest.fail("the paid seam ran"))
    asked: list[dict] = []

    def decline(manifest: dict) -> bool:
        asked.append(manifest)
        return False

    with pytest.raises(RunError) as declined:
        tr.run_training_pass(paid, confirm=decline)
    assert declined.value.quiet and asked[0]["run_id"] == preview.manifest["run_id"]
    assert not (tmp_path / "m.json").exists()

    with pytest.raises(ConfigError, match="no corpus"):
        tr.run_training_pass(tr.TrainOpts(**{**opts.__dict__, "data_dir": tmp_path / "none"}))
