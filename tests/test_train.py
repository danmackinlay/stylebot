"""Tests for the Phase-3 styler trainer (`stylebot.train`).

Assembly, the data-policy hooks, the deterministic by-POST val split, and the
manifest round-trip — all keyless, no network, no tinker. The paid path is
exercised through the `runner` seam with a fake.
"""

from __future__ import annotations

import inspect
import json

import pytest

from stylebot import provenance
from stylebot import splits as sp
from stylebot import train as tr

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


def test_selector_and_per_target_hooks(write_pairs_corpus):
    corpus_path = write_pairs_corpus(n_posts=6, **CORPUS)

    def drop_synth(rec):
        return not (rec.get("meta") or {}).get("synthetic")

    corpus = tr.assemble_training_corpus(corpus_path, selector=drop_synth, val_frac=0)
    assert corpus.n_synthetic == 0 and corpus.dropped["selector"] == 4

    def cap_one(recs):
        return recs[:1]

    # Every target body is unique in the fixture, so cap-one keeps everything…
    corpus = tr.assemble_training_corpus(corpus_path, per_target=cap_one, val_frac=0)
    assert corpus.dropped["per_target"] == 0 and corpus.n_train == 12
    # …and dropping everything via the hook zeroes the corpus.
    corpus = tr.assemble_training_corpus(corpus_path, per_target=lambda recs: [], val_frac=0)
    assert corpus.n_train == 0 and corpus.dropped["per_target"] == 12


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
    with pytest.raises(ValueError, match="pairs contract"):
        tr.assemble_training_corpus(bad)


def test_dry_run_writes_manifest_without_spending(tmp_path, write_pairs_corpus):
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
    manifest = json.loads(manifest_out.read_text())
    assert manifest["dry_run"] is True and manifest["result"] is None
    assert manifest["run_id"] == "testrun"
    assert manifest["data"]["pairs_sha256"] == result.corpus.pairs_sha256
    assert manifest["data"]["n_train"] + manifest["data"]["n_val"] == 12
    assert manifest["estimate"]["cost_usd"] > 0
    assert manifest["hyperparameters"]["base_model"] == tr.DEFAULT_BASE_MODEL


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


def test_file_sha256_still_reachable_as_train_attribute(tmp_path):
    """`file_sha256` moved to `stylebot.provenance` but stays a `train` name."""
    path = tmp_path / "x.jsonl"
    path.write_text("hello\n", encoding="utf-8")
    digest = tr.file_sha256(path)
    assert len(digest) == 16 and digest == provenance.file_sha256(path)


def test_presplit_builder_seed_has_no_silent_default():
    """`--seed` must reach the epoch shuffle, not just the val split.

    The builder body needs the trainer extra, so this pins the seam instead:
    `shuffle_seed` is a REQUIRED keyword. It was hardcoded to 0 at the
    construction site until 2026-08-09, which made two runs differing only in
    `--seed` shuffle identically while the manifest claimed otherwise.
    """
    param = inspect.signature(tr._make_presplit_builder).parameters["shuffle_seed"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_estimate_tokens_scales_with_text():
    recs = [{"messages": [{"role": "user", "content": "x" * 400}]}]
    assert tr.estimate_tokens(recs) == 100
