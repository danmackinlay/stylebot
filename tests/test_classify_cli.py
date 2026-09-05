"""`classify_cli.train_clf` — the `train-clf` pass as a library (D78), keyless.

Nothing here embeds: the dry-run path stops before the backbone, and the fit
path stubs `classify_train.train` to check what the pass hands it and how it
renders what comes back.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylebot import classify_cli as cc
from stylebot import classify_train as ct
from stylebot import splits as sp
from stylebot.errors import ConfigError

POSTS = [f"post/p{p}.qmd" for p in range(4)]


def _recording_emitter():
    lines: list[tuple[str, str]] = []

    def emit(message: str, /, *, level: str = "info") -> None:
        lines.append((level, message))

    return emit, lines


def _splits_file(tmp_path: Path) -> Path:
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(sp.make_splits(POSTS, seed=1)), encoding="utf-8")
    return path


def _opts(pairs: Path, tmp_path: Path, **kw) -> cc.TrainClfOpts:
    return cc.TrainClfOpts(pairs_path=pairs, out_dir=tmp_path / "clf", **kw)


def test_train_clf_opts_defaults_are_the_trainer_constants():
    opts = cc.TrainClfOpts()
    assert (opts.test_size, opts.n_splits) == (ct.DEFAULT_TEST_SIZE, ct.DEFAULT_N_SPLITS)
    assert opts.holdout_frac == ct.DEFAULT_HOLDOUT_FRAC
    assert opts.holdout_seed == ct.DEFAULT_HOLDOUT_SEED
    assert opts.embed_model == ct.DEFAULT_EMBED_MODEL
    assert {"opts": cc.TrainClfOpts} == cc.TRAIN_CLF_PARAMS


def test_train_clf_dry_run_reports_assembly_and_the_splits_table(tmp_path, write_pairs_corpus):
    """The count line, one row per role, `check_splits` at `warn`, no artifact."""
    pairs = write_pairs_corpus(n_posts=4, chunks=2)
    emit, lines = _recording_emitter()
    opts = _opts(pairs, tmp_path, splits_path=_splits_file(tmp_path), dry_run=True)

    assert cc.train_clf(opts, emit=emit) is None

    messages = [m for _, m in lines]
    assert messages[0] == (
        "8 content-matched pair(s) from 4 post(s)  [8 slop / 8 author; 0 free positives]"
    )
    for role in sp.ROLES:
        assert any(m.startswith(f"  {role:>8}: ") and "post(s)" in m for m in messages), role
    assert any(level == "warn" and m.startswith("WARNING: ") for level, m in lines)
    assert messages[-1] == "[dry-run] no embedding, no artifact written"
    assert not opts.out_dir.exists()


def test_train_clf_dry_run_runs_the_pinned_post_gate(tmp_path, write_pairs_corpus):
    """`blog_root` without the eval posts on disk fails before the dry-run exit (D34)."""
    pairs = write_pairs_corpus(n_posts=4, chunks=1)
    blog_root = tmp_path / "blog"
    blog_root.mkdir()
    opts = _opts(
        pairs, tmp_path, splits_path=_splits_file(tmp_path), blog_root=blog_root, dry_run=True
    )
    with pytest.raises(ConfigError, match="pinned eval post"):
        cc.train_clf(opts)


def test_train_clf_refuses_an_empty_corpus(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="no content-matched pairs") as info:
        cc.train_clf(_opts(pairs, tmp_path, dry_run=True))
    assert info.value.param_hint == "--pairs"


def test_train_clf_refuses_splits_with_a_holdout(tmp_path, write_pairs_corpus):
    pairs = write_pairs_corpus(n_posts=2, chunks=1)
    splits = _splits_file(tmp_path)
    listing = tmp_path / "hold.txt"
    listing.write_text("post/p0.qmd\n", encoding="utf-8")
    for kw in ({"holdout_frac": 0.5}, {"holdout_posts": listing}):
        with pytest.raises(ConfigError, match="mutually exclusive"):
            cc.train_clf(_opts(pairs, tmp_path, splits_path=splits, **kw))


def test_train_clf_counts_what_the_gate_drops(tmp_path, write_pairs_corpus):
    """The caller's `pair_ok` is wrapped in a counting gate and reported by name."""
    pairs = write_pairs_corpus(n_posts=3, chunks=2)

    def not_p0(rec: dict) -> bool:
        return rec["meta"]["source"] != "post/p0.qmd"

    emit, lines = _recording_emitter()
    cc.train_clf(_opts(pairs, tmp_path, dry_run=True), pair_ok=not_p0, emit=emit)
    messages = [m for _, m in lines]
    assert messages[0] == "not_p0: dropped 2 pair(s) from 1 post(s)"
    assert messages[1].startswith("4 content-matched pair(s) from 2 post(s)")


def test_train_clf_fits_through_classify_train_and_reports_the_artifact(
    tmp_path, write_pairs_corpus, monkeypatch
):
    """The fit path: the holdout list is read, every hook reaches `ct.train`,
    the metrics render, and `meta.json`'s own C and warnings are surfaced."""
    pairs = write_pairs_corpus(n_posts=3, chunks=1)
    listing = tmp_path / "hold.txt"
    listing.write_text("post/p2.qmd\n\n", encoding="utf-8")
    metrics = {
        "mode": "holdout",
        "pairwise_accuracy": {"mean": 0.9, "std": 0.0},
        "auc": {"mean": 0.8, "std": 0.0},
        "C": {"mode": "fixed", "value": 3.0},
        "styler_holdout": {
            "n_splits": 8,
            "pairwise_accuracy": {"mean": 0.7, "std": 0.1},
            "auc": {"mean": 0.6, "std": 0.1},
        },
    }
    seen: dict = {}

    def fake_train(pairs_path, out_dir, **kw):
        seen.update(kw, pairs_path=pairs_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True)
        meta = {"head_C": {"value": 3.0, "mode": "fixed"}, "split": {"warnings": ["tiny"]}}
        (out_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return out_dir, metrics

    monkeypatch.setattr(ct, "train", fake_train)

    def pair_ok(rec: dict) -> bool:
        return True

    emit, lines = _recording_emitter()
    opts = _opts(pairs, tmp_path, holdout_posts=listing, C=3.0, blog_root=tmp_path)
    out, got = cc.train_clf(opts, pair_ok=pair_ok, git_repo=tmp_path, emit=emit)

    assert (out, got) == (opts.out_dir, metrics)
    assert seen["holdout_posts"] == ["post/p2.qmd"]
    assert seen["pair_ok"] is pair_ok and seen["blog_root"] == tmp_path
    assert seen["git_repo"] == tmp_path and seen["C"] == 3.0 and seen["splits_path"] is None
    messages = [m for _, m in lines]
    assert "holdout split: list of 1 posts — head ships fit on the train posts only" in messages
    assert f"embedding with {ct.DEFAULT_EMBED_MODEL} (downloads on first use) ..." in messages
    assert "holdout (unseen posts): pairwise_acc=0.9±0.0  auc=0.8±0.0  [C fixed at 3.0]" in messages
    assert "  cross-val (8 POST folds): pairwise_acc=0.7±0.1  auc=0.6±0.1" in messages
    assert "shipped head C: 3.0 (fixed)" in messages
    assert ("warn", "WARNING: tiny") in lines
    assert messages[-1] == f"wrote artifact -> {out}/ (head.json, meta.json)"


def test_format_clf_metrics_renders_nested_c_and_the_provenance_facet():
    metrics = {
        "n_splits": 8,
        "pairwise_accuracy": {"mean": 0.91, "std": 0.02},
        "auc": {"mean": 0.95, "std": 0.01},
        "C": {"mode": "nested_cv", "selected": [1.0, 3.0]},
        "by_provenance": {
            "real": {
                "n_pairs": 10,
                "pairwise_accuracy": {"mean": 0.9, "std": 0.1},
                "auc": {"mean": 0.8, "std": 0.2},
            }
        },
    }
    assert cc.format_clf_metrics(metrics, indent="> ") == [
        "> cross-val (8 POST folds): pairwise_acc=0.91±0.02  auc=0.95±0.01  "
        "[C: nested, per-fold [1.0, 3.0]]",
        ">   real (10 pairs): pairwise_acc=0.9±0.1  auc=0.8±0.2",
    ]
    bare = {"pairwise_accuracy": {"mean": 1, "std": 0}, "auc": {"mean": 1, "std": 0}}
    assert cc.format_clf_metrics(bare) == ["cross-val (None POST folds): pairwise_acc=1±0  auc=1±0"]


def test_ai_style_train_clf_is_the_pass_behind_flags(tmp_path, write_pairs_corpus):
    from click.testing import CliRunner

    from stylebot.bin.ai_style import main

    pairs = write_pairs_corpus(n_posts=2, chunks=1)
    out = str(tmp_path / "clf")
    args = ["train-clf", "--pairs", str(pairs), "--out", out, "--dry-run"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "2 content-matched pair(s) from 2 post(s)" in result.output
    assert "[dry-run] no embedding, no artifact written" in result.output

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    refused = CliRunner().invoke(main, ["train-clf", "--pairs", str(empty), "--out", out])
    assert refused.exit_code == 2 and "--pairs" in refused.output

    help_text = CliRunner().invoke(main, ["train-clf", "--help"]).output
    assert "Voice classifier" in help_text
    assert "--holdout-frac" in help_text and f"[default: {ct.DEFAULT_TEST_SIZE}]" in help_text


def test_blog_root_is_eager_wherever_a_bundle_declares_it(tmp_path, write_pairs_corpus):
    """D79: a corpus default that reads `--blog-root` off the context must find
    it already processed, whatever the flag order — so the three bundles that
    declare `--blog-root` mark it eager, and a default pinned to a callable sees
    the root even when `--blog-root` is the LAST flag on the line."""
    import click
    from click.testing import CliRunner

    from stylebot import train as tr
    from stylebot.cli_options import bundles_from_kwargs

    for options in (cc.train_clf_options, sp.extend_eval_options, tr.train_options):

        @click.command()
        @options()
        def cmd(**kw) -> None:
            pass

        (param,) = [p for p in cmd.params if "--blog-root" in p.opts]
        assert param.is_eager, options.__name__

    root = tmp_path / "blog"
    root.mkdir()
    corpus = write_pairs_corpus(path=root / "pairs.jsonl", n_posts=2, chunks=1)
    seen: dict = {}

    def corpus_under_root() -> Path:
        return click.get_current_context().params["blog_root"] / "pairs.jsonl"

    @click.command()
    @cc.train_clf_options(defaults=[cc.TrainClfOpts(pairs_path=corpus_under_root)])
    def train_clf(**kw) -> None:
        seen.update(bundles_from_kwargs(kw, cc.TRAIN_CLF_PARAMS)["opts"].__dict__)

    args = ["--out", str(tmp_path / "clf"), "--dry-run", "--blog-root", str(root)]
    result = CliRunner().invoke(train_clf, args)
    assert result.exit_code == 0, result.output
    assert seen["pairs_path"] == corpus and seen["blog_root"] == root
