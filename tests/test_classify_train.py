"""Tests for the generic classifier trainer (`stylebot.classify_train`).

These exercise the training mechanics — dataset assembly from a pairs corpus,
the leakage-safe POST-split metric, head extraction, holdout partitioning, and
the artifact round-trip through `stylebot.classify` — WITHOUT any model
download: embeddings are hand-made separable matrices. They need the dev-group
`numpy` + `scikit-learn` (sentence-transformers is never touched) and skip
cleanly on a minimal install.

The runtime seam's dependency-freedom is tested separately in
`test_classify.py`; nothing here weakens that guarantee.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from stylebot import classify_train as ct  # noqa: E402
from stylebot.ai_core import STYLE_SYSTEM  # noqa: E402
from stylebot.synth import Target  # noqa: E402


def _write_fake_corpus(path, n_posts: int = 4, chunks: int = 3) -> None:
    """A minimal pairs.jsonl: per (post, chunk) a slop user-turn + an author assistant-turn."""
    with path.open("w", encoding="utf-8") as fp:
        for p in range(n_posts):
            for c in range(chunks):
                rec = {
                    "messages": [
                        {"role": "system", "content": STYLE_SYSTEM},
                        {"role": "user", "content": f"It is worth noting that post {p} chunk {c} delves into the tapestry."},
                        {"role": "assistant", "content": f"Post {p} chunk {c}: the tight version."},
                    ],
                    "meta": {"source": f"post/p{p}.qmd", "chunk_index": c, "capture_id": f"cap{p}"},
                }
                fp.write(json.dumps(rec) + "\n")


def test_assemble_dataset_labels_and_groups(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    _write_fake_corpus(corpus, n_posts=4, chunks=3)
    ds = ct.assemble_dataset(corpus)

    assert ds.n_pairs == 12
    assert ds.n_posts == 4
    # Two rows per pair (slop then author), labelled slop=1 / author=0.
    assert len(ds.texts) == 24
    assert ds.labels.count(ct.LABEL_SLOP) == 12
    assert ds.labels.count(ct.LABEL_DAN) == 12
    # Twins share a POST group and the slop row precedes the author row.
    for s_row, a_row in ds.pair_rows:
        assert ds.labels[s_row] == ct.LABEL_SLOP and ds.labels[a_row] == ct.LABEL_DAN
        assert ds.groups[s_row] == ds.groups[a_row]
    assert sum(ds.is_free) == 0


def test_assemble_dataset_extra_positives_are_injected_and_flagged(tmp_path):
    """Free positives are caller-supplied (Targets or tuples), flagged, metric-excluded."""
    corpus = tmp_path / "pairs.jsonl"
    _write_fake_corpus(corpus, n_posts=2, chunks=1)
    extra = [
        Target(text="A free-standing authored paragraph.", source="post/free1.qmd", chunk_index=0, chunk_total=1),
        ("A plain-tuple paragraph.", "post/free2.qmd"),
        ("   ", "post/blank.qmd"),  # blank -> dropped
    ]
    ds = ct.assemble_dataset(corpus, extra_positives=extra)
    assert ds.n_pairs == 2  # pairs unchanged — extras never become twins
    assert sum(ds.is_free) == 2
    free_rows = [i for i, f in enumerate(ds.is_free) if f]
    assert all(ds.labels[i] == ct.LABEL_DAN for i in free_rows)
    matched_rows = {r for pair in ds.pair_rows for r in pair}
    assert not matched_rows.intersection(free_rows)


def _separable_dataset():
    """A Dataset + a separable embedding matrix: slop near [+1,0], author near [-1,0]."""
    ds = ct.Dataset()
    rng = np.random.default_rng(0)
    X = []
    for p in range(6):
        for c in range(3):
            s_row = ds._add(f"slop {p}.{c}", ct.LABEL_SLOP, f"post/p{p}.qmd")
            a_row = ds._add(f"author {p}.{c}", ct.LABEL_DAN, f"post/p{p}.qmd")
            ds.pair_rows.append((s_row, a_row))
            X.append([1.0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])  # slop
            X.append([-1.0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])  # author
    return ds, np.asarray(X, dtype=np.float32)


def test_evaluate_separates_classes():
    ds, X = _separable_dataset()
    metrics = ct.evaluate(X, ds, n_splits=4, test_size=0.34)
    # On cleanly separable data, pick-the-author's-version accuracy is ~perfect.
    assert metrics["pairwise_accuracy"]["mean"] is not None
    assert metrics["pairwise_accuracy"]["mean"] > 0.95
    assert metrics["auc"]["mean"] > 0.95
    assert metrics["pairwise_accuracy"]["n"] == 4
    assert metrics["mode"] == "cross_val"


def test_fit_head_polarity_and_artifact_roundtrip(tmp_path):
    ds, X = _separable_dataset()
    clf, coef, intercept = ct.fit_head(X, ds)
    # P(slop) for a slop-like vector should exceed that for an author-like vector.
    from stylebot.classify import _logreg_p_slop

    assert _logreg_p_slop([1.0, 0.0], coef, intercept) > 0.5
    assert _logreg_p_slop([-1.0, 0.0], coef, intercept) < 0.5

    out = ct.write_artifact(
        tmp_path / "voice-clf", clf=clf, coef=coef, intercept=intercept, ds=ds,
        metrics={"pairwise_accuracy": {"mean": 1.0}}, save_joblib=True,
    )
    # Contract files exist and meta pins the embedder + polarity + split mode.
    meta = json.loads((out / "meta.json").read_text())
    assert meta["embed_model"] == ct.DEFAULT_EMBED_MODEL
    assert meta["embed_dim"] == 2
    assert meta["label_polarity"] == "p_slop"
    assert meta["split"]["mode"] == "fit_all"
    assert (out / "head.json").exists() and (out / "model.joblib").exists()

    # The artifact serves through the dep-free runtime with a STUB embed_fn.
    from stylebot import classify

    def stub_embed(prose: str):
        return [1.0, 0.0] if prose.startswith("slop") else [-1.0, 0.0]

    det = classify.sklearn_detector(out, embed_fn=stub_embed)
    slop, author = det("slop please"), det("author please")
    assert slop["score"] > 0.5 > author["score"]
    assert slop["p_dan"] == 1.0 - slop["score"]
    assert slop["name"] == "voice-clf"


def test_partition_posts_deterministic_and_explicit():
    sources = [f"post/p{p}.qmd" for p in range(10) for _ in range(2)]
    # Deterministic frac split: same seed -> same holdout; ~30% of 10 posts = 3.
    train1, hold1 = ct.partition_posts(sources, holdout_frac=0.3, holdout_seed=7)
    train2, hold2 = ct.partition_posts(sources, holdout_frac=0.3, holdout_seed=7)
    assert hold1 == hold2 and len(hold1) == 3
    assert train1.isdisjoint(hold1) and train1 | hold1 == set(sources)
    # Explicit list pins the exact partition (intersected with present posts).
    train3, hold3 = ct.partition_posts(sources, holdout_posts=["post/p0.qmd", "post/missing.qmd"])
    assert hold3 == {"post/p0.qmd"}
    # No holdout by default.
    assert ct.partition_posts(sources)[1] == set()


def test_holdout_eval_and_subset_fit_are_leakage_safe():
    ds, X = _separable_dataset()  # 6 posts
    _, holdout = ct.partition_posts(ds.groups, holdout_frac=0.34, holdout_seed=1)
    assert holdout  # non-empty
    metrics = ct.evaluate_holdout(X, ds, holdout)
    assert metrics["mode"] == "holdout"
    assert metrics["pairwise_accuracy"]["mean"] > 0.95  # separable -> easy on unseen posts

    # The shipped head, fit on train posts only, has not seen the holdout rows.
    train_rows = [i for i, g in enumerate(ds.groups) if g not in holdout]
    _, coef, intercept = ct.fit_head(X, ds, rows=train_rows)
    from stylebot.classify import _logreg_p_slop

    assert _logreg_p_slop([1.0, 0.0], coef, intercept) > 0.5 > _logreg_p_slop([-1.0, 0.0], coef, intercept)
