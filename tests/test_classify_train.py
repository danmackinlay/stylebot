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


def _write_fake_corpus(path, n_posts: int = 4, chunks: int = 3, n_synth_posts: int = 0) -> None:
    """A minimal pairs.jsonl: per (post, chunk) a slop user-turn + an author assistant-turn.

    The last `n_synth_posts` posts are marked `meta.synthetic` (the `ai-style
    synth` stratum); the rest look like real edit captures (no key).
    """
    with path.open("w", encoding="utf-8") as fp:
        for p in range(n_posts):
            for c in range(chunks):
                meta = {"source": f"post/p{p}.qmd", "chunk_index": c, "capture_id": f"cap{p}"}
                if p >= n_posts - n_synth_posts:
                    meta["synthetic"] = True
                rec = {
                    "messages": [
                        {"role": "system", "content": STYLE_SYSTEM},
                        {"role": "user", "content": f"It is worth noting that post {p} chunk {c} delves into the tapestry."},
                        {"role": "assistant", "content": f"Post {p} chunk {c}: the tight version."},
                    ],
                    "meta": meta,
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
    # No meta.synthetic anywhere -> all-real.
    assert ds.n_pairs_real == 12 and ds.n_pairs_synthetic == 0


def test_assemble_dataset_records_synthetic_provenance(tmp_path):
    """`meta.synthetic` is carried into pair_synth so metrics can facet by it."""
    corpus = tmp_path / "pairs.jsonl"
    _write_fake_corpus(corpus, n_posts=4, chunks=2, n_synth_posts=1)
    ds = ct.assemble_dataset(corpus)
    assert ds.n_pairs == 8
    assert ds.n_pairs_synthetic == 2 and ds.n_pairs_real == 6
    # The synthetic pairs are exactly those from the marked post.
    for k, (s_row, _) in enumerate(ds.pair_rows):
        assert ds.pair_is_synthetic(k) == (ds.groups[s_row] == "post/p3.qmd")


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
            ds._add_pair(f"slop {p}.{c}", f"author {p}.{c}", f"post/p{p}.qmd")
            X.append([1.0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])  # slop
            X.append([-1.0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])  # author
    return ds, np.asarray(X, dtype=np.float32)


def _mixed_dataset():
    """Real pairs separable one way, synthetic pairs the OPPOSITE way.

    The real stratum dominates (10 posts vs 2), so any train subset fits the real
    orientation — real facet ~1.0, synthetic facet ~0.0. Only a correctly wired
    facet (right pairs in the right stratum) reproduces that split.
    """
    ds = ct.Dataset()
    rng = np.random.default_rng(1)
    X = []
    for p in range(12):
        synth = p >= 10
        sign = -1.0 if synth else 1.0
        for c in range(3):
            ds._add_pair(f"slop {p}.{c}", f"author {p}.{c}", f"post/p{p}.qmd", synthetic=synth)
            X.append([sign * 1.0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])  # slop
            X.append([sign * -1.0 + rng.normal(0, 0.05), rng.normal(0, 0.05)])  # author
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
    # Single-stratum corpus -> no provenance facet (it would repeat the headline).
    assert "by_provenance" not in metrics
    # Default C=None -> nested CV: a per-fold C chosen from the grid, recorded.
    assert metrics["C"]["mode"] == "nested_cv"
    assert len(metrics["C"]["selected"]) == 4
    assert all(c in ct.C_GRID for c in metrics["C"]["selected"])
    # An explicit C is an override, recorded as fixed.
    fixed = ct.evaluate(X, ds, n_splits=2, test_size=0.34, C=1.0)
    assert fixed["C"] == {"mode": "fixed", "value": 1.0}


def test_select_C_nested_and_fallback():
    ds, X = _separable_dataset()  # 6 posts, separable at any sane C
    C, info = ct.select_C(X, ds, list(range(len(ds.texts))))
    assert C in ct.C_GRID
    assert info["mode"] == "nested_cv"
    assert max(v for v in info["scores"].values() if v is not None) > 0.9

    # Too few posts for an inner split -> DEFAULT_C fallback, flagged as such.
    tiny = ct.Dataset()
    Xt = []
    for p in range(2):
        for c in range(2):
            tiny._add_pair(f"slop {p}.{c}", f"author {p}.{c}", f"post/p{p}.qmd")
            Xt += [[1.0, 0.0], [-1.0, 0.0]]
    C2, info2 = ct.select_C(np.asarray(Xt, dtype=np.float32), tiny, list(range(len(tiny.texts))))
    assert C2 == ct.DEFAULT_C
    assert info2["mode"] == "fallback"


def test_subset_dataset_remaps_pairs_and_aligns_rows():
    ds, X = _mixed_dataset()
    keep = {"post/p0.qmd", "post/p10.qmd"}  # one real + one synthetic post
    sub, rows = ct.subset_dataset(ds, keep)
    assert set(sub.groups) == keep
    assert sub.n_pairs == 6 and sub.n_pairs_synthetic == 3 and sub.n_pairs_real == 3
    # rows are the kept original indices in order -> X[rows] aligns with sub.
    assert [ds.texts[i] for i in rows] == sub.texts
    for k, (s_row, a_row) in enumerate(sub.pair_rows):
        assert sub.labels[s_row] == ct.LABEL_SLOP and sub.labels[a_row] == ct.LABEL_DAN
        assert sub.groups[s_row] == sub.groups[a_row]
        assert sub.texts[s_row].startswith("slop") and sub.texts[a_row].startswith("author")


def test_evaluate_facets_by_provenance():
    ds, X = _mixed_dataset()
    metrics = ct.evaluate(X, ds, n_splits=6, test_size=0.25)
    by = metrics["by_provenance"]
    assert by["real"]["n_pairs"] == 30 and by["synthetic"]["n_pairs"] == 6
    # The head follows the (majority) real orientation, so the facets diverge
    # sharply — real near-perfect, synthetic (anti-oriented) near-zero.
    assert by["real"]["pairwise_accuracy"]["mean"] > 0.9
    assert by["synthetic"]["pairwise_accuracy"]["mean"] < 0.1
    assert by["real"]["auc"]["mean"] > 0.9
    assert by["synthetic"]["auc"]["mean"] < 0.1
    # The blended headline sits between the facets: the facet is what keeps a
    # synth-heavy corpus from flattering itself.
    overall = metrics["pairwise_accuracy"]["mean"]
    assert by["synthetic"]["pairwise_accuracy"]["mean"] < overall < by["real"]["pairwise_accuracy"]["mean"]


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
    assert meta["n_pairs_real"] == 18 and meta["n_pairs_synthetic"] == 0
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
    assert "by_provenance" not in metrics  # all-real corpus

    # The shipped head, fit on train posts only, has not seen the holdout rows.
    train_rows = [i for i, g in enumerate(ds.groups) if g not in holdout]
    _, coef, intercept = ct.fit_head(X, ds, rows=train_rows)
    from stylebot.classify import _logreg_p_slop

    assert _logreg_p_slop([1.0, 0.0], coef, intercept) > 0.5 > _logreg_p_slop([-1.0, 0.0], coef, intercept)


def test_train_with_splits_contract(tmp_path, monkeypatch):
    """End-to-end splits mode: eval posts never embedded, styler holdout reported,
    the contract + danger report recorded in meta, artifact serves via the runtime."""
    import hashlib

    from stylebot import splits as sp

    corpus = tmp_path / "pairs.jsonl"
    _write_fake_corpus(corpus, n_posts=12, chunks=2, n_synth_posts=4)
    ds_all = ct.assemble_dataset(corpus)
    posts = sorted(set(ds_all.groups))
    real_posts = sorted({
        ds_all.groups[s] for k, (s, _) in enumerate(ds_all.pair_rows) if not ds_all.pair_is_synthetic(k)
    })
    splits_doc = sp.make_splits(posts, eval_frac=0.25, detector_frac=0.5, seed=0, eval_candidates=real_posts)
    splits_path = sp.save_splits(splits_doc, tmp_path / "splits.json")
    roles = {p: sp.role_of(p, splits_doc) for p in posts}
    assert set(roles.values()) == {"eval", "detector", "styler"}  # precondition

    embedded: list[str] = []

    def stub_embed(texts, **kw):
        embedded.extend(texts)
        out = []
        for t in texts:
            jitter = (int(hashlib.sha256(t.encode()).hexdigest()[:4], 16) / 65535 - 0.5) * 0.1
            base = 1.0 if t.startswith("It is worth noting") else -1.0  # the slop side
            out.append([base + jitter, jitter])
        return np.asarray(out, dtype=np.float32)

    monkeypatch.setattr(ct, "embed_texts", stub_embed)
    out, metrics = ct.train(corpus, tmp_path / "clf", splits_path=splits_path)

    # The frozen eval stratum was never even encoded.
    post_of_text = {t: ds_all.groups[i] for i, t in enumerate(ds_all.texts)}
    eval_posts = {p for p, r in roles.items() if r == "eval"}
    assert embedded and all(post_of_text[t] not in eval_posts for t in embedded)

    # Styler posts give the bonus unseen-posts metric.
    assert metrics["styler_holdout"]["mode"] == "holdout"
    assert metrics["styler_holdout"]["pairwise_accuracy"]["mean"] is not None

    meta = json.loads((out / "meta.json").read_text())
    assert meta["split"]["mode"] == "splits"
    assert meta["split"]["rest_rule"] == splits_doc["rest_rule"]
    assert meta["split"]["roles"]["eval"]["pairs"] > 0  # counted, just untrained-on
    assert meta["head_C"]["value"] in set(ct.C_GRID) | {ct.DEFAULT_C}
    # Tiny fixture strata -> the danger report fires and travels with the artifact.
    assert any("DANGEROUSLY SMALL" in w for w in meta["split"]["warnings"])
    # n_pairs describes the fit data (detector pool only).
    assert meta["n_pairs"] == meta["split"]["roles"]["detector"]["pairs"]

    # Round-trip: the artifact serves through the dep-free runtime.
    from stylebot import classify

    def one_embed(prose: str):
        jitter = (int(hashlib.sha256(prose.encode()).hexdigest()[:4], 16) / 65535 - 0.5) * 0.1
        base = 1.0 if prose.startswith("It is worth noting") else -1.0
        return [base + jitter, jitter]

    det = classify.sklearn_detector(out, embed_fn=one_embed)
    assert det("It is worth noting that this delves.")["score"] > 0.5 > det("The tight version.")["score"]


def test_train_splits_mutually_exclusive_with_holdout(tmp_path):
    corpus = tmp_path / "pairs.jsonl"
    _write_fake_corpus(corpus, n_posts=3, chunks=1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        ct.train(corpus, tmp_path / "clf", splits_path=tmp_path / "splits.json", holdout_frac=0.25)


def test_holdout_eval_facets_by_provenance():
    ds, X = _mixed_dataset()  # posts p0..p9 real, p10..p11 synthetic (anti-oriented)
    holdout = {"post/p0.qmd", "post/p1.qmd", "post/p10.qmd"}  # both strata held out
    metrics = ct.evaluate_holdout(X, ds, holdout)
    by = metrics["by_provenance"]
    # Stratum counts are corpus-wide; the per-facet metric n counts holdout twins.
    assert by["real"]["n_pairs"] == 30 and by["synthetic"]["n_pairs"] == 6
    assert by["real"]["pairwise_accuracy"]["n"] == 6  # 2 held-out real posts x 3 chunks
    assert by["synthetic"]["pairwise_accuracy"]["n"] == 3
    # Head fit on the (real-dominated) train posts -> facets diverge on the holdout.
    assert by["real"]["pairwise_accuracy"]["mean"] > 0.9
    assert by["synthetic"]["pairwise_accuracy"]["mean"] < 0.1


def test_assemble_drops_near_identity_pairs(tmp_path):
    # Near-identical slop/author sides with opposite labels are label noise;
    # the transform_sim gate drops them (None-covariate pairs are kept).
    import json

    from stylebot.classify_train import assemble_dataset

    def rec(slop, author, sim=None):
        meta = {"source": "p.qmd"}
        if sim is not None:
            meta["transform_sim"] = sim
        return {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": slop},
                {"role": "assistant", "content": author},
            ],
            "meta": meta,
        }

    path = tmp_path / "pairs.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                rec("proper slop text", "proper author text", sim=0.3),
                rec("nearly identical text", "nearly identical text!", sim=0.97),  # dropped
                rec("legacy pair no covariate", "legacy author no covariate"),  # kept
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ds = assemble_dataset(path)
    assert ds.n_pairs == 2
    ds_all = assemble_dataset(path, max_transform_sim=None)  # gate off
    assert ds_all.n_pairs == 3
