"""Eval integrity — the instrument, the interval, the stratum (W10, 2026-09-03).

Four questions the harness could not previously answer, each with its own
failure already on record:

1. **Which head produced this number?** One classifier is the best-of reranker,
   the do-no-harm guard AND the verdict metric, and it is refit whenever the
   corpus grows. Run 4a's detector was refit the same day as the verdict it
   produced, and the score rows recorded `name`/`embed_model` only.
2. **What is the effective n?** 77 of the 103 frozen-eval pairs come from
   `notebook/automatic_coding.qmd`, and the run-4 CI treated 127 chunks as 127
   independent observations.
3. **Real or synthetic?** 62 of those 103 pairs are our own generators'
   paraphrases; a headline mean over them measures the slop model.
4. **Was this scored at chunk scale or document scale?** Everything until now
   scored pre-cut chunks, so a chunk-diet switch could not move its own verdict.

No network, no key, no model: the detector is a stub head, the Pangram transport
is injected, and the document-scale styler is a stub `rewrite`.
"""

from __future__ import annotations

import json

import pytest

from stylebot import classify, synth
from stylebot import eval as ev
from stylebot import splits as sp
from stylebot.classify_train import Dataset

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_pairs(data_dir, specs):
    """A valid pairs.jsonl via the real synth path. `specs`: (text, source)."""
    for i, (text, source) in enumerate(specs):
        target = synth.Target(text=text, source=source, chunk_index=0, chunk_total=1, context="")
        gen = synth.Generator(name=f"m{i}", generate=lambda s: "[slop] " + s, strategy="polish")
        synth.synthesize_pairs([target], data_dir, [gen])
    return data_dir / "pairs.jsonl"


def _stub_detector(*, git_sha: str, score: float = 0.4, trained_at: str = "2026-01-01", n=10):
    """A real `embedding_classifier_detector` over a one-dimensional head."""
    return classify.embedding_classifier_detector(
        embed_fn=lambda prose: [score],
        head={"coef": [1.0], "intercept": 0.0},
        meta={
            "embed_model": "stub-embed",
            "git_sha": git_sha,
            "trained_at": trained_at,
            "n_pairs": n,
        },
    )


def _score_rows(ids_and_values, *, git_sha, source_of=None, synthetic=False):
    """Minimal scores.jsonl records: one detector score per id."""
    return [
        {
            "id": rid,
            "meta": {
                "source": (source_of or {}).get(rid, "post/a.qmd"),
                "synthetic": synthetic,
            },
            "scores": {
                "output": {
                    "detector": {
                        "score": value,
                        "configured": True,
                        "name": "voice-clf",
                        "embed_model": "stub-embed",
                        "git_sha": git_sha,
                        "trained_at": "2026-01-01",
                        "n_pairs": 10,
                    }
                }
            },
        }
        for rid, value in ids_and_values
    ]


# ---------------------------------------------------------------------------
# 1. the instrument travels with the reading
# ---------------------------------------------------------------------------


def test_detector_stamps_its_artifact_identity_on_every_score():
    detector = _stub_detector(git_sha="abc123")
    result = detector("some prose")
    assert result["git_sha"] == "abc123"
    assert result["trained_at"] == "2026-01-01"
    assert result["n_pairs"] == 10
    # And answerable without spending an embed, which is how the guards ask.
    assert ev.detector_identity(detector)["git_sha"] == "abc123"


def test_null_detector_has_an_identity_too():
    identity = ev.detector_identity(ev.null_detector)
    assert identity["name"] == "null"
    assert identity["git_sha"] is None


def test_score_rows_carry_the_detector_identity(tmp_path):
    pairs = _make_pairs(tmp_path / "c", [("A paragraph long enough to score.", "post/a.qmd")])
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, detector=_stub_detector(git_sha="deadbeef"))
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["scores"]["slop"]["detector"]["git_sha"] == "deadbeef"
    assert ev.detector_identities(out) == [
        {
            "name": "voice-clf",
            "embed_model": "stub-embed",
            "git_sha": "deadbeef",
            "trained_at": "2026-01-01",
            "n_pairs": 10,
        }
    ]


def test_resuming_with_a_refit_detector_is_refused(tmp_path):
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A first paragraph, long enough to score properly.", "post/a.qmd"),
            ("A second paragraph, also long enough to score.", "post/b.qmd"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, detector=_stub_detector(git_sha="old"), limit=1)

    with pytest.raises(ValueError, match="blend two instruments"):
        ev.score_pairs_file(pairs, out, detector=_stub_detector(git_sha="new"))

    # The escape hatch exists, and says so in the file afterwards.
    ev.score_pairs_file(
        pairs, out, detector=_stub_detector(git_sha="new"), allow_mixed_detector=True
    )
    assert len(ev.detector_identities(out)) == 2


def test_an_unconfigured_detector_never_trips_the_guard(tmp_path):
    """A keyless re-run adds no detector scores, so it cannot blend the means."""
    pairs = _make_pairs(tmp_path / "c", [("A paragraph long enough to score.", "post/a.qmd")])
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, detector=_stub_detector(git_sha="old"))
    ev.score_pairs_file(pairs, out, detector=ev.null_detector)  # no raise


def test_compare_refuses_across_detector_artifacts():
    a = _score_rows([("p1", 0.2), ("p2", 0.3)], git_sha="old")
    b = _score_rows([("p1", 0.25), ("p2", 0.35)], git_sha="new")
    with pytest.raises(ValueError, match="across detector artifacts"):
        ev.compare_scores(a, b)

    forced = ev.compare_scores(a, b, allow_cross_detector=True)
    assert forced["cross_detector"] is True
    assert forced["delta"]["mean"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# 2. the interval knows about clusters
# ---------------------------------------------------------------------------


def test_cluster_bootstrap_is_wider_than_the_iid_one_on_a_concentrated_stratum():
    """The whole point: 40 chunks from 2 posts are not 40 observations.

    The differences below are perfectly correlated within a post and opposite
    between posts, so the chunk-level interval collapses while the post-level one
    must straddle both post means.
    """
    a = [1.0] * 20 + [0.0] * 20
    b = [0.0] * 40
    clusters = ["post/one.qmd"] * 20 + ["post/two.qmd"] * 20

    iid = ev.paired_delta(a, b, n_boot=500, seed=0)
    clustered = ev.paired_delta(a, b, clusters=clusters, n_boot=500, seed=0)

    assert clustered["method"] == "cluster_bootstrap"
    assert clustered["n"] == 40
    assert clustered["n_clusters"] == 2
    assert iid["mean"] == clustered["mean"] == pytest.approx(0.5)
    iid_width = iid["ci_high"] - iid["ci_low"]
    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    assert clustered_width > iid_width
    # And the clustered interval spans zero, which is the honest answer here.
    assert clustered["ci_low"] <= 0.0 <= clustered["ci_high"]


def test_paired_delta_declines_to_interval_a_single_cluster():
    delta = ev.paired_delta([1.0, 2.0], [0.0, 0.0], clusters=["p", "p"])
    assert delta["method"] == "single-cluster"
    assert delta["ci_low"] is None and delta["ci_high"] is None


def test_compare_clusters_by_post_by_default():
    sources = {"p1": "post/a.qmd", "p2": "post/a.qmd", "p3": "post/b.qmd"}
    a = _score_rows([("p1", 0.2), ("p2", 0.2), ("p3", 0.4)], git_sha="x", source_of=sources)
    b = _score_rows([("p1", 0.3), ("p2", 0.3), ("p3", 0.5)], git_sha="x", source_of=sources)
    verdict = ev.compare_scores(a, b)
    assert verdict["delta"]["n"] == 3
    assert verdict["delta"]["n_clusters"] == 2  # two posts, not three chunks
    assert verdict["verdict"] == "a better"  # lower P(slop) wins


# ---------------------------------------------------------------------------
# 3. the headline is the real facet
# ---------------------------------------------------------------------------


def test_summary_facets_by_provenance_and_names_the_headline():
    rows = _score_rows([("r1", 0.2)], git_sha="x") + _score_rows(
        [("s1", 0.8), ("s2", 0.9)], git_sha="x", synthetic=True
    )
    summary = ev.summarize_scores(rows)
    assert summary["headline_facet"] == "real"
    assert summary["by_provenance"]["real"]["pairs"] == 1
    assert summary["by_provenance"]["synthetic"]["pairs"] == 2
    real = summary["by_provenance"]["real"]["fields"]["output"]
    assert real["mean_detector_score"] == pytest.approx(0.2)
    # The all-rows mean is still reported — it is just not the headline.
    assert summary["fields"]["output"]["mean_detector_score"] == pytest.approx(
        (0.2 + 0.8 + 0.9) / 3
    )
    assert ev.format_summary(summary)[0].startswith("HEADLINE")


def test_compare_scores_reports_the_real_facet_by_default():
    real = _score_rows([("r1", 0.2), ("r2", 0.3)], git_sha="x")
    synth_rows = _score_rows([("s1", 0.9)], git_sha="x", synthetic=True)
    b_real = _score_rows([("r1", 0.4), ("r2", 0.5)], git_sha="x")
    b_synth = _score_rows([("s1", 0.1)], git_sha="x", synthetic=True)

    default = ev.compare_scores(real + synth_rows, b_real + b_synth)
    assert default["provenance"] == "real"
    assert default["delta"]["n"] == 2
    assert default["delta"]["mean"] == pytest.approx(-0.2)

    everything = ev.compare_scores(real + synth_rows, b_real + b_synth, provenance=None)
    assert everything["delta"]["n"] == 3


# ---------------------------------------------------------------------------
# 4. document-scale eval, and the Pangram cross-check
# ---------------------------------------------------------------------------


def _archive(data_dir, capture_id, source, before, after):
    d = data_dir / "captures" / capture_id
    d.mkdir(parents=True)
    (d / "before.md").write_text(before, encoding="utf-8")
    (d / "after.md").write_text(after, encoding="utf-8")
    (d / "meta.json").write_text(
        json.dumps(
            {
                "capture_id": capture_id,
                "source": source,
                "snapped_at": "2026-01-01T00:00:00+00:00",
                "whole": True,
            }
        ),
        encoding="utf-8",
    )


def test_document_scale_rewrite_emits_a_scorable_output_jsonl(tmp_path):
    data_dir = tmp_path / "c"
    pairs = _make_pairs(data_dir, [("A paragraph long enough to score.", "post/a.qmd")])
    _archive(data_dir, "cap-eval", "post/eval.qmd", "Before doc.\n", "After doc.\n")
    _archive(data_dir, "cap-styler", "post/styler.qmd", "Other before.\n", "Other after.\n")

    splits = {
        "schema_version": 1,
        "eval_posts": ["post/eval.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    out = tmp_path / "documents.jsonl"
    calls = []

    def fake_rewrite(text, styler, **kw):
        calls.append((text, kw))
        return type("R", (), {"text": text.upper(), "n_chunks": 1, "n_kept_input": 0})()

    result = ev.rewrite_documents_file(
        pairs, out, styler=object(), splits=splits, rewrite=fake_rewrite, max_chunk_chars=4500
    )
    assert result.written == 1  # the styler-role archive was not touched
    assert calls[0][1]["max_chunk_chars"] == 4500

    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["output"] == "BEFORE DOC.\n"
    assert rec["meta"]["capture_id"] == "doc:cap-eval"
    assert rec["meta"]["scale"] == "document"

    # It scores through the ordinary path, with no orphan pruning (it is a
    # DIFFERENT corpus from pairs.jsonl and every id would read as orphaned).
    scores = tmp_path / "doc-scores.jsonl"
    ev.score_pairs_file(
        out,
        scores,
        fields=("slop", "target", "output"),
        detector=_stub_detector(git_sha="x"),
        prune_orphans=False,
    )
    scored = json.loads(scores.read_text().splitlines()[0])
    assert set(scored["scores"]) == {"slop", "target", "output"}

    # Resumable: a second pass rewrites nothing.
    again = ev.rewrite_documents_file(
        pairs, out, styler=object(), splits=splits, rewrite=fake_rewrite
    )
    assert again.written == 0


def test_pangram_detector_scores_fraction_ai_through_an_injected_transport():
    posted = []

    def fake_post(url, payload, headers):
        posted.append((url, payload, headers))
        return {"fraction_ai": 0.83, "prediction": "AI"}

    detector = ev.pangram_detector(api_key="k", post=fake_post)
    result = detector("some prose")
    assert result["score"] == pytest.approx(0.83)  # polarity: higher = more AI
    assert result["name"] == "pangram"
    assert posted[0][1] == {"text": "some prose"}
    assert posted[0][2]["x-api-key"] == "k"


def test_pangram_says_none_rather_than_zero_when_the_field_is_absent():
    detector = ev.pangram_detector(api_key="k", post=lambda *a: {"unexpected": 1})
    assert detector("prose")["score"] is None


# ---------------------------------------------------------------------------
# the stratum: versioned, add-only, and honest about concentration
# ---------------------------------------------------------------------------


def _ds(posts_pairs, synth_posts=()):
    ds = Dataset()
    for post, n in posts_pairs.items():
        for c in range(n):
            ds._add_pair(f"s{c}", f"a{c}", post, synthetic=post in synth_posts)
    return ds


def test_extend_eval_is_add_only_and_versions_the_stratum():
    base = sp.make_splits([f"post/p{i}.qmd" for i in range(20)], seed=0)
    assert sp.eval_version(base) == 1

    grown = sp.extend_eval(base, ["post/new.qmd"], note="eval_v2")
    assert sp.eval_version(grown) == 2
    assert set(base["eval_posts"]) < set(grown["eval_posts"])
    assert grown["eval_history"][-1]["added"] == ["post/new.qmd"]
    assert sp.role_of("post/new.qmd", grown) == "eval"
    # The input is untouched: an extension is a new document, not a mutation.
    assert "eval_history" not in base

    # Adding nothing new is a no-op, not a version bump.
    assert sp.extend_eval(grown, grown["eval_posts"]) is grown


def test_extend_eval_counts_a_renamed_post_as_already_pinned():
    base = {
        "schema_version": 1,
        "eval_posts": ["notebook/old.qmd"],
        "aliases": {"notebook/new.qmd": "notebook/old.qmd"},
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    assert sp.extend_eval(base, ["notebook/new.qmd"]) is base


def test_check_splits_flags_a_stratum_dominated_by_one_post():
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/big.qmd", "post/small.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    ds = _ds({"post/big.qmd": 77, "post/small.qmd": 26})
    roles = sp.summarize_roles(splits, ds)
    assert roles["eval"]["top_post"] == "post/big.qmd"
    assert roles["eval"]["top_post_share"] == pytest.approx(77 / 103)
    assert any(w.startswith("CONCENTRATED") for w in sp.check_splits(splits, ds))


def test_check_splits_flags_a_synthetic_majority_eval_stratum():
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/a.qmd", "post/b.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    ds = _ds({"post/a.qmd": 41, "post/b.qmd": 62}, synth_posts={"post/b.qmd"})
    warnings = [w for w in sp.check_splits(splits, ds) if w.startswith("EVAL PURITY")]
    assert warnings and "headline verdict must be the REAL facet" in warnings[0]


def test_splits_sha256_moves_when_the_partition_does(tmp_path):
    path = tmp_path / "splits.json"
    base = sp.make_splits([f"post/p{i}.qmd" for i in range(20)], seed=0)
    sp.save_splits(base, path)
    before = sp.splits_sha256(path)
    sp.save_splits(sp.extend_eval(base, ["post/new.qmd"]), path)
    assert sp.splits_sha256(path) != before
