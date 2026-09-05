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

No network, no key, no model: the detector is a stub head, and the
document-scale styler is a stub `rewrite`.
"""

from __future__ import annotations

import json

import pytest
from conftest import corpus_file

from stylebot import classify, synth
from stylebot import eval as ev
from stylebot import splits as sp
from stylebot.classify_train import Dataset
from stylebot.errors import ConfigError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_pairs(data_dir, specs):
    """A valid corpus via the real synth path, as the view of its draws (D84).

    `specs`: (text, source).
    """
    for i, (text, source) in enumerate(specs):
        target = synth.Target(text=text, source=source, chunk_index=0, chunk_total=1, context="")
        gen = synth.Generator(name=f"m{i}", generate=lambda s: "[slop] " + s, strategy="polish")
        synth.synthesize_pairs([target], data_dir, [gen])
    return corpus_file(data_dir, data_dir / "pairs.jsonl")


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


def _with_judge(rows, scores):
    for row, score in zip(rows, scores, strict=True):
        row["scores"]["output"]["judge"] = (
            {"score": score} if score is not None else {"score": None, "parse_error": True}
        )
    return rows


def test_compare_scores_counts_a_refusal_as_unscored_not_a_3():
    # A judge refusal on one side drops that pair from the paired delta and is
    # reported as unscored beside the mean it did not enter (C4).
    ids = [("p1", 0.1), ("p2", 0.1), ("p3", 0.1)]
    a = _with_judge(_score_rows(ids, git_sha="x"), [4, None, 5])
    b = _with_judge(_score_rows(ids, git_sha="x"), [3, 3, 3])
    verdict = ev.compare_scores(a, b, signal="judge")
    assert verdict["n_unscored"] == {"a": 1, "b": 0}
    assert verdict["mean_a"] == pytest.approx(4.5)
    assert verdict["mean_b"] == pytest.approx(3.0)
    assert verdict["delta"]["n"] == 2
    assert verdict["delta"]["mean"] == pytest.approx(1.5)
    lines = ev.format_verdict(verdict)
    assert any("unscored 1" in line for line in lines if "a =" in line)


# ---------------------------------------------------------------------------
# 4. document-scale eval
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


# ---------------------------------------------------------------------------
# `extend-eval`: the operation around the transform (preview, refuse, write)
# ---------------------------------------------------------------------------


def _splits_file(tmp_path, posts=("post/a.qmd",)):
    path = tmp_path / "splits.json"
    sp.save_splits(
        {
            "schema_version": 1,
            "eval_posts": list(posts),
            "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
        },
        path,
    )
    return path


def _emitted(lines):
    def emit(message, /, *, level="info"):
        lines.append((level, message))

    return emit


def test_extend_eval_dry_run_reports_the_move_and_writes_nothing(tmp_path):
    path = _splits_file(tmp_path)
    before = path.read_text(encoding="utf-8")
    lines: list[tuple[str, str]] = []

    report = sp.apply_eval_extension(
        sp.ExtendEvalOpts(splits_path=path, posts=("post/b.qmd", "post/a.qmd")),
        emit=_emitted(lines),
    )

    assert report["applied"] is False
    assert path.read_text(encoding="utf-8") == before
    assert report["already_eval"] == ["post/a.qmd"]
    (addition,) = report["additions"]
    assert addition["post"] == "post/b.qmd"
    # It names the pool the prose is leaving, which is the whole cost of the move.
    assert addition["from_role"] == sp.role_of("post/b.qmd", sp.load_splits(path))
    assert report["eval_version_before"] == 1 and report["eval_version_after"] == 2
    assert any("dry run" in message for _, message in lines)


def test_extend_eval_apply_writes_the_bumped_stratum(tmp_path):
    path = _splits_file(tmp_path)
    before_sha = sp.splits_sha256(path)

    report = sp.apply_eval_extension(
        sp.ExtendEvalOpts(splits_path=path, posts=("post/b.qmd",), note="eval_v2", apply=True)
    )

    written = sp.load_splits(path)
    assert report["applied"] is True
    assert report["splits_sha256_after"] != before_sha
    assert sp.eval_version(written) == 2
    assert sp.role_of("post/b.qmd", written) == "eval"
    assert written["eval_history"][-1]["note"] == "eval_v2"


def test_extend_eval_refuses_a_post_that_is_not_on_disk(tmp_path):
    path = _splits_file(tmp_path)
    (tmp_path / "post").mkdir()
    (tmp_path / "post" / "b.qmd").write_text("x", encoding="utf-8")

    with pytest.raises(ConfigError, match="not on disk"):
        sp.apply_eval_extension(
            sp.ExtendEvalOpts(
                splits_path=path,
                posts=("post/b.qmd", "post/typo.qmd"),
                blog_root=tmp_path,
                apply=True,
            )
        )
    assert sp.eval_version(sp.load_splits(path)) == 1


def test_extend_eval_needs_posts_and_leaves_a_no_op_unversioned(tmp_path):
    path = _splits_file(tmp_path)
    with pytest.raises(ConfigError, match="no posts to add"):
        sp.apply_eval_extension(sp.ExtendEvalOpts(splits_path=path))

    report = sp.apply_eval_extension(
        sp.ExtendEvalOpts(splits_path=path, posts=("post/a.qmd",), apply=True)
    )
    assert report["applied"] is False
    assert sp.eval_version(sp.load_splits(path)) == 1


def test_extend_eval_reads_a_posts_file(tmp_path):
    path = _splits_file(tmp_path)
    listing = tmp_path / "posts.txt"
    listing.write_text("# the new cut\npost/b.qmd\n\n  post/c.qmd  # why\n", encoding="utf-8")

    report = sp.apply_eval_extension(sp.ExtendEvalOpts(splits_path=path, posts_file=listing))
    assert [a["post"] for a in report["additions"]] == ["post/b.qmd", "post/c.qmd"]


def test_extend_eval_cli_is_a_dry_run_by_default(tmp_path):
    from click.testing import CliRunner

    from stylebot.bin.ai_style import main

    path = _splits_file(tmp_path)
    argv = ["extend-eval", "--splits", str(path), "--post", "post/b.qmd"]
    result = CliRunner().invoke(main, argv)
    assert result.exit_code == 0, result.output
    assert "would re-role 1 post(s)" in result.output
    assert sp.eval_version(sp.load_splits(path)) == 1

    result = CliRunner().invoke(main, [*argv, "--apply"])
    assert result.exit_code == 0, result.output
    assert sp.eval_version(sp.load_splits(path)) == 2


# ---------------------------------------------------------------------------
# 5. document scale, part two: score chunk-wise, and do not share a process
#    with the sampler (2026-09-03, W10 item 7's follow-up)
# ---------------------------------------------------------------------------
#
# The first live `--documents` run found two things no stub had: the command
# segfaults (exit 139, hidden by a `| tail`), and the verdict it would have
# produced has no resolution — mean P(slop) 0.4880 / 0.4951 / 0.4914 for the
# draft, Dan's edit and the styler's output, with byte-identical min and max
# across all three fields. Both are fixed here, and both get a test that could
# have caught them.


DOC = (
    "---\ntitle: T\n---\n"
    "Alpha paragraph, prose the styler would rewrite.\n"
    "\n"
    "```python\nx = 1\n```\n"
    "\n"
    "Bravo paragraph, also prose.\n"
    "\n"
    "Charlie paragraph, the third.\n"
)


def _recording_detector(scores_by_chunk):
    """A detector that answers per chunk and records every text it saw."""
    seen = []

    def detect(prose):
        seen.append(prose)
        return {
            "score": scores_by_chunk.get(prose[:5], 0.5),
            "p_dan": None,
            "configured": True,
            "name": "stub",
            "embed_model": "stub-embed",
            "git_sha": "abc",
            "trained_at": "2026-01-01",
            "n_pairs": 10,
        }

    detect.identity = {
        "name": "stub",
        "embed_model": "stub-embed",
        "git_sha": "abc",
        "trained_at": "2026-01-01",
        "n_pairs": 10,
    }
    return detect, seen


def test_chunked_detector_scores_chunks_not_the_document():
    detector, seen = _recording_detector({"Alpha": 0.9, "Bravo": 0.3, "Charl": 0.6})
    chunked = ev.chunked_detector(detector, max_chunk_chars=40)

    result = chunked(DOC)

    # Three prose chunks — never the whole document, never the fence, never
    # the frontmatter. That is the whole point: over 25k chars the head has no
    # resolution, so it is never handed 25k chars.
    assert seen == [
        "Alpha paragraph, prose the styler would rewrite.",
        "Bravo paragraph, also prose.",
        "Charlie paragraph, the third.\n",
    ]
    assert result["n_chunks"] == 3
    assert result["score"] == pytest.approx((0.9 + 0.3 + 0.6) / 3)
    assert result["p_dan"] == pytest.approx(1 - (0.9 + 0.3 + 0.6) / 3)
    assert result["sd"] == pytest.approx(0.3)


def test_chunked_detector_reports_no_signal_for_a_document_with_no_prose():
    """`None`, not 0.0 — an absent reading is said out loud, as everywhere else."""
    detector, seen = _recording_detector({})
    chunked = ev.chunked_detector(detector, max_chunk_chars=40)
    result = chunked("# Only a heading\n\n```\ncode\n```\n")
    assert seen == []
    assert result["score"] is None and result["n_chunks"] == 0 and result["sd"] is None


def test_chunked_detector_carries_the_head_identity_with_the_scale_appended():
    detector, _ = _recording_detector({})
    chunked = ev.chunked_detector(detector, max_chunk_chars=4500, inline_headings=True)
    identity = ev.detector_identity(chunked)
    assert identity["git_sha"] == "abc"  # same artifact
    assert identity["name"] == "stub/chunked:4500:inline-headings"  # different instrument


def test_a_chunk_wise_pass_refuses_to_append_to_a_whole_document_file(tmp_path):
    """Two scales of one head are two instruments, and must not blend into a mean.

    This is precisely D39's guard doing the job it was built for: the
    whole-document numbers and the chunk-wise ones differ by more than anything
    being measured across them.
    """
    data_dir = tmp_path / "c"
    pairs = _make_pairs(data_dir, [("A paragraph long enough to score.", "post/a.qmd")])
    out = tmp_path / "scores.jsonl"
    whole, _ = _recording_detector({})
    ev.score_pairs_file(pairs, out, detector=whole, fields=("slop",))

    chunked = ev.chunked_detector(whole, max_chunk_chars=4500)
    with pytest.raises(ConfigError, match="blend two instruments"):
        ev.score_pairs_file(pairs, out, detector=chunked, fields=("slop",))
    # A fresh --out is the fix, and it is not refused.
    ev.score_pairs_file(pairs, tmp_path / "chunked.jsonl", detector=chunked, fields=("slop",))


def _divs_are_prose(cls: str) -> bool:
    """A `div_is_prose` policy, at module level so its qualname is stable."""
    return cls == "callout-note"


def test_a_div_policy_is_a_different_instrument_at_the_same_scale(tmp_path):
    """`div_is_prose` decides which text is scored, so it belongs in `name`.

    Two document-scale runs at one scale under two div policies read different
    prose out of the same documents. Without the policy in the identity both
    call themselves `stub/chunked:4500` and D39's guard cannot separate them.
    """
    detector, _ = _recording_detector({})
    plain = ev.chunked_detector(detector, max_chunk_chars=4500)
    policied = ev.chunked_detector(detector, max_chunk_chars=4500, div_is_prose=_divs_are_prose)
    plain_name = ev.detector_identity(plain)["name"]
    policied_name = ev.detector_identity(policied)["name"]
    assert plain_name == "stub/chunked:4500"
    assert policied_name.startswith("stub/chunked:4500:divs=")
    assert policied_name.endswith("._divs_are_prose")  # module.qualname, not an address

    data_dir = tmp_path / "c"
    pairs = _make_pairs(data_dir, [("A paragraph long enough to score.", "post/a.qmd")])
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, detector=plain, fields=("slop",))
    with pytest.raises(ConfigError, match="blend two instruments"):
        ev.score_pairs_file(pairs, out, detector=policied, fields=("slop",))
    ev.score_pairs_file(pairs, tmp_path / "divs.jsonl", detector=policied, fields=("slop",))


def test_chunk_wise_document_scores_keep_the_ordinary_scores_jsonl_shape(tmp_path):
    """`summarize_scores` / `format_summary` / the report see nothing new.

    The aggregate lands in the same `scores[field]["detector"]["score"]` slot,
    with `n_chunks` and `sd` beside it, so every consumer keeps working.
    """
    data_dir = tmp_path / "c"
    pairs = _make_pairs(data_dir, [("A paragraph long enough to score.", "post/a.qmd")])
    _archive(data_dir, "cap-eval", "post/eval.qmd", DOC, DOC.replace("Bravo", "Delta"))
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/eval.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    docs = tmp_path / "documents.jsonl"
    ev.rewrite_documents_file(
        pairs,
        docs,
        styler=object(),
        splits=splits,
        rewrite=lambda text, styler, **kw: type("R", (), {"text": text, "n_chunks": 3})(),
        max_chunk_chars=40,
    )

    detector, _ = _recording_detector({"Alpha": 0.9, "Bravo": 0.3, "Charl": 0.6, "Delta": 0.1})
    scores = tmp_path / "doc-scores.jsonl"
    ev.score_pairs_file(
        docs,
        scores,
        fields=("slop", "target", "output"),
        detector=ev.chunked_detector(detector, max_chunk_chars=40),
        prune_orphans=False,
    )

    rec = json.loads(scores.read_text().splitlines()[0])
    slop = rec["scores"]["slop"]["detector"]
    assert slop["n_chunks"] == 3 and slop["sd"] is not None
    assert slop["score"] == pytest.approx((0.9 + 0.3 + 0.6) / 3)
    # Dan's side differs (Bravo -> Delta), which the whole-document reading
    # could not see at all.
    assert rec["scores"]["target"]["detector"]["score"] == pytest.approx((0.9 + 0.1 + 0.6) / 3)

    summary = ev.summarize_scores(scores)
    assert summary["fields"]["slop"]["mean_detector_score"] == pytest.approx(0.6)
    assert any("slop" in line for line in ev.format_summary(summary))
    assert summary["detectors"][0]["name"] == "stub/chunked:40"


CALLOUT_DOC = (
    "---\ntitle: T\n---\n"
    "Alpha paragraph, ordinary prose outside any div.\n"
    "\n"
    "::: {.callout-note}\n"
    "Bravo paragraph, callout prose in Dan's own voice.\n"
    ":::\n"
    "\n"
    "::: {#fig-one}\n"
    "Charlie caption, markup rather than prose.\n"
    ":::\n"
)


def _blog_div_is_prose(attrs: str) -> bool:
    """The blog's policy in miniature: `callout-*` wraps prose, a figure does not."""
    return "callout-" in attrs


def test_the_callout_policy_crosses_the_document_scale_seam(tmp_path):
    """A `.callout-note` paragraph is BOTH rewritten and scored (F4).

    `run`, `style` and `serve` pass the blog's `div_is_prose`; the
    document-scale eval did not, so the verdict committed on 2026-09-03 was
    computed on a diet its own serving path does not use. Both halves take the
    hook here — the real `infer.rewrite_text` under a stub styler, and a real
    `chunked_detector` — and the no-hook run beside it is the control.
    """
    data_dir = tmp_path / "c"
    pairs = _make_pairs(data_dir, [("A paragraph long enough to score.", "post/a.qmd")])
    _archive(data_dir, "cap-eval", "post/eval.qmd", CALLOUT_DOC, CALLOUT_DOC)
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/eval.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }

    def run(out_name, div_is_prose):
        sent = []

        def styler(messages, num_samples=1):
            sent.append(messages[-1]["content"])
            return [messages[-1]["content"] + " Styled."] * num_samples

        docs = tmp_path / out_name
        rewritten = ev.rewrite_documents_file(
            pairs,
            docs,
            styler,
            splits=splits,
            max_chunk_chars=200,
            div_is_prose=div_is_prose,
        )
        detector, seen = _recording_detector({})
        scores = tmp_path / f"scores-{out_name}"
        ev.score_pairs_file(
            docs,
            scores,
            fields=("slop", "output"),
            prune_orphans=False,
            detector=ev.chunked_detector(detector, max_chunk_chars=200, div_is_prose=div_is_prose),
        )
        rec = json.loads(scores.read_text().splitlines()[0])
        return rewritten, sent, seen, json.loads(docs.read_text().splitlines()[0]), rec

    without, sent_w, seen_w, _, rec_w = run("plain.jsonl", None)
    with_hook, sent_h, seen_h, doc_h, rec_h = run("callout.jsonl", _blog_div_is_prose)

    # The rewrite half: the callout paragraph is only ever sent to the styler
    # when the hook is passed, and the figure caption never is.
    assert not any("Bravo paragraph" in c for c in sent_w)
    assert any("Bravo paragraph" in c for c in sent_h)
    assert not any("Charlie caption" in c for c in sent_h)
    assert with_hook.n_chunks > without.n_chunks

    # The `:::` marker lines carry class semantics and stay verbatim; what they
    # delimit is what got rewritten.
    styled_callout = (
        "::: {.callout-note}\nBravo paragraph, callout prose in Dan's own voice. Styled.\n:::"
    )
    assert styled_callout in doc_h["output"]
    assert "Charlie caption, markup rather than prose." in doc_h["output"]
    assert "Charlie caption, markup rather than prose. Styled." not in doc_h["output"]

    # The scoring half: the same chunk reaches the detector, so the verdict is
    # computed on the diet that was served.
    assert not any("Bravo paragraph" in t for t in seen_w)
    assert any("Bravo paragraph" in t for t in seen_h)
    for field in ("slop", "output"):
        assert (
            rec_h["scores"][field]["detector"]["n_chunks"]
            > rec_w["scores"][field]["detector"]["n_chunks"]
        )
