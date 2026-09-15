"""Tests for the generic linear-head classifier seam — NO ML deps.

The whole point of `stylebot.classify` is that scoring a passage needs no
numpy/sklearn/sentence-transformers: a stub `embed_fn` + a hand-made `head.json`
must produce a valid `P(slop)` detector. These tests prove that seam is
dependency-free (they import only stdlib + stylebot), and pin the `score = P(slop)`
polarity that lets the trained detector compose with `mean_detector_score`.
"""

from __future__ import annotations

import json

import pytest

from stylebot import classify
from stylebot.errors import RunError


def test_sigmoid_is_stable_at_extremes():
    # No overflow on large-magnitude logits.
    assert classify._sigmoid(1000.0) == pytest.approx(1.0)
    assert classify._sigmoid(-1000.0) == pytest.approx(0.0)


def test_detector_returns_p_slop_polarity_and_p_dan():
    # A slop-leaning embedding (aligned with +coef) scores HIGH (more AI-like);
    # a Dan-leaning one (anti-aligned) scores LOW. score == P(slop).
    head = {"coef": [2.0, 0.0], "intercept": 0.0}

    def stub_embed(prose: str):
        # "slop" -> aligned with coef (high P(slop)); "dan" -> anti-aligned.
        return [3.0, 0.0] if "slop" in prose else [-3.0, 0.0]

    det = classify.embedding_classifier_detector(
        embed_fn=stub_embed, head=head, name="test-clf", meta={"embed_model": "stub"}
    )

    slop = det("this is slop")
    dan = det("this is dan")

    for rec in (slop, dan):
        assert 0.0 <= rec["score"] <= 1.0
        assert rec["p_dan"] == pytest.approx(1.0 - rec["score"])
        assert rec["configured"] is True
        assert rec["name"] == "test-clf"
        assert rec["embed_model"] == "stub"

    # Polarity: slop is MORE AI-like than Dan.
    assert slop["score"] > 0.5 > dan["score"]
    assert slop["score"] > dan["score"]


def test_detector_dim_mismatch_raises():
    head = {"coef": [1.0, 1.0, 1.0], "intercept": 0.0}
    det = classify.embedding_classifier_detector(
        embed_fn=lambda _p: [1.0, 1.0], head=head, meta={"embed_model": "x"}
    )
    with pytest.raises(RunError, match="does not match"):
        det("anything")


def test_load_linear_head_roundtrip_and_validation(tmp_path):
    good = tmp_path / "head.json"
    good.write_text(json.dumps({"coef": [0.1, 0.2], "intercept": -0.3, "calibration": None}))
    head = classify.load_linear_head(good)
    assert head["coef"] == [0.1, 0.2]
    assert head["intercept"] == -0.3

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"intercept": 0.0}))  # missing coef
    with pytest.raises(RunError, match="coef"):
        classify.load_linear_head(bad)


def test_detector_composes_into_score_candidate():
    """The detector drops into `eval.score_candidate(detector=…)` unchanged."""
    from stylebot import eval as ev

    head = {"coef": [1.0], "intercept": 0.0}
    det = classify.embedding_classifier_detector(
        embed_fn=lambda _p: [2.0], head=head, meta={"embed_model": "stub"}
    )
    rec = ev.score_candidate("some prose", judge=None, detector=det)
    assert rec["detector"]["score"] == pytest.approx(classify._sigmoid(2.0))
    assert rec["detector"]["name"] == "voice-clf"


def test_runtime_import_pulls_no_ml_deps():
    """The dep-free guarantee, enforced: importing the runtime (classify) — and
    even the trainer module (classify_train, whose heavy imports are lazy) —
    must not load sklearn/numpy/sentence_transformers/torch. Run in a subprocess
    so this process's own imports can't contaminate the check."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import stylebot.classify, stylebot.classify_train, stylebot.splits\n"
        "bad = [m for m in ('sklearn', 'numpy', 'sentence_transformers', 'torch')\n"
        "       if any(k == m or k.startswith(m + '.') for k in sys.modules)]\n"
        "assert not bad, f'ML deps loaded at import time: {bad}'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_embedding_classifier_detector_exposes_a_batch_path_that_agrees():
    """`.batch_detect` must be the same instrument as `detect`, in one encode.

    The audit finding behind it (D57): `sentence_transformers_batch_embed_fn`
    has always returned a batch encoder, and every caller was handing it one
    text at a time — 22.2 s of batch-of-one encodes against 14.2 s batched over
    the 1,690 serving-scale chunks of the document-scale eval.
    """
    head = {"coef": [1.0, -1.0], "intercept": 0.0}
    calls: list[int] = []

    def encode(texts, *, batch_size: int = 32):
        texts = list(texts)
        calls.append(len(texts))
        return [[len(t) / 10.0, 1.0] for t in texts]

    def embed(prose: str):
        return encode([prose])[0]

    det = classify.embedding_classifier_detector(
        embed_fn=embed, head=head, name="stub", batch_embed_fn=encode
    )
    texts = ["a", "bb", "ccc"]
    one_at_a_time = [det(t) for t in texts]
    assert calls == [1, 1, 1]
    calls.clear()
    assert det.batch_detect(texts) == one_at_a_time
    assert calls == [3]
    # And `detect_many` is how a caller holding many texts reaches it.
    calls.clear()
    assert classify.detect_many(det, texts) == one_at_a_time
    assert calls == [3]
    assert det.batch_detect([]) == []


def test_detect_many_falls_back_when_the_detector_has_no_batch_path():
    calls: list[int] = []

    def plain(prose: str) -> dict:
        calls.append(1)
        return {"score": 0.5, "configured": True, "name": "plain"}

    assert [r["score"] for r in classify.detect_many(plain, ["a", "b", "c"])] == [0.5] * 3
    assert calls == [1, 1, 1]  # the obvious loop; the caller never had to ask


# ---------------------------------------------------------------------------
# Marker-threshold calibration (D105): real pairs of one role, gated, quantile-cut
# ---------------------------------------------------------------------------


def _calibration_corpus(tmp_path):
    """Three real posts + one synthetic, two edited pairs each; slop sides say
    'slop', Dan sides say 'dan', and a stub detector reads the word."""
    from stylebot import splits as sp
    from stylebot.ai_core import STYLE_SYSTEM

    path = tmp_path / "pairs.jsonl"
    posts = ["post/a.qmd", "post/b.qmd", "post/c.qmd", "post/z.qmd"]
    with path.open("w", encoding="utf-8") as fp:
        for i, post in enumerate(posts):
            for c in range(2):
                slop = f"slop {i}{c}: the model wrote an even, tidy paragraph about {post}."
                dan = (
                    f"dan {i}{c}: I'd say {post} is odd, though — it wanders off, and that is fine."
                )
                rec = {
                    "messages": [
                        {"role": "system", "content": STYLE_SYSTEM},
                        {"role": "user", "content": slop},
                        {"role": "assistant", "content": dan},
                    ],
                    "meta": {"source": post, "synthetic": post == "post/z.qmd"},
                }
                fp.write(json.dumps(rec) + "\n")
    splits = sp.make_splits(posts, eval_frac=0.5, eval_candidates=posts[:2], seed=0)

    def stub_detector(prose: str) -> dict:
        n = int(prose.split(":")[0].split()[-1])  # the pair index, for spread
        return {"score": (0.6 if prose.startswith("slop") else 0.3) + n / 100, "configured": True}

    stub_detector.identity = {"name": "stub", "git_sha": "abc123"}
    return path, splits, stub_detector


def test_calibrate_scores_only_the_roles_real_pairs(tmp_path):
    path, splits, det = _calibration_corpus(tmp_path)
    cal = classify.calibrate_thresholds(det, path, splits=splits, role="eval")
    assert set(splits["eval_posts"]) == {"post/a.qmd", "post/b.qmd"}
    assert cal["n_pairs"] == 4 and cal["n_posts"] == 2 and cal["role"] == "eval"
    assert cal["instrument"]["name"] == "stub"
    # Every dan score sits below every slop score, so the rule's edges ascend.
    assert cal["quantiles"]["dan"][95] < cal["quantiles"]["slop"][5]
    assert cal["thresholds"] == sorted(cal["thresholds"]) and cal["ascending"]
    assert cal["rule"] == [["dan", 75], ["slop", 50], ["slop", 75], ["slop", 95]]


def test_calibrate_without_splits_takes_every_real_pair_never_synthetic(tmp_path):
    path, _, det = _calibration_corpus(tmp_path)
    cal = classify.calibrate_thresholds(det, path)
    assert cal["n_pairs"] == 6 and cal["n_posts"] == 3 and cal["role"] is None


def test_calibrate_refuses_an_empty_population(tmp_path):
    from stylebot.errors import ConfigError

    path, splits, det = _calibration_corpus(tmp_path)
    with pytest.raises(ConfigError, match="nothing to calibrate"):
        classify.calibrate_thresholds(det, path, splits=splits, role="eval", max_transform_sim=0.0)


def test_threshold_rule_parses_and_rejects():
    from stylebot.errors import ConfigError

    assert classify.parse_threshold_rule("dan:75, slop:50") == (("dan", 75), ("slop", 50))
    for bad in ("human:75", "dan:0", "dan:100", "dan", "slop:x"):
        with pytest.raises(ConfigError):
            classify.parse_threshold_rule(bad)


def test_format_calibration_warns_when_edges_do_not_ascend(tmp_path):
    path, splits, det = _calibration_corpus(tmp_path)
    cal = classify.calibrate_thresholds(det, path, splits=splits, rule=(("slop", 95), ("dan", 5)))
    lines = classify.format_calibration(cal)
    assert lines[0].startswith(
        "stub@abc123 on 4 real pairs (eval role, transform_sim ≤ 0.85), 2 posts"
    )
    assert any(line.startswith("thresholds [") for line in lines)
    assert any("WARNING" in line for line in lines) and not cal["ascending"]
