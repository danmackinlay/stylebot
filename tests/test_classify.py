"""Tests for the generic linear-head classifier seam — NO ML deps.

The whole point of `stylebot.classify` is that scoring a passage needs no
numpy/sklearn/sentence-transformers: a stub `embed_fn` + a hand-made `head.json`
must produce a valid `P(slop)` detector. These tests prove that seam is
dependency-free (they import only stdlib + stylebot), and pin the `score = P(slop)`
polarity that lets the trained detector compose with `mean_detector_score`.
"""

from __future__ import annotations

import json
import math

import pytest

from stylebot import classify


def test_logreg_p_slop_matches_sigmoid():
    # P(slop) = sigmoid(intercept + coef·vec); a hand-computed reference.
    vec = [1.0, 2.0, -1.0]
    coef = [0.5, -0.25, 1.0]
    intercept = 0.1
    z = 0.1 + (0.5 * 1.0 + -0.25 * 2.0 + 1.0 * -1.0)
    assert classify._logreg_p_slop(vec, coef, intercept) == pytest.approx(1 / (1 + math.exp(-z)))


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
    with pytest.raises(ValueError, match="does not match"):
        det("anything")


def test_load_linear_head_roundtrip_and_validation(tmp_path):
    good = tmp_path / "head.json"
    good.write_text(json.dumps({"coef": [0.1, 0.2], "intercept": -0.3, "calibration": None}))
    head = classify.load_linear_head(good)
    assert head["coef"] == [0.1, 0.2]
    assert head["intercept"] == -0.3

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"intercept": 0.0}))  # missing coef
    with pytest.raises(ValueError, match="coef"):
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
