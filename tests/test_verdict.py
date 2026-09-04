"""`stylebot.verdict` — the paired comparison refuses a circular instrument.

Best-of reranking lets the detector pick the output; scoring that output with
the same detector measures agreement with itself (C5, D39). The output row
carries `meta.reranker`, and a detector verdict on such rows is refused.
"""

from __future__ import annotations

import json

import pytest

from stylebot import classify
from stylebot import eval as ev
from stylebot.errors import ConfigError

_HEAD = {
    "name": "voice-clf",
    "embed_model": "stub-embed",
    "git_sha": "abc",
    "trained_at": "2026-01-01",
    "n_pairs": 10,
}


def _rows(ids_and_values, *, reranker=None, judge=3.0):
    """Score rows as `scoring._score_record` writes them, one head throughout."""
    return [
        {
            "id": rid,
            "meta": {"source": "post/a.qmd", "synthetic": False, "reranker": reranker},
            "scores": {
                "output": {
                    "detector": {"score": value, "configured": True, **_HEAD},
                    "judge": {"score": judge},
                }
            },
        }
        for rid, value in ids_and_values
    ]


_PAIRS = [("r1", 0.2), ("r2", 0.3), ("r3", 0.4)]


def test_detector_verdict_refuses_rows_its_own_reranker_chose():
    a = _rows(_PAIRS, reranker=dict(_HEAD))
    b = _rows(_PAIRS)
    with pytest.raises(ConfigError, match=r"--compare-signal judge"):
        ev.compare_scores(a, b, signal="detector")
    # Either side trips it: the circularity is per file, not per pair of files.
    with pytest.raises(ConfigError, match=r"--compare-signal judge"):
        ev.compare_scores(b, a)


def test_judge_verdict_is_allowed_on_rows_the_reranker_chose():
    a = _rows(_PAIRS, reranker=dict(_HEAD), judge=4.0)
    b = _rows(_PAIRS, judge=3.0)
    verdict = ev.compare_scores(a, b, signal="judge")
    assert verdict["signal"] == "judge"
    assert verdict["n_shared_ids"] == 3


def test_reranker_null_or_another_head_does_not_trip_the_refusal():
    other = {**_HEAD, "git_sha": "def", "trained_at": "2026-02-02"}
    unreranked = _rows(_PAIRS)
    reranked_elsewhere = _rows(_PAIRS, reranker=other)
    assert ev.compare_scores(unreranked, unreranked)["n_shared_ids"] == 3
    assert ev.compare_scores(reranked_elsewhere, unreranked)["n_shared_ids"] == 3


def test_reranker_identity_is_compared_on_the_d39_keys_only():
    # A stamp carrying extra keys (a future field, a score) still matches.
    stamped = {**_HEAD, "score": 0.1, "extra": "x"}
    with pytest.raises(ConfigError, match=r"reranked|picked"):
        ev.compare_scores(_rows(_PAIRS, reranker=stamped), _rows(_PAIRS))


def _stub_detector(*, git_sha: str, score: float = 0.4):
    return classify.embedding_classifier_detector(
        embed_fn=lambda prose: [score],
        head={"coef": [1.0], "intercept": 0.0},
        meta={
            "embed_model": "stub-embed",
            "git_sha": git_sha,
            "trained_at": "2026-01-01",
            "n_pairs": 10,
        },
    )


def test_a_scored_output_file_carries_the_reranker_to_the_verdict(tmp_path):
    from stylebot.ai_core import STYLE_SYSTEM

    detector = _stub_detector(git_sha="abc")
    out = tmp_path / "outputs.jsonl"
    row = {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": "slop"},
            {"role": "assistant", "content": "dan"},
        ],
        "output": "picked by the head",
        "meta": {
            "source": "post/a.qmd",
            "capture_id": "c1",
            "chunk_index": 0,
            "reranker": ev.detector_identity(detector),
        },
    }
    out.write_text(json.dumps(row) + "\n", encoding="utf-8")
    scores = tmp_path / "scores.jsonl"
    ev.score_pairs_file(out, scores, fields=("output",), detector=detector, prune_orphans=False)
    with pytest.raises(ConfigError, match=r"--compare-signal judge"):
        ev.compare_scores(scores, scores)
