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


# --- `EvalOpts` and `report_eval`: the eval surface as a library (D78) ------


def test_eval_opts_defaults_are_the_library_functions_defaults():
    import inspect

    from stylebot import report, scoring
    from stylebot.verdict import EVAL_PARAMS, VERDICT_PARAMS, DocumentOpts, EvalOpts, VerdictOpts

    def default_of(fn, name):
        return inspect.signature(fn).parameters[name].default

    opts = EvalOpts()
    assert opts.judge_model == ev.DEFAULT_JUDGE_MODEL == default_of(ev.openrouter_judge, "model")
    assert opts.judge_model == default_of(ev.openrouter_judge_async, "model")
    assert opts.max_workers == scoring.DEFAULT_MAX_WORKERS
    # The typed keywords live on the async entry point; the sync one forwards.
    assert opts.max_workers == default_of(scoring.score_pairs_file_async, "max_workers")
    assert opts.report_max_rows == report.DEFAULT_MAX_ROWS
    assert opts.report_max_rows == default_of(report.render_scores_report, "max_rows")
    assert opts.facet_by == report.DEFAULT_FACET_BY
    assert opts.facet_by == default_of(report.render_scores_report, "facet_by")
    assert {"eval": EvalOpts, "verdict": VerdictOpts, "documents": DocumentOpts} == EVAL_PARAMS
    assert {"verdict": VerdictOpts, "documents": DocumentOpts} == VERDICT_PARAMS
    assert EvalOpts(pairs_path="c/pairs.jsonl").scores_path().as_posix() == "c/scores.jsonl"
    assert EvalOpts(fields=("output",)).score_fields() == ("output",)
    assert EvalOpts().score_fields() == ("slop", "target")


def _scores_file(path, ids_and_values):
    path.write_text("".join(json.dumps(r) + "\n" for r in _rows(ids_and_values)), encoding="utf-8")
    return path


def test_report_eval_renders_the_verdict_and_the_report(tmp_path):
    from stylebot.verdict import EvalOpts, VerdictOpts, report_eval

    a = _scores_file(tmp_path / "a.jsonl", _PAIRS)
    b = _scores_file(tmp_path / "b.jsonl", [("r1", 0.5), ("r2", 0.6), ("r3", 0.7)])
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("", encoding="utf-8")
    lines: list[tuple[str, str]] = []
    opts = EvalOpts(pairs_path=pairs, report_path=tmp_path / "scores.html", report_max_rows=0)

    report_eval(
        a,
        pairs,
        ("output",),
        VerdictOpts(compare_path=b),
        opts,
        emit=lambda message, /, *, level="info": lines.append((level, message)),
    )

    messages = [m for _, m in lines]
    assert messages[0] == ""
    assert messages[1].startswith("verdict — detector on 'output', real pairs, clustered by source")
    assert any(f"a = {a}" in m for m in messages)
    assert messages[-1] == f"wrote report -> {opts.report_path}"
    assert opts.report_path.exists()
    assert not any(level == "warn" for level, _ in lines)
