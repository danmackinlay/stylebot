"""Tests for the NDJSON scoring sidecar — NO ML deps.

`serve_loop` takes an injected detector, so a stub stands in for the real
voice classifier; the tests drive the loop over in-memory streams exactly the
way an editor client drives the child process's pipes.
"""

from __future__ import annotations

import io
import json

from stylebot import serve


def stub_detector(prose: str) -> dict:
    # Deterministic and inspectable: "slop" texts score high, others low.
    return {"score": 0.9 if "slop" in prose else 0.1, "p_dan": None, "configured": True}


def run_lines(*requests: object, rewriter=None, prefer=None) -> list[dict]:
    """Feed newline-joined requests through serve_loop, return parsed responses."""
    lines = [r if isinstance(r, str) else json.dumps(r) for r in requests]
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    serve.serve_loop(
        stub_detector,
        stdin,
        stdout,
        meta={"name": "stub-clf", "embed_dim": 2},
        rewriter=rewriter,
        prefer=prefer,
    )
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_score_roundtrip_preserves_order_and_id():
    (resp,) = run_lines({"id": 7, "op": "score", "texts": ["pure slop here", "plain prose"]})
    assert resp == {"id": 7, "scores": [0.9, 0.1]}


def test_info_returns_meta_as_ready_handshake():
    (resp,) = run_lines({"id": 1, "op": "info"})
    assert resp == {"id": 1, "meta": {"name": "stub-clf", "embed_dim": 2}}


def test_empty_texts_is_valid_and_cheap():
    (resp,) = run_lines({"id": 2, "op": "score", "texts": []})
    assert resp == {"id": 2, "scores": []}


def test_malformed_lines_error_without_killing_the_loop():
    responses = run_lines(
        "this is not json",
        {"id": 3, "op": "score", "texts": "not-a-list"},
        {"id": 4, "op": "nonsense"},
        "",  # blank line: ignored, no response
        {"id": 5, "op": "score", "texts": ["still alive slop"]},
    )
    assert len(responses) == 4
    assert responses[0]["id"] is None and "bad request line" in responses[0]["error"]
    assert responses[1] == {"id": 3, "error": "'texts' must be a list of strings"}
    assert responses[2]["id"] == 4 and "unknown op" in responses[2]["error"]
    assert responses[3] == {"id": 5, "scores": [0.9]}


def test_rewrite_and_prefer_unarmed_error_politely():
    responses = run_lines(
        {"id": 1, "op": "rewrite", "text": "some prose"},
        {"id": 2, "op": "prefer", "decision": {"chosen": 0}},
    )
    assert "without a styler" in responses[0]["error"]
    assert "without a preferences sink" in responses[1]["error"]


def test_rewrite_delegates_and_flattens_result():
    calls = []

    def rewriter(text, *, context=None, best_of=4):
        calls.append((text, context, best_of))
        return {
            "input_score": 0.5,
            "candidates": [{"text": "better", "score": 0.2, "anchors_lost": []}],
        }

    (resp,) = run_lines(
        {"id": 3, "op": "rewrite", "text": "meh prose", "context": "## H", "best_of": 2},
        rewriter=rewriter,
    )
    assert calls == [("meh prose", "## H", 2)]
    assert resp == {
        "id": 3,
        "input_score": 0.5,
        "candidates": [{"text": "better", "score": 0.2, "anchors_lost": []}],
    }


def test_rewrite_rejects_blank_text_and_survives_rewriter_errors():
    def rewriter(text, *, context=None, best_of=4):
        raise ValueError("selection contains a fence — select prose only")

    responses = run_lines(
        {"id": 4, "op": "rewrite", "text": "   "},
        {"id": 5, "op": "rewrite", "text": "prose"},
        {"id": 6, "op": "score", "texts": ["still alive slop"]},
        rewriter=rewriter,
    )
    assert "'text' must be a non-empty string" in responses[0]["error"]
    assert "select prose only" in responses[1]["error"]
    assert responses[2] == {"id": 6, "scores": [0.9]}  # loop survived


def test_prefer_hands_decision_to_sink():
    seen = []
    responses = run_lines(
        {"id": 7, "op": "prefer", "decision": {"chosen": 1, "kept_input": False}},
        {"id": 8, "op": "prefer", "decision": "not-an-object"},
        prefer=seen.append,
    )
    assert responses[0] == {"id": 7, "ok": True}
    assert seen == [{"chosen": 1, "kept_input": False}]
    assert "'decision' must be an object" in responses[1]["error"]


def test_detector_exception_becomes_error_response():
    def broken(prose: str) -> dict:
        raise RuntimeError("embedder fell over")

    stdin = io.StringIO(json.dumps({"id": 9, "op": "score", "texts": ["x"]}) + "\n")
    stdout = io.StringIO()
    serve.serve_loop(broken, stdin, stdout)
    (resp,) = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert resp["id"] == 9
    assert "RuntimeError: embedder fell over" in resp["error"]
