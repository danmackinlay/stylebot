"""NDJSON scoring sidecar — the editor-facing seam over a detector.

`ai-style serve` keeps a detector (typically the trained voice classifier,
`classify.sklearn_detector`) resident and scores batches of texts over
stdin/stdout, one JSON object per line each way. The expensive part — the
embedding model — loads once at startup (~5 s for StyleDistance); after that a
whole document scores in tens of milliseconds, which is what makes a live
editor marker (see `_plans/vscode-marker.md`) viable.

The protocol is deliberately minimal (NDJSON, not LSP or JSON-RPC framing):

    -> {"id": 1, "op": "info"}
    <- {"id": 1, "meta": {...artifact meta.json...}}
    -> {"id": 2, "op": "score", "texts": ["para one", "para two"]}
    <- {"id": 2, "scores": [0.72, 0.31]}

Two further ops arm only when their seam is injected; an unarmed op errors
politely. They are the editor rewrite loop (`_plans/decisions.md` D20):

    -> {"id": 3, "op": "rewrite", "text": "...", "context": "## Heading",
        "best_of": 4}
    <- {"id": 3, "input_score": 0.52,
        "candidates": [{"text": ..., "score": 0.31, "anchors_lost": []}, ...]}
    -> {"id": 4, "op": "prefer", "decision": {...}}
    <- {"id": 4, "ok": true}

`rewrite` delegates to an injected `rewriter(text, context, best_of)` (typically
`stylebot.infer.rewrite_candidates` closed over a styler + scorer); `prefer`
hands the client's decision dict to an injected sink (the preference-data seam —
the caller owns the schema).

`info` doubles as the client's ready handshake: the first response arrives
only after the model has loaded. Errors never kill the loop — a malformed or
failing request gets `{"id": ..., "error": "..."}` and the server keeps
reading. EOF on stdin is the shutdown signal.

Like `stylebot.classify`, this module is ML-dependency-free: the detector,
rewriter, and prefer sink are injected, so tests drive `serve_loop` with stubs.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import IO


def handle_request(
    request: dict,
    detector: Callable[[str], dict],
    meta: dict,
    *,
    rewriter: Callable[..., dict] | None = None,
    prefer: Callable[[dict], None] | None = None,
) -> dict:
    """Compute the response for one parsed request object (pure, testable).

    Unknown ops and detector failures come back as an ``error`` response with
    the request's ``id`` echoed, so the client can reject the matching promise
    instead of stalling.
    """
    rid = request.get("id")
    op = request.get("op")
    try:
        if op == "info":
            return {"id": rid, "meta": meta}
        if op == "score":
            texts = request.get("texts")
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return {"id": rid, "error": "'texts' must be a list of strings"}
            return {"id": rid, "scores": [detector(t)["score"] for t in texts]}
        if op == "rewrite":
            if rewriter is None:
                return {"id": rid, "error": "serve started without a styler — no rewrite"}
            text = request.get("text")
            if not isinstance(text, str) or not text.strip():
                return {"id": rid, "error": "'text' must be a non-empty string"}
            result = rewriter(
                text,
                context=request.get("context"),
                best_of=int(request.get("best_of") or 4),
            )
            return {"id": rid, **result}
        if op == "prefer":
            if prefer is None:
                return {"id": rid, "error": "serve started without a preferences sink"}
            decision = request.get("decision")
            if not isinstance(decision, dict):
                return {"id": rid, "error": "'decision' must be an object"}
            prefer(decision)
            return {"id": rid, "ok": True}
        return {
            "id": rid,
            "error": f"unknown op {op!r} (expected 'score', 'info', 'rewrite', or 'prefer')",
        }
    except Exception as exc:  # a bad text must not kill the server
        return {"id": rid, "error": f"{type(exc).__name__}: {exc}"}


def serve_loop(
    detector: Callable[[str], dict],
    stdin: IO[str],
    stdout: IO[str],
    *,
    meta: dict | None = None,
    rewriter: Callable[..., dict] | None = None,
    prefer: Callable[[dict], None] | None = None,
) -> None:
    """Serve NDJSON requests until EOF on `stdin`.

    Every response is a single line, flushed immediately — the client blocks
    on the next line, so buffering here would deadlock it. Blank lines are
    ignored; unparseable lines respond with ``id: null`` (the request id is
    unrecoverable).
    """
    meta = meta or {}
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
        except ValueError as exc:
            response = {"id": None, "error": f"bad request line: {exc}"}
        else:
            response = handle_request(request, detector, meta, rewriter=rewriter, prefer=prefer)
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()
