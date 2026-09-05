"""NDJSON scoring sidecar — the editor-facing seam over a detector (D20, D44).

`ai-style serve` keeps a detector resident and answers one JSON object per
line over stdin/stdout:

    -> {"id": 1, "op": "info"}                       <- {"id": 1, "meta": {...}}
    -> {"id": 2, "op": "score", "texts": [...]}      <- {"id": 2, "scores": [...]}
    -> {"id": 5, "op": "segment", "text": ..., "min_chars": 80}
    <- {"id": 5, "segments": [{"start", "end", "kind", "context"}, ...]}
    -> {"id": 3, "op": "rewrite", "text": ..., "context": ..., "best_of": 4}
    <- {"id": 3, "input_score": 0.52, "candidates": [{"text", "score",
                 "anchors_lost", "anchors_gained"}, ...]}
    -> {"id": 4, "op": "prefer", "decision": {...}}  <- {"id": 4, "ok": true}

`segment` answers from `stylebot.markdown` under the sidecar's `div_is_prose`.
`rewrite` and `prefer` arm only when their seam is injected; `info` is the
ready handshake. One request in flight at a time; an error never kills the
loop; EOF shuts down; all ML is injected. `serve_sidecar` is the command body
as a library, `preference_recorder` its `prefer` sink (D78).
"""

from __future__ import annotations

import bisect
import json
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import IO, TYPE_CHECKING

from stylebot import markdown
from stylebot.classify import detect_many

if TYPE_CHECKING:
    from stylebot.backends import BackendOpts
    from stylebot.infer import Styler


def _utf16_mapper(text: str) -> Callable[[int], int]:
    """Map a Python code-point offset into `text` to a UTF-16 code-unit offset,
    which is what an editor (VS Code, LSP) indexes by; identity when the
    document has no astral characters."""
    astral = [i for i, ch in enumerate(text) if ch > "\uffff"]
    if not astral:
        return lambda offset: offset
    return lambda offset: offset + bisect.bisect_left(astral, offset)


def prose_paragraphs(
    text: str,
    *,
    min_chars: int = 0,
    div_is_prose: markdown.DivIsProse | None = None,
) -> list[dict]:
    """The scoreable prose paragraphs of a markdown document, with offsets.

    A paragraph is a maximal run of lines `markdown.line_kinds` calls prose;
    `context` is the nearest heading above; frontmatter is dropped and
    `min_chars` drops short paragraphs. Offsets are UTF-16 code units into the
    original document, so the response carries no text.
    """
    from stylebot.infer import split_frontmatter

    frontmatter, body = split_frontmatter(text)
    base = len(frontmatter)
    to_utf16 = _utf16_mapper(text)

    lines = body.split("\n")
    kinds = markdown.line_kinds(body, headings=True, div_is_prose=div_is_prose)
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1

    out: list[dict] = []
    context: str | None = None
    run: list[int] = []

    def flush() -> None:
        if not run:
            return
        start = base + starts[run[0]]
        end = base + starts[run[-1]] + len(lines[run[-1]])
        if end - start >= min_chars:
            out.append(
                {
                    "start": to_utf16(start),
                    "end": to_utf16(end),
                    "kind": markdown.PROSE,
                    "context": context,
                }
            )
        run.clear()

    for i, line in enumerate(lines):
        if kinds[i] is None and line.strip():
            run.append(i)
            continue
        flush()
        if kinds[i] == markdown.HEADING:
            context = line.strip()
    flush()
    return out


def handle_request(
    request: dict,
    detector: Callable[[str], dict],
    meta: dict,
    *,
    rewriter: Callable[..., dict] | None = None,
    prefer: Callable[[dict], None] | None = None,
    div_is_prose: markdown.DivIsProse | None = None,
) -> dict:
    """Compute the response for one parsed request object; pure.

    Unknown ops and detector failures come back as an ``error`` response with
    the ``id`` echoed. `div_is_prose` applies to the `segment` op and must be
    the object the injected `rewriter` closed over.
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
            # One `encode` for the whole request, not one per text (D59).
            return {"id": rid, "scores": [r["score"] for r in detect_many(detector, texts)]}
        if op == "segment":
            text = request.get("text")
            if not isinstance(text, str):
                return {"id": rid, "error": "'text' must be a string"}
            return {
                "id": rid,
                "segments": prose_paragraphs(
                    text,
                    min_chars=int(request.get("min_chars") or 0),
                    div_is_prose=div_is_prose,
                ),
            }
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
            "error": (
                f"unknown op {op!r} (expected 'score', 'segment', 'info', 'rewrite', or 'prefer')"
            ),
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
    div_is_prose: markdown.DivIsProse | None = None,
) -> None:
    """Serve NDJSON requests until EOF on `stdin`: one flushed line per
    response (the client blocks on it); blank lines ignored; an unparseable
    line answers ``id: null``."""
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
            response = handle_request(
                request,
                detector,
                meta,
                rewriter=rewriter,
                prefer=prefer,
                div_is_prose=div_is_prose,
            )
        stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        stdout.flush()


def preference_recorder(
    path: str | Path, *, run_id: str | None = None, default_chooser: str | None = None
) -> Callable[[dict], None]:
    """A `prefer` sink appending `{ts, run_id, chooser, **decision}` to a JSONL;
    `chooser` is the decision's, else `default_chooser`."""
    from stylebot.jsonl import append_jsonl
    from stylebot.provenance import now_iso

    def prefer(decision: dict) -> None:
        rest = dict(decision)
        chooser = rest.pop("chooser", default_chooser)
        append_jsonl(path, {"ts": now_iso(), "run_id": run_id, "chooser": chooser, **rest})

    return prefer


def serve_sidecar(
    backend: BackendOpts,
    *,
    detector: Callable[[str], dict],
    meta: dict,
    make_styler: Callable[[], Styler] | None,
    prefer: Callable[[dict], None] | None = None,
    div_is_prose: markdown.DivIsProse | None = None,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
) -> None:
    """The sidecar: `serve_loop` with the `rewrite` op armed from `make_styler`.

    The styler is built lazily on the first `rewrite`, so a backend that dials
    out cannot take scoring down at editor start; `None` leaves the op
    unarmed. The reranker is `infer.detector_scorer(detector)`; the diet is
    `backend`'s; `div_is_prose` reaches the `segment` op only.
    """
    from stylebot import infer

    rewriter = None
    if make_styler is not None:
        styler = cache(make_styler)
        scorer = infer.detector_scorer(detector)
        chunk_chars, inline_headings = backend.chunk_chars(), backend.headings()

        def rewrite_op(text: str, *, context: str | None = None, best_of: int = 4) -> dict:
            return infer.rewrite_candidates(
                text,
                styler(),
                context=context,
                best_of=best_of,
                scorer=scorer,
                max_chunk_chars=chunk_chars,
                inline_headings=inline_headings,
                system=backend.system(),
            )

        rewriter = rewrite_op

    serve_loop(
        detector,
        stdin if stdin is not None else sys.stdin,
        stdout if stdout is not None else sys.stdout,
        meta=meta,
        rewriter=rewriter,
        prefer=prefer,
        div_is_prose=div_is_prose,
    )
