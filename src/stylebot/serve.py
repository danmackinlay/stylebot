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
    -> {"id": 5, "op": "segment", "text": "<whole document>", "min_chars": 80}
    <- {"id": 5, "segments": [{"start": 0, "end": 42, "kind": "prose",
                               "context": "## A heading"}, ...]}

`segment` is what stops the client owning a segmenter of its own. It answers
"which spans of this document are scoreable prose, and what heading is each
under?" from `stylebot.markdown`, under the `div_is_prose` policy the sidecar
was started with — so a `.callout-note` paragraph is prose here exactly when it
is a training target, which no client-side port could know.

Two further ops arm only when their seam is injected; an unarmed op errors
politely. They are the editor rewrite loop (`_plans/decisions.md` D20):

    -> {"id": 3, "op": "rewrite", "text": "...", "context": "## Heading",
        "best_of": 4}
    <- {"id": 3, "input_score": 0.52,
        "candidates": [{"text": ..., "score": 0.31, "anchors_lost": [],
                        "anchors_gained": []}, ...]}
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

import bisect
import json
from collections.abc import Callable
from typing import IO

from stylebot import markdown
from stylebot.eval import detect_many


def _utf16_mapper(text: str) -> Callable[[int], int]:
    """Map a Python code-point offset into `text` to a UTF-16 code-unit offset.

    The `segment` op's offsets are consumed by an editor, and every editor this
    protocol will ever meet (VS Code, and the LSP spec it follows) indexes
    documents in UTF-16 code units. Python indexes code points, so the two agree
    until the document contains an astral character — one emoji or one
    ``\\U0001d53c`` in a paragraph and every decoration below it lands one
    character short. Converting here rather than in the client keeps the
    client's arithmetic to none at all: `positionAt(start)` just works.

    Fast path: a document with no astral characters needs no conversion, which
    is nearly all of them.
    """
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

    A paragraph is a maximal run of adjacent lines that `markdown.line_kinds`
    calls prose — so blank lines, protected blocks (fences, display math,
    blockquotes, divs) and heading lines all end one. `context` is the nearest
    heading line above, by the same `markdown.ATX_HEADING_RE` training uses.
    Leading YAML frontmatter is dropped; `min_chars` drops short paragraphs,
    which score noisily (`_plans/vscode-marker.md`).

    Offsets are UTF-16 code units into the ORIGINAL document, frontmatter
    included — see `_utf16_mapper`. Slicing the document by them reproduces the
    paragraph, which is why the response carries no text: the client already
    holds the document it sent.

    This is the composition the VS Code marker used to re-implement in
    TypeScript (`vscode-voice-marker/src/segment.ts`, deleted 2026-09-03). It
    lives here rather than in `stylebot.markdown` because it needs two things
    that module refuses on purpose — frontmatter knowledge, and a paragraph
    granularity that is the *editor's* unit rather than the chunk budget's.
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
    """Compute the response for one parsed request object (pure, testable).

    Unknown ops and detector failures come back as an ``error`` response with
    the request's ``id`` echoed, so the client can reject the matching promise
    instead of stalling.

    `div_is_prose` is the caller's div policy, applied to the `segment` op — it
    must be the SAME object the injected `rewriter` closed over, or the marker
    will offer to rewrite paragraphs the styler then refuses.
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
            # One `encode` for the whole request, not one per text: the
            # marker sends a document's paragraphs as a single `score` op,
            # and batch-of-one encodes cost 1.56x (`_plans/decisions.md` D57).
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
