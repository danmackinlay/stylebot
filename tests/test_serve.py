"""Tests for the NDJSON scoring sidecar — NO ML deps.

`serve_loop` takes an injected detector, so a stub stands in for the real
voice classifier; the tests drive the loop over in-memory streams exactly the
way an editor client drives the child process's pipes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from click.testing import CliRunner

from stylebot import backends, classify, markdown, serve
from stylebot.bin import ai_style
from stylebot.errors import RunError

PARITY = Path(__file__).parent / "fixtures" / "protection_parity.qmd"


def stub_detector(prose: str) -> dict:
    # Deterministic and inspectable: "slop" texts score high, others low.
    return {"score": 0.9 if "slop" in prose else 0.1, "p_dan": None, "configured": True}


def run_lines(*requests: object, rewriter=None, prefer=None, div_is_prose=None) -> list[dict]:
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
        div_is_prose=div_is_prose,
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


# ---------------------------------------------------------------------------
# The `segment` op — the marker's segmentation, which used to be a TypeScript
# port in `vscode-voice-marker/src/segment.ts` (deleted 2026-09-03, W4).
# ---------------------------------------------------------------------------


def callout_is_prose(attrs: str) -> bool:
    """A stand-in for the blog's `training_targets.div_is_prose` policy."""
    return ".callout-" in attrs


def markdown_expectation(text: str, div_is_prose=None) -> list[str]:
    """The marker's prose paragraphs, re-derived from `stylebot.markdown`.

    Deliberately NOT the route `prose_paragraphs` takes: it walks a per-character
    mask built from `protected_spans` where the implementation reads
    `line_kinds`. The two agree only if the module says one thing, which is the
    whole point of the op existing — a second implementation is exactly what W4
    deleted, so the test may not contain one either.
    """
    from stylebot.infer import split_frontmatter

    _frontmatter, body = split_frontmatter(text)
    blocked = bytearray(len(body) + 1)
    for start, end, _kind in markdown.protected_spans(
        body, block_only=True, div_is_prose=div_is_prose
    ):
        for i in range(start, end):
            blocked[i] = 1

    paragraphs: list[str] = []
    run: list[str] = []
    pos = 0
    for line in body.split("\n"):
        # A line is protected when any span reaches into it, its own newline
        # included — `line_kinds`' ownership rule, spelled out.
        protected = any(blocked[pos : pos + len(line) + 1])
        if line.strip() and not protected and not markdown.is_heading_line(line):
            run.append(line)
        elif run:
            paragraphs.append("\n".join(run))
            run = []
        pos += len(line) + 1
    if run:
        paragraphs.append("\n".join(run))
    return paragraphs


def segment_op(text: str, *, min_chars: int = 0, div_is_prose=None) -> list[dict]:
    (resp,) = run_lines(
        {"id": 1, "op": "segment", "text": text, "min_chars": min_chars},
        div_is_prose=div_is_prose,
    )
    assert "error" not in resp, resp
    return resp["segments"]


def test_segment_op_matches_stylebot_markdown_on_the_parity_fixture():
    text = PARITY.read_text(encoding="utf-8")
    for policy in (None, callout_is_prose):
        segments = segment_op(text, div_is_prose=policy)
        assert [text[s["start"] : s["end"]] for s in segments] == markdown_expectation(
            text, div_is_prose=policy
        )


def test_segment_offsets_slice_back_to_the_paragraph():
    text = PARITY.read_text(encoding="utf-8")
    for seg in segment_op(text):
        sliced = text[seg["start"] : seg["end"]]
        assert sliced == sliced.strip()
        assert seg["kind"] == markdown.PROSE


def test_segment_frees_callout_prose_only_under_the_policy():
    text = PARITY.read_text(encoding="utf-8")
    callout = "Callout prose paragraph two, still the author's voice."
    assert callout not in [text[s["start"] : s["end"]] for s in segment_op(text)]
    freed = [text[s["start"] : s["end"]] for s in segment_op(text, div_is_prose=callout_is_prose)]
    assert callout in freed
    # The `:::` fence lines stay protected even when their contents are prose.
    assert not any(p.startswith(":::") and p.endswith(":::") for p in freed)


def test_segment_protects_an_unclosed_fence_to_end_of_document():
    # The live case: notebook/vs_code.qmd ends on a bare ``` with no closer, and
    # the deleted TypeScript port scored it as the tail of the paragraph above.
    text = "Prose before the runaway fence.\n```\nnot prose at all\nstill not prose\n"
    assert [text[s["start"] : s["end"]] for s in segment_op(text)] == [
        "Prose before the runaway fence."
    ]


def test_segment_splits_a_heading_glued_to_prose():
    # 28 of the 29 blog files where the port disagreed were this: the port kept
    # heading and prose in one paragraph, then dropped the whole thing when the
    # heading came first — losing the prose entirely.
    text = "### A heading\nProse directly under it.\nMore of the same paragraph.\n"
    segments = segment_op(text)
    assert [text[s["start"] : s["end"]] for s in segments] == [
        "Prose directly under it.\nMore of the same paragraph."
    ]
    assert segments[0]["context"] == "### A heading"


def test_segment_reports_the_nearest_heading_as_context():
    text = "## One\n\nAlpha paragraph.\n\n## Two\n\nBeta paragraph.\n"
    assert [(text[s["start"] : s["end"]], s["context"]) for s in segment_op(text)] == [
        ("Alpha paragraph.", "## One"),
        ("Beta paragraph.", "## Two"),
    ]


def test_segment_drops_frontmatter_but_keeps_document_offsets():
    text = "---\ntitle: A post\n---\n\nThe body paragraph.\n"
    (seg,) = segment_op(text)
    assert text[seg["start"] : seg["end"]] == "The body paragraph."
    assert seg["context"] is None


def test_segment_min_chars_drops_short_paragraphs_only():
    text = "Short.\n\nA paragraph long enough to clear the minimum characters bar.\n"
    assert len(segment_op(text)) == 2
    assert [text[s["start"] : s["end"]] for s in segment_op(text, min_chars=20)] == [
        "A paragraph long enough to clear the minimum characters bar."
    ]


def test_segment_offsets_are_utf16_code_units():
    # VS Code (and LSP) index documents in UTF-16 code units; Python indexes code
    # points. One astral character before a paragraph and every decoration below
    # it lands a character short — so the conversion happens here, not client-side.
    text = "An emoji 🚧 sits above.\n\nThe paragraph to decorate.\n"
    (_first, second) = segment_op(text)
    utf16 = text.encode("utf-16-le")
    assert utf16[second["start"] * 2 : second["end"] * 2].decode("utf-16-le") == (
        "The paragraph to decorate."
    )
    assert second["start"] != text.index("The paragraph")  # the naive answer differs


def test_segment_rejects_a_non_string_and_survives():
    responses = run_lines(
        {"id": 1, "op": "segment", "text": ["not", "a", "string"]},
        {"id": 2, "op": "segment", "text": "Still alive and segmenting."},
    )
    assert responses[0] == {"id": 1, "error": "'text' must be a string"}
    assert len(responses[1]["segments"]) == 1


def test_score_op_uses_the_detector_batch_path_when_there_is_one():
    """The marker sends a document's paragraphs as one `score` op — so send them
    to the backbone as one encode, not one per paragraph (D57)."""
    sizes: list[int] = []

    def detect(prose: str) -> dict:
        sizes.append(1)
        return {"score": 0.1, "configured": True}

    def batch_detect(texts):
        texts = list(texts)
        sizes.append(len(texts))
        return [{"score": 0.5, "configured": True} for _ in texts]

    detect.batch_detect = batch_detect
    resp = serve.handle_request(
        {"id": 1, "op": "score", "texts": ["one", "two", "three"]}, detect, {}
    )
    assert resp == {"id": 1, "scores": [0.5, 0.5, 0.5]}
    assert sizes == [3]


def test_score_op_still_works_for_a_detector_with_no_batch_path():
    def detect(prose: str) -> dict:
        return {"score": 0.9 if "slop" in prose else 0.1, "configured": True}

    resp = serve.handle_request({"id": 2, "op": "score", "texts": ["slop", "ok"]}, detect, {})
    assert resp == {"id": 2, "scores": [0.9, 0.1]}


def test_serve_answers_a_failed_styler_build_and_lives(monkeypatch, tmp_path):
    """`handle_request` catches `Exception`, so `build_styler`'s `RunError`
    reaches the client under its own D47 name with no re-raise in `serve_cmd`."""
    monkeypatch.setattr(classify, "load_artifact_meta", lambda p: {"name": "stub"})
    monkeypatch.setattr(classify, "sklearn_detector", lambda p: stub_detector)

    def unready(opts):
        raise RunError("TINKER_API_KEY is not set")

    monkeypatch.setattr(backends, "build_styler", unready)
    requests = [
        {"id": 1, "op": "rewrite", "text": "some prose"},
        {"id": 2, "op": "score", "texts": ["still slop"]},
    ]
    result = CliRunner().invoke(
        ai_style.main,
        ["serve", "--detector-model", str(tmp_path), "--backend", "tinker"]
        + ["--sampler-path", "s", "--base-model", "b"],
        input="".join(json.dumps(r) + "\n" for r in requests),
    )
    assert result.exit_code == 0, result.output
    replies = [json.loads(line) for line in result.output.splitlines()]
    assert replies == [
        {"id": 1, "error": "RunError: TINKER_API_KEY is not set"},
        {"id": 2, "scores": [0.9]},
    ]
