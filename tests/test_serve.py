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

FIXTURE = Path(__file__).parent / "fixtures" / "hard" / "segmenter_conformance.qmd"


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
    assert resp == {"id": 7, "scores": [0.9, 0.1], "detector": "stub-clf"}


def test_info_returns_meta_as_ready_handshake():
    (resp,) = run_lines({"id": 1, "op": "info"})
    assert resp == {"id": 1, "meta": {"name": "stub-clf", "embed_dim": 2}}


def test_empty_texts_is_valid_and_cheap():
    (resp,) = run_lines({"id": 2, "op": "score", "texts": []})
    assert resp == {"id": 2, "scores": [], "detector": "stub-clf"}


def test_malformed_lines_error_without_killing_the_loop():
    responses = run_lines(
        "this is not json",
        {"id": 3, "op": "score", "texts": "not-a-list"},
        {"id": 4, "op": "nonsense"},
        "",  # blank line: ignored, no response
        {"id": 5, "op": "segment", "text": ["not", "a", "string"]},
        {"id": 6, "op": "score", "texts": ["still alive slop"]},
    )
    assert len(responses) == 5
    assert responses[0]["id"] is None and "bad request line" in responses[0]["error"]
    assert responses[1] == {"id": 3, "error": "'texts' must be a list of strings"}
    assert responses[2]["id"] == 4 and "unknown op" in responses[2]["error"]
    assert responses[3] == {"id": 5, "error": "'text' must be a string"}
    assert responses[4] == {"id": 6, "scores": [0.9], "detector": "stub-clf"}


def test_rewrite_and_prefer_unarmed_error_politely():
    responses = run_lines(
        {"id": 1, "op": "rewrite", "text": "some prose"},
        {"id": 2, "op": "prefer", "decision": {"chosen": 0}},
    )
    assert "without a styler" in responses[0]["error"]
    assert "without a preferences sink" in responses[1]["error"]


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
    assert responses[2] == {"id": 6, "scores": [0.9], "detector": "stub-clf"}  # loop survived


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


def test_segment_op_matches_stylebot_markdown_on_the_hard_fixture():
    text = FIXTURE.read_text(encoding="utf-8")
    for policy in (None, callout_is_prose):
        segments = segment_op(text, div_is_prose=policy)
        assert [text[s["start"] : s["end"]] for s in segments] == markdown_expectation(
            text, div_is_prose=policy
        )


def test_segment_offsets_slice_back_to_the_paragraph():
    text = FIXTURE.read_text(encoding="utf-8")
    for seg in segment_op(text):
        sliced = text[seg["start"] : seg["end"]]
        assert sliced == sliced.strip()
        assert seg["kind"] == markdown.PROSE


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
    assert resp == {"id": 1, "scores": [0.5, 0.5, 0.5], "detector": "detector"}
    assert sizes == [3]


def test_score_op_picks_the_named_detector_and_reports_its_stats():
    """Several instruments armed at once (D106): `score` picks one by name,
    else the first; `stats` rides along when the detector keeps any; a name
    nobody armed is an error that lists who is."""

    def hosted(prose: str) -> dict:
        return {"score": 0.7, "configured": True}

    hosted.stats = {"requests": 3, "chars_sent": 1200}
    detectors = {"stub-clf": stub_detector, "sapling-x": hosted}
    meta = serve.sidecar_meta(detectors, {"embed_model": "e"})
    assert meta == {
        "embed_model": "e",
        "name": "stub-clf",
        "detectors": ["stub-clf", "sapling-x"],
        "detector": "stub-clf",
    }
    stdin = io.StringIO(
        "".join(
            json.dumps(r) + "\n"
            for r in (
                {"id": 1, "op": "score", "texts": ["slop"]},
                {"id": 2, "op": "score", "texts": ["slop"], "detector": "sapling-x"},
                {"id": 3, "op": "score", "texts": ["slop"], "detector": "nobody"},
            )
        )
    )
    stdout = io.StringIO()
    serve.serve_loop(detectors, stdin, stdout, meta=meta)
    first, named, unknown = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert first == {"id": 1, "scores": [0.9], "detector": "stub-clf"}
    assert named == {
        "id": 2,
        "scores": [0.7],
        "detector": "sapling-x",
        "stats": {"requests": 3, "chars_sent": 1200},
    }
    assert unknown["id"] == 3 and "'nobody'" in unknown["error"]
    assert "['stub-clf', 'sapling-x']" in unknown["error"]


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
        {"id": 2, "scores": [0.9], "detector": "stub_detector"},
    ]


def _fake_sapling(posted: list) -> object:
    """A Sapling transport that records payloads and scores everything 0.42;
    stands in for `_urllib_post_json`, so it takes its `timeout=`."""

    def post(url, payload, headers, **_):
        posted.append((url, payload, headers))
        return {"results": [{"score": 0.42} for _ in payload["texts"]]}

    return post


def test_serve_arms_sapling_alone_and_never_rewrites_with_it(monkeypatch):
    """`--detector sapling` without the voice head: the marker scores through
    the hosted API, and `rewrite` stays unarmed even with a --backend — the
    hosted instrument is never the reranker (D106)."""
    from stylebot import eval as ev

    posted: list = []
    monkeypatch.setenv("SAPLING_API_KEY", "k")
    monkeypatch.setattr(ev, "_urllib_post_json", _fake_sapling(posted))
    monkeypatch.setattr(backends, "build_styler", lambda opts: lambda m, num_samples=1: ["x"])
    long = "prose " * 80
    requests = [
        {"id": 1, "op": "info"},
        {"id": 2, "op": "score", "texts": [long, "short"]},
        {"id": 3, "op": "rewrite", "text": long},
    ]
    result = CliRunner().invoke(
        ai_style.main,
        ["serve", "--detector", "sapling", "--backend", "tinker"]
        + ["--sampler-path", "s", "--base-model", "b"],
        input="".join(json.dumps(r) + "\n" for r in requests),
    )
    assert result.exit_code == 0, result.output
    info, scores, rewrite = [json.loads(line) for line in result.output.splitlines()]
    assert info["meta"]["detectors"] == ["sapling-20251027"]
    assert info["meta"]["name"] == "sapling-20251027"
    assert scores["scores"] == [0.42, None]  # the short text abstained, unsent
    assert scores["stats"]["texts_scored"] == 1 and scores["stats"]["cache_hits"] == 0
    assert [len(p["texts"]) for _, p, _ in posted] == [1]
    assert "without a styler" in rewrite["error"]


def test_serve_marks_with_sapling_but_reranks_with_the_voice_head(monkeypatch, tmp_path):
    """Both armed, Sapling first: `score` defaults to Sapling, `rewrite` ranks
    candidates by the voice head and sends nothing to the API (D106)."""
    from stylebot import eval as ev

    posted: list = []
    monkeypatch.setenv("SAPLING_API_KEY", "k")
    monkeypatch.setattr(ev, "_urllib_post_json", _fake_sapling(posted))
    monkeypatch.setattr(classify, "load_artifact_meta", lambda p: {"name": "stub"})
    monkeypatch.setattr(classify, "sklearn_detector", lambda p: stub_detector)
    monkeypatch.setattr(
        backends, "build_styler", lambda opts: lambda m, num_samples=1: ["slop"] * num_samples
    )
    long = "prose " * 80
    requests = [
        {"id": 1, "op": "score", "texts": [long]},
        {"id": 2, "op": "score", "texts": [long]},
        {"id": 3, "op": "rewrite", "text": "Plain prose here.", "best_of": 2},
    ]
    result = CliRunner().invoke(
        ai_style.main,
        ["serve", "--detector", "sapling", "--detector", "voice-clf", "--detector-model"]
        + [str(tmp_path), "--backend", "tinker", "--sampler-path", "s", "--base-model", "b"],
        input="".join(json.dumps(r) + "\n" for r in requests),
    )
    assert result.exit_code == 0, result.output
    first, again, rewrite = [json.loads(line) for line in result.output.splitlines()]
    assert first["detector"] == "sapling-20251027" and first["scores"] == [0.42]
    assert again["stats"]["cache_hits"] == 1 and len(posted) == 1  # the rescore was free
    assert rewrite["candidates"][0]["score"] == 0.9  # the stub head, not 0.42
    assert len(posted) == 1  # the rewrite spent nothing


# ---------------------------------------------------------------------------
# The sidecar as a library: `serve_sidecar` and `preference_recorder` (D78).
# ---------------------------------------------------------------------------


def sidecar_lines(
    *requests: object, make_styler=None, prefer=None, div_is_prose=None, backend=None
) -> list[dict]:
    """Drive `serve_sidecar` over in-memory streams, the way `run_lines` drives the loop."""
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    serve.serve_sidecar(
        backend or backends.BackendOpts(),
        detectors=stub_detector,
        meta={"name": "stub-clf"},
        make_styler=make_styler,
        reranker=stub_detector,
        prefer=prefer,
        div_is_prose=div_is_prose,
        stdin=stdin,
        stdout=stdout,
    )
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def test_serve_sidecar_scores_and_segments_without_a_styler():
    document = "## H\n\n::: {.callout-note}\nCallout prose.\n:::\n"
    info, scores, unarmed, segments = sidecar_lines(
        {"id": 1, "op": "info"},
        {"id": 2, "op": "score", "texts": ["pure slop", "prose"]},
        {"id": 3, "op": "rewrite", "text": "prose"},
        {"id": 4, "op": "segment", "text": document},
        div_is_prose=callout_is_prose,
    )
    assert info == {"id": 1, "meta": {"name": "stub-clf"}}
    assert scores == {"id": 2, "scores": [0.9, 0.1], "detector": "stub-clf"}
    assert "without a styler" in unarmed["error"]
    # The div policy reaches the `segment` op — the loop — and nothing else.
    assert [document[s["start"] : s["end"]] for s in segments["segments"]] == ["Callout prose."]


def test_serve_sidecar_builds_the_styler_lazily_and_once():
    requests = [
        {"id": 1, "op": "score", "texts": ["x"]},
        {"id": 2, "op": "rewrite", "text": "Plain prose here.", "best_of": 2},
        {"id": 3, "op": "rewrite", "text": "More prose here.", "best_of": 1},
    ]
    stdin = io.StringIO("".join(json.dumps(r) + "\n" for r in requests))
    stdout = io.StringIO()
    answered_before_build: list[int] = []

    def make_styler():
        answered_before_build.append(stdout.getvalue().count("\n"))
        return lambda messages, num_samples=1: ["Rewritten slop."] * num_samples

    serve.serve_sidecar(
        backends.BackendOpts(max_chunk_chars=400),
        detectors=stub_detector,
        meta={},
        make_styler=make_styler,
        reranker=stub_detector,
        stdin=stdin,
        stdout=stdout,
    )
    scored, first, second = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert scored == {"id": 1, "scores": [0.1], "detector": "detector"}
    assert answered_before_build == [1]  # built once, and only after scoring was answered
    assert first["n_sampled"] == 2 and first["input_score"] == 0.1
    assert {c["text"] for c in first["candidates"]} == {"Rewritten slop."}
    assert first["candidates"][0]["score"] == 0.9  # the stub detector, as the reranker
    assert second["id"] == 3 and "candidates" in second


def test_the_rewrite_op_serves_the_backends_prompt():
    """D85: the manifest's prompt, carried on `BackendOpts`, is what the editor's chunk sees."""
    seen: list[str] = []

    def make_styler():
        def styler(messages, num_samples=1):
            seen.append(messages[0]["content"])
            return ["Rewritten."] * num_samples

        return styler

    sidecar_lines(
        {"id": 1, "op": "rewrite", "text": "Plain prose here.", "best_of": 1},
        make_styler=make_styler,
        backend=backends.BackendOpts(system_prompt="the served prompt"),
    )
    assert seen == ["the served prompt"]


def test_serve_sidecar_hands_a_failed_build_to_the_client_and_lives():
    def unready():
        raise RunError("TINKER_API_KEY is not set")

    failed, alive = sidecar_lines(
        {"id": 1, "op": "rewrite", "text": "some prose"},
        {"id": 2, "op": "score", "texts": ["still slop"]},
        make_styler=unready,
    )
    assert failed == {"id": 1, "error": "RunError: TINKER_API_KEY is not set"}
    assert alive == {"id": 2, "scores": [0.9], "detector": "stub-clf"}


def test_preference_recorder_stamps_the_record_and_creates_the_directory(tmp_path):
    path = tmp_path / "nested" / "preferences.jsonl"
    prefer = serve.preference_recorder(path)
    (ok,) = sidecar_lines(
        {"id": 1, "op": "prefer", "decision": {"chosen": 1, "kept_input": False}},
        prefer=prefer,
    )
    assert ok == {"id": 1, "ok": True}
    (record,) = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert record["ts"].startswith("20") and "T" in record["ts"]
    assert {k: v for k, v in record.items() if k != "ts"} == {
        "run_id": None,
        "chooser": None,
        "chosen": 1,
        "kept_input": False,
    }

    # The blog's shape: a run id, a default chooser, and the client's own wins.
    tagged = serve.preference_recorder(path, run_id="r7", default_chooser="human")
    tagged({"chosen": 0})
    tagged({"chosen": 2, "chooser": "detector"})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [(r["run_id"], r["chooser"], r["chosen"]) for r in rows[1:]] == [
        ("r7", "human", 0),
        ("r7", "detector", 2),
    ]
