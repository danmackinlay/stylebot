"""Tests for Phase-4 inference (`stylebot.infer`). No network: stylers are
deterministic fakes."""

from __future__ import annotations

import json

from stylebot import infer
from stylebot.ai_core import STYLE_SYSTEM


def _identity_styler(messages, num_samples=1):
    return [messages[1]["content"]] * num_samples


def _upper_styler(messages, num_samples=1):
    return [messages[1]["content"].upper()] * num_samples


BODY = (
    "First paragraph, plain prose.\n"
    "Second line of it.\n"
    "\n"
    "```python\n"
    "x = 1\n"
    "\n"
    "y = 2  # blank line INSIDE the fence must not split blocks\n"
    "```\n"
    "\n"
    "\n"
    "Last paragraph."
)


def test_chunk_roundtrip_exact():
    chunks, seps = infer.chunk_body(BODY, max_chars=10)  # force every block separate
    reassembled = chunks[0] + "".join(s + c for s, c in zip(seps, chunks[1:]))
    assert reassembled == BODY
    # The fence is one block despite its interior blank line.
    assert any(c.startswith("```python") and c.endswith("```") for c in chunks)


def test_chunk_grouping_respects_budget():
    chunks, _seps = infer.chunk_body(BODY, max_chars=len(BODY) + 10)
    assert len(chunks) == 1  # everything fits in one chunk
    chunks, _seps = infer.chunk_body(BODY, max_chars=60)
    assert all(len(c) <= 60 or "```" in c for c in chunks)


def test_rewrite_identity_preserves_text_exactly():
    text = "---\ntitle: T\n---\n" + BODY
    result = infer.rewrite_text(text, _identity_styler, max_chunk_chars=40)
    assert result.text == text
    assert result.n_chunks >= 2


def test_rewrite_frontmatter_never_sent_to_styler():
    seen = []

    def spy(messages, num_samples=1):
        seen.append(messages[1]["content"])
        assert messages[0]["content"] == STYLE_SYSTEM
        return [messages[1]["content"]]

    text = "---\ntitle: Secret\n---\nJust prose."
    infer.rewrite_text(text, spy)
    assert all("Secret" not in chunk for chunk in seen)


def test_rewrite_transforms_and_reassembles():
    result = infer.rewrite_text("one two\n\nthree four", _upper_styler, max_chunk_chars=8)
    assert result.text == "ONE TWO\n\nTHREE FOUR"


def test_empty_styler_answer_keeps_input():
    result = infer.rewrite_text("keep me", lambda m, num_samples=1: ["   "])
    assert result.text == "keep me"


def test_best_of_reranks_with_detector():
    def two_candidates(messages, num_samples=1):
        return ["sloppy version", "dan version"]

    def fake_detector(text):
        return {"score": 0.1 if "dan" in text else 0.9}

    result = infer.rewrite_text(
        "x", two_candidates, best_of=2,
        rerank=infer.detector_reranker(fake_detector),
    )
    assert result.text == "dan version"
    assert result.n_candidates == 2


def test_rewrite_pairs_file_writes_output_and_resumes(tmp_path):
    from stylebot.eval import FIELD_EXTRACTORS

    pairs = tmp_path / "pairs.jsonl"
    recs = []
    for i in range(3):
        recs.append({
            "messages": [
                {"role": "system", "content": STYLE_SYSTEM},
                {"role": "user", "content": f"ctx\n\nslop {i}"},
                {"role": "assistant", "content": f"ctx\n\ndan {i}"},
            ],
            "meta": {"source": f"p{i}", "capture_id": f"c{i}", "chunk_index": 0,
                     "context": "ctx"},
        })
    pairs.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")

    out = tmp_path / "outputs.jsonl"
    n = infer.rewrite_pairs_file(pairs, out, _upper_styler, limit=2)
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["output"] == "CTX\n\nSLOP 0"
    # The eval extractor strips the heading context from the output field...
    assert FIELD_EXTRACTORS["output"](rows[0]) == "CTX\n\nSLOP 0"  # ctx uppercased -> prefix miss is kept
    # ...and a well-behaved styler that PRESERVES the context prefix strips clean.
    rows[0]["output"] = "ctx\n\nSLOP 0"
    assert FIELD_EXTRACTORS["output"](rows[0]) == "SLOP 0"

    # Resumable: a second run only does the remaining record.
    n = infer.rewrite_pairs_file(pairs, out, _upper_styler)
    assert n == 1
    assert len(out.read_text().splitlines()) == 3


def test_headings_and_fences_protected_and_context_flows():
    """Inference mirrors training: headings/fences never reach the styler;
    the nearest heading rides along as the build_pair_content context and is
    stripped from the sample."""
    seen = []

    def spy(messages, num_samples=1):
        seen.append(messages[1]["content"])
        return [messages[1]["content"].upper()]

    text = (
        "Intro paragraph.\n"
        "\n"
        "## The Heading\n"
        "\n"
        "Under the heading.\n"
        "\n"
        "```py\ncode = 1\n```\n"
        "\n"
        "After the fence."
    )
    result = infer.rewrite_text(text, spy, max_chunk_chars=10)
    # Headings/fences survive verbatim; prose transformed (context stripped back off).
    assert "## The Heading" in result.text
    assert "```py\ncode = 1\n```" in result.text
    assert "UNDER THE HEADING." in result.text and "AFTER THE FENCE." in result.text
    # The styler saw the heading as a context PREFIX, never as a chunk.
    assert seen[0] == "Intro paragraph."
    assert seen[1] == "## The Heading\n\nUnder the heading."
    assert seen[2] == "## The Heading\n\nAfter the fence."
