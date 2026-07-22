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


def test_best_of_picks_best_candidate_and_reports():
    def two_candidates(messages, num_samples=1):
        return ["sloppy version", "dan version"]

    def fake_detector(text):
        return {"score": 0.1 if "dan" in text else 0.9}

    result = infer.rewrite_text(
        "sloppy input", two_candidates, best_of=2,
        scorer=infer.detector_scorer(fake_detector),
    )
    assert result.text == "dan version"
    assert result.n_candidates == 2 and result.n_kept_input == 0
    assert "sample 2/2" in result.decisions[0]


def test_do_no_harm_guard_keeps_better_input():
    """With a scorer, the input competes: no candidate beating it => no edit."""

    def worse_candidates(messages, num_samples=1):
        return ["sloppy attempt A", "sloppy attempt B"]

    def fake_detector(text):
        return {"score": 0.2 if "dan" in text else 0.8}

    result = infer.rewrite_text(
        "dan-grade input prose", worse_candidates, best_of=2,
        scorer=infer.detector_scorer(fake_detector),
    )
    assert result.text == "dan-grade input prose"
    assert result.n_kept_input == 1
    assert "kept input" in result.decisions[0]


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


def test_anchor_guard_rejects_information_loss():
    """A candidate that drops a link/citation is disqualified before voice
    scoring; a candidate keeping all anchors wins even if it scores worse."""
    chunk = "See [the notes](./notes.qmd#sec) and [@Halpern2016Actual] for more."

    def candidates(messages, num_samples=1):
        return [
            "Tighter, very Dan, but the link and citation are gone.",
            "See [the notes](./notes.qmd#sec) and [@Halpern2016Actual]; tighter.",
        ]

    def fake_detector(text):
        # The lossy candidate would win on voice alone; the input is sloppiest.
        if "gone" in text:
            return {"score": 0.1}
        return {"score": 0.5 if "for more" in text else 0.3}

    result = infer.rewrite_text(
        chunk, candidates, best_of=2, scorer=infer.detector_scorer(fake_detector),
    )
    assert "notes.qmd#sec" in result.text and "@Halpern2016Actual" in result.text
    assert "tighter" in result.text  # the intact rewrite shipped
    assert result.n_anchor_rejected == 1


def test_anchor_guard_reverts_when_all_samples_lose_content():
    chunk = "Read [this](https://example.org/x)."
    result = infer.rewrite_text(
        chunk, lambda m, num_samples=1: ["Read this — trust me."],
    )
    assert result.text == chunk
    assert result.n_kept_input == 1 and result.n_anchor_rejected == 1
    assert "lost anchors" in result.decisions[0]
    assert "example.org" in result.decisions[0]


def test_on_decision_reports_candidates_scores_and_choice():
    """The preference seam: per prose chunk the hook sees the candidate
    texts, aligned scores, and which one shipped."""
    calls = []

    def two_candidates(messages, num_samples=1):
        return ["sloppy version", "dan version"]

    def fake_detector(text):
        return {"score": 0.1 if "dan" in text else 0.9}

    infer.rewrite_text(
        "sloppy input", two_candidates, best_of=2,
        scorer=infer.detector_scorer(fake_detector),
        on_decision=lambda *args: calls.append(args),
    )
    assert len(calls) == 1
    chunk, context, candidates, scores, chosen, kept_input = calls[0]
    assert chunk == "sloppy input" and context is None
    assert candidates == ["sloppy version", "dan version"]
    assert scores == [0.9, 0.1]
    assert chosen == 1 and kept_input is False


def test_on_decision_kept_input_has_no_chosen_index():
    calls = []

    def worse_candidates(messages, num_samples=1):
        return ["sloppy attempt A", "sloppy attempt B"]

    def fake_detector(text):
        return {"score": 0.2 if "dan" in text else 0.8}

    infer.rewrite_text(
        "dan-grade input prose", worse_candidates, best_of=2,
        scorer=infer.detector_scorer(fake_detector),
        on_decision=lambda *args: calls.append(args),
    )
    (_chunk, _ctx, candidates, scores, chosen, kept_input) = calls[0]
    assert len(candidates) == 2 and scores == [0.8, 0.8]
    assert chosen is None and kept_input is True


def test_on_decision_anchor_rejected_candidate_is_unscored():
    """Anchor-disqualified samples still appear as rejected candidates —
    that's the preference signal — but carry score None (never voice-scored)."""
    calls = []
    chunk = "See [the notes](./notes.qmd#sec) for more."

    def candidates(messages, num_samples=1):
        return [
            "Lost the link entirely.",
            "See [the notes](./notes.qmd#sec); tighter.",
        ]

    def fake_detector(text):
        return {"score": 0.3 if "tighter" in text else 0.5}

    infer.rewrite_text(
        chunk, candidates, best_of=2,
        scorer=infer.detector_scorer(fake_detector),
        on_decision=lambda *args: calls.append(args),
    )
    (_chunk, _ctx, cands, scores, chosen, kept_input) = calls[0]
    assert cands == ["Lost the link entirely.", "See [the notes](./notes.qmd#sec); tighter."]
    assert scores[0] is None and scores[1] == 0.3
    assert chosen == 1 and kept_input is False


def test_on_decision_receives_heading_context_and_fires_per_prose_chunk():
    calls = []
    text = "Intro paragraph.\n\n## The Heading\n\nUnder the heading."
    infer.rewrite_text(
        text, _upper_styler, max_chunk_chars=10,
        on_decision=lambda *args: calls.append(args),
    )
    assert [(c[0], c[1]) for c in calls] == [
        ("Intro paragraph.", None),
        ("Under the heading.", "## The Heading"),
    ]
    # Without a scorer the first usable sample ships: chosen index 0.
    assert all(c[4] == 0 and c[5] is False for c in calls)


def test_content_anchors_counting():
    text = "A [link](./a.qmd), a bare https://b.org/c url, and [@Cite2020] twice: [@Cite2020]."
    anchors = infer.content_anchors(text)
    assert anchors["./a.qmd"] == 1
    assert anchors["https://b.org/c"] == 1
    assert anchors["@Cite2020"] == 2
    assert infer.missing_anchors(text, text.replace(" [@Cite2020].", ".")) == ["@Cite2020"]
