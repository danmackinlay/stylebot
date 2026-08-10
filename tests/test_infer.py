"""Tests for Phase-4 inference (`stylebot.infer`). No network: stylers are
deterministic fakes."""

from __future__ import annotations

import json

import pytest

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


# ---------------------------------------------------------------------------
# rewrite_candidates — the editor seam (all candidates, none auto-chosen)
# ---------------------------------------------------------------------------


def test_rewrite_candidates_scores_and_flags_anchors():
    chunk = "See [that](./a.qmd) and slop besides."

    def styler(messages, num_samples=1):
        assert messages[0]["content"] == STYLE_SYSTEM
        return ["See [that](./a.qmd). Fine.", "Dropped the link entirely."]

    scores = {"See [that](./a.qmd). Fine.": 0.2, "Dropped the link entirely.": 0.1, chunk: 0.6}
    result = infer.rewrite_candidates(chunk, styler, best_of=2, scorer=scores.__getitem__)
    assert result["input_score"] == 0.6
    assert result["n_sampled"] == 2
    kept, dropped = result["candidates"]
    assert kept == {
        "text": "See [that](./a.qmd). Fine.", "score": 0.2,
        "anchors_lost": [], "identity": False,
    }
    # Anchor loss FLAGS, never disqualifies — the human decides.
    assert dropped["anchors_lost"] == ["./a.qmd"]


def test_rewrite_candidates_dedupes_and_flags_identity():
    chunk = "Already Dan prose."

    def mode_collapsed(messages, num_samples=1):
        # Low temperature in miniature: 3 identical identity draws + 1 rewrite.
        return [chunk, chunk, chunk, "Dan prose, already."]

    result = infer.rewrite_candidates(chunk, mode_collapsed, best_of=4)
    assert result["n_sampled"] == 4  # the draw count stays honest
    assert [c["text"] for c in result["candidates"]] == [chunk, "Dan prose, already."]
    assert [c["identity"] for c in result["candidates"]] == [True, False]


def test_rewrite_candidates_strips_heading_context_from_samples():
    def styler(messages, num_samples=1):
        # The context heading rides along on the user content (build_pair_content)
        assert messages[1]["content"].startswith("## Topic\n")
        return ["## Topic\n\nRewritten body."]

    result = infer.rewrite_candidates("Original body.", styler, context="## Topic", best_of=1)
    assert result["candidates"][0]["text"] == "Rewritten body."


def test_rewrite_candidates_enforces_the_single_prose_chunk_contract():
    import pytest

    with pytest.raises(ValueError, match="frontmatter"):
        infer.rewrite_candidates("---\nt: x\n---\nprose", _identity_styler)
    with pytest.raises(ValueError, match="heading"):
        infer.rewrite_candidates("## H\n\nprose", _identity_styler)
    with pytest.raises(ValueError, match="fence"):
        infer.rewrite_candidates("```py\nx\n```", _identity_styler)
    with pytest.raises(ValueError, match="chunk budget"):
        infer.rewrite_candidates("one two\n\nthree four", _identity_styler, max_chunk_chars=5)


# --- inline headings (v2 cross-heading diet) ------------------------------

_DOC = "First para of prose here.\n\n## A Section\n\nSecond para under it.\n\n### Deeper\n\nThird para.\n"


def test_segment_default_protects_headings():
    items, seps = infer._segment(_DOC, max_chars=5000)
    kinds = [k for k, _ in items]
    assert kinds.count("heading") == 2  # headings break prose grouping
    assert "".join(t + (seps[i] if i < len(seps) else "") for i, (_, t) in enumerate(items)) == _DOC


def test_segment_inline_headings_groups_across_sections():
    items, _seps = infer._segment(_DOC, max_chars=5000, inline_headings=True)
    assert len(items) == 1
    kind, text = items[0]
    assert kind == "prose"  # a group is chunk content, whatever it contains
    assert "## A Section" in text and "### Deeper" in text


def test_segment_inline_headings_is_lossless():
    items, seps = infer._segment(_DOC, max_chars=60, inline_headings=True)
    rebuilt = "".join(t + (seps[i] if i < len(seps) else "") for i, (_, t) in enumerate(items))
    assert rebuilt == _DOC  # budget forces splits, bytes still reproduce exactly


def test_segment_inline_headings_still_protects_fences():
    doc = "Prose here.\n\n```python\nx = 1\n```\n\n## H\n\nMore prose.\n"
    items, _ = infer._segment(doc, max_chars=5000, inline_headings=True)
    kinds = [k for k, _ in items]
    assert "fence" in kinds  # never groupable — absent from the corpus either way


def test_rewrite_candidates_rejects_heading_by_default():
    styler = lambda messages, num_samples=1: ["x"] * num_samples  # noqa: E731
    with pytest.raises(ValueError, match="heading"):
        infer.rewrite_candidates("## H\n\nSome prose.", styler, max_chunk_chars=5000)


def test_rewrite_candidates_accepts_section_under_inline_headings():
    """The editor's 'rewrite section' mode: a multi-section selection is one
    rewrite when that is what the adapter was trained on."""
    styler = lambda messages, num_samples=1: ["## H\n\nDan prose."] * num_samples  # noqa: E731
    out = infer.rewrite_candidates(
        "## H\n\nSome prose.", styler, max_chunk_chars=5000, inline_headings=True
    )
    assert out["candidates"] and out["candidates"][0]["text"] == "## H\n\nDan prose."


def test_rewrite_candidates_still_rejects_fences_under_inline_headings():
    styler = lambda messages, num_samples=1: ["x"] * num_samples  # noqa: E731
    with pytest.raises(ValueError, match="fence"):
        infer.rewrite_candidates(
            "Prose.\n\n```py\nx=1\n```", styler, max_chunk_chars=5000, inline_headings=True
        )


def test_max_tokens_scales_with_chunk_budget():
    assert infer.max_tokens_for(1500) == infer.DEFAULT_MAX_TOKENS  # floor holds
    assert infer.max_tokens_for(4500) > infer.DEFAULT_MAX_TOKENS   # 4.5k chunk needs room


# --- the `run` / `serve` CLI seam ------------------------------------------
#
# `infer` has supported both heading diets since the v2 corpus, but the CLI
# passed neither — leaving stylebot's own entry points stuck on v1 and strictly
# less capable than the library. Serving at the wrong diet is out-of-
# distribution and makes the styler copy its input verbatim (observed
# 2026-07-22), so the flag has to actually reach the library.


def _spy_rewrite_text(monkeypatch):
    """Capture the kwargs `run` hands to `infer.rewrite_text`. No model."""
    seen: dict = {}

    def fake(text, styler, **kw):
        seen.update(kw)
        return infer.RewriteResult(text="rewritten")

    monkeypatch.setattr(infer, "openai_backend", lambda *a, **k: _identity_styler)
    monkeypatch.setattr(infer, "rewrite_text", fake)
    return seen


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--inline-headings", True), ("--no-inline-headings", False), (None, False)],
)
def test_run_cli_threads_inline_headings_into_rewrite_text(tmp_path, monkeypatch, flag, expected):
    from click.testing import CliRunner

    from stylebot.bin.ai_style import main

    seen = _spy_rewrite_text(monkeypatch)
    doc = tmp_path / "d.qmd"
    doc.write_text("## H\n\nSome prose.\n", encoding="utf-8")

    args = ["run", str(doc), "--backend", "openai", "--base-url", "http://x/v1", "--model", "m"]
    result = CliRunner().invoke(main, args + ([flag] if flag else []))
    assert result.exit_code == 0, result.output
    assert seen["inline_headings"] is expected
