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
        "sloppy input",
        two_candidates,
        best_of=2,
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
        "dan-grade input prose",
        worse_candidates,
        best_of=2,
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
        recs.append(
            {
                "messages": [
                    {"role": "system", "content": STYLE_SYSTEM},
                    {"role": "user", "content": f"ctx\n\nslop {i}"},
                    {"role": "assistant", "content": f"ctx\n\ndan {i}"},
                ],
                "meta": {
                    "source": f"p{i}",
                    "capture_id": f"c{i}",
                    "chunk_index": 0,
                    "context": "ctx",
                },
            }
        )
    pairs.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")

    out = tmp_path / "outputs.jsonl"
    n = infer.rewrite_pairs_file(pairs, out, _upper_styler, limit=2)
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["output"] == "CTX\n\nSLOP 0"
    # The eval extractor strips the heading context from the output field...
    # ctx uppercased -> prefix miss is kept
    assert FIELD_EXTRACTORS["output"](rows[0]) == "CTX\n\nSLOP 0"
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
        chunk,
        candidates,
        best_of=2,
        scorer=infer.detector_scorer(fake_detector),
    )
    assert "notes.qmd#sec" in result.text and "@Halpern2016Actual" in result.text
    assert "tighter" in result.text  # the intact rewrite shipped
    assert result.n_anchor_rejected == 1


def test_anchor_guard_reverts_when_all_samples_lose_content():
    chunk = "Read [this](https://example.org/x)."
    result = infer.rewrite_text(
        chunk,
        lambda m, num_samples=1: ["Read this — trust me."],
    )
    assert result.text == chunk
    assert result.n_kept_input == 1 and result.n_anchor_rejected == 1
    assert "lost anchors" in result.decisions[0]
    assert "example.org" in result.decisions[0]


def test_anchor_guard_rejects_invention():
    """The half the guard was blind to until 2026-08-12, and the half the
    training corpus actively teaches: a candidate conjuring a URL the input
    never named is disqualified however Dan it sounds (stylebot#1)."""
    chunk = "The tinker cookbook merges the adapter back into the base weights."

    def candidates(messages, num_samples=1):
        return [
            "The [tinker cookbook](https://not-a-real.example/docs) merges it back.",
            "tinker-cookbook merges the adapter back into the base weights.",
        ]

    # The fabricating candidate would win on voice alone.
    scores = {
        "The [tinker cookbook](https://not-a-real.example/docs) merges it back.": 0.1,
        "tinker-cookbook merges the adapter back into the base weights.": 0.3,
        chunk: 0.6,
    }
    result = infer.rewrite_text(chunk, candidates, best_of=2, scorer=scores.__getitem__)
    assert "not-a-real.example" not in result.text
    assert result.n_anchor_rejected == 1
    assert result.n_anchor_invented == 1 and result.n_anchor_lost == 0


def test_anchor_guard_reverts_and_names_invention():
    chunk = "Read the notes."
    result = infer.rewrite_text(
        chunk,
        lambda m, num_samples=1: ["Read [the notes](https://made-up.example/n)."],
    )
    assert result.text == chunk
    assert result.n_kept_input == 1 and result.n_anchor_rejected == 1
    # Named as invention, not filed under loss: the two want different fixes.
    assert "invented anchors" in result.decisions[0]
    assert "made-up.example" in result.decisions[0]


def test_repeating_a_source_anchor_is_not_invention():
    """The case the one-directional check never had to answer. A count increase
    on a key the source already carried points somewhere the source vouched for,
    so splitting a sentence and re-citing the same link is a legitimate edit."""
    chunk = "See [the notes](./n.qmd) for the argument and the numbers."
    twice = "See [the notes](./n.qmd) for the argument. See [the notes](./n.qmd) for numbers."
    assert infer.invented_anchors(chunk, twice) == []
    result = infer.rewrite_text(chunk, lambda m, num_samples=1: [twice])
    assert result.text == twice
    assert result.n_anchor_rejected == 0


def test_anchor_guard_counts_both_causes_for_one_sample():
    """Loss and invention overlap rather than partition: a sample that swaps one
    link for another did both, and `n_anchor_rejected` must not double-count."""
    chunk = "Read [the notes](./real.qmd)."
    result = infer.rewrite_text(
        chunk,
        lambda m, num_samples=1: ["Read [the notes](https://fake.example/x)."],
    )
    assert result.n_anchor_rejected == 1
    assert result.n_anchor_lost == 1 and result.n_anchor_invented == 1
    assert "lost anchors" in result.decisions[0] and "invented anchors" in result.decisions[0]


def test_anchor_audit_records_every_candidate(tmp_path):
    """Step 0 of `_plans/anchor-integrity.md`: the counters must survive the run.
    The denominator is candidates not marked truncated; the numerator is those
    with a non-empty lost/gained."""
    chunk = "Read [the notes](./real.qmd) and think about it for a while, honestly."
    result = infer.rewrite_text(
        chunk,
        lambda m, num_samples=3: [
            "Read [the notes](./real.qmd); think.",  # clean
            "Read [the notes](https://fake.example/x); think about it a while.",  # invented + lost
            "Short.",  # truncated
        ],
        best_of=3,
    )
    (entry,) = result.anchor_audit
    assert entry["chunk"] == 1 and entry["source_anchors"] == ["./real.qmd"]
    clean, damaged, cut = entry["candidates"]
    assert clean["lost"] == [] and clean["gained"] == []
    assert damaged["gained"] == ["https://fake.example/x"] and damaged["lost"] == ["./real.qmd"]
    # A truncated sample gets no anchor verdict at all: its delta measures the
    # token budget, not the model's link handling.
    assert cut == {"truncated": True, "chars": len("Short.")}

    path = tmp_path / "anchors.jsonl"
    assert infer.write_anchor_audit(path, result, source="draft.qmd") == 1
    (line,) = [json.loads(x) for x in path.read_text().splitlines()]
    assert line["source"] == "draft.qmd" and line["at"] and line["chunk"] == 1
    # Anchor keys and lengths, never candidate text: joinable, not a second copy.
    assert "text" not in json.dumps(line["candidates"])


def test_on_decision_reports_candidates_scores_and_choice():
    """The preference seam: per prose chunk the hook sees the candidate
    texts, aligned scores, and which one shipped."""
    calls = []

    def two_candidates(messages, num_samples=1):
        return ["sloppy version", "dan version"]

    def fake_detector(text):
        return {"score": 0.1 if "dan" in text else 0.9}

    infer.rewrite_text(
        "sloppy input",
        two_candidates,
        best_of=2,
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
        "dan-grade input prose",
        worse_candidates,
        best_of=2,
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
        chunk,
        candidates,
        best_of=2,
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
        text,
        _upper_styler,
        max_chunk_chars=10,
        on_decision=lambda *args: calls.append(args),
    )
    assert [(c[0], c[1]) for c in calls] == [
        ("Intro paragraph.", None),
        ("Under the heading.", "## The Heading"),
    ]
    # Without a scorer the first usable sample ships: chosen index 0.
    assert all(c[4] == 0 and c[5] is False for c in calls)


def test_inline_headings_records_no_context():
    # Defect A4: `rewrite_text` set `context` from any heading item, so under
    # --inline-headings the heading was prepended to every following chunk —
    # while cross-heading training pairs record no `meta.context` at all.
    # (A heading too big to group stays a standalone "heading" item and is
    # passed through, hence two decisions rather than three.)
    calls = []
    text = "Intro paragraph.\n\n## The Heading\n\nUnder the heading."
    infer.rewrite_text(
        text,
        _upper_styler,
        max_chunk_chars=10,
        inline_headings=True,
        on_decision=lambda *args: calls.append(args),
    )
    assert [c[1] for c in calls] == [None, None]
    # The same document without the flag DOES carry the heading as context.
    calls.clear()
    infer.rewrite_text(
        text,
        _upper_styler,
        max_chunk_chars=10,
        on_decision=lambda *args: calls.append(args),
    )
    assert [c[1] for c in calls] == [None, "## The Heading"]


def test_protected_block_does_not_reset_heading_context():
    # `_split_sections` runs AFTER `editable_prose`, so a protected block is
    # invisible to it and never breaks a section. The heading must survive one.
    calls = []
    text = "## The Heading\n\nFirst para.\n\n> A quote in between.\n\nSecond para."
    infer.rewrite_text(
        text,
        _upper_styler,
        max_chunk_chars=10,
        on_decision=lambda *args: calls.append(args),
    )
    assert [c[1] for c in calls] == ["## The Heading", "## The Heading"]


def test_content_anchors_counting():
    text = "A [link](./a.qmd), a bare https://b.org/c url, and [@Cite2020] twice: [@Cite2020]."
    anchors = infer.content_anchors(text)
    assert anchors["./a.qmd"] == 1
    assert anchors["https://b.org/c"] == 1
    assert anchors["@Cite2020"] == 2
    assert infer.missing_anchors(text, text.replace(" [@Cite2020].", ".")) == ["@Cite2020"]


# --- inline anchors (code spans + inline math), off by default -------------


def test_inline_anchors_are_off_by_default():
    text = "Set `PATH` and note $x_i$ carefully."
    assert infer.content_anchors(text) == {}
    assert set(infer.content_anchors(text, inline=True)) == {"PATH", "x_i"}


def test_inline_math_ignores_dollar_amounts():
    # The loose `\$([^$\n]+?)\$` calls "5 and " an anchor here, so every price
    # in a paragraph becomes a false positive. Requiring non-space at both ends
    # of the span is what kills it.
    assert infer.content_anchors("it costs $5 and $10", inline=True) == {}


def test_inline_math_and_code_spans_do_not_confuse_each_other():
    got = infer.content_anchors("`$PATH` and $x_i$", inline=True)
    # `$PATH` is a code span, not maths: the math pattern must not pair its
    # dollar with the one opening $x_i$.
    assert got == {"$PATH": 1, "x_i": 1}


def test_inline_code_anchor_does_not_swallow_a_fenced_block():
    # Single-line by construction — with DOTALL this pattern eats whole fences.
    text = "```\na\n```\ntext `x` more"
    assert infer.content_anchors(text, inline=True) == {"x": 1}


def test_inline_anchors_skip_escaped_dollars_and_display_math():
    assert infer.content_anchors(r"a \$5 fee and $y$", inline=True) == {"y": 1}
    assert infer.content_anchors("$$x$$", inline=True) == {}


def test_inline_anchors_skip_double_backtick_spans():
    # Deliberately conservative: a missed anchor costs nothing, a false one
    # rejects a good rewrite.
    assert infer.content_anchors("``a ` b`` and `c`", inline=True) == {"c": 1}


def test_rewrite_text_inline_anchors_disqualify_a_lost_code_span():
    text = "Run `make check` before pushing anything at all."
    dropped = "Run make check before pushing anything at all."
    styler = lambda messages, num_samples=1: [dropped] * num_samples  # noqa: E731

    loose = infer.rewrite_text(text, styler)
    assert loose.text == dropped  # today's guard lets it through

    strict = infer.rewrite_text(text, styler, inline_anchors=True)
    assert strict.text == text
    assert strict.n_anchor_rejected == 1


def test_rewrite_candidates_always_flags_inline_anchors():
    chunk = "Run `make check` first, honestly."
    styler = lambda messages, num_samples=1: ["Run make check first."]  # noqa: E731
    out = infer.rewrite_candidates(chunk, styler, best_of=1)
    assert out["candidates"][0]["anchors_lost"] == ["make check"]


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
        "text": "See [that](./a.qmd). Fine.",
        "score": 0.2,
        "anchors_lost": [],
        "anchors_gained": [],
        "truncated": False,
        "identity": False,
    }
    # Anchor loss FLAGS, never disqualifies — the human decides.
    assert dropped["anchors_lost"] == ["./a.qmd"]


def test_rewrite_candidates_flags_invention_without_disqualifying():
    """The editor seam flags BOTH directions and rejects neither: a human
    picking from the slate is the one who can tell a real URL from a plausible
    one, which is exactly why `rewrite_text` may not make this call itself."""
    chunk = "See that page and slop besides."
    styler = lambda m, num_samples=1: ["See [that page](https://invented.example/x)."]  # noqa: E731
    out = infer.rewrite_candidates(chunk, styler, best_of=1)
    (only,) = out["candidates"]
    assert only["anchors_gained"] == ["https://invented.example/x"]
    assert only["anchors_lost"] == []


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


def test_rewrite_candidates_names_an_empty_selection_as_empty():
    """Nothing selected must not be reported as over the chunk budget."""
    import pytest

    for blank in ("", "   \n\n  ", "\n"):
        with pytest.raises(ValueError, match="selection is empty") as excinfo:
            infer.rewrite_candidates(blank, _identity_styler)
        assert "budget" not in str(excinfo.value)


def test_rewrite_candidates_still_reports_a_real_overrun_as_over_budget():
    import pytest

    with pytest.raises(ValueError, match="over the 5-char chunk budget"):
        infer.rewrite_candidates("one two\n\nthree four", _identity_styler, max_chunk_chars=5)


# --- inline headings (v2 cross-heading diet) ------------------------------

_DOC = (
    "First para of prose here.\n\n## A Section\n\n"
    "Second para under it.\n\n### Deeper\n\nThird para.\n"
)


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


def test_segment_sees_an_indented_heading():
    # Defect A3: `infer` used to anchor headings at column 0 while `targets` and
    # `capture` allowed up to 3 spaces, so a legal CommonMark `  ## X` was
    # rewritable prose at inference and a section boundary in training.
    doc = "Prose here.\n\n  ## Indented\n\nMore prose.\n"
    items, _ = infer._segment(doc, max_chars=5000)
    assert [k for k, _ in items] == ["prose", "heading", "prose"]


def test_segment_rejects_a_tab_indented_heading():
    # A tab indents by 4 under CommonMark, so it cannot open a heading.
    doc = "Prose here.\n\n\t## Tabbed\n\nMore prose.\n"
    items, _ = infer._segment(doc, max_chars=5000)
    assert "heading" not in [k for k, _ in items]


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
    assert infer.max_tokens_for(4500) > infer.DEFAULT_MAX_TOKENS  # 4.5k chunk needs room


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


# --- truncation, told apart from anchor damage -----------------------------
#
# `_plans/placeholder-design-2026-08.md`: a model that runs out of budget looks
# exactly like a model that deletes your links, and conflating them contaminates
# every span-damage number. The floor is ours, not that document's: a style
# rewrite's length ratio spans 0.18-9.76 over the corpus and leans compressive
# only on average (median 0.79; 13.6% of real rewrites expand), so its 0.9 would
# call 70% of genuine rewrites truncated. Hence a floor, never a band.


def _truncating_styler(messages, num_samples=1):
    """A model that stopped a fifth of the way through."""
    body = messages[-1]["content"]
    return [body[: max(1, len(body) // 5)]] * num_samples


def test_truncation_is_reported_as_truncation_not_anchor_loss():
    chunk = (
        "Some prose about [the paper](https://example.com/a) and a good deal more "
        "discussion besides, which continues for a while before it finally reaches "
        "[a second link](https://example.com/b) at the very end of the passage."
    )
    result = infer.rewrite_text(chunk, _truncating_styler, best_of=1)
    assert result.n_truncated == 1
    assert result.n_anchor_rejected == 0  # NOT filed as anchor damage
    assert result.n_kept_input == 1
    (decision,) = result.decisions
    assert "truncated" in decision and "max-tokens" in decision
    assert "anchor" not in decision
    assert result.text == chunk  # do no harm: the input survives


def test_truncation_is_caught_even_with_no_anchors_to_lose():
    """The case the anchor guard cannot see: no link, so nothing to lose."""
    chunk = (
        "A paragraph of entirely ordinary prose carrying no links, no citation "
        "keys and no inline code whatsoever, long enough to be a real chunk."
    )
    assert infer.content_anchors(chunk) == {}  # nothing for anchors to protect
    result = infer.rewrite_text(chunk, _truncating_styler, best_of=1)
    assert result.n_truncated == 1
    assert result.text == chunk  # would have shipped a fifth of the prose


def test_legitimate_compression_is_not_truncation():
    """Real rewrites compress hard at the low end; the floor must not eat them."""
    chunk = "This is a passage that a good rewrite would tighten considerably indeed."

    def compressing_styler(messages, num_samples=1):
        # ~0.45 of input: past the corpus's 10th percentile, well above the floor.
        return ["A passage a good rewrite would tighten."] * num_samples

    result = infer.rewrite_text(chunk, compressing_styler, best_of=1)
    assert result.n_truncated == 0
    assert result.text.strip() == "A passage a good rewrite would tighten."


def test_expansion_is_never_truncation():
    """13.6% of real rewrites are LONGER than their input. A floor-only guard
    must be silent on all of them — there is no ceiling, deliberately."""
    chunk = "A terse claim."

    def expanding_styler(messages, num_samples=1):
        return ["A claim, unpacked at considerably greater length than the original."]

    result = infer.rewrite_text(chunk, expanding_styler, best_of=1)
    assert result.n_truncated == 0
    assert result.text.strip().startswith("A claim, unpacked")


def test_is_truncated_boundary():
    src = "x" * 100
    assert infer.is_truncated(src, "y" * 29)
    assert not infer.is_truncated(src, "y" * 31)
    assert not infer.is_truncated("", "")  # nothing to collapse from


def test_rewrite_candidates_flags_truncation_without_filtering():
    chunk = "A passage of prose long enough that a fifth of it is plainly a cut-off."
    out = infer.rewrite_candidates(chunk, _truncating_styler, best_of=1, scorer=None)
    assert len(out["candidates"]) == 1  # flagged, not filtered
    assert out["candidates"][0]["truncated"] is True
