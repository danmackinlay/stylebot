"""Tests for Phase 2 synthetic-pair generation: the corpus write contract.

Generators are injected, so these exercise the real `synthesize_pairs` /
`iter_targets` path with deterministic fakes — no API keys, no network — and
gate the output through the same `validate_pairs_file` contract Phase 3 uses.

Sibling modules carved off this one along its section seams (2026-08-11):
`test_synth_assignment.py` (arm rotation), `test_synth_targets.py` (selection
and chunk hygiene), `test_synth_generator.py` (the OpenRouter wire),
`test_synth_sessions.py` (the async run engine). What stays here is what
`synthesize_pairs` writes, dedupes, records and refuses.
"""

from __future__ import annotations

import pytest
from synth_helpers import _fake, _targets

from stylebot import synth
from stylebot.pairs import validate_pairs_file


def test_pairs_validate_and_carry_synthetic_meta(tmp_path, make_blog):
    root = make_blog()
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(targets, data_dir, [_fake("claude-x"), _fake("gpt-y")])
    assert result.written == len(targets)

    pairs_path = data_dir / "pairs.jsonl"
    assert validate_pairs_file(pairs_path) == []  # empty == valid

    import json

    recs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    assert all(r["meta"]["synthetic"] is True for r in recs)
    assert all("synthetic" in r["meta"]["tags"] for r in recs)
    # user = slop, assistant = the real target
    for r in recs:
        assert r["messages"][1]["content"] != r["messages"][2]["content"]


def test_idempotent_resume(tmp_path, make_blog):
    root = make_blog()
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)
    gens = [_fake("claude-x"), _fake("gpt-y")]

    first = synth.synthesize_pairs(targets, data_dir, gens)
    assert first.written == len(targets)

    second = synth.synthesize_pairs(targets, data_dir, gens)
    assert second.written == 0
    assert second.skipped_existing == len(targets)

    # File did not grow.
    pairs_path = data_dir / "pairs.jsonl"
    n_lines = len([ln for ln in pairs_path.read_text().splitlines() if ln.strip()])
    assert n_lines == len(targets)


def test_per_generator_mode_doubles_pairs(tmp_path, make_blog):
    root = make_blog()
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [_fake("claude-x"), _fake("gpt-y")], per_generator=True
    )
    assert result.written == 2 * len(targets)


def test_heading_context_prepended_both_sides(tmp_path):
    # A target carrying context -> heading is the verbatim prefix of BOTH the
    # slop (user) and the Dan body (assistant); slop is generated from the body
    # only (the fake echoes its input, so the heading must NOT appear in the
    # generated portion).
    import json

    from stylebot.pairs import validate_pairs_file

    data_dir = tmp_path / "corpus"
    target = synth.Target(
        text="A real paragraph of prose long enough to be a worthwhile target here.",
        source="post/x.qmd",
        chunk_index=0,
        chunk_total=1,
        context="## A Heading",
    )
    # Fake generator echoes the body it is GIVEN — assert it was given the body,
    # not the heading.
    given = {}

    def gen(body):
        given["arg"] = body
        return "[slop] " + body

    result = synth.synthesize_pairs([target], data_dir, [synth.Generator("g", generate=gen)])
    assert result.written == 1
    assert "## A Heading" not in given["arg"]  # generator saw the body only

    pairs_path = data_dir / "pairs.jsonl"
    assert validate_pairs_file(pairs_path) == []  # incl. the context-prefix invariant
    rec = json.loads(pairs_path.read_text().splitlines()[0])
    assert rec["messages"][1]["content"].startswith("## A Heading\n\n")
    assert rec["messages"][2]["content"].startswith("## A Heading\n\n")
    assert rec["meta"]["context"] == "## A Heading"
    assert rec["meta"]["context_mode"] == "immediate"


def test_context_changes_synth_key():
    # Same body, different context -> different synth_key (regenerates).
    from stylebot.synth import _synth_key

    assert _synth_key("g", "body") != _synth_key("g", "body", "## H")


def test_strategy_changes_synth_key():
    # Same generator/context/body, different slop strategy -> different key, so
    # iterating on the prompt regenerates rather than colliding on resume.
    from stylebot.synth import _synth_key

    assert _synth_key("g", "body") != _synth_key("g", "body", "", "catalogue")
    assert _synth_key("g", "body", "", "engaging") != _synth_key("g", "body", "", "catalogue")


def test_reasoning_and_prompt_change_synth_key():
    # Reasoning effort and prompt id are part of the key, so sweeping them
    # regenerates rather than colliding on resume. Temperature is NOT a key axis.
    import inspect

    from stylebot.synth import _synth_key

    base = _synth_key("g", "body")
    assert base != _synth_key("g", "body", "", "polish", "low")  # reasoning effort
    assert base != _synth_key("g", "body", "", "polish", "high", "pid123")  # prompt id
    assert "temperature" not in inspect.signature(_synth_key).parameters


def test_meta_records_slop_strategy(tmp_path):
    import json

    data_dir = tmp_path / "corpus"
    target = synth.Target(
        text="A paragraph long enough to be a worthwhile target.",
        source="post/x.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    gen = synth.Generator(name="m", generate=lambda t: "[slop] " + t, strategy="catalogue")

    result = synth.synthesize_pairs([target], data_dir, [gen])
    assert result.written == 1
    rec = json.loads((data_dir / "pairs.jsonl").read_text().splitlines()[0])
    assert rec["meta"]["slop_strategy"] == "catalogue"
    assert validate_pairs_file(data_dir / "pairs.jsonl") == []


def test_strategies_coexist_no_collision(tmp_path):
    # The same target generated under two strategies (same model name) yields two
    # distinct pairs in one file — the experimental "one run per strategy" loop.
    import json

    data_dir = tmp_path / "corpus"
    target = synth.Target(
        text="A paragraph long enough to be a worthwhile target.",
        source="post/x.qmd",
        chunk_index=0,
        chunk_total=1,
    )

    def make(strategy):
        return synth.Generator(name="m", generate=lambda t: f"[{strategy}] " + t, strategy=strategy)

    synth.synthesize_pairs([target], data_dir, [make("polish")])
    second = synth.synthesize_pairs([target], data_dir, [make("catalogue")])  # must NOT be skipped
    assert second.written == 1

    lines = (data_dir / "pairs.jsonl").read_text().splitlines()
    recs = [json.loads(ln) for ln in lines if ln.strip()]
    assert {r["meta"]["slop_strategy"] for r in recs} == {"polish", "catalogue"}


def test_resolve_strategy():
    label, system, version, prompt_id = synth.resolve_strategy("polish")
    assert label == "polish" and system == synth.STRATEGIES["polish"].system
    assert version == 1 and prompt_id == synth.prompt_id_of(system)
    # An explicit system overrides the registry under any label (version 0, hashed id).
    label, system, version, prompt_id = synth.resolve_strategy("dan-catalogue", "CUSTOM PROMPT")
    assert label == "dan-catalogue" and system == "CUSTOM PROMPT"
    assert version == 0 and prompt_id == synth.prompt_id_of("CUSTOM PROMPT")
    # Unknown name with no custom prompt is an error.
    import pytest

    with pytest.raises(ValueError):
        synth.resolve_strategy("nonsense")


def test_prompt_id_stable_and_distinct():
    assert synth.prompt_id_of("abc") == synth.prompt_id_of("abc")  # stable
    assert synth.prompt_id_of("abc") != synth.prompt_id_of("abd")  # content-sensitive
    # Registry strategies have distinct prompt ids (distinct prompt texts).
    ids = {synth.resolve_strategy(n)[3] for n in synth.STRATEGIES}
    assert len(ids) == len(synth.STRATEGIES)


def test_dry_run_writes_nothing(tmp_path, make_blog):
    root = make_blog()
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [synth.Generator("claude-x"), synth.Generator("gpt-y")], dry_run=True
    )
    assert result.written == 0
    assert result.planned == len(targets)
    assert not (data_dir / "pairs.jsonl").exists()


def test_gen_output_meta_recorded(tmp_path):
    # A generator returning GenOutput records its covariates under meta.gen.
    import json

    data_dir = tmp_path / "corpus"
    target = synth.Target(
        text="A paragraph long enough to be a worthwhile target.",
        source="post/x.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    gen = synth.Generator(
        name="m",
        generate=lambda t: synth.GenOutput(
            "[slop] " + t, {"model": "m", "reasoning_effort": "low", "completion_tokens": 5}
        ),
    )
    synth.synthesize_pairs([target], data_dir, [gen])
    rec = json.loads((data_dir / "pairs.jsonl").read_text().splitlines()[0])
    assert rec["meta"]["gen"] == {"model": "m", "reasoning_effort": "low", "completion_tokens": 5}
    assert validate_pairs_file(data_dir / "pairs.jsonl") == []


def test_bare_string_generator_has_no_gen_meta(tmp_path):
    # Bare-string fakes still work (back-compat) and record no meta.gen.
    import json

    data_dir = tmp_path / "corpus"
    target = synth.Target(
        text="A paragraph long enough to be a worthwhile target.",
        source="post/x.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    synth.synthesize_pairs(
        [target], data_dir, [synth.Generator(name="m", generate=lambda t: "[slop] " + t)]
    )
    rec = json.loads((data_dir / "pairs.jsonl").read_text().splitlines()[0])
    assert "gen" not in rec["meta"]


def test_model_stats_fold_run_economics(tmp_path):
    # Cost/cache/token stats accumulate per model from meta.gen as the run
    # goes, and the live SynthResult reaches the progress callback so the CLI
    # heartbeat can show spend without new plumbing.
    def gen(text, history=None):
        return synth.GenOutput(
            "[slop] " + text,
            {
                "cost": 0.01,
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "reasoning_tokens": 5,
                "cached_tokens": 60,
                "gen_seconds": 2.0,
            },
        )

    seen_costs: list[float] = []
    result = synth.synthesize_pairs(
        _targets(2),
        tmp_path / "c",
        [synth.Generator("g", generate=gen)],
        on_progress=lambda i, total, live: seen_costs.append(live.total_cost),
    )
    ms = result.model_stats["g"]
    assert ms["pairs"] == 2 and ms["cost"] == 0.02
    assert ms["prompt_tokens"] == 200 and ms["cached_tokens"] == 120
    assert ms["gen_seconds"] == 4.0 and ms["reasoning_tokens"] == 10
    assert result.total_cost == 0.02
    assert seen_costs == [0.01, 0.02]  # live accumulation, visible mid-run
    # Generators reporting no usage stay out of the stats' way.
    bare = synth.synthesize_pairs(
        _targets(3)[2:],
        tmp_path / "c2",
        [synth.Generator("plain", generate=lambda t, history=None: "[slop] " + t)],
    )
    assert bare.model_stats["plain"]["cost"] == 0.0 and bare.total_cost == 0.0


def test_replicate_mints_new_cells(tmp_path):
    # A replicate label deliberately resamples the same substrate: same target,
    # same config, new cells — recorded as meta.gen.replicate for faceting.
    import json

    targets = _targets(1)
    gen = synth.Generator("g", generate=lambda t, history=None: "[slop] " + t)
    first = synth.synthesize_pairs(targets, tmp_path / "c", [gen])
    again = synth.synthesize_pairs(targets, tmp_path / "c", [gen])
    draw2 = synth.synthesize_pairs(targets, tmp_path / "c", [gen], replicate="draw2")
    assert (first.written, again.written, draw2.written) == (1, 0, 1)
    recs = [json.loads(ln) for ln in (tmp_path / "c" / "pairs.jsonl").read_text().splitlines()]
    assert "replicate" not in (recs[0]["meta"].get("gen") or {})  # base corpus stays unlabelled
    assert recs[1]["meta"]["gen"]["replicate"] == "draw2"


def test_skip_covered_prevents_cross_epoch_target_doubling(tmp_path):
    # The cross-epoch scenario: the corpus holds a pair for target A from an
    # old config (effort=high, WITH heading context). A new run under a new
    # config would mint a fresh cell for A — doubling A's training weight.
    # skip_covered says a target with ANY pair is done; only B generates.
    # Coverage is context-agnostic (the recorded meta.context prefix is
    # stripped), so config drift in heading context does not defeat it.
    import json

    a, b = _targets(2)
    old = synth.Generator(
        "g", generate=lambda t, history=None: "[old slop] " + t, reasoning_effort="high"
    )
    first = synth.synthesize_pairs([a], tmp_path / "c", [old], context_dropout=0.0)
    assert first.written == 1
    # Give the recorded pair a heading context, as the old corpus has.
    rec_path = tmp_path / "c" / "pairs.jsonl"
    rec = json.loads(rec_path.read_text())
    rec["messages"][2]["content"] = "## Heading\n\n" + rec["messages"][2]["content"]
    rec["meta"]["context"] = "## Heading"
    rec_path.write_text(json.dumps(rec) + "\n")

    new = synth.Generator("g", generate=lambda t, history=None: "[new slop] " + t)  # effort=off
    covered_run = synth.synthesize_pairs([a, b], tmp_path / "c", [new], skip_covered=True)
    assert covered_run.written == 1  # B only
    assert covered_run.skipped_covered == 1
    recs = [json.loads(ln) for ln in rec_path.read_text().splitlines()]
    a_pairs = [r for r in recs if r["messages"][2]["content"].endswith(a.text)]
    assert len(a_pairs) == 1  # still only the old-epoch pair for A

    # Without the flag the same run WOULD double target A (distinct effort cell).
    doubled = synth.synthesize_pairs([a, b], tmp_path / "c", [new])
    assert doubled.written == 1 and doubled.skipped_existing == 1  # A regenerates, B dedupes


# --- coverage-key normalisation ----------------------------------------------
# The blog is edited between synth runs and most of that drift is mechanical:
# link-anchor repair scripts, a bulk quote-curling commit, a sentence-per-line
# reflow. None of it is a difference the styler could learn, so it must not mint
# a second pair for the same passage. Measured over the 3686-cell corpus these
# folds produce zero collisions between genuinely distinct passages.

_COSMETIC = [
    pytest.param(
        "See [the interaction notes](refs.qmd#interaction) for why.",
        "See [the interaction notes](refs.qmd#interacting) for why.",
        id="link-fragment-repair",
    ),
    pytest.param(
        "See [the interaction notes](refs.qmd#interaction) for why.",
        "See [the interaction notes](https://example.org/refs.html) for why.",
        id="link-destination-rewrite",
    ),
    pytest.param(
        'It doesn\'t matter, he said "loudly".',
        "It doesn’t matter, he said “loudly”.",
        id="quote-curling",
    ),
    pytest.param(
        "One sentence here.\nAnother sentence there.",
        "One sentence here. Another sentence there.",
        id="whitespace-join",
    ),
]


@pytest.mark.parametrize(("before", "after"), _COSMETIC)
def test_coverage_key_folds_cosmetic_drift(before, after):
    assert before != after
    assert synth.coverage_key(before) == synth.coverage_key(after)


@pytest.mark.parametrize(("before", "after"), _COSMETIC)
def test_skip_covered_survives_cosmetic_drift(tmp_path, before, after):
    # End to end: a pair exists for the pre-edit prose; today's segmenter hands
    # back the post-edit text. That is the same passage and must not be paid for
    # twice.
    old = synth.Target(text=before, source="post/0.qmd", chunk_index=0, chunk_total=1)
    new = synth.Target(text=after, source="post/0.qmd", chunk_index=0, chunk_total=1)
    gen = synth.Generator("g", generate=lambda t, history=None: "[slop] " + t)
    assert synth.synthesize_pairs([old], tmp_path / "c", [gen]).written == 1
    run = synth.synthesize_pairs([new], tmp_path / "c", [gen], skip_covered=True)
    assert (run.written, run.skipped_covered) == (0, 1)


def test_coverage_key_keeps_genuine_edits_distinct():
    # The fold must stay narrow: an authorial edit is a different passage, and a
    # false "covered" silently costs that passage its only training pair.
    a = "Robertson et al have shown it to be handy for rhythm."
    b = "Robertson et al have shown them to be handy for rhythm."
    assert synth.coverage_key(a) != synth.coverage_key(b)
    # Link TEXT is prose and is kept, even when the destination is dropped.
    assert synth.coverage_key("[the home page](x.qmd)") != synth.coverage_key("[the index](x.qmd)")
    # Case and punctuation are not folded.
    assert synth.coverage_key("Pi's philosophy") != synth.coverage_key("pi's philosophy")
    assert synth.coverage_key("badlogic, MIT) a.k.a") != synth.coverage_key("badlogic, MIT), a.k.a")


def test_coverage_key_leaves_unbalanced_link_alone():
    # A link with no closing paren is malformed markdown, not a destination the
    # regex may guess at; repairing one IS a prose change from our side.
    broken = "[recommendations](./ai_search.qmd#indexing-the-blog for posts."
    assert synth.coverage_key(broken) == broken
    assert synth.coverage_key(broken) != synth.coverage_key("[recommendations]() for posts.")


def test_skip_covered_still_generates_a_genuinely_edited_passage(tmp_path):
    old = synth.Target(
        text="Robertson et al have shown it to be handy for rhythm.",
        source="post/0.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    new = synth.Target(
        text="Robertson et al have shown them to be handy for rhythm.",
        source="post/0.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    gen = synth.Generator("g", generate=lambda t, history=None: "[slop] " + t)
    assert synth.synthesize_pairs([old], tmp_path / "c", [gen]).written == 1
    run = synth.synthesize_pairs([new], tmp_path / "c", [gen], skip_covered=True)
    assert (run.written, run.skipped_covered) == (1, 0)


def test_coverage_normalisation_does_not_reach_synth_key():
    # The hard constraint: `_synth_key` is the corpus's cell identity. Folding it
    # the way coverage folds would orphan every synthetic pair already generated.
    # (The golden hashes in tests/test_synth_keys.py pin the recipe itself.)
    for before, after in [(p.values[0], p.values[1]) for p in _COSMETIC]:
        assert synth._synth_key("g", before) != synth._synth_key("g", after)


def test_prompts_sidecar_written_and_deduped(tmp_path):
    import json

    gen = synth.Generator(
        name="m",
        generate=lambda t: "[slop] " + t,
        strategy="catalogue",
        prompt_id="abc123",
        prompt_version=2,
        prompt_system="You are a typical AI writing assistant...",
    )
    synth.synthesize_pairs(_targets(2), tmp_path / "c", [gen])
    synth.synthesize_pairs(_targets(2), tmp_path / "c", [gen])  # resume: no dup entry

    lines = (tmp_path / "c" / "prompts.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry == {
        "prompt_id": "abc123",
        "label": "catalogue",
        "version": 2,
        "system": "You are a typical AI writing assistant...",
    }


def test_transform_similarity_bounds():
    # whitespace-insensitive
    assert synth.transform_similarity("same text here", "same text  here") == 1.0
    assert synth.transform_similarity("", "") == 1.0
    long_a = "completely original prose about macroeconomic weather patterns " * 5
    long_b = "utterly different words concerning tidal marmalade physics today " * 5
    assert synth.transform_similarity(long_a, long_b) < 0.4
    assert synth.transform_similarity(long_a, long_a) == 1.0


def test_transform_sim_recorded_on_pairs(tmp_path):
    import json

    target = synth.Target(
        text="A paragraph long enough to be a worthwhile target here.",
        source="p.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    echo = synth.Generator("echo", generate=lambda t: t + " ")  # near-verbatim no-op
    rewrite = synth.Generator(
        "rw",
        generate=lambda t: (
            "Entirely different slop with none of the original words at all, padded out somewhat."
        ),
    )
    synth.synthesize_pairs([target], tmp_path / "a", [echo])
    synth.synthesize_pairs([target], tmp_path / "b", [rewrite])
    sim_a = json.loads((tmp_path / "a" / "pairs.jsonl").read_text())["meta"]["transform_sim"]
    sim_b = json.loads((tmp_path / "b" / "pairs.jsonl").read_text())["meta"]["transform_sim"]
    assert sim_a == 1.0
    assert sim_b < 0.5


def test_measured_strategy_registered():
    label, system, version, prompt_id = synth.resolve_strategy("measured")
    assert label == "measured" and version == 1
    # The whole point: describe texture, never quote a stock phrase.
    assert "In today's world" not in system


# ---------------------------------------------------------------------------
# Reasoning-trace capture (the diagnosis instrument for reasoning blowouts)
# ---------------------------------------------------------------------------


def test_reasoning_text_routed_to_sidecar_not_corpus(tmp_path):
    import json

    def gen(text, history=None):
        return synth.GenOutput(
            "[slop] " + text,
            {
                "reasoning_tokens": 4321,
                "reasoning_text": "Hmm, let me deliberate at absurd length...",
            },
        )

    data_dir = tmp_path / "corpus"
    result = synth.synthesize_pairs(_targets(1), data_dir, [synth.Generator("g", generate=gen)])
    assert result.written == 1

    rec = json.loads((data_dir / "pairs.jsonl").read_text().splitlines()[0])
    assert "reasoning_text" not in rec["meta"]["gen"]  # corpus stays lean
    side = json.loads((data_dir / "reasoning.jsonl").read_text().splitlines()[0])
    assert side["synth_key"] == rec["meta"]["synth_key"]  # joinable to its pair
    assert side["generator"] == "g" and side["reasoning_tokens"] == 4321
    assert "deliberate" in side["reasoning"]


def test_no_sidecar_without_capture(tmp_path):
    result = synth.synthesize_pairs(_targets(1), tmp_path / "c", [_fake("g")])
    assert result.written == 1
    assert not (tmp_path / "c" / "reasoning.jsonl").exists()


def test_max_transform_sim_gate_drops_identity_outputs(tmp_path):
    """A generator that returns the input verbatim is degenerate: with the gate
    set, its pairs are counted (skipped_degenerate) and never written; healthy
    transforms pass. (The 2026-07-21 corpus QA fix — identity pairs teach the
    styler to copy.)"""
    data_dir = tmp_path / "corpus"
    targets = _targets(6)

    echo = synth.Generator(name="echo-model", generate=lambda text: text)
    result = synth.synthesize_pairs(targets, data_dir, [echo], max_transform_sim=0.95)
    assert result.skipped_degenerate == 6
    assert result.written == 0
    pairs = data_dir / "pairs.jsonl"
    assert not pairs.exists() or not pairs.read_text().strip()
    # The spend is still accounted even though nothing was written.
    assert result.model_stats["echo-model"]["pairs"] == 6

    healthy = _fake("ok-model")
    result = synth.synthesize_pairs(targets, data_dir, [healthy], max_transform_sim=0.95)
    assert result.written == 6 and result.skipped_degenerate == 0

    # Unset gate (default): even identity output is written — covariate only.
    #
    # Deliberately the SAME data_dir, generator and texts as the gated run
    # above, because that pins a second property: a dropped cell is never
    # keyed, so it must be RE-ATTEMPTED rather than counted as already
    # present. `skipped_existing == 0` is what actually says so — without it
    # `written == 3` would also pass if the gate had (wrongly) written and
    # something else had changed the key.
    retried = targets[:3]
    result = synth.synthesize_pairs(retried, data_dir, [echo])
    assert result.skipped_existing == 0, "a gated-and-dropped cell must not count as present"
    assert result.skipped_degenerate == 0 and result.written == 3


def test_report_result_echoes_degenerate_count(capsys, tmp_path):
    """The CLI summary path must render the degenerate-drop line (regression:
    it referenced a name not in scope and crashed after a paid run)."""
    from stylebot import synth_cli

    result = synth.SynthResult(planned=3, skipped_degenerate=3)
    synth_cli._report_result(
        result,
        tmp_path / "pairs.jsonl",
        dry_run=False,
        session_turns=1,
        max_transform_sim=0.95,
        emit=synth_cli.click_emitter(),
    )
    out = capsys.readouterr().out
    # The whole line, not "0.95" loose in the summary: a bare substring passes
    # if the threshold happens to appear anywhere, including in a line that
    # never mentions the drop.
    assert "3 identity pair(s) DROPPED (transform_sim > 0.95)" in out


def test_max_length_ratio_gate_drops_inflated_outputs(tmp_path):
    """A generator that emits several times the target length is degenerate:
    with the inflation gate set, its pairs are counted (skipped_inflated) and
    never written (the qwen3-32b failure mode, 2026-07-22 QA)."""
    data_dir = tmp_path / "corpus"
    targets = _targets(4)

    inflate = synth.Generator(name="windbag", generate=lambda text: ("verily " + text + " ") * 5)
    result = synth.synthesize_pairs(targets, data_dir, [inflate], max_length_ratio=3.0)
    assert result.skipped_inflated == 4 and result.written == 0

    healthy = _fake("ok-model")
    result = synth.synthesize_pairs(targets, data_dir, [healthy], max_length_ratio=3.0)
    assert result.written == 4 and result.skipped_inflated == 0


def _delinker(name: str) -> synth.Generator:
    """A generator that strips the link while 'improving' the passage — the exact
    move that, once the edit is reversed, makes the target look like it invented
    one (`_plans/anchor-integrity.md`)."""
    return synth.Generator(
        name=name, generate=lambda text: f"[{name}-slop] " + text.replace("[docs](./d.qmd)", "docs")
    )


def _linked_targets(n: int) -> list[synth.Target]:
    import dataclasses

    return [
        dataclasses.replace(t, text=t.text + " See the [docs](./d.qmd) for the rest.")
        for t in _targets(n)
    ]


def test_invented_anchor_gate_drops_fabrication_lessons(tmp_path):
    """The most-taught failure in the corpus (876 pairs, 19.1%) and the one no
    gate saw until 0.3.0: a target carrying a link its slop never had."""
    data_dir = tmp_path / "corpus"
    targets = _linked_targets(4)

    result = synth.synthesize_pairs(
        targets, data_dir, [_delinker("dropper")], drop_invented_anchors=True
    )
    assert result.skipped_invented_anchors == 4 and result.written == 0

    # Off by default — the gate is newer than the corpus, so a run that does not
    # ask for it behaves exactly as it did before.
    result = synth.synthesize_pairs(targets, data_dir, [_delinker("dropper")])
    assert result.written == 4 and result.skipped_invented_anchors == 0


def test_invented_anchor_gate_keeps_a_faithful_generator(tmp_path):
    """The gate must not fire on a generator that carries the link through: it
    reads the anchor multiset, not the amount of rewriting."""
    data_dir = tmp_path / "corpus"
    result = synth.synthesize_pairs(
        _linked_targets(4), data_dir, [_fake("keeps-links")], drop_invented_anchors=True
    )
    assert result.written == 4 and result.skipped_invented_anchors == 0


def test_is_anchor_faithful_selects_over_an_existing_corpus(tmp_path):
    """Step 2's other half: the corpus is a frozen record, so pairs already
    written are filtered out of the training SIGNAL rather than migrated away."""
    from stylebot import infer
    from stylebot.jsonl import iter_jsonl

    data_dir = tmp_path / "corpus"
    synth.synthesize_pairs(_linked_targets(2), data_dir, [_delinker("dropper")])
    synth.synthesize_pairs(_linked_targets(2), data_dir, [_fake("keeps-links")])
    records = list(iter_jsonl(data_dir / "pairs.jsonl"))
    assert len(records) == 4
    kept = [r for r in records if infer.is_anchor_faithful(r)]
    assert len(kept) == 2
    assert all(r["meta"]["generator"] == "keeps-links" for r in kept)
    # The name lands in the training manifest, which a lambda could not do.
    assert infer.is_anchor_faithful.__name__ == "is_anchor_faithful"
