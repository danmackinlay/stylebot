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

import json

import pytest
from conftest import corpus_file, corpus_records
from synth_helpers import _fake, _targets

from stylebot import synth
from stylebot.pairs import validate_pairs_file


def test_pairs_validate_and_carry_synthetic_meta(tmp_path, make_blog):
    root = make_blog()
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(targets, data_dir, [_fake("claude-x"), _fake("gpt-y")])
    assert result.written == len(targets)

    assert validate_pairs_file(corpus_file(data_dir, tmp_path / "p.jsonl")) == []
    recs = corpus_records(data_dir)
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

    # The corpus did not grow.
    assert len(corpus_records(data_dir)) == len(targets)


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

    assert validate_pairs_file(corpus_file(data_dir, tmp_path / "p.jsonl")) == []
    rec = corpus_records(data_dir)[0]
    assert rec["messages"][1]["content"].startswith("## A Heading\n\n")
    assert rec["messages"][2]["content"].startswith("## A Heading\n\n")
    assert rec["meta"]["context"] == "## A Heading"
    assert rec["meta"]["context_mode"] == "immediate"


def test_strategies_coexist_no_collision(tmp_path):
    # The same target generated under two strategies (same model name) yields two
    # distinct pairs in one file — the experimental "one run per strategy" loop.

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

    recs = corpus_records(data_dir)
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


def test_argued_prompt_quotes_no_connective_it_is_measured_on():
    """The texture-not-tics rule made mechanical (D9): none of the stage-0
    justificatory connectives may appear verbatim in the prompt text."""
    from stylebot import features as ft
    from stylebot.generators import STRATEGIES

    prompt = STRATEGIES["argued"].system.lower()
    quoted = [c for c in ft.CONNECTIVES if c in prompt]
    assert not quoted, quoted
    assert "rather than" not in prompt  # the subordinator Dan strips most (the plan's "Risks")
    assert synth.resolve_strategy("argued").version == 2


def test_draws_carry_the_runs_system_prompt(tmp_path):
    """D85: the author slot is a run parameter; the default renders the frozen bytes."""
    from stylebot.ai_core import STYLE_SYSTEM, style_system

    other = style_system("Ada Lovelace")
    synth.synthesize_pairs(_targets(2), tmp_path / "ada", [_fake("g")], system_prompt=other)
    assert {r["messages"][0]["content"] for r in corpus_records(tmp_path / "ada")} == {other}
    synth.synthesize_pairs(_targets(2), tmp_path / "plain", [_fake("g")])
    stamped = {r["messages"][0]["content"] for r in corpus_records(tmp_path / "plain")}
    assert stamped == {STYLE_SYSTEM}


def test_dry_run_writes_nothing(tmp_path, make_blog):
    root = make_blog()
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [synth.Generator("claude-x"), synth.Generator("gpt-y")], dry_run=True
    )
    assert result.written == 0
    assert result.planned == len(targets)
    assert not (data_dir / "draws").exists()


def test_gen_output_meta_recorded(tmp_path):
    # A generator returning GenOutput records its covariates under meta.gen.

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
    rec = corpus_records(data_dir)[0]
    assert rec["meta"]["gen"] == {"model": "m", "reasoning_effort": "low", "completion_tokens": 5}
    assert validate_pairs_file(corpus_file(data_dir, tmp_path / "p.jsonl")) == []


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

    targets = _targets(1)
    gen = synth.Generator("g", generate=lambda t, history=None: "[slop] " + t)
    first = synth.synthesize_pairs(targets, tmp_path / "c", [gen])
    again = synth.synthesize_pairs(targets, tmp_path / "c", [gen])
    draw2 = synth.synthesize_pairs(targets, tmp_path / "c", [gen], replicate="draw2")
    assert (first.written, again.written, draw2.written) == (1, 0, 1)
    recs = corpus_records(tmp_path / "c")
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
    # Give the recorded pair a heading context, as the old corpus has. Editing
    # a shard by hand is test surgery; no command rewrites a draw (D84).
    shard = next(iter((tmp_path / "c" / "draws").glob("*.jsonl")))
    drawn = json.loads(shard.read_text())
    rec = drawn["record"]
    rec["messages"][2]["content"] = "## Heading\n\n" + rec["messages"][2]["content"]
    rec["meta"]["context"] = "## Heading"
    shard.write_text(json.dumps(drawn) + "\n")

    new = synth.Generator("g", generate=lambda t, history=None: "[new slop] " + t)  # effort=off
    covered_run = synth.synthesize_pairs([a, b], tmp_path / "c", [new], skip_covered=True)
    assert covered_run.written == 1  # B only
    assert covered_run.skipped_covered == 1
    recs = corpus_records(tmp_path / "c")
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
    sim_a = corpus_records(tmp_path / "a")[0]["meta"]["transform_sim"]
    sim_b = corpus_records(tmp_path / "b")[0]["meta"]["transform_sim"]
    assert sim_a == 1.0
    assert sim_b < 0.5


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

    rec = corpus_records(data_dir)[0]
    assert "reasoning_text" not in rec["meta"]["gen"]  # corpus stays lean
    side = json.loads((data_dir / "reasoning.jsonl").read_text().splitlines()[0])
    assert side["synth_key"] == rec["meta"]["synth_key"]  # joinable to its pair
    assert side["generator"] == "g" and side["reasoning_tokens"] == 4321
    assert "deliberate" in side["reasoning"]


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
    assert corpus_records(data_dir) == []
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
    one (D32)."""
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

    # On by default, unlike the two threshold gates: there is no number to tune
    # and no defensible other side (D32).
    result = synth.synthesize_pairs(targets, data_dir, [_delinker("dropper")])
    assert result.skipped_invented_anchors == 4 and result.written == 0

    # Opting out is what reproduces a pre-0.3.0 corpus.
    result = synth.synthesize_pairs(
        targets, data_dir, [_delinker("dropper")], drop_invented_anchors=False
    )
    assert result.written == 4 and result.skipped_invented_anchors == 0


def test_invented_anchor_gate_keeps_a_faithful_generator(tmp_path):
    """The gate must not fire on a generator that carries the link through: it
    reads the anchor multiset, not the amount of rewriting."""
    data_dir = tmp_path / "corpus"
    result = synth.synthesize_pairs(
        _linked_targets(4), data_dir, [_fake("keeps-links")], drop_invented_anchors=True
    )
    assert result.written == 4 and result.skipped_invented_anchors == 0


def test_gate_retries_redraw_and_recover(tmp_path):
    """A gated draw is paid for either way, so a redraw costs no more than the
    drop. Without it a gate biases coverage toward what the roster finds easy."""
    data_dir = tmp_path / "corpus"
    calls = {"n": 0}

    def flaky(text: str) -> str:
        # Drops the link on the first draw for each cell, keeps it after.
        calls["n"] += 1
        return "[flaky-slop] " + (
            text.replace("[docs](./d.qmd)", "docs") if calls["n"] % 2 else text
        )

    gen = synth.Generator(name="flaky", generate=flaky)
    result = synth.synthesize_pairs(_linked_targets(4), data_dir, [gen], gate_retries=2)
    assert result.written == 4 and result.skipped_invented_anchors == 0
    assert result.redrawn_cells == 4 and result.gate_redraws == 4
    assert result.redraws_recovered == 4
    assert calls["n"] == 8  # four cells, two draws each — the redraws were paid for


def test_gate_drops_sidecar_recovers_rate_and_wasted_spend(tmp_path):
    """The two questions a gated corpus can no longer answer about itself: what
    is this arm's rejection rate, and what did its failures cost? Both must be
    joinable from `gate_drops.jsonl` after the run ends."""
    import json

    data_dir = tmp_path / "corpus"

    def gen_out(text: str):
        # A GenOutput carries the cost/token metadata a real backend reports.
        return synth.GenOutput(
            text="[dropper-slop] " + text.replace("[docs](./d.qmd)", "docs"),
            meta={"cost": 0.004, "prompt_tokens": 300, "completion_tokens": 120},
        )

    gen = synth.Generator(name="dropper", generate=gen_out)
    result = synth.synthesize_pairs(_linked_targets(3), data_dir, [gen], gate_retries=1)
    assert result.written == 0 and result.skipped_invented_anchors == 3
    assert result.drops_by_generator == {"dropper": 6}  # 3 cells x (1 draw + 1 redraw)

    rows = [json.loads(x) for x in (data_dir / "lost_draws.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {r["gate"] for r in rows} == {"skipped_invented_anchors"}
    assert {r["kind"] for r in rows} == {"gate"}
    assert {r["generator"] for r in rows} == {"dropper"}
    # A redrawn draw is marked as such; the last attempt per cell is not.
    assert sorted(r["redrawn"] for r in rows) == [False] * 3 + [True] * 3
    assert sorted(r["attempt"] for r in rows) == [0, 0, 0, 1, 1, 1]
    # The spend the corpus cannot see, because nothing was written.
    assert sum(r["cost"] for r in rows) == pytest.approx(0.024)
    # And the reason blames the generator, not Dan's prose.
    assert all(r["reason"].startswith("generator deleted") for r in rows)


class _Boom(Exception):
    """Stands in for `openai.APIStatusError`, which `_is_transient` duck-types."""

    def __init__(self, status: int) -> None:
        super().__init__(f"Error code: {status} - Provider returned error")
        self.status_code = status


def test_transient_provider_error_is_redrawn(tmp_path, monkeypatch):
    """The 502-from-Cloudflare case. One flaky draw must not cost the cell —
    that is the difference between finishing a run and re-running it."""
    monkeypatch.setattr(synth.asyncio, "sleep", lambda _d: asyncio_noop())
    calls = {"n": 0}

    def flaky(text: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Boom(502)
        return "[slop] " + text

    result = synth.synthesize_pairs(
        _targets(1), tmp_path / "c", [synth.Generator(name="flaky", generate=flaky)]
    )
    assert result.written == 1 and result.errors == []
    assert result.gen_retries_spent == 1 and calls["n"] == 2


def test_permanent_error_is_not_retried(tmp_path):
    """A 400 is our bug or our config; retrying it just burns budget slower."""
    calls = {"n": 0}

    def broken(text: str) -> str:
        calls["n"] += 1
        raise _Boom(400)

    result = synth.synthesize_pairs(
        _targets(1), tmp_path / "c", [synth.Generator(name="broken", generate=broken)]
    )
    assert result.written == 0 and len(result.errors) == 1
    assert calls["n"] == 1  # exactly one attempt
    assert result.gen_retries_spent == 0


def test_error_lands_in_the_lost_draws_sidecar(tmp_path):
    """Answers 'is this run dying of QA or of provider errors?' — the question
    that was unanswerable while both were printed to stderr and discarded."""
    import json

    data_dir = tmp_path / "c"

    def broken(text: str) -> str:
        raise _Boom(400)

    synth.synthesize_pairs(_targets(2), data_dir, [synth.Generator(name="broken", generate=broken)])
    rows = [json.loads(x) for x in (data_dir / "lost_draws.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert {r["kind"] for r in rows} == {"error"}
    assert all(r["gate"] is None for r in rows)
    assert all("400" in r["reason"] for r in rows)


def test_gate_redraw_failure_does_not_cost_the_session(tmp_path):
    """A provider error DURING a gate redraw is one lost draw, once (C14): the
    gated first draw is recorded and charged once, the failed redraw is one
    `kind="error"` row and one attributed error, and the session's gate
    accounting does not run for a redraw that never happened. The remaining
    turns reflow rather than being abandoned (the 2026-08-12 regression) —
    until C14 this path recorded the gated draw twice and the error nowhere.

    The redraw raises a PERMANENT error on purpose — a transient one would be
    retried by `_draw` and never reach the give-up path this pins."""
    import json

    data_dir = tmp_path / "c"
    seen = {"n": 0}

    def gen(text: str, history=None):
        seen["n"] += 1
        if seen["n"] == 1:  # first cell: a gated draw, so a redraw is attempted
            return synth.GenOutput(
                text="[slop] " + text.replace("[docs](./d.qmd)", "docs"), meta={"cost": 0.004}
            )
        if seen["n"] == 2:  # ... and the redraw dies unrecoverably
            raise _Boom(400)
        return "[slop] " + text  # every later cell is fine

    result = synth.synthesize_pairs(
        _linked_targets(3),
        data_dir,
        [synth.Generator(name="flaky", generate=gen)],
        gate_retries=1,
        session_turns=8,
    )
    # The other two cells still get written: the session's leftover turns reflow.
    assert result.written == 2 and result.reflowed_turns == 2
    # The failed redraw is one error, attributed to its generator...
    assert len(result.errors) == 1 and "400" in result.errors[0][1]
    assert result.errors_by_generator == {"flaky": 1}
    # ... and not a gate drop: a redraw that was never drawn is not a paid redraw.
    assert result.skipped_invented_anchors == 0
    assert result.gate_redraws == 0 and result.redrawn_cells == 0
    # The gated draw is recorded once (it was paid for), the failed redraw once.
    rows = [json.loads(x) for x in (data_dir / "lost_draws.jsonl").read_text().splitlines()]
    assert [(r["kind"], r["redrawn"]) for r in rows] == [("gate", True), ("error", False)]
    # `attempt` means one thing on both rows: the draw's index within its cell.
    assert [r["attempt"] for r in rows] == [0, 1]
    assert result.drops_by_generator == {"flaky": 1}
    assert result.model_stats["flaky"]["cost"] == pytest.approx(0.004)  # charged once


def asyncio_noop():
    """An awaitable that returns immediately — keeps backoff out of test time."""

    async def _noop():
        return None

    return _noop()


def test_gate_retries_give_up_and_still_drop(tmp_path):
    """A passage no draw clears is a real drop, not an infinite redraw loop —
    and the recovery count is how the operator tells the two apart."""
    data_dir = tmp_path / "corpus"
    result = synth.synthesize_pairs(
        _linked_targets(2), data_dir, [_delinker("dropper")], gate_retries=3
    )
    assert result.written == 0 and result.skipped_invented_anchors == 2
    assert result.redrawn_cells == 2 and result.gate_redraws == 6
    assert result.redraws_recovered == 0


def test_is_anchor_faithful_selects_over_an_existing_corpus(tmp_path):
    """Step 2's other half: the corpus is a frozen record, so pairs already
    written are filtered out of the training SIGNAL rather than migrated away."""
    from stylebot import infer

    data_dir = tmp_path / "corpus"
    # Gate off: the corpus this selector exists for was written before it.
    synth.synthesize_pairs(
        _linked_targets(2), data_dir, [_delinker("dropper")], drop_invented_anchors=False
    )
    synth.synthesize_pairs(_linked_targets(2), data_dir, [_fake("keeps-links")])
    records = corpus_records(data_dir)
    assert len(records) == 4
    kept = [r for r in records if infer.is_anchor_faithful(r)]
    assert len(kept) == 2
    assert all(r["meta"]["generator"] == "keeps-links" for r in kept)
    # The name lands in the training manifest, which a lambda could not do.
    assert infer.is_anchor_faithful.__name__ == "is_anchor_faithful"


# --- the record contract: provenance and one-record-per-cell ------------------


def test_written_records_carry_the_five_provenance_keys(tmp_path, make_blog):
    """W8: every record written after 2026-09 says which trees and which cut
    produced it, and what the source frontmatter said AT THAT MOMENT."""

    from stylebot.pairs import PAIR_SCHEMA_VERSION
    from stylebot.provenance import PROVENANCE_KEYS, policy_digest
    from stylebot.targets import ChunkPolicy

    root = make_blog(ai=False)
    data_dir = tmp_path / "corpus"
    policy = ChunkPolicy(merge=True, merge_max_chars=1500)
    targets = synth.iter_targets(blog_root=root, policy=policy)
    synth.synthesize_pairs(targets, data_dir, [_fake("g")], blog_root=root)

    recs = corpus_records(data_dir)
    assert recs
    for r in recs:
        meta = r["meta"]
        assert meta["schema_version"] == PAIR_SCHEMA_VERSION
        # The prompt claim is the view's, not the draw's (D36, W4 step 5).
        assert "system_sha" not in meta
        prov = meta["provenance"]
        assert sorted(prov) == sorted(PROVENANCE_KEYS)
        # The blog fixture is not a git checkout, so the tree fields degrade to
        # null rather than raising — but the policy and the selector's view of
        # the post are always answerable.
        assert prov["policy_digest"] == policy_digest(policy)
        assert prov["source_automation"] == 0


def test_provenance_records_a_higher_automation_level_it_was_handed(tmp_path, make_blog):
    """The 75 pairs sitting on `automation: 2` posts had nothing recording that.
    A pre-selected FILES list bypasses the selector by design, so the record is
    the only place the level can be caught."""

    root = make_blog()  # includes the AI-touched post the default selector drops
    ai_post = root / "post" / "ai.qmd"
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(files=[ai_post])
    assert targets
    synth.synthesize_pairs(targets, data_dir, [_fake("g")])

    levels = {r["meta"]["provenance"]["source_automation"] for r in corpus_records(data_dir)}
    assert levels == {2}


def test_a_cell_claimed_mid_run_by_another_writer_is_not_duplicated(tmp_path, make_blog):
    """The 2026-08-12 defect, deterministically: `synthesize_pairs` plans against
    the keys it read at startup, so a second process appending the same cell
    between plan and write used to produce two records with one `record_id`.

    Injected through the generator rather than a real race, so the interleaving
    is exact rather than hoped for."""

    from stylebot import draws
    from stylebot.pairs import validate_pairs_file

    root = make_blog(ai=False)
    data_dir = tmp_path / "corpus"
    draws_dir = data_dir / "draws"
    draws_dir.mkdir(parents=True)
    targets = synth.iter_targets(blog_root=root)
    assert len(targets) >= 2

    # Plan the run once so we know the keys the "other process" would write.
    plan = synth.synthesize_pairs(targets, data_dir, [_fake("g")], dry_run=True)
    assert plan.planned == len(targets)

    competitor: dict[str, dict] = {}

    def _rival_then_slop(text: str) -> str:
        """Impersonate a concurrent run: append this cell's draw first."""
        for t in targets:
            if t.text == text:
                key = synth._synth_key("g", t.text, t.context)
                rec = synth._build_record(
                    slop=f"[rival] {text}",
                    target=t,
                    generator_name="g",
                    synth_key=key,
                    system=synth.style_system(),
                    context=t.context,
                )
                draws.append(draws_dir, draws.draw_from_pair(rec))
                competitor[key] = rec
                break
        return f"[g-slop] {text}"

    result = synth.synthesize_pairs(
        targets, data_dir, [synth.Generator(name="g", generate=_rival_then_slop)]
    )

    recs = corpus_records(data_dir)
    assert len(recs) == len(targets), "one record per cell, whoever wrote it"
    assert result.written == 0 and result.skipped_existing == len(targets)
    # The rival's draw is the one on disk — first writer wins, as `record_id`
    # consumers already assume.
    assert all(r["messages"][1]["content"].startswith("[rival]") for r in recs)
    assert validate_pairs_file(corpus_file(data_dir, tmp_path / "p.jsonl")) == []


def test_brief_then_write_composes_two_calls_and_sums_their_meta(tmp_path):
    """D92: the brief sees the passage and a sampled objection, the write call
    sees both plus the passage; meta sums, prompt_id hashes both prompts, the
    objection's text never reaches the corpus."""
    from stylebot.generators import GenOutput, brief_then_write, prompt_id_of

    seen: dict[str, str] = {}

    def brief_fn(text, history=None):
        seen["brief"] = text
        return GenOutput(
            "- claim A\n  rules out B [answers]", {"prompt_tokens": 100, "cost": 0.001}
        )

    async def write_fn(text, history=None):
        seen["write"] = text
        seen["history"] = history
        return GenOutput(
            "[argued2] " + text.rsplit("\n\n", 1)[-1], {"prompt_tokens": 300, "cost": 0.002}
        )

    brief = synth.Generator(name="b", generate=brief_fn, prompt_system="BRIEF", prompt_id="x")
    write = synth.Generator(name="w", generate=write_fn, prompt_system="WRITE", prompt_id="y")
    gen = brief_then_write(brief, write, objections=["Why argue this at all?"], seed=0)
    assert gen.strategy == "argued2" and gen.name == "w+brief"
    assert (
        gen.prompt_id == prompt_id_of("BRIEF\n---\nWRITE")
        and gen.prompt_system == "BRIEF\n---\nWRITE"
    )

    target = synth.Target(
        text="A passage long enough to matter.", source="post/x.qmd", chunk_index=0, chunk_total=1
    )
    result = synth.synthesize_pairs([target], tmp_path / "corpus", [gen])
    assert result.written == 1
    rec = corpus_records(tmp_path / "corpus")[0]
    assert rec["messages"][1]["content"] == "[argued2] A passage long enough to matter."
    assert "Why argue this at all?" in seen["brief"] and "rules out B" in seen["write"]
    assert seen["history"] is None
    m = rec["meta"]
    assert m["slop_strategy"] == "argued2"
    assert m["gen"]["prompt_tokens"] == 400 and m["gen"]["cost"] == 0.003
    assert m["gen"]["brief_model"] == "b" and m["gen"]["objection_chars"] == len(
        "Why argue this at all?"
    )
    assert "Why argue" not in json.dumps(rec)
    archived = [
        json.loads(line)
        for line in (tmp_path / "corpus" / "prompts.jsonl").read_text().splitlines()
    ]
    assert archived[0]["system"] == "BRIEF\n---\nWRITE"
