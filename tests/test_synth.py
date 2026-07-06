"""Tests for Phase 2 synthetic-pair generation.

Generators are injected, so these exercise the real `synthesize_pairs` /
`iter_targets` path with deterministic fakes — no API keys, no network — and
gate the output through the same `validate_pairs_file` contract Phase 3 uses.
"""

from __future__ import annotations

from stylebot import synth
from stylebot.pairs import validate_pairs_file


def _fake(name: str) -> synth.Generator:
    # Deterministic "slop": tag the text so source != target but stays non-empty.
    return synth.Generator(name=name, generate=lambda text: f"[{name}-slop] {text}")


HUMAN_POST = """---
title: A Human Post
automation: 0
---

This is a sufficiently long human-authored paragraph about graph theory and the
quiet dignity of well-chosen abstractions, long enough to clear the min-chars bar.

Another paragraph, also comfortably past the minimum length, musing on the way
synthetic data both helps and quietly lies to the model that consumes it.
"""

AI_POST = """---
title: An AI-Touched Post
automation: 2
---

This paragraph was heavily AI-assisted and must never become a training target,
no matter how long and superficially reasonable it manages to appear here.
"""


def _make_blog(tmp_path):
    root = tmp_path / "blog"
    (root / "post").mkdir(parents=True)
    (root / "post" / "human.qmd").write_text(HUMAN_POST, encoding="utf-8")
    (root / "post" / "ai.qmd").write_text(AI_POST, encoding="utf-8")
    return root


def test_selector_keeps_only_human_authored(tmp_path):
    root = _make_blog(tmp_path)
    targets = synth.iter_targets(blog_root=root)
    assert targets, "expected human-authored chunks"
    assert {t.source for t in targets} == {"post/human.qmd"}
    assert len(targets) == 2  # two prose paragraphs in the human post


def test_pairs_validate_and_carry_synthetic_meta(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [_fake("claude-x"), _fake("gpt-y")]
    )
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


def test_rotation_uses_at_least_two_generators(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [_fake("claude-x"), _fake("gpt-y")]
    )
    assert set(result.per_generator) == {"claude-x", "gpt-y"}


def test_idempotent_resume(tmp_path):
    root = _make_blog(tmp_path)
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


def test_per_generator_mode_doubles_pairs(tmp_path):
    root = _make_blog(tmp_path)
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

    result = synth.synthesize_pairs(
        [target], data_dir, [synth.Generator("g", generate=gen)]
    )
    assert result.written == 1
    assert "## A Heading" not in given["arg"]  # generator saw the body only

    pairs_path = data_dir / "pairs.jsonl"
    assert validate_pairs_file(pairs_path) == []  # incl. the context-prefix invariant
    rec = json.loads(pairs_path.read_text().splitlines()[0])
    assert rec["messages"][1]["content"].startswith("## A Heading\n\n")
    assert rec["messages"][2]["content"].startswith("## A Heading\n\n")
    assert rec["meta"]["context"] == "## A Heading"
    assert rec["meta"]["context_mode"] == "immediate"


def test_context_changes_synth_key(tmp_path):
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
    target = synth.Target(text="A paragraph long enough to be a worthwhile target.", source="post/x.qmd", chunk_index=0, chunk_total=1)
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
    target = synth.Target(text="A paragraph long enough to be a worthwhile target.", source="post/x.qmd", chunk_index=0, chunk_total=1)

    def make(strategy):
        return synth.Generator(name="m", generate=lambda t: f"[{strategy}] " + t, strategy=strategy)

    synth.synthesize_pairs([target], data_dir, [make("polish")])
    second = synth.synthesize_pairs([target], data_dir, [make("catalogue")])  # must NOT be skipped
    assert second.written == 1

    recs = [json.loads(ln) for ln in (data_dir / "pairs.jsonl").read_text().splitlines() if ln.strip()]
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


def test_openrouter_generator_name_and_strategy():
    # Construction is offline (no request); pass an explicit key so it doesn't
    # require OPENROUTER_API_KEY from the environment.
    gen = synth.openrouter_generator(model="anthropic/claude-opus-4-8", api_key="x", strategy="catalogue")
    assert gen.name == "openrouter/anthropic/claude-opus-4-8"
    assert gen.strategy == "catalogue"


def test_dry_run_writes_nothing(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [synth.Generator("claude-x"), synth.Generator("gpt-y")], dry_run=True
    )
    assert result.written == 0
    assert result.planned == len(targets)
    assert not (data_dir / "pairs.jsonl").exists()


def test_pre_selected_files_skip_selector(tmp_path):
    # A pre-selected list is taken as-is — even an AI-touched post — because the
    # caller owns selection in that mode.
    root = _make_blog(tmp_path)
    targets = synth.iter_targets(files=[root / "post" / "ai.qmd"])
    assert targets
    assert {t.source for t in targets} == {str(root / "post" / "ai.qmd")}


# --- chunk hygiene -------------------------------------------------------

POST_WITH_DIV = """---
title: Has a figure div
automation: 0
---

A genuine paragraph of prose that is comfortably longer than the minimum and
carries actual voice worth learning to reproduce in the styler.

:::{#fig-x .figure .illustration}
![](/images/x.png)
:::

A second genuine paragraph, also well past the minimum length, with real
sentences that a human plausibly wrote and edited by hand.
"""

POST_WITH_STUB = """---
title: Has a stub
automation: 0
---

A real paragraph long enough to be kept as a synthesis target, written in a
recognisably human register with some actual content.

🚧TODO🚧 flesh this section out later — it is a stub placeholder and must never
become a synthesis target, even though it is long enough to clear the min-chars bar.
"""


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_prose_only_drops_divs(tmp_path):
    root = tmp_path / "blog"
    _write(root, "post/div.qmd", POST_WITH_DIV)
    targets = synth.iter_targets(blog_root=root)
    assert len(targets) == 2  # two prose paragraphs, the :::div excluded
    assert all("fig-x" not in t.text and "images/x.png" not in t.text for t in targets)


def test_ignore_markers_drops_stub_paragraphs(tmp_path):
    root = tmp_path / "blog"
    _write(root, "post/stub.qmd", POST_WITH_STUB)
    kept = synth.iter_targets(blog_root=root, ignore_markers=["🚧TODO🚧"])
    assert len(kept) == 1
    assert all("🚧TODO🚧" not in t.text for t in kept)
    # Without the marker filter, the stub paragraph survives.
    unfiltered = synth.iter_targets(blog_root=root)
    assert len(unfiltered) == 2


def test_link_dump_dropped(tmp_path):
    root = tmp_path / "blog"
    links = "\n".join(f"* [resource {i}](https://example.com/{i})" for i in range(40))
    body = f"""---
title: Blogroll
automation: 0
---

A normal paragraph with a single [link](https://example.com) that should be kept
because it is mostly prose and only incidentally references one URL out there.

{links}
"""
    _write(root, "post/roll.qmd", body)
    targets = synth.iter_targets(blog_root=root)
    assert len(targets) == 1  # prose kept, the link-dump dropped
    assert "resource 0" not in targets[0].text


def test_max_chars_caps_giant_chunk(tmp_path):
    root = tmp_path / "blog"
    giant = "word " * 3000  # ~15k chars, single paragraph
    body = f"---\ntitle: Giant\nautomation: 0\n---\n\n{giant}\n"
    _write(root, "post/giant.qmd", body)
    assert synth.iter_targets(blog_root=root) == []
    # Raising the cap keeps it.
    assert len(synth.iter_targets(blog_root=root, max_chars=None)) == 1


def test_backward_compat_defaults_unchanged(tmp_path):
    # The original two-paragraph human post still yields exactly two targets.
    root = _make_blog(tmp_path)
    assert len(synth.iter_targets(blog_root=root)) == 2


POST_WITH_INCOMING = """---
title: Has an Incoming dump
automation: 0
---

A real authored paragraph that should survive, long enough to clear the minimum
and carrying genuine voice worth learning from in the styler.

## Incoming

A pile of undigested quotes and links with zero authored signal, long enough to
otherwise look like a real paragraph but sitting below the cut header.

Another undigested blob that must also be excluded along with everything after
the Incoming header, no matter how prose-like it appears.
"""


def test_stop_at_headers_truncates_trailing_section(tmp_path):
    root = tmp_path / "blog"
    _write(root, "post/incoming.qmd", POST_WITH_INCOMING)
    kept = synth.iter_targets(blog_root=root, stop_at_headers=["## Incoming"])
    assert len(kept) == 1  # only the paragraph before "## Incoming"
    assert "authored paragraph that should survive" in kept[0].text
    assert all("undigested" not in t.text for t in kept)
    # Level-agnostic / case-insensitive: "### incoming" also cuts.
    assert len(synth.iter_targets(blog_root=root, stop_at_headers=["incoming"])) == 1
    # Without the option, the post-Incoming paragraphs survive.
    assert len(synth.iter_targets(blog_root=root)) == 3


# ---------------------------------------------------------------------------
# Generator request-shaping + response guards + recorded covariates. These mock
# the openai client to assert the request we send (reasoning-effort mapping,
# sampling) and the GenOutput meta we record — no network, no keys.
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content="paraphrase", finish_reason="stop"):
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=None, completion_tokens=None):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeResponse:
    def __init__(self, choices, usage=None):
        self.choices = choices
        self.usage = usage


def _patch_openai(monkeypatch, response):
    """Patch ``openai.AsyncOpenAI`` with a client that records ``create()`` kwargs."""
    calls: dict = {}

    class _Completions:
        async def create(self, **kwargs):
            calls.update(kwargs)
            return response

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

        def __init__(self, **kwargs):
            pass

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    return calls


def _run_gen(gen, text, history=None):
    """Drive a factory generator's async `generate` from a sync test."""
    import asyncio

    return asyncio.run(gen.generate(text, history=history))


def test_openrouter_reasoning_effort_enum(monkeypatch):
    # Effort-enum families (claude/gpt/…) get {"effort": <level>}; default is high.
    # Provider routing defaults to throughput sort (price-balanced routing can
    # land on ~10 tok/s upstreams).
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice("slop out")]))
    out = _run_gen(synth.openrouter_generator(model="anthropic/claude-opus-4.8", api_key="x"), "rewrite me")
    assert out.text == "slop out"  # GenOutput, not a bare string
    assert calls["extra_body"] == {"reasoning": {"effort": "high"}, "provider": {"sort": "throughput"}, "usage": {"include": True}}
    assert out.meta["provider_sort"] == "throughput"  # the routing request, recorded


def test_openrouter_reasoning_budget_family(monkeypatch):
    # Budget-style families (qwen/nvidia/google/…) get a token budget instead.
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", reasoning_effort="medium"), "x")
    assert calls["extra_body"] == {"reasoning": {"max_tokens": 4000}, "provider": {"sort": "throughput"}, "usage": {"include": True}}


def test_openrouter_reasoning_off(monkeypatch):
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", reasoning_effort="off"), "x")
    assert calls["extra_body"] == {"reasoning": {"enabled": False}, "provider": {"sort": "throughput"}, "usage": {"include": True}}


def test_openrouter_provider_sort_none_omits_field(monkeypatch):
    # provider_sort=None restores OpenRouter's own load-balancing: no provider
    # field on the wire, no provider_sort covariate recorded.
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    out = _run_gen(synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", provider_sort=None), "x")
    assert "provider" not in calls["extra_body"]
    assert "provider_sort" not in out.meta


def test_reasoning_effort_recorded_regardless_of_shape(monkeypatch):
    # A budget-family model still RECORDS the requested effort string ("medium"),
    # not the wire budget int — the covariate is the request, not what the API did.
    _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    out = _run_gen(synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", reasoning_effort="medium"), "x")
    assert out.meta["reasoning_effort"] == "medium"


def test_max_tokens_threaded(monkeypatch):
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", max_tokens=1234), "x")
    assert calls["max_tokens"] == 1234


def test_temperature_top_p_passed_and_recorded(monkeypatch):
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()], usage=_FakeUsage(11, 22)))
    out = _run_gen(
        synth.openrouter_generator(model="anthropic/claude-opus-4.8", api_key="x", temperature=0.3, top_p=0.9), "x"
    )
    assert calls["temperature"] == 0.3 and calls["top_p"] == 0.9
    assert out.meta["temperature"] == 0.3 and out.meta["top_p"] == 0.9
    assert out.meta["prompt_tokens"] == 11 and out.meta["completion_tokens"] == 22


def test_temperature_omitted_when_none(monkeypatch):
    # Unset sampling params are NOT sent, so providers keep their defaults.
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(synth.openrouter_generator(model="anthropic/claude-opus-4.8", api_key="x"), "x")
    assert "temperature" not in calls and "top_p" not in calls


def test_empty_choices_raises_clear_error(monkeypatch):
    import pytest

    _patch_openai(monkeypatch, _FakeResponse(None))  # provider returned choices=None
    gen = synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x")
    with pytest.raises(RuntimeError, match="no choices"):  # not an opaque TypeError
        _run_gen(gen, "rewrite me")


def test_truncated_slop_raises(monkeypatch):
    import pytest

    _patch_openai(monkeypatch, _FakeResponse([_FakeChoice(finish_reason="length")]))
    gen = synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x")
    with pytest.raises(RuntimeError, match="truncated"):
        _run_gen(gen, "rewrite me")


def test_gen_output_meta_recorded(tmp_path):
    # A generator returning GenOutput records its covariates under meta.gen.
    import json

    data_dir = tmp_path / "corpus"
    target = synth.Target(text="A paragraph long enough to be a worthwhile target.", source="post/x.qmd", chunk_index=0, chunk_total=1)
    gen = synth.Generator(
        name="m",
        generate=lambda t: synth.GenOutput("[slop] " + t, {"model": "m", "reasoning_effort": "low", "completion_tokens": 5}),
    )
    synth.synthesize_pairs([target], data_dir, [gen])
    rec = json.loads((data_dir / "pairs.jsonl").read_text().splitlines()[0])
    assert rec["meta"]["gen"] == {"model": "m", "reasoning_effort": "low", "completion_tokens": 5}
    assert validate_pairs_file(data_dir / "pairs.jsonl") == []


def test_bare_string_generator_has_no_gen_meta(tmp_path):
    # Bare-string fakes still work (back-compat) and record no meta.gen.
    import json

    data_dir = tmp_path / "corpus"
    target = synth.Target(text="A paragraph long enough to be a worthwhile target.", source="post/x.qmd", chunk_index=0, chunk_total=1)
    synth.synthesize_pairs([target], data_dir, [synth.Generator(name="m", generate=lambda t: "[slop] " + t)])
    rec = json.loads((data_dir / "pairs.jsonl").read_text().splitlines()[0])
    assert "gen" not in rec["meta"]


# ---------------------------------------------------------------------------
# Parallelism + live sessions (async engine, window-position covariates)
# ---------------------------------------------------------------------------


def _targets(n: int) -> list[synth.Target]:
    return [
        synth.Target(
            text=f"Paragraph {i}: " + "sufficiently long prose about abstraction. " * 3,
            source=f"post/{i % 3}.qmd",
            chunk_index=i,
            chunk_total=n,
        )
        for i in range(n)
    ]


def test_parallel_stateless_run_writes_each_pair_once(tmp_path):
    import json

    targets = _targets(10)
    gens = [_fake("a"), _fake("b")]
    first = synth.synthesize_pairs(targets, tmp_path / "c", gens, max_workers=4)
    assert first.written == 10
    assert first.per_generator == {"a": 5, "b": 5}

    second = synth.synthesize_pairs(targets, tmp_path / "c", gens, max_workers=4)
    assert second.written == 0 and second.skipped_existing == 10

    keys = [
        json.loads(ln)["meta"]["synth_key"]
        for ln in (tmp_path / "c" / "pairs.jsonl").read_text().splitlines()
    ]
    assert len(keys) == 10 == len(set(keys))


def test_async_injected_generator_works(tmp_path):
    async def gen(text, history=None):
        return "[async-slop] " + text

    result = synth.synthesize_pairs(_targets(2), tmp_path / "c", [synth.Generator("ag", generate=gen)])
    assert result.written == 2


def test_stateless_keys_have_no_session_component():
    # session_turns=1 must key exactly like bare _synth_key, independent of the
    # rest of the target list — corpus resume must survive the blog growing.
    t = _targets(1)[0]
    g = synth.Generator(name="g")
    [sess] = synth._plan_sessions([(t, g, "")], session_turns=1)
    assert sess.session_id == ""
    assert sess.turns[0].key == synth._synth_key("g", t.text)


def test_session_keys_distinct_per_turn_and_stable():
    t1, t2 = _targets(2)
    g = synth.Generator(name="g")
    plan = lambda: synth._plan_sessions([(t1, g, ""), (t2, g, "")], session_turns=2)  # noqa: E731
    [sess] = plan()
    keys = [t.key for t in sess.turns]
    assert len(set(keys)) == 2
    assert keys[0] != synth._synth_key("g", t1.text)  # session component folded in
    assert [t.key for t in plan()[0].turns] == keys  # deterministic replan
    # Even a duplicated target text gets distinct per-turn keys inside a session.
    [dup] = synth._plan_sessions([(t1, g, ""), (t1, g, "")], session_turns=2)
    assert len({t.key for t in dup.turns}) == 2


def test_session_history_accumulates_and_covariates_recorded(tmp_path):
    import json

    targets = _targets(3)
    seen_hist: list[list[dict]] = []

    def gen(text, history=None):
        seen_hist.append(list(history or []))
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 100 * (len(seen_hist))})

    result = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=gen)],
        session_turns=3, context_windows={"g": 10_000},
    )
    assert result.written == 3
    assert [len(h) for h in seen_hist] == [0, 2, 4]  # turn N sees N-1 exchanges
    assert seen_hist[1][0] == {"role": "user", "content": targets[0].text}
    assert seen_hist[1][1]["content"] == "[slop] " + targets[0].text

    recs = [json.loads(ln) for ln in (tmp_path / "c" / "pairs.jsonl").read_text().splitlines()]
    assert [r["meta"]["gen"]["session_turn"] for r in recs] == [1, 2, 3]
    assert len({r["meta"]["gen"]["session_id"] for r in recs}) == 1
    assert recs[1]["meta"]["gen"]["context_window"] == 10_000
    assert recs[1]["meta"]["gen"]["window_fill"] == round(200 / 10_000, 4)


def test_session_resume_replays_recorded_history(tmp_path):
    targets = _targets(3)
    calls = {"n": 0}

    def flaky(text, history=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("boom")
        return "[slop] " + text

    first = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=flaky)], session_turns=3
    )
    assert first.written == 2 and len(first.errors) == 1

    hists: list[list[dict]] = []

    def good(text, history=None):
        hists.append(list(history or []))
        return "[slop] " + text

    second = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=good)], session_turns=3
    )
    assert second.written == 1 and second.skipped_existing == 2
    # The retried turn saw the two recorded exchanges replayed from the file.
    assert len(hists) == 1 and len(hists[0]) == 4
    assert hists[0][1]["content"] == "[slop] " + targets[0].text


def test_turn_error_ends_only_its_session(tmp_path):
    targets = _targets(4)  # round-robin: bad gets 0,2; good gets 1,3

    def bad(text, history=None):
        raise RuntimeError("api down")

    result = synth.synthesize_pairs(
        targets, tmp_path / "c",
        [synth.Generator("bad", generate=bad), synth.Generator("good", generate=lambda t, history=None: "[slop] " + t)],
        session_turns=2,
    )
    assert result.per_generator.get("good") == 2
    assert "bad" not in result.per_generator
    assert len(result.errors) == 1  # session ended at its first failure


def test_session_budget_stops_session(tmp_path):
    targets = _targets(3)

    def gen(text, history=None):
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 10_000, "completion_tokens": 50})

    result = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=gen)],
        session_turns=3, session_max_tokens=100,
    )
    assert result.written == 1  # turn 2 estimate blows the budget
    assert not result.errors  # a budget stop is not an error


def test_generator_session_budget_beats_global(tmp_path):
    targets = _targets(3)

    def gen(text, history=None):
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 60})

    g = synth.Generator("g", generate=gen, session_budget=50)
    result = synth.synthesize_pairs(
        targets, tmp_path / "c", [g], session_turns=3, session_max_tokens=1_000_000
    )
    assert result.written == 1


def test_window_cap_stops_session(tmp_path):
    targets = _targets(3)

    def gen(text, history=None):
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 790, "completion_tokens": 50})

    result = synth.synthesize_pairs(
        targets, tmp_path / "c", [synth.Generator("g", generate=gen)],
        session_turns=3, session_max_tokens=None, context_windows={"g": 1000},
    )
    # cap = 0.8 * 1000 = 800; next-turn estimate from prompt_tokens=790 exceeds it.
    assert result.written == 1


def test_openrouter_context_windows_registry(monkeypatch):
    import io
    import json as _json
    import urllib.request

    payload = {"data": [
        {"id": "a/b", "context_length": 32768},
        {"id": "c/d", "context_length": 200000},
        {"id": "e/no-window"},
    ]}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(url, timeout=None):
        assert url.endswith("/models")
        return _Resp(_json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    windows = synth.openrouter_context_windows(base_url="https://registry.test/api/v1")
    assert windows == {"a/b": 32768, "c/d": 200000}
