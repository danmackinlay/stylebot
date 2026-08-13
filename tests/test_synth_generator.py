"""Phase 2 generator wire behaviour: the request we send, the response we trust.

Carved verbatim out of ``test_synth.py``. These mock the openai client — no
network, no keys — and assert the OpenRouter request shape (reasoning effort,
sampling, provider routing, cache breakpoints) and the ``GenOutput`` meta the
run records.
"""

from __future__ import annotations

from stylebot import synth


def test_openrouter_generator_name_and_strategy():
    # Construction is offline (no request); pass an explicit key so it doesn't
    # require OPENROUTER_API_KEY from the environment.
    gen = synth.openrouter_generator(
        model="anthropic/claude-opus-4-8", api_key="x", strategy="casual"
    )
    assert gen.name == "openrouter/anthropic/claude-opus-4-8"
    assert gen.strategy == "casual"


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
    # Effort-enum families (claude/gpt/…) get {"effort": <level>}. Asked for
    # explicitly: the DEFAULT is "off" (see DEFAULT_REASONING_EFFORT), which maps to
    # {"enabled": False} and is covered below — this case is the enum mapping itself.
    # Provider routing defaults to throughput sort (price-balanced routing can
    # land on ~10 tok/s upstreams).
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice("slop out")]))
    out = _run_gen(
        synth.openrouter_generator(
            model="anthropic/claude-opus-4.8", api_key="x", reasoning_effort="high"
        ),
        "rewrite me",
    )
    assert out.text == "slop out"  # GenOutput, not a bare string
    assert calls["extra_body"] == {
        "reasoning": {"effort": "high"},
        "provider": {"sort": "throughput"},
        "usage": {"include": True},
    }
    assert out.meta["provider_sort"] == "throughput"  # the routing request, recorded


def test_gen_meta_keys_match_schema_constants(monkeypatch):
    # The recorded meta.gen dict must match the schema constants in
    # stylebot.pairs EXACTLY — eval and report facet by those names, so a key
    # renamed here but not there silently produces empty columns downstream.
    #
    # Both directions, deliberately. `writer ⊆ declaration` alone (what this
    # asserted until 0.3.1) lets the declaration drift AHEAD of the writer: a
    # key nothing emits sits in the list looking like a schema, and the facet
    # built on it comes back empty — indistinguishable from "that covariate
    # never varied", which is a conclusion, not a bug report.
    from stylebot import pairs

    calls = _patch_openai(
        monkeypatch,
        _FakeResponse([_FakeChoice("slop out")], _FakeUsage(prompt_tokens=10, completion_tokens=5)),
    )
    out = _run_gen(synth.openrouter_generator(model="anthropic/claude-opus-4.8", api_key="x"), "x")
    assert calls  # the fake client was exercised
    # openrouter_generator = the base keys plus its own extra_meta contribution.
    assert set(out.meta) == set(pairs.GEN_BASE_KEYS) | set(pairs.GEN_EXTRA_KEYS)


def test_gen_meta_base_keys_are_exactly_the_plain_writer(monkeypatch):
    # The same equality against `openai_generator` alone, which contributes no
    # extra_meta — so GEN_BASE_KEYS is pinned independently of OpenRouter's
    # routing knobs rather than only as a subtraction from them.
    from stylebot import pairs
    from stylebot.generators import openai_generator

    _patch_openai(monkeypatch, _FakeResponse([_FakeChoice("slop")], _FakeUsage()))
    out = _run_gen(openai_generator(model="some/model", api_key="x"), "x")
    assert set(out.meta) == set(pairs.GEN_BASE_KEYS)


def test_openrouter_reasoning_budget_family(monkeypatch):
    # Budget-style families (qwen/nvidia/google/…) get a token budget instead.
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(
        synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", reasoning_effort="medium"),
        "x",
    )
    assert calls["extra_body"] == {
        "reasoning": {"max_tokens": 4000},
        "provider": {"sort": "throughput"},
        "usage": {"include": True},
    }


def test_openrouter_reasoning_off(monkeypatch):
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(
        synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", reasoning_effort="off"), "x"
    )
    assert calls["extra_body"] == {
        "reasoning": {"enabled": False},
        "provider": {"sort": "throughput"},
        "usage": {"include": True},
    }


def test_openrouter_provider_sort_none_omits_field(monkeypatch):
    # provider_sort=None restores OpenRouter's own load-balancing: no provider
    # field on the wire, no provider_sort covariate recorded.
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    out = _run_gen(
        synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", provider_sort=None), "x"
    )
    assert "provider" not in calls["extra_body"]
    assert "provider_sort" not in out.meta


def test_reasoning_effort_recorded_regardless_of_shape(monkeypatch):
    # A budget-family model still RECORDS the requested effort string ("medium"),
    # not the wire budget int — the covariate is the request, not what the API did.
    _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    out = _run_gen(
        synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", reasoning_effort="medium"),
        "x",
    )
    assert out.meta["reasoning_effort"] == "medium"


def test_max_tokens_threaded(monkeypatch):
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()]))
    _run_gen(synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", max_tokens=1234), "x")
    assert calls["max_tokens"] == 1234


def test_temperature_top_p_passed_and_recorded(monkeypatch):
    calls = _patch_openai(monkeypatch, _FakeResponse([_FakeChoice()], usage=_FakeUsage(11, 22)))
    out = _run_gen(
        synth.openrouter_generator(
            model="anthropic/claude-opus-4.8", api_key="x", temperature=0.3, top_p=0.9
        ),
        "x",
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
    # The message says where the tokens went (completion vs reasoning) and
    # names both remedies — that is the attribution for a never-written pair.
    with pytest.raises(RuntimeError, match="truncated.*reasoning-effort"):
        _run_gen(gen, "rewrite me")


def test_openrouter_context_windows_registry(monkeypatch):
    import io
    import json as _json
    import urllib.request

    payload = {
        "data": [
            {"id": "a/b", "context_length": 32768},
            {"id": "c/d", "context_length": 200000},
            {"id": "e/no-window"},
        ]
    }

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


# ---------------------------------------------------------------------------
# Session cost hygiene: sticky provider routing + Anthropic cache breakpoints
# ---------------------------------------------------------------------------


def _patch_openai_multi(monkeypatch, response):
    """Like _patch_openai but captures kwargs per call (list, not last-wins)."""
    seen: list[dict] = []

    class _Completions:
        async def create(self, **kwargs):
            seen.append(kwargs)
            return response

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

        def __init__(self, **kwargs):
            pass

    import openai

    monkeypatch.setattr(openai, "AsyncOpenAI", _Client)
    return seen


def test_sticky_provider_pins_session_after_first_turn(monkeypatch):
    import asyncio

    resp = _FakeResponse([_FakeChoice()])
    resp.provider = "Groq"
    seen = _patch_openai_multi(monkeypatch, resp)
    gen = synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x")

    session_call = gen.begin_session()
    hist = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "s"}]
    asyncio.run(session_call("one"))
    asyncio.run(session_call("two", history=hist))
    assert seen[0]["extra_body"]["provider"] == {"sort": "throughput"}  # turn 1 routes free
    assert seen[1]["extra_body"]["provider"] == {"order": ["Groq"], "allow_fallbacks": False}

    # The stateless path never pins; a NEW session re-pins from scratch.
    asyncio.run(gen.generate("three"))
    assert seen[2]["extra_body"]["provider"] == {"sort": "throughput"}
    asyncio.run(gen.begin_session()("four"))
    assert seen[3]["extra_body"]["provider"] == {"sort": "throughput"}


def test_no_sticky_provider_disables_begin_session(monkeypatch):
    _patch_openai_multi(monkeypatch, _FakeResponse([_FakeChoice()]))
    gen = synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", sticky_provider=False)
    assert gen.begin_session is None


def test_anthropic_cache_breakpoint_marks_last_history_message(monkeypatch):
    import asyncio

    seen = _patch_openai_multi(monkeypatch, _FakeResponse([_FakeChoice()]))
    hist = [{"role": "user", "content": "p1"}, {"role": "assistant", "content": "s1"}]

    gen = synth.openrouter_generator(model="anthropic/claude-sonnet-4.6", api_key="x")
    asyncio.run(gen.generate("p2", history=hist))
    last_hist_msg = seen[0]["messages"][-2]
    assert last_hist_msg["content"] == [
        {"type": "text", "text": "s1", "cache_control": {"type": "ephemeral"}}
    ]
    assert seen[0]["messages"][-1] == {"role": "user", "content": "p2"}

    # No history -> nothing to cache, plain string content everywhere.
    asyncio.run(gen.generate("p2"))
    assert all(isinstance(m["content"], str) for m in seen[1]["messages"])

    # Non-Anthropic models are never marked (their providers cache
    # automatically or not at all; structured content buys nothing).
    gen_qwen = synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x")
    asyncio.run(gen_qwen.generate("p2", history=hist))
    assert seen[2]["messages"][-2]["content"] == "s1"

    # And --no-prompt-cache turns it off for Anthropic too.
    gen_off = synth.openrouter_generator(
        model="anthropic/claude-sonnet-4.6", api_key="x", prompt_cache=False
    )
    asyncio.run(gen_off.generate("p2", history=hist))
    assert seen[3]["messages"][-2]["content"] == "s1"


def test_wire_capture_reasoning_and_truncation_tail(monkeypatch):
    import pytest

    # Provider sends a reasoning trace: capture puts it in gen_meta...
    resp = _FakeResponse([_FakeChoice("slop out")], usage=_FakeUsage(10, 20))
    resp.choices[0].message.reasoning = "Step 1: overthink. Step 2: overthink more."
    _patch_openai(monkeypatch, resp)
    out = _run_gen(
        synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x", capture_reasoning=True), "x"
    )
    assert out.meta["reasoning_text"].startswith("Step 1")

    # ...and a truncated response includes the trace tail in the error even
    # without the capture flag (that IS the truncation diagnosis).
    trunc = _FakeResponse([_FakeChoice(finish_reason="length")])
    trunc.choices[0].message.reasoning = "x" * 500 + " FINAL THOUGHT"
    _patch_openai(monkeypatch, trunc)
    gen = synth.openrouter_generator(model="qwen/qwen3-8b", api_key="x")
    with pytest.raises(RuntimeError, match="FINAL THOUGHT"):
        _run_gen(gen, "rewrite me")
