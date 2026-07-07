"""Generators — the slop producers: strategy prompts + provider factories.

The slop-prompt registry (`STRATEGIES`, `resolve_strategy`, `prompt_id_of`),
the `Generator`/`GenOutput` contract, and the three provider factories
(`openai_generator`, `local_generator`, `openrouter_generator`) with their
wire-level helpers (reasoning mapping, the OpenRouter models registry).

`stylebot.synth` re-exports every public name here — external callers keep
importing via `stylebot.synth`, and tests monkeypatch the factories on that
module; this module is the implementation home.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

# Instruction we hand a generic LLM to manufacture "slop" from Dan's prose.
# It mirrors STYLE_SYSTEM's structure-preservation clause so the synthetic
# source differs from the target in *style*, not markdown shape — we want the
# styler to learn the voice transform, not a reformatting.
# Shared tail for every slop strategy: the formatting contract (preserve
# structure, return only the passage). Identical across strategies so the only
# thing that varies between them is the *flavour* of slop requested.
_SLOP_PRESERVE = (
    "Preserve all markdown structure (code fences, math, links, headings, "
    "list markers, blank lines) verbatim. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten passage, nothing else."
)

# Named slop strategies: label -> the system prompt that produces that flavour of
# slop. The label is recorded as `meta.slop_strategy` and folded into `synth_key`,
# so pairs from different strategies neither collide on resume nor blur together —
# you can ablate "which flavour of slop teaches the styler best". These are
# GENERIC AI-prose flavours; an author's own slop catalogue is injected as a
# custom prompt (CLI `--slop-system-file` / library `system=`), keeping stylebot
# free of any one author's slop definition.
SLOP_SYSTEM = (  # "polish": the neutral baseline (clearer / more professional)
    "You are a writing assistant that polishes prose. Rewrite the user's "
    "passage to be clearer, more professional, and more engaging. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_ENGAGING = (  # "engaging": hooks, signposting, surfaced takeaways
    "You are an enthusiastic content editor. Rewrite the user's passage to be "
    "maximally engaging and accessible to a broad audience: open with a hook, "
    "add helpful signposting, surface the key takeaways, and keep the reader "
    "moving. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_CASUAL = (  # "casual": the friendly-technical-blog register
    "You are a friendly technical blogger. Rewrite the user's passage in the "
    "approachable register of a popular developer blog: conversational and "
    "upbeat, address the reader as 'you', break dense phrasing into short "
    "clear sentences, add light connective tissue ('so', 'basically', 'the "
    "nice thing is'), and round each idea off so nothing lands abruptly. "
    + _SLOP_PRESERVE
)
SLOP_SYSTEM_MEASURED = (  # "measured": the mild stereotypical-LLM register — texture, not tokens
    "You are a careful AI writing assistant. Rewrite the user's passage in a "
    "measured, well-organized explanatory register: smooth transitions, gentle "
    "hedging of strong claims, tidy structure, slightly more formal vocabulary, "
    "and an even sentence rhythm. Let the register show in the overall texture, "
    "not in any stock phrase — do not open with generic scene-setting, and vary "
    "your sentence openings and structure naturally from passage to passage. "
    + _SLOP_PRESERVE
)
# "measured" replaces a removed "catalogue" strategy (2026-07-07) that QUOTED
# the stereotypical tics ("In today's world", ...) — samplers converge hard on
# quoted tokens, so every output opened identically and the slop was cartoonish.
# Describing the texture and banning stock phrases keeps the register while
# preserving output diversity. Old catalogue pairs remain resolvable via their
# data-dir's prompts.jsonl; the prompt text is in git history.


@dataclass(frozen=True)
class SlopStrategy:
    """A named slop-prompt flavour: a human label, the system prompt, a version.

    `version` is bumped by hand when the prompt text changes meaningfully; the
    stable `prompt_id` (a content hash, see `prompt_id_of`) is what actually
    identifies the prompt for faceting/dedup, so editing a prompt changes its id
    regardless of the version bump.
    """

    label: str
    system: str
    version: int = 1


STRATEGIES: dict[str, SlopStrategy] = {
    "polish": SlopStrategy("polish", SLOP_SYSTEM, version=1),
    "engaging": SlopStrategy("engaging", SLOP_SYSTEM_ENGAGING, version=1),
    "casual": SlopStrategy("casual", SLOP_SYSTEM_CASUAL, version=1),
    "measured": SlopStrategy("measured", SLOP_SYSTEM_MEASURED, version=1),
}
DEFAULT_STRATEGY = "polish"

# Reasoning is a recorded *covariate*, not a silent default. Slop generation is a
# paraphrase, but real AI prose is often produced at high reasoning, so we default
# HIGH and let experiments sweep down (see `_reasoning_extra_body`).
DEFAULT_REASONING_EFFORT = "high"


def prompt_id_of(system_text: str) -> str:
    """Stable content id for ANY slop system prompt (registry or custom file).

    Hashing the actual prompt text means a custom `--slop-system-file` gets a
    stable id and is faceted/deduped exactly like a registry strategy, and editing
    a registry prompt changes its id (so old and new pairs stay distinguishable).
    """
    return hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:12]


class ResolvedStrategy(NamedTuple):
    """A slop strategy resolved to its prompt-content identity.

    A NamedTuple on purpose: existing ``label, system, version, prompt_id``
    unpack sites keep working; new code reads the attributes. ``prompt_id``
    (content hash) and ``version`` feed ``synth_key`` / ``meta.gen``.
    """

    label: str
    system: str
    version: int
    prompt_id: str


def resolve_strategy(name: str, system: str | None = None) -> ResolvedStrategy:
    """Resolve a strategy name to a `ResolvedStrategy`.

    An explicit ``system`` overrides the registry, so a caller can inject a custom
    (e.g. blog-specific) slop prompt under any label without stylebot needing to
    know that author's catalogue; such a prompt has version 0 and is identified by
    its content hash. A name absent from the registry is an error *unless* an
    explicit ``system`` is supplied.
    """
    if system is not None:
        return ResolvedStrategy(name, system, 0, prompt_id_of(system))
    try:
        strat = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(
            f"unknown slop strategy {name!r}; known: {known} "
            f"(or pass an explicit system prompt / --slop-system-file)"
        ) from None
    return ResolvedStrategy(strat.label, strat.system, strat.version, prompt_id_of(strat.system))

# Generous output budget for slop generation. Slop is an *expansion* of the
# target (AI prose runs longer than the human source — often 1.5-3x), so the
# cap must comfortably exceed the target's own token count, not match it. With
# targets capped at MAX_CHUNK_CHARS (~2k tokens), ~8k output tokens leaves room
# for 3-4x expansion. Well under every provider's non-streaming ceiling.
DEFAULT_SLOP_MAX_TOKENS = 8192
# Per-request HTTP timeout for slop generation. Without one, the openai SDK
# waits 600s per attempt (x its automatic retries) — a bad upstream stalls a
# sequential run for half an hour in silence. 300s clears even slow
# high-reasoning generations (~60-120s observed) with headroom; a timed-out
# pair is recorded in SynthResult.errors and the run continues.
DEFAULT_GEN_TIMEOUT = 300.0




@dataclass(frozen=True)
class GenOutput:
    """A generator's output: the slop text plus per-call generation covariates.

    A generator's `generate` may return a bare ``str`` (test fakes / simple
    callables) or a ``GenOutput`` whose ``meta`` carries the recorded generation
    covariates (model, reasoning_effort, temperature, top_p, max_tokens, token
    usage, finish_reason, prompt id/version). `synthesize_pairs` coerces either via
    `_normalize_gen_output`, so bare-string callables keep working unchanged.
    """

    text: str
    meta: dict = field(default_factory=dict)


def _normalize_gen_output(out: "str | GenOutput") -> tuple[str, dict]:
    """Coerce a generator return (``str`` or ``GenOutput``) to ``(text, gen_meta)``."""
    if isinstance(out, GenOutput):
        return out.text, dict(out.meta)
    return out, {}


@dataclass
class Generator:
    """A named slop producer.

    `name` becomes `meta.generator` (the model id); `strategy` becomes
    `meta.slop_strategy` (which slop *prompt* produced the pair). `reasoning_effort`
    and `prompt_id` also feed the `synth_key`, so the same model under two
    strategies / reasoning levels / prompts yields distinct, non-colliding pairs.

    `generate` may be sync or async (the session loop awaits the result only if
    it is awaitable) and may return a bare `str` or a `GenOutput` (text +
    recorded covariates). A plain 1-arg callable is fine for stateless
    (`session_turns=1`) runs; multi-turn sessions call it with a `history`
    kwarg (list of ``{"role","content"}`` messages), so a session-capable
    callable must accept `(text, history=None)`.

    `session_budget` optionally overrides the global per-session prompt-token
    budget for this generator (policy hook — normally unset; the registry
    window × `SESSION_WINDOW_FILL_CAP` still caps it).

    `begin_session`, when set, returns a fresh generate-callable holding
    per-session state; the session loop calls it once per multi-turn session
    (openrouter uses it for sticky provider routing — stay on whichever
    provider served turn 1, so its prefix cache stays hot and the serving
    stack is constant within a session).
    """

    name: str
    generate: Callable[..., "str | GenOutput"] | None = None
    strategy: str = DEFAULT_STRATEGY
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    prompt_id: str = ""
    prompt_version: int = 0
    # Full system-prompt text (factories set it); synthesize_pairs archives it
    # to <data-dir>/prompts.jsonl so a prompt_id is always resolvable to the
    # exact prompt that produced the pairs sitting next to it.
    prompt_system: str = ""
    session_budget: int | None = None
    begin_session: Callable[[], Callable[..., "str | GenOutput"]] | None = None

    def __call__(self, target_text: str, history: list[dict] | None = None):
        if self.generate is None:
            raise RuntimeError(f"generator {self.name!r} has no callable (dry-run/name-only stub)")
        if history:
            return self.generate(target_text, history=history)
        return self.generate(target_text)


# Approximate per-family reasoning budgets for upstreams that take a token budget
# instead of an effort enum.
_REASONING_MAX_TOKENS = {"high": 8000, "medium": 4000, "low": 1500}
# OpenRouter model-id prefixes whose upstreams take a `max_tokens` reasoning budget
# rather than the OpenAI/Anthropic `effort` enum (best-effort; OpenRouter normalizes
# the rest, and the REQUESTED effort is recorded regardless of the wire shape).
_REASONING_BUDGET_FAMILIES = ("google/", "qwen/", "nvidia/", "deepseek/")


def _reasoning_extra_body(model: str, effort: str) -> dict | None:
    """Map a requested reasoning effort to OpenRouter's `reasoning` request field.

    `off` disables reasoning; budget-style families get a token budget; everyone
    else gets the effort enum. Best-effort across heterogeneous upstreams — the
    *requested* effort is recorded in `meta.gen` independent of what the provider
    honors, and `finish_reason`/`completion_tokens` let you detect a model that
    reasoned anyway.
    """
    if effort == "off":
        return {"reasoning": {"enabled": False}}
    if model.startswith(_REASONING_BUDGET_FAMILIES):
        return {"reasoning": {"max_tokens": _REASONING_MAX_TOKENS[effort]}}
    return {"reasoning": {"effort": effort}}


_CONTEXT_WINDOWS_CACHE: dict[str, dict[str, int]] = {}


def openrouter_context_windows(base_url: str | None = None) -> dict[str, int]:
    """Fetch ``{model_id: context_length}`` from the OpenRouter models registry.

    Ground truth for per-model window sizes (hand-annotating them would go
    stale). The endpoint is keyless; one GET per base_url per process
    (module-level cache). Raises URLError/HTTPError on network failure — the
    caller decides whether windows are required.
    """
    from urllib.request import urlopen

    base = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
    if base not in _CONTEXT_WINDOWS_CACHE:
        with urlopen(f"{base}/models", timeout=30) as resp:
            data = json.load(resp)
        _CONTEXT_WINDOWS_CACHE[base] = {
            m["id"]: int(m["context_length"])
            for m in data.get("data", [])
            if m.get("id") and m.get("context_length")
        }
    return _CONTEXT_WINDOWS_CACHE[base]


def _reasoning_text_of(message) -> str | None:
    """The reasoning/thinking trace of a response message, if the provider sent one.

    OpenRouter normalizes most reasoning models to `message.reasoning` (a plain
    string); some providers instead return `message.reasoning_details` (a list
    of typed blocks). Returns None when neither is present (non-reasoning model,
    reasoning disabled, or an upstream that withholds traces).
    """
    text = getattr(message, "reasoning", None)
    if text:
        return text
    details = getattr(message, "reasoning_details", None)
    if details:
        parts = [d.get("text") if isinstance(d, dict) else getattr(d, "text", None) for d in details]
        joined = "\n".join(p for p in parts if p)
        return joined or None
    return None


def openai_generator(
    *,
    model: str = "gpt-4o",
    strategy: str = DEFAULT_STRATEGY,
    system: str | None = None,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    name: str | None = None,
    extra_body: dict | None = None,
    extra_meta: dict | None = None,
    timeout: float | None = DEFAULT_GEN_TIMEOUT,
    sticky_provider: bool = False,
    cache_breakpoints: bool = False,
    capture_reasoning: bool = False,
) -> Generator:
    """OpenAI-compatible slop generator (`openai` SDK; key `OPENAI_API_KEY`).

    `base_url` repoints at any OpenAI-compatible endpoint — `local_generator` and
    `openrouter_generator` use that to drive a base model / OpenRouter. `extra_body`
    passes provider-specific knobs through (e.g. OpenRouter's `reasoning` field, set
    by `openrouter_generator`). `reasoning_effort` is recorded verbatim as the
    *requested* covariate regardless of whether/how the provider honors it; sampling
    params (`temperature`/`top_p`) are sent only when set, and recorded. `generate`
    returns a `GenOutput` carrying these covariates plus token usage.
    """
    import openai

    from stylebot import config

    label, system, prompt_version, prompt_id = resolve_strategy(strategy, system)
    # Async client: synthesize_pairs runs sessions concurrently on an event
    # loop (no threads, no locks); the sync CLI surface is preserved by an
    # asyncio.run inside synthesize_pairs.
    client = openai.AsyncOpenAI(
        api_key=api_key or config.require_key("OPENAI_API_KEY"),
        base_url=base_url,
        timeout=timeout,
    )

    def _make_generate(pin_state: dict | None):
        async def generate(text: str, history: list[dict] | None = None) -> GenOutput:
            messages = [
                {"role": "system", "content": system},
                *(history or []),
                {"role": "user", "content": text},
            ]
            if cache_breakpoints and history:
                # Anthropic prompt caching: one moving breakpoint on the last
                # history message marks system+history as the reusable prefix
                # (cache reads at 0.1x after a 1.25x write; the API ignores
                # breakpoints under its ~1024-token minimum, so this is inert
                # early in a session and on stateless calls).
                last = dict(messages[-2])
                last["content"] = [
                    {"type": "text", "text": last["content"], "cache_control": {"type": "ephemeral"}}
                ]
                messages[-2] = last
            kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
            # Send sampling/reasoning knobs only when set, so providers keep their
            # defaults (and so the recorded request mirrors what was actually sent).
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            body = dict(extra_body) if extra_body else {}
            if pin_state is not None and pin_state.get("provider"):
                # Session-sticky routing: after turn 1 stay on the provider that
                # served it, so its prefix cache stays hot and the serving stack
                # (quantization etc.) is constant within the session. If it goes
                # down the turn errors, the session ends, and the retry next run
                # re-pins fresh.
                body["provider"] = {"order": [pin_state["provider"]], "allow_fallbacks": False}
            if body:
                kwargs["extra_body"] = body
            t0 = time.monotonic()
            resp = await client.chat.completions.create(**kwargs)
            gen_seconds = time.monotonic() - t0
            # Some providers return choices=None/[] on an upstream error rather than
            # raising — surface a clear, catchable message, not an opaque TypeError.
            if not resp.choices:
                raise RuntimeError(f"{model}: provider returned no choices (upstream error?)")
            choice = resp.choices[0]
            # A truncated slop (finish_reason "length") is a broken pair — fail loudly,
            # and say where the tokens went: reasoning eating the whole budget wants
            # a lower --reasoning-effort, an actually-long answer wants --max-tokens.
            if choice.finish_reason == "length":
                trunc_usage = getattr(resp, "usage", None)
                trunc_details = getattr(trunc_usage, "completion_tokens_details", None)
                # The tail of the trace is the diagnosis: a deliberation loop
                # ("wait, let me reconsider...") vs a genuinely long rewrite.
                trace = _reasoning_text_of(choice.message)
                tail = f"; reasoning tail: ...{trace[-240:]}" if trace else ""
                raise RuntimeError(
                    f"slop truncated at max_tokens={max_tokens} "
                    f"(completion={getattr(trunc_usage, 'completion_tokens', '?')}, "
                    f"reasoning={getattr(trunc_details, 'reasoning_tokens', '?')} — "
                    f"raise --max-tokens or lower --reasoning-effort){tail}"
                )
            served_provider = getattr(resp, "provider", None)
            if pin_state is not None and served_provider:
                pin_state.setdefault("provider", served_provider)  # pin to turn 1's provider
            usage = getattr(resp, "usage", None)
            # OpenRouter/OpenAI split reasoning tokens out of completion_tokens here
            # (None when the provider doesn't report it). Latency + this split let a
            # slow run be diagnosed from the corpus alone: reasoning blowout shows as
            # reasoning_tokens ~ its budget; a slow upstream shows as low
            # completion_tokens / gen_seconds.
            details = getattr(usage, "completion_tokens_details", None)
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            gen_meta = {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "finish_reason": choice.finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": getattr(details, "reasoning_tokens", None),
                # Billing ground truth (OpenRouter, when usage.include is on):
                # cached_tokens = prompt prefix billed at the provider's cache-read
                # discount (0/None on providers with no cache pricing), cost = the
                # actual credits charged for THIS request. Session cost analysis
                # sums these instead of trusting token arithmetic.
                "cached_tokens": getattr(prompt_details, "cached_tokens", None),
                "cost": getattr(usage, "cost", None),
                "gen_seconds": round(gen_seconds, 2),
                # OpenRouter reports which upstream provider actually served the
                # request (None elsewhere) — the routing outcome, next to the
                # routing *request* in extra_meta (e.g. provider_sort).
                "provider": served_provider,
                "prompt_id": prompt_id,
                "prompt_version": prompt_version,
                "prompt_label": label,
            }
            if extra_meta:
                gen_meta.update(extra_meta)
            if capture_reasoning:
                # Routed by synthesize_pairs to <data-dir>/reasoning.jsonl —
                # never into pairs.jsonl (traces are diagnostics, not corpus).
                gen_meta["reasoning_text"] = _reasoning_text_of(choice.message)
            return GenOutput((choice.message.content or "").strip(), gen_meta)

        return generate

    return Generator(
        name=name or model,
        generate=_make_generate(None),
        strategy=label,
        reasoning_effort=reasoning_effort,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        prompt_system=system,
        # Multi-turn sessions get a per-session closure whose pin_state makes
        # routing sticky after turn 1 (only when requested — presets/local have
        # no provider routing to pin).
        begin_session=(lambda: _make_generate({})) if sticky_provider else None,
    )


def local_generator(
    *,
    model: str | None = None,
    strategy: str = DEFAULT_STRATEGY,
    base_url: str | None = None,
    api_key: str | None = None,
    system: str | None = None,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    timeout: float | None = DEFAULT_GEN_TIMEOUT,
    capture_reasoning: bool = False,
) -> Generator:
    """Local/utility base-model generator via an OpenAI-compatible endpoint.

    Reads `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` / `LOCAL_LLM_API_KEY` from the
    environment when not passed explicitly. Tagged `local-<model>` so its pairs
    are distinguishable in `meta.generator`. `reasoning_effort` is recorded but no
    reasoning wire-param is sent (local OpenAI-compatible servers vary).
    """
    from stylebot import config

    base_url = base_url or config.get_key("LOCAL_LLM_BASE_URL") or "http://localhost:8080/v1"
    model = model or config.get_key("LOCAL_LLM_MODEL") or "local"
    api_key = api_key or config.get_key("LOCAL_LLM_API_KEY") or "not-needed"
    return openai_generator(
        model=model,
        strategy=strategy,
        system=system,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url,
        name=f"local-{model}",
        timeout=timeout,
        capture_reasoning=capture_reasoning,
    )


def openrouter_generator(
    *,
    model: str,
    strategy: str = DEFAULT_STRATEGY,
    system: str | None = None,
    max_tokens: int = DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float | None = DEFAULT_GEN_TIMEOUT,
    provider_sort: str | None = "throughput",
    sticky_provider: bool = True,
    prompt_cache: bool = True,
    capture_reasoning: bool = False,
) -> Generator:
    """OpenRouter slop generator — one key, many upstream models.

    OpenRouter is OpenAI-compatible, so this is `openai_generator` pointed at the
    OpenRouter endpoint. `model` is an OpenRouter model id (e.g.
    ``anthropic/claude-opus-4.8``, ``qwen/qwen3-8b``), which makes multi-source slop
    rotation a single-credential affair. Tagged ``openrouter/<model>`` in
    `meta.generator` so its pairs stay distinguishable.

    Reads `OPENROUTER_API_KEY` (required) and optional `OPENROUTER_BASE_URL`
    (default ``https://openrouter.ai/api/v1``) from the environment / `.env`.

    `reasoning_effort` (high|medium|low|off) is a recorded covariate. Many models
    (Qwen3, Nemotron, …) reason by default, which on a paraphrase burns the token
    budget (≈14× completion tokens) and truncates the output; `_reasoning_extra_body`
    maps the requested effort to OpenRouter's `reasoning` field per model family.
    Default is HIGH (real AI prose is often produced at high reasoning); sweep down
    for experiments.
    """
    from stylebot import config

    base_url = base_url or config.get_key("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    # Provider routing: OpenRouter's default load-balancing favours price and
    # can land on ~10 tok/s upstreams; sort=throughput picks the fastest. The
    # requested sort is recorded (extra_meta) next to the served `provider`.
    extra_body = dict(_reasoning_extra_body(model, reasoning_effort) or {})
    if provider_sort:
        extra_body["provider"] = {"sort": provider_sort}
    # Ask OpenRouter to return billing ground truth in usage: per-request cost
    # (credits) and cached_tokens (cache-read-discounted prefix) — recorded in
    # meta.gen so session cost curves are measured, not inferred from tokens.
    extra_body["usage"] = {"include": True}
    return openai_generator(
        model=model,
        strategy=strategy,
        system=system,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key or config.require_key("OPENROUTER_API_KEY"),
        base_url=base_url,
        name=f"openrouter/{model}",
        extra_body=extra_body,
        extra_meta={"provider_sort": provider_sort} if provider_sort else None,
        timeout=timeout,
        # Session cost/covariate hygiene: pin each live session to the provider
        # that served its first turn (prefix cache stays hot; serving stack
        # constant within a session), and let Anthropic models cache the
        # session history via a moving cache_control breakpoint (0.1x reads;
        # other families cache automatically or not at all).
        sticky_provider=sticky_provider,
        cache_breakpoints=prompt_cache and model.startswith("anthropic/"),
        capture_reasoning=capture_reasoning,
    )
