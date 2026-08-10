"""Generators — the slop producers: strategy prompts + provider factories.

The slop-prompt registry (`STRATEGIES`, `resolve_strategy`, `prompt_id_of`), the
`Generator`/`GenOutput` contract, and the provider factories (`openai_generator`
and `openrouter_generator`, built on it) with their wire-level helpers.
`stylebot.synth` re-exports the subset the CLI kit reaches through it — not
everything here. Import from this module directly unless you specifically need
`synth.<name>` attribute lookup (which is how `bin/synth_cli.py` resolves the
generator factories, and therefore how tests monkeypatch them).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

# Shared tail for every strategy, so only the *flavour* of slop varies. Mirrors
# STYLE_SYSTEM's structure-preservation clause: the synthetic source must differ
# from the target in style, not markdown shape, or the styler learns reformatting.
_SLOP_PRESERVE = (
    "Preserve all markdown structure (code fences, math, links, "
    "list markers, blank lines) verbatim. "
    "Section headings may be rewritten in your own style, but keep each "
    "heading's level and meaning. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten passage, nothing else."
)

# Named slop strategies: label -> the system prompt producing that flavour. These
# are GENERIC AI-prose flavours; an author's own catalogue is injected as a custom
# prompt (`--slop-system-file` / `system=`), keeping stylebot free of any one
# author's slop definition. When writing a new one, DESCRIBE THE TEXTURE, never
# quote the tics: samplers converge hard on quoted tokens, so a prompt listing
# stock phrases makes every output open identically and the slop go cartoonish.
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
    "nice thing is'), and round each idea off so nothing lands abruptly. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_MEASURED = (  # "measured": the mild stereotypical-LLM register — texture, not tokens
    "You are a careful AI writing assistant. Rewrite the user's passage in a "
    "measured, well-organized explanatory register: smooth transitions, gentle "
    "hedging of strong claims, tidy structure, slightly more formal vocabulary, "
    "and an even sentence rhythm. Let the register show in the overall texture, "
    "not in any stock phrase — do not open with generic scene-setting, and vary "
    "your sentence openings and structure naturally from passage to passage. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_LINKEDIN = (  # "linkedin": the personal-brand / thought-leadership register
    "You are a personal-brand content strategist. Rewrite the user's passage so it "
    "performs on LinkedIn: lead with the payoff instead of the setup, recast "
    "observations as hard-won professional lessons, replace the original's hedging "
    "with flat confidence, tell the reader what to take away, and land each point "
    "as its own short declarative sentence. Aim for a brisk motivational cadence "
    "and an air of earned authority. Let the register show in the texture, not in "
    "any stock phrase — no emoji, no hashtags, no stock openers or sign-offs, and "
    "vary your sentence openings naturally from passage to passage. " + _SLOP_PRESERVE
)
SLOP_SYSTEM_VERBOSE = (  # "verbose": moderate inflation — the compression-training flavour
    "You are a thorough writing assistant. Rewrite the user's passage at "
    "greater length — aim for one-and-a-half to two times the original — by "
    "explaining each point more fully: unpack compressed claims, spell out "
    "implications the original leaves implicit, add gentle hedges and "
    "qualifications, and restate the key idea once in different words. Add "
    "no new facts, examples, or opinions — only elaboration of what is "
    "already there. Never exceed roughly double the original length. Let the "
    "expansion live in the texture of full explanation, not in filler "
    "phrases — no generic openers, no summarising sign-off. " + _SLOP_PRESERVE
)
# "verbose" exists because real human rewrites *compress*: the styler needs
# deliberately inflated slop to learn compression from, bounded by max_length_ratio.
#
# "linkedin" bans emoji/hashtags despite them being the format's loudest tell:
# they are lexical giveaways, so the classifier would learn "emoji -> slop" and
# stop reading the prose — a shortcut that inflates the detector's score while
# teaching the styler nothing. The target is the rhetorical move (payoff-first,
# hedges stripped, lesson-ised, reader instructed).


@dataclass(frozen=True)
class SlopStrategy:
    """A named slop-prompt flavour: a human label, the system prompt, a version.

    `version` is bumped by hand; the `prompt_id` content hash is what actually
    identifies a prompt, so editing the text changes its id either way.
    """

    label: str
    system: str
    version: int = 1


STRATEGIES: dict[str, SlopStrategy] = {
    "polish": SlopStrategy("polish", SLOP_SYSTEM, version=1),
    "engaging": SlopStrategy("engaging", SLOP_SYSTEM_ENGAGING, version=1),
    "casual": SlopStrategy("casual", SLOP_SYSTEM_CASUAL, version=1),
    "measured": SlopStrategy("measured", SLOP_SYSTEM_MEASURED, version=1),
    "linkedin": SlopStrategy("linkedin", SLOP_SYSTEM_LINKEDIN, version=1),
    "verbose": SlopStrategy("verbose", SLOP_SYSTEM_VERBOSE, version=1),
}
DEFAULT_STRATEGY = "polish"

# Reasoning is a recorded *covariate*, not a silent default. `off` because a
# measured sweep found the detector gap indistinguishable between `off` and
# `medium` for ~3x the wall clock: raising it costs money and buys nothing unless
# a model proves effort-sensitive. Providers may ignore the request entirely
# (some models reason at `off` anyway), so what is recorded is what was *asked*.
DEFAULT_REASONING_EFFORT = "off"


def prompt_id_of(system_text: str) -> str:
    """Stable content id for ANY slop system prompt (registry or custom file).

    Hashing the text means a custom `--slop-system-file` is faceted and deduped
    like a registry strategy, and editing a prompt changes its id.
    """
    return hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:12]


class ResolvedStrategy(NamedTuple):
    """A slop strategy resolved to its prompt-content identity.

    A NamedTuple, so callers may unpack or read attributes. ``prompt_id``
    (content hash) and ``version`` feed ``synth_key`` / ``meta.gen``.
    """

    label: str
    system: str
    version: int
    prompt_id: str


def resolve_strategy(name: str, system: str | None = None) -> ResolvedStrategy:
    """Resolve a strategy name to a `ResolvedStrategy`.

    An explicit ``system`` overrides the registry, letting a caller inject a
    custom slop prompt under any label (version 0, identified by content hash).
    An unknown name is an error *unless* ``system`` is supplied.
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


# Slop is an *expansion* of the target (AI prose runs 1.5-3x longer), so the cap
# must comfortably exceed the target's own token count: targets cap at
# MAX_CHUNK_CHARS (~2k tokens), leaving room for 3-4x.
DEFAULT_SLOP_MAX_TOKENS = 8192
# Per-request HTTP timeout. Without one the openai SDK waits 600s per attempt
# (times its retries), so a bad upstream stalls a run for half an hour in silence.
DEFAULT_GEN_TIMEOUT = 300.0


@dataclass(frozen=True)
class GenOutput:
    """A generator's output: the slop text plus per-call generation covariates.

    `generate` may return a bare ``str`` (test fakes, simple callables) or a
    ``GenOutput`` whose ``meta`` carries the recorded covariates;
    `_normalize_gen_output` coerces either.
    """

    text: str
    meta: dict = field(default_factory=dict)


def _normalize_gen_output(out: str | GenOutput) -> tuple[str, dict]:
    """Coerce a generator return (``str`` or ``GenOutput``) to ``(text, gen_meta)``."""
    if isinstance(out, GenOutput):
        return out.text, dict(out.meta)
    return out, {}


@dataclass
class Generator:
    """A named slop producer.

    `name` becomes `meta.generator`, `strategy` becomes `meta.slop_strategy`;
    both, plus `reasoning_effort` and `prompt_id`, feed the `synth_key`.

    `generate` may be sync or async and may return a bare `str` or a `GenOutput`.
    A plain 1-arg callable is fine for stateless (`session_turns=1`) runs;
    multi-turn sessions pass a `history` kwarg, so a session-capable callable
    must accept `(text, history=None)`.

    `session_budget` optionally overrides the global per-session prompt-token
    budget (window × `SESSION_WINDOW_FILL_CAP` still caps it). `begin_session`,
    when set, returns a fresh generate-callable holding per-session state —
    openrouter uses it for sticky provider routing.
    """

    name: str
    generate: Callable[..., str | GenOutput] | None = None
    strategy: str = DEFAULT_STRATEGY
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    prompt_id: str = ""
    prompt_version: int = 0
    # Full system-prompt text; synthesize_pairs archives it to prompts.jsonl so
    # a prompt_id is always resolvable to the prompt that produced the pairs.
    prompt_system: str = ""
    session_budget: int | None = None
    begin_session: Callable[[], Callable[..., str | GenOutput]] | None = None

    def __call__(self, target_text: str, history: list[dict] | None = None):
        if self.generate is None:
            raise RuntimeError(f"generator {self.name!r} has no callable (dry-run/name-only stub)")
        if history:
            return self.generate(target_text, history=history)
        return self.generate(target_text)


# Approximate reasoning budgets, and the model-id prefixes of the upstreams that
# take a budget rather than an effort enum (OpenRouter normalizes the rest).
_REASONING_MAX_TOKENS = {"high": 8000, "medium": 4000, "low": 1500}
_REASONING_BUDGET_FAMILIES = ("google/", "qwen/", "nvidia/", "deepseek/")


def _reasoning_extra_body(model: str, effort: str) -> dict | None:
    """Map a requested reasoning effort to OpenRouter's `reasoning` request field.

    `off` disables reasoning; budget-style families get a token budget; everyone
    else gets the effort enum. The *requested* effort is recorded regardless of
    what the provider honors — `finish_reason` and `completion_tokens` are how
    you detect a model that reasoned anyway.
    """
    if effort == "off":
        return {"reasoning": {"enabled": False}}
    if model.startswith(_REASONING_BUDGET_FAMILIES):
        return {"reasoning": {"max_tokens": _REASONING_MAX_TOKENS[effort]}}
    return {"reasoning": {"effort": effort}}


_CONTEXT_WINDOWS_CACHE: dict[str, dict[str, int]] = {}


def openrouter_context_windows(base_url: str | None = None) -> dict[str, int]:
    """Fetch ``{model_id: context_length}`` from the OpenRouter models registry.

    Ground truth for per-model window sizes; hand-annotating them would go stale.
    Keyless, one GET per base_url per process. Raises URLError/HTTPError on
    network failure — the caller decides whether windows are required.
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

    Most models normalize to `message.reasoning` (a plain string); some providers
    return `message.reasoning_details` (typed blocks). None when neither exists.
    """
    text = getattr(message, "reasoning", None)
    if text:
        return text
    details = getattr(message, "reasoning_details", None)
    if details:
        parts = [
            d.get("text") if isinstance(d, dict) else getattr(d, "text", None) for d in details
        ]
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

    `base_url` repoints at any OpenAI-compatible endpoint — the seam
    `openrouter_generator` uses; `extra_body` passes provider-specific knobs
    through. `generate` returns a `GenOutput` of covariates plus token usage.
    """
    import openai

    from stylebot import config

    label, system, prompt_version, prompt_id = resolve_strategy(strategy, system)
    # Async client: synthesize_pairs runs sessions concurrently on an event loop
    # (no threads, no locks) and keeps a sync surface via its own asyncio.run.
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
                # history message marks system+history as the reusable prefix.
                # The API ignores breakpoints under its ~1024-token minimum, so
                # this is inert early in a session.
                last = dict(messages[-2])
                last["content"] = [
                    {
                        "type": "text",
                        "text": last["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
                messages[-2] = last
            kwargs: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
            # Send sampling knobs only when set, so providers keep their defaults
            # and the recorded request mirrors what was actually sent.
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            body = dict(extra_body) if extra_body else {}
            if pin_state is not None and pin_state.get("provider"):
                # Session-sticky routing: after turn 1 stay on the provider that
                # served it, so its prefix cache stays hot and the serving stack
                # (quantization etc.) is constant within the session.
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
                # The tail of the trace diagnoses which: a deliberation loop
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
            # Reasoning tokens are split out of completion_tokens here (None when
            # unreported). With gen_seconds this diagnoses a slow run from the
            # corpus alone: blowout shows as reasoning_tokens ~ budget, a slow
            # upstream as low completion_tokens / gen_seconds.
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
                # credits actually charged, and the prefix billed at the cache-read
                # discount. Sum these rather than trusting token arithmetic.
                "cached_tokens": getattr(prompt_details, "cached_tokens", None),
                "cost": getattr(usage, "cost", None),
                "gen_seconds": round(gen_seconds, 2),
                # Which upstream served the request — the routing outcome, next to
                # the routing *request* in extra_meta (e.g. provider_sort).
                "provider": served_provider,
                "prompt_id": prompt_id,
                "prompt_version": prompt_version,
                "prompt_label": label,
            }
            if extra_meta:
                gen_meta.update(extra_meta)
            if capture_reasoning:
                # synthesize_pairs routes this to reasoning.jsonl, never into
                # pairs.jsonl — traces are diagnostics, not corpus.
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
        # routing sticky after turn 1. Off by default: a plain OpenAI endpoint
        # has no provider routing to pin.
        begin_session=(lambda: _make_generate({})) if sticky_provider else None,
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
    OpenRouter endpoint; pairs are tagged ``openrouter/<model>`` in
    `meta.generator`. Reads `OPENROUTER_API_KEY` (required) and optional
    `OPENROUTER_BASE_URL` from the environment.

    Many models (Qwen3, Nemotron, …) reason by default, which on a paraphrase
    burns the token budget (≈14× completion tokens) and truncates the output;
    `_reasoning_extra_body` maps `reasoning_effort` per model family. See
    `DEFAULT_REASONING_EFFORT`.
    """
    from stylebot import config

    base_url = base_url or config.get_key("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    # Provider routing: OpenRouter's default load-balancing favours price and can
    # land on ~10 tok/s upstreams; sort=throughput picks the fastest.
    extra_body = dict(_reasoning_extra_body(model, reasoning_effort) or {})
    if provider_sort:
        extra_body["provider"] = {"sort": provider_sort}
    extra_body["usage"] = {"include": True}  # billing ground truth in usage
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
        # Session cost/covariate hygiene: pin each session to the provider that
        # served its first turn, and let Anthropic models cache session history
        # via a moving breakpoint (other families cache automatically or not).
        sticky_provider=sticky_provider,
        cache_breakpoints=prompt_cache and model.startswith("anthropic/"),
        capture_reasoning=capture_reasoning,
    )
