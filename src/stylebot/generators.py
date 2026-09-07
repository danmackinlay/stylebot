"""Generators — the slop producers: strategy prompts + provider factories.

The slop-prompt registry (`STRATEGIES`, `resolve_strategy`, `prompt_id_of`),
the `Generator`/`GenOutput` contract, and the provider factories
(`openai_generator`, `openrouter_generator`). Import from here unless you
need the `synth.<name>` attribute lookup tests monkeypatch.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from stylebot.errors import ConfigError, RunError

# Shared tail for every strategy, so only the *flavour* of slop varies. Mirrors
# the style prompt's structure-preservation clause (`ai_core`): the synthetic
# source must differ in style, not markdown shape, or the styler learns reformatting.
_SLOP_PRESERVE = (
    "Preserve all markdown structure (code fences, math, links, "
    "list markers, blank lines) verbatim. "
    "Section headings may be rewritten in your own style, but keep each "
    "heading's level and meaning. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten passage, nothing else."
)

# Named slop strategies: generic AI-prose flavours (an author's own catalogue
# is a custom prompt). Describe the texture, never quote the tics: samplers
# converge on quoted tokens (D9).
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
SLOP_SYSTEM_ARGUED = (  # "argued": the concessive register of a model explaining itself
    "You are a careful technical writer reporting the outcome of an investigation "
    "to a collaborator. Rewrite the user's passage so that its conclusions read as "
    "reasoned instead of announced: for the main claim of each paragraph, say in "
    "the same sentence what it follows from, and where the passage itself rules "
    "something out, keep that rejection and give its reason. Do this at most once "
    "a paragraph and leave the other sentences plain. Carry the reasoning in the "
    "syntax — subordinate and relative clauses, a closing clause that names what "
    "follows — never in stock phrases, and never stage an imagined reader, "
    "objector or hypothetical doubt: address only what the passage says. Mix "
    "sentence lengths, one long qualified sentence beside short direct ones. Add "
    "no facts, examples or opinions the passage does not contain or clearly "
    "imply. Keep every link, citation and code span exactly as written. Keep the "
    "length within about one-and-a-quarter times the original. Do not open with "
    "a summary and do not close with one, and vary your sentence openings and "
    "structure from passage to passage. " + _SLOP_PRESERVE
)
# "verbose" gives the styler inflated slop to learn compression from;
# "linkedin" bans emoji/hashtags because a lexical giveaway would let the
# classifier stop reading the prose; "argued" asks for the shape real captures
# carry and synthesis lacked (`_plans/argued-slop-arm.md`), in texture only.


@dataclass(frozen=True)
class SlopStrategy:
    """A named slop-prompt flavour: label, system prompt, hand-bumped version;
    the `prompt_id` content hash is what identifies it."""

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
    "argued": SlopStrategy("argued", SLOP_SYSTEM_ARGUED, version=2),
}
DEFAULT_STRATEGY = "polish"

# Reasoning effort is a recorded covariate; `off` by default (D10). What is
# recorded is what was asked, since providers may ignore it.
DEFAULT_REASONING_EFFORT = "off"


def prompt_id_of(system_text: str) -> str:
    """Stable content id for any slop system prompt, registry or custom file."""
    return hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:12]


class ResolvedStrategy(NamedTuple):
    """A slop strategy resolved to its prompt-content identity; ``prompt_id``
    and ``version`` feed ``synth_key`` / ``meta.gen``."""

    label: str
    system: str
    version: int
    prompt_id: str


def resolve_strategy(name: str, system: str | None = None) -> ResolvedStrategy:
    """Resolve a strategy name to a `ResolvedStrategy`; an explicit ``system``
    overrides the registry (version 0), and an unknown name without one raises."""
    if system is not None:
        return ResolvedStrategy(name, system, 0, prompt_id_of(system))
    try:
        strat = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ConfigError(
            f"unknown slop strategy {name!r}; known: {known} "
            f"(or pass an explicit system prompt / --slop-system-file)"
        ) from None
    return ResolvedStrategy(strat.label, strat.system, strat.version, prompt_id_of(strat.system))


# Slop expands the target, so the cap leaves room for 3-4x its tokens.
DEFAULT_SLOP_MAX_TOKENS = 8192
# Per-request HTTP timeout. Without one the openai SDK waits 600s per attempt
# (times its retries), so a bad upstream stalls a run for half an hour in silence.
DEFAULT_GEN_TIMEOUT = 300.0


@dataclass(frozen=True)
class GenOutput:
    """A generator's output: the slop text plus per-call covariates.
    `generate` may return a bare ``str`` instead; `_normalize_gen_output` coerces."""

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

    `name`, `strategy`, `reasoning_effort` and `prompt_id` feed the
    `synth_key`. `generate` may be sync or async, returning `str` or
    `GenOutput`; a session-capable callable accepts `(text, history=None)`.
    `session_budget` overrides the per-session token budget; `begin_session`
    returns a fresh callable holding per-session state.
    """

    name: str
    generate: Callable[..., str | GenOutput] | None = None
    strategy: str = DEFAULT_STRATEGY
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    prompt_id: str = ""
    prompt_version: int = 0
    # Full system-prompt text; `synthesize_pairs` archives it to prompts.jsonl.
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
    """Map a requested reasoning effort to OpenRouter's `reasoning` request
    field: `off` disables, budget-style families get a token budget, the rest
    the effort enum."""
    if effort == "off":
        return {"reasoning": {"enabled": False}}
    if model.startswith(_REASONING_BUDGET_FAMILIES):
        return {"reasoning": {"max_tokens": _REASONING_MAX_TOKENS[effort]}}
    return {"reasoning": {"effort": effort}}


_CONTEXT_WINDOWS_CACHE: dict[str, dict[str, int]] = {}


def openrouter_context_windows(base_url: str | None = None) -> dict[str, int]:
    """Fetch ``{model_id: context_length}`` from the OpenRouter models registry:
    keyless, one GET per base_url per process; raises on network failure."""
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
    """The reasoning trace of a response message (`reasoning`, else
    `reasoning_details` blocks), None when the provider sent none."""
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
    model: str,
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

    `base_url` repoints at any compatible endpoint; `extra_body` passes
    provider knobs through. `model` has no default: a forgotten argument must
    not silently bill a model nobody chose.
    """
    import openai

    from stylebot import config

    label, system, prompt_version, prompt_id = resolve_strategy(strategy, system)
    # Async client: `synthesize_pairs` runs sessions on one event loop (D56).
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
                # history message; inert under the API's token minimum.
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
                # Session-sticky routing: stay on the provider that served turn 1.
                body["provider"] = {"order": [pin_state["provider"]], "allow_fallbacks": False}
            if body:
                kwargs["extra_body"] = body
            t0 = time.monotonic()
            resp = await client.chat.completions.create(**kwargs)
            gen_seconds = time.monotonic() - t0
            # Some providers return choices=None/[] on an upstream error rather than
            # raising — surface a clear, catchable message, not an opaque TypeError.
            if not resp.choices:
                raise RunError(f"{model}: provider returned no choices (upstream error?)")
            choice = resp.choices[0]
            # A truncated slop is a broken pair: fail loudly and say where the
            # tokens went, since reasoning and a long answer want different fixes.
            if choice.finish_reason == "length":
                trunc_usage = getattr(resp, "usage", None)
                trunc_details = getattr(trunc_usage, "completion_tokens_details", None)
                # The tail of the trace diagnoses which: a deliberation loop
                # ("wait, let me reconsider...") vs a genuinely long rewrite.
                trace = _reasoning_text_of(choice.message)
                tail = f"; reasoning tail: ...{trace[-240:]}" if trace else ""
                raise RunError(
                    f"slop truncated at max_tokens={max_tokens} "
                    f"(completion={getattr(trunc_usage, 'completion_tokens', '?')}, "
                    f"reasoning={getattr(trunc_details, 'reasoning_tokens', '?')} — "
                    f"raise --max-tokens or lower --reasoning-effort){tail}"
                )
            served_provider = getattr(resp, "provider", None)
            if pin_state is not None and served_provider:
                pin_state.setdefault("provider", served_provider)  # pin to turn 1's provider
            usage = getattr(resp, "usage", None)
            # Reasoning tokens split out of completion_tokens (None when
            # unreported), so a slow run is diagnosable from the corpus alone.
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
                # Billing ground truth: credits charged, and the cached prefix.
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
        # Per-session closure with sticky routing; off for a plain OpenAI endpoint.
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
    """OpenRouter slop generator — `openai_generator` at the OpenRouter endpoint,
    tagged ``openrouter/<model>``; reads `OPENROUTER_API_KEY` and optional
    `OPENROUTER_BASE_URL`. `reasoning_effort` maps per model family (D10)."""
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
        # Pin each session to its first provider; Anthropic models cache via a
        # moving breakpoint.
        sticky_provider=sticky_provider,
        cache_breakpoints=prompt_cache and model.startswith("anthropic/"),
        capture_reasoning=capture_reasoning,
    )
