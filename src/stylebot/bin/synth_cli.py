"""Shared CLI kit for synth-style commands.

`ai-style synth` and downstream blog wrappers (e.g. livingthing's
`train-targets`) present the same knobs over the same engine; only *selection*
(which posts, which selector) and a few pinned defaults differ per caller.
This module is the single source for the shared surface so specializations
override instead of copy-paste (the fork drift this replaces: divergent exit
codes, missing progress heartbeat, duplicated dry-run preset names):

- `synth_options(exclude=…, **default_overrides)` — decorator applying the
  shared chunking/inspection/generation option set from one spec table. A
  wrapper overrides defaults by param name and excludes options it pins as
  policy constants.
- `run_synth(targets, …)` — the whole command body after target selection:
  count echo, `--sample`/`--report` short-circuit, generator construction
  (incl. dry-run stubs), `synthesize_pairs`, result reporting, and the
  non-zero exit on generation errors.

Per-command concerns stay with the caller: target selection options, the
`--data-dir` resolution policy (pass a callable to defer it past the
inspection modes), and provenance tags.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from stylebot import synth
from stylebot.splits import ROLES as SPLIT_ROLES

# Progress heartbeat cadence for generation (seconds of wall-clock silence).
_HEARTBEAT_SECS = 20.0

# Direct-API slop-generator presets selectable on the CLI (each needs that
# provider's own key). OpenRouter models are selected separately via
# --openrouter-model (one key, many upstream models). Library callers inject
# their own `synth.Generator`s.
PRESETS = ("gpt", "local")


def sort_key(name: str):
    """Map the --sort choice to an iter_targets sort_key callable."""
    if name == "none":
        return None
    if name == "length":
        return lambda t: len(t.text)
    if name == "source":
        return lambda t: (t.source, t.chunk_index)
    raise click.BadParameter(f"unknown sort key {name!r}")


# Shared params consumed by iter_targets (selection-side); the rest of the
# shared set feeds run_synth. Lets a wrapper collect the whole shared surface
# as **kw and split it without restating ~30 parameter names.
CHUNK_PARAM_NAMES = (
    "min_chars", "max_chars", "prose_only", "ignore_markers", "stop_at_headers",
    "drop_link_dumps", "drop_list_items", "merge", "merge_max_chars",
    "heading_context", "sort_name",
)


def pop_chunk_kwargs(kw: dict) -> dict:
    """Pop the chunking/hygiene params out of a **kw catch-all, shaped for
    iter_targets (--max-chars 0 -> no cap; --sort name -> sort_key callable)."""
    out = {k: kw.pop(k) for k in CHUNK_PARAM_NAMES if k in kw}
    if "max_chars" in out:
        out["max_chars"] = out["max_chars"] or None
    if "sort_name" in out:
        out["sort_key"] = sort_key(out.pop("sort_name"))
    return out


# The shared option surface, one spec per param name (the key MUST match the
# callback parameter). Order here is help order. Wrappers override `default`
# by name via synth_options(**overrides) or drop an option via exclude=….
_OPTION_SPECS: dict[str, tuple[tuple[str, ...], dict]] = {
    # -- chunking / hygiene (feed iter_targets) --
    "min_chars": (("--min-chars",), dict(default=synth.MIN_CHUNK_CHARS, show_default=True, type=int, help="Skip prose chunks shorter than this.")),
    "max_chars": (("--max-chars",), dict(default=synth.MAX_CHUNK_CHARS, show_default=True, type=int, help="Drop prose chunks longer than this (0 disables the cap).")),
    "prose_only": (("--prose-only/--no-prose-only",), dict(default=True, show_default=True, help="Drop code/math/:::divs/blockquotes before chunking.")),
    "ignore_markers": (("--ignore-marker", "ignore_markers"), dict(multiple=True, help="Drop any chunk containing this literal (repeatable), e.g. a stub marker.")),
    "stop_at_headers": (("--stop-at-header", "stop_at_headers"), dict(multiple=True, help="Truncate each post body at this section header (repeatable), e.g. '## Incoming'.")),
    "drop_link_dumps": (("--drop-link-dumps/--keep-link-dumps",), dict(default=True, show_default=True, help="Drop chunks that are mostly markdown links (little authored prose).")),
    "drop_list_items": (("--drop-list-items/--keep-list-items",), dict(default=False, show_default=True, help="Drop chunks that are entirely markdown list items.")),
    "merge": (("--merge/--no-merge",), dict(default=False, show_default=True, help="Pack consecutive prose paragraphs within a section into multi-paragraph passages.")),
    "merge_max_chars": (("--merge-max-chars",), dict(default=synth.MERGE_MAX_CHUNK_CHARS, show_default=True, type=int, help="Soft char budget per packed passage (merge mode).")),
    "heading_context": (("--heading-context",), dict(type=click.Choice(["none", "immediate"]), default="none", show_default=True, help="Prepend the section heading (verbatim) to both sides of each pair as fixed context.")),
    "context_dropout": (("--context-dropout",), dict(default=0.0, show_default=True, type=float, help="Fraction of pairs to keep heading-less (deterministic) so the styler doesn't require a heading.")),
    "sort_name": (("--sort", "sort_name"), dict(type=click.Choice(["none", "length", "source"]), default="none", show_default=True)),
    # -- read-only inspection (handled inside run_synth, exit before generation) --
    "report_path": (("--report", "report_path"), dict(type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a self-contained HTML report of the selected targets and exit (pre-generation — vets selection, generates nothing; browse generated pairs via ai-style eval --report).")),
    "report_max_rows": (("--report-max-rows",), dict(default=2000, show_default=True, type=int, help="Cap table rows in the HTML report (0 = all).")),
    "sample_n": (("--sample", "sample_n"), dict(type=int, default=None, help="Print N random targets to stdout and exit (no generation).")),
    # -- generation --
    "generator_names": (("--generator", "generator_names"), dict(type=click.Choice(PRESETS), multiple=True, default=(), show_default=True, help="Direct-API slop generators to rotate across (repeatable; each needs that provider's key). Combine with or replace by --openrouter-model.")),
    "openrouter_models": (("--openrouter-model", "openrouter_models"), dict(multiple=True, help="OpenRouter model id to generate slop with (repeatable), e.g. anthropic/claude-opus-4-8. One key ($OPENROUTER_API_KEY) reaches many upstream models — ideal for multi-source rotation.")),
    "provider_sort": (("--provider-sort",), dict(type=click.Choice(["throughput", "price", "latency", "none"]), default="throughput", show_default=True, help="OpenRouter provider-routing preference (default load-balancing favours price and can land on ~10 tok/s upstreams; 'none' restores it). OpenRouter models only; recorded in meta.gen next to the served provider.")),
    "sticky_provider": (("--sticky-provider/--no-sticky-provider",), dict(default=True, show_default=True, help="Pin each live session to the provider that served its first turn — keeps its prefix cache hot AND the serving stack constant, so window-position covariates aren't confounded by provider hops. OpenRouter models only; no effect on stateless runs.")),
    "prompt_cache": (("--prompt-cache/--no-prompt-cache",), dict(default=True, show_default=True, help="Anthropic models: mark session history with a moving cache_control breakpoint (cache reads at 0.1x after a 1.25x write; inert under ~1024-token prefixes and for other model families). Verify via meta.gen cached_tokens/cost.")),
    "slop_strategies": (("--slop-strategy", "slop_strategies"), dict(multiple=True, default=(synth.DEFAULT_STRATEGY,), show_default=True, help=f"Slop-prompt flavour ({', '.join(synth.STRATEGIES)}); repeatable — the rotation becomes models x strategies, so one run diversifies prompts like it already diversifies models. Recorded as meta.slop_strategy and folded into synth_key. With --slop-system-file, pass exactly one label for the custom prompt.")),
    "slop_system_file": (("--slop-system-file",), dict(type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Override the slop system prompt with this file's contents (e.g. an author's own slop catalogue), labelled by --slop-strategy. Keeps blog-specific prompts out of stylebot.")),
    "gpt_model": (("--gpt-model",), dict(default="gpt-4o", show_default=True)),
    "local_model": (("--local-model",), dict(default="", help="Local model id (else $LOCAL_LLM_MODEL).")),
    "max_tokens": (("--max-tokens",), dict(default=synth.DEFAULT_SLOP_MAX_TOKENS, show_default=True, type=int, help="Max completion tokens per slop generation. Raise if a model truncates.")),
    "timeout": (("--timeout",), dict(default=synth.DEFAULT_GEN_TIMEOUT, show_default=True, type=float, help="Per-request HTTP timeout (seconds) for slop generation. A timed-out pair is reported immediately and recorded as an error; it ends its own session (remaining turns defer to the next run) while other sessions continue.")),
    "reasoning_efforts": (("--reasoning-effort", "reasoning_efforts"), dict(type=click.Choice(["high", "medium", "low", "off"]), multiple=True, default=(synth.DEFAULT_REASONING_EFFORT,), show_default=True, help="Reasoning effort for the slop generator ('off' disables reasoning); repeatable — the rotation becomes models x strategies x efforts, so effort joins the sweep (facet eval by reasoning_effort). Recorded in meta.gen and folded into synth_key; the requested effort is recorded regardless of what the provider honors.")),
    "capture_reasoning": (("--capture-reasoning",), dict(is_flag=True, help="Save each pair's reasoning/thinking trace to <data-dir>/reasoning.jsonl (keyed by synth_key; never enters pairs.jsonl). Diagnose reasoning blowouts by READING what the model deliberated about; costs nothing extra (the tokens were already paid for).")),
    "temperature": (("--temperature",), dict(type=float, default=None, help="Sampling temperature, sent to the API and recorded (provider default if unset).")),
    "top_p": (("--top-p", "top_p"), dict(type=float, default=None, help="Nucleus top_p, sent to the API and recorded (provider default if unset).")),
    "per_generator": (("--per-generator",), dict(is_flag=True, help="Emit a pair from EVERY generator per target (n× cost) — the fully-crossed within-target design for experiments; default assigns each target ONE arm by content hash.")),
    "assign_seed": (("--assign-seed",), dict(default="", help="Salt for the content-hash arm assignment. Change it to re-randomize which (model, strategy, effort) arm each target gets — a fresh replicate of the design (changed assignments regenerate on resume).")),
    "max_workers": (("--max-workers",), dict(default=0, show_default="auto", type=int, help="Concurrent sessions (asyncio). 0 = auto: 16 when all generators are OpenRouter-backed, 1 when a gpt/local preset is in the mix (a local server wants sequential).")),
    "session_turns": (("--session-turns",), dict(default=1, show_default=True, type=int, help="Session mode switch + backstop: 1 = stateless; >1 groups turns into live sessions (each turn sees the real prior passage->slop exchanges, meta.gen session_turn / window_fill). Depth is controlled by --session-max-tokens, NOT this number — sessions end when the token budget binds and leftover turns reflow into fresh sessions; the value only caps turns per session as a runaway guard.")),
    "session_max_tokens": (("--session-max-tokens",), dict(default=synth.DEFAULT_SESSION_MAX_TOKENS, show_default=True, type=int, help="THE session depth control: per-session prompt-token budget (absolute; input cost grows ~quadratically with it). Also capped at 80% of each model's context window. Turns beyond the budget reflow into fresh sessions, never dropped.")),
    "replicate": (("--replicate",), dict(default="", show_default=True, help="Deliberate-resample label folded into synth_key: the same substrate under a new label mints new cells without colliding with the base corpus (e.g. 'deep128k' for a deep-window arm, 'draw2' for a second sample). Recorded as meta.gen.replicate. Empty = the base corpus.")),
    "skip_covered": (("--skip-covered/--no-skip-covered",), dict(default=False, show_default=True, help="Corpus-building coverage mode: skip any target whose text already has >=1 pair in the data-dir under ANY config or epoch (cell-level dedup still applies to the rest). Use when the goal is one pair per passage — without it, a re-key epoch (e.g. an effort-default change) regenerates already-covered targets and doubles their training weight.")),
    "max_transform_sim": (("--max-transform-sim",), dict(type=float, default=None, help="Degenerate-output gate: drop (don't write) any generated pair whose transform_sim exceeds this — the model returned (nearly) the input, which would teach the styler to copy. Dropped pairs are counted loudly (skipped_degenerate) and their cells retry next run. Unset = record everything (the covariate is always stored).")),
    "context_window": (("--context-window",), dict(type=int, default=None, help="Context window (tokens) for gpt/local preset generators when sweeping sessions — OpenRouter models resolve theirs from the models registry automatically.")),
    # -- substrate selection: WHICH population this run measures (before --limit) --
    "splits_path": (("--splits", "splits_path"), dict(type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="splits.json of by-post roles (see ai-style make-splits). Required by --role.")),
    "role": (("--role",), dict(type=click.Choice(SPLIT_ROLES), default=None, help="Keep only chunks whose post holds this role. Measurement runs want 'styler': disjoint from the detector's training pool, so the voice classifier stays an unbiased instrument.")),
    "sample_frac": (("--sample-frac",), dict(type=float, default=None, help="Keep this fraction of chunks by content hash. Growth-stable: a later run on a larger corpus is a SUPERSET of this one, so the two stay comparable (--limit is not, its first N shifts as posts are added).")),
    "sample_salt": (("--sample-salt",), dict(default="", show_default=True, help="Re-draw an independent subsample of the same size (a fresh replicate of the design).")),
    "limit": (("--limit",), dict(type=int, default=None, help="Cap the number of target chunks (cost control / smoke runs). A cap on the substrate, not a definition of it — see --sample-frac.")),
    "dry_run": (("--dry-run",), dict(is_flag=True, help="Report planned targets/pairs without calling any API or writing.")),
}


@dataclass(frozen=True)
class Arm:
    """One rotation arm: a (model, slop strategy, reasoning effort) combination.

    Reifies the unit the round-robin assignment spreads targets across
    (previously "arm" existed only in comments). ``name`` is the single
    computation of the generator name that feeds ``synth_key`` — dry-run stubs
    and real factory generators both take it from here, so the two paths
    cannot drift apart.
    """

    kind: str  # "gpt" | "local" | "openrouter"
    model: str  # resolved model id (env fallbacks already applied)
    strategy: synth.ResolvedStrategy
    effort: str

    @property
    def name(self) -> str:
        if self.kind == "openrouter":
            return f"openrouter/{self.model}"
        if self.kind == "local":
            return f"local-{self.model}"
        return self.model  # gpt preset: the raw model id


def iter_arms(
    resolved_strategies: Sequence[synth.ResolvedStrategy],
    efforts: Sequence[str],
    generator_names: Sequence[str],
    openrouter_models: Sequence[str],
    *,
    gpt_model: str,
    local_model: str,
) -> list[Arm]:
    """The rotation cross product, in corpus-affecting enumeration order.

    Strategies (outer) × efforts × [presets in flag order, then OpenRouter
    models in flag order]. `synth._assign` picks ``generators[digest % n]``,
    so this enumeration order decides which arm owns which target — reordering
    it reassigns the corpus on resume. The local model id is resolved here
    (flag, else ``$LOCAL_LLM_MODEL``, else ``"local"`` — mirroring
    `synth.local_generator`) so dry-run stub names match a real run's even
    when the env var supplies the model.
    """
    from stylebot import config

    models: list[tuple[str, str]] = []
    for g in generator_names:
        if g == "gpt":
            models.append(("gpt", gpt_model))
        else:
            models.append(("local", local_model or config.get_key("LOCAL_LLM_MODEL") or "local"))
    models += [("openrouter", m) for m in openrouter_models]
    return [
        Arm(kind=kind, model=model, strategy=rs, effort=effort)
        for rs in resolved_strategies
        for effort in efforts
        for kind, model in models
    ]


def build_generators(
    arms: Sequence[Arm],
    *,
    dry_run: bool,
    custom_system: str | None,
    max_tokens: int,
    timeout: float | None,
    capture_reasoning: bool,
    temperature: float | None,
    top_p: float | None,
    provider_sort: str,
    sticky_provider: bool,
    prompt_cache: bool,
) -> list[synth.Generator]:
    """One generator per arm.

    Dry-run stubs (name-only, no API clients, no keys) and real factory
    generators share ``arm.name`` and the key covariates, so their synth_keys
    are identical by construction — dry-run resume counts are the real run's.
    Factories are reached as ``synth.<factory>`` attribute lookups on purpose:
    tests monkeypatch them on the synth module.
    """
    out: list[synth.Generator] = []
    for arm in arms:
        if dry_run:
            out.append(
                synth.Generator(
                    name=arm.name,
                    strategy=arm.strategy.label,
                    reasoning_effort=arm.effort,
                    prompt_id=arm.strategy.prompt_id,
                    prompt_version=arm.strategy.version,
                )
            )
            continue
        gen_kw = dict(
            strategy=arm.strategy.label, system=custom_system, max_tokens=max_tokens,
            reasoning_effort=arm.effort, temperature=temperature, top_p=top_p,
            timeout=timeout, capture_reasoning=capture_reasoning,
        )
        if arm.kind == "gpt":
            out.append(synth.openai_generator(model=arm.model, **gen_kw))
        elif arm.kind == "local":
            out.append(synth.local_generator(model=arm.model, **gen_kw))
        else:
            out.append(synth.openrouter_generator(
                model=arm.model,
                provider_sort=None if provider_sort == "none" else provider_sort,
                sticky_provider=sticky_provider, prompt_cache=prompt_cache, **gen_kw,
            ))
    return out


def synth_options(*, exclude: Iterable[str] = (), **default_overrides):
    """Decorator applying the shared synth option set.

    `default_overrides` swaps an option's default by param name (a wrapper's
    policy defaults); `exclude` omits options the wrapper pins as constants
    (its callback then must not take those params, and passes the pinned
    values straight to iter_targets/run_synth). Unknown names raise at import
    time — a typo must not silently leave a stylebot default in force.
    """
    exclude = set(exclude)
    unknown = (set(default_overrides) | exclude) - set(_OPTION_SPECS)
    if unknown:
        raise ValueError(f"unknown synth option param(s): {sorted(unknown)}")

    def deco(f):
        # Apply in reverse so help order matches _OPTION_SPECS order.
        for name in reversed(list(_OPTION_SPECS)):
            if name in exclude:
                continue
            decls, kw = _OPTION_SPECS[name]
            if name in default_overrides:
                kw = {**kw, "default": default_overrides[name], "show_default": True}
            f = click.option(*decls, **kw)(f)
        return f

    return deco


def _inspect_targets(
    targets: list[synth.Target],
    *,
    sample_n: int | None,
    report_path: Path | None,
    report_max_rows: int,
    report_title: str | None,
    data_dir: Path | Callable[[], Path],
    generation_flags_set: bool,
) -> None:
    """The read-only --sample/--report modes: print/write and return, before
    any generation — no --data-dir resolution, no API keys."""
    from stylebot import report

    click.echo("[inspection] targets only — nothing generated.", err=True)
    # Generation flags alongside --report/--sample are a classic trap: the
    # run exits before any generator fires. Say so, and point at the pair
    # browser (the eval scores report) that DOES show generated slop.
    if generation_flags_set:
        pairs_hint = f"{data_dir}/pairs.jsonl" if isinstance(data_dir, Path) else "<data-dir>/pairs.jsonl"
        click.echo(
            "[inspection] generation flags ignored; to generate then browse pairs: "
            "re-run without --report/--sample, then: "
            f"ai-style eval --pairs {pairs_hint} --report FILE.html --facet-by generator",
            err=True,
        )
    if sample_n is not None:
        click.echo(report.format_sample(report.sample_targets(targets, sample_n)))
    if report_path is not None:
        title_kw = {"title": report_title} if report_title else {}
        written = report.render_targets_report(
            targets, report_path, max_rows=(None if report_max_rows == 0 else report_max_rows), **title_kw
        )
        click.echo(f"wrote report -> {written}")


def _resolve_windows(
    generators: Sequence[synth.Generator],
    *,
    session_turns: int,
    dry_run: bool,
    context_window: int | None,
) -> dict[str, int]:
    """Per-generator context-window sizes — live sessions only (the overflow
    guard + the window_fill covariate). OpenRouter models resolve from the
    models registry (ground truth); presets / injected generators take
    --context-window."""
    windows: dict[str, int] = {}
    if session_turns <= 1 or dry_run:
        return windows
    or_names = [g.name for g in generators if g.name.startswith("openrouter/")]
    if or_names:
        try:
            registry = synth.openrouter_context_windows()
        except Exception as exc:
            raise click.ClickException(
                f"could not fetch OpenRouter context windows for the session sweep: {exc}"
            )
        for n in or_names:
            w = registry.get(n.removeprefix("openrouter/"))
            if w:
                windows[n] = w
    if context_window:
        for g in generators:
            windows.setdefault(g.name, context_window)
    return windows


def _model_economics(ms: dict | None) -> str:
    """Per-model cost/cache/throughput suffix for the exit summary.

    Everything here is folded from meta.gen covariates already recorded on the
    pairs — this prints run economics, it does not measure anything new. The
    cached share is the direct check on whether session prefix caching pays;
    tok/s (visible completion over wall generation time) is the provider-
    throughput check. Empty string when the generator reported no usage.
    """
    if not ms or not (ms["cost"] or ms["prompt_tokens"]):
        return ""
    parts = [f"${ms['cost']:.3f} (${ms['cost'] / ms['pairs']:.4f}/pair)"]
    if ms["prompt_tokens"]:
        parts.append(f"cached {100 * ms['cached_tokens'] / ms['prompt_tokens']:.0f}%")
    if ms["gen_seconds"] and ms["completion_tokens"]:
        parts.append(f"{ms['completion_tokens'] / ms['gen_seconds']:.0f} tok/s")
    if ms["reasoning_tokens"]:
        parts.append(f"{ms['reasoning_tokens'] / ms['pairs']:.0f} reasoning tok/pair")
    return "  " + "  ".join(parts)


def _report_result(
    result: synth.SynthResult, pairs_path: Path, *, dry_run: bool, session_turns: int,
    max_transform_sim: float | None = None,
) -> None:
    """Echo the run summary; exit 1 when any generation failed."""
    if dry_run:
        click.echo(
            f"[dry-run] would write {result.planned - result.skipped_existing} new pair(s) "
            f"({result.skipped_existing} already present) to {pairs_path}"
        )
        if result.skipped_covered:
            click.echo(f"  {result.skipped_covered} target(s) already covered, skipped (--skip-covered)")
        if session_turns > 1:
            click.echo(f"  {result.planned_sessions} live session(s) of <= {session_turns} turn(s)")
        for name, count in sorted(result.per_generator.items()):
            click.echo(f"  {name}: {count}")
        return
    click.echo(
        f"wrote {result.written} pair(s) "
        f"({result.skipped_existing} already present, skipped) -> {pairs_path}"
    )
    if result.skipped_covered:
        click.echo(f"  {result.skipped_covered} target(s) already covered, skipped (--skip-covered)")
    if result.skipped_degenerate:
        click.secho(
            f"  {result.skipped_degenerate} degenerate pair(s) DROPPED "
            f"(transform_sim > {max_transform_sim}) — paid for, not written; "
            "a persistently degenerate model should leave the roster",
            fg="yellow",
        )
    for name, count in sorted(result.per_generator.items()):
        click.echo(f"  {name}: {count}{_model_economics(result.model_stats.get(name))}")
    if result.total_cost:
        click.echo(f"  total cost: ${result.total_cost:.2f}")
    if result.budget_bound_sessions or result.reflow_sessions:
        click.echo(
            f"  {result.budget_bound_sessions} session(s) ended on the token budget; "
            f"{result.reflowed_turns} leftover turn(s) reflowed into "
            f"{result.reflow_sessions} fresh session(s)"
        )
    # The post-mortem invariant: every planned turn is written, skipped, or
    # individually errored. Anything else is silent work-dropping — the bug
    # class that cost 46% of a paid run in 2026-07 — so make it loud.
    unattempted = (
        result.planned - result.skipped_existing - result.written
        - result.skipped_degenerate - len(result.errors)
    )
    if unattempted > 0:
        click.secho(
            f"WARNING: {unattempted} planned turn(s) were never attempted — "
            "this should be impossible now that leftover turns reflow; report it.",
            fg="yellow",
            err=True,
        )
    if result.errors:
        click.echo(f"{len(result.errors)} generation error(s):", err=True)
        for key, msg in result.errors[:10]:
            click.echo(f"  {key}: {msg}", err=True)
        raise SystemExit(1)


def run_synth(
    targets: list[synth.Target],
    *,
    data_dir: Path | Callable[[], Path],
    generator_names: Sequence[str] = (),
    openrouter_models: Sequence[str] = (),
    provider_sort: str = "throughput",
    sticky_provider: bool = True,
    prompt_cache: bool = True,
    slop_strategies: Sequence[str] = (synth.DEFAULT_STRATEGY,),
    slop_system_file: Path | None = None,
    gpt_model: str = "gpt-4o",
    local_model: str = "",
    max_tokens: int = synth.DEFAULT_SLOP_MAX_TOKENS,
    timeout: float | None = synth.DEFAULT_GEN_TIMEOUT,
    capture_reasoning: bool = False,
    reasoning_efforts: Sequence[str] = (synth.DEFAULT_REASONING_EFFORT,),
    temperature: float | None = None,
    top_p: float | None = None,
    per_generator: bool = False,
    assign_seed: str = "",
    context_dropout: float = 0.0,
    max_workers: int = 0,
    session_turns: int = 1,
    session_max_tokens: int = synth.DEFAULT_SESSION_MAX_TOKENS,
    replicate: str = "",
    skip_covered: bool = False,
    max_transform_sim: float | None = None,
    context_window: int | None = None,
    session_budgets: Mapping[str, int] | None = None,
    splits_path: Path | None = None,
    role: str | None = None,
    sample_frac: float | None = None,
    sample_salt: str = "",
    limit: int | None = None,
    dry_run: bool = False,
    sample_n: int | None = None,
    report_path: Path | None = None,
    report_max_rows: int = 2000,
    report_title: str | None = None,
    count_note: str = "",
    extra_tags: Sequence[str] = (),
    generators: Sequence[synth.Generator] | None = None,
) -> synth.SynthResult | None:
    """Everything after target selection, shared by all synth-style commands.

    `data_dir` may be a callable so callers can defer resolution (and its
    "no silent default" guard) past the inspection modes, which need no
    corpus location or API keys. Library callers may inject ready-made
    `generators`, bypassing the preset/OpenRouter construction. Returns the
    SynthResult, or None when an inspection mode short-circuited. Exits 1 on
    generation errors.
    """
    # Substrate selection runs BEFORE --limit: role and sample-frac define
    # *which* population this run measures, --limit is only a cost cap on it.
    if role is not None or sample_frac is not None:
        from stylebot import splits as splits_mod
        from stylebot.targets import select_targets

        before = len(targets)
        loaded = splits_mod.load_splits(splits_path) if role is not None else None
        if role is not None and splits_path is None:
            raise click.UsageError("--role requires --splits")
        targets = select_targets(
            targets, splits=loaded, role=role, sample_frac=sample_frac, sample_salt=sample_salt
        )
        detail = ", ".join(
            p for p in (f"role={role}" if role else "", f"sample-frac={sample_frac}" if sample_frac else "") if p
        )
        click.echo(f"substrate: {before} -> {len(targets)} target chunk(s) ({detail})")

    if limit is not None:
        targets = targets[:limit]

    n_sources = len({t.source for t in targets})
    suffix = f" {count_note}" if count_note else ""
    click.echo(f"{len(targets)} target chunk(s) from {n_sources} source(s){suffix}")
    if not targets:
        click.echo("nothing to synthesize", err=True)
        return None

    # Read-only inspection modes — print/write and exit before any generation,
    # so no --data-dir or API keys are needed to vet the targets.
    if sample_n is not None or report_path is not None:
        _inspect_targets(
            targets,
            sample_n=sample_n,
            report_path=report_path,
            report_max_rows=report_max_rows,
            report_title=report_title,
            data_dir=data_dir,
            generation_flags_set=bool(
                generator_names or openrouter_models
                or tuple(slop_strategies) != (synth.DEFAULT_STRATEGY,)
            ),
        )
        return None

    if generators is None:
        # At least one slop source is required (no silent default — synth spends money
        # and may hit a provider key the operator hasn't configured).
        if not generator_names and not openrouter_models:
            raise click.UsageError(
                "no generators selected: pass --generator {gpt,local} and/or "
                "--openrouter-model MODEL (synth needs at least one slop source)"
            )
        # Resolve every requested slop prompt up front (fail fast on a bad name;
        # dedupe repeats, order-preserving). For a registry strategy the *name*
        # (system=None) goes down to the factory so it resolves the registry entry
        # and keeps its version; a custom --slop-system-file overrides the system
        # text and is single-strategy by construction (one file = one prompt).
        custom_system = slop_system_file.read_text(encoding="utf-8") if slop_system_file else None
        strategy_names = list(dict.fromkeys(slop_strategies)) or [synth.DEFAULT_STRATEGY]
        if custom_system is not None and len(strategy_names) > 1:
            raise click.UsageError(
                "--slop-system-file provides ONE custom prompt; pass exactly one "
                "--slop-strategy label for it (registry strategies can rotate, a file cannot)"
            )
        try:
            resolved_strategies = [synth.resolve_strategy(s, custom_system) for s in strategy_names]
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--slop-strategy")
        efforts = list(dict.fromkeys(reasoning_efforts)) or [synth.DEFAULT_REASONING_EFFORT]

        # The rotation is the models x strategies x efforts cross product:
        # round-robin assignment then spreads targets across every combination,
        # so one run diversifies prompts and reasoning depth the same way it
        # diversifies models (same cost — still one generation per target, just
        # a finer rotation).
        arms = iter_arms(
            resolved_strategies, efforts, generator_names, openrouter_models,
            gpt_model=gpt_model, local_model=local_model,
        )
        try:
            generators = build_generators(
                arms,
                dry_run=dry_run,
                custom_system=custom_system,
                max_tokens=max_tokens,
                timeout=timeout,
                capture_reasoning=capture_reasoning,
                temperature=temperature,
                top_p=top_p,
                provider_sort=provider_sort,
                sticky_provider=sticky_provider,
                prompt_cache=prompt_cache,
            )
        except RuntimeError as exc:  # missing key, surfaced by config.require_key
            raise click.ClickException(str(exc))

    # Per-model session budgets (policy hook, e.g. a blog's model->budget map).
    if session_budgets:
        for gen in generators:
            budget = session_budgets.get(gen.name) or session_budgets.get(
                gen.name.removeprefix("openrouter/")
            )
            if budget:
                gen.session_budget = budget

    # auto concurrency: OpenRouter multiplexes fine; a local/preset endpoint
    # (or injected generators we know nothing about) gets sequential.
    if max_workers <= 0:
        openrouter_only = bool(openrouter_models) and not generator_names
        max_workers = 16 if (openrouter_only and not dry_run) else 1

    windows = _resolve_windows(
        generators, session_turns=session_turns, dry_run=dry_run, context_window=context_window
    )

    resolved_dir = data_dir() if callable(data_dir) else data_dir

    # Heartbeat on wall-clock, not pair count: at high reasoning effort a pair
    # takes ~1-2 min, so count-gated echoes (every 25th) sit silent for half an
    # hour and look like a hang. on_progress fires before each pair, so after
    # any slow pair the very next tick prints.
    last_echo = [0.0]

    def _progress(i: int, total: int, live: synth.SynthResult) -> None:
        now = time.monotonic()
        if i == 1 or i == total or now - last_echo[0] >= _HEARTBEAT_SECS:
            last_echo[0] = now
            # Billed credits so far (OpenRouter usage.include ground truth);
            # silent for generators that report no cost.
            spent = f"  ≈${live.total_cost:.2f}" if live.total_cost else ""
            click.echo(f"  ... {i}/{total} pairs{spent}", err=True)

    def _on_error(key: str, msg: str) -> None:
        # Surface failures the moment they happen, not just in the exit summary.
        click.echo(f"  !! {key[:12]}: {msg}", err=True)

    result = synth.synthesize_pairs(
        targets,
        resolved_dir,
        generators,
        per_generator=per_generator,
        assign_seed=assign_seed,
        dry_run=dry_run,
        extra_tags=list(extra_tags),
        context_dropout=context_dropout,
        on_progress=None if dry_run else _progress,
        on_error=None if dry_run else _on_error,
        max_workers=max_workers,
        session_turns=session_turns,
        session_max_tokens=session_max_tokens,
        context_windows=windows or None,
        replicate=replicate,
        skip_covered=skip_covered,
        max_transform_sim=max_transform_sim,
    )

    _report_result(
        result, resolved_dir / "pairs.jsonl", dry_run=dry_run, session_turns=session_turns,
        max_transform_sim=max_transform_sim,
    )
    return result
