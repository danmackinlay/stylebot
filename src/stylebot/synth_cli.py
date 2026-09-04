"""Shared CLI kit for synth-style commands.

`ai-style synth` and downstream blog wrappers present the same knobs over the
same engine; only *selection* and a few pinned defaults differ per caller. This
module is the single source for the shared surface:

- the six frozen **option bundles** below, grouped by the consumer each feeds.
  Each field declares its own flag via `cli_options.opt()`, so a default exists
  in exactly one place;
- `synth_options(bundles=..., defaults=..., exclude=...)` — the click decorator
  over them;
- `run_synth(targets, ...)` — the whole command body after target selection:
  count echo, `--sample`/`--report` short-circuit, generator construction,
  `synthesize_pairs`, result reporting, and `SynthGenerationError`.

Per-command concerns stay with the caller: target selection options, the
`--data-dir` resolution policy, provenance tags. A callback's `**kw` becomes
bundles via `pop_bundles`. `ChunkPolicy` (re-exported from `stylebot.targets`)
feeds `iter_targets` on the *selection* side, before `run_synth` sees a target.

**click declares this module's options; it does not decide its control flow or
own its output.** The library entry points raise the `SynthError` family and
never exit, and they write to a caller-supplied `emit` sink defaulting to
silence; `bin/ai_style.py` decides exit codes and rendering (D47).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from stylebot import config, synth
from stylebot.cli_options import (
    Emitter,
    SectionedCommand,
    bundle_options,
    bundles_from_kwargs,
    click_emitter,
    field_owners,
    opt,
    silent,
)
from stylebot.errors import ConfigError, RunError, StylebotError
from stylebot.splits import ROLES as SPLIT_ROLES
from stylebot.targets import ChunkPolicy

__all__ = [
    "Arm",
    "BUNDLE_PARAMS",
    "ChunkPolicy",
    "CorpusOpts",
    "DEFAULT_BUNDLES",
    "Emitter",
    "GenOpts",
    "InspectOpts",
    "SectionedCommand",
    "SessionOpts",
    "SubstrateOpts",
    "SynthConfigError",
    "SynthError",
    "SynthGenerationError",
    "SynthRunError",
    "build_generators",
    "click_emitter",
    "iter_arms",
    "pop_bundles",
    "run_synth",
    "synth_options",
]

# Progress heartbeat cadence for generation (seconds of wall-clock silence).
_HEARTBEAT_SECS = 20.0


# ---------------------------------------------------------------------------
# The error vocabulary: three named specializations of `stylebot.errors`' two
# classes (D47). A library function reachable without click must not hand its
# caller a CLI exception, so nothing here raises one or exits the process;
# `cli_options.translate_errors` is where they become exit codes.
# ---------------------------------------------------------------------------


class SynthError(StylebotError):
    """Base for every failure this kit raises on purpose."""


class SynthConfigError(SynthError, ConfigError):
    """A synth-shaped `stylebot.errors.ConfigError` — exit 2.

    Kept as a name so `except SynthConfigError` can mean "this run's flags",
    not "some flag somewhere"; the exit code and the `param_hint` come from the
    package vocabulary.
    """


class SynthRunError(SynthError, RunError):
    """A synth-shaped `stylebot.errors.RunError` — exit 1.

    A missing API key, an unreachable model registry.
    """


class SynthGenerationError(SynthRunError):
    """The run completed but some pairs failed to generate.

    Carries the `SynthResult` so a library caller can still see what landed.
    `quiet`, because the per-error detail and the run summary have already been
    reported by the time this is raised: exit 1 with nothing further to print.
    """

    def __init__(self, result: synth.SynthResult) -> None:
        super().__init__(f"{len(result.errors)} generation error(s)", quiet=True)
        self.result = result


# Slop models are selected by `--openrouter-model` alone: one key reaches many
# upstream models, which is what a multi-source rotation wants. There is no
# direct-API preset path (decisions.md D30). A library caller that wants a
# different provider injects its own `synth.Generator`s.


# ---------------------------------------------------------------------------
# The option bundles: the surface grouped along its seams, one frozen dataclass
# per consumer. Each field carries its own flag (`opt(default, *decls, **kw)`),
# so a default is declared exactly once and `run_synth` cannot drift from the
# CLI; a field WITHOUT `opt()` is library-only. `ChunkPolicy` is the one bundle
# not declared here — it belongs to `stylebot.targets` — but is re-exported, so
# a wrapper reaches it as `synth_cli.ChunkPolicy`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InspectOpts:
    """Read-only inspection — `--sample` / `--report`, handled by
    `_inspect_targets`, which exits before any generation (no data-dir, no
    keys)."""

    cli_section = "Inspection (read-only; exits before generation)"

    report_path: Path | None = opt(
        None,
        "--report",
        "report_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Write a self-contained HTML report of the selected targets and exit "
        "(pre-generation — vets selection, generates nothing; browse generated pairs via "
        "ai-style eval --report).",
    )
    sample_n: int | None = opt(
        None,
        "--sample",
        "sample_n",
        type=int,
        help="Print N random targets to stdout and exit (no generation).",
    )
    # Library-only, no flags. `report_title` is a caller-supplied label;
    # `report_max_rows` caps table rows (0 = all) and lost its `--report-max-rows`
    # flag in 0.3.1 (synth-flag-audit C2) — the default had never been raised or
    # zeroed, and a caller wanting another cap is already writing code.
    report_title: str | None = None
    report_max_rows: int = 2000


@dataclass(frozen=True)
class GenOpts:
    """Rotation / generation — which arms exist and how each request is made.

    Feeds `iter_arms` (the models x strategies x efforts cross product, whose
    *membership* — not its order — decides which arm owns which target) and
    `build_generators` (per-request knobs). `dry_run` lives here because it is
    a generator-construction switch: stubs instead of API clients. It is read
    downstream too (concurrency, windows, reporting).
    """

    cli_section = "Generation"

    openrouter_models: Sequence[str] = opt(
        (),
        "--openrouter-model",
        "openrouter_models",
        multiple=True,
        help="OpenRouter model id to generate slop with (repeatable), e.g. "
        "anthropic/claude-opus-4-8. One key ($OPENROUTER_API_KEY) reaches many upstream "
        "models — ideal for multi-source rotation.",
    )
    provider_sort: str = opt(
        "throughput",
        "--provider-sort",
        type=click.Choice(["throughput", "price", "latency", "none"]),
        show_default=True,
        help="OpenRouter provider-routing preference (default load-balancing favours price and "
        "can land on ~10 tok/s upstreams; 'none' restores it). OpenRouter models only; "
        "recorded in meta.gen next to the served provider.",
    )
    sticky_provider: bool = opt(
        True,
        "--sticky-provider/--no-sticky-provider",
        show_default=True,
        help="Pin each live session to the provider that served its first turn — keeps its "
        "prefix cache hot AND the serving stack constant, so window-position covariates "
        "aren't confounded by provider hops. OpenRouter models only; no effect on stateless "
        "runs.",
    )
    prompt_cache: bool = opt(
        True,
        "--prompt-cache/--no-prompt-cache",
        show_default=True,
        help="Anthropic models: mark session history with a moving cache_control breakpoint "
        "(cache reads at 0.1x after a 1.25x write; inert under ~1024-token prefixes and for "
        "other model families). Verify via meta.gen cached_tokens/cost.",
    )
    slop_strategies: Sequence[str] = opt(
        (synth.DEFAULT_STRATEGY,),
        "--slop-strategy",
        "slop_strategies",
        multiple=True,
        show_default=True,
        help=f"Slop-prompt flavour ({', '.join(synth.STRATEGIES)}); repeatable — the rotation "
        "becomes models x strategies, so one run diversifies prompts like it already "
        "diversifies models. Recorded as meta.slop_strategy and folded into synth_key. With "
        "--slop-system-file, pass exactly one label for the custom prompt.",
    )
    slop_system_file: Path | None = opt(
        None,
        "--slop-system-file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Override the slop system prompt with this file's contents (e.g. an author's own "
        "slop catalogue), labelled by --slop-strategy. Keeps blog-specific prompts out of "
        "stylebot.",
    )
    max_tokens: int = opt(
        synth.DEFAULT_SLOP_MAX_TOKENS,
        "--max-tokens",
        show_default=True,
        type=int,
        help="Max completion tokens per slop generation. Raise if a model truncates.",
    )
    timeout: float | None = opt(
        synth.DEFAULT_GEN_TIMEOUT,
        "--timeout",
        show_default=True,
        type=float,
        help="Per-request HTTP timeout (seconds) for slop generation. A timed-out pair is "
        "reported immediately and recorded as an error; it ends its own session (remaining "
        "turns defer to the next run) while other sessions continue.",
    )
    reasoning_efforts: Sequence[str] = opt(
        (synth.DEFAULT_REASONING_EFFORT,),
        "--reasoning-effort",
        "reasoning_efforts",
        type=click.Choice(["high", "medium", "low", "off"]),
        multiple=True,
        show_default=True,
        help="Reasoning effort for the slop generator ('off' disables reasoning); repeatable — "
        "the rotation becomes models x strategies x efforts, so effort joins the sweep "
        "(facet eval by reasoning_effort). Recorded in meta.gen and folded into synth_key; "
        "the requested effort is recorded regardless of what the provider honors.",
    )
    capture_reasoning: bool = opt(
        False,
        "--capture-reasoning",
        is_flag=True,
        help="Save each pair's reasoning/thinking trace to <data-dir>/reasoning.jsonl (keyed by "
        "synth_key; never enters pairs.jsonl). Diagnose reasoning blowouts by READING what "
        "the model deliberated about; costs nothing extra (the tokens were already paid for).",
    )
    temperature: float | None = opt(
        None,
        "--temperature",
        type=float,
        help="Sampling temperature, sent to the API and recorded (provider default if unset).",
    )
    top_p: float | None = opt(
        None,
        "--top-p",
        "top_p",
        type=float,
        help="Nucleus top_p, sent to the API and recorded (provider default if unset).",
    )
    dry_run: bool = opt(
        False,
        "--dry-run",
        is_flag=True,
        help="Report planned targets/pairs without calling any API or writing.",
    )


@dataclass(frozen=True)
class SessionOpts:
    """Live-session shape and the concurrency that runs them."""

    cli_section = "Sessions / concurrency"

    session_turns: int = opt(
        1,
        "--session-turns",
        show_default=True,
        type=int,
        help="Session mode switch + backstop: 1 = stateless; >1 groups turns into live sessions "
        "(each turn sees the real prior passage->slop exchanges, meta.gen session_turn / "
        "window_fill). Depth is controlled by --session-max-tokens, NOT this number — "
        "sessions end when the token budget binds and leftover turns reflow into fresh "
        "sessions; the value only caps turns per session as a runaway guard.",
    )
    session_max_tokens: int = opt(
        synth.DEFAULT_SESSION_MAX_TOKENS,
        "--session-max-tokens",
        show_default=True,
        type=int,
        help="THE session depth control: per-session prompt-token budget (absolute; input cost "
        "grows ~quadratically with it). Also capped at 80% of each model's context window. "
        "Turns beyond the budget reflow into fresh sessions, never dropped.",
    )
    max_workers: int = opt(
        0,
        "--max-workers",
        show_default="auto",
        type=int,
        help="Concurrent sessions (asyncio). 0 = auto: 16 for the OpenRouter rotation this CLI "
        "builds, 1 for a library caller's injected generators (an unknown endpoint may want "
        "sequential).",
    )
    # Library-only policy hook: a wrapper's per-model budget map (the blog pins
    # one), applied on top of `session_max_tokens`.
    session_budgets: Mapping[str, int] | None = None


@dataclass(frozen=True)
class SubstrateOpts:
    """WHICH population this run measures — applied before `--limit`.

    `role`/`sample_frac` define the substrate (`select_targets`); `limit` is
    only a cost cap on it, which is why it is applied afterwards.
    """

    cli_section = "Substrate (which population this run measures)"

    splits_path: Path | None = opt(
        None,
        "--splits",
        "splits_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="splits.json of by-post roles (see ai-style make-splits). Required by --role.",
    )
    role: str | None = opt(
        None,
        "--role",
        type=click.Choice(SPLIT_ROLES),
        help="Keep only chunks whose post holds this role. Measurement runs want 'styler': "
        "disjoint from the detector's training pool, so the voice classifier stays an "
        "unbiased instrument.",
    )
    sample_frac: float | None = opt(
        None,
        "--sample-frac",
        type=float,
        help="Keep this fraction of chunks by content hash. Growth-stable: a later run on a "
        "larger corpus is a SUPERSET of this one, so the two stay comparable (--limit is "
        "not, its first N shifts as posts are added).",
    )
    sample_salt: str = opt(
        "",
        "--sample-salt",
        show_default=True,
        help="Re-draw an independent subsample of the same size (a fresh replicate of the design).",
    )
    limit: int | None = opt(
        None,
        "--limit",
        type=int,
        help="Cap the number of target chunks (cost control / smoke runs). A cap on the "
        "substrate, not a definition of it — see --sample-frac.",
    )


@dataclass(frozen=True)
class CorpusOpts:
    """What gets written, under which design cell — feeds `synthesize_pairs`.

    Two clusters that both bear on corpus identity: the arm-assignment design
    (`per_generator` / `assign_seed` / `replicate` / `skip_covered`, which
    decide *which cells exist* and therefore what resume regenerates) and the
    write gates (`context_dropout` and the two hygiene thresholds, which decide
    which generated pairs survive to the file).
    """

    cli_section = "Corpus (design cell + write gates)"

    per_generator: bool = opt(
        False,
        "--per-generator",
        is_flag=True,
        help="Emit a pair from EVERY generator per target (n× cost) — the fully-crossed "
        "within-target design for experiments; default assigns each target ONE arm by "
        "content hash.",
    )
    assign_seed: str = opt(
        "",
        "--assign-seed",
        help="Salt for the content-hash arm assignment. Change it to re-randomize which (model, "
        "strategy, effort) arm each target gets — a fresh replicate of the design (changed "
        "assignments regenerate on resume).",
    )
    context_dropout: float = opt(
        0.0,
        "--context-dropout",
        show_default=True,
        type=float,
        help="Fraction of pairs to keep heading-less (deterministic) so the styler doesn't "
        "require a heading.",
    )
    replicate: str = opt(
        "",
        "--replicate",
        show_default=True,
        help="Deliberate-resample label folded into synth_key: the same substrate under a new "
        "label mints new cells without colliding with the base corpus (e.g. 'deep128k' for a "
        "deep-window arm, 'draw2' for a second sample). Recorded as meta.gen.replicate. "
        "Empty = the base corpus.",
    )
    skip_covered: bool = opt(
        False,
        "--skip-covered/--no-skip-covered",
        show_default=True,
        help="Corpus-building coverage mode: skip any target whose text already has >=1 pair in "
        "the data-dir under ANY config or epoch (cell-level dedup still applies to the "
        "rest). Matching ignores cosmetic drift the styler cannot see — rewritten link "
        "destinations, curly-vs-straight quotes, whitespace runs — so a passage edited by a "
        "link-repair or reflow pass since it was generated still reads as covered. Use when "
        "the goal is one pair per passage — without it, a re-key epoch (e.g. an "
        "effort-default change) regenerates already-covered targets and doubles their "
        "training weight.",
    )
    max_transform_sim: float | None = opt(
        None,
        "--max-transform-sim",
        type=float,
        help="Identity gate: drop (don't write) any generated pair whose transform_sim exceeds "
        "this — the model returned (nearly) the input, which would teach the styler to copy. "
        "Dropped pairs are counted loudly (skipped_degenerate) and their cells retry next "
        "run. Unset = record everything (the covariate is always stored).",
    )
    max_length_ratio: float | None = opt(
        None,
        "--max-length-ratio",
        type=float,
        help="Inflation gate: drop any generated pair whose slop/target length ratio exceeds "
        "this — a manically inflating model (qwen3-32b ran 4-7x) teaches manic deletion. "
        "Counted as skipped_inflated; cells retry next run. Unset = record everything.",
    )
    drop_invented_anchors: bool = opt(
        True,
        "--drop-invented-anchors/--keep-invented-anchors",
        show_default=True,
        help="Fabrication gate: drop any generated pair whose Dan-side target carries a link, "
        "URL or citation key its slop side never had. The generator dropped the link while "
        "'improving' the passage, and reversing the edit turns that into a lesson in "
        "conjuring destinations the model cannot know — the most-taught failure in the "
        "corpus (18.6% of pairs) and the one nothing checked until 0.3.0. Counted as "
        "skipped_invented_anchors; cells retry next run. On by default (unlike the two "
        "threshold gates) because there is no number to tune and no defensible other side.",
    )
    gen_retries: int = opt(
        synth.DEFAULT_GEN_RETRIES,
        "--gen-retries",
        show_default=True,
        type=int,
        help="Extra draws after a TRANSIENT upstream failure (502/429/timeout), asking for a "
        "fresh provider each time. Not the same as --gate-retries, which redraws a draw that "
        "ARRIVED and was rejected on quality: an unreachable provider and a careless one want "
        "different remedies, and pooling the counts hides which is eating a run. 0 disables.",
    )
    gate_retries: int = opt(
        0,
        "--gate-retries",
        show_default=True,
        type=int,
        help="On a hygiene-gate rejection, redraw the same cell up to N more times before "
        "giving up on it. A rejected draw is paid for either way, so a redraw costs no more "
        "than the drop already did — and rejection is mostly a per-sample accident, so most "
        "clear. Worth setting whenever a gate is on: without it a gate biases coverage "
        "toward the passages the roster finds easy (link-dense prose is rejected twice as "
        "often as single-link prose), which is the opposite of what the gate is for.",
    )


# The shared surface, keyed by the parameter each bundle binds to downstream:
# "chunk" goes to `iter_targets`, the rest are `run_synth`'s own bundle params.
BUNDLE_PARAMS: dict[str, type] = {
    "chunk": ChunkPolicy,
    "inspection": InspectOpts,
    "gen": GenOpts,
    "session": SessionOpts,
    "substrate": SubstrateOpts,
    "corpus": CorpusOpts,
}
DEFAULT_BUNDLES: tuple[type, ...] = tuple(BUNDLE_PARAMS.values())

# Two bundles must not claim one field name, and no field may collide with a
# parameter `run_synth` already has — either would make routing ambiguous.
field_owners(
    DEFAULT_BUNDLES,
    reserved=BUNDLE_PARAMS.keys()
    | {
        "targets",
        "data_dir",
        "count_note",
        "emit",
        "extra_tags",
        "generators",
    },
)


def synth_options(
    *,
    bundles: Sequence[type] = DEFAULT_BUNDLES,
    defaults: Iterable[object] = (),
    exclude: Iterable[str] = (),
):
    """Decorator applying the shared synth option set.

    `defaults` re-pins flag defaults as bundle instances
    (`defaults=[GenOpts(max_tokens=900)]`); `exclude` drops flags the wrapper
    pins as policy constants (its callback never sees them, and it sets those
    fields on the bundle itself); `bundles` is the extension point — another
    repo passes `[*DEFAULT_BUNDLES, MyRepoOpts]` to add its own options.

    Use `cls=SectionedCommand` on the command for sectioned `--help`.
    """
    return bundle_options(bundles, defaults=defaults, exclude=exclude)


def pop_bundles(kw: dict, bundles: Mapping[str, type] = BUNDLE_PARAMS) -> dict[str, object]:
    """Fold a click callback's `**kw` into bundle instances, keyed by the
    parameter each binds to (`chunk` for `iter_targets`, the rest for
    `run_synth`). `kw` is left holding only the command's own parameters; a
    wrapper with extra bundles passes its own `{name: cls}` map."""
    return bundles_from_kwargs(kw, bundles)


@dataclass(frozen=True)
class Arm:
    """One rotation arm: a (model, slop strategy, reasoning effort) combination.

    Reifies the unit rendezvous assignment spreads targets across. ``name`` is
    the single computation of the generator name that feeds ``synth_key`` —
    dry-run stubs and real factory generators both take it from here, so the
    two paths cannot drift apart. Every arm is OpenRouter-served, so ``name``
    is unconditionally ``openrouter/<model>``: that spelling is a corpus
    invariant, not a formatting choice, since changing it re-keys every cell.
    """

    model: str  # OpenRouter model id
    strategy: synth.ResolvedStrategy
    effort: str

    @property
    def name(self) -> str:
        return f"openrouter/{self.model}"


def iter_arms(
    resolved_strategies: Sequence[synth.ResolvedStrategy],
    efforts: Sequence[str],
    openrouter_models: Sequence[str],
) -> list[Arm]:
    """The rotation cross product: strategies (outer) × efforts × OpenRouter
    models in flag order.

    `synth._assign` scores arms by rendezvous hashing on arm identity, so this
    enumeration order does not affect assignment — reordering it reassigns
    nothing. Adding or dropping an arm moves only that arm's own share.
    """
    return [
        Arm(model=model, strategy=rs, effort=effort)
        for rs in resolved_strategies
        for effort in efforts
        for model in openrouter_models
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
    The factory is reached as ``synth.openrouter_generator`` (attribute lookup
    on purpose: tests monkeypatch it on the synth module).
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
        out.append(
            synth.openrouter_generator(
                model=arm.model,
                strategy=arm.strategy.label,
                system=custom_system,
                max_tokens=max_tokens,
                reasoning_effort=arm.effort,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                capture_reasoning=capture_reasoning,
                provider_sort=None if provider_sort == "none" else provider_sort,
                sticky_provider=sticky_provider,
                prompt_cache=prompt_cache,
            )
        )
    return out


def _inspect_targets(
    targets: list[synth.Target],
    opts: InspectOpts,
    *,
    data_dir: Path | None,
    generation_flags_set: bool,
    emit: Emitter,
) -> None:
    """The read-only --sample/--report modes: print/write and return, before
    any generation — no API keys, and `data_dir` may be None (the caller skips
    resolving it when nothing will be written).

    `emit` is required (not defaulted): a helper whose whole job is output must
    never guess where it goes — see the module docstring."""
    from stylebot import report

    emit("[inspection] targets only — nothing generated.", level="status")
    # Generation flags alongside --report/--sample are a classic trap: the
    # run exits before any generator fires. Say so, and point at the pair
    # browser (the eval scores report) that DOES show generated slop.
    if generation_flags_set:
        pairs_hint = f"{data_dir}/pairs.jsonl" if data_dir is not None else "<data-dir>/pairs.jsonl"
        emit(
            "[inspection] generation flags ignored; to generate then browse pairs: "
            "re-run without --report/--sample, then: "
            f"ai-style eval --pairs {pairs_hint} --report FILE.html --facet-by generator",
            level="status",
        )
    if opts.sample_n is not None:
        emit(report.format_sample(report.sample_targets(targets, opts.sample_n)))
    if opts.report_path is not None:
        title_kw = {"title": opts.report_title} if opts.report_title else {}
        written = report.render_targets_report(
            targets,
            opts.report_path,
            max_rows=(None if opts.report_max_rows == 0 else opts.report_max_rows),
            **title_kw,
        )
        emit(f"wrote report -> {written}")


def _resolve_windows(
    generators: Sequence[synth.Generator],
    *,
    session_turns: int,
    dry_run: bool,
) -> dict[str, int]:
    """Per-generator context-window sizes — live sessions only (the overflow
    guard + the window_fill covariate). Resolved from the OpenRouter models
    registry, which is ground truth — there is no manual-override flag, and
    every window value ever recorded on the corpus came from the registry. An
    injected generator with a non-OpenRouter name simply gets no window (no
    overflow guard, no window_fill); a library caller that needs one passes
    `context_windows=` to `synth.synthesize_pairs` directly."""
    windows: dict[str, int] = {}
    if session_turns <= 1 or dry_run:
        return windows
    or_names = [g.name for g in generators if g.name.startswith("openrouter/")]
    if or_names:
        try:
            registry = synth.openrouter_context_windows()
        except Exception as exc:
            raise SynthRunError(
                f"could not fetch OpenRouter context windows for the session sweep: {exc}"
            ) from exc
        for n in or_names:
            w = registry.get(n.removeprefix("openrouter/"))
            if w:
                windows[n] = w
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
    result: synth.SynthResult,
    pairs_path: Path,
    *,
    dry_run: bool,
    session_turns: int,
    emit: Emitter,
    max_transform_sim: float | None = None,
    max_length_ratio: float | None = None,
) -> None:
    """Render the run summary to `emit` — reporting only.

    Reporting only: deciding the run failed is `run_synth`'s (it raises
    `SynthGenerationError`), and deciding what that costs the process is the
    CLI's. `emit` is required rather than defaulted for the same reason: a
    helper whose whole job is output decides *what* to say, never *where* it
    lands.
    """
    if dry_run:
        emit(
            f"[dry-run] would write {result.planned - result.skipped_existing} new pair(s) "
            f"({result.skipped_existing} already present) to {pairs_path}"
        )
        if result.skipped_covered:
            emit(f"  {result.skipped_covered} target(s) already covered, skipped (--skip-covered)")
        if session_turns > 1:
            emit(f"  {result.planned_sessions} live session(s) of <= {session_turns} turn(s)")
        for name, count in sorted(result.per_generator.items()):
            emit(f"  {name}: {count}")
        return
    emit(
        f"wrote {result.written} pair(s) "
        f"({result.skipped_existing} already present, skipped) -> {pairs_path}"
    )
    if result.skipped_covered:
        emit(f"  {result.skipped_covered} target(s) already covered, skipped (--skip-covered)")
    if result.skipped_degenerate:
        emit(
            f"  {result.skipped_degenerate} identity pair(s) DROPPED "
            f"(transform_sim > {max_transform_sim}) — paid for, not written; "
            "a persistently degenerate model should leave the roster",
            level="note",
        )
    if result.skipped_inflated:
        emit(
            f"  {result.skipped_inflated} inflated pair(s) DROPPED "
            f"(length ratio > {max_length_ratio}) — paid for, not written; "
            "a persistently inflating model should leave the roster",
            level="note",
        )
    if result.skipped_invented_anchors:
        emit(
            f"  {result.skipped_invented_anchors} pair(s) DROPPED where the generator "
            "deleted a link the target keeps — paid for, not written; reversed, those "
            "pairs would teach the styler to invent links",
            level="note",
        )
    if result.gen_retries_spent or result.errors_by_generator:
        # The split the printed-and-discarded errors could never show: how much
        # of a run's shortfall is an unreachable provider versus a careless one.
        lost = sum(result.errors_by_generator.values())
        emit(
            f"  {result.gen_retries_spent} transient upstream failure(s) survived by redrawing"
            + (f"; {lost} cell(s) lost to one anyway" if lost else "")
            + (" — see lost_draws.jsonl (kind=error)" if lost else "")
        )
    if result.redrawn_cells:
        # The redraw budget's own report card: cells recovered vs draws spent.
        # A low recovery rate means the gate is finding something structural
        # (a passage no arm handles) and more retries will not fix it.
        emit(
            f"  {result.redrawn_cells} cell(s) hit a gate and were redrawn "
            f"({result.gate_redraws} extra draw(s) spent, "
            f"{result.redraws_recovered} recovered)"
        )
    for name, count in sorted(result.per_generator.items()):
        # Written, then rejected-as-a-share-of-draws. The rate is the roster
        # signal — an arm can be cheap per written pair and still be the most
        # expensive one, because rejected draws are paid for and invisible in
        # `pairs.jsonl` (llama-3.3-70b left the roster on exactly this number).
        dropped = result.drops_by_generator.get(name, 0)
        rate = (
            f"  {dropped} dropped ({100 * dropped / (count + dropped):.0f}% of draws)"
            if dropped
            else ""
        )
        economics = _model_economics(result.model_stats.get(name))
        emit(f"  {name}: {count}{rate}{economics}")
    for name, dropped in sorted(result.drops_by_generator.items()):
        # An arm that wrote NOTHING never reaches the loop above, which is the
        # case most worth shouting about.
        if name not in result.per_generator:
            emit(f"  {name}: 0 written, {dropped} draw(s) all rejected", level="note")
    if result.total_cost:
        emit(f"  total cost: ${result.total_cost:.2f}")
    if result.budget_bound_sessions or result.reflow_sessions:
        emit(
            f"  {result.budget_bound_sessions} session(s) ended on the token budget; "
            f"{result.reflowed_turns} leftover turn(s) reflowed into "
            f"{result.reflow_sessions} fresh session(s)"
        )
    # The post-mortem invariant: every planned turn is written, skipped, or
    # individually errored. Anything else is silent work-dropping — the bug
    # class that cost 46% of a paid run in 2026-07 — so make it loud.
    unattempted = (
        result.planned
        - result.skipped_existing
        - result.written
        - result.skipped_degenerate
        - result.skipped_inflated
        - result.skipped_invented_anchors
        - len(result.errors)
    )
    if unattempted > 0:
        emit(
            f"WARNING: {unattempted} planned turn(s) were never attempted — "
            "this should be impossible now that leftover turns reflow; report it.",
            level="warn",
        )
    if result.errors:
        emit(f"{len(result.errors)} generation error(s):", level="error")
        for key, msg in result.errors[:10]:
            emit(f"  {key}: {msg}", level="error")


def run_synth(
    targets: list[synth.Target],
    *,
    data_dir: Path | None,
    gen: GenOpts | None = None,
    session: SessionOpts | None = None,
    substrate: SubstrateOpts | None = None,
    inspection: InspectOpts | None = None,
    corpus: CorpusOpts | None = None,
    count_note: str = "",
    emit: Emitter | None = None,
    extra_tags: Sequence[str] = (),
    generators: Sequence[synth.Generator] | None = None,
) -> synth.SynthResult | None:
    """Everything after target selection, shared by all synth-style commands.

    The knobs arrive as **option bundles** — ``run_synth(targets, data_dir=d,
    gen=GenOpts(max_tokens=900))``, or splatted from ``pop_bundles(kw)``.
    `ChunkPolicy` is not among them: it is consumed on the selection side.
    `data_dir` may be None **only** for the inspection modes, which write
    nothing and need no keys; the caller resolves it (and owns its "no silent
    default" message) after seeing `--sample`/`--report`. Library callers may
    inject ready-made `generators`. Returns the `SynthResult`, or None when
    short-circuited.

    `emit` takes the counts, heartbeat, errors and summary and **defaults to
    silence**; a CLI passes `click_emitter()`. Raises `SynthConfigError`,
    `SynthRunError` or `SynthGenerationError` (which carries the result), and
    never exits the process (D47).
    """
    say: Emitter = emit if emit is not None else silent
    gen = gen if gen is not None else GenOpts()
    session = session if session is not None else SessionOpts()
    substrate = substrate if substrate is not None else SubstrateOpts()
    inspection = inspection if inspection is not None else InspectOpts()
    corpus = corpus if corpus is not None else CorpusOpts()

    from stylebot import splits as splits_mod

    loaded: dict | None = None

    # Substrate selection runs BEFORE --limit: role and sample-frac define
    # *which* population this run measures, --limit is only a cost cap on it.
    if substrate.role is not None or substrate.sample_frac is not None:
        from stylebot.targets import select_targets

        before = len(targets)
        if substrate.role is not None and substrate.splits_path is None:
            raise SynthConfigError("--role requires --splits")
        loaded = (
            splits_mod.load_splits(substrate.splits_path) if substrate.role is not None else None
        )
        targets = select_targets(
            targets,
            splits=loaded,
            role=substrate.role,
            sample_frac=substrate.sample_frac,
            sample_salt=substrate.sample_salt,
        )
        detail = ", ".join(
            p
            for p in (
                f"role={substrate.role}" if substrate.role else "",
                f"sample-frac={substrate.sample_frac}" if substrate.sample_frac else "",
            )
            if p
        )
        say(f"substrate: {before} -> {len(targets)} target chunk(s) ({detail})")

    # The frozen-eval guard (D34): cutting eval-pinned prose as a synthesis
    # target trains the styler on the metric, and `--role` is optional, so this
    # does not wait to be asked — it consults the corpus's own `splits.json`
    # when `--splits` was not given, and only `--role eval` gets past it.
    if substrate.role != "eval":
        eval_splits = loaded
        if eval_splits is None:
            eval_splits = (
                splits_mod.load_splits(substrate.splits_path)
                if substrate.splits_path is not None
                else splits_mod.load_corpus_splits(data_dir)
                if data_dir is not None
                else None
            )
        if eval_splits is not None:
            from stylebot.targets import split_off_role

            refused, targets = split_off_role(targets, splits=eval_splits, role="eval")
            if refused:
                posts = sorted({t.source for t in refused})
                shown = ", ".join(posts[:3]) + (" ..." if len(posts) > 3 else "")
                say(
                    f"eval guard: refused {len(refused)} target chunk(s) from "
                    f"{len(posts)} frozen-eval post(s) ({shown}) — pass --role eval "
                    f"to synthesize on the eval stratum deliberately",
                    level="warn",
                )

    if substrate.limit is not None:
        targets = targets[: substrate.limit]

    n_sources = len({t.source for t in targets})
    suffix = f" {count_note}" if count_note else ""
    say(f"{len(targets)} target chunk(s) from {n_sources} source(s){suffix}")
    if not targets:
        say("nothing to synthesize", level="status")
        return None

    # Read-only inspection modes — print/write and exit before any generation,
    # so no --data-dir or API keys are needed to vet the targets.
    if inspection.sample_n is not None or inspection.report_path is not None:
        _inspect_targets(
            targets,
            inspection,
            data_dir=data_dir,
            generation_flags_set=bool(
                gen.openrouter_models or tuple(gen.slop_strategies) != (synth.DEFAULT_STRATEGY,)
            ),
            emit=say,
        )
        return None

    if generators is None:
        # At least one slop source is required (no silent default — synth spends money
        # and may hit a provider key the operator hasn't configured).
        if not gen.openrouter_models:
            raise SynthConfigError(
                "no generators selected: pass --openrouter-model MODEL "
                "(synth needs at least one slop source)"
            )
        # Resolve every requested slop prompt up front (fail fast on a bad name;
        # dedupe repeats, order-preserving). For a registry strategy the *name*
        # (system=None) goes down to the factory so it resolves the registry entry
        # and keeps its version; a custom --slop-system-file overrides the system
        # text and is single-strategy by construction (one file = one prompt).
        custom_system = (
            gen.slop_system_file.read_text(encoding="utf-8") if gen.slop_system_file else None
        )
        strategy_names = list(dict.fromkeys(gen.slop_strategies)) or [synth.DEFAULT_STRATEGY]
        if custom_system is not None and len(strategy_names) > 1:
            raise SynthConfigError(
                "--slop-system-file provides ONE custom prompt; pass exactly one "
                "--slop-strategy label for it (registry strategies can rotate, a file cannot)"
            )
        try:
            resolved_strategies = [synth.resolve_strategy(s, custom_system) for s in strategy_names]
        except ValueError as exc:
            raise SynthConfigError(str(exc), param_hint="--slop-strategy") from exc
        efforts = list(dict.fromkeys(gen.reasoning_efforts)) or [synth.DEFAULT_REASONING_EFFORT]

        # The rotation is the models x strategies x efforts cross product:
        # rendezvous assignment then spreads targets across every combination,
        # so one run diversifies prompts and reasoning depth the same way it
        # diversifies models (same cost — still one generation per target, just
        # a finer rotation).
        arms = iter_arms(resolved_strategies, efforts, gen.openrouter_models)
        try:
            generators = build_generators(
                arms,
                dry_run=gen.dry_run,
                custom_system=custom_system,
                max_tokens=gen.max_tokens,
                timeout=gen.timeout,
                capture_reasoning=gen.capture_reasoning,
                temperature=gen.temperature,
                top_p=gen.top_p,
                provider_sort=gen.provider_sort,
                sticky_provider=gen.sticky_provider,
                prompt_cache=gen.prompt_cache,
            )
        except RuntimeError as exc:  # missing key, surfaced by config.require_key
            raise SynthRunError(str(exc)) from exc

    # Per-model session budgets (policy hook, e.g. a blog's model->budget map).
    if session.session_budgets:
        for generator in generators:
            budget = session.session_budgets.get(generator.name) or session.session_budgets.get(
                generator.name.removeprefix("openrouter/")
            )
            if budget:
                generator.session_budget = budget

    # auto concurrency: OpenRouter multiplexes fine, so a run we built ourselves
    # gets 16. Injected generators (a library caller's own endpoint, which we
    # know nothing about) still get sequential.
    max_workers = session.max_workers
    if max_workers <= 0:
        max_workers = 16 if (gen.openrouter_models and not gen.dry_run) else 1

    windows = _resolve_windows(
        generators,
        session_turns=session.session_turns,
        dry_run=gen.dry_run,
    )

    # Everything below writes: a data-dir is mandatory from here on. The
    # inspection modes returned above, which is why the caller is free to hand
    # in None for them (and to keep its own "no silent default" policy message
    # where it belongs — see `synth_cmd`).
    if data_dir is None:
        raise SynthConfigError("a data-dir is required to write pairs.jsonl")
    resolved_dir = data_dir

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
            say(f"  ... {i}/{total} pairs{spent}", level="status")

    def _on_error(key: str, msg: str) -> None:
        # Surface failures the moment they happen, not just in the exit summary.
        say(f"  !! {key[:12]}: {msg}", level="error")

    result = synth.synthesize_pairs(
        targets,
        resolved_dir,
        generators,
        per_generator=corpus.per_generator,
        assign_seed=corpus.assign_seed,
        dry_run=gen.dry_run,
        extra_tags=list(extra_tags),
        context_dropout=corpus.context_dropout,
        on_progress=None if gen.dry_run else _progress,
        on_error=None if gen.dry_run else _on_error,
        max_workers=max_workers,
        session_turns=session.session_turns,
        session_max_tokens=session.session_max_tokens,
        context_windows=windows or None,
        replicate=corpus.replicate,
        skip_covered=corpus.skip_covered,
        max_transform_sim=corpus.max_transform_sim,
        max_length_ratio=corpus.max_length_ratio,
        drop_invented_anchors=corpus.drop_invented_anchors,
        gate_retries=corpus.gate_retries,
        gen_retries=corpus.gen_retries,
    )

    _report_result(
        result,
        config.pairs_path(resolved_dir),
        dry_run=gen.dry_run,
        session_turns=session.session_turns,
        emit=say,
        max_transform_sim=corpus.max_transform_sim,
        max_length_ratio=corpus.max_length_ratio,
    )
    if result.errors:
        # `_report_result` has already printed each error and the summary; this
        # only signals the failure upward, carrying the result with it.
        raise SynthGenerationError(result)
    return result
