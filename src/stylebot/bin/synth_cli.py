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
`--data-dir` resolution policy (the caller checks for the inspection modes
first and passes None when nothing will be written, so its own "no silent
default" message stays in its own command), and provenance tags.

**One list, not three.** The shared flags are described *once*, in
`_OPTION_SPECS`; everything downstream is derived from it instead of restating
it. Each spec name is a field of exactly one frozen **option bundle**
(`ChunkOpts`, `InspectOpts`, `GenOpts`, `SessionOpts`, `SubstrateOpts`,
`CorpusOpts`) — grouped along the seams of the surface, i.e. by the consumer
each group feeds — and an import-time check refuses to load the module if a
spec name has no bundle field. That is the guard the surface used to lack:
`CHUNK_PARAM_NAMES` was a hand-written tuple and `run_synth` restated 42
keywords by hand, so a flag could be added to the table and quietly reach
nothing. `CHUNK_PARAM_NAMES` is now `fields(ChunkOpts)` and `run_synth` takes
bundles.

`run_synth` still accepts every flag as a **loose keyword** — external
wrappers (livingthing's `dan-style synth`) splat the click callback's `**kw`
straight in, and pass individual overrides by param name — but the loose names
are now *routed* into the bundles by the same registry, and an unrecognised
name raises `TypeError` instead of being swallowed. Typed callers can pass the
bundles directly (`gen=GenOpts(...)`); loose keywords override bundle fields.

`ChunkOpts` is the CLI-side twin of `targets.ChunkPolicy` (the same hygiene
knobs, plus `--sort`, which `iter_targets` takes as a `sort_key` callable).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
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
    "cross_heading": (("--cross-heading/--no-cross-heading",), dict(default=False, show_default=True, help="Pack across section boundaries with headings as chunk content (needs --merge; overrides --heading-context). This, not --merge-max-chars, is what buys chunk length.")),
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
    "max_transform_sim": (("--max-transform-sim",), dict(type=float, default=None, help="Identity gate: drop (don't write) any generated pair whose transform_sim exceeds this — the model returned (nearly) the input, which would teach the styler to copy. Dropped pairs are counted loudly (skipped_degenerate) and their cells retry next run. Unset = record everything (the covariate is always stored).")),
    "max_length_ratio": (("--max-length-ratio",), dict(type=float, default=None, help="Inflation gate: drop any generated pair whose slop/target length ratio exceeds this — a manically inflating model (qwen3-32b ran 4-7x) teaches manic deletion. Counted as skipped_inflated; cells retry next run. Unset = record everything.")),
    "context_window": (("--context-window",), dict(type=int, default=None, help="Context window (tokens) for gpt/local preset generators when sweeping sessions — OpenRouter models resolve theirs from the models registry automatically.")),
    # -- substrate selection: WHICH population this run measures (before --limit) --
    "splits_path": (("--splits", "splits_path"), dict(type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="splits.json of by-post roles (see ai-style make-splits). Required by --role.")),
    "role": (("--role",), dict(type=click.Choice(SPLIT_ROLES), default=None, help="Keep only chunks whose post holds this role. Measurement runs want 'styler': disjoint from the detector's training pool, so the voice classifier stays an unbiased instrument.")),
    "sample_frac": (("--sample-frac",), dict(type=float, default=None, help="Keep this fraction of chunks by content hash. Growth-stable: a later run on a larger corpus is a SUPERSET of this one, so the two stay comparable (--limit is not, its first N shifts as posts are added).")),
    "sample_salt": (("--sample-salt",), dict(default="", show_default=True, help="Re-draw an independent subsample of the same size (a fresh replicate of the design).")),
    "limit": (("--limit",), dict(type=int, default=None, help="Cap the number of target chunks (cost control / smoke runs). A cap on the substrate, not a definition of it — see --sample-frac.")),
    "dry_run": (("--dry-run",), dict(is_flag=True, help="Report planned targets/pairs without calling any API or writing.")),
}


# ---------------------------------------------------------------------------
# Option bundles: the surface grouped along its seams, one frozen dataclass
# per consumer. These replace `run_synth`'s 42 hand-restated keywords — the
# names now live in exactly two places (the spec table, which owns the flag
# spelling/help/default, and one bundle field, which owns the plumbing), and
# `_FIELD_OWNER` below welds the two together at import time.
#
# Defaults MUST equal the spec defaults: `run_synth` is a public library entry
# point reachable without click, so a default that drifts from its flag
# silently changes what a paid run generates. `tests/test_synth_cli.py`
# asserts the equality mechanically.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkOpts:
    """Chunking / hygiene — feeds `iter_targets`, i.e. WHAT counts as a target.

    Consumed on the *selection* side, before `run_synth` ever sees the targets:
    the CLI splats these into `synth.iter_targets` via `pop_chunk_kwargs`. The
    twin of `targets.ChunkPolicy` with `sort_name` added (the flag is a choice
    string; `iter_targets` wants a `sort_key` callable).
    """

    min_chars: int = synth.MIN_CHUNK_CHARS
    max_chars: int = synth.MAX_CHUNK_CHARS  # 0 -> None (uncapped) at translation
    prose_only: bool = True
    ignore_markers: Sequence[str] = ()
    stop_at_headers: Sequence[str] = ()
    drop_link_dumps: bool = True
    drop_list_items: bool = False
    merge: bool = False
    merge_max_chars: int = synth.MERGE_MAX_CHUNK_CHARS
    heading_context: str = "none"
    cross_heading: bool = False
    sort_name: str = "none"

    def iter_targets_kwargs(self) -> dict:
        """This bundle shaped for `iter_targets` (the typed path; the CLI's
        `**kw` path goes through `pop_chunk_kwargs`, which shares the rules)."""
        return pop_chunk_kwargs({f.name: getattr(self, f.name) for f in fields(self)})


@dataclass(frozen=True)
class InspectOpts:
    """Read-only inspection — `--sample` / `--report`, handled by
    `_inspect_targets`, which exits before any generation (no data-dir, no
    keys). `report_title` has no flag: it is a caller-supplied label."""

    report_path: Path | None = None
    report_max_rows: int = 2000
    sample_n: int | None = None
    report_title: str | None = None


@dataclass(frozen=True)
class GenOpts:
    """Rotation / generation — which arms exist and how each request is made.

    Feeds `iter_arms` (the models x strategies x efforts cross product, whose
    *enumeration order* decides which arm owns which target) and
    `build_generators` (per-request knobs). `dry_run` lives here because it is
    a generator-construction switch: stubs instead of API clients. It is read
    downstream too (concurrency, windows, reporting).
    """

    generator_names: Sequence[str] = ()
    openrouter_models: Sequence[str] = ()
    provider_sort: str = "throughput"
    sticky_provider: bool = True
    prompt_cache: bool = True
    slop_strategies: Sequence[str] = (synth.DEFAULT_STRATEGY,)
    slop_system_file: Path | None = None
    gpt_model: str = "gpt-4o"
    local_model: str = ""
    max_tokens: int = synth.DEFAULT_SLOP_MAX_TOKENS
    timeout: float | None = synth.DEFAULT_GEN_TIMEOUT
    reasoning_efforts: Sequence[str] = (synth.DEFAULT_REASONING_EFFORT,)
    capture_reasoning: bool = False
    temperature: float | None = None
    top_p: float | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class SessionOpts:
    """Live-session shape and the concurrency that runs them.

    `session_budgets` has no flag — it is the policy hook a wrapper uses to
    hand in a per-model budget map (the blog pins one), applied on top of
    `session_max_tokens`.
    """

    session_turns: int = 1
    session_max_tokens: int = synth.DEFAULT_SESSION_MAX_TOKENS
    context_window: int | None = None
    max_workers: int = 0  # 0 = auto, decided from the generator mix
    session_budgets: Mapping[str, int] | None = None


@dataclass(frozen=True)
class SubstrateOpts:
    """WHICH population this run measures — applied before `--limit`.

    `role`/`sample_frac` define the substrate (`select_targets`); `limit` is
    only a cost cap on it, which is why it is applied afterwards.
    """

    splits_path: Path | None = None
    role: str | None = None
    sample_frac: float | None = None
    sample_salt: str = ""
    limit: int | None = None


@dataclass(frozen=True)
class CorpusOpts:
    """What gets written, under which design cell — feeds `synthesize_pairs`.

    Two clusters that both bear on corpus identity: the arm-assignment design
    (`per_generator` / `assign_seed` / `replicate` / `skip_covered`, which
    decide *which cells exist* and therefore what resume regenerates) and the
    write gates (`context_dropout` and the two hygiene thresholds, which decide
    which generated pairs survive to the file).
    """

    per_generator: bool = False
    assign_seed: str = ""
    context_dropout: float = 0.0
    replicate: str = ""
    skip_covered: bool = False
    max_transform_sim: float | None = None
    max_length_ratio: float | None = None


# The bundle registry: field name -> owning bundle. This is the weld between
# the spec table and the plumbing, and the reason the two lists cannot drift.
_BUNDLES: tuple[type, ...] = (
    ChunkOpts, InspectOpts, GenOpts, SessionOpts, SubstrateOpts, CorpusOpts,
)

_FIELD_OWNER: dict[str, type] = {}
for _bundle_cls in _BUNDLES:
    for _f in fields(_bundle_cls):
        if _f.name in _FIELD_OWNER:
            raise RuntimeError(
                f"synth option {_f.name!r} claimed by two bundles "
                f"({_FIELD_OWNER[_f.name].__name__}, {_bundle_cls.__name__}) — "
                "routing would be ambiguous"
            )
        _FIELD_OWNER[_f.name] = _bundle_cls

# THE drift guard. A flag added to `_OPTION_SPECS` with no bundle field would
# reach no consumer: the click callback would collect it into **kw and
# `run_synth` would have nowhere to route it. Fail loudly at import, not on the
# operator's paid run.
_ORPHAN_SPECS = sorted(set(_OPTION_SPECS) - set(_FIELD_OWNER))
if _ORPHAN_SPECS:
    raise RuntimeError(
        f"synth option(s) {_ORPHAN_SPECS} are declared in _OPTION_SPECS but are "
        "not a field of any option bundle, so nothing consumes them. Add each to "
        "the bundle whose consumer should read it (see _BUNDLES)."
    )


# `run_synth`'s own parameters must never collide with a routed flag name, or
# the loose keyword would bind to the parameter instead of reaching its bundle.
_RUN_SYNTH_OWN_PARAMS = frozenset({
    "targets", "data_dir", "gen", "session", "substrate", "inspection", "corpus",
    "count_note", "extra_tags", "generators",
})
_COLLIDING_PARAMS = sorted(_RUN_SYNTH_OWN_PARAMS & set(_FIELD_OWNER))
if _COLLIDING_PARAMS:
    raise RuntimeError(
        f"synth option(s) {_COLLIDING_PARAMS} collide with run_synth's own "
        "parameter names and could never be routed to a bundle — rename one side."
    )


def split_bundle_kwargs(flat: dict) -> dict[type, dict]:
    """Route loose param names to their owning bundle (mutating `flat`).

    Whatever is left in `flat` afterwards belongs to no bundle — the caller
    decides whether that is an error (`run_synth`: yes) or its own param.
    """
    routed: dict[type, dict] = {}
    for name in [n for n in flat if n in _FIELD_OWNER]:
        routed.setdefault(_FIELD_OWNER[name], {})[name] = flat.pop(name)
    return routed


def _bundle(cls, given, routed: dict[type, dict]):
    """One bundle instance: the caller's object (if any) with loose overrides
    applied on top, else a fresh one from the loose values / defaults."""
    over = routed.pop(cls, {})
    if given is None:
        return cls(**over)
    return replace(given, **over) if over else given


# Shared params consumed by iter_targets (selection-side); the rest of the
# shared set feeds run_synth. Lets a wrapper collect the whole shared surface
# as **kw and split it without restating ~30 parameter names. DERIVED from
# ChunkOpts — this tuple used to be hand-written next to the spec table, which
# is exactly how it could fall behind it.
CHUNK_PARAM_NAMES = tuple(f.name for f in fields(ChunkOpts))


def pop_chunk_kwargs(kw: dict) -> dict:
    """Pop the chunking/hygiene params out of a **kw catch-all, shaped for
    iter_targets (--max-chars 0 -> no cap; --sort name -> sort_key callable).

    Only the keys actually PRESENT are returned: a wrapper that pins some chunk
    knobs as policy constants (`synth_options(exclude=...)`) passes those to
    `iter_targets` itself, and injecting defaults for them here would collide.
    """
    out = {k: kw.pop(k) for k in CHUNK_PARAM_NAMES if k in kw}
    if "max_chars" in out:
        out["max_chars"] = out["max_chars"] or None
    if "sort_name" in out:
        out["sort_key"] = sort_key(out.pop("sort_name"))
    return out


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
    opts: InspectOpts,
    *,
    data_dir: Path | None,
    generation_flags_set: bool,
) -> None:
    """The read-only --sample/--report modes: print/write and return, before
    any generation — no API keys, and `data_dir` may be None (the caller skips
    resolving it when nothing will be written)."""
    from stylebot import report

    click.echo("[inspection] targets only — nothing generated.", err=True)
    # Generation flags alongside --report/--sample are a classic trap: the
    # run exits before any generator fires. Say so, and point at the pair
    # browser (the eval scores report) that DOES show generated slop.
    if generation_flags_set:
        pairs_hint = f"{data_dir}/pairs.jsonl" if data_dir is not None else "<data-dir>/pairs.jsonl"
        click.echo(
            "[inspection] generation flags ignored; to generate then browse pairs: "
            "re-run without --report/--sample, then: "
            f"ai-style eval --pairs {pairs_hint} --report FILE.html --facet-by generator",
            err=True,
        )
    if opts.sample_n is not None:
        click.echo(report.format_sample(report.sample_targets(targets, opts.sample_n)))
    if opts.report_path is not None:
        title_kw = {"title": opts.report_title} if opts.report_title else {}
        written = report.render_targets_report(
            targets, opts.report_path,
            max_rows=(None if opts.report_max_rows == 0 else opts.report_max_rows),
            **title_kw,
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
    max_length_ratio: float | None = None,
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
            f"  {result.skipped_degenerate} identity pair(s) DROPPED "
            f"(transform_sim > {max_transform_sim}) — paid for, not written; "
            "a persistently degenerate model should leave the roster",
            fg="yellow",
        )
    if result.skipped_inflated:
        click.secho(
            f"  {result.skipped_inflated} inflated pair(s) DROPPED "
            f"(length ratio > {max_length_ratio}) — paid for, not written; "
            "a persistently inflating model should leave the roster",
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
        - result.skipped_degenerate - result.skipped_inflated - len(result.errors)
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
    data_dir: Path | None,
    gen: GenOpts | None = None,
    session: SessionOpts | None = None,
    substrate: SubstrateOpts | None = None,
    inspection: InspectOpts | None = None,
    corpus: CorpusOpts | None = None,
    count_note: str = "",
    extra_tags: Sequence[str] = (),
    generators: Sequence[synth.Generator] | None = None,
    **flat,
) -> synth.SynthResult | None:
    """Everything after target selection, shared by all synth-style commands.

    The knobs arrive as **option bundles** (see the module docstring). Two
    calling styles, both first-class:

    - typed: ``run_synth(targets, data_dir=d, gen=GenOpts(max_tokens=900))``;
    - loose: every flag is still accepted by its own param name, e.g.
      ``run_synth(targets, data_dir=d, max_tokens=900, session_turns=4)``, or
      splatted wholesale from a click callback's ``**kw`` — which is how
      `ai-style synth` and the blog's `dan-style synth` wrapper call it.
      Loose values override the matching field of a bundle passed alongside.

    An unrecognised keyword is a `TypeError`, not a silent drop.

    `data_dir` may be None **only** for the inspection modes, which write
    nothing and need no corpus location or API keys: the caller decides
    whether to resolve it (and owns its own "no silent default" message)
    before calling, having seen `--sample`/`--report` first. Library callers
    may inject ready-made `generators`, bypassing the preset/OpenRouter
    construction. Returns the SynthResult, or None when an inspection mode
    short-circuited. Exits 1 on generation errors.
    """
    routed = split_bundle_kwargs(flat)
    if flat:  # whatever survived routing matches no flag and no bundle field
        raise TypeError(
            f"run_synth() got unexpected keyword argument(s): {sorted(flat)}. "
            "Every synth flag routes to an option bundle; a name that does not "
            "is a typo or a flag that was never wired up."
        )
    gen = _bundle(GenOpts, gen, routed)
    session = _bundle(SessionOpts, session, routed)
    substrate = _bundle(SubstrateOpts, substrate, routed)
    inspection = _bundle(InspectOpts, inspection, routed)
    corpus = _bundle(CorpusOpts, corpus, routed)
    # ChunkOpts is consumed on the selection side (iter_targets), so its names
    # are already gone by the time the caller reaches here; a stray one means
    # the caller forgot `pop_chunk_kwargs`.
    if routed:
        raise TypeError(
            f"run_synth() got selection-side keyword argument(s): "
            f"{sorted(n for b in routed.values() for n in b)} — pass them through "
            "pop_chunk_kwargs() to iter_targets instead."
        )

    # Substrate selection runs BEFORE --limit: role and sample-frac define
    # *which* population this run measures, --limit is only a cost cap on it.
    if substrate.role is not None or substrate.sample_frac is not None:
        from stylebot import splits as splits_mod
        from stylebot.targets import select_targets

        before = len(targets)
        loaded = splits_mod.load_splits(substrate.splits_path) if substrate.role is not None else None
        if substrate.role is not None and substrate.splits_path is None:
            raise click.UsageError("--role requires --splits")
        targets = select_targets(
            targets, splits=loaded, role=substrate.role,
            sample_frac=substrate.sample_frac, sample_salt=substrate.sample_salt,
        )
        detail = ", ".join(
            p for p in (
                f"role={substrate.role}" if substrate.role else "",
                f"sample-frac={substrate.sample_frac}" if substrate.sample_frac else "",
            ) if p
        )
        click.echo(f"substrate: {before} -> {len(targets)} target chunk(s) ({detail})")

    if substrate.limit is not None:
        targets = targets[:substrate.limit]

    n_sources = len({t.source for t in targets})
    suffix = f" {count_note}" if count_note else ""
    click.echo(f"{len(targets)} target chunk(s) from {n_sources} source(s){suffix}")
    if not targets:
        click.echo("nothing to synthesize", err=True)
        return None

    # Read-only inspection modes — print/write and exit before any generation,
    # so no --data-dir or API keys are needed to vet the targets.
    if inspection.sample_n is not None or inspection.report_path is not None:
        _inspect_targets(
            targets,
            inspection,
            data_dir=data_dir,
            generation_flags_set=bool(
                gen.generator_names or gen.openrouter_models
                or tuple(gen.slop_strategies) != (synth.DEFAULT_STRATEGY,)
            ),
        )
        return None

    if generators is None:
        # At least one slop source is required (no silent default — synth spends money
        # and may hit a provider key the operator hasn't configured).
        if not gen.generator_names and not gen.openrouter_models:
            raise click.UsageError(
                "no generators selected: pass --generator {gpt,local} and/or "
                "--openrouter-model MODEL (synth needs at least one slop source)"
            )
        # Resolve every requested slop prompt up front (fail fast on a bad name;
        # dedupe repeats, order-preserving). For a registry strategy the *name*
        # (system=None) goes down to the factory so it resolves the registry entry
        # and keeps its version; a custom --slop-system-file overrides the system
        # text and is single-strategy by construction (one file = one prompt).
        custom_system = gen.slop_system_file.read_text(encoding="utf-8") if gen.slop_system_file else None
        strategy_names = list(dict.fromkeys(gen.slop_strategies)) or [synth.DEFAULT_STRATEGY]
        if custom_system is not None and len(strategy_names) > 1:
            raise click.UsageError(
                "--slop-system-file provides ONE custom prompt; pass exactly one "
                "--slop-strategy label for it (registry strategies can rotate, a file cannot)"
            )
        try:
            resolved_strategies = [synth.resolve_strategy(s, custom_system) for s in strategy_names]
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--slop-strategy")
        efforts = list(dict.fromkeys(gen.reasoning_efforts)) or [synth.DEFAULT_REASONING_EFFORT]

        # The rotation is the models x strategies x efforts cross product:
        # round-robin assignment then spreads targets across every combination,
        # so one run diversifies prompts and reasoning depth the same way it
        # diversifies models (same cost — still one generation per target, just
        # a finer rotation).
        arms = iter_arms(
            resolved_strategies, efforts, gen.generator_names, gen.openrouter_models,
            gpt_model=gen.gpt_model, local_model=gen.local_model,
        )
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
            raise click.ClickException(str(exc))

    # Per-model session budgets (policy hook, e.g. a blog's model->budget map).
    if session.session_budgets:
        for generator in generators:
            budget = session.session_budgets.get(generator.name) or session.session_budgets.get(
                generator.name.removeprefix("openrouter/")
            )
            if budget:
                generator.session_budget = budget

    # auto concurrency: OpenRouter multiplexes fine; a local/preset endpoint
    # (or injected generators we know nothing about) gets sequential.
    max_workers = session.max_workers
    if max_workers <= 0:
        openrouter_only = bool(gen.openrouter_models) and not gen.generator_names
        max_workers = 16 if (openrouter_only and not gen.dry_run) else 1

    windows = _resolve_windows(
        generators, session_turns=session.session_turns, dry_run=gen.dry_run,
        context_window=session.context_window,
    )

    # Everything below writes: a data-dir is mandatory from here on. The
    # inspection modes returned above, which is why the caller is free to hand
    # in None for them (and to keep its own "no silent default" policy message
    # where it belongs — see `synth_cmd`).
    if callable(data_dir):  # transitional: the old deferred-resolution shape
        raise TypeError(
            "run_synth(data_dir=...) no longer takes a callable — resolve the path "
            "eagerly, and pass None only for --sample/--report inspection runs."
        )
    if data_dir is None:
        raise click.UsageError("a data-dir is required to write pairs.jsonl")
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
            click.echo(f"  ... {i}/{total} pairs{spent}", err=True)

    def _on_error(key: str, msg: str) -> None:
        # Surface failures the moment they happen, not just in the exit summary.
        click.echo(f"  !! {key[:12]}: {msg}", err=True)

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
    )

    _report_result(
        result, resolved_dir / "pairs.jsonl", dry_run=gen.dry_run,
        session_turns=session.session_turns,
        max_transform_sim=corpus.max_transform_sim,
        max_length_ratio=corpus.max_length_ratio,
    )
    return result
