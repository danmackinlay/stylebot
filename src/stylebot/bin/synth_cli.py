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

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import click

from stylebot import synth

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
    "slop_strategy": (("--slop-strategy",), dict(default=synth.DEFAULT_STRATEGY, show_default=True, help=f"Named slop-prompt flavour ({', '.join(synth.STRATEGIES)}); recorded as meta.slop_strategy and folded into synth_key. With --slop-system-file the name is just a label.")),
    "slop_system_file": (("--slop-system-file",), dict(type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Override the slop system prompt with this file's contents (e.g. an author's own slop catalogue), labelled by --slop-strategy. Keeps blog-specific prompts out of stylebot.")),
    "gpt_model": (("--gpt-model",), dict(default="gpt-4o", show_default=True)),
    "local_model": (("--local-model",), dict(default="", help="Local model id (else $LOCAL_LLM_MODEL).")),
    "max_tokens": (("--max-tokens",), dict(default=synth.DEFAULT_SLOP_MAX_TOKENS, show_default=True, type=int, help="Max completion tokens per slop generation. Raise if a model truncates.")),
    "reasoning_effort": (("--reasoning-effort",), dict(type=click.Choice(["high", "medium", "low", "off"]), default=synth.DEFAULT_REASONING_EFFORT, show_default=True, help="Reasoning effort for the slop generator, recorded as a covariate in meta.gen ('off' disables reasoning). Mapping to each provider is best-effort; the requested effort is recorded regardless of what the provider honors.")),
    "temperature": (("--temperature",), dict(type=float, default=None, help="Sampling temperature, sent to the API and recorded (provider default if unset).")),
    "top_p": (("--top-p", "top_p"), dict(type=float, default=None, help="Nucleus top_p, sent to the API and recorded (provider default if unset).")),
    "per_generator": (("--per-generator",), dict(is_flag=True, help="Emit a pair from EVERY generator per target (n× cost), not round-robin.")),
    "limit": (("--limit",), dict(type=int, default=None, help="Cap the number of target chunks (cost control / smoke runs).")),
    "dry_run": (("--dry-run",), dict(is_flag=True, help="Report planned targets/pairs without calling any API or writing.")),
}


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


def run_synth(
    targets: list[synth.Target],
    *,
    data_dir: Path | Callable[[], Path],
    generator_names: Sequence[str] = (),
    openrouter_models: Sequence[str] = (),
    slop_strategy: str = synth.DEFAULT_STRATEGY,
    slop_system_file: Path | None = None,
    gpt_model: str = "gpt-4o",
    local_model: str = "",
    max_tokens: int = synth.DEFAULT_SLOP_MAX_TOKENS,
    reasoning_effort: str = synth.DEFAULT_REASONING_EFFORT,
    temperature: float | None = None,
    top_p: float | None = None,
    per_generator: bool = False,
    context_dropout: float = 0.0,
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
        from stylebot import report

        click.echo("[inspection] targets only — nothing generated.", err=True)
        # Generation flags alongside --report/--sample are a classic trap: the
        # run exits before any generator fires. Say so, and point at the pair
        # browser (the eval scores report) that DOES show generated slop.
        if generator_names or openrouter_models or slop_strategy != synth.DEFAULT_STRATEGY:
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
        return None

    if generators is None:
        # At least one slop source is required (no silent default — synth spends money
        # and may hit a provider key the operator hasn't configured).
        if not generator_names and not openrouter_models:
            raise click.UsageError(
                "no generators selected: pass --generator {gpt,local} and/or "
                "--openrouter-model MODEL (synth needs at least one slop source)"
            )
        # Resolve the slop prompt once (fail fast on a bad strategy name). For a registry
        # strategy we pass the *name* (system=None) down to the factory so it resolves the
        # registry entry and keeps its version; only a custom --slop-system-file overrides
        # the system text. prompt_id/version computed here drive the dry-run stubs so their
        # synth_key matches a real run's.
        custom_system = slop_system_file.read_text(encoding="utf-8") if slop_system_file else None
        try:
            strategy_label, _slop_system, prompt_version, prompt_id = synth.resolve_strategy(
                slop_strategy, custom_system
            )
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--slop-strategy")

        if dry_run:
            # Name-only stubs — no API clients, no keys needed to vet selection. Carry the
            # covariates that feed synth_key so dry-run resume counts match a real run.
            preset_names = {"gpt": gpt_model, "local": f"local-{local_model or 'local'}"}

            def _stub(name: str) -> synth.Generator:
                return synth.Generator(
                    name=name,
                    strategy=strategy_label,
                    reasoning_effort=reasoning_effort,
                    prompt_id=prompt_id,
                    prompt_version=prompt_version,
                )

            generators = [_stub(preset_names[g]) for g in generator_names]
            generators += [_stub(f"openrouter/{m}") for m in openrouter_models]
        else:
            gen_kw = dict(
                strategy=strategy_label, system=custom_system, max_tokens=max_tokens,
                reasoning_effort=reasoning_effort, temperature=temperature, top_p=top_p,
            )
            try:
                generators = []
                for g in generator_names:
                    if g == "gpt":
                        generators.append(synth.openai_generator(model=gpt_model, **gen_kw))
                    elif g == "local":
                        generators.append(synth.local_generator(model=local_model or None, **gen_kw))
                for m in openrouter_models:
                    generators.append(synth.openrouter_generator(model=m, **gen_kw))
            except RuntimeError as exc:  # missing key, surfaced by config.require_key
                raise click.ClickException(str(exc))

    resolved_dir = data_dir() if callable(data_dir) else data_dir

    def _progress(i: int, total: int) -> None:
        if i == 1 or i % 25 == 0 or i == total:
            click.echo(f"  ... {i}/{total} pairs", err=True)

    result = synth.synthesize_pairs(
        targets,
        resolved_dir,
        generators,
        per_generator=per_generator,
        dry_run=dry_run,
        extra_tags=list(extra_tags),
        context_dropout=context_dropout,
        on_progress=None if dry_run else _progress,
    )

    pairs_path = resolved_dir / "pairs.jsonl"
    if dry_run:
        click.echo(
            f"[dry-run] would write {result.planned - result.skipped_existing} new pair(s) "
            f"({result.skipped_existing} already present) to {pairs_path}"
        )
        for name, count in sorted(result.per_generator.items()):
            click.echo(f"  {name}: {count}")
        return result

    click.echo(
        f"wrote {result.written} pair(s) "
        f"({result.skipped_existing} already present, skipped) -> {pairs_path}"
    )
    for name, count in sorted(result.per_generator.items()):
        click.echo(f"  {name}: {count}")
    if result.errors:
        click.echo(f"{len(result.errors)} generation error(s):", err=True)
        for key, msg in result.errors[:10]:
            click.echo(f"  {key}: {msg}", err=True)
        raise SystemExit(1)
    return result
