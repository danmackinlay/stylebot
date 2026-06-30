"""ai-style: the one-entry-point CLI (subcommands), thin over the libraries.

Per OVERVIEW "Interfaces": one command, subcommands `synth | split | train |
eval` — not a scatter of loose scripts. Each subcommand is a thin `click`
wrapper that parses flags and calls a typed library function. `synth` (Phase 2)
and `eval` (the offline scorer, `stylebot.eval`) are built; `split` / `train`
land with their phases.

The shipped `ai-style-log` (Phase 1) stays its own console script — it is
daily-used and its CLI predates this group.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import click

from stylebot import config, synth
from stylebot.lib import is_human_authored

# Direct-API slop-generator presets selectable on the CLI (each needs that
# provider's own key). OpenRouter models are selected separately via
# --openrouter-model (one key, many upstream models). Library callers inject
# their own `synth.Generator`s.
_PRESETS = ("gpt", "local")


def _sort_key(name: str):
    if name == "none":
        return None
    if name == "length":
        return lambda t: len(t.text)
    if name == "source":
        return lambda t: (t.source, t.chunk_index)
    raise click.BadParameter(f"unknown sort key {name!r}")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """stylebot tooling — synthetic pairs, training, eval (subcommands)."""


@main.command("synth")
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--blog-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Read source: walk this blog for posts (mutually exclusive with FILES).",
)
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Read/write state: where pairs.jsonl is appended. Required (no default) "
    "unless $STYLEBOT_DATA_DIR is set — never silently grow the wrong corpus.",
)
@click.option("--field", default="automation", show_default=True, help="Frontmatter field the default selector reads.")
@click.option("--max-level", default=0, show_default=True, type=int, help="Default selector keeps posts with field <= this.")
@click.option("--all", "select_all", is_flag=True, help="Disable selection: take every post (ignore the selector).")
@click.option("--glob", default="**/*.qmd", show_default=True, help="Glob for the blog-root walk.")
@click.option("--min-chars", default=synth.MIN_CHUNK_CHARS, show_default=True, type=int, help="Skip prose chunks shorter than this.")
@click.option("--max-chars", default=synth.MAX_CHUNK_CHARS, show_default=True, type=int, help="Drop prose chunks longer than this (0 disables the cap).")
@click.option("--prose-only/--no-prose-only", default=True, show_default=True, help="Drop code/math/:::divs/blockquotes before chunking.")
@click.option("--ignore-marker", "ignore_markers", multiple=True, help="Drop any chunk containing this literal (repeatable), e.g. a stub marker.")
@click.option("--stop-at-header", "stop_at_headers", multiple=True, help="Truncate each post body at this section header (repeatable), e.g. '## Incoming'.")
@click.option("--drop-link-dumps/--keep-link-dumps", default=True, show_default=True, help="Drop chunks that are mostly markdown links (little authored prose).")
@click.option("--drop-list-items/--keep-list-items", default=False, show_default=True, help="Drop chunks that are entirely markdown list items.")
@click.option("--merge/--no-merge", default=False, show_default=True, help="Pack consecutive prose paragraphs within a section into multi-paragraph passages.")
@click.option("--merge-max-chars", default=synth.MERGE_MAX_CHUNK_CHARS, show_default=True, type=int, help="Soft char budget per packed passage (merge mode).")
@click.option("--heading-context", type=click.Choice(["none", "immediate"]), default="none", show_default=True, help="Prepend the section heading (verbatim) to both sides of each pair as fixed context.")
@click.option("--context-dropout", default=0.0, show_default=True, type=float, help="Fraction of pairs to keep heading-less (deterministic) so the styler doesn't require a heading.")
@click.option("--sort", "sort_name", type=click.Choice(["none", "length", "source"]), default="none", show_default=True)
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a self-contained HTML report of the selected targets and exit (no generation).")
@click.option("--report-max-rows", default=2000, show_default=True, type=int, help="Cap table rows in the HTML report (0 = all).")
@click.option("--sample", "sample_n", type=int, default=None, help="Print N random targets to stdout and exit (no generation).")
@click.option(
    "--generator",
    "generator_names",
    type=click.Choice(_PRESETS),
    multiple=True,
    default=(),
    show_default=True,
    help="Direct-API slop generators to rotate across (repeatable; each needs that provider's key). "
    "Combine with or replace by --openrouter-model.",
)
@click.option(
    "--openrouter-model",
    "openrouter_models",
    multiple=True,
    help="OpenRouter model id to generate slop with (repeatable), e.g. anthropic/claude-opus-4-8. "
    "One key ($OPENROUTER_API_KEY) reaches many upstream models — ideal for multi-source rotation.",
)
@click.option(
    "--slop-strategy",
    default=synth.DEFAULT_STRATEGY,
    show_default=True,
    help=f"Named slop-prompt flavour ({', '.join(synth.STRATEGIES)}); recorded as meta.slop_strategy "
    "and folded into synth_key. With --slop-system-file the name is just a label.",
)
@click.option(
    "--slop-system-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Override the slop system prompt with this file's contents (e.g. an author's own slop "
    "catalogue), labelled by --slop-strategy. Keeps blog-specific prompts out of stylebot.",
)
@click.option("--gpt-model", default="gpt-4o", show_default=True)
@click.option("--local-model", default="", help="Local model id (else $LOCAL_LLM_MODEL).")
@click.option(
    "--max-tokens",
    default=synth.DEFAULT_SLOP_MAX_TOKENS,
    show_default=True,
    type=int,
    help="Max completion tokens per slop generation. Raise if a model truncates.",
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(["high", "medium", "low", "off"]),
    default=synth.DEFAULT_REASONING_EFFORT,
    show_default=True,
    help="Reasoning effort for the slop generator, recorded as a covariate in meta.gen "
    "('off' disables reasoning). Mapping to each provider is best-effort; the requested "
    "effort is recorded regardless of what the provider honors.",
)
@click.option("--temperature", type=float, default=None, help="Sampling temperature, sent to the API and recorded (provider default if unset).")
@click.option("--top-p", "top_p", type=float, default=None, help="Nucleus top_p, sent to the API and recorded (provider default if unset).")
@click.option("--per-generator", is_flag=True, help="Emit a pair from EVERY generator per target (n× cost), not round-robin.")
@click.option("--limit", type=int, default=None, help="Cap the number of target chunks (cost control / smoke runs).")
@click.option("--tag", "tags", multiple=True, help="Extra provenance tag(s) added to meta.tags.")
@click.option("--dry-run", is_flag=True, help="Report planned targets/pairs without calling any API or writing.")
def synth_cmd(
    files: tuple[Path, ...],
    blog_root: Path | None,
    data_dir: Path | None,
    field: str,
    max_level: int,
    select_all: bool,
    glob: str,
    min_chars: int,
    max_chars: int,
    prose_only: bool,
    ignore_markers: tuple[str, ...],
    stop_at_headers: tuple[str, ...],
    drop_link_dumps: bool,
    drop_list_items: bool,
    merge: bool,
    merge_max_chars: int,
    heading_context: str,
    context_dropout: float,
    sort_name: str,
    report_path: Path | None,
    report_max_rows: int,
    sample_n: int | None,
    generator_names: tuple[str, ...],
    openrouter_models: tuple[str, ...],
    slop_strategy: str,
    slop_system_file: Path | None,
    gpt_model: str,
    local_model: str,
    max_tokens: int,
    reasoning_effort: str,
    temperature: float | None,
    top_p: float | None,
    per_generator: bool,
    limit: int | None,
    tags: tuple[str, ...],
    dry_run: bool,
) -> None:
    """Manufacture synthetic (slop -> Dan) pairs from Dan's own prose.

    Provide targets one of two ways: a pre-selected FILES list, or --blog-root
    plus a selector (default: is_human_authored, i.e. automation: 0). Writes the
    Phase-1 pairs.jsonl schema with meta.synthetic / meta.generator. Idempotent
    and resumable — re-running never duplicates.
    """
    if bool(files) == bool(blog_root):
        raise click.UsageError("provide either FILES (pre-selected) or --blog-root, not both/neither")

    # Selector: the bundled is_human_authored unless --all disables selection.
    if select_all:
        selector = lambda meta: True  # noqa: E731 — trivial inline policy
    else:
        selector = partial(is_human_authored, field=field, max_level=max_level)

    targets = synth.iter_targets(
        files=list(files) or None,
        blog_root=blog_root,
        selector=selector,
        glob=glob,
        min_chars=min_chars,
        max_chars=max_chars or None,
        prose_only=prose_only,
        ignore_markers=ignore_markers,
        stop_at_headers=stop_at_headers,
        drop_link_dumps=drop_link_dumps,
        drop_list_items=drop_list_items,
        merge=merge,
        merge_max_chars=merge_max_chars,
        heading_context=heading_context,
        sort_key=_sort_key(sort_name),
    )
    if limit is not None:
        targets = targets[:limit]

    n_sources = len({t.source for t in targets})
    click.echo(f"{len(targets)} target chunk(s) from {n_sources} source(s)")
    if not targets:
        click.echo("nothing to synthesize", err=True)
        return

    # Read-only inspection modes — print/write and exit before any generation,
    # so no --data-dir or API keys are needed to vet the targets.
    if sample_n is not None or report_path is not None:
        from stylebot import report

        if sample_n is not None:
            click.echo(report.format_sample(report.sample_targets(targets, sample_n)))
        if report_path is not None:
            written = report.render_targets_report(
                targets, report_path, max_rows=(None if report_max_rows == 0 else report_max_rows)
            )
            click.echo(f"wrote report -> {written}")
        return

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

    # Resolve the corpus location: explicit flag > $STYLEBOT_DATA_DIR. Refuse the
    # bare cwd default — synth costs money and grows the corpus; be explicit.
    if data_dir is None and not config.get_key("STYLEBOT_DATA_DIR"):
        raise click.UsageError(
            "no --data-dir and $STYLEBOT_DATA_DIR unset; pass --data-dir explicitly "
            "(synth appends to the corpus and costs API spend — never the silent default)"
        )
    resolved_dir = config.resolve_data_dir(data_dir)

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
        try:
            generators = []
            for g in generator_names:
                if g == "gpt":
                    generators.append(
                        synth.openai_generator(
                            model=gpt_model, strategy=strategy_label, system=custom_system,
                            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                            temperature=temperature, top_p=top_p,
                        )
                    )
                elif g == "local":
                    generators.append(
                        synth.local_generator(
                            model=local_model or None, strategy=strategy_label, system=custom_system,
                            max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                            temperature=temperature, top_p=top_p,
                        )
                    )
            for m in openrouter_models:
                generators.append(
                    synth.openrouter_generator(
                        model=m, strategy=strategy_label, system=custom_system,
                        max_tokens=max_tokens, reasoning_effort=reasoning_effort,
                        temperature=temperature, top_p=top_p,
                    )
                )
        except RuntimeError as exc:  # missing key, surfaced by config.require_key
            raise click.ClickException(str(exc))

    def _progress(i: int, total: int) -> None:
        if i == 1 or i % 25 == 0 or i == total:
            click.echo(f"  ... {i}/{total} pairs", err=True)

    result = synth.synthesize_pairs(
        targets,
        resolved_dir,
        generators,
        per_generator=per_generator,
        dry_run=dry_run,
        extra_tags=tags,
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
        return

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
        sys.exit(1)


@main.command("eval")
@click.option("--pairs", "pairs_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="The pairs.jsonl corpus to score (slop + Dan sides of every pair).")
@click.option("--field", "fields", type=click.Choice(["slop", "target"]), multiple=True, default=(), help="Which side(s) to score (repeatable; default both). slop=messages[1], target=Dan/messages[2].")
@click.option("--judge/--no-judge", default=False, show_default=True, help="Run the LLM judge (needs $OPENROUTER_API_KEY); off = Vale + null detector, no spend.")
@click.option("--judge-model", default="anthropic/claude-opus-4-8", show_default=True, help="OpenRouter model id for the judge.")
@click.option("--detector-model", "detector_model", type=click.Path(exists=True, file_okay=False, path_type=Path), default=None, help="A trained voice-classifier artifact dir (head.json + meta.json) to score detector.score; needs the pinned embedder (sentence-transformers) installed.")
@click.option("--vale-config", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Vale config (.vale.ini); omitted = Vale's own discovery (or Vale absent -> skipped).")
@click.option("--max-workers", default=8, show_default=True, type=int, help="Concurrent scoring workers (judge/Vale are IO-bound).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="scores.jsonl to append (default: <pairs-dir>/scores.jsonl). Resumable — scored ids are skipped.")
@click.option("--summary", "summary_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Also write the aggregate summary JSON here.")
@click.option("--by", "by", default=None, help="Facet the summary by a meta key, e.g. slop_strategy or generator.")
@click.option("--limit", type=int, default=None, help="Cap pairs scored (smoke / cost control).")
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a self-contained HTML scores report (slop↔Dan + judge scores, sortable, faceted by strategy).")
@click.option("--report-max-rows", default=2000, show_default=True, type=int, help="Cap rows in the HTML report (0 = all).")
@click.option("--facet-by", "facet_by", default="slop_strategy", show_default=True, help="Group the HTML report headline by this meta covariate (e.g. reasoning_effort, prompt_id, generator).")
@click.option("--sample", "sample_n", type=int, default=None, help="Print N random scored pairs (slop vs Dan + scores) to stdout.")
def eval_cmd(
    pairs_path: Path,
    fields: tuple[str, ...],
    judge: bool,
    judge_model: str,
    detector_model: Path | None,
    vale_config: Path | None,
    max_workers: int,
    out_path: Path | None,
    summary_path: Path | None,
    by: str | None,
    limit: int | None,
    report_path: Path | None,
    report_max_rows: int,
    facet_by: str,
    sample_n: int | None,
) -> None:
    """Score a pairs.jsonl corpus across the eval signals (offline, batched).

    Reads the corpus, scores the slop and Dan sides of each pair, and appends one
    id-keyed record per pair to scores.jsonl (joinable back to the corpus, and
    resumable — re-running skips already-scored ids). Runs keyless by default
    (Vale + null detector); add --judge to score voice via OpenRouter. The
    summary aggregates per field, and per --by facet (e.g. slop_strategy) so you
    can compare strategies with a number. A Phase-4 styler run scores the same way
    once it emits an output-bearing JSONL (add an "output" field then).
    """
    import json

    from stylebot import eval as ev

    out = out_path or (pairs_path.parent / "scores.jsonl")
    score_fields = fields or ("slop", "target")

    try:
        judge_fn = ev.openrouter_judge(model=judge_model) if judge else None
    except RuntimeError as exc:  # missing key, surfaced by config.require_key
        raise click.ClickException(str(exc))

    # A trained voice-classifier artifact, if given, supplies a real detector
    # (P(slop)); else the null detector. The embedder is rebuilt lazily from the
    # artifact's pinned meta.embed_model — heavy imports stay inside classify.
    detector_fn = ev.null_detector
    if detector_model is not None:
        from stylebot import classify

        try:
            detector_fn = classify.sklearn_detector(detector_model)
        except ImportError as exc:
            raise click.ClickException(
                f"--detector-model needs the embedder installed (sentence-transformers): {exc}"
            )

    def _progress(i: int, total: int) -> None:
        if i == 1 or i % 25 == 0 or i == total:
            click.echo(f"  ... {i}/{total} scored", err=True)

    result = ev.score_pairs_file(
        pairs_path,
        out,
        fields=score_fields,
        judge=judge_fn,
        detector=detector_fn,
        vale_config=vale_config,
        max_workers=max_workers,
        limit=limit,
        on_progress=_progress,
    )
    click.echo(
        f"scored {result.written} pair(s) "
        f"({result.skipped_existing} already scored, skipped) -> {out}"
    )
    if result.errors:
        click.echo(f"{len(result.errors)} scoring error(s):", err=True)
        for rid, msg in result.errors[:10]:
            click.echo(f"  {rid}: {msg}", err=True)

    summary = ev.summarize_scores(out, by=by)
    for name, agg in summary["fields"].items():
        click.echo(
            f"{name}: n={agg['n']} judge={agg['mean_judge_score']} "
            f"vale_alerts={agg['mean_vale_alerts']} detector={agg['mean_detector_score']}"
        )
    if by is not None:
        click.echo(f"by {by}:")
        for facet, fld in summary["by"].items():
            parts = ", ".join(f"{fn}={a['mean_judge_score']}" for fn, a in fld.items())
            click.echo(f"  {facet}: {parts}")
    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"wrote summary -> {summary_path}")

    # Read-only visualisations over the (just-written) scores.jsonl + the corpus.
    # No extra scoring — re-rendering an already-scored corpus is a no-op pass.
    if sample_n is not None or report_path is not None:
        from stylebot import report

        if sample_n is not None:
            click.echo(report.format_scores_sample(out, pairs_path, sample_n, fields=score_fields))
        if report_path is not None:
            written = report.render_scores_report(
                out, pairs_path, report_path, fields=score_fields,
                max_rows=(None if report_max_rows == 0 else report_max_rows),
                facet_by=facet_by,
            )
            click.echo(f"wrote report -> {written}")


if __name__ == "__main__":
    main()
