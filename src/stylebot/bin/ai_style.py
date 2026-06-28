"""ai-style: the one-entry-point CLI (subcommands), thin over the libraries.

Per OVERVIEW "Interfaces": one command, subcommands `synth | split | train |
eval` — not a scatter of loose scripts. Each subcommand is a thin `click`
wrapper that parses flags and calls a typed library function. Only `synth` is
built so far (Phase 2); the rest land with their phases.

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

# Preset slop generators selectable on the CLI. Library callers inject their own
# `synth.Generator`s; these are the bundled defaults (multi-source by design).
_GENERATOR_FACTORIES = {
    "claude": lambda models: synth.anthropic_generator(model=models["claude"]),
    "gpt": lambda models: synth.openai_generator(model=models["gpt"]),
    "local": lambda models: synth.local_generator(model=models["local"] or None),
}


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
@click.option("--sort", "sort_name", type=click.Choice(["none", "length", "source"]), default="none", show_default=True)
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a self-contained HTML report of the selected targets and exit (no generation).")
@click.option("--report-max-rows", default=2000, show_default=True, type=int, help="Cap table rows in the HTML report (0 = all).")
@click.option("--sample", "sample_n", type=int, default=None, help="Print N random targets to stdout and exit (no generation).")
@click.option(
    "--generator",
    "generator_names",
    type=click.Choice(list(_GENERATOR_FACTORIES)),
    multiple=True,
    default=("claude", "gpt"),
    show_default=True,
    help="Slop generators to rotate across (repeat for ≥2; multi-source by design).",
)
@click.option("--claude-model", default="claude-opus-4-8", show_default=True)
@click.option("--gpt-model", default="gpt-4o", show_default=True)
@click.option("--local-model", default="", help="Local model id (else $LOCAL_LLM_MODEL).")
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
    sort_name: str,
    report_path: Path | None,
    report_max_rows: int,
    sample_n: int | None,
    generator_names: tuple[str, ...],
    claude_model: str,
    gpt_model: str,
    local_model: str,
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

    # Resolve the corpus location: explicit flag > $STYLEBOT_DATA_DIR. Refuse the
    # bare cwd default — synth costs money and grows the corpus; be explicit.
    if data_dir is None and not config.get_key("STYLEBOT_DATA_DIR"):
        raise click.UsageError(
            "no --data-dir and $STYLEBOT_DATA_DIR unset; pass --data-dir explicitly "
            "(synth appends to the corpus and costs API spend — never the silent default)"
        )
    resolved_dir = config.resolve_data_dir(data_dir)

    models = {"claude": claude_model, "gpt": gpt_model, "local": local_model}
    if dry_run:
        # Name-only stubs — no API clients, no keys needed to vet selection.
        generators = [
            synth.Generator(
                name={"claude": claude_model, "gpt": gpt_model, "local": f"local-{local_model or 'local'}"}[g]
            )
            for g in generator_names
        ]
    else:
        try:
            generators = [_GENERATOR_FACTORIES[g](models) for g in generator_names]
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


if __name__ == "__main__":
    main()
