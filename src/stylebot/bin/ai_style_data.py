"""
ai-style data: one-off corpus migrations. A **hidden** subcommand group.

The mechanism is `stylebot.migrations`; this is the click shell, in the same
shape as `bin/ai_style_log.py` — declare flags, call a typed function, print.

**Why hidden.** These are the commands the "one entry point, subcommands" rule
in CLAUDE.md used to carve an exception *out* of, on the reasoning that a
migration is run once, by one person, per schema change, and would otherwise
clutter the `--help` tree of a tool nobody should be migrating with twice. That
reasoning was right about the `--help` tree and wrong about the remedy: loose
files under `scripts/` are not in the wheel, so `capture.iter_archives` printed
a command the user did not have, and the blog's RUNBOOK had to reach across the
repo boundary with a `../stylebot/scripts/...` path. `hidden=True` keeps the
tidy `--help` tree and ships the code (W5, 2026-09-03).

So: absent from `ai-style --help`, fully documented under `ai-style data
--help`, and named by every error message and runbook step that wants one.

Both commands **dry-run by default** and need `--apply` to write, because both
are destructive in the one direction that matters — one unlinks legacy archive
files, the other deletes corpus records. `prune-invented-anchors` also keeps a
`.bak`. A migration is deleted once the only corpus that exists has been through
it; see the module docstring in `stylebot.migrations`.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group(hidden=True, context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """One-off corpus migrations (hidden: run once per schema change)."""


@main.command("migrate-captures")
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True
)
@click.option("--apply", is_flag=True, help="Actually convert (default: dry run).")
def migrate_captures_cmd(data_dir: Path, apply: bool) -> None:
    """Convert legacy JSON capture archives and snapshots to directories.

    The superseded `captures/<id>.json` blobs held whole documents as escaped
    strings inside one-line JSON: unreadable, unopenable and undiffable, which
    is perverse given the before->after diff IS the training signal. Support was
    removed in 0.3 and `iter_archives` now REFUSES a leftover blob rather than
    skipping it, so run this until `captures/` holds no `*.json`.

        uv run ai-style data migrate-captures _training_pairs --apply
    """
    from stylebot import migrations

    try:
        migrations.migrate_capture_archives(data_dir, apply=apply, log=click.echo)
    except migrations.MigrationError as exc:
        raise click.ClickException(str(exc)) from exc


@main.command("prune-invented-anchors")
@click.argument(
    "pairs", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--apply", is_flag=True, help="Rewrite pairs.jsonl (default: dry run).")
@click.option(
    "--include-real",
    is_flag=True,
    help="Also cull captured (non-synthetic) pairs. They cannot be regenerated.",
)
def prune_invented_anchors_cmd(pairs: Path, apply: bool, include_real: bool) -> None:
    """Drop pairs whose target carries a link/URL/citation its slop never had.

    The one-off for the corpus that predates `--drop-invented-anchors` (D32),
    which gates this at synthesis time and defaults on. Synthetic pairs only
    unless `--include-real`: a culled synthetic cell is regenerable by the next
    `synth` run, a captured one is not.
    """
    from stylebot import migrations

    keep, report = migrations.find_inventing(pairs, include_real=include_real)
    n_dropped = sum(report["dropped"].values())

    click.echo(f"records          : {report['records']}")
    click.echo(
        f"dropped          : {n_dropped} "
        f"({report['dropped']['synthetic']} synthetic, {report['dropped']['real']} real)"
    )
    if report["spared_real"]:
        click.echo(
            f"real, left alone : {report['spared_real']} "
            "(captured records — pass --include-real to cull, or exclude them at "
            "training assembly with infer.is_anchor_faithful)"
        )
    click.echo(f"remaining        : {len(keep)}")
    for kind, n in report["kinds"]:
        click.echo(f"    {kind:18} {n}")
    click.echo(
        f"targets emptied  : {report['targets_wiped']} of {report['targets']} "
        "— these go back to uncovered, so the next synth run refills them"
    )
    for source, anchors in report["samples"]:
        click.echo(f"  e.g. {source}: invented {', '.join(anchors)}")

    if not n_dropped:
        click.echo("\nnothing to prune.")
        return
    if not apply:
        click.echo("\ndry run — nothing written. Re-run with --apply to prune.")
        return

    backup = migrations.write_pruned(pairs, keep)
    click.echo(f"\npruned {n_dropped} pair(s) -> {pairs} (backup at {backup})")
    click.echo("next: `ai-style synth --gate-retries 2` to refill the emptied cells,")
    click.echo("      then `ai-style eval --prune-orphans` if you keep scores.jsonl.")
