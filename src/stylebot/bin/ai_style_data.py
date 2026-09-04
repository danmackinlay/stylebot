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

Every command **dry-runs by default** and needs `--apply` to write, because all
of them are destructive in the one direction that matters — one unlinks legacy
archive files, the other deletes corpus records. The prune keeps a `.bak`. A
migration is deleted once the only corpus that exists has been through it; see
the module docstring in `stylebot.migrations`.
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

    migrations.migrate_capture_archives(data_dir, apply=apply, log=click.echo)


@main.command("shard-pairs")
@click.argument(
    "data_dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True
)
@click.option("--apply", is_flag=True, help="Actually write pairs/ (default: dry run).")
@click.option(
    "--keep-file/--remove-file",
    default=False,
    show_default=True,
    help="With --apply, whether to leave the original pairs.jsonl in place. Removing it is "
    "what makes the shards authoritative — `config.pairs_path` prefers pairs/ the moment it "
    "exists, so a left-behind file is a stale copy nothing reads.",
)
def shard_pairs_cmd(data_dir: Path, apply: bool, keep_file: bool) -> None:
    """Split `<data-dir>/pairs.jsonl` into the `<data-dir>/pairs/` shard layout.

    One file becomes `captures.jsonl` plus one `synth-<YYYY-MM>.jsonl` per month
    (`stylebot.corpus.shard_of` is the rule, D50 the argument). Nothing else
    changes: same records, same order within a shard, same ids.

    A one-off: run it once per corpus and never again. `config.pairs_path`
    prefers `pairs/` from the moment it exists, so every reader follows.

        uv run ai-style data shard-pairs _training_pairs --apply
    """
    from stylebot import corpus
    from stylebot.jsonl import iter_jsonl

    src = data_dir / "pairs.jsonl"
    out = data_dir / corpus.PAIRS_DIR_NAME
    if not src.exists():
        raise click.ClickException(f"no {src} to shard")
    if out.exists():
        raise click.ClickException(f"{out} already exists — this corpus is already sharded")

    counts: dict[str, int] = {}
    total = 0
    for rec in iter_jsonl(src, keep_undecodable=True):
        total += 1
        name = corpus.shard_of(rec)
        counts[name] = counts.get(name, 0) + 1
    click.echo(f"{src}: {total} record(s) -> {len(counts)} shard(s)")
    for name, n in sorted(counts.items()):
        click.echo(f"  {name:<26} {n:>6}")

    if not apply:
        click.echo("\ndry run — nothing written. Re-run with --apply to shard.")
        return

    written = corpus.shard_corpus(src, out)
    round_tripped = sum(written.values())
    if round_tripped != total:
        raise click.ClickException(
            f"round-trip mismatch: {total} in, {round_tripped} out — {out} left in place for "
            "inspection, {src} untouched"
        )
    click.echo(f"\nwrote {round_tripped} record(s) -> {out}/")
    if keep_file:
        click.echo(f"left {src} in place; NOTHING READS IT NOW — delete it once you have checked")
    else:
        src.unlink()
        click.echo(f"removed {src} (the shards are the corpus now)")
    click.echo("next: `ai-style validate-pairs <data-dir>/pairs` to confirm 0 problems.")


def _import_attr(spec: str, hint: str):
    """Resolve a `module:attribute` string to the object it names."""
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise click.BadParameter(f"expected 'module:callable', got {spec!r}", param_hint=hint)
    from importlib import import_module

    try:
        return getattr(import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise click.BadParameter(f"cannot import {spec!r}: {exc}", param_hint=hint) from exc


def _load_policy(spec: str):
    """Import `module:callable` and call it for a `ChunkPolicy`.

    **The one knob that cannot be a flag**, and the reason this command takes a
    factory rather than a flag bundle: `div_is_prose` is a *predicate*, and
    re-composing the blog's diet out of flags is the hand-composition that
    charged the corpus 70 cells when one composer dropped it (D29). The user
    names the composed policy object instead:

        --policy livingthing.training_targets:chunk_policy

    stylebot still knows nothing about the blog; it imports what it is told to,
    the way a `selector` arrives as a callable.
    """
    return _import_attr(spec, "--policy")()


def _echo_composition(label: str, comp: dict) -> None:
    click.echo(
        f"  {label:<22} {comp['total']:>6} total  {comp['real']:>4} real  "
        f"{comp['synthetic']:>6} synthetic  {comp['posts']:>4} posts"
    )


@main.command("prune-stale")
@click.argument("pairs", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--blog-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Source tree to re-cut against. Required unless --no-orphans.",
)
@click.option(
    "--splits",
    "splits_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="splits.json. Required unless --no-eval-synthetic.",
)
@click.option(
    "--policy",
    "policy_spec",
    metavar="MODULE:CALLABLE",
    help="Zero-arg factory returning the ChunkPolicy to re-cut under, e.g. "
    "'livingthing.training_targets:chunk_policy'. Required unless --no-orphans: a "
    "predicate like div_is_prose cannot be a flag, and re-cutting under an "
    "approximation of the live diet measures nothing.",
)
@click.option(
    "--orphans/--no-orphans",
    default=True,
    show_default=True,
    help="Drop synthetic pairs whose target no current cut reproduces.",
)
@click.option(
    "--eval-synthetic/--no-eval-synthetic",
    default=True,
    show_default=True,
    help="Drop synthetic pairs whose source resolves to the frozen eval role.",
)
@click.option(
    "--selector",
    "selector_spec",
    metavar="MODULE:CALLABLE",
    help="Pair-level training selector for the passage-scale report only "
    "(e.g. 'livingthing.bin.train_styler:styler_pair_ok'). Never filters what is written.",
)
@click.option("--top", default=10, show_default=True, type=int, help="Posts to list.")
@click.option("--apply", is_flag=True, help="Rewrite pairs.jsonl (default: dry run).")
def prune_stale_cmd(
    pairs: Path,
    blog_root: Path | None,
    splits_path: Path | None,
    policy_spec: str | None,
    orphans: bool,
    eval_synthetic: bool,
    selector_spec: str | None,
    top: int,
    apply: bool,
) -> None:
    """Drop cells the current policy would not produce, and eval contamination.

    The corpus consequence of two fixes rather than of rot: an ORPHAN (no future
    `synth` run regenerates that cell) and eval contamination (D53). **Real
    captures are never dropped** — the orphan question does not apply to them,
    so the report says UNCHECKED rather than clean.

    An orphan is not automatically a mistake. Read the split this prints between
    a policy change and prose Dan has simply edited since; the latter is the
    corpus being a frozen record of prose that moved, not damage.

        uv run ai-style data prune-stale _training_pairs/pairs.jsonl \\
            --blog-root . --splits _training_pairs/splits.json \\
            --policy livingthing.training_targets:chunk_policy
    """
    from stylebot import migrations
    from stylebot.splits import load_splits

    if orphans and (blog_root is None or policy_spec is None):
        raise click.UsageError(
            "--orphans needs both --blog-root and --policy (or pass --no-orphans)"
        )
    if eval_synthetic and splits_path is None:
        raise click.UsageError("--eval-synthetic needs --splits (or pass --no-eval-synthetic)")

    policy = _load_policy(policy_spec) if policy_spec else None
    splits = load_splits(splits_path) if splits_path else None
    selector = _import_attr(selector_spec, "--selector") if selector_spec else None

    try:
        plan = migrations.plan_stale_prune(
            pairs,
            blog_root=blog_root,
            policy=policy,
            splits=splits,
            drop_orphans=orphans,
            drop_eval=eval_synthetic,
        )
    except migrations.MigrationError as exc:
        raise click.ClickException(str(exc)) from exc

    if plan.orphan_report is not None:
        r = plan.orphan_report
        click.echo("orphans — cells no current cut reproduces")
        click.echo(f"  synthetic pairs checked : {r.synthetic}")
        click.echo(
            f"  real pairs              : {r.real_unchecked} UNCHECKED "
            "(cut from capture diffs, not by iter_targets — never orphanable, never dropped)"
        )
        click.echo(f"  distinct cells          : {r.cells}, of which orphaned {r.orphan_cells}")
        click.echo(f"  orphaned pairs          : {r.orphans} across {r.sources_touched} posts")
        if r.orphans_missing_source:
            click.echo(
                f"  ...source missing       : {r.orphans_missing_source} "
                "(post renamed without a splits alias, or deleted — fix the alias, "
                "do not prune)"
            )
        for diet, n in sorted(r.per_diet.items(), key=lambda kv: -kv[1]):
            click.echo(f"      {diet[0]:<16} @{diet[1]:<5} {n}")
        for source, n in r.per_post.most_common(top):
            click.echo(f"      {n:>5}  {source}")

    if plan.eval_report is not None:
        e = plan.eval_report
        click.echo("\neval contamination — synthetic pairs cut from frozen-eval prose")
        click.echo(f"  synthetic dropped       : {e.synthetic_dropped}")
        click.echo(f"  real, left alone        : {e.real_spared}")
        for source, n in e.per_post.most_common():
            click.echo(f"      {n:>5}  {source}")
        for gen, n in e.per_generator.most_common():
            click.echo(f"      {n:>5}  {gen}")

    click.echo(
        f"\noverlap                   : {plan.n_both} pair(s) in both sets "
        f"({plan.n_orphans} orphan + {plan.n_eval} eval - {plan.n_both} = {plan.n_dropped} dropped)"
    )
    click.echo("\ncomposition")
    for label, comp in plan.composition.items():
        _echo_composition(label, comp)

    if splits is not None:
        click.echo("\nstyler train set (role filter + passage containment; no val split)")
        for label, recs in plan.variants.items():
            scale = migrations.styler_passage_scale(recs, splits=splits, selector=selector)
            click.echo(
                f"  {label:<22} {scale['pairs']:>5} pairs / {scale['passages']:>5} passages"
                f"  mean {scale['mean_pairs_per_passage']:.2f}"
                f"  real {scale['real_pair_share']:.2f}% of pairs,"
                f" {scale['real_passage_share']:.2f}% of passages"
            )

    if not plan.n_dropped:
        click.echo("\nnothing to prune.")
        return
    if not apply:
        click.echo("\ndry run — nothing written. Re-run with --apply to prune.")
        return

    backup = migrations.write_pruned(pairs, plan.dropped)
    click.echo(f"\npruned {plan.n_dropped} pair(s) -> {pairs} (backup at {backup})")
    click.echo("next: `ai-style eval --prune-orphans` if you keep scores.jsonl, and a")
    click.echo("      `synth` run to refill the cells whose diet is still live policy.")
