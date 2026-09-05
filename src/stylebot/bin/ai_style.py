"""ai-style: the one-entry-point CLI (subcommands), thin over the libraries.

Per OVERVIEW "Interfaces": one command, subcommands — not a scatter of loose
scripts. Each subcommand is a thin `click` wrapper that parses flags and calls a
typed library function: `log` (Phase 1 capture; a subcommand, not a console
script, since 2026-07-20), `synth` (Phase 2), `make-splits` / `check-splits` /
`extend-eval` (the by-post splits contract), `train-clf` (the voice
classifier), `train` (Phase 3 LoRA SFT) and `promote` (which run serves, D74),
`run` (Phase 4 inference), `eval` (the offline scorer), `export-table` (scores
-> tidy CSV), `merge-adapter` (`stylebot.export`), `serve` (the NDJSON editor
sidecar), `validate-pairs` (the record contract on demand) and `status`
(`stylebot.status` — what the corpus holds, which run serves, which detector
scored it, and where those disagree; the numbers docs may not carry, D52).

**Exit codes are not written here.** The libraries raise `stylebot.errors`'
two classes and `ErrorTranslatingGroup` (on `main`, so the whole subtree
inherits it) turns them into 2 and 1. Until 2026-09-03 this file had six
hand-written `except ValueError:` blocks, all mapping to exit 1, which is how
"no training pairs left after filtering" came to exit 1 while the same class of
mistake in `synth` exited 2. What is still written here is what click alone can
know: flag *combinations* (`--pairs` with a FILE, `--best-of` without a
detector), which are `click.UsageError` at the point of parsing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import click

# Library modules are imported at module scope so click can own their defaults
# and option decorators directly (`default=tr.DEFAULT_LORA_RANK`,
# `synth_options()`); all import free. `infer` stays lazy — it owns no default
# declared here. D49 has the `--help` numbers.
from stylebot import backends, config, draws, pairs, synth, synth_cli
from stylebot import classify_cli as cc
from stylebot import classify_train as ct
from stylebot import eval as ev
from stylebot import splits as sp
from stylebot import train as tr
from stylebot.backends import BACKEND_PARAMS, BackendOpts, backend_options
from stylebot.bin.ai_style_log import main as _log
from stylebot.cli_options import (
    ErrorTranslatingGroup,
    SectionedCommand,
    bundle_options,
    bundles_from_kwargs,
    click_emitter,
)
from stylebot.errors import RunError
from stylebot.lib import is_human_authored


@click.group(
    cls=ErrorTranslatingGroup, context_settings={"help_option_names": ["-h", "--help"]}
)
def main() -> None:
    """stylebot tooling — capture, synthetic pairs, training, eval (subcommands)."""


# Phase 1 capture (bin/ai_style_log.py), mounted here as its entry point. The
# hidden `data` group of one-off migrations emptied on 2026-09-05 (D38).
main.add_command(_log, name="log")


# The styler backend (`run`, `serve`) is one surface, `stylebot.backends`, a
# LIBRARY — its docstring says why. What stays per-command is deliberate:
# `run` defaults --backend to tinker, `serve` runs hotter and requires
# --detector-model (it IS the service) where run's only feeds --best-of.


@main.command("synth", cls=synth_cli.SectionedCommand)
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
    help="Read/write state: where the corpus is appended. Required (no default) "
    "unless $STYLEBOT_DATA_DIR is set — never silently grow the wrong corpus.",
)
@click.option(
    "--field",
    default="automation",
    show_default=True,
    help="Frontmatter field the default selector reads.",
)
@click.option(
    "--max-level",
    default=0,
    show_default=True,
    type=int,
    help="Default selector keeps posts with field <= this.",
)
@click.option("--glob", default="**/*.qmd", show_default=True, help="Glob for the blog-root walk.")
@synth_cli.synth_options()
def synth_cmd(
    files: tuple[Path, ...],
    blog_root: Path | None,
    data_dir: Path | None,
    field: str,
    max_level: int,
    glob: str,
    **kw,
) -> None:
    """Manufacture synthetic (slop -> Dan) pairs from Dan's own prose.

    Provide targets one of two ways: a pre-selected FILES list, or --blog-root
    plus a selector (default: is_human_authored, i.e. automation: 0). Writes the
    Phase-1 pairs.jsonl schema with meta.synthetic / meta.generator. Idempotent
    and resumable — re-running never duplicates.

    To take every post regardless of the frontmatter field, raise --max-level
    above every value it holds; to bypass selection entirely, pass FILES.
    """
    if bool(files) == bool(blog_root):
        raise click.UsageError(
            "provide either FILES (pre-selected) or --blog-root, not both/neither"
        )

    selector = partial(is_human_authored, field=field, max_level=max_level)

    # The shared surface arrives as **kw; fold it back into its option bundles.
    opts = synth_cli.pop_bundles(kw)

    def _data_dir(*, required: bool) -> Path | None:
        # Resolve the corpus location: explicit flag > $STYLEBOT_DATA_DIR. Refuse the
        # bare cwd default — synth costs money and grows the corpus; be explicit.
        if data_dir is None and not config.get_key("STYLEBOT_DATA_DIR"):
            if not required:
                return None  # inspection: nothing to write, nothing to resolve
            raise click.UsageError(
                "no --data-dir and $STYLEBOT_DATA_DIR unset; pass --data-dir explicitly "
                "(synth appends to the corpus and costs API spend — never the silent default)"
            )
        return config.resolve_data_dir(data_dir)

    targets = synth.iter_targets(
        files=list(files) or None,
        blog_root=blog_root,
        selector=selector,
        glob=glob,
        policy=opts.pop("chunk"),
    )

    # The read-only inspection modes (--sample/--report) exit before generation:
    # they write nothing, so they need no corpus location. Deciding that HERE —
    # above the resolution — is what keeps `run_synth`'s data_dir a plain path
    # while this command keeps its own tailored refusal message.
    inspection = opts["inspection"]
    inspecting = inspection.sample_n is not None or inspection.report_path is not None

    # Still resolved for inspection runs when it IS available, so the inspection
    # hints can name the real pairs path.
    # `run_synth` is silent unless handed a sink — this is the CLI, so it
    # gets the click one (stream + colour routing lives in
    # `cli_options.EMIT_LEVELS`, shared with livingthing's `dan-style synth`
    # so the two front-ends cannot render a run differently).
    # No `extra_tags=` here: provenance tagging is a library parameter
    # (`run_synth(extra_tags=…)`), which is how the one tag on the live
    # corpus was ever set. A wrapper that wants tags passes them in code.
    synth_cli.run_synth(
        targets,
        data_dir=_data_dir(required=not inspecting),
        emit=synth_cli.click_emitter(),
        **opts,
    )


@main.command("eval", cls=SectionedCommand)
@ev.eval_options()
def eval_cmd(**kw) -> None:
    """Score a pairs corpus across the eval signals (offline, batched).

    Scores the slop and Dan sides of each pair and appends one id-keyed record
    per pair to scores.jsonl (joinable back to the corpus; resumable — re-running
    skips scored ids). Keyless by default (Vale + null detector); --judge scores
    voice via OpenRouter. The summary aggregates per field and per --by facet
    (e.g. slop_strategy). --report is also the *pair browser*: a keyless run is
    the cheap way to eyeball a fresh synth batch, slop↔Dan side by side with
    generation covariates, faceted and sortable. A styler run scores the same
    way from an output-bearing JSONL (`run --pairs --out`, then --field output).
    """
    parsed = bundles_from_kwargs(kw, ev.EVAL_PARAMS)
    opts, verdict_opts, doc_opts = parsed["eval"], parsed["verdict"], parsed["documents"]
    emit = click_emitter()
    detector = ev.null_detector
    if opts.detector_model is not None:
        from stylebot import classify

        detector = classify.sklearn_detector(opts.detector_model)
    documents = None
    if doc_opts.documents:
        if doc_opts.documents_manifest is None:
            raise click.UsageError("--documents needs --documents-manifest (the adapter to serve)")
        splits_file = config.splits_path(opts.pairs_path.parent)
        documents = ev.document_pass(
            opts.pairs_path,
            opts.scores_path(),
            doc_opts,
            opts.score_fields(),
            manifest=backends.load_manifest(
                doc_opts.documents_manifest, param_hint="--documents-manifest"
            ),
            splits=sp.load_splits(splits_file) if splits_file.exists() else None,
            emit=emit,
        )
    ev.run_eval(opts, verdict_opts, detector=detector, documents=documents, emit=emit)


@main.command("export-table")
@click.option(
    "--scores",
    "scores_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The scores.jsonl to project.",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="CSV to write; a <stem>.meta.json provenance sidecar lands beside it.",
)
@click.option(
    "--note",
    default=None,
    help="Free-text description of the run, recorded in the sidecar.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="A JSON experiment config to embed in the sidecar (the pinned arms/substrate that "
    "produced these scores).",
)
def export_table_cmd(
    scores_path: Path, out_path: Path, note: str | None, config_path: Path | None
) -> None:
    """Project a scores.jsonl to a committable tidy CSV (+ provenance sidecar).

    One row per pair, no prose text: small enough to version next to a write-up,
    and sufficient to redraw every figure — including re-cuts nobody planned at
    run time. The sidecar records the hoisted run-constants, a digest of the
    source scores file and the experiment config, so a later reader can tell
    whether a re-run is comparable or merely similar.

        ai-style export-table --scores sweep/scores.jsonl --out notebook/data/sweep.csv
    """
    from stylebot.eval import write_covariate_table

    extra: dict = {}
    if note:
        extra["note"] = note
    if config_path:
        extra["config"] = json.loads(config_path.read_text(encoding="utf-8"))

    res = write_covariate_table(scores_path, out_path, extra=extra or None)
    size = Path(res["csv"]).stat().st_size
    click.echo(
        f"wrote {res['rows']} rows x {len(res['columns'])} cols -> "
        f"{res['csv']} ({size / 1024:.1f} KB)"
    )
    if res["constants"]:
        click.echo(
            f"hoisted {len(res['constants'])} run-constant column(s) into {res['sidecar'].name}"
        )


@main.command("serve", cls=SectionedCommand)
# --detector-model is excluded from the shared bundle and redeclared here: for
# `serve` it is the service itself, not run's optional --best-of reranker.
@click.option(
    "--detector-model",
    "detector_model",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A trained voice-classifier artifact dir (head.json + meta.json); needs the "
    "pinned embedder (sentence-transformers) installed.",
)
@click.option(
    "--preferences",
    "preferences_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Arm the 'prefer' op: append client decisions (plus a UTC ts) to this JSONL.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=backends.MANIFEST_HELP,
)
@backend_options(defaults=[BackendOpts(temperature=0.7)], exclude=["detector_model"])
def serve_cmd(
    detector_model: Path,
    preferences_path: Path | None,
    manifest_path: Path | None,
    **kw,
) -> None:
    """Score texts over stdin/stdout NDJSON (long-lived editor sidecar).

    Loads the detector once (the embedding model is the slow part), then serves
    {"id", "op": "score", "texts": [...]} -> {"id", "scores": [P(slop), ...]}
    one JSON object per line until EOF. {"op": "info"} returns the artifact's
    meta.json and doubles as the client's ready handshake. With a --backend,
    the 'rewrite' op returns best-of candidates for one prose chunk (scored,
    anchor-flagged, none auto-chosen); with --preferences, the 'prefer' op
    records the client's choice. Protocol details: stylebot.serve.
    """
    from stylebot import classify, serve

    manifest = backends.load_manifest(manifest_path)
    opts = backends.opts_from_manifest(
        manifest,
        replace(bundles_from_kwargs(kw, BACKEND_PARAMS)["opts"], detector_model=detector_model),
    )
    try:
        meta = classify.load_artifact_meta(detector_model)
        detector = classify.sklearn_detector(detector_model)
    except ImportError as exc:
        raise click.ClickException(
            f"--detector-model needs the embedder installed (sentence-transformers): {exc}"
        ) from exc
    prefer = None
    if preferences_path is not None:
        prefer = serve.preference_recorder(preferences_path, run_id=manifest.get("run_id"))
    serve.serve_sidecar(
        opts,
        detector=detector,
        meta=meta,
        make_styler=partial(backends.build_styler, opts) if opts.backend is not None else None,
        prefer=prefer,
    )


@main.command("make-splits")
@click.option(
    "--pairs",
    "pairs_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
    help="The pairs corpus whose posts get partitioned (pairs.jsonl or a pairs/ directory).",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write splits.json (commit it next to the corpus).",
)
@click.option(
    "--eval-frac",
    default=sp.DEFAULT_EVAL_FRAC,
    show_default=True,
    type=float,
    help="Fraction of posts pinned as the frozen eval stratum.",
)
@click.option(
    "--detector-frac",
    default=sp.DEFAULT_DETECTOR_FRAC,
    show_default=True,
    type=float,
    help="Hash-rule share of NON-eval posts routed to the detector pool (rest -> styler).",
)
@click.option(
    "--seed",
    default=0,
    show_default=True,
    type=int,
    help="Seed for the eval sample + the hash salt.",
)
@click.option(
    "--allow-synthetic-eval",
    is_flag=True,
    help="Let synth-only posts into the eval stratum (default: eval is sampled from posts with "
    ">=1 REAL pair, so the frozen eval never consists of our own paraphrases).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite an existing splits file. Re-cutting the partition un-freezes eval — only do "
    "this knowingly.",
)
def make_splits_cmd(
    pairs_path: Path,
    out_path: Path,
    eval_frac: float,
    detector_frac: float,
    seed: int,
    allow_synthetic_eval: bool,
    force: bool,
) -> None:
    """Create the shared by-POST splits contract (eval / styler / detector).

    Pins a frozen eval post list and a deterministic hash rule that routes every
    other post — including future ones — to the styler or detector role. All
    three stages (detector-train via `train-clf --splits`, Phase-3 styler-train,
    the frozen eval) must consume this one file. Prints the role table and the
    danger report (dangerously small strata). Needs no ML deps.
    """
    if out_path.exists() and not force:
        raise click.ClickException(
            f"{out_path} exists — the eval stratum is FROZEN by design. "
            "Re-cutting the partition invalidates every consumer of the old one; "
            "pass --force only if you mean that."
        )

    ds = ct.assemble_dataset(pairs_path)
    if ds.n_pairs == 0:
        raise click.ClickException(f"no content-matched pairs in {pairs_path}")
    posts = sorted(set(ds.groups))
    real_posts = sorted(
        {
            ds.groups[s_row]
            for k, (s_row, _) in enumerate(ds.pair_rows)
            if not ds.pair_is_synthetic(k)
        }
    )
    candidates = None if allow_synthetic_eval else real_posts
    if candidates is not None and not candidates:
        raise click.ClickException(
            "no posts with real pairs to sample eval from — pass --allow-synthetic-eval to override"
        )
    splits_doc = sp.make_splits(
        posts,
        eval_frac=eval_frac,
        detector_frac=detector_frac,
        seed=seed,
        eval_candidates=candidates,
    )
    sp.save_splits(splits_doc, out_path)

    click.echo("\n".join(sp.role_lines(splits_doc, ds)))
    for warning in sp.check_splits(splits_doc, ds):
        click.secho(f"WARNING: {warning}", fg="yellow", err=True)
    click.echo(f"wrote splits -> {out_path} (seed {seed}; eval frozen, rest by hash rule)")


@main.command("check-splits")
@click.option(
    "--splits",
    "splits_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The splits.json to check.",
)
@click.option(
    "--blog-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    show_default=True,
    help="Source tree the pinned eval posts must still exist in.",
)
@click.option(
    "--pairs",
    "pairs_path",
    type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
    default=None,
    help="Also print the per-role post/pair table for this corpus, and the danger report.",
)
@draws.view_options()
def check_splits_cmd(
    splits_path: Path,
    blog_root: Path,
    pairs_path: Path | None,
    draws_dir: Path | None,
    policy_path: Path | None,
) -> None:
    """Health-check the splits contract: eval posts present, strata big enough.

    The read-only counterpart of `make-splits`, and the thing to run after any
    `post-op move`: a renamed post silently changes role, which is how two
    frozen-eval posts leaked into the styler/detector pools (2026-09-03). Exits
    non-zero when a pinned eval post is no longer on disk under --blog-root;
    small-stratum findings stay warnings. The table reads --pairs, or the view
    --draws builds — what `train --draws` would train on (D84).
    """
    pairs_path = draws.corpus_or_view(pairs_path, draws_dir, policy_path)
    splits_doc = sp.load_splits(splits_path)
    spellings = sp.eval_post_spellings(splits_doc)
    missing = set(sp.missing_eval_posts(splits_doc, blog_root))
    click.echo(
        f"eval stratum v{sp.eval_version(splits_doc)}  "
        f"splits_sha256={sp.splits_sha256(splits_path)}"
    )
    click.echo(
        f"eval posts on disk: {len(spellings) - len(missing)}/{len(spellings)} "
        f"(under {blog_root})"
    )
    for canonical, paths in spellings.items():
        mark = "MISSING" if canonical in missing else "ok"
        others = [p for p in paths if p != canonical]
        alias_note = f"  [also: {', '.join(others)}]" if others else ""
        click.echo(f"  {mark:>7}  {canonical}{alias_note}")

    if pairs_path is not None:
        ds = ct.assemble_dataset(pairs_path)
        click.echo("\n".join(sp.role_lines(splits_doc, ds, top=True)))
        report = sp.check_splits(splits_doc, ds, blog_root=blog_root)
    else:
        # No corpus, no danger report — the on-disk contract is checkable alone.
        report = [
            f"{sp.ERROR_PREFIX} pinned eval post {canonical!r} is not on disk under {blog_root}"
            for canonical in sorted(missing)
        ]

    errors = [line for line in report if sp.is_error(line)]
    for line in report:
        if sp.is_error(line):
            click.secho(line, fg="red", err=True)
        else:
            click.secho(f"WARNING: {line}", fg="yellow", err=True)
    if errors:
        raise click.ClickException(f"{len(errors)} pinned eval post(s) missing from {blog_root}")


@main.command("extend-eval", cls=SectionedCommand)
@sp.extend_eval_options()
def extend_eval_cmd(**kw) -> None:
    """Grow the frozen eval stratum: `eval_v(n+1) = eval_v(n) u posts`.

    Add-only and dry-run by default. Posts named here move OUT of the styler /
    detector pools and INTO the frozen eval — the safe direction, since prose
    only ever leaves training. It bumps `eval_version`, appends an
    `eval_history` entry, and needs no migration of existing pairs: role is
    derived from each record's `meta.source` at read time.

    Shrinking the stratum is not expressible: dropping a post out of eval would
    retroactively license training on prose an existing verdict was computed
    over. Anything already trained under the old stratum (the detector head
    above all) must be refit afterwards.
    """
    opts = bundles_from_kwargs(kw, {"opts": sp.ExtendEvalOpts})["opts"]
    sp.apply_eval_extension(opts, emit=click_emitter())


@main.command("train-clf", cls=SectionedCommand)
@cc.train_clf_options()
def train_clf_cmd(**kw) -> None:
    """Train a voice classifier (linear head over a style embedding).

    Generic slop-vs-author trainer over a pairs corpus: embeds both sides of every
    content-matched pair, fits a logistic regression, evaluates leakage-safely (by
    POST, C by nested group-CV), and writes the head.json + meta.json artifact that
    `--detector-model` / `serve` consume. Needs the `classifier` extra. Extra
    positives are a library argument (`classify_train.train(extra_positives=…)`).
    """
    opts = bundles_from_kwargs(kw, cc.TRAIN_CLF_PARAMS)["opts"]
    cc.train_clf(opts, emit=click_emitter())


@main.command("train", cls=SectionedCommand)
@click.option(
    "--data-dir",
    "data_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Data dir holding the corpus (pairs/ or pairs.jsonl). Required, no default: "
    "the explicit path is the reproducibility record — never silently train on the "
    "wrong corpus.",
)
@click.option(
    "--manifest-out",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the run manifest JSON (the ONLY artifact that gets committed).",
)
@click.option(
    "--work-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Scratch dir for assembled data, tinker logs, and the exported adapter (never committed).",
)
@click.option(
    "--splits",
    "splits_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The shared splits.json: train on STYLER-role posts only (the contract; see "
    "make-splits). Omit only for fixture/experiment corpora.",
)
@click.option(
    "--blog-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Tree the pinned eval posts must still exist in; a renamed one fails the run (D34).",
)
@click.option(
    "--val-frac",
    default=tr.DEFAULT_VAL_FRAC,
    show_default=True,
    type=float,
    help="Fraction of POSTs held out for val loss (by-post, deterministic).",
)
@click.option(
    "--run-id",
    default=None,
    help="Run id for the manifest. Default: a UTC timestamp minted at launch "
    "(train._default_run_id).",
)
@click.option(
    "--price-per-mtok",
    "train_price_per_mtok",
    default=None,
    type=float,
    help="Train price $/1M tokens for the cost estimate (see the Tinker models page; "
    "deliberately not hardcoded).",
)
@click.option(
    "--download-adapter/--no-download-adapter",
    default=True,
    show_default=True,
    help="Export the trained LoRA as a PEFT adapter into work-dir after the run.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Assemble, print the manifest + token/cost estimate, and stop. Writes nothing; "
    "no key, no spend.",
)
@click.option("--yes", "assume_yes", is_flag=True, help="Skip the pre-spend confirmation prompt.")
@draws.view_options()
@bundle_options([tr.Hyperparameters])
def train_cmd(
    data_dir: Path,
    manifest_out: Path,
    work_dir: Path,
    splits_path: Path | None,
    blog_root: Path | None,
    val_frac: float,
    run_id: str | None,
    train_price_per_mtok: float | None,
    download_adapter: bool,
    dry_run: bool,
    assume_yes: bool,
    draws_dir: Path | None,
    policy_path: Path | None,
    **kw,
) -> None:
    """Phase 3: LoRA-SFT the styler on Tinker (stylebot.train.run_training).

    Assembles the corpus (styler-role posts only under --splits, near-copy pairs
    KEPT) — or, with --draws, the view of those draws — cuts a by-POST val
    split, writes the manifest, runs the Tinker recipe. Needs the `trainer`
    extra and TINKER_API_KEY; --dry-run needs neither.
    """
    pairs_path = config.pairs_path(data_dir)
    if draws_dir is None and not pairs_path.exists():
        raise click.ClickException(f"no corpus (pairs/ or pairs.jsonl) in {data_dir}")

    kwargs = dict(
        draws_dir=draws_dir,
        policy=draws.load_policy(policy_path),
        hp=bundles_from_kwargs(kw, {"hp": tr.Hyperparameters})["hp"],
        splits_path=splits_path,
        blog_root=blog_root,
        val_frac=val_frac,
        run_id=run_id,
        train_price_per_mtok=train_price_per_mtok,
        download_adapter=download_adapter,
    )

    # Always preview first: the dry-run manifest is also the pre-spend estimate.
    preview = tr.run_training(pairs_path, work_dir, manifest_out, dry_run=True, **kwargs)
    data = preview.manifest["data"]
    est = preview.manifest["estimate"]
    click.echo(
        f"{data['n_train']} train / {data['n_val']} val pair(s) "
        f"({data['n_real']} real / {data['n_synthetic']} synthetic; "
        f"dropped {data['dropped']}) from {data['n_source_records']} record(s), "
        f"pairs sha256 {data['pairs_sha256']}"
    )
    click.echo(tr.format_passage_scale(data["chunk_scale"]))
    cost = f" ≈ ${est['cost_usd']}" if est["cost_usd"] is not None else ""
    click.echo(
        f"~{est['train_tokens']:,} train tokens ({est['basis']}){cost} on "
        f"{preview.manifest['hyperparameters']['base_model']}"
    )
    if dry_run:
        click.echo(f"[dry-run] the manifest a paid run would write to {preview.manifest_path}:")
        click.echo(json.dumps(preview.manifest, indent=2))
        return

    if not config.get_key("TINKER_API_KEY"):
        raise click.ClickException(
            "TINKER_API_KEY is not set — the paid run needs it. Export it, or "
            "run under direnv (`direnv exec <dir-with-an-.envrc> uv run ai-style train ...`)."
        )
    if not assume_yes:
        click.confirm("Launch the paid Tinker run?", abort=True)

    result = tr.run_training(pairs_path, work_dir, manifest_out, dry_run=False, **kwargs)

    run_result = result.manifest["result"] or {}
    for key in ("train_mean_nll", "test/nll", "train_tokens"):
        if key in run_result:
            click.echo(f"{key}: {run_result[key]}")
    checkpoints = run_result.get("checkpoints") or {}
    if checkpoints.get("sampler_path"):
        click.echo(f"sampler checkpoint: {checkpoints['sampler_path']}")
    if result.adapter_dir:
        click.echo(f"PEFT adapter -> {result.adapter_dir} (never commit this)")
    click.echo(f"manifest -> {result.manifest_path} (commit this)")


@main.command("promote")
@click.option(
    "--runs",
    "runs_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory of run manifests (where `train --manifest-out` wrote them).",
)
@click.argument("run_id")
def promote_cmd(runs_dir: Path, run_id: str) -> None:
    """Make RUN_ID the serving run: `promoted: true` on it, false on the rest (D74)."""
    click.echo(f"promoted {run_id}: {tr.promote(runs_dir, run_id)} (commit the manifests)")


@main.command("merge-adapter")
@click.option(
    "--adapter",
    "adapter_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The PEFT adapter directory written by `train --download-adapter`.",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where to write the merged HF checkpoint (never commit it — it is base-model sized).",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Training-run manifest to read the base model from. Either this or --base-model.",
)
@click.option(
    "--base-model",
    default=None,
    help="Base model the adapter was trained on; overrides the manifest's.",
)
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Scratch for the key-inverted adapter copy. [default: beside --out]",
)
def merge_adapter_cmd(
    adapter_dir: Path,
    out_dir: Path,
    manifest_path: Path | None,
    base_model: str | None,
    work_dir: Path | None,
) -> None:
    """Merge a trained LoRA adapter into its base weights (stylebot.export).

    The step between `train` and serving the styler off somebody else's
    computer: emits one merged HF checkpoint that a local runtime takes from
    there — `mlx_lm.convert` to quantize for MLX, llama.cpp, vLLM, or plain
    transformers.

    It exists because the adapter `train --download-adapter` writes is in
    SERVING key format while the cookbook's merge expects RAW Tinker keys, so
    feeding stylebot's own export to its own next step fails; this inverts the
    keys first. Quantization, config surgery and chat-template edits are
    deliberately not here — they are per-runtime policy, so wrap this.

    Needs the `trainer` extra. Idempotent: an existing --out is left alone.
    """
    from stylebot import export

    if base_model is None:
        if manifest_path is None:
            raise click.UsageError("pass --base-model, or --manifest to read it from")
        try:
            base_model = export.base_model_of(json.loads(manifest_path.read_text(encoding="utf-8")))
        except RunError as exc:
            raise click.ClickException(f"{manifest_path}: {exc}") from exc

    click.echo(f"merging {adapter_dir} into {base_model} ...")
    out = export.merge_adapter(
        adapter_dir,
        out_dir,
        base_model=base_model,
        work_dir=work_dir,
    )
    click.echo(f"merged model -> {out} (never commit this)")
    click.echo("quantize it for local serving with e.g. `mlx_lm.convert --hf-path <it> -q`")


@main.command("run", cls=SectionedCommand)
@click.argument(
    "file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
@backends.run_options(defaults=[BackendOpts(backend="tinker")])
def run_cmd(file: Path | None, **kw) -> None:
    """Phase 4: rewrite FILE (or stdin) through the trained styler.

    Chunks prose at training granularity (frontmatter and code fences
    protected, original whitespace reassembled verbatim), sends each chunk
    with the frozen STYLE_SYSTEM, and prints the rewrite to stdout (--write
    edits in place with a backup). See `stylebot.infer` for the library seam.

    With --pairs/--out it runs in **corpus mode** instead: the styler rewrites
    each pair's slop side and the result lands as a top-level "output" field,
    which `ai-style eval --field output` scores beside the same corpus's slop
    and Dan sides. That is the shape the eval harness has always documented and
    nothing produced — the missing half of comparing two adapters.
    """
    from stylebot import infer

    parsed = bundles_from_kwargs(kw, backends.RUN_PARAMS)
    run: backends.RunOpts = parsed["run"]
    # The manifest is the committed reproducibility record: it carries the
    # checkpoint, the base model, the renderer AND the chunk diet the adapter
    # was trained on. Explicit flags still win over every one of them.
    opts = backends.opts_from_manifest(backends.load_manifest(run.manifest_path), parsed["opts"])
    corpus = backends.corpus_mode(run, source=file)
    styler = backends.build_styler(opts)
    scorer = None
    if run.best_of > 1:
        if opts.detector_model is None:
            raise click.UsageError("--best-of needs --detector-model for the p_dan scoring")
        from stylebot.classify import sklearn_detector

        scorer = infer.detector_scorer(sklearn_detector(opts.detector_model))
    emit = click_emitter()
    if corpus:
        backends.rewrite_corpus(run, styler, scorer=scorer, emit=emit)
        return
    text = file.read_text(encoding="utf-8") if file else click.get_text_stream("stdin").read()
    result = backends.rewrite_document(
        text, run, opts, styler, scorer=scorer, source=file, emit=emit
    )
    if not (run.write_in_place and file):
        click.echo(result.text, nl=not result.text.endswith("\n"))


@main.command("status")
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Resolves --pairs and --splits when they are not given. "
    "[default: --data-dir > $STYLEBOT_DATA_DIR > ./_training_pairs]",
)
@click.option(
    "--pairs",
    "pairs_path",
    type=click.Path(path_type=Path),
    default=None,
    help="The corpus: a pairs.jsonl or a pairs/ shard directory.",
)
@click.option(
    "--splits",
    "splits_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="The splits.json whose eval stratum and role table to report.",
)
@click.option(
    "--runs",
    "runs_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory of committed run manifests. No default: stylebot writes a "
    "manifest wherever `train --manifest-out` says, and invents no convention.",
)
@click.option(
    "--detector-model",
    "detector_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="A trained voice-classifier artifact directory (head.json + meta.json).",
)
@click.option(
    "--blog-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Source tree the pinned eval posts must still exist in. Without it, "
    "post presence is not checked.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit the raw Status as JSON.")
@draws.view_options()
def status_cmd(
    data_dir: Path | None,
    pairs_path: Path | None,
    splits_path: Path | None,
    runs_dir: Path | None,
    detector_dir: Path | None,
    blog_root: Path | None,
    as_json: bool,
    draws_dir: Path | None,
    policy_path: Path | None,
) -> None:
    """Where this pipeline stands: corpus (or view), splits, promoted run, detector.

    The command the documentation points at instead of carrying numbers it
    cannot keep true. Read-only, keyless, and free.

    Nothing is discovered: stylebot does not know where your blog, runs or
    detector artifact are, so `--runs` / `--detector-model` / `--blog-root` /
    `--draws` have no defaults and unnamed artifacts are reported as "not given"
    rather than guessed at. `--pairs`/`--splits` fall back to the usual
    `--data-dir` > `$STYLEBOT_DATA_DIR` > `./_training_pairs` precedence. A
    policy mirror like `dan-style status` is this command with all six filled in.

    The last block is the cross-check: a detector fit under a different
    `splits_sha256` than `splits.json` now carries, a promoted run that trained
    on a different partition, plus the `check-splits` danger report.
    """
    from stylebot import status as st

    resolved = config.resolve_data_dir(data_dir)
    if pairs_path is None:
        candidate = config.pairs_path(resolved)
        pairs_path = candidate if candidate.exists() else None
    if splits_path is None:
        candidate = config.splits_path(resolved)
        splits_path = candidate if candidate.exists() else None

    report = st.collect(
        pairs=pairs_path,
        splits=splits_path,
        runs_dir=runs_dir,
        detector=detector_dir,
        blog_root=blog_root,
        draws_dir=draws_dir,
        policy=draws.load_policy(policy_path),
    )
    if as_json:
        click.echo(json.dumps(asdict(report), indent=2, default=str))
        return
    click.echo(st.format_status(report))


@main.command("validate-pairs")
@click.argument("pairs_file", type=click.Path(exists=True, path_type=Path))
@click.option("--limit", type=int, default=20, show_default=True, help="Problem lines to print.")
def validate_pairs_cmd(pairs_file: Path, limit: int) -> None:
    """Check a corpus against the record contract. Read-only (D84).

    The gate `train` runs on its input, exposed so it can be run on demand: the
    message triple and the frozen system prompt, meta types and domains, and
    `record_id` uniqueness across the whole corpus. A duplicate is fixed by a
    policy clause and a rebuilt view, never by rewriting the file.

    PAIRS_FILE is a `pairs.jsonl` or a `pairs/` shard directory. Uniqueness is
    checked across every shard, which is the reason to point this at the
    directory rather than at each file in turn.
    """
    problems = pairs.validate_pairs_file(pairs_file)
    dupes = [(ln, e) for ln, e in problems if any("duplicate record_id" in x for x in e)]
    click.echo(
        f"{pairs_file}: {len(problems)} problem line(s), {len(dupes)} duplicate record_id(s)"
    )
    for where, errs in problems[:limit]:
        # int for a single file, "<shard>.jsonl:<line>" for a sharded corpus.
        loc = f"line {where}" if isinstance(where, int) else where
        click.echo(f"  {loc}: {'; '.join(errs)}")
    if len(problems) > limit:
        click.echo(f"  ... and {len(problems) - limit} more")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
