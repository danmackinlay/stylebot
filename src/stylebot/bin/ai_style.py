"""ai-style: the one-entry-point CLI (subcommands), thin over the libraries (D38).

Each subcommand parses flags and calls a typed library function; the phase
map is OVERVIEW's. Exit codes are not written here: the libraries raise
`stylebot.errors`' two classes and `ErrorTranslatingGroup` on `main` turns
them into 2 and 1 (D47). What stays here is what click alone can know — flag
combinations, which are `click.UsageError` at parse time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import click

# Library modules import at module scope so click can own their defaults; all
# import free. `infer` stays lazy (D49).
from stylebot import backends, classify, config, draws, pairs, synth, synth_cli
from stylebot import classify_cli as cc
from stylebot import classify_train as ct
from stylebot import eval as ev
from stylebot import features as ft
from stylebot import fit_judge as fj
from stylebot import splits as sp
from stylebot import train as tr
from stylebot.backends import BACKEND_PARAMS, BackendOpts, backend_options
from stylebot.bin.ai_style_log import main as _log
from stylebot.cli_options import (
    ErrorTranslatingGroup,
    SectionedCommand,
    bundles_from_kwargs,
    click_emitter,
)
from stylebot.errors import RunError
from stylebot.lib import SourcePolicy


@click.group(cls=ErrorTranslatingGroup, context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """stylebot tooling — capture, synthetic pairs, training, eval (subcommands)."""


# Phase 1 capture (bin/ai_style_log.py), mounted here as its entry point.
main.add_command(_log, name="log")


# The styler backend (`run`, `serve`) is `stylebot.backends` (D35); per
# command, `run` defaults --backend to tinker and `serve` requires --detector-model.


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
    help="Frontmatter field the default selector reads (`lib.SourcePolicy`).",
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
    """Manufacture synthetic (slop -> author) pairs from human-authored prose.

    Targets are a pre-selected FILES list, or --blog-root plus a source policy
    (--glob/--field/--max-level; raise --max-level above every value to take
    every post). Appends draws; idempotent and resumable.
    """
    if bool(files) == bool(blog_root):
        raise click.UsageError(
            "provide either FILES (pre-selected) or --blog-root, not both/neither"
        )

    source_policy = SourcePolicy(glob=glob, meta_field=field, max_level=max_level)

    # Fold the shared surface back into its option bundles.
    opts = synth_cli.pop_bundles(kw)

    def _data_dir(*, required: bool) -> Path | None:
        # Explicit flag > $STYLEBOT_DATA_DIR; no cwd default where money is spent.
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
        source_policy=source_policy,
        policy=opts.pop("chunk"),
    )

    # The inspection modes (--sample/--report) write nothing and need no corpus
    # location; deciding that here keeps `run_synth`'s data_dir a plain path.
    inspection = opts["inspection"]
    inspecting = inspection.sample_n is not None or inspection.report_path is not None

    # Still resolved for inspection runs, so the hints can name the real path.
    # `run_synth` is silent unless handed a sink; the CLI hands it the click
    # one. Provenance tags are a library parameter (`extra_tags=`), not a flag.
    synth_cli.run_synth(
        targets,
        data_dir=_data_dir(required=not inspecting),
        emit=synth_cli.click_emitter(),
        **opts,
    )


@main.command("eval", cls=SectionedCommand)
@ev.eval_options()
def eval_cmd(**kw) -> None:
    """Score a corpus (--draws, or a scratch --pairs) across the eval signals.

    One id-keyed row per (pair, field, signal) appended to scores.jsonl,
    resumable; keyless by default (Vale + null detector), --judge scores voice
    via OpenRouter. --report is the pair browser and --by facets the summary. A
    styler's output scores from `run --pairs --out`, then --field output.
    """
    parsed = bundles_from_kwargs(kw, ev.EVAL_PARAMS)
    opts, verdict_opts, doc_opts = parsed["eval"], parsed["verdict"], parsed["documents"]
    opts = replace(opts, pairs_path=parsed["view"].corpus_or_view(opts.pairs_path, required=True))
    emit = click_emitter()
    detector = ev.null_detector
    if opts.detector != "voice-clf" or opts.detector_model is not None:
        detector = ev.build_detector(opts.detector, opts.detector_model)
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


@main.command("features", cls=SectionedCommand)
@ft.features_options()
def features_cmd(**kw) -> None:
    """Syntactic proxies per stratum: real vs synthetic (or --by a meta key),
    pooled over one side of --draws' view and every --pairs; --separability adds
    each facet's grouped-CV AUC against real. Keyless, read-only (D88)."""
    parsed = bundles_from_kwargs(kw, ft.FEATURES_PARAMS)
    click.echo(ft.features_pass(parsed["opts"], parsed["view"]))


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
    """Project a scores.jsonl to a committable tidy CSV (+ provenance sidecar):
    one row per pair, no prose text; the sidecar records the hoisted
    run-constants and a digest of the source file.

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
@click.option(
    "--detector",
    "detector_kinds",
    multiple=True,
    type=click.Choice(list(ev.DETECTOR_KINDS)),
    help="Arm a detector for the 'score' op (repeatable; the first is the default). voice-clf "
    "needs --detector-model; sapling needs $SAPLING_API_KEY and spends per character.  "
    "[default: voice-clf]",
)
# --detector-model is redeclared here: for `serve` it is the service itself.
@click.option(
    "--detector-model",
    "detector_model",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A trained voice-classifier artifact dir (head.json + meta.json); needs the "
    "pinned embedder (sentence-transformers) installed. Required with --detector voice-clf, "
    "and the only rewrite reranker (D106).",
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
    detector_kinds: tuple[str, ...],
    detector_model: Path | None,
    preferences_path: Path | None,
    manifest_path: Path | None,
    **kw,
) -> None:
    """Score texts over stdin/stdout NDJSON (long-lived editor sidecar).

    Arms each --detector once, then serves one JSON object per line until
    EOF; with a --backend the 'rewrite' op returns best-of candidates (reranked
    by the voice head only), with --preferences the 'prefer' op records the
    choice. Protocol: stylebot.serve.
    """
    from stylebot import classify, serve

    kinds = list(detector_kinds) or ["voice-clf"]
    manifest = backends.load_manifest(manifest_path)
    opts = backends.opts_from_manifest(
        manifest,
        replace(bundles_from_kwargs(kw, BACKEND_PARAMS)["opts"], detector_model=detector_model),
    )
    detectors = ev.arm_detectors(kinds, detector_model)
    base_meta = classify.load_artifact_meta(detector_model) if "voice-clf" in kinds else {}
    reranker = dict(zip(kinds, detectors.values(), strict=True)).get("voice-clf")
    prefer = None
    if preferences_path is not None:
        prefer = serve.preference_recorder(preferences_path, run_id=manifest.get("run_id"))
    serve.serve_sidecar(
        opts,
        detectors=detectors,
        meta=serve.sidecar_meta(detectors, base_meta),
        make_styler=partial(backends.build_styler, opts) if opts.backend is not None else None,
        reranker=reranker,
        prefer=prefer,
    )


@main.command("make-splits", cls=SectionedCommand)
@click.option(
    "--pairs",
    "pairs_path",
    type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
    help="A scratch corpus whose posts get partitioned (pairs.jsonl or a shard directory). "
    "The corpus itself is --draws.",
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
@draws.view_options()
def make_splits_cmd(
    pairs_path: Path | None,
    out_path: Path,
    eval_frac: float,
    detector_frac: float,
    seed: int,
    allow_synthetic_eval: bool,
    force: bool,
    **kw,
) -> None:
    """Create the shared by-post splits contract (eval / styler / detector):
    a frozen eval list plus a hash rule for every other post, present and
    future. Prints the role table and the danger report. Needs no ML deps."""
    view = bundles_from_kwargs(kw, {"view": draws.ViewOpts})["view"]
    pairs_path = view.corpus_or_view(pairs_path, required=True)
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


@main.command("check-splits", cls=SectionedCommand)
@sp.check_splits_options()
def check_splits_cmd(**kw) -> None:
    """Health-check the splits contract: eval posts present, strata big enough.

    Read-only, and the thing to run after any `post-op move` (D34). Exits 1 when
    a pinned post or corpus source is gone from --blog-root; small strata are warnings.
    """
    parsed = bundles_from_kwargs(kw, {"opts": sp.CheckSplitsOpts, "view": draws.ViewOpts})
    sp.check_splits_pass(parsed["opts"], parsed["view"], emit=click_emitter())


@main.command("extend-eval", cls=SectionedCommand)
@sp.extend_eval_options()
def extend_eval_cmd(**kw) -> None:
    """Grow the frozen eval stratum: `eval_v(n+1) = eval_v(n) u posts`.

    Add-only and dry-run by default; bumps `eval_version` and appends an
    `eval_history` entry. Shrinking is not expressible (D42). Anything trained
    under the old stratum, the detector head above all, must be refit.
    """
    opts = bundles_from_kwargs(kw, {"opts": sp.ExtendEvalOpts})["opts"]
    sp.apply_eval_extension(opts, emit=click_emitter())


@main.command("train-clf", cls=SectionedCommand)
@cc.train_clf_options()
def train_clf_cmd(**kw) -> None:
    """Train a voice classifier (linear head over a style embedding): fit by
    post-split nested group-CV, written as the head.json + meta.json artifact
    that `--detector-model` / `serve` consume. Needs the `classifier` extra."""
    parsed = bundles_from_kwargs(kw, cc.TRAIN_CLF_PARAMS)
    opts = parsed["opts"]
    opts = replace(opts, pairs_path=parsed["view"].corpus_or_view(opts.pairs_path, required=True))
    cc.train_clf(opts, emit=click_emitter())


@main.command("train", cls=SectionedCommand)
@draws.view_options()
@tr.train_options()
def train_cmd(**kw) -> None:
    """Phase 3: LoRA-SFT the styler on Tinker (stylebot.train.run_training_pass).

    Corpus (styler-role posts under --splits, or the view of --draws) → by-POST
    val split → manifest → Tinker. Needs `trainer` + TINKER_API_KEY; --dry-run neither.
    """
    # ViewOpts first: TrainOpts also names `draws_dir`, as a library-only field.
    parsed = bundles_from_kwargs(kw, {"view": draws.ViewOpts, **tr.TRAIN_PARAMS})
    view = parsed["view"]
    opts = replace(
        parsed["opts"],
        hp=parsed["hp"],
        draws_dir=view.draws_dir,
        policy=view.view_policy(),
        views_root=view.views_root,
    )
    confirm = None if opts.assume_yes else (lambda _: click.confirm("Launch the paid Tinker run?"))
    tr.run_training_pass(opts, confirm=confirm, emit=click_emitter())


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
    """Merge a trained LoRA adapter into its base weights (stylebot.export, D31):
    one merged HF checkpoint a local runtime takes from there. Quantization,
    config surgery and chat-template edits are per-runtime policy, not here.
    Needs the `trainer` extra. Idempotent: an existing --out is left alone."""
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

    Chunks prose at training granularity and prints the rewrite to stdout
    (--write edits in place with a backup). With --pairs/--out it runs in
    corpus mode: each pair's slop side is rewritten into a top-level "output"
    field, which `ai-style eval --field output` scores.
    """
    from stylebot import infer

    parsed = bundles_from_kwargs(kw, backends.RUN_PARAMS)
    run: backends.RunOpts = parsed["run"]
    # The manifest supplies checkpoint, base model, renderer and diet; flags win.
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
    **kw,
) -> None:
    """Where this pipeline stands: corpus (or view), splits, promoted run,
    detector, and the cross-checks between them (D52). Read-only and keyless.

    The command the documentation points at instead of carrying numbers (D52).
    Nothing is discovered: an artifact no flag names is "not given", and only
    `--pairs`/`--splits` fall back to the `--data-dir` precedence. The last
    block is the cross-check between the four (`stylebot.status`).
    """
    from stylebot import status as st

    view = bundles_from_kwargs(kw, {"view": draws.ViewOpts})["view"]

    resolved = config.resolve_data_dir(data_dir)
    if pairs_path is None and config.pairs_path(resolved).exists():
        pairs_path = config.pairs_path(resolved)
    if splits_path is None and config.splits_path(resolved).exists():
        splits_path = config.splits_path(resolved)

    report = st.collect(
        pairs=pairs_path,
        splits=splits_path,
        runs_dir=runs_dir,
        detector=detector_dir,
        blog_root=blog_root,
        draws_dir=view.draws_dir,
        policy=view.view_policy(),
        views_root=view.views_root,
    )
    if as_json:
        click.echo(json.dumps(asdict(report), indent=2, default=str))
        return
    click.echo(st.format_status(report))


@main.command("calibrate", cls=SectionedCommand)
@cc.calibrate_options()
def calibrate_cmd(**kw) -> None:
    """Cut the editor marker's P(slop) bucket edges from a detector's score
    quantiles on real pairs it never trained on (D105). Keyless; one local
    encode. Paste `thresholds` into the extension's `voiceMarker.thresholds`."""
    opts = bundles_from_kwargs(kw, cc.CALIBRATE_PARAMS)["opts"]
    cal = cc.calibrate(opts)
    click.echo(
        json.dumps(cal, indent=2) if opts.as_json else "\n".join(classify.format_calibration(cal))
    )


@main.command("build-view", cls=SectionedCommand)
@draws.view_options()
def build_view_cmd(**kw) -> None:
    """Build (or reuse) the view of --draws under --policy; print its path (D84)."""
    view = bundles_from_kwargs(kw, {"view": draws.ViewOpts})["view"].view()
    side = draws.read_sidecar(view) or {}
    click.echo(
        f"{side.get('records')} record(s) content_sha256={side.get('content_sha256')}", err=True
    )
    click.echo(view)


@main.command("validate-pairs", cls=SectionedCommand)
@click.argument("pairs_file", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--limit", type=int, default=20, show_default=True, help="Problem lines to print.")
@draws.view_options()
def validate_pairs_cmd(pairs_file: Path | None, limit: int, **kw) -> None:
    """Check a corpus against the record contract. Read-only (D36, D84).

    The gate `train` runs on its input: the message triple and the frozen
    system prompt, meta types and domains, and `record_id` uniqueness across
    every shard. The corpus is the view of --draws; PAIRS_FILE is a scratch
    file. A duplicate is a policy clause plus a rebuilt view, never an edit.
    """
    view = bundles_from_kwargs(kw, {"view": draws.ViewOpts})["view"]
    pairs_file = view.corpus_or_view(pairs_file, required=True)
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


@main.command("judge-fit", cls=SectionedCommand)
@fj.fit_options()
def judge_fit_cmd(**kw) -> None:
    """Compare two arms at CHUNK BOUNDARIES: does each rewrite fit its neighbours?

    The fifth signal (D97). The styler rewrites a document chunk by chunk, each
    call blind to the others; every glue point is where a renamed referent or a
    restarted metaphor shows up, and the other four signals each score one
    passage with no neighbours. This shows a judge the fixed prose either side
    of a span and both arms' middles, and asks which fits — forced choice, ties
    allowed, order randomised per case and recorded. Offline only: never wire
    it into inference. Run --field-a target --field-b slop as the control.
    """
    parsed = bundles_from_kwargs(kw, fj.FIT_PARAMS)
    text = fj.judge_fit_pass(parsed["fit"], parsed["view"], emit=click_emitter())
    if text:
        click.echo(text)


if __name__ == "__main__":
    main()
