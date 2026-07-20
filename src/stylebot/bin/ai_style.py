"""ai-style: the one-entry-point CLI (subcommands), thin over the libraries.

Per OVERVIEW "Interfaces": one command, subcommands `synth | split | train |
eval | serve` — not a scatter of loose scripts. Each subcommand is a thin
`click` wrapper that parses flags and calls a typed library function. `synth`
(Phase 2), `eval` (the offline scorer, `stylebot.eval`) and `serve` (the NDJSON
scoring sidecar, `stylebot.serve`) are built; `split` / `train` land with their
phases.

Phase 1's capture logger is `ai-style log` here, and is ALSO installed as the
standalone `ai-style-log` console script. Same group object, two spellings: the
subcommand keeps the surface consistent (and discoverable in `--help`), the
short script keeps the daily muscle memory. Neither is a fork — adding a command
to `ai_style_log.main` surfaces it under both.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import click

from stylebot import config, synth
from stylebot.bin import synth_cli
from stylebot.bin.ai_style_log import main as _log
from stylebot.lib import is_human_authored


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """stylebot tooling — capture, synthetic pairs, training, eval (subcommands)."""


# Phase 1 capture. The same group backs the standalone `ai-style-log` script;
# this is the consistent spelling, that one is the shorthand.
main.add_command(_log, name="log")


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
@synth_cli.synth_options()
@click.option("--tag", "tags", multiple=True, help="Extra provenance tag(s) added to meta.tags.")
def synth_cmd(
    files: tuple[Path, ...],
    blog_root: Path | None,
    data_dir: Path | None,
    field: str,
    max_level: int,
    select_all: bool,
    glob: str,
    tags: tuple[str, ...],
    **kw,
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
        **synth_cli.pop_chunk_kwargs(kw),
    )

    def _data_dir() -> Path:
        # Resolve the corpus location: explicit flag > $STYLEBOT_DATA_DIR. Refuse the
        # bare cwd default — synth costs money and grows the corpus; be explicit.
        # Deferred (callable) so the inspection modes need no data-dir at all.
        if data_dir is None and not config.get_key("STYLEBOT_DATA_DIR"):
            raise click.UsageError(
                "no --data-dir and $STYLEBOT_DATA_DIR unset; pass --data-dir explicitly "
                "(synth appends to the corpus and costs API spend — never the silent default)"
            )
        return config.resolve_data_dir(data_dir)

    # Resolve eagerly when the flag was given (inspection hints then name the real
    # pairs path); defer only to keep flag-less inspection runs guard-free.
    resolved = config.resolve_data_dir(data_dir) if data_dir is not None else _data_dir
    synth_cli.run_synth(targets, data_dir=resolved, extra_tags=tags, **kw)


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
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Write a self-contained HTML scores report — the pair browser: slop↔Dan side by side, judge/detector/Vale chips, generation covariates, facet dropdowns. Works keyless (judge badges show — without --judge).")
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
    can compare strategies with a number. --report is also the *pair browser* —
    a keyless run (no --judge) is the cheap way to eyeball a fresh synth batch:
    slop↔Dan side by side with generation covariates (model, strategy, prompt,
    reasoning), faceted and sortable. A Phase-4 styler run scores the same way
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


@main.command("serve")
@click.option(
    "--detector-model",
    "detector_model",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A trained voice-classifier artifact dir (head.json + meta.json); needs the "
    "pinned embedder (sentence-transformers) installed.",
)
def serve_cmd(detector_model: Path) -> None:
    """Score texts over stdin/stdout NDJSON (long-lived editor sidecar).

    Loads the detector once (the embedding model is the slow part), then serves
    {"id", "op": "score", "texts": [...]} -> {"id", "scores": [P(slop), ...]}
    one JSON object per line until EOF. {"op": "info"} returns the artifact's
    meta.json and doubles as the client's ready handshake. Protocol details:
    stylebot.serve.
    """
    from stylebot import classify, serve

    try:
        meta = classify.load_artifact_meta(detector_model)
        detector = classify.sklearn_detector(detector_model)
    except ImportError as exc:
        raise click.ClickException(
            f"--detector-model needs the embedder installed (sentence-transformers): {exc}"
        )
    except ValueError as exc:
        raise click.ClickException(str(exc))
    serve.serve_loop(detector, sys.stdin, sys.stdout, meta=meta)


def _echo_clf_metrics(metrics: dict, indent: str = "") -> None:
    """Echo a classify_train metrics dict: headline, per-fold Cs, provenance facet."""
    pa, auc = metrics["pairwise_accuracy"], metrics["auc"]
    label = (
        "holdout (unseen posts)"
        if metrics.get("mode") == "holdout"
        else f"cross-val ({metrics.get('n_splits')} POST folds)"
    )
    c_info = metrics.get("C") or {}
    c_note = (
        f"  [C: nested, per-fold {c_info['selected']}]"
        if c_info.get("mode") == "nested_cv"
        else (f"  [C fixed at {c_info['value']}]" if c_info else "")
    )
    click.echo(f"{indent}{label}: pairwise_acc={pa['mean']}±{pa['std']}  auc={auc['mean']}±{auc['std']}{c_note}")
    for name, facet in (metrics.get("by_provenance") or {}).items():
        fpa, fauc = facet["pairwise_accuracy"], facet["auc"]
        click.echo(
            f"{indent}  {name} ({facet['n_pairs']} pairs): pairwise_acc={fpa['mean']}±{fpa['std']}  "
            f"auc={fauc['mean']}±{fauc['std']}"
        )


@main.command("make-splits")
@click.option("--pairs", "pairs_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="The pairs.jsonl corpus whose posts get partitioned.")
@click.option("--out", "out_path", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Where to write splits.json (commit it next to the corpus).")
@click.option("--eval-frac", default=None, type=float, help="Fraction of posts pinned as the frozen eval stratum. [default: 0.2]")
@click.option("--detector-frac", default=None, type=float, help="Hash-rule share of NON-eval posts routed to the detector pool (rest -> styler). [default: 0.4]")
@click.option("--seed", default=0, show_default=True, type=int, help="Seed for the eval sample + the hash salt.")
@click.option("--allow-synthetic-eval", is_flag=True, help="Let synth-only posts into the eval stratum (default: eval is sampled from posts with >=1 REAL pair, so the frozen eval never consists of our own paraphrases).")
@click.option("--force", is_flag=True, help="Overwrite an existing splits file. Re-cutting the partition un-freezes eval — only do this knowingly.")
def make_splits_cmd(
    pairs_path: Path,
    out_path: Path,
    eval_frac: float | None,
    detector_frac: float | None,
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
    from stylebot import classify_train as ct
    from stylebot import splits as sp

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
    real_posts = sorted({
        ds.groups[s_row]
        for k, (s_row, _) in enumerate(ds.pair_rows)
        if not ds.pair_is_synthetic(k)
    })
    candidates = None if allow_synthetic_eval else real_posts
    if candidates is not None and not candidates:
        raise click.ClickException(
            "no posts with real pairs to sample eval from — pass --allow-synthetic-eval to override"
        )
    splits_doc = sp.make_splits(
        posts,
        eval_frac=eval_frac if eval_frac is not None else sp.DEFAULT_EVAL_FRAC,
        detector_frac=detector_frac if detector_frac is not None else sp.DEFAULT_DETECTOR_FRAC,
        seed=seed,
        eval_candidates=candidates,
    )
    sp.save_splits(splits_doc, out_path)

    roles = sp.summarize_roles(splits_doc, ds)
    for role in sp.ROLES:
        entry = roles[role]
        click.echo(
            f"{role:>8}: {entry['posts']} post(s), {entry['pairs']} pair(s) "
            f"({entry['pairs_real']} real / {entry['pairs_synthetic']} synthetic)"
        )
    for warning in sp.check_splits(splits_doc, ds):
        click.secho(f"WARNING: {warning}", fg="yellow", err=True)
    click.echo(f"wrote splits -> {out_path} (seed {seed}; eval frozen, rest by hash rule)")


@main.command("train-clf")
@click.option("--pairs", "pairs_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="The content-matched pairs.jsonl corpus to train on.")
@click.option("--out", "out_dir", required=True, type=click.Path(file_okay=False, path_type=Path), help="Artifact directory to write (head.json + meta.json).")
@click.option("--splits", "splits_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="The shared splits.json (make-splits): fit on the detector pool only, never touch the frozen eval posts, report a styler-posts holdout metric. Mutually exclusive with --holdout-*.")
@click.option("--test-size", default=0.25, show_default=True, type=float, help="Held-out POST fraction per CV split.")
@click.option("--n-splits", default=8, show_default=True, type=int, help="GroupShuffleSplit folds for the CV metric.")
@click.option("--C", "C", default=None, type=float, help="FIXED C override. Default (unset) selects C per training side by inner group-CV over C_GRID — do NOT sweep this flag against the printed metric; that is tuning on the test set.")
@click.option("--embed-model", default=None, help="Style backbone (sentence-transformers id); pinned into meta.json. [default: StyleDistance/styledistance]")
@click.option("--holdout-frac", default=0.0, show_default=True, type=float, help="Ad-hoc: hold out this fraction of POSTs and ship the head fit on the rest. Prefer --splits for the shared contract.")
@click.option("--holdout-seed", default=0, show_default=True, type=int, help="Seed for the deterministic by-POST holdout split.")
@click.option("--holdout-posts", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="Ad-hoc: file of POST source ids (one per line) to hold out. Prefer --splits.")
@click.option("--save-joblib/--no-save-joblib", default=False, show_default=True, help="Also dump the fitted sklearn estimator (redundant binary pickle; no code path reads it).")
@click.option("--dry-run", is_flag=True, help="Report data assembly (n_pairs, n_posts, class balance) without embedding or writing.")
def train_clf_cmd(
    pairs_path: Path,
    out_dir: Path,
    splits_path: Path | None,
    test_size: float,
    n_splits: int,
    C: float | None,
    embed_model: str | None,
    holdout_frac: float,
    holdout_seed: int,
    holdout_posts: Path | None,
    save_joblib: bool,
    dry_run: bool,
) -> None:
    """Train a voice classifier (linear head over a style embedding).

    Generic slop-vs-author trainer over a pairs.jsonl corpus: embeds both sides
    of every content-matched pair, fits a logistic regression, evaluates
    leakage-safely (split by POST, C selected by nested group-CV), and writes
    the head.json + meta.json artifact that `--detector-model` / `serve`
    consume. Needs the `classifier` extra (`uv add 'stylebot[classifier]'`).
    With --splits, honours the shared three-role contract (see make-splits).
    Free-standing extra positives are a library-level policy argument
    (`classify_train.train(extra_positives=…)`) — callers with a corpus policy
    wrap this, as livingthing's train-voice-clf does.
    """
    import json as _json

    from stylebot import classify_train as ct

    if splits_path is not None and (holdout_frac or holdout_posts is not None):
        raise click.UsageError("--splits is mutually exclusive with --holdout-frac/--holdout-posts")

    ds = ct.assemble_dataset(pairs_path)
    n_slop = sum(1 for label in ds.labels if label == ct.LABEL_SLOP)
    n_author = sum(1 for label in ds.labels if label == ct.LABEL_DAN)
    provenance = (
        f"; {ds.n_pairs_real} real / {ds.n_pairs_synthetic} synthetic"
        if ds.n_pairs_synthetic else ""
    )
    click.echo(
        f"{ds.n_pairs} content-matched pair(s) from {ds.n_posts} post(s)  "
        f"[{n_slop} slop / {n_author} author{provenance}]"
    )
    if ds.n_pairs == 0:
        raise click.ClickException(f"no content-matched pairs in {pairs_path}")
    if dry_run:
        if splits_path is not None:
            from stylebot import splits as sp

            splits_doc = sp.load_splits(splits_path)
            for role, entry in sp.summarize_roles(splits_doc, ds).items():
                click.echo(
                    f"  {role:>8}: {entry['posts']} post(s), {entry['pairs']} pair(s) "
                    f"({entry['pairs_real']} real / {entry['pairs_synthetic']} synthetic)"
                )
            for warning in sp.check_splits(splits_doc, ds):
                click.secho(f"WARNING: {warning}", fg="yellow", err=True)
        click.echo("[dry-run] no embedding, no artifact written")
        return

    holdout_list = None
    if holdout_posts is not None:
        holdout_list = [ln.strip() for ln in holdout_posts.read_text().splitlines() if ln.strip()]

    model_id = embed_model or ct.DEFAULT_EMBED_MODEL
    click.echo(f"embedding with {model_id} (downloads on first use) ...")
    try:
        out, metrics = ct.train(
            pairs_path, out_dir, test_size=test_size, n_splits=n_splits, C=C,
            embed_model=model_id, save_joblib=save_joblib,
            holdout_frac=holdout_frac, holdout_seed=holdout_seed, holdout_posts=holdout_list,
            splits_path=splits_path,
        )
    except ImportError as exc:
        raise click.ClickException(str(exc))
    except ValueError as exc:
        raise click.ClickException(str(exc))

    _echo_clf_metrics(metrics)
    if "styler_holdout" in metrics:
        click.echo("styler posts (unseen by this head — the reward-relevant number):")
        _echo_clf_metrics(metrics["styler_holdout"], indent="  ")

    # The artifact is the source of truth for what shipped: surface its own
    # record of the chosen C and any danger warnings.
    meta = _json.loads((out / "meta.json").read_text(encoding="utf-8"))
    head_c = meta.get("head_C") or {}
    if head_c:
        chosen = head_c.get("value")
        click.echo(f"shipped head C: {chosen} ({head_c.get('mode')})")
    for warning in (meta.get("split") or {}).get("warnings") or []:
        click.secho(f"WARNING: {warning}", fg="yellow", err=True)
    files = "head.json, meta.json" + (", model.joblib" if save_joblib else "")
    click.echo(f"wrote artifact -> {out}/ ({files})")


if __name__ == "__main__":
    main()
