"""ai-style: the one-entry-point CLI (subcommands), thin over the libraries.

Per OVERVIEW "Interfaces": one command, subcommands — not a scatter of loose
scripts. Each subcommand is a thin `click` wrapper that parses flags and calls a
typed library function: `log` (Phase 1 capture), `synth` (Phase 2), `make-splits`
(the shared by-post splits contract), `check-splits` (its health check) and
`extend-eval` (the add-only re-cut of the frozen eval stratum),
`train-clf` (the voice classifier),
`train` (Phase 3 LoRA SFT), `run` (Phase 4 inference), `eval` (the offline
scorer, `stylebot.eval`), `export-table` (scores -> tidy CSV), `merge-adapter`
(adapter + base -> one merged checkpoint, `stylebot.export`), `serve` (the
NDJSON editor sidecar, `stylebot.serve`), `validate-pairs` (the record
contract on demand) and `status` (`stylebot.status` — what the corpus holds,
which run serves, which detector scored it, and where those three disagree;
the numbers documentation is no longer allowed to carry, D52).

Phase 1's capture logger is `ai-style log` — a subcommand like every other, not
a separate console script. It shipped as a standalone `ai-style-log` until
2026-07-20; that alias is gone, one canonical spelling.

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

import datetime
import json
import sys
from dataclasses import replace
from functools import cache, partial
from pathlib import Path

import click

# `classify_train`, `splits` and `train` are imported at module scope so click
# can own their defaults directly (`default=tr.DEFAULT_LORA_RANK`), replacing a
# triple declaration: `default=None`, a hand-written `[default: X]` in the help
# string, and an `x if x is not None else DEFAULT_X` at the call site. All three
# import free (tinker/sklearn/sentence-transformers stay lazy) and reuse the
# substrate `synth` already pulls in: 344 µs measured, against 4.5 ms for click.
# `infer` stays lazy — it owns no default declared here.
#
# `synth`/`synth_cli` are here for the same reason (`synth_options()` is a
# decorator; a default cannot be lazy). D49 has the `--help` numbers and why
# neither a lazy-command group nor deferring `asyncio` earned its keep.
from stylebot import backends, config, pairs, synth, synth_cli
from stylebot import classify_train as ct
from stylebot import eval as ev
from stylebot import splits as sp
from stylebot import train as tr
from stylebot.backends import BACKEND_PARAMS, BackendOpts, backend_options
from stylebot.bin.ai_style_data import main as _data
from stylebot.bin.ai_style_log import main as _log
from stylebot.cli_options import (
    ErrorTranslatingGroup,
    SectionedCommand,
    bundle_options,
    bundles_from_kwargs,
    click_emitter,
)
from stylebot.errors import RunError
from stylebot.jsonl import append_jsonl
from stylebot.lib import is_human_authored


@click.group(
    cls=ErrorTranslatingGroup, context_settings={"help_option_names": ["-h", "--help"]}
)
def main() -> None:
    """stylebot tooling — capture, synthetic pairs, training, eval (subcommands)."""


# Phase 1 capture — the group defined in bin/ai_style_log.py, mounted here as
# its only entry point.
main.add_command(_log, name="log")

# One-off data migrations — hidden; see CLAUDE.md's carve-out paragraph.
main.add_command(_data, name="data")


# ---------------------------------------------------------------------------
# The styler backend: one surface, two commands.
#
# `run` (batch, one file) and `serve` (long-lived editor sidecar) arm the SAME
# styler from the SAME flags — the sidecar's rewrite op is `run` held open. The
# surface itself (`BackendOpts`, `build_styler`, the token-scaling rule and the
# manifest diet) is `stylebot.backends`, a LIBRARY: it lived here until
# 2026-09-03, where livingthing could not import it, reimplemented it, and lost
# `max_tokens` in the copy. Nothing of it is left here — not even the exit-code
# translation, which `ErrorTranslatingGroup` now does for the whole tree.
#
# The per-command differences below are deliberate and stay per-command:
# `run` defaults --backend to tinker (serve omits it for score-only serving),
# `serve` defaults --temperature hotter, and serve's --detector-model is
# required (it IS the service) where run's only feeds --best-of.
# ---------------------------------------------------------------------------


MANIFEST_HELP = (
    "A training-run manifest to serve: supplies --sampler-path, --base-model, --renderer and "
    "the chunk diet (--max-chunk-chars/--inline-headings) the adapter was TRAINED on. Explicit "
    "flags win over it. Serving at the wrong diet is out-of-distribution — the styler falls back "
    "to copying its input — so prefer this over retyping four flags."
)


def _load_manifest(manifest_path: Path | None) -> dict:
    """A training manifest as a dict, or `{}` when none was named."""
    if manifest_path is None:
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise click.ClickException(f"{manifest_path}: not valid JSON ({exc})") from exc


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


def _document_pass(pairs_path: Path, out: Path, doc_opts, score_fields):
    """Rewrite whole eval-stratum documents; return what to score instead.

    Returns `(pairs_path, fields, prune_orphans, chunk_scale)` for the scoring
    pass that follows: the output JSONL is a DIFFERENT corpus from `pairs.jsonl`,
    so orphan pruning must be off (every row would read as orphaned) and `output`
    joins the scored fields. `chunk_scale` is the manifest's serving diet, handed
    to `eval.chunked_detector` so the documents are scored at the scale the
    detector head was fit on rather than as one 25k-char embedding.
    """
    if doc_opts.documents_manifest is None:
        raise click.UsageError("--documents needs --documents-manifest (the adapter to serve)")
    manifest = _load_manifest(doc_opts.documents_manifest)
    styler = backends.build_styler(
        backends.opts_from_manifest(manifest, BackendOpts(backend="tinker"))
    )
    max_chunk_chars, inline_headings = backends.serving_diet(manifest)

    splits_file = config.splits_path(pairs_path.parent)
    splits_doc = sp.load_splits(splits_file) if splits_file.exists() else None
    role = None if doc_opts.documents_role == "all" else doc_opts.documents_role
    if role is not None and splits_doc is None:
        raise click.UsageError(
            f"--documents-role {role} needs a splits contract; none at {splits_file}"
        )

    docs_out = doc_opts.documents_out or (out.parent / "documents.jsonl")

    def _doc_progress(i: int, total: int) -> None:
        click.echo(f"  ... {i}/{total} documents rewritten", err=True)

    result = ev.rewrite_documents_file(
        pairs_path,
        docs_out,
        styler,
        splits=splits_doc,
        role=role,
        max_chunk_chars=max_chunk_chars,
        inline_headings=inline_headings,
        on_progress=_doc_progress,
    )
    click.echo(
        f"rewrote {result.written} document(s) ({result.n_chunks} chunks, "
        f"{result.n_kept_input} kept verbatim) -> {docs_out}"
    )
    for capture_id, message in result.errors[:10]:
        click.secho(f"  {capture_id}: {message}", fg="yellow", err=True)
    fields = (*score_fields, *(() if "output" in score_fields else ("output",)))
    chunk_scale = {"max_chunk_chars": max_chunk_chars, "inline_headings": inline_headings}
    return docs_out, fields, False, chunk_scale


def _echo_summary(summary: dict) -> None:
    for line in ev.format_summary(summary):
        click.echo(line)


def _echo_verdict(verdict: dict, a_path: Path, b_path: Path) -> None:
    click.echo("")
    for line in ev.format_verdict(verdict, a_path, b_path):
        if line.lstrip().startswith("WARNING:"):
            click.secho(line, fg="yellow", err=True)
        else:
            click.echo(line)


@main.command("eval")
@click.option(
    "--pairs",
    "pairs_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
    help="The pairs corpus to score (slop + Dan sides of every pair): a "
    "pairs.jsonl or a pairs/ shard directory.",
)
@click.option(
    "--field",
    "fields",
    type=click.Choice(["slop", "target"]),
    multiple=True,
    default=(),
    help="Which side(s) to score (repeatable; default both). slop=messages[1], "
    "target=Dan/messages[2].",
)
@click.option(
    "--judge/--no-judge",
    default=False,
    show_default=True,
    help="Run the LLM judge (needs $OPENROUTER_API_KEY); off = Vale + null detector, no spend.",
)
@click.option(
    "--judge-model",
    default="anthropic/claude-opus-4-8",
    show_default=True,
    help="OpenRouter model id for the judge.",
)
@click.option(
    "--detector-model",
    "detector_model",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="A trained voice-classifier artifact dir (head.json + meta.json) to score "
    "detector.score; needs the pinned embedder (sentence-transformers) installed.",
)
@click.option(
    "--vale-config",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Vale config (.vale.ini); omitted = Vale's own discovery (or Vale absent -> skipped).",
)
@click.option(
    "--max-workers",
    default=8,
    show_default=True,
    type=int,
    help="Concurrent judge calls (asyncio). Vale is a batched pre-pass, not a worker (D59).",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="scores.jsonl to append (default: <pairs-dir>/scores.jsonl). Resumable — scored ids are "
    "skipped.",
)
@click.option(
    "--summary",
    "summary_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also write the aggregate summary JSON here.",
)
@click.option(
    "--by",
    "by",
    default=None,
    help="Facet the summary by a meta key, e.g. slop_strategy or generator.",
)
@click.option("--limit", type=int, default=None, help="Cap pairs scored (smoke / cost control).")
@click.option(
    "--prune-orphans/--no-prune-orphans",
    default=True,
    show_default=True,
    help="Drop score records whose pair is no longer in --pairs (re-cut or regenerated), so "
    "every summary and exported table counts live pairs only. Turn OFF only when --out belongs "
    "to a DIFFERENT corpus, where every record reads as orphaned.",
)
@click.option(
    "--report",
    "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a self-contained HTML scores report — the pair browser: slop↔Dan side by side, "
    "judge/detector/Vale chips, generation covariates, facet dropdowns. Works keyless (judge "
    "badges show — without --judge).",
)
@click.option(
    "--report-max-rows",
    default=2000,
    show_default=True,
    type=int,
    help="Cap rows in the HTML report (0 = all).",
)
@click.option(
    "--facet-by",
    "facet_by",
    default="slop_strategy",
    show_default=True,
    help="Group the HTML report headline by this meta covariate (e.g. reasoning_effort, "
    "prompt_id, generator).",
)
@click.option(
    "--sample",
    "sample_n",
    type=int,
    default=None,
    help="Print N random scored pairs (slop vs Dan + scores) to stdout.",
)
@ev.verdict_options()
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
    prune_orphans: bool,
    report_path: Path | None,
    report_max_rows: int,
    facet_by: str,
    sample_n: int | None,
    **kw,
) -> None:
    """Score a pairs corpus across the eval signals (offline, batched).

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
    parsed = bundles_from_kwargs(kw, ev.VERDICT_PARAMS)
    verdict_opts, doc_opts = parsed["verdict"], parsed["documents"]

    out = out_path or (pairs_path.parent / "scores.jsonl")
    score_fields = fields or ("slop", "target")

    judge_fn = ev.openrouter_judge_async(model=judge_model) if judge else None

    def _progress(i: int, total: int) -> None:
        if i == 1 or i % 25 == 0 or i == total:
            click.echo(f"  ... {i}/{total} scored", err=True)

    chunk_scale = None
    if doc_opts.documents:
        # --documents samples the adapter first, then scores what came back in
        # the same process (D59).
        pairs_path, score_fields, prune_orphans, chunk_scale = _document_pass(
            pairs_path, out, doc_opts, score_fields
        )

    # A trained voice-classifier artifact, if given, supplies a real detector
    # (P(slop)); else the null detector. The embedder is rebuilt lazily from
    # the artifact's pinned meta.embed_model — heavy imports stay inside
    # classify.
    detector_fn = ev.null_detector
    if detector_model is not None:
        from stylebot import classify

        try:
            detector_fn = classify.sklearn_detector(detector_model)
        except ImportError as exc:
            raise click.ClickException(
                f"--detector-model needs the embedder installed (sentence-transformers): {exc}"
            ) from exc
    if chunk_scale is not None:
        # Score a document at the scale it was served at, not whole (D54).
        detector_fn = ev.chunked_detector(detector_fn, **chunk_scale)

    result = ev.score_pairs_file(
        pairs_path,
        out,
        fields=score_fields,
        judge=judge_fn,
        detector=detector_fn,
        vale_config=vale_config,
        max_workers=max_workers,
        limit=limit,
        prune_orphans=prune_orphans,
        allow_mixed_detector=verdict_opts.allow_mixed_detector,
        on_progress=_progress,
    )
    # Loud, never silent: the prune REWROTE a file the operator may have
    # committed, and a big count usually means --out belongs to another corpus.
    if result.pruned_orphans:
        click.echo(
            f"pruned {result.pruned_orphans} orphan score record(s) from {out} "
            f"— their pair is no longer in {pairs_path}",
            err=True,
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
    _echo_summary(summary)
    if by is not None:
        click.echo(f"by {by}:")
        for facet, fld in summary["by"].items():
            parts = ", ".join(f"{fn}={a['mean_judge_score']}" for fn, a in fld.items())
            click.echo(f"  {facet}: {parts}")
    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        click.echo(f"wrote summary -> {summary_path}")

    if verdict_opts.compare_path is not None:
        verdict = ev.compare_scores(
            out,
            verdict_opts.compare_path,
            field=verdict_opts.compare_field,
            signal=verdict_opts.compare_signal,
            cluster_by=verdict_opts.cluster_by or None,
            allow_cross_detector=verdict_opts.allow_cross_detector,
        )
        _echo_verdict(verdict, out, verdict_opts.compare_path)

    # Read-only visualisations over the (just-written) scores.jsonl + the corpus.
    # No extra scoring — re-rendering an already-scored corpus is a no-op pass.
    if sample_n is not None or report_path is not None:
        from stylebot import report

        if sample_n is not None:
            click.echo(report.format_scores_sample(out, pairs_path, sample_n, fields=score_fields))
        if report_path is not None:
            written = report.render_scores_report(
                out,
                pairs_path,
                report_path,
                fields=score_fields,
                max_rows=(None if report_max_rows == 0 else report_max_rows),
                facet_by=facet_by,
            )
            click.echo(f"wrote report -> {written}")


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
    help=MANIFEST_HELP,
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
    from stylebot import classify, infer, serve

    opts = backends.opts_from_manifest(
        _load_manifest(manifest_path),
        replace(bundles_from_kwargs(kw, BACKEND_PARAMS)["opts"], detector_model=detector_model),
    )

    try:
        meta = classify.load_artifact_meta(detector_model)
        detector = classify.sklearn_detector(detector_model)
    except ImportError as exc:
        raise click.ClickException(
            f"--detector-model needs the embedder installed (sentence-transformers): {exc}"
        ) from exc

    chunk_chars = opts.chunk_chars()

    rewriter = None
    if opts.backend is not None:
        # Build the styler LAZILY: the tinker client dials out at construction,
        # and a flaky network at editor start must not take scoring down with it.
        # `cache` is what makes it lazy-once — the first rewrite op pays, later
        # ones reuse the client.
        @cache
        def make_styler():
            return backends.build_styler(opts)

        scorer = infer.detector_scorer(detector)

        def rewrite_op(text, *, context=None, best_of=4):
            return infer.rewrite_candidates(
                text,
                make_styler(),
                context=context,
                best_of=best_of,
                scorer=scorer,
                max_chunk_chars=chunk_chars,
                inline_headings=opts.headings(),
            )

        rewriter = rewrite_op

    prefer = None
    if preferences_path is not None:

        def prefer_op(decision: dict) -> None:
            record = {
                "ts": datetime.datetime.now(datetime.UTC).isoformat(),
                **decision,
            }
            append_jsonl(preferences_path, record)

        prefer = prefer_op

    serve.serve_loop(detector, sys.stdin, sys.stdout, meta=meta, rewriter=rewriter, prefer=prefer)


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
    click.echo(
        f"{indent}{label}: pairwise_acc={pa['mean']}±{pa['std']}  "
        f"auc={auc['mean']}±{auc['std']}{c_note}"
    )
    for name, facet in (metrics.get("by_provenance") or {}).items():
        fpa, fauc = facet["pairwise_accuracy"], facet["auc"]
        click.echo(
            f"{indent}  {name} ({facet['n_pairs']} pairs): "
            f"pairwise_acc={fpa['mean']}±{fpa['std']}  "
            f"auc={fauc['mean']}±{fauc['std']}"
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
def check_splits_cmd(splits_path: Path, blog_root: Path, pairs_path: Path | None) -> None:
    """Health-check the splits contract: eval posts present, strata big enough.

    The read-only counterpart of `make-splits`, and the thing to run after any
    `post-op move`: a renamed post silently changes role, which is how two
    frozen-eval posts leaked into the styler/detector pools (2026-09-03). Exits
    non-zero when a pinned eval post is no longer on disk under --blog-root;
    small-stratum findings stay warnings.
    """
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
        roles = sp.summarize_roles(splits_doc, ds)
        for role in sp.ROLES:
            entry = roles[role]
            top = (
                f"; top post {entry['top_post']} {entry['top_post_share']:.0%}"
                if entry["top_post"]
                else ""
            )
            click.echo(
                f"{role:>8}: {entry['posts']} post(s), {entry['pairs']} pair(s) "
                f"({entry['pairs_real']} real / {entry['pairs_synthetic']} synthetic){top}"
            )
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


@main.command("train-clf")
@click.option(
    "--pairs",
    "pairs_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
    help="The content-matched pairs corpus to train on (pairs.jsonl or a pairs/ directory).",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Artifact directory to write (head.json + meta.json).",
)
@click.option(
    "--splits",
    "splits_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="The shared splits.json (make-splits): fit on the detector pool only, never touch the "
    "frozen eval posts, report a styler-posts holdout metric. Mutually exclusive with "
    "--holdout-*.",
)
@click.option(
    "--blog-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Tree the pinned eval posts must still exist in; a renamed one fails the run (D34).",
)
@click.option(
    "--test-size",
    default=ct.DEFAULT_TEST_SIZE,
    show_default=True,
    type=float,
    help="Held-out POST fraction per CV split.",
)
@click.option(
    "--n-splits",
    default=ct.DEFAULT_N_SPLITS,
    show_default=True,
    type=int,
    help="GroupShuffleSplit folds for the CV metric.",
)
@click.option(
    "--C",
    "C",
    default=None,
    type=float,
    help="FIXED C override. Default (unset) selects C per training side by inner group-CV over "
    "C_GRID — do NOT sweep this flag against the printed metric; that is tuning on the test set.",
)
@click.option(
    "--embed-model",
    default=ct.DEFAULT_EMBED_MODEL,
    show_default=True,
    help="Style backbone (sentence-transformers id); pinned into meta.json.",
)
@click.option(
    "--holdout-frac",
    default=0.0,
    show_default=True,
    type=float,
    help="Ad-hoc: hold out this fraction of POSTs and ship the head fit on the rest. Prefer "
    "--splits for the shared contract.",
)
@click.option(
    "--holdout-seed",
    default=0,
    show_default=True,
    type=int,
    help="Seed for the deterministic by-POST holdout split.",
)
@click.option(
    "--holdout-posts",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Ad-hoc: file of POST source ids (one per line) to hold out. Prefer --splits.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report data assembly (n_pairs, n_posts, class balance) without embedding or writing.",
)
def train_clf_cmd(
    pairs_path: Path,
    out_dir: Path,
    splits_path: Path | None,
    blog_root: Path | None,
    test_size: float,
    n_splits: int,
    C: float | None,
    embed_model: str,
    holdout_frac: float,
    holdout_seed: int,
    holdout_posts: Path | None,
    dry_run: bool,
) -> None:
    """Train a voice classifier (linear head over a style embedding).

    Generic slop-vs-author trainer over a pairs corpus: embeds both sides of every
    content-matched pair, fits a logistic regression, evaluates leakage-safely (by
    POST, C by nested group-CV), and writes the head.json + meta.json artifact that
    `--detector-model` / `serve` consume. Needs the `classifier` extra. Extra
    positives are a library argument (`classify_train.train(extra_positives=…)`).
    """
    if splits_path is not None and (holdout_frac or holdout_posts is not None):
        raise click.UsageError("--splits is mutually exclusive with --holdout-frac/--holdout-posts")

    ds = ct.assemble_dataset(pairs_path)
    n_slop = sum(1 for label in ds.labels if label == ct.LABEL_SLOP)
    n_author = sum(1 for label in ds.labels if label == ct.LABEL_DAN)
    provenance = (
        f"; {ds.n_pairs_real} real / {ds.n_pairs_synthetic} synthetic"
        if ds.n_pairs_synthetic
        else ""
    )
    click.echo(
        f"{ds.n_pairs} content-matched pair(s) from {ds.n_posts} post(s)  "
        f"[{n_slop} slop / {n_author} author{provenance}]"
    )
    if ds.n_pairs == 0:
        raise click.ClickException(f"no content-matched pairs in {pairs_path}")
    if dry_run:
        if splits_path is not None:
            splits_doc = sp.load_splits(splits_path)
            if blog_root is not None:
                sp.require_pinned_posts(splits_doc, blog_root)
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

    click.echo(f"embedding with {embed_model} (downloads on first use) ...")
    out, metrics = ct.train(
        pairs_path,
        out_dir,
        test_size=test_size,
        n_splits=n_splits,
        C=C,
        embed_model=embed_model,
        holdout_frac=holdout_frac,
        holdout_seed=holdout_seed,
        holdout_posts=holdout_list,
        splits_path=splits_path,
        blog_root=blog_root,
    )

    _echo_clf_metrics(metrics)
    if "styler_holdout" in metrics:
        click.echo("styler posts (unseen by this head — the reward-relevant number):")
        _echo_clf_metrics(metrics["styler_holdout"], indent="  ")

    # The artifact is the source of truth for what shipped: surface its own
    # record of the chosen C and any danger warnings.
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    head_c = meta.get("head_C") or {}
    if head_c:
        chosen = head_c.get("value")
        click.echo(f"shipped head C: {chosen} ({head_c.get('mode')})")
    for warning in (meta.get("split") or {}).get("warnings") or []:
        click.secho(f"WARNING: {warning}", fg="yellow", err=True)
    click.echo(f"wrote artifact -> {out}/ (head.json, meta.json)")


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
    help="Assemble, write the manifest preview + token/cost estimate, and stop. No key, no spend.",
)
@click.option("--yes", "assume_yes", is_flag=True, help="Skip the pre-spend confirmation prompt.")
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
    **kw,
) -> None:
    """Phase 3: LoRA-SFT the styler on Tinker (stylebot.train.run_training).

    Assembles the corpus (styler-role posts only under --splits, near-copy pairs
    KEPT), cuts a by-POST val split, writes the manifest, runs the Tinker recipe.
    Needs the `trainer` extra and TINKER_API_KEY; --dry-run needs neither.
    """
    pairs_path = config.pairs_path(data_dir)
    if not pairs_path.exists():
        raise click.ClickException(f"no corpus (pairs/ or pairs.jsonl) in {data_dir}")

    kwargs = dict(
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
        click.echo(f"[dry-run] manifest preview -> {preview.manifest_path}")
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
@click.option(
    "--best-of",
    default=1,
    show_default=True,
    type=int,
    help="Sample N candidates per chunk and keep the most Dan-scored (needs --detector-model).",
)
@click.option(
    "--strict-anchors",
    is_flag=True,
    help="Also require inline code spans and inline math to survive a rewrite, not just links "
    "and citations. Measured over the corpus this roughly triples the rate at which every "
    "candidate is rejected and the input is kept verbatim — hence off by default.",
)
@click.option(
    "--anchor-log",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Append per-chunk anchor accounting to this JSONL (anchor keys and lengths, never "
    "candidate text). The headline counts say whether a run went wrong; this says how the "
    "model behaves, which is what decides whether masking is worth building at all "
    "(D32, D33).",
)
@click.option(
    "--write",
    "write_in_place",
    is_flag=True,
    help="Rewrite FILE in place (a .bak backup is left beside it). Default prints to stdout.",
)
@click.option(
    "--pairs",
    "pairs_path",
    type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
    help="Corpus mode: rewrite the slop side of every pair in this pairs.jsonl instead of a "
    "file, writing an output-bearing JSONL for `ai-style eval --field output`. This is how a "
    "trained adapter gets a number rather than an impression. Needs --out.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="[--pairs] Where the output-bearing JSONL goes. Appended to and resumable: ids already "
    "present are skipped, so an interrupted sweep continues rather than re-paying for them.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="[--pairs] Cap how many pairs are rewritten.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=MANIFEST_HELP,
)
@backend_options(defaults=[BackendOpts(backend="tinker")])
def run_cmd(
    file: Path | None,
    best_of: int,
    strict_anchors: bool,
    anchor_log: Path | None,
    write_in_place: bool,
    pairs_path: Path | None,
    out_path: Path | None,
    limit: int | None,
    manifest_path: Path | None,
    **kw,
) -> None:
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

    # The manifest is the committed reproducibility record: it carries the
    # checkpoint, the base model, the renderer AND the chunk diet the adapter
    # was trained on. Explicit flags still win over every one of them.
    opts: BackendOpts = backends.opts_from_manifest(
        _load_manifest(manifest_path),
        bundles_from_kwargs(kw, BACKEND_PARAMS)["opts"],
    )
    if pairs_path is not None:
        if file is not None:
            raise click.UsageError("--pairs rewrites a corpus; pass it or a FILE, not both")
        if out_path is None:
            raise click.UsageError("--pairs needs --out (the output-bearing JSONL to write)")
    elif out_path is not None or limit is not None:
        raise click.UsageError("--out/--limit only apply in corpus mode (--pairs)")

    styler = backends.build_styler(opts)

    scorer = None
    if best_of > 1:
        if opts.detector_model is None:
            raise click.UsageError("--best-of needs --detector-model for the p_dan scoring")
        from stylebot.classify import sklearn_detector

        scorer = infer.detector_scorer(sklearn_detector(opts.detector_model))

    if pairs_path is not None:
        def _progress(i: int, total: int) -> None:
            if i == 1 or i % 10 == 0 or i == total:
                click.echo(f"  ... {i}/{total} rewritten", err=True)

        written = infer.rewrite_pairs_file(
            pairs_path, out_path, styler,
            limit=limit, best_of=best_of, scorer=scorer, on_progress=_progress,
        )
        click.echo(f"rewrote {written} pair(s) -> {out_path}", err=True)
        return

    text = file.read_text(encoding="utf-8") if file else click.get_text_stream("stdin").read()
    result = infer.rewrite_text(
        text,
        styler,
        max_chunk_chars=opts.chunk_chars(),
        best_of=best_of,
        scorer=scorer,
        inline_headings=opts.headings(),
        inline_anchors=strict_anchors,
    )
    if scorer is not None:
        for line in result.decisions:
            click.echo(f"  {line}", err=True)
    if anchor_log is not None:
        n = infer.write_anchor_audit(anchor_log, result, source=str(file) if file else None)
        click.echo(f"  {n} chunk(s) of anchor accounting -> {anchor_log}", err=True)
    click.echo(
        f"[{result.n_chunks} chunk(s), {result.n_candidates} sample(s)"
        + (f", {result.n_kept_input} kept as input" if result.n_kept_input else "")
        + (
            # Broken out by cause: losing a link wants a better sample, inventing
            # one wants a model that stops making things up.
            f", {result.n_anchor_rejected} sample(s) rejected for anchor damage "
            f"({result.n_anchor_lost} lost, {result.n_anchor_invented} invented)"
            if result.n_anchor_rejected
            else ""
        )
        + (
            # Reported apart from anchor loss on purpose: this one usually means
            # the token budget, not the sample.
            f", {result.n_truncated} truncated (raise --max-tokens?)" if result.n_truncated else ""
        )
        + "]",
        err=True,
    )
    if write_in_place and file:
        file.with_suffix(file.suffix + ".bak").write_text(text, encoding="utf-8")
        file.write_text(result.text, encoding="utf-8")
        click.echo(f"rewrote {file} (backup at {file}.bak)", err=True)
    else:
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
def status_cmd(
    data_dir: Path | None,
    pairs_path: Path | None,
    splits_path: Path | None,
    runs_dir: Path | None,
    detector_dir: Path | None,
    blog_root: Path | None,
    as_json: bool,
) -> None:
    """Where this pipeline stands: corpus, splits, promoted run, detector.

    The command the documentation points at instead of carrying numbers it
    cannot keep true. Read-only, keyless, and free.

    Nothing is discovered: stylebot does not know where your blog, runs or
    detector artifact are, so `--runs` / `--detector-model` / `--blog-root` have
    no defaults and unnamed artifacts are reported as "not given" rather than
    guessed at. `--pairs`/`--splits` fall back to the usual `--data-dir` >
    `$STYLEBOT_DATA_DIR` > `./_training_pairs` precedence. A policy mirror like
    `dan-style status` is this command with all five filled in.

    The last block is the cross-check: a detector fit under a different
    `splits_sha256` than `splits.json` now carries, a promoted run that trained
    on a different partition, plus the `check-splits` danger report.
    """
    from dataclasses import asdict

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
    )
    if as_json:
        click.echo(json.dumps(asdict(report), indent=2, default=str))
        return
    click.echo(st.format_status(report))


@main.command("validate-pairs")
@click.argument("pairs_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--fix",
    is_flag=True,
    help="Drop duplicate record_ids (keeping the first of each) and rewrite the file.",
)
@click.option(
    "--backup/--no-backup",
    default=True,
    show_default=True,
    help="With --fix, keep a .bak beside the corpus before rewriting.",
)
@click.option("--limit", type=int, default=20, show_default=True, help="Problem lines to print.")
def validate_pairs_cmd(pairs_file: Path, fix: bool, backup: bool, limit: int) -> None:
    """Check a corpus against the record contract, and optionally de-duplicate.

    The gate `train` runs on its input, exposed so it can be run on demand: the
    message triple and the frozen system prompt, meta types and domains, and
    `record_id` uniqueness across the whole corpus. `--fix` addresses only the
    last of those — a duplicate is mechanically removable, a malformed record
    is not.

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

    if fix:
        report = pairs.dedupe_pairs_file(pairs_file, apply=True, backup=backup)
        click.echo(
            f"deduped: dropped {report.dropped} of {report.total} record(s) across "
            f"{report.duplicate_ids} duplicated id(s); {report.kept} kept"
            + (f"; backup at {report.backup}" if report.backup else "")
        )
        remaining = pairs.validate_pairs_file(pairs_file)
        click.echo(f"re-validated: {len(remaining)} problem line(s)")
        if remaining:
            raise SystemExit(1)
    elif problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
