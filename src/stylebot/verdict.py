"""The verdict — a paired comparison of two scored runs, clustered by post (D40).

`compare_scores` joins two `scores.jsonl` on the pair id with a cluster
bootstrap over posts; `paired_delta` is that interval on bare sequences. The
option bundles and `format_*` renderers serve both CLIs; `report_eval` is
the tail of the `eval` pass (D78). Cross-detector and circular verdicts are
refused (D39).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import click

from stylebot import report
from stylebot.classify import DETECTOR_IDENTITY_KEYS
from stylebot.cli_options import Emitter, opt, silent
from stylebot.draws import ViewOpts
from stylebot.errors import ConfigError
from stylebot.eval import DEFAULT_JUDGE_MODEL, DETECTOR_KINDS, SCHEMA_VERSION
from stylebot.scoring import (
    DEFAULT_MAX_WORKERS,
    _identity_label,
    assert_single_detector,
    load_scores,
)

#: How to pull one scalar out of a scored field; an unavailable Vale is `null`,
#: not 0, so it drops out rather than scoring perfect.
SIGNAL_EXTRACTORS: dict[str, Callable[[dict], float | None]] = {
    "detector": lambda signals: (signals.get("detector") or {}).get("score"),
    "judge": lambda signals: (signals.get("judge") or {}).get("score"),
    "vale": lambda signals: (
        (signals.get("vale") or {}).get("alerts")
        if (signals.get("vale") or {}).get("available")
        else None
    ),
}


def paired_delta(
    a: Sequence[float | None],
    b: Sequence[float | None],
    *,
    clusters: Sequence[object] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict | None:
    """Mean of `a - b` with a percentile bootstrap interval over `clusters` (D40).

    Aligned sequences; a None on either side drops out. Returns None with
    fewer than two usable differences; the interval is None when they all fall
    in one cluster.
    """
    clustered = clusters is not None
    if clusters is None:
        clusters = range(len(a))
    diffs: list[float] = []
    labels: list[object] = []
    for x, y, cluster in zip(a, b, clusters, strict=True):
        if x is None or y is None:
            continue
        diffs.append(float(x) - float(y))
        labels.append(cluster)
    if len(diffs) < 2:
        return None

    grouped: dict[object, list[float]] = {}
    for label, diff in zip(labels, diffs, strict=True):
        grouped.setdefault(label, []).append(diff)
    point = mean(diffs)
    base = {
        "mean": point,
        "n": len(diffs),
        "n_clusters": len(grouped),
    }

    if len(grouped) < 2:
        return {**base, "ci_low": None, "ci_high": None, "method": "single-cluster", "n_boot": 0}

    keys = list(grouped)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        drawn: list[float] = []
        for _ in keys:
            drawn.extend(grouped[keys[rng.randrange(len(keys))]])
        if drawn:
            means.append(mean(drawn))
    means.sort()
    lo = means[max(0, int(alpha / 2 * len(means)) - 1)]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return {
        **base,
        "ci_low": lo,
        "ci_high": hi,
        "method": "cluster_bootstrap" if clustered else "bootstrap",
        "n_boot": len(means),
        "alpha": alpha,
    }


@dataclass(frozen=True)
class EvalOpts:
    """The scoring surface of `eval` — what `scoring.run_eval` takes (D78)."""

    cli_section = "Scoring"

    pairs_path: Path | None = opt(
        None,
        "--pairs",
        "pairs_path",
        type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
        help="A scratch corpus to score (slop + Dan sides of every pair): a pairs.jsonl "
        "or a shard directory. The corpus itself is --draws.",
    )
    fields: tuple[str, ...] = opt(
        (),
        "--field",
        "fields",
        type=click.Choice(["slop", "target", "output"]),
        multiple=True,
        help="Which side(s) to score (repeatable; default both). slop=messages[1], "
        "target=Dan/messages[2], output=a styler's rewrite (`run --pairs --out`).",
    )
    judge: bool = opt(
        False,
        "--judge/--no-judge",
        show_default=True,
        help="Run the LLM judge (needs $OPENROUTER_API_KEY); off = Vale + null detector, no spend.",
    )
    judge_model: str = opt(
        DEFAULT_JUDGE_MODEL,
        "--judge-model",
        show_default=True,
        help="OpenRouter model id for the judge.",
    )
    detector: str = opt(
        "voice-clf",
        "--detector",
        show_default=True,
        type=click.Choice(list(DETECTOR_KINDS)),
        help="Which instrument scores detector.score: the trained head at --detector-model "
        "(null when none is given), or a hosted one (sapling: $SAPLING_API_KEY, spends per "
        "character).",
    )
    detector_model: Path | None = opt(
        None,
        "--detector-model",
        "detector_model",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="A trained voice-classifier artifact dir (head.json + meta.json) to score "
        "detector.score; needs the pinned embedder (sentence-transformers) installed.",
    )
    vale_config: Path | None = opt(
        None,
        "--vale-config",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Vale config (.vale.ini); omitted = Vale's own discovery (or Vale absent -> skipped).",
    )
    max_workers: int = opt(
        DEFAULT_MAX_WORKERS,
        "--max-workers",
        show_default=True,
        type=int,
        help="Concurrent judge calls (asyncio). Vale is a batched pre-pass, not a worker (D59).",
    )
    out_path: Path | None = opt(
        None,
        "--out",
        "out_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="scores.jsonl to append (default: <pairs-dir>/scores.jsonl). Resumable — scored "
        "ids are skipped.",
    )
    summary_path: Path | None = opt(
        None,
        "--summary",
        "summary_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Also write the aggregate summary JSON here.",
    )
    by: str | None = opt(
        None,
        "--by",
        "by",
        help="Facet the summary by a meta key, e.g. slop_strategy or generator.",
    )
    limit: int | None = opt(
        None, "--limit", type=int, help="Cap pairs scored (smoke / cost control)."
    )
    prune_orphans: bool = opt(
        True,
        "--prune-orphans/--no-prune-orphans",
        show_default=True,
        help="Drop score records whose pair is no longer in --pairs (re-cut or regenerated), so "
        "every summary and exported table counts live pairs only. Turn OFF only when --out belongs "
        "to a DIFFERENT corpus, where every record reads as orphaned.",
    )
    report_path: Path | None = opt(
        None,
        "--report",
        "report_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Write a self-contained HTML scores report — the pair browser: slop↔Dan side by side, "
        "judge/detector/Vale chips, generation covariates, facet dropdowns. Works keyless (judge "
        "badges show — without --judge).",
    )
    report_max_rows: int = opt(
        report.DEFAULT_MAX_ROWS,
        "--report-max-rows",
        show_default=True,
        type=int,
        help="Cap rows in the HTML report (0 = all).",
    )
    facet_by: str = opt(
        report.DEFAULT_FACET_BY,
        "--facet-by",
        "facet_by",
        show_default=True,
        help="Group the HTML report headline by this meta covariate (e.g. reasoning_effort, "
        "prompt_id, generator).",
    )
    sample_n: int | None = opt(
        None,
        "--sample",
        "sample_n",
        type=int,
        help="Print N random scored pairs (slop vs Dan + scores) to stdout.",
    )

    def scores_path(self) -> Path:
        """`--out`, else `scores.jsonl` beside the corpus."""
        return Path(self.out_path or (Path(self.pairs_path).parent / "scores.jsonl"))

    def score_fields(self) -> tuple[str, ...]:
        """`--field`s, else both sides of the pair."""
        return tuple(self.fields) or ("slop", "target")


@dataclass(frozen=True)
class VerdictOpts:
    """The comparison surface, declared once for every CLI that offers it (D78)."""

    cli_section = "Verdict / comparison"

    allow_mixed_detector: bool = opt(
        False,
        "--allow-mixed-detector",
        is_flag=True,
        help="Append to a scores file written by a DIFFERENT detector artifact. Off by default: "
        "resume skips by pair id, so a refit head would blend two instruments into one mean.",
    )
    compare_path: Path | None = opt(
        None,
        "--compare",
        "compare_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="A second scores.jsonl to compare against, paired on pair id: prints the mean "
        "difference with a bootstrap CI clustered by POST. Refuses a cross-detector comparison.",
    )
    compare_field: str = opt(
        "output",
        "--compare-field",
        show_default=True,
        help="Which scored field --compare pits against the other file.",
    )
    compare_signal: str = opt(
        "detector",
        "--compare-signal",
        show_default=True,
        type=click.Choice(["detector", "judge", "vale"]),
        help="Which signal --compare reports.",
    )
    cluster_by: str = opt(
        "source",
        "--cluster-by",
        show_default=True,
        help="Meta key the --compare bootstrap resamples (the POST). Empty string = treat every "
        "chunk as independent, which over a concentrated stratum makes the CI too narrow.",
    )
    allow_cross_detector: bool = opt(
        False,
        "--allow-cross-detector",
        is_flag=True,
        help="Let --compare span two detector artifacts. A refit head moves P(slop) on its own, "
        "so the delta then measures the instrument as much as the adapter.",
    )


@dataclass(frozen=True)
class DocumentOpts:
    """The document-scale eval surface — `rewrite_documents_file`'s flags. Only
    four: the styler is configured by the manifest, so the diet it records is
    the one scored (D54)."""

    cli_section = "Document-scale eval"

    documents: bool = opt(
        False,
        "--documents",
        is_flag=True,
        help="Rewrite WHOLE capture-archive documents through the styler instead of scoring "
        "pre-cut chunks, then score the result. This is the only mode that exercises "
        "segmentation, protected blocks and the chunk diet — a diet switch cannot move a "
        "number computed on chunks that were cut before it. PAID: it samples the adapter.",
    )
    documents_manifest: Path | None = opt(
        None,
        "--documents-manifest",
        "documents_manifest",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Training manifest whose adapter and chunk diet --documents serves.",
    )
    documents_out: Path | None = opt(
        None,
        "--documents-out",
        "documents_out",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Where --documents writes its output JSONL. Default: documents.jsonl beside --out.",
    )
    documents_role: str = opt(
        "eval",
        "--documents-role",
        show_default=True,
        help="Restrict --documents to this splits role. 'all' takes every archive; anything but "
        "'eval' is measuring memorisation, since the styler trained on those posts.",
    )


VERDICT_PARAMS: dict[str, type] = {"verdict": VerdictOpts, "documents": DocumentOpts}
EVAL_PARAMS: dict[str, type] = {"eval": EvalOpts, **VERDICT_PARAMS, "view": ViewOpts}


def verdict_options(**kw):
    """The shared verdict/comparison + document-scale flags, one decorator;
    `defaults=`/`exclude=` as elsewhere."""
    from stylebot.cli_options import bundle_options

    return bundle_options([VerdictOpts, DocumentOpts], **kw)


def eval_options(**kw):
    """The whole `eval` surface — `EVAL_PARAMS`'s four bundles, one decorator."""
    from stylebot.cli_options import bundle_options

    return bundle_options(list(EVAL_PARAMS.values()), **kw)


def report_eval(
    out: Path,
    pairs_path: Path,
    fields: Sequence[str],
    verdict: VerdictOpts,
    opts: EvalOpts,
    *,
    emit: Emitter = silent,
) -> None:
    """The tail of the `eval` pass: A-vs-B verdict, sample, HTML report (D78)."""
    if verdict.compare_path is not None:
        comparison = compare_scores(
            out,
            verdict.compare_path,
            field=verdict.compare_field,
            signal=verdict.compare_signal,
            cluster_by=verdict.cluster_by or None,
            allow_cross_detector=verdict.allow_cross_detector,
        )
        emit("")
        for line in format_verdict(comparison, out, verdict.compare_path):
            emit(line, level="warn" if line.lstrip().startswith("WARNING:") else "info")
    if opts.sample_n is not None:
        emit(report.format_scores_sample(out, pairs_path, opts.sample_n, fields=fields))
    if opts.report_path is not None:
        written = report.render_scores_report(
            out,
            pairs_path,
            opts.report_path,
            fields=fields,
            max_rows=(None if opts.report_max_rows == 0 else opts.report_max_rows),
            facet_by=opts.facet_by,
        )
        emit(f"wrote report -> {written}")


def format_summary(summary: dict) -> list[str]:
    """The aggregate as printable lines, real facet first (D40); a renderer,
    so both CLIs show the same thing."""
    lines: list[str] = []

    def block(prefix: str, fields: dict) -> None:
        for name, agg in fields.items():
            lines.append(
                f"{prefix}{name}: n={agg['n']} judge={agg['mean_judge_score']} "
                f"unscored={agg['n_unscored']} vale_alerts={agg['mean_vale_alerts']} "
                f"detector={agg['mean_detector_score']}"
            )

    facets = summary.get("by_provenance") or {}
    real, synthetic = facets.get("real") or {}, facets.get("synthetic") or {}
    if real.get("pairs") and synthetic.get("pairs"):
        lines.append(f"HEADLINE — real pairs only ({real['pairs']}):")
        block("  ", real["fields"])
        lines.append(f"synthetic pairs, reported separately ({synthetic['pairs']}):")
        block("  ", synthetic["fields"])
        lines.append(f"all pairs ({summary['pairs']}):")
        block("  ", summary["fields"])
    else:
        block("", summary["fields"])
    for identity in summary.get("detectors") or []:
        lines.append(f"detector: {_identity_label(identity)}")
    return lines


def format_verdict(verdict: dict, a_label: object = "a", b_label: object = "b") -> list[str]:
    """The paired comparison as printable lines, CI and cluster count included."""
    delta, unscored = verdict["delta"], verdict["n_unscored"]
    lines = [
        f"verdict — {verdict['signal']} on '{verdict['field']}', "
        f"{verdict['provenance'] or 'all'} pairs, clustered by {verdict['cluster_by']}",
        f"  a = {a_label}  mean {verdict['mean_a']}  unscored {unscored['a']}",
        f"  b = {b_label}  mean {verdict['mean_b']}  unscored {unscored['b']}",
    ]
    if delta is None:
        lines.append("  not enough shared, scored ids to compare")
        return lines
    ci = (
        f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
        if delta.get("ci_low") is not None
        else "[no interval: one cluster]"
    )
    lines.append(
        f"  paired a-b: {delta['mean']:+.4f}  95% CI {ci}  "
        f"n={delta['n']} over {delta['n_clusters']} cluster(s) [{delta['method']}]"
        f"  -> {verdict['verdict']}"
    )
    if verdict["cross_detector"]:
        lines.append(
            "  WARNING: the two sides were scored by DIFFERENT detector artifacts; "
            "this delta measures the instrument as well as the adapter"
        )
    return lines


def _verdict_label(delta: dict | None, *, lower_is_better: bool) -> str:
    if delta is None or delta.get("ci_low") is None:
        return "undetermined"
    if delta["ci_high"] < 0:
        return "a better" if lower_is_better else "b better"
    if delta["ci_low"] > 0:
        return "b better" if lower_is_better else "a better"
    return "indistinguishable"


def _refuse_circular_verdict(rows: Iterable[dict], identities: list[dict], side: str) -> None:
    """A detector verdict on rows the same head chose is self-agreement (D39)."""
    for row in rows:
        reranker = (row.get("meta") or {}).get("reranker")
        if not reranker:
            continue
        picked_by = {key: reranker.get(key) for key in DETECTOR_IDENTITY_KEYS}
        if picked_by in identities:
            raise ConfigError(
                f"refusing a detector verdict on {side}: its outputs were picked by the same "
                f"head that scores them, {_identity_label(picked_by)} (best-of reranking), so "
                f"P(slop) can only agree with itself. Compare another instrument, "
                f"--compare-signal judge, or rewrite with --best-of 1"
            )


def compare_scores(
    scores_a: str | Path | Iterable[dict],
    scores_b: str | Path | Iterable[dict],
    *,
    field: str = "output",
    signal: str = "detector",
    cluster_by: str | None = "source",
    provenance: str | None = "real",
    n_boot: int = 2000,
    seed: int = 0,
    allow_cross_detector: bool = False,
) -> dict:
    """Paired verdict between two scored runs over the ids both carry.

    Bootstrap clustered on `meta[cluster_by]`, the post by default; `provenance`
    restricts to one facet, `"real"` by default, None for all (D40). A row with
    no `signal` value on one side is dropped into ``n_unscored``. Refuses a
    cross-detector comparison unless `allow_cross_detector` (D39), which the
    returned dict records.
    """
    if signal not in SIGNAL_EXTRACTORS:
        raise ConfigError(f"unknown signal {signal!r}; known: {sorted(SIGNAL_EXTRACTORS)}")
    rows_a = {str(r["id"]): r for r in load_scores(scores_a) if r.get("id") is not None}
    rows_b = {str(r["id"]): r for r in load_scores(scores_b) if r.get("id") is not None}

    ids_a = assert_single_detector(rows_a.values(), allow_mixed=allow_cross_detector)
    ids_b = assert_single_detector(rows_b.values(), allow_mixed=allow_cross_detector)
    if not allow_cross_detector and ids_a and ids_b and ids_a[0] != ids_b[0]:
        raise ConfigError(
            f"refusing to compare verdicts across detector artifacts: "
            f"{_identity_label(ids_a[0])} vs {_identity_label(ids_b[0])} — a refit head moves "
            f"P(slop) by more than most effects being measured. Re-score both sides with one "
            f"head, or pass allow_cross_detector=True / --allow-cross-detector"
        )
    if signal == "detector":
        _refuse_circular_verdict(rows_a.values(), ids_a, "scores_a")
        _refuse_circular_verdict(rows_b.values(), ids_b, "scores_b")
    extract = SIGNAL_EXTRACTORS[signal]
    shared = sorted(set(rows_a) & set(rows_b))
    a_vals: list[float | None] = []
    b_vals: list[float | None] = []
    clusters: list[object] = []
    for rid in shared:
        meta = rows_a[rid].get("meta") or {}
        facet = "synthetic" if meta.get("synthetic") else "real"
        if provenance is not None and facet != provenance:
            continue
        a_vals.append(extract((rows_a[rid].get("scores") or {}).get(field) or {}))
        b_vals.append(extract((rows_b[rid].get("scores") or {}).get(field) or {}))
        clusters.append(meta.get(cluster_by) if cluster_by else rid)

    delta = paired_delta(
        a_vals,
        b_vals,
        clusters=clusters if cluster_by else None,
        n_boot=n_boot,
        seed=seed,
    )
    usable_a = [v for v in a_vals if v is not None]
    usable_b = [v for v in b_vals if v is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "field": field,
        "signal": signal,
        "provenance": provenance,
        "cluster_by": cluster_by,
        "n_shared_ids": len(shared),
        "mean_a": mean(usable_a) if usable_a else None,
        "mean_b": mean(usable_b) if usable_b else None,
        "n_unscored": {"a": a_vals.count(None), "b": b_vals.count(None)},
        "delta": delta,
        "verdict": _verdict_label(delta, lower_is_better=(signal != "judge")),
        "detector_a": ids_a[0] if ids_a else None,
        "detector_b": ids_b[0] if ids_b else None,
        "cross_detector": bool(allow_cross_detector and ids_a and ids_b and ids_a[0] != ids_b[0]),
    }
