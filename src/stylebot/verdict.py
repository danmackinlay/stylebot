"""The verdict — a paired comparison of two scored runs, clustered by post.

`compare_scores` joins two `scores.jsonl` files on the pair id and reports the
paired difference with a cluster bootstrap over posts; `paired_delta` is that
interval on bare sequences. `VerdictOpts` / `DocumentOpts` declare the flags
both CLIs offer, and `format_summary` / `format_verdict` render for both.

Why posts and not chunks, and why the real facet is the headline: D40. Why a
cross-detector comparison is refused: D39.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import click

from stylebot.cli_options import opt
from stylebot.errors import ConfigError
from stylebot.eval import SCHEMA_VERSION
from stylebot.scoring import _identity_label, assert_single_detector, load_scores

#: How to pull one scalar out of a scored field. Vale reports `null` when the
#: binary is absent rather than 0, so an unavailable run drops out of the
#: comparison instead of scoring perfect.
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
    """Mean of `a - b` with a confidence interval; clustered when asked (D40).

    `a`, `b` and `clusters` are aligned; a position where either value is None
    drops out. With `clusters`, whole clusters are resampled with replacement
    `n_boot` times and the interval is the percentile interval of the resampled
    means; without them every observation is its own cluster, which is the
    ordinary paired bootstrap.

    Returns None when fewer than two usable differences remain; the interval is
    None when they all fall in one cluster — one post is not an interval.
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
class VerdictOpts:
    """The comparison surface, declared once for every CLI that offers it.

    A `cli_options` bundle because `ai-style eval` and the blog's `dan-style
    eval` mirror each other flag for flag, pinned by the blog's mirror test.
    """

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
    """The document-scale eval surface — `rewrite_documents_file`'s flags.

    Deliberately only four: the styler is configured entirely by the training
    manifest (`backends.opts_from_manifest`), because the whole point of
    document-scale eval is to exercise the diet the manifest records. A flag
    soup here would let you score a diet no adapter was trained on.
    """

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


def verdict_options(**kw):
    """The shared verdict/comparison + document-scale flags, one decorator.

    Both bundles at once: a surface split across two decorators is two chances
    to add a flag to one CLI and forget the other. `defaults=`/`exclude=` as
    elsewhere.
    """
    from stylebot.cli_options import bundle_options

    return bundle_options([VerdictOpts, DocumentOpts], **kw)


def format_summary(summary: dict) -> list[str]:
    """The aggregate as printable lines, **real facet first** (D40).

    A renderer, not a printer, so `ai-style eval` and the blog's mirror show the
    same thing without either owning the layout.
    """
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
    """Paired verdict between two scored runs over the same ids.

    Joins on the score `id`, keeps the ids both files carry, and reports the
    paired difference with a bootstrap clustered on `meta[cluster_by]` — the
    post, by default. `provenance` restricts the comparison to one facet,
    `"real"` by default; pass None for all rows (D40). A row one side has no
    `signal` value for (a judge refusal) is dropped and counted in ``n_unscored``.

    **Refuses a cross-detector comparison** (D39): both files must have been
    written by the same detector artifact. `allow_cross_detector=True` says you
    mean it, and the returned dict records that you did.
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
