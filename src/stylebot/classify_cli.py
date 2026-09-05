"""`train-clf` as a library pass: `TrainClfOpts`, `train_clf`, `format_clf_metrics` (D78)."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import click

from stylebot import classify_train as ct
from stylebot import splits as sp
from stylebot.cli_options import Emitter, bundle_options, opt, silent
from stylebot.errors import ConfigError


@dataclass(frozen=True)
class TrainClfOpts:
    """The `train-clf` surface — what `train_clf` takes (D78).

    Every default is the constant `classify_train.train` uses.
    """

    cli_section = "Voice classifier"

    pairs_path: Path | None = opt(
        None,
        "--pairs",
        "pairs_path",
        required=True,
        type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
        help="The content-matched pairs corpus to train on (pairs.jsonl or a pairs/ directory).",
    )
    out_dir: Path | None = opt(
        None,
        "--out",
        "out_dir",
        required=True,
        type=click.Path(file_okay=False, path_type=Path),
        help="Artifact directory to write (head.json + meta.json).",
    )
    splits_path: Path | None = opt(
        None,
        "--splits",
        "splits_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="The shared splits.json (make-splits): fit on the detector pool only, never touch "
        "the frozen eval posts, report a styler-posts holdout metric. Mutually exclusive with "
        "--holdout-*.",
    )
    blog_root: Path | None = opt(
        None,
        "--blog-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        is_eager=True,  # processed before any corpus default that reads it (D79)
        help="Tree the pinned eval posts must still exist in; a renamed one fails the run (D34).",
    )
    test_size: float = opt(
        ct.DEFAULT_TEST_SIZE,
        "--test-size",
        show_default=True,
        type=float,
        help="Held-out POST fraction per CV split.",
    )
    n_splits: int = opt(
        ct.DEFAULT_N_SPLITS,
        "--n-splits",
        show_default=True,
        type=int,
        help="GroupShuffleSplit folds for the CV metric.",
    )
    C: float | None = opt(
        None,
        "--C",
        "C",
        type=float,
        help="FIXED C override. Default (unset) selects C per training side by inner group-CV "
        "over C_GRID — do NOT sweep this flag against the printed metric; that is tuning on "
        "the test set.",
    )
    embed_model: str = opt(
        ct.DEFAULT_EMBED_MODEL,
        "--embed-model",
        show_default=True,
        help="Style backbone (sentence-transformers id); pinned into meta.json.",
    )
    holdout_frac: float = opt(
        ct.DEFAULT_HOLDOUT_FRAC,
        "--holdout-frac",
        show_default=True,
        type=float,
        help="Ad-hoc: hold out this fraction of POSTs and ship the head fit on the rest. "
        "Prefer --splits for the shared contract.",
    )
    holdout_seed: int = opt(
        ct.DEFAULT_HOLDOUT_SEED,
        "--holdout-seed",
        show_default=True,
        type=int,
        help="Seed for the deterministic by-POST holdout split.",
    )
    holdout_posts: Path | None = opt(
        None,
        "--holdout-posts",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Ad-hoc: file of POST source ids (one per line) to hold out. Prefer --splits.",
    )
    dry_run: bool = opt(
        False,
        "--dry-run",
        is_flag=True,
        help="Report data assembly (n_pairs, n_posts, class balance) without embedding or writing.",
    )


TRAIN_CLF_PARAMS: dict[str, type] = {"opts": TrainClfOpts}


def train_clf_options(**kw):
    """The whole `train-clf` surface as one decorator; `kw` is `bundle_options`'."""
    return bundle_options([TrainClfOpts], **kw)


def format_clf_metrics(metrics: dict, indent: str = "") -> list[str]:
    """Render a `classify_train` metrics dict: headline, chosen C, provenance facet."""
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
    lines = [
        f"{indent}{label}: pairwise_acc={pa['mean']}±{pa['std']}  "
        f"auc={auc['mean']}±{auc['std']}{c_note}"
    ]
    for name, facet in (metrics.get("by_provenance") or {}).items():
        fpa, fauc = facet["pairwise_accuracy"], facet["auc"]
        lines.append(
            f"{indent}  {name} ({facet['n_pairs']} pairs): "
            f"pairwise_acc={fpa['mean']}±{fpa['std']}  "
            f"auc={fauc['mean']}±{fauc['std']}"
        )
    return lines


def _counting_gate(pair_ok: Callable[[dict], bool], dropped: list) -> Callable[[dict], bool]:
    """`pair_ok` wrapped to record the source of every pair it refuses."""

    def counted(rec: dict) -> bool:
        keep = pair_ok(rec)
        if not keep:
            dropped.append((rec.get("meta") or {}).get("source"))
        return keep

    counted.__name__ = pair_ok.__name__
    return counted


def train_clf(
    opts: TrainClfOpts,
    *,
    extra_positives: Iterable[object] | None = None,
    pair_ok: Callable[[dict], bool] | None = None,
    git_repo: str | Path | None = None,
    emit: Emitter = silent,
) -> tuple[Path, dict] | None:
    """The `train-clf` pass: assemble, report, then fit unless `opts.dry_run`.

    `extra_positives` and `pair_ok` are the caller's selection policy
    (`classify_train.assemble_dataset`), `git_repo` the tree whose sha the
    artifact records. Returns `(artifact_dir, metrics)`, or `None` after a
    dry run. Raises `ConfigError` on a conflicting `--splits`/`--holdout-*`
    or an empty corpus.
    """
    if opts.splits_path is not None and (opts.holdout_frac or opts.holdout_posts is not None):
        raise ConfigError("--splits is mutually exclusive with --holdout-frac/--holdout-posts")

    dropped: list = []
    gate = _counting_gate(pair_ok, dropped) if pair_ok is not None else None
    ds = ct.assemble_dataset(opts.pairs_path, extra_positives=extra_positives, pair_ok=gate)
    if pair_ok is not None:
        emit(f"{pair_ok.__name__}: dropped {len(dropped)} pair(s) from {len(set(dropped))} post(s)")
    n_slop = sum(1 for label in ds.labels if label == ct.LABEL_SLOP)
    n_author = sum(1 for label in ds.labels if label == ct.LABEL_DAN)
    provenance = (
        f"; {ds.n_pairs_real} real / {ds.n_pairs_synthetic} synthetic"
        if ds.n_pairs_synthetic
        else ""
    )
    emit(
        f"{ds.n_pairs} content-matched pair(s) from {ds.n_posts} post(s)  "
        f"[{n_slop} slop / {n_author} author{provenance}; {sum(ds.is_free)} free positives]"
    )
    if ds.n_pairs == 0:
        raise ConfigError(f"no content-matched pairs in {opts.pairs_path}", param_hint="--pairs")
    if opts.dry_run:
        if opts.splits_path is not None:
            splits_doc = sp.load_splits(opts.splits_path)
            if opts.blog_root is not None:
                sp.require_sources_on_disk(splits_doc, set(ds.groups), opts.blog_root)
            for role, entry in sp.summarize_roles(splits_doc, ds).items():
                emit(
                    f"  {role:>8}: {entry['posts']} post(s), {entry['pairs']} pair(s) "
                    f"({entry['pairs_real']} real / {entry['pairs_synthetic']} synthetic)"
                )
            for warning in sp.check_splits(splits_doc, ds):
                emit(f"WARNING: {warning}", level="warn")
        emit("[dry-run] no embedding, no artifact written")
        return None

    holdout_list = None
    if opts.holdout_posts is not None:
        holdout_list = [
            ln.strip() for ln in opts.holdout_posts.read_text().splitlines() if ln.strip()
        ]
    if opts.splits_path is not None:
        emit(f"using shared splits contract: {opts.splits_path} (eval posts untouched)")
    elif holdout_list or opts.holdout_frac:
        how = (
            f"list of {len(holdout_list)} posts"
            if holdout_list
            else f"{opts.holdout_frac:.0%} of posts (seed {opts.holdout_seed})"
        )
        emit(f"holdout split: {how} — head ships fit on the train posts only")
    emit(f"embedding with {opts.embed_model} (downloads on first use) ...")
    return _fit_and_report(
        opts,
        extra_positives=extra_positives,
        pair_ok=pair_ok,
        git_repo=git_repo,
        holdout_list=holdout_list,
        emit=emit,
    )


def _fit_and_report(
    opts: TrainClfOpts,
    *,
    extra_positives: Iterable[object] | None,
    pair_ok: Callable[[dict], bool] | None,
    git_repo: str | Path | None,
    holdout_list: list[str] | None,
    emit: Emitter,
) -> tuple[Path, dict]:
    """`classify_train.train`, then the metrics and the artifact's own record."""
    out, metrics = ct.train(
        opts.pairs_path,
        opts.out_dir,
        extra_positives=extra_positives,
        test_size=opts.test_size,
        n_splits=opts.n_splits,
        C=opts.C,
        embed_model=opts.embed_model,
        git_repo=git_repo,
        holdout_frac=opts.holdout_frac,
        holdout_seed=opts.holdout_seed,
        holdout_posts=holdout_list,
        splits_path=opts.splits_path,
        blog_root=opts.blog_root,
        pair_ok=pair_ok,
    )
    for line in format_clf_metrics(metrics):
        emit(line)
    if "styler_holdout" in metrics:
        emit("styler posts (unseen by this head — the reward-relevant number):")
        for line in format_clf_metrics(metrics["styler_holdout"], indent="  "):
            emit(line)

    # The artifact is the source of truth for what shipped: surface its own
    # record of the chosen C and any danger warnings.
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    head_c = meta.get("head_C") or {}
    if head_c:
        emit(f"shipped head C: {head_c.get('value')} ({head_c.get('mode')})")
    for warning in (meta.get("split") or {}).get("warnings") or []:
        emit(f"WARNING: {warning}", level="warn")
    emit(f"wrote artifact -> {out}/ (head.json, meta.json)")
    return out, metrics
