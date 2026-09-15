"""Voice-classifier training — the trainer behind `stylebot.classify` (D7).

Assemble a dataset from a pairs corpus, embed it with a style backbone, fit a
logistic regression slop-vs-author, evaluate it split by post, and write
`head.json` + `meta.json`. Generic over authors: the caller injects
`pairs_path`, `embed_model`, `extra_positives` and `pair_ok`. Needs the
``classifier`` extra; heavy imports are lazy.

Train style, not topic: content-matched pairs, split by post, never by
paragraph. The headline metric is content-matched pairwise accuracy plus AUC,
faceted by provenance on a mixed corpus (D40). Slop is the positive class
(`LABEL_SLOP=1`), so the served ``score = P(slop)`` and `p_dan = 1 - score`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from stylebot import classify, synth
from stylebot.errors import ConfigError, MissingExtra
from stylebot.pairs import extract_slop, extract_target, iter_pairs
from stylebot.provenance import git_sha

logger = logging.getLogger(__name__)

# The default style backbone (D7); whatever an author's own bake-off picks is
# pinned into the artifact's meta.json and enforced at serve time.
DEFAULT_EMBED_MODEL = "StyleDistance/styledistance"
NORMALIZE = True
SCHEMA_VERSION = 1

LABEL_SLOP = 1
LABEL_DAN = 0  # the author class ("Dan" for historical reasons; generic in use)

# Leakage-safe evaluation defaults.
DEFAULT_TEST_SIZE = 0.25
DEFAULT_N_SPLITS = 8
DEFAULT_RANDOM_STATE = 0
DEFAULT_HOLDOUT_FRAC = 0.0  # no ad-hoc by-POST holdout: CV metric, fit on all
DEFAULT_HOLDOUT_SEED = 0

# Regularization. C=None means nested selection by inner group-CV over C_GRID;
# DEFAULT_C is the fallback when the fit pool has too few posts for an inner split.
DEFAULT_C = 1.0
# The grid's top is above 10.0 because a refit once chose 10.0 in every fold:
# a boundary chosen unanimously has been clipped, not searched.
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
MIN_POSTS_FOR_C_SELECTION = 4

_EXTRA_HINT = (
    "stylebot.classify_train needs the 'classifier' extra: "
    "run `uv add 'stylebot[classifier]'` (or `pip install 'stylebot[classifier]'`) "
    "to get scikit-learn/numpy/sentence-transformers. The stylebot runtime "
    "(stylebot.classify) stays dependency-free without it."
)


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingExtra(_EXTRA_HINT) from exc
    return np


# ---------------------------------------------------------------------------
# Data assembly — learn STYLE not TOPIC (content-matched pairs, grouped by post)
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """The assembled training matrix inputs, before embedding.

    `texts[i]` has label `labels[i]` and post `groups[i]`; `pair_rows` are the
    `(slop_row, author_row)` twins the headline metric scores; `pair_synth[k]`
    marks a synthetic pair; `is_free[i]` marks a free-standing positive, which
    augments the fit but never the metric.
    """

    texts: list[str] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    groups: list[str] = field(default_factory=list)
    pair_rows: list[tuple[int, int]] = field(default_factory=list)
    pair_synth: list[bool] = field(default_factory=list)
    is_free: list[bool] = field(default_factory=list)

    @property
    def n_pairs(self) -> int:
        return len(self.pair_rows)

    @property
    def n_posts(self) -> int:
        return len({g for g in self.groups})

    def pair_is_synthetic(self, k: int) -> bool:
        # Tolerate hand-built Datasets that never filled pair_synth: all-real.
        return self.pair_synth[k] if k < len(self.pair_synth) else False

    @property
    def n_pairs_synthetic(self) -> int:
        return sum(1 for k in range(self.n_pairs) if self.pair_is_synthetic(k))

    @property
    def n_pairs_real(self) -> int:
        return self.n_pairs - self.n_pairs_synthetic

    def _add(self, text: str, label: int, group: str, *, free: bool = False) -> int:
        self.texts.append(text)
        self.labels.append(label)
        self.groups.append(group)
        self.is_free.append(free)
        return len(self.texts) - 1

    def _add_pair(
        self, slop: str, author: str, group: str, *, synthetic: bool = False
    ) -> tuple[int, int]:
        s_row = self._add(slop, LABEL_SLOP, group)
        a_row = self._add(author, LABEL_DAN, group)
        self.pair_rows.append((s_row, a_row))
        self.pair_synth.append(synthetic)
        return s_row, a_row


def assemble_dataset(
    pairs_path: str | Path,
    *,
    extra_positives: Iterable[object] | None = None,
    max_transform_sim: float | None = 0.85,
    pair_ok: Callable[[dict], bool] | None = None,
) -> Dataset:
    """Build the slop-vs-author dataset from a content-matched pairs corpus.

    Each pair contributes both sides, heading-context stripped, grouped by
    `meta.source`; a pair over `max_transform_sim` is dropped as label noise
    (D18; None disables the gate), the value computed from the two sides when
    the record stamps none (D104). `extra_positives` (`synth.Target`s or
    `(text, source)` tuples) are flagged `is_free` and kept out of the metric.
    `pair_ok` is the caller's pair-level policy hook, never hardcoded (D85).
    """
    ds = Dataset()
    n_near_identity = 0
    n_selector = 0
    for rec in iter_pairs(pairs_path):
        slop = extract_slop(rec).strip()
        author = extract_target(rec).strip()
        if not slop or not author:
            continue
        if pair_ok is not None and not pair_ok(rec):
            n_selector += 1
            continue
        meta = rec.get("meta") or {}
        if max_transform_sim is not None:
            sim = meta.get("transform_sim")
            if sim is None:  # a capture stamps none (D104)
                sim = synth.transform_similarity(slop, author)
            if sim > max_transform_sim:
                n_near_identity += 1
                continue
        source = meta.get("source") or "?"
        ds._add_pair(slop, author, source, synthetic=bool(meta.get("synthetic")))
    if n_near_identity:
        logger.info(
            "dropped %d near-identity pair(s) (transform_sim > %s) — label noise",
            n_near_identity,
            max_transform_sim,
        )
    if n_selector:
        logger.info(
            "dropped %d pair(s) by the caller's pair_ok policy (%s)",
            n_selector,
            getattr(pair_ok, "__name__", repr(pair_ok)),
        )

    if extra_positives is not None:
        n_extra = 0
        for item in extra_positives:
            if isinstance(item, tuple):
                text, source = item
            else:  # a synth.Target or anything shaped like one
                text, source = item.text, item.source
            text = (text or "").strip()
            if not text:
                continue
            ds._add(text, LABEL_DAN, source, free=True)
            n_extra += 1
        logger.info("added %d free positives (excluded from the metric)", n_extra)

    return ds


def subset_dataset(ds: Dataset, keep_groups) -> tuple[Dataset, list[int]]:
    """Restrict a Dataset to the rows whose post is in `keep_groups`; returns
    `(sub, rows)` with `rows` the kept original indices, so `X[rows]` aligns.
    Twins share a post, so pairs survive whole."""
    keep_groups = set(keep_groups)
    rows = [i for i, g in enumerate(ds.groups) if g in keep_groups]
    remap = {old: new for new, old in enumerate(rows)}
    sub = Dataset(
        texts=[ds.texts[i] for i in rows],
        labels=[ds.labels[i] for i in rows],
        groups=[ds.groups[i] for i in rows],
        is_free=[ds.is_free[i] for i in rows],
    )
    for k, (s_row, a_row) in enumerate(ds.pair_rows):
        if s_row in remap:  # twins share a group: both kept or both dropped
            sub.pair_rows.append((remap[s_row], remap[a_row]))
            sub.pair_synth.append(ds.pair_is_synthetic(k))
    return sub, rows


# ---------------------------------------------------------------------------
# Embedding (the heavy bit)
# ---------------------------------------------------------------------------


def embed_texts(
    texts: Sequence[str],
    *,
    model_id: str = DEFAULT_EMBED_MODEL,
    normalize: bool = NORMALIZE,
    batch_size: int = 32,
):
    """Batch-encode `texts` with the style backbone -> [N, D] float32, through
    the one encode call-site shared with serving."""
    np = _np()
    try:
        encode = classify.sentence_transformers_batch_embed_fn(model_id, normalize=normalize)
    except ImportError as exc:
        raise MissingExtra(_EXTRA_HINT) from exc
    return encode(texts, batch_size=batch_size).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Fit + leakage-safe evaluation
# ---------------------------------------------------------------------------


def _make_logreg(C: float = DEFAULT_C):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise MissingExtra(_EXTRA_HINT) from exc

    # class_weight='balanced' guards the small/imbalanced negative set.
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=C)


def _pairwise_score(p_slop, pairs, test_set: set) -> tuple[float, float]:
    """(correct, total) over the content-matched twins fully inside `test_set`:
    P(slop|slop_side) > P(slop|author_side), ties 0.5. `p_slop` is indexed by
    absolute row but read only at `test_set` positions."""
    correct = total = 0.0
    for s_row, a_row in pairs:
        if s_row in test_set and a_row in test_set:
            total += 1
            ps, pa = p_slop[s_row], p_slop[a_row]
            correct += 1.0 if ps > pa else (0.5 if ps == pa else 0.0)
    return correct, total


def select_C(X, ds: Dataset, rows, *, grid=C_GRID, n_inner: int = 4) -> tuple[float, dict]:
    """Pick C by inner GroupKFold over the posts within `rows`, scored by
    pairwise accuracy; ties to the smallest C. Under
    MIN_POSTS_FOR_C_SELECTION posts, DEFAULT_C, and the info says so. Call it
    strictly inside a training side."""
    np = _np()
    from sklearn.model_selection import GroupKFold

    rows = list(rows)
    y = np.asarray(ds.labels)
    groups = np.asarray(ds.groups)
    n_posts = len(set(groups[rows].tolist()))
    if n_posts < MIN_POSTS_FOR_C_SELECTION:
        info = {
            "mode": "fallback",
            "value": DEFAULT_C,
            "reason": f"only {n_posts} post(s) in the fit pool "
            f"(< {MIN_POSTS_FOR_C_SELECTION}) — no room for an inner split",
        }
        return DEFAULT_C, info

    rows_arr = np.asarray(rows)
    n_folds = min(n_inner, n_posts)
    gkf = GroupKFold(n_splits=n_folds)
    folds = [
        (rows_arr[tr], rows_arr[va], set(rows_arr[va].tolist()))
        for tr, va in gkf.split(X[rows_arr], y[rows_arr], groups[rows_arr])
    ]

    scores: dict[float, float | None] = {}
    best_C, best_score = DEFAULT_C, -1.0
    for C in grid:  # ascending grid + strict '>' ⇒ ties resolve to the smallest C
        fold_accs = []
        for fit_rows, val_rows, val_set in folds:
            clf = _make_logreg(C)
            clf.fit(X[fit_rows], y[fit_rows])
            # Score the validation rows only, into a full-length vector indexed
            # by absolute row; the untouched NaNs are unreachable.
            p_slop = np.full(len(y), np.nan)
            p_slop[val_rows] = clf.predict_proba(X[val_rows])[
                :, list(clf.classes_).index(LABEL_SLOP)
            ]
            correct, total = _pairwise_score(p_slop, ds.pair_rows, val_set)
            if total:
                fold_accs.append(correct / total)
        score = float(np.mean(fold_accs)) if fold_accs else None
        scores[C] = None if score is None else round(score, 4)
        if score is not None and score > best_score:
            best_C, best_score = C, score
    if best_score < 0:  # no fold had scorable twins
        return DEFAULT_C, {
            "mode": "fallback",
            "value": DEFAULT_C,
            "reason": "no scorable inner folds",
        }
    return best_C, {
        "mode": "nested_cv",
        "value": best_C,
        "grid": list(grid),
        "n_inner": n_folds,
        "scores": scores,
    }


STRATA = ("real", "synthetic")


def _provenance_strata(ds: Dataset) -> tuple[dict[str, list[int]], dict[int, str]]:
    """`(stratum -> pair indices, matched row -> stratum)` for real/synthetic
    faceting; empty on a single-provenance corpus."""
    pair_idx = {name: [] for name in STRATA}
    row_stratum: dict[int, str] = {}
    for k, (s_row, a_row) in enumerate(ds.pair_rows):
        name = "synthetic" if ds.pair_is_synthetic(k) else "real"
        pair_idx[name].append(k)
        row_stratum[s_row] = row_stratum[a_row] = name
    if not all(pair_idx.values()):  # single-stratum corpus -> no facet
        return {}, row_stratum
    return pair_idx, row_stratum


def _stat(xs: list[float]) -> dict:
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    np = _np()
    return {"mean": round(float(np.mean(xs)), 4), "std": round(float(np.std(xs)), 4), "n": len(xs)}


def _fold_metrics(p_slop, y, ds: Dataset, test_rows: list[int], *, pair_idx, row_stratum) -> dict:
    """Content-matched metrics for one evaluation split: pairwise twin accuracy
    + AUC over the matched test rows, globally and per provenance stratum, as
    raw `(value_or_None, n)` tuples."""
    from sklearn.metrics import roc_auc_score

    matched_rows = {r for pair in ds.pair_rows for r in pair}
    test_set = set(test_rows)

    def _auc(rows: list[int]) -> tuple[float | None, int]:
        if len(set(y[rows].tolist())) == 2:
            return float(roc_auc_score(y[rows], p_slop[rows])), len(rows)
        return None, len(rows)

    def _pairwise(pairs) -> tuple[float | None, int]:
        correct, total = _pairwise_score(p_slop, pairs, test_set)
        return (correct / total if total else None), int(total)

    matched_test = [r for r in test_rows if r in matched_rows]
    return {
        "auc": _auc(matched_test),
        "pairwise": _pairwise(ds.pair_rows),
        "facets": {
            name: {
                "auc": _auc([r for r in matched_test if row_stratum[r] == name]),
                "pairwise": _pairwise(ds.pair_rows[k] for k in pair_idx[name]),
            }
            for name in pair_idx
        },
    }


def evaluate(
    X,
    ds: Dataset,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    C: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """Post-split content-matched pairwise accuracy + per-text AUC.

    Per `GroupShuffleSplit` fold: fit on the train rows, measure `_fold_metrics`
    on the held-out side. `C=None` runs nested CV (`select_C` over each fold's
    train posts); an explicit C skips it. A mixed corpus gains `by_provenance`.
    """
    np = _np()
    from sklearn.model_selection import GroupShuffleSplit

    y = np.asarray(ds.labels)
    groups = np.asarray(ds.groups)
    pair_idx, row_stratum = _provenance_strata(ds)

    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    pair_accs: list[float] = []
    aucs: list[float] = []
    selected_Cs: list[float] = []
    facet_pair_accs: dict[str, list[float]] = {name: [] for name in pair_idx}
    facet_aucs: dict[str, list[float]] = {name: [] for name in pair_idx}
    for train_idx, test_idx in gss.split(X, y, groups):
        fold_C = C if C is not None else select_C(X, ds, train_idx.tolist())[0]
        selected_Cs.append(fold_C)
        clf = _make_logreg(fold_C)
        clf.fit(X[train_idx], y[train_idx])
        p_slop = clf.predict_proba(X)[:, list(clf.classes_).index(LABEL_SLOP)]

        fm = _fold_metrics(
            p_slop, y, ds, test_idx.tolist(), pair_idx=pair_idx, row_stratum=row_stratum
        )
        if (auc := fm["auc"][0]) is not None:
            aucs.append(auc)
        if (pa := fm["pairwise"][0]) is not None:
            pair_accs.append(pa)
        for name, facet in fm["facets"].items():
            if (f_auc := facet["auc"][0]) is not None:
                facet_aucs[name].append(f_auc)
            if (f_pa := facet["pairwise"][0]) is not None:
                facet_pair_accs[name].append(f_pa)

    result = {
        "pairwise_accuracy": _stat(pair_accs),
        "auc": _stat(aucs),
        "n_splits": n_splits,
        "test_size": test_size,
        "mode": "cross_val",
        "C": (
            {"mode": "nested_cv", "selected": selected_Cs, "grid": list(C_GRID)}
            if C is None
            else {"mode": "fixed", "value": C}
        ),
    }
    if pair_idx:
        result["by_provenance"] = {
            name: {
                "pairwise_accuracy": _stat(facet_pair_accs[name]),
                "auc": _stat(facet_aucs[name]),
                "n_pairs": len(pair_idx[name]),
            }
            for name in STRATA
        }
    return result


def partition_posts(
    sources,
    *,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
    holdout_posts=None,
) -> tuple[set, set]:
    """Split the distinct posts into `(train, holdout)` (D17, D42): an explicit
    `holdout_posts` pins the partition, else it derives from `holdout_frac` +
    `holdout_seed`; `holdout_frac=0` and no list is no holdout."""
    posts = sorted(set(sources))
    if holdout_posts:
        hold = set(holdout_posts) & set(posts)
    elif holdout_frac and holdout_frac > 0:
        import random

        shuffled = list(posts)
        random.Random(holdout_seed).shuffle(shuffled)
        n_hold = max(1, round(len(posts) * holdout_frac))
        hold = set(shuffled[:n_hold])
    else:
        hold = set()
    return set(posts) - hold, hold


def evaluate_holdout(X, ds: Dataset, holdout: set, *, C: float | None = None) -> dict:
    """Single by-post holdout eval: fit on the train posts, score the holdout
    twins. `C=None` selects over the train posts only."""
    np = _np()

    y = np.asarray(ds.labels)
    train_rows = [i for i, g in enumerate(ds.groups) if g not in holdout]
    test_rows = [i for i, g in enumerate(ds.groups) if g in holdout]
    if C is None:
        C, C_info = select_C(X, ds, train_rows)
    else:
        C_info = {"mode": "fixed", "value": C}
    clf = _make_logreg(C)
    clf.fit(X[train_rows], y[train_rows])
    p_slop = clf.predict_proba(X)[:, list(clf.classes_).index(LABEL_SLOP)]
    pair_idx, row_stratum = _provenance_strata(ds)

    def _rounded(pair: tuple[float | None, int]) -> tuple[float | None, int]:
        value, n = pair
        return (round(value, 4) if value is not None else None), n

    fm = _fold_metrics(p_slop, y, ds, test_rows, pair_idx=pair_idx, row_stratum=row_stratum)
    pa_mean, n_pairs = _rounded(fm["pairwise"])
    auc, n_auc = _rounded(fm["auc"])
    result = {
        "pairwise_accuracy": {"mean": pa_mean, "std": None, "n": n_pairs},
        "auc": {"mean": auc, "std": None, "n": n_auc},
        "mode": "holdout",
        "C": C_info,
    }
    if pair_idx:
        by = {}
        for name in STRATA:
            f_pa, f_n = _rounded(fm["facets"][name]["pairwise"])
            f_auc, f_n_auc = _rounded(fm["facets"][name]["auc"])
            by[name] = {
                "pairwise_accuracy": {"mean": f_pa, "std": None, "n": f_n},
                "auc": {"mean": f_auc, "std": None, "n": f_n_auc},
                "n_pairs": len(pair_idx[name]),
            }
        result["by_provenance"] = by
    return result


def fit_head(X, ds: Dataset, *, C: float = DEFAULT_C, rows: list[int] | None = None):
    """Fit the final logreg (on `rows`, if given) and return `(coef, intercept)`
    for the P(slop) class, which `classify._logreg_p_slop` reproduces exactly."""
    np = _np()

    if rows is None:
        X_fit, y_fit = X, ds.labels
    else:
        X_fit, y_fit = X[rows], np.asarray(ds.labels)[rows]
    clf = _make_logreg(C)
    clf.fit(X_fit, y_fit)
    slop_col = list(clf.classes_).index(LABEL_SLOP)
    # One weight row for the positive class (classes_[1]); index defensively.
    coef_row = clf.coef_[slop_col] if clf.coef_.shape[0] > 1 else clf.coef_[0]
    intercept = clf.intercept_[slop_col] if clf.intercept_.shape[0] > 1 else clf.intercept_[0]
    return [float(c) for c in coef_row], float(intercept)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def write_artifact(
    out_dir: str | Path,
    *,
    coef: list[float],
    intercept: float,
    ds: Dataset,
    metrics: dict,
    embed_model: str = DEFAULT_EMBED_MODEL,
    normalize: bool = NORMALIZE,
    threshold: float = 0.5,
    git_repo: str | Path | None = None,
    split: dict | None = None,
    head_C: dict | None = None,
    pair_ok_name: str | None = None,
    splits_sha256: str | None = None,
) -> Path:
    """Write `head.json` (the plain-JSON head `stylebot.classify` serves) +
    `meta.json` (the embedder pin and the held-out metrics). No estimator pickle."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    head = {"coef": coef, "intercept": intercept, "calibration": None}
    (out_dir / "head.json").write_text(json.dumps(head), encoding="utf-8")

    meta = {
        "schema_version": SCHEMA_VERSION,
        "name": "voice-clf",
        "embed_model": embed_model,
        "embed_dim": len(coef),
        "normalize": normalize,
        "label_polarity": "p_slop",
        "threshold": threshold,
        "trained_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(git_repo),
        "n_pairs": ds.n_pairs,
        "n_pairs_real": ds.n_pairs_real,
        "n_pairs_synthetic": ds.n_pairs_synthetic,
        "n_posts": ds.n_posts,
        "n_free_positives": sum(ds.is_free),
        # The caller's pair-level selector, by name, so two data policies are
        # distinguishable.
        "pair_ok": pair_ok_name,
        "head_C": head_C,  # how the shipped head's regularization was chosen
        "splits_sha256": splits_sha256,
        "split": split
        or {
            "mode": "fit_all",
            "note": (
                "head fit on ALL posts; honest for measuring an INDEPENDENT styler, but "
                "NOT for grading a styler trained on these posts — use holdout_frac/"
                "holdout_posts for the shared by-POST split when the detector is a reward."
            ),
        },
        "metrics": metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def _resolve_head_C(X, ds: Dataset, rows, C: float | None) -> tuple[float, dict]:
    """The shipped head's C: the explicit override, or inner group-CV over `rows`."""
    if C is not None:
        return C, {"mode": "fixed", "value": C}
    return select_C(X, ds, rows)


def _ship_head(
    X,
    ds: Dataset,
    *,
    rows,
    head_C: float,
    head_C_info: dict,
    metrics: dict,
    split: dict | None,
    ship_kw: dict,
) -> tuple[Path, dict]:
    """Fit the final head (on `rows`, if given) and write the artifact: the
    shared tail of every train mode. C resolution stays with the mode."""
    coef, intercept = fit_head(X, ds, C=head_C, rows=rows)
    out = write_artifact(
        coef=coef,
        intercept=intercept,
        ds=ds,
        metrics=metrics,
        split=split,
        head_C=head_C_info,
        **ship_kw,
    )
    return out, metrics


def _train_splits_mode(
    ds: Dataset,
    splits_path: str | Path,
    *,
    blog_root: str | Path | None,
    test_size: float,
    n_splits: int,
    C: float | None,
    embed_model: str,
    normalize: bool,
    batch_size: int,
    ship_kw: dict,
) -> tuple[Path, dict]:
    """The three-role contract: fit on the detector pool only; eval posts are
    excluded before embedding; styler posts supply the unseen-posts metric."""
    from stylebot import splits as splits_mod

    sp = splits_mod.load_splits(splits_path)
    if blog_root is not None:
        splits_mod.require_sources_on_disk(sp, set(ds.groups), blog_root)
    warnings = splits_mod.check_splits(sp, ds)
    for w in warnings:
        logger.warning("splits: %s", w)
    role = {g: splits_mod.role_of(g, sp) for g in set(ds.groups)}
    det_groups = {g for g, r in role.items() if r == "detector"}
    sty_groups = {g for g, r in role.items() if r == "styler"}

    work_ds, _ = subset_dataset(ds, det_groups | sty_groups)
    det_ds, det_rows = subset_dataset(work_ds, det_groups)
    if det_ds.n_pairs == 0:
        raise ConfigError("splits: the detector pool has no pairs — nothing to train on")
    logger.info(
        "embedding %d passages with %s (%d eval-role posts excluded)",
        len(work_ds.texts),
        embed_model,
        sum(1 for r in role.values() if r == "eval"),
    )
    X = embed_texts(work_ds.texts, model_id=embed_model, normalize=normalize, batch_size=batch_size)
    X_det = X[det_rows]

    metrics = evaluate(X_det, det_ds, test_size=test_size, n_splits=n_splits, C=C)
    head_C, head_C_info = _resolve_head_C(X_det, det_ds, list(range(len(det_ds.texts))), C)
    if any(g in sty_groups for g in work_ds.groups):
        # The detector-pool head scored on the styler's (unseen) posts.
        metrics["styler_holdout"] = evaluate_holdout(X, work_ds, sty_groups, C=head_C)
    split = {
        "mode": "splits",
        "path": str(splits_path),
        "seed": sp.get("seed"),
        # The partition this head was fit under, pinned by content and version (D42).
        "eval_version": sp.get("eval_version"),
        "rest_rule": sp["rest_rule"],
        "roles": splits_mod.summarize_roles(sp, ds),
        "warnings": warnings,
    }
    return _ship_head(
        X_det,
        det_ds,
        rows=None,
        head_C=head_C,
        head_C_info=head_C_info,
        metrics=metrics,
        split=split,
        ship_kw={**ship_kw, "splits_sha256": splits_mod.splits_sha256(splits_path)},
    )


def _train_holdout_mode(
    X,
    ds: Dataset,
    train_posts: set,
    holdout: set,
    *,
    holdout_frac: float,
    holdout_seed: int,
    C: float | None,
    ship_kw: dict,
) -> tuple[Path, dict]:
    """Hold out POSTs: report the single-split unseen-posts metric and ship a
    head fit on the train posts ONLY."""
    metrics = evaluate_holdout(X, ds, holdout, C=C)
    train_rows = [i for i, g in enumerate(ds.groups) if g in train_posts]
    head_C, head_C_info = _resolve_head_C(X, ds, train_rows, C)
    split = {
        "mode": "holdout",
        "holdout_frac": holdout_frac,
        "holdout_seed": holdout_seed,
        "n_train_posts": len(train_posts),
        "n_holdout_posts": len(holdout),
        "holdout_posts": sorted(holdout),
    }
    return _ship_head(
        X,
        ds,
        rows=train_rows,
        head_C=head_C,
        head_C_info=head_C_info,
        metrics=metrics,
        split=split,
        ship_kw=ship_kw,
    )


def _train_fit_all(
    X,
    ds: Dataset,
    *,
    test_size: float,
    n_splits: int,
    C: float | None,
    ship_kw: dict,
) -> tuple[Path, dict]:
    """Cross-validated metric over POST splits, then ship the head fit on ALL posts."""
    metrics = evaluate(X, ds, test_size=test_size, n_splits=n_splits, C=C)
    head_C, head_C_info = _resolve_head_C(X, ds, list(range(len(ds.texts))), C)
    return _ship_head(
        X,
        ds,
        rows=None,
        head_C=head_C,
        head_C_info=head_C_info,
        metrics=metrics,
        split=None,
        ship_kw=ship_kw,
    )


def train(
    pairs_path: str | Path,
    out_dir: str | Path,
    *,
    extra_positives: Iterable[object] | None = None,
    test_size: float = DEFAULT_TEST_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    C: float | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    normalize: bool = NORMALIZE,
    batch_size: int = 32,
    git_repo: str | Path | None = None,
    holdout_frac: float = DEFAULT_HOLDOUT_FRAC,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
    holdout_posts=None,
    splits_path: str | Path | None = None,
    blog_root: str | Path | None = None,
    pair_ok: Callable[[dict], bool] | None = None,
) -> tuple[Path, dict]:
    """End-to-end: assemble → embed → evaluate → fit → write artifact.

    Returns `(artifact_dir, metrics)`. `C=None` selects by inner group-CV;
    an explicit C is recorded as an override. Modes: default (CV metric, head
    fit on all posts); holdout (`holdout_frac`/`holdout_posts`: single-split
    unseen-posts metric, head fit on the rest); splits (`splits_path`: the
    three-role contract, `_train_splits_mode`). `blog_root` fails the run on a
    pinned eval post or corpus source that has left the tree (D34).
    """
    ds = assemble_dataset(pairs_path, extra_positives=extra_positives, pair_ok=pair_ok)
    if ds.n_pairs == 0:
        raise ConfigError(f"no content-matched pairs found in {pairs_path}")

    ship_kw = dict(
        out_dir=out_dir,
        embed_model=embed_model,
        normalize=normalize,
        git_repo=git_repo,
        pair_ok_name=getattr(pair_ok, "__name__", None) if pair_ok else None,
    )

    if splits_path is not None:
        if holdout_frac or holdout_posts:
            raise ConfigError("splits_path is mutually exclusive with holdout_frac/holdout_posts")
        return _train_splits_mode(
            ds,
            splits_path,
            blog_root=blog_root,
            test_size=test_size,
            n_splits=n_splits,
            C=C,
            embed_model=embed_model,
            normalize=normalize,
            batch_size=batch_size,
            ship_kw=ship_kw,
        )

    logger.info("embedding %d passages with %s", len(ds.texts), embed_model)
    X = embed_texts(ds.texts, model_id=embed_model, normalize=normalize, batch_size=batch_size)

    train_posts, holdout = partition_posts(
        ds.groups, holdout_frac=holdout_frac, holdout_seed=holdout_seed, holdout_posts=holdout_posts
    )
    if holdout:
        return _train_holdout_mode(
            X,
            ds,
            train_posts,
            holdout,
            holdout_frac=holdout_frac,
            holdout_seed=holdout_seed,
            C=C,
            ship_kw=ship_kw,
        )
    return _train_fit_all(X, ds, test_size=test_size, n_splits=n_splits, C=C, ship_kw=ship_kw)
