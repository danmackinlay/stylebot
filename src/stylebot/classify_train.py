"""Generic voice-classifier *training* — the other half of `stylebot.classify`.

`classify.py` is the runtime seam (dep-free import, pure-Python scoring of a
plain-JSON linear head). This module is the trainer that *produces* that
artifact: assemble a dataset from a `pairs.jsonl` corpus, embed it with a style
backbone, fit a logistic regression slop-vs-author, evaluate it leakage-safely
(split by POST), and write `head.json` + `meta.json`.

It is generic over authors by construction — every input and output is a
stylebot contract (`pairs.jsonl` in, `head.json`/`meta.json` out), and the
caller-specific choices are injected:

- **Which pairs**: the `pairs_path` argument.
- **Extra free-standing positives**: `extra_positives` — an iterable of
  `(text, source)` pairs (or `synth.Target`s) the *caller* selected under its own
  policy (cf. `iter_targets(selector=…)`). Off by default: free positives
  reintroduce a topic signal, so they are flagged and excluded from the metric.
- **The backbone**: `embed_model`, defaulting to `DEFAULT_EMBED_MODEL` below.

**Dependencies.** This module needs the ``classifier`` extra —
``uv add 'stylebot[classifier]'`` (scikit-learn, numpy, sentence-transformers).
All heavy imports are lazy (inside functions), so importing the module is free
and `import stylebot.classify` stays ML-dep-free; calling a trainer function
without the extra raises an actionable ImportError.

**The methodology crux — train STYLE not TOPIC (leakage safety).** Use
content-matched pairs (slop vs author of the *same* passage), split by POST
(`GroupShuffleSplit` — never by paragraph), `class_weight='balanced'`. The
headline metric is content-matched pairwise accuracy ("pick the author's version
of the same passage") plus AUC. Two training modes:

- **fit-all (default)**: cross-validated metric over POST splits, head shipped
  fit on all posts. Honest for *measuring* an independent styler; NOT for
  grading a styler trained on the same posts.
- **holdout** (`holdout_frac`/`holdout_posts`): hold out whole POSTs, report the
  single-split unseen-posts metric, ship the head fit on the train posts only —
  the leakage-safe configuration when the detector is used as a *reward*. The
  resolved partition is recorded in `meta.split` so styler-train + eval reuse it.

On a mixed corpus (real edit-pairs + `meta.synthetic` machine paraphrases) both
modes additionally facet the metric by provenance (`by_provenance.real` /
`.synthetic`): synthetic pairs may augment the *fit*, but "how well do we detect
our own paraphrase generator" must never masquerade as the honest number — that
is `by_provenance.real`.

Polarity: slop is the positive class (`LABEL_SLOP=1`), so
`predict_proba[:, slop] = P(slop)` and the served detector's ``score = P(slop)``
composes with `stylebot.eval.mean_detector_score`; `p_dan = 1 - score`.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from stylebot import classify
from stylebot.eval import extract_slop, extract_target
from stylebot.pairs import iter_pairs

logger = logging.getLogger(__name__)

# The default style backbone. Chosen by a measured bake-off on Dan's
# content-matched corpus (2026-06-30): StyleDistance 0.78 pairwise / 0.72 AUC,
# beating LUAR (0.76/0.71), Wegmann CISR (0.73/0.67) and the mxbai *semantic*
# baseline (0.75/0.62) — content-independent style embeddings hold where topic-
# dominated semantic geometry collapses. A different author should re-run that
# bake-off on their own pairs rather than trust this default blindly; whatever
# wins is pinned into the artifact's meta.json and enforced at serve time.
DEFAULT_EMBED_MODEL = "StyleDistance/styledistance"
NORMALIZE = True
SCHEMA_VERSION = 1

# Label encoding: slop = positive class (1) so predict_proba[:, 1] = P(slop).
LABEL_SLOP = 1
LABEL_DAN = 0  # the author class ("Dan" for historical reasons; generic in use)

# Leakage-safe evaluation defaults.
DEFAULT_TEST_SIZE = 0.25
DEFAULT_N_SPLITS = 8
DEFAULT_RANDOM_STATE = 0

# Regularization. C=None everywhere means "select by inner group-CV over C_GRID"
# (nested CV: selection happens strictly inside each training side, so the
# reported metric covers the tuning step and hand-sweeping --C against the
# printed number is never needed). DEFAULT_C is only the fallback when the fit
# pool has too few posts for an inner split.
DEFAULT_C = 1.0
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
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
        raise ImportError(_EXTRA_HINT) from exc
    return np


# ---------------------------------------------------------------------------
# Data assembly — learn STYLE not TOPIC (content-matched pairs, grouped by post)
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    """The assembled training matrix inputs, before embedding.

    `texts[i]` has label `labels[i]` and belongs to POST `groups[i]`. `pair_rows`
    lists `(slop_row, author_row)` index twins (the content-matched pairs) — the
    only rows the headline metric scores. `pair_synth[k]` marks pair `k` as
    synthetic (`meta.synthetic` — a machine paraphrase rather than a real edit
    capture), so metrics can facet real vs synthetic. `is_free[i]` marks
    free-standing positives (off by default); these augment the *fit* but never
    the metric, so their topic signal can't inflate the reported numbers.
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

    def _add_pair(self, slop: str, author: str, group: str, *, synthetic: bool = False) -> tuple[int, int]:
        s_row = self._add(slop, LABEL_SLOP, group)
        a_row = self._add(author, LABEL_DAN, group)
        self.pair_rows.append((s_row, a_row))
        self.pair_synth.append(synthetic)
        return s_row, a_row


def assemble_dataset(
    pairs_path: str | Path,
    *,
    extra_positives: Iterable[object] | None = None,
) -> Dataset:
    """Build the slop-vs-author dataset from a content-matched pairs corpus.

    Each pair contributes its slop side (`messages[1]`, label slop) and its
    author side (`messages[2]`, label author), heading-context stripped (via
    `eval.extract_slop`/`extract_target`), grouped by `meta.source` (the POST).
    `meta.synthetic` (truthy on `ai-style synth` output, absent on real edit
    captures) is carried into `pair_synth` so metrics can facet by provenance.

    `extra_positives` enriches the author class with free-standing prose the
    *caller* selected under its own policy — each item is a `synth.Target` (or
    any object with `.text` and `.source`) or a plain `(text, source)` tuple.
    They are flagged `is_free` and excluded from the headline metric, because
    free positives reintroduce a topic signal that would inflate it.
    """
    ds = Dataset()
    for rec in iter_pairs(pairs_path):
        slop = extract_slop(rec).strip()
        author = extract_target(rec).strip()
        if not slop or not author:
            continue
        meta = rec.get("meta") or {}
        source = meta.get("source") or "?"
        ds._add_pair(slop, author, source, synthetic=bool(meta.get("synthetic")))

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
    """Restrict a Dataset to the rows whose POST is in `keep_groups`.

    Returns `(sub, rows)` where `rows` are the kept original row indices in
    original order — so an embedding matrix subsets in lockstep: `X[rows]`
    aligns with `sub`. Twins always share a POST, so pairs survive whole.
    Used for role filtering *before* embedding: excluded posts (e.g. the frozen
    eval stratum) are never even encoded.
    """
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
    """Batch-encode `texts` with the style backbone; returns an [N, D] float32 array.

    Wraps `stylebot.classify.sentence_transformers_batch_embed_fn` — the ONE
    encode call-site shared with serving, so normalize/encode semantics can't
    drift between training and inference. Heavy imports stay inside the call so
    importing this module is cheap.
    """
    np = _np()
    try:
        encode = classify.sentence_transformers_batch_embed_fn(model_id, normalize=normalize)
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc
    return encode(texts, batch_size=batch_size).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Fit + leakage-safe evaluation
# ---------------------------------------------------------------------------


def _make_logreg(C: float = DEFAULT_C):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(_EXTRA_HINT) from exc

    # class_weight='balanced' guards the small/imbalanced negative set.
    return LogisticRegression(max_iter=2000, class_weight="balanced", C=C)


def _pairwise_score(p_slop, pairs, test_set: set) -> tuple[float, float]:
    """(correct, total) over the content-matched twins fully inside `test_set`.

    Correct = P(slop|slop_side) > P(slop|author_side); ties score 0.5.
    """
    correct = total = 0.0
    for s_row, a_row in pairs:
        if s_row in test_set and a_row in test_set:
            total += 1
            ps, pa = p_slop[s_row], p_slop[a_row]
            correct += 1.0 if ps > pa else (0.5 if ps == pa else 0.0)
    return correct, total


def select_C(X, ds: Dataset, rows, *, grid=C_GRID, n_inner: int = 4) -> tuple[float, dict]:
    """Pick C by inner GroupKFold over the posts within `rows` (the fit pool).

    Scored by content-matched pairwise accuracy on the validation twins — the
    same quantity as the headline metric. Ties break toward the smallest C
    (more regularization). With fewer than MIN_POSTS_FOR_C_SELECTION distinct
    posts an inner split is meaningless: falls back to DEFAULT_C and says so in
    the returned info (callers surface it).

    Callers use this strictly *inside* a training side (an outer fold's train
    posts, or the detector pool for the shipped head) — never on rows the
    resulting metric is reported over. That nesting is what keeps the reported
    number honest about the tuning step.
    """
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
        (rows_arr[tr], set(rows_arr[va].tolist()))
        for tr, va in gkf.split(X[rows_arr], y[rows_arr], groups[rows_arr])
    ]

    scores: dict[float, float | None] = {}
    best_C, best_score = DEFAULT_C, -1.0
    for C in grid:  # ascending grid + strict '>' ⇒ ties resolve to the smallest C
        fold_accs = []
        for fit_rows, val_set in folds:
            clf = _make_logreg(C)
            clf.fit(X[fit_rows], y[fit_rows])
            p_slop = clf.predict_proba(X)[:, list(clf.classes_).index(LABEL_SLOP)]
            correct, total = _pairwise_score(p_slop, ds.pair_rows, val_set)
            if total:
                fold_accs.append(correct / total)
        score = float(np.mean(fold_accs)) if fold_accs else None
        scores[C] = None if score is None else round(score, 4)
        if score is not None and score > best_score:
            best_C, best_score = C, score
    if best_score < 0:  # no fold had scorable twins
        return DEFAULT_C, {"mode": "fallback", "value": DEFAULT_C, "reason": "no scorable inner folds"}
    return best_C, {"mode": "nested_cv", "value": best_C, "grid": list(grid), "n_inner": n_folds, "scores": scores}


STRATA = ("real", "synthetic")


def _provenance_strata(ds: Dataset) -> tuple[dict[str, list[int]], dict[int, str]]:
    """`(stratum -> pair indices, matched row -> stratum)` for real/synthetic faceting.

    The stratum dict is empty unless the corpus is genuinely mixed — on an
    all-real (or all-synthetic) corpus the facet would just repeat the headline
    metric, so callers skip `by_provenance` entirely.
    """
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


def evaluate(
    X,
    ds: Dataset,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    C: float | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict:
    """POST-split content-matched pairwise accuracy + per-text AUC.

    For each `GroupShuffleSplit` fold (grouped by POST, so no post appears in both
    train and test), fit logreg on the train rows and, on the held-out side,
    measure: (1) **pairwise accuracy** — for each content-matched twin whose POST
    is in test, did `P(slop | slop_side) > P(slop | author_side)`; (2) **AUC**
    over the matched test rows. Free positives never enter the metric.

    `C=None` (the default) runs **nested CV**: each fold selects C by an inner
    GroupKFold over its *train* posts only (`select_C`), so the reported metric
    is honest about the tuning step. An explicit C skips selection.

    On a mixed real/synthetic corpus the result gains `by_provenance` — the same
    two metrics per stratum. The `real` facet is the honest number when synthetic
    pairs augment the fit: it can't be inflated by "detecting our own paraphrase
    generator".
    """
    np = _np()
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupShuffleSplit

    y = np.asarray(ds.labels)
    groups = np.asarray(ds.groups)
    matched_rows = {r for pair in ds.pair_rows for r in pair}
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

        test_set = set(test_idx.tolist())
        # AUC over matched test rows only (free positives excluded).
        matched_test = [r for r in test_idx.tolist() if r in matched_rows]
        ytest = y[matched_test]
        if len(set(ytest.tolist())) == 2:
            aucs.append(float(roc_auc_score(ytest, p_slop[matched_test])))
        for name in pair_idx:
            rows = [r for r in matched_test if row_stratum[r] == name]
            if len(set(y[rows].tolist())) == 2:
                facet_aucs[name].append(float(roc_auc_score(y[rows], p_slop[rows])))

        correct, total = _pairwise_score(p_slop, ds.pair_rows, test_set)
        if total:
            pair_accs.append(correct / total)
        for name in pair_idx:
            c, t = _pairwise_score(p_slop, (ds.pair_rows[k] for k in pair_idx[name]), test_set)
            if t:
                facet_pair_accs[name].append(c / t)

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
    holdout_frac: float = 0.0,
    holdout_seed: int = 0,
    holdout_posts=None,
) -> tuple[set, set]:
    """Split the distinct POSTs into `(train, holdout)` for the shared by-POST split.

    The leakage-safe contract: the detector must NOT be trained on the posts the
    styler trains on or that the final eval scores (see `_plans/eval-harness.md`
    "the split contract"). An explicit `holdout_posts` list pins the exact
    partition so all three stages (styler-train / detector-train / eval) can reuse
    it; otherwise the holdout is derived deterministically from `holdout_frac` +
    `holdout_seed`. `holdout_frac=0` and no list ⇒ no holdout (empty set).
    """
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
    """Single by-POST holdout eval: fit on the train posts, score the holdout twins.

    The honest "unseen posts" number for a head that ships fit on the train posts
    only — content-matched pairwise accuracy + AUC over the held-out posts. Like
    `evaluate`, gains a `by_provenance` real/synthetic facet on a mixed corpus.
    `C=None` selects C by inner group-CV over the *train* posts only (never the
    holdout), mirroring the nested contract.
    """
    np = _np()
    from sklearn.metrics import roc_auc_score

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

    def _auc(rows: list[int]) -> tuple[float | None, int]:
        if len(set(y[rows].tolist())) == 2:
            return round(float(roc_auc_score(y[rows], p_slop[rows])), 4), len(rows)
        return None, len(rows)

    matched = {r for pair in ds.pair_rows for r in pair}
    matched_test = [r for r in test_rows if r in matched]
    auc, n_auc = _auc(matched_test)

    test_set = set(test_rows)

    def _pairwise(pairs: Iterable[tuple[int, int]]) -> tuple[float | None, int]:
        correct, total = _pairwise_score(p_slop, pairs, test_set)
        return (round(correct / total, 4) if total else None), int(total)

    pa_mean, n_pairs = _pairwise(ds.pair_rows)
    result = {
        "pairwise_accuracy": {"mean": pa_mean, "std": None, "n": n_pairs},
        "auc": {"mean": auc, "std": None, "n": n_auc},
        "mode": "holdout",
        "C": C_info,
    }
    if pair_idx:
        by = {}
        for name in STRATA:
            f_pa, f_n = _pairwise(ds.pair_rows[k] for k in pair_idx[name])
            f_auc, f_n_auc = _auc([r for r in matched_test if row_stratum[r] == name])
            by[name] = {
                "pairwise_accuracy": {"mean": f_pa, "std": None, "n": f_n},
                "auc": {"mean": f_auc, "std": None, "n": f_n_auc},
                "n_pairs": len(pair_idx[name]),
            }
        result["by_provenance"] = by
    return result


def fit_head(X, ds: Dataset, *, C: float = DEFAULT_C, rows: list[int] | None = None):
    """Fit the final logreg (optionally on a row subset) and return `(clf, coef, intercept)`.

    `coef`/`intercept` are extracted for the P(slop) class so the pure-Python
    `stylebot.classify._logreg_p_slop` reproduces `clf.predict_proba[:, slop]`
    exactly. (Binary logreg keeps one weight row; we index it by the slop class.)

    `rows` restricts the fit to a subset of row indices — used by the held-out
    split mode to ship a head that has NOT seen the holdout posts.
    """
    np = _np()

    if rows is None:
        X_fit, y_fit = X, ds.labels
    else:
        X_fit, y_fit = X[rows], np.asarray(ds.labels)[rows]
    clf = _make_logreg(C)
    clf.fit(X_fit, y_fit)
    slop_col = list(clf.classes_).index(LABEL_SLOP)
    # Binary logistic regression stores a single weight row for the positive
    # class (classes_[1]); index defensively in case the row count ever differs.
    coef_row = clf.coef_[slop_col] if clf.coef_.shape[0] > 1 else clf.coef_[0]
    intercept = clf.intercept_[slop_col] if clf.intercept_.shape[0] > 1 else clf.intercept_[0]
    return clf, [float(c) for c in coef_row], float(intercept)


# ---------------------------------------------------------------------------
# Artifact I/O
# ---------------------------------------------------------------------------


def _git_sha(repo: str | Path | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or "."), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def write_artifact(
    out_dir: str | Path,
    *,
    clf,
    coef: list[float],
    intercept: float,
    ds: Dataset,
    metrics: dict,
    embed_model: str = DEFAULT_EMBED_MODEL,
    normalize: bool = NORMALIZE,
    threshold: float = 0.5,
    git_repo: str | Path | None = None,
    save_joblib: bool = False,
    split: dict | None = None,
    head_C: dict | None = None,
) -> Path:
    """Write `head.json` (the runtime contract) + `meta.json` [+ optional `model.joblib`].

    `head.json` is the plain-JSON linear head `stylebot.classify` serves with no
    ML deps; `meta.json` pins the embedder + records the held-out metrics. These
    two are the committable artifact: tiny, diff-able, version-independent, and
    the only way to use the detector without re-deriving it from the (private)
    corpus.

    `model.joblib` (the full sklearn estimator) is OFF by default: it is a
    redundant, sklearn-version-fragile binary pickle of the same weights that no
    code path loads (both `classify.sklearn_detector` and `--detector-model`
    read `head.json`).
    """
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
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(git_repo),
        "n_pairs": ds.n_pairs,
        "n_pairs_real": ds.n_pairs_real,
        "n_pairs_synthetic": ds.n_pairs_synthetic,
        "n_posts": ds.n_posts,
        "n_free_positives": sum(ds.is_free),
        "head_C": head_C,  # how the shipped head's regularization was chosen
        "split": split or {"mode": "fit_all", "note": (
            "head fit on ALL posts; honest for measuring an INDEPENDENT styler, but "
            "NOT for grading a styler trained on these posts — use holdout_frac/"
            "holdout_posts for the shared by-POST split when the detector is a reward."
        )},
        "metrics": metrics,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if save_joblib:
        import joblib

        joblib.dump(clf, out_dir / "model.joblib")
    return out_dir


def _resolve_head_C(X, ds: Dataset, rows, C: float | None) -> tuple[float, dict]:
    """The shipped head's C: the explicit override, or inner group-CV over `rows`."""
    if C is not None:
        return C, {"mode": "fixed", "value": C}
    return select_C(X, ds, rows)


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
    save_joblib: bool = False,
    holdout_frac: float = 0.0,
    holdout_seed: int = 0,
    holdout_posts=None,
    splits_path: str | Path | None = None,
) -> tuple[Path, dict]:
    """End-to-end: assemble → embed → evaluate → fit → write artifact.

    Returns `(artifact_dir, metrics)`. `C=None` (default) selects the
    regularization by inner group-CV everywhere (see `select_C`); an explicit C
    is an override, recorded as such. Three modes:

    - **default (no holdout):** cross-validated held-out metric over POST splits,
      then ship the head fit on ALL posts. Honest for *measuring* an independent
      styler; not for grading one trained on these posts.
    - **holdout** (`holdout_frac`>0 or `holdout_posts`): hold out a set of POSTs,
      report the single-split unseen-posts metric, and ship the head fit on the
      train posts ONLY.
    - **splits** (`splits_path`): the shared three-role contract
      (`stylebot.splits`). The head is fit on the **detector** pool only; the
      frozen **eval** posts are never even embedded; the **styler** posts supply
      a bonus unseen-posts metric (`metrics.styler_holdout`). Role counts, the
      danger-report warnings, and the rest-rule are recorded in `meta.split` so
      styler-train + eval consume the same partition.
    """
    ds = assemble_dataset(pairs_path, extra_positives=extra_positives)
    if ds.n_pairs == 0:
        raise ValueError(f"no content-matched pairs found in {pairs_path}")

    if splits_path is not None:
        if holdout_frac or holdout_posts:
            raise ValueError("splits_path is mutually exclusive with holdout_frac/holdout_posts")
        from stylebot import splits as splits_mod

        sp = splits_mod.load_splits(splits_path)
        warnings = splits_mod.check_splits(sp, ds)
        for w in warnings:
            logger.warning("splits: %s", w)
        role = {g: splits_mod.role_of(g, sp) for g in set(ds.groups)}
        det_groups = {g for g, r in role.items() if r == "detector"}
        sty_groups = {g for g, r in role.items() if r == "styler"}

        # Eval posts are excluded BEFORE embedding — the frozen stratum is never
        # touched by this trainer, not even to encode it.
        work_ds, _ = subset_dataset(ds, det_groups | sty_groups)
        det_ds, det_rows = subset_dataset(work_ds, det_groups)
        if det_ds.n_pairs == 0:
            raise ValueError("splits: the detector pool has no pairs — nothing to train on")
        logger.info(
            "embedding %d passages with %s (%d eval-role posts excluded)",
            len(work_ds.texts), embed_model, sum(1 for r in role.values() if r == "eval"),
        )
        X = embed_texts(work_ds.texts, model_id=embed_model, normalize=normalize, batch_size=batch_size)
        X_det = X[det_rows]

        metrics = evaluate(X_det, det_ds, test_size=test_size, n_splits=n_splits, C=C)
        head_C, head_C_info = _resolve_head_C(X_det, det_ds, list(range(len(det_ds.texts))), C)
        if any(g in sty_groups for g in work_ds.groups):
            # Bonus honest number: the detector-pool head scored on the styler's
            # (unseen) posts — the deployment-relevant condition for reward use.
            metrics["styler_holdout"] = evaluate_holdout(X, work_ds, sty_groups, C=head_C)
        clf, coef, intercept = fit_head(X_det, det_ds, C=head_C)
        split = {
            "mode": "splits",
            "path": str(splits_path),
            "seed": sp.get("seed"),
            "rest_rule": sp["rest_rule"],
            "roles": splits_mod.summarize_roles(sp, ds),
            "warnings": warnings,
        }
        out = write_artifact(
            out_dir, clf=clf, coef=coef, intercept=intercept, ds=det_ds, metrics=metrics,
            embed_model=embed_model, normalize=normalize, git_repo=git_repo,
            save_joblib=save_joblib, split=split, head_C=head_C_info,
        )
        return out, metrics

    logger.info("embedding %d passages with %s", len(ds.texts), embed_model)
    X = embed_texts(ds.texts, model_id=embed_model, normalize=normalize, batch_size=batch_size)

    train_posts, holdout = partition_posts(
        ds.groups, holdout_frac=holdout_frac, holdout_seed=holdout_seed, holdout_posts=holdout_posts
    )
    if holdout:
        metrics = evaluate_holdout(X, ds, holdout, C=C)
        train_rows = [i for i, g in enumerate(ds.groups) if g in train_posts]
        head_C, head_C_info = _resolve_head_C(X, ds, train_rows, C)
        clf, coef, intercept = fit_head(X, ds, C=head_C, rows=train_rows)
        split = {
            "mode": "holdout",
            "holdout_frac": holdout_frac,
            "holdout_seed": holdout_seed,
            "n_train_posts": len(train_posts),
            "n_holdout_posts": len(holdout),
            "holdout_posts": sorted(holdout),
        }
    else:
        metrics = evaluate(X, ds, test_size=test_size, n_splits=n_splits, C=C)
        head_C, head_C_info = _resolve_head_C(X, ds, list(range(len(ds.texts))), C)
        clf, coef, intercept = fit_head(X, ds, C=head_C)
        split = None
    out = write_artifact(
        out_dir, clf=clf, coef=coef, intercept=intercept, ds=ds, metrics=metrics,
        embed_model=embed_model, normalize=normalize, git_repo=git_repo,
        save_joblib=save_joblib, split=split, head_C=head_C_info,
    )
    return out, metrics
