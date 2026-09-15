"""Linear-head voice classifier — the runtime seam (D7).

Scoring is a dot product of an embedding against a logistic head stored as
plain JSON (`head.json`: `coef` + `intercept`), so the module imports with no
ML dependency; heavy imports live inside functions. Entry points:
`embedding_classifier_detector` (inject an `embed_fn`) and `sklearn_detector`
(rebuild the pinned embedder from an artifact dir). Polarity: ``score =
P(slop)``, higher more AI-like; ``p_dan = 1 - P(slop)`` beside it.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

from stylebot.errors import ConfigError, RunError

# An embedder: passage -> fixed-length vector. Injected, never imported here.
EmbedFn = Callable[[str], Sequence[float]]

#: What identifies the instrument on a score, as opposed to the score (D39);
#: `git_sha` / `trained_at` / `n_pairs` distinguish two refits of one recipe.
DETECTOR_IDENTITY_KEYS = ("name", "embed_model", "git_sha", "trained_at", "n_pairs")


def _sigmoid(z: float) -> float:
    """Numerically-stable logistic sigmoid (no overflow on large |z|)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logreg_p_slop(vec: Sequence[float], coef: Sequence[float], intercept: float) -> float:
    """P(slop) for one embedding vector: ``sigmoid(intercept + coef · vec)``, stdlib."""
    # strict=False preserves truncate-on-mismatch; strict=True is a behaviour change.
    z = intercept + math.fsum(v * c for v, c in zip(vec, coef, strict=False))
    return _sigmoid(z)


def load_linear_head(path: str | Path) -> dict:
    """Load a `head.json` and validate the linear-head contract: ``coef``
    (list[float]) and ``intercept`` (float), ``calibration`` passed through.
    Raises ``ValueError`` on a malformed field."""
    path = Path(path)
    with path.open(encoding="utf-8") as fp:
        head = json.load(fp)
    if not isinstance(head, dict):
        raise RunError(f"{path}: head.json must be a JSON object")
    coef = head.get("coef")
    intercept = head.get("intercept")
    if not isinstance(coef, list) or not coef or not all(isinstance(c, (int, float)) for c in coef):
        raise RunError(f"{path}: head.json 'coef' must be a non-empty list of numbers")
    if not isinstance(intercept, (int, float)):
        raise RunError(f"{path}: head.json 'intercept' must be a number")
    return head


def embedding_classifier_detector(
    *,
    embed_fn: EmbedFn,
    head: dict,
    name: str = "voice-clf",
    meta: dict | None = None,
    batch_embed_fn: Callable[..., object] | None = None,
) -> Callable[[str], dict]:
    """Compose an embedder + a linear head into a `stylebot.eval.Detector`.

    Returns ``{"score": P(slop), "p_dan", "configured"}`` plus every
    `DETECTOR_IDENTITY_KEYS` field `meta` answers (D39); the identity is also
    attached as ``.identity``. Raises ``ValueError`` when the embedding
    dimension mismatches ``len(head["coef"])``. Given `batch_embed_fn` it
    carries ``.batch_detect(texts)`` (D59), reached through `detect_many`.
    """
    coef = head["coef"]
    intercept = float(head["intercept"])
    dim = len(coef)
    meta = meta or {}
    embed_model = meta.get("embed_model")
    identity = {key: meta.get(key) for key in DETECTOR_IDENTITY_KEYS}
    identity["name"] = name

    def _from_vec(vec) -> dict:
        if len(vec) != dim:
            raise RunError(
                f"embedding dim {len(vec)} != head dim {dim} "
                f"(embed_model={embed_model!r}); the embedder does not match the trained head"
            )
        p_slop = _logreg_p_slop(vec, coef, intercept)
        return {
            "score": p_slop,
            "p_dan": 1.0 - p_slop,
            "configured": True,
            **identity,
        }

    def detect(prose: str) -> dict:
        return _from_vec(embed_fn(prose))

    detect.identity = dict(identity)
    if batch_embed_fn is not None:

        def batch_detect(texts: Sequence[str]) -> list[dict]:
            texts = list(texts)
            if not texts:
                return []
            return [_from_vec(vec) for vec in batch_embed_fn(texts)]

        detect.batch_detect = batch_detect
    return detect


def detect_many(detector: Callable[[str], dict], texts: Sequence[str]) -> list[dict]:
    """Score many passages with `detector`: through ``.batch_detect`` when it
    carries one, else the obvious loop."""
    texts = list(texts)
    batch = getattr(detector, "batch_detect", None)
    if batch is None:
        return [detector(text) for text in texts]
    return list(batch(texts))


def load_artifact_meta(artifact_dir: str | Path) -> dict:
    """Read an artifact's `meta.json` (embed_model, embed_dim, normalize, …)."""
    meta_path = Path(artifact_dir) / "meta.json"
    with meta_path.open(encoding="utf-8") as fp:
        meta = json.load(fp)
    if not isinstance(meta, dict) or not meta.get("embed_model"):
        raise RunError(f"{meta_path}: meta.json must carry an 'embed_model' id")
    return meta


def sentence_transformers_batch_embed_fn(model_id: str, *, normalize: bool = True):
    """Build a batch encoder backed by a sentence-transformers model (lazy import).

    Returns ``encode(texts, batch_size=32) -> [N, D] numpy array``: the one
    encode call-site, wrapped by serving and training alike. Single-threaded
    by contract and unguarded (D59).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)

    def encode(texts: Sequence[str], *, batch_size: int = 32):
        return model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

    return encode


def sentence_transformers_embed_fn(model_id: str, *, normalize: bool = True) -> EmbedFn:
    """A per-passage `embed_fn` over the shared batch encoder (lazy import)."""
    encode = sentence_transformers_batch_embed_fn(model_id, normalize=normalize)

    def embed(prose: str) -> list[float]:
        return encode([prose])[0].tolist()

    # The batch encoder underneath, for `sklearn_detector` -> `.batch_detect`.
    embed.batch = encode
    return embed


def sklearn_detector(
    artifact_dir: str | Path, *, embed_fn: EmbedFn | None = None
) -> Callable[[str], dict]:
    """Build a detector from an artifact directory: ``meta.json`` + ``head.json``,
    the pinned embedder rebuilt lazily unless `embed_fn` is injected. No sklearn."""
    artifact_dir = Path(artifact_dir)
    meta = load_artifact_meta(artifact_dir)
    head = load_linear_head(artifact_dir / "head.json")
    if embed_fn is None:
        embed_fn = sentence_transformers_embed_fn(
            meta["embed_model"], normalize=bool(meta.get("normalize", True))
        )
    return embedding_classifier_detector(
        embed_fn=embed_fn,
        head=head,
        name=meta.get("name", "voice-clf"),
        meta=meta,
        # Absent on an injected per-passage `embed_fn`: no batch path.
        batch_embed_fn=getattr(embed_fn, "batch", None),
    )


# ---------------------------------------------------------------------------
# Marker-threshold calibration (D105)
# ---------------------------------------------------------------------------

#: One `(side, percentile)` per ascending bucket edge, sides `dan` and `slop`:
#: an example the caller re-pins as policy (D43), not a gate.
DEFAULT_THRESHOLD_RULE: tuple[tuple[str, int], ...] = (
    ("dan", 75), ("slop", 50), ("slop", 75), ("slop", 95),
)  # fmt: skip
DEFAULT_THRESHOLD_RULE_SPEC = "dan:75,slop:50,slop:75,slop:95"
#: The percentiles every calibration reports, the rule's added.
QUANTILE_POINTS = (5, 25, 50, 75, 95)
SIDES = ("dan", "slop")


def parse_threshold_rule(spec: str) -> tuple[tuple[str, int], ...]:
    """`"dan:75,slop:50,…"` -> the rule; a side outside `SIDES` or a percentile
    outside 1–99 is a `ConfigError`."""
    rule = []
    for item in spec.split(","):
        side, _, pct = item.strip().partition(":")
        if side not in SIDES or not pct.isdigit() or not 1 <= int(pct) <= 99:
            raise ConfigError(
                f"threshold rule item {item!r}: want side:percentile, side dan|slop, 1–99"
            )
        rule.append((side, int(pct)))
    return tuple(rule)


def _percentiles(scores: list[float], points) -> dict[int, float]:
    """Inclusive percentiles at `points`; one score answers every point."""
    if len(scores) == 1:
        return dict.fromkeys(points, scores[0])
    cuts = statistics.quantiles(scores, n=100, method="inclusive")
    return {p: cuts[p - 1] for p in points}


def calibrate_thresholds(
    detector: Callable[[str], dict],
    pairs_path: str | Path,
    *,
    splits: dict | None = None,
    role: str = "eval",
    rule=DEFAULT_THRESHOLD_RULE,
    max_transform_sim: float | None = 0.85,
) -> dict:
    """Quantile-cut marker thresholds (D105): score the real pairs of one splits
    `role` (every real pair without `splits`), gated as the detector's training
    is (D18/D104), and cut one edge per `(side, percentile)` of `rule`. Returns
    the instrument identity, the population, both sides' percentiles at
    `QUANTILE_POINTS` plus the rule's, `thresholds`, and whether they ascend."""
    from stylebot.classify_train import LABEL_SLOP, assemble_dataset
    from stylebot.splits import role_of

    ds = assemble_dataset(pairs_path, max_transform_sim=max_transform_sim)
    rows: list[int] = []
    for k, (s_row, a_row) in enumerate(ds.pair_rows):
        if ds.pair_is_synthetic(k):
            continue
        if splits is not None and role_of(ds.groups[s_row], splits) != role:
            continue
        rows.extend((s_row, a_row))
    if not rows:
        where = f"the {role!r} role of" if splits is not None else ""
        raise ConfigError(
            f"no real pairs in {where} {pairs_path} survive the gate; nothing to calibrate"
        )
    scored = detect_many(detector, [ds.texts[r] for r in rows])
    by_side: dict[str, list[float]] = {side: [] for side in SIDES}
    for r, rec in zip(rows, scored, strict=True):
        by_side["slop" if ds.labels[r] == LABEL_SLOP else "dan"].append(float(rec["score"]))
    points = sorted({*QUANTILE_POINTS, *(p for _, p in rule)})
    quantiles = {side: _percentiles(sorted(v), points) for side, v in by_side.items()}
    thresholds = [round(quantiles[side][p], 2) for side, p in rule]
    return {
        "instrument": dict(getattr(detector, "identity", {}) or {}),
        "role": role if splits is not None else None,
        "n_pairs": len(rows) // 2,
        "n_posts": len({ds.groups[r] for r in rows}),
        "max_transform_sim": max_transform_sim,
        "points": points,
        "quantiles": quantiles,
        "rule": [list(item) for item in rule],
        "thresholds": thresholds,
        "ascending": thresholds == sorted(thresholds),
    }


def format_calibration(cal: dict) -> list[str]:
    """The calibration as terminal lines: instrument and population, a
    percentile table, the edges and the rule they were cut by."""
    ident = cal["instrument"]
    role = f"{cal['role']} role, " if cal["role"] else ""
    gate = (
        f"transform_sim ≤ {cal['max_transform_sim']}"
        if cal["max_transform_sim"] is not None
        else "ungated"
    )
    who = f"{ident.get('name', '?')}@{(ident.get('git_sha') or '?')[:10]}"
    lines = [
        f"{who} on {cal['n_pairs']} real pairs ({role}{gate}), {cal['n_posts']} posts",
        "side  " + "  ".join(f"p{p:<3}" for p in cal["points"]),
    ]
    for side in SIDES:
        lines.append(
            f"{side:<5} " + "  ".join(f"{cal['quantiles'][side][p]:.3f}" for p in cal["points"])
        )
    rule = ", ".join(f"{side} p{p}" for side, p in cal["rule"])
    lines.append(f"thresholds {cal['thresholds']}  = {rule}")
    if not cal["ascending"]:
        lines.append(
            "WARNING: the edges do not ascend; the rule does not fit this score distribution"
        )
    return lines
