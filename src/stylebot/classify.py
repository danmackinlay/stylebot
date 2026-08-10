"""Generic linear-head voice/style classifier — the runtime seam (mechanism).

The serving half of the trained voice classifier; the trainer that produces the
artifact is `stylebot.classify_train` (behind the ``classifier`` extra). This
module is deliberately **ML-dependency-free at import**: scoring a passage is a
plain dot product of an embedding vector against a logistic-regression head
stored as plain JSON (`head.json` = a `coef` list + an `intercept`). No numpy,
no sklearn, no sentence-transformers needed to import or run the default path.

Two entry points: `embedding_classifier_detector` (inject your own `embed_fn`)
and `sklearn_detector` (rebuild the pinned embedder from an artifact dir). Every
heavy import happens INSIDE a function, so importing this module stays
dependency-free. stylebot never imports livingthing.

**Polarity.** The detector returns ``score = P(slop)`` (higher = more AI-like),
matching stylebot's detector convention (`eval.null_detector`,
`mean_detector_score`) so it composes with the existing aggregation unchanged.
It *also* returns ``p_dan = 1 - P(slop)`` explicitly, so reward callers
(Phase-4 best-of-N, Phase-3 weighting) never have to flip a sign.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path

# An embedder: maps a passage to a fixed-length vector. Injected, never imported
# here — livingthing supplies a sentence-transformers one; tests supply a stub.
EmbedFn = Callable[[str], Sequence[float]]


def _sigmoid(z: float) -> float:
    """Numerically-stable logistic sigmoid (no overflow on large |z|)."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logreg_p_slop(vec: Sequence[float], coef: Sequence[float], intercept: float) -> float:
    """P(slop) for one embedding vector: ``sigmoid(intercept + coef · vec)``.

    Pure stdlib — the load-bearing reason stylebot needs no numpy/sklearn to
    *serve* the classifier.
    """
    # strict=False preserves today's truncate-on-mismatch; see B905 note in the
    # lint sweep — tightening this to strict=True is a behaviour change.
    z = intercept + math.fsum(v * c for v, c in zip(vec, coef, strict=False))
    return _sigmoid(z)


def load_linear_head(path: str | Path) -> dict:
    """Load a `head.json` and validate the linear-head contract.

    Returns the parsed dict with at least ``coef`` (list[float]) and
    ``intercept`` (float); an optional ``calibration`` block passes through
    untouched. Raises ``ValueError`` on missing/malformed fields, so a broken
    artifact fails loudly at load rather than silently scoring nonsense.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fp:
        head = json.load(fp)
    if not isinstance(head, dict):
        raise ValueError(f"{path}: head.json must be a JSON object")
    coef = head.get("coef")
    intercept = head.get("intercept")
    if not isinstance(coef, list) or not coef or not all(isinstance(c, (int, float)) for c in coef):
        raise ValueError(f"{path}: head.json 'coef' must be a non-empty list of numbers")
    if not isinstance(intercept, (int, float)):
        raise ValueError(f"{path}: head.json 'intercept' must be a number")
    return head


def embedding_classifier_detector(
    *,
    embed_fn: EmbedFn,
    head: dict,
    name: str = "voice-clf",
    meta: dict | None = None,
) -> Callable[[str], dict]:
    """Compose an embedder + a linear head into a `stylebot.eval.Detector`.

    The returned callable embeds the passage, applies the logistic head, and
    returns ``{"score": P(slop), "p_dan": 1 - P(slop), "name", "configured",
    "embed_model"}``.

    The embedding dimension MUST match ``len(head["coef"])`` — a mismatch means
    the wrong embedder was wired, which would score garbage, so it raises
    ``ValueError`` rather than guessing.
    """
    coef = head["coef"]
    intercept = float(head["intercept"])
    dim = len(coef)
    embed_model = (meta or {}).get("embed_model")

    def detect(prose: str) -> dict:
        vec = embed_fn(prose)
        if len(vec) != dim:
            raise ValueError(
                f"embedding dim {len(vec)} != head dim {dim} "
                f"(embed_model={embed_model!r}); the embedder does not match the trained head"
            )
        p_slop = _logreg_p_slop(vec, coef, intercept)
        return {
            "score": p_slop,
            "p_dan": 1.0 - p_slop,
            "name": name,
            "configured": True,
            "embed_model": embed_model,
        }

    return detect


def load_artifact_meta(artifact_dir: str | Path) -> dict:
    """Read an artifact's `meta.json` (embed_model, embed_dim, normalize, …)."""
    meta_path = Path(artifact_dir) / "meta.json"
    with meta_path.open(encoding="utf-8") as fp:
        meta = json.load(fp)
    if not isinstance(meta, dict) or not meta.get("embed_model"):
        raise ValueError(f"{meta_path}: meta.json must carry an 'embed_model' id")
    return meta


def sentence_transformers_batch_embed_fn(model_id: str, *, normalize: bool = True):
    """Build a batch encoder backed by a sentence-transformers model (LAZY import).

    Returns ``encode(texts, batch_size=32) -> [N, D] numpy array``. The **one**
    encode call-site for the classifier: the per-passage `embed_fn` below and
    the training-time `classify_train.embed_texts` both wrap it, so
    normalize/encode semantics can't drift between training and serving.
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
    """A per-passage `embed_fn` over the shared batch encoder (LAZY import).

    The model is loaded once and reused per call.
    """
    encode = sentence_transformers_batch_embed_fn(model_id, normalize=normalize)

    def embed(prose: str) -> list[float]:
        return encode([prose])[0].tolist()

    return embed


def sklearn_detector(
    artifact_dir: str | Path, *, embed_fn: EmbedFn | None = None
) -> Callable[[str], dict]:
    """Build a detector from a self-contained artifact directory (LAZY embedder).

    Reads ``<artifact_dir>/meta.json`` + ``head.json``; if no `embed_fn` is
    injected, rebuilds the pinned embedder from ``meta.embed_model`` /
    ``meta.normalize`` via sentence-transformers. Named for the sklearn-trained
    head it serves, though it needs no sklearn to serve it: the weights arrive
    as the plain-JSON `head.json`, so no estimator pickle has to load here.
    """
    artifact_dir = Path(artifact_dir)
    meta = load_artifact_meta(artifact_dir)
    head = load_linear_head(artifact_dir / "head.json")
    if embed_fn is None:
        embed_fn = sentence_transformers_embed_fn(
            meta["embed_model"], normalize=bool(meta.get("normalize", True))
        )
    return embedding_classifier_detector(
        embed_fn=embed_fn, head=head, name=meta.get("name", "voice-clf"), meta=meta
    )
