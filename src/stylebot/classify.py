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
from collections.abc import Callable, Sequence
from pathlib import Path

from stylebot.errors import RunError

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
