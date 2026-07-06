"""Generic linear-head voice/style classifier — the runtime seam (mechanism).

This is the stylebot-side half of the trained Dan-voice classifier. It is
deliberately **ML-dependency-free at import**: scoring a passage is a plain dot
product of an embedding vector against a logistic-regression head stored as
plain JSON (`head.json` = a `coef` list + an `intercept`). No numpy, no sklearn,
no sentence-transformers needed to import or run the default path.

The split mirrors `synth.iter_targets` (mechanism) vs `livingthing.training_targets`
(policy): the *heavy* embedder and the Dan-specific trainer live in livingthing,
which composes its embedder with the JSON head and hands the finished detector to
stylebot's eval functions. stylebot never imports livingthing.

Two entry points:

- `embedding_classifier_detector(*, embed_fn, head, name, meta)` — the contract.
  Compose an injected `embed_fn` (str -> list[float]) with a loaded `head` into a
  `stylebot.eval.Detector`. This is what livingthing's `build_detector` uses, and
  what the dep-free unit test exercises with a stub `embed_fn`.
- `sklearn_detector(artifact_dir)` — a lazy, *self-contained* loader for the
  `ai-style eval --detector-model` path. It reads the artifact's `meta.json`,
  rebuilds the pinned embedder via `sentence-transformers`, and loads the head —
  all heavy imports happen INSIDE the function (the lazy-`openai` convention at
  `eval.py`), so importing this module stays dependency-free.

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
    """P(slop) for one embedding vector under a linear logistic head.

    ``sigmoid(intercept + coef · vec)``. Pure stdlib — the load-bearing reason
    stylebot needs no numpy/sklearn to *serve* the classifier.
    """
    z = intercept + math.fsum(v * c for v, c in zip(vec, coef))
    return _sigmoid(z)


def load_linear_head(path: str | Path) -> dict:
    """Load a `head.json` (stdlib json) and validate the linear-head contract.

    Returns the parsed dict with at least ``coef`` (list[float]) and ``intercept``
    (float); an optional ``calibration`` block is passed through untouched. Raises
    a clear ``ValueError`` if the required fields are missing or malformed, so a
    broken artifact fails loudly at load rather than silently scoring nonsense.
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

    `embed_fn` maps prose -> an embedding vector; `head` is a loaded `head.json`
    (`coef` + `intercept`). The returned callable embeds the passage and applies
    the logistic head, returning::

        {"score": P(slop), "p_dan": 1 - P(slop), "name", "configured": True,
         "embed_model": <meta.embed_model or None>}

    The embedding dimension MUST match ``len(head["coef"])`` — a mismatch means
    the wrong embedder was wired (e.g. mxbai head served with a style embedder),
    which would score garbage, so it raises ``ValueError`` rather than guessing.
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

    The heavy import lives inside this function (the lazy-`openai` convention in
    `eval.py`) so importing `stylebot.classify` never pulls in torch /
    sentence-transformers. Returns ``encode(texts, batch_size=32) -> [N, D] array``
    (numpy, as sentence-transformers produces). This is the **one** encode
    call-site for the classifier: the per-passage `embed_fn` below and
    livingthing's training-time `embed_texts` both wrap it, so normalize/encode
    semantics can't drift between training and serving.
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

    The model is loaded once and reused per call. Used by `sklearn_detector` for
    the self-contained `--detector-model` path; livingthing composes the same
    batch encoder for training.
    """
    encode = sentence_transformers_batch_embed_fn(model_id, normalize=normalize)

    def embed(prose: str) -> list[float]:
        return encode([prose])[0].tolist()

    return embed


def sklearn_detector(artifact_dir: str | Path, *, embed_fn: EmbedFn | None = None) -> Callable[[str], dict]:
    """Build a detector from a self-contained artifact directory (LAZY embedder).

    Reads ``<artifact_dir>/meta.json`` + ``head.json``; if no `embed_fn` is
    injected, rebuilds the pinned embedder from ``meta.embed_model`` /
    ``meta.normalize`` via sentence-transformers (heavy import deferred to call
    time). This backs ``ai-style eval --detector-model <dir>`` for operators who
    have the embedder installed; the dependency-free default path is livingthing's
    ``build_detector``, which injects its own embedder into
    `embedding_classifier_detector`.

    Named for the sklearn-trained head it serves; it loads the plain-JSON `head.json`
    (version-independent), not `model.joblib` (whose reuse is livingthing-side).
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
