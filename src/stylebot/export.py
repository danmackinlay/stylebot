"""Merge a trained LoRA adapter back into its base weights.

The step between `ai-style train` and serving the styler off somebody else's
computer: `train --download-adapter` writes a PEFT adapter, and this turns that
plus the base model into one merged HF checkpoint that a local runtime
(`mlx_lm.convert`, llama.cpp, vLLM, plain transformers) can take from there.

**Why this is stylebot's problem and not the caller's.** The adapter
`train.py` exports is in *serving* key format, and
`tinker_cookbook.weights.build_hf_model` — the merge it is meant to be fed to —
expects *raw* Tinker keys and applies that same remap itself. Hand it stylebot's
own output and the `language_model` prefix doubles and the merge dies with
`WeightsMergeError`. So stylebot emits an artifact its own documented next step
cannot consume, and every user hits that wall. `SERVING_TO_RAW` below is the
inversion; it is the reason this module exists.

**What stays with the caller.** Where manifests, adapters and outputs live, and
what the merged model is named, are policy — this module takes explicit paths
like every other phase. So is anything downstream of the merge: quantization
settings, the config-key surgery a particular server needs, chat-template
edits. Those are per-runtime and per-model, and baking one server's quirks in
here is exactly the blog/generic boundary CLAUDE.md warns about.

Heavy imports (`safetensors`, `huggingface_hub`, `tinker_cookbook`) are lazy, so
importing this module costs nothing and the key-inversion logic above stays
testable without the ``trainer`` extra.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from stylebot.errors import RunError

logger = logging.getLogger(__name__)

_EXTRA_HINT = (
    "merging needs the 'trainer' extra: run `uv add 'stylebot[trainer]'` "
    "(or `pip install 'stylebot[trainer]'`) for tinker-cookbook and its "
    "safetensors/huggingface-hub dependencies."
)

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

#: Serving-format prefix -> raw-Tinker prefix, applied longest-match-first by
#: `raw_adapter_key`. Inverts the remap `tinker_cookbook.weights` performs when
#: it builds the serving adapter, so the merge planner sees the key convention
#: it expects (`model.layers.*`, `model.unembed_tokens`).
#:
#: Safe because the planner validates every mapped key against the base state
#: dict — a wrong rename raises rather than silently corrupting weights — and
#: because `merge_adapter` refuses an adapter whose `lora_alpha != r` (see
#: `check_adapter_scaling`), which is the one case where a rename could change
#: what the merged weights mean rather than just where they land.
SERVING_TO_RAW: tuple[tuple[str, str], ...] = (
    ("base_model.model.model.language_model.", "base_model.model.model."),
    ("base_model.model.lm_head.", "base_model.model.model.unembed_tokens."),
)


# ---------------------------------------------------------------------------
# Key inversion — pure, no dependencies, the part worth testing
# ---------------------------------------------------------------------------


def raw_adapter_key(key: str) -> str:
    """One adapter tensor key in raw-Tinker form; unchanged if already raw.

    At most one prefix applies (they are disjoint), and the longest is tried
    first so a future overlapping entry cannot silently half-match.
    """
    for old, new in sorted(SERVING_TO_RAW, key=lambda pair: -len(pair[0])):
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def is_serving_format(keys: Iterable[str]) -> bool:
    """True if any key carries a serving-format prefix, i.e. needs inverting."""
    prefixes = tuple(old for old, _ in SERVING_TO_RAW)
    return any(key.startswith(prefixes) for key in keys)


def check_adapter_scaling(config: dict) -> None:
    """Refuse an adapter whose LoRA scaling the rename would misrepresent.

    The inversion moves where a tensor lands, not what it means — provided
    `lora_alpha == r`, so the scaling factor `alpha / r` is 1 and no value is
    at stake. Every adapter `stylebot.train` produces satisfies that (the
    cookbook's SFT recipe sets them equal), but an adapter from elsewhere need
    not, and silently merging one at the wrong scale would produce a model that
    is subtly, unfixably wrong rather than broken. So it is checked, not
    assumed.
    """
    r, alpha = config.get("r"), config.get("lora_alpha")
    if r is None or alpha is None:
        raise RunError(
            f"{ADAPTER_CONFIG} is missing 'r' or 'lora_alpha', so the LoRA scaling "
            "cannot be verified — refusing to merge an adapter of unknown scale"
        )
    if r != alpha:
        raise RunError(
            f"adapter has lora_alpha={alpha} != r={r} (scaling {alpha / r:g}), but the "
            "serving->raw key inversion assumes a scaling of 1. Merging would apply the "
            "wrong scale. Merge it with PEFT directly, or re-export with alpha == r."
        )


# ---------------------------------------------------------------------------
# The merge
# ---------------------------------------------------------------------------


def normalize_adapter(adapter_dir: Path | str, out_dir: Path | str) -> Path:
    """A raw-Tinker-keyed copy of a PEFT adapter, for the cookbook merge.

    Returns `adapter_dir` unchanged when the keys are already raw (nothing to
    do, nothing copied). Otherwise writes the renamed weights plus the original
    `adapter_config.json` to `out_dir` and returns that.
    """
    adapter_dir = Path(adapter_dir)
    weights = adapter_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise RunError(f"no {ADAPTER_WEIGHTS} in {adapter_dir}")
    config_path = adapter_dir / ADAPTER_CONFIG
    if not config_path.is_file():
        raise RunError(f"no {ADAPTER_CONFIG} in {adapter_dir}")
    check_adapter_scaling(json.loads(config_path.read_text(encoding="utf-8")))

    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:  # pragma: no cover - needs the extra
        raise ImportError(_EXTRA_HINT) from exc

    tensors = load_file(str(weights))
    if not is_serving_format(tensors):
        logger.info("adapter keys are already raw-Tinker; no inversion needed")
        return adapter_dir

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    save_file({raw_adapter_key(k): v for k, v in tensors.items()}, str(out_dir / ADAPTER_WEIGHTS))
    shutil.copyfile(config_path, out_dir / ADAPTER_CONFIG)
    logger.info("serving-format adapter keys inverted to raw -> %s", out_dir)
    return out_dir


def resolve_base_model(base_model: str) -> str:
    """The local HF-cache snapshot dir for `base_model`, else the hub name.

    Load-bearing rather than an optimization: given a local directory,
    `tinker_cookbook.weights` reads `config.json` itself and never consults
    `transformers.AutoConfig`, which does not necessarily know a
    just-released architecture and raises when it doesn't. The base weights are
    normally already cached from the training export. Falls back to the hub
    name with a warning, since that path does work for architectures
    transformers knows.
    """
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(base_model, local_files_only=True)
    except Exception:
        logger.warning(
            "%s is not in the local HF cache; passing the hub name through, which "
            "fails if transformers does not know the architecture",
            base_model,
        )
        return base_model


def base_model_of(manifest: dict) -> str:
    """The base model a training manifest was trained against."""
    base_model = (manifest.get("hyperparameters") or {}).get("base_model")
    if not base_model:
        raise RunError("manifest has no hyperparameters.base_model")
    return str(base_model)


def merge_adapter(
    adapter_dir: Path | str,
    out_dir: Path | str,
    *,
    base_model: str,
    work_dir: Path | str | None = None,
    merge_fn: Callable[..., None] | None = None,
) -> Path:
    """Merge the adapter into `base_model`, writing a merged HF checkpoint.

    Idempotent in the way the rest of the pipeline is: the merge writes to a
    sibling `.partial` directory and renames on success, so an interrupted run
    leaves debris rather than a half-written model that looks finished. An
    existing `out_dir` is returned untouched — delete it to force a re-merge.

    `work_dir` holds the key-inverted adapter copy (default: beside `out_dir`).
    `merge_fn` is the paid-execution seam, defaulting to
    `tinker_cookbook.weights.build_hf_model`; tests inject a fake so the whole
    path above is exercised without torch or a 9B download.
    """
    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    if (out_dir / "config.json").is_file():
        logger.info("%s already merged — nothing to do", out_dir)
        return out_dir

    work_dir = Path(work_dir) if work_dir is not None else out_dir.parent
    raw_adapter = normalize_adapter(adapter_dir, work_dir / f"{out_dir.name}-adapter-raw")

    if merge_fn is None:
        try:
            from tinker_cookbook.weights import build_hf_model as merge_fn  # noqa: N813
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ImportError(_EXTRA_HINT) from exc

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.parent.mkdir(parents=True, exist_ok=True)
    logger.info("merging %s into %s -> %s", raw_adapter, base_model, out_dir)
    merge_fn(
        base_model=resolve_base_model(base_model),
        adapter_path=str(raw_adapter),
        output_path=str(partial),
        trust_remote_code=True,
    )
    if not (partial / "config.json").is_file():
        raise RunError(f"the merge left no config.json in {partial} — inspect it")
    partial.rename(out_dir)

    if raw_adapter != adapter_dir:
        shutil.rmtree(raw_adapter, ignore_errors=True)
    return out_dir
