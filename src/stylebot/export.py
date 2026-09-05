"""Merge a trained LoRA adapter back into its base weights (D31).

`train --download-adapter` writes a PEFT adapter in serving key format;
`tinker_cookbook.weights.build_hf_model` expects raw Tinker keys, so
`SERVING_TO_RAW` inverts them and the merge emits one HF checkpoint a local
runtime takes from there. Quantization, config surgery and chat-template
edits are per-runtime policy and stay with the caller. Heavy imports are lazy.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path

from stylebot.errors import MissingExtra, RunError

logger = logging.getLogger(__name__)

_EXTRA_HINT = (
    "merging needs the 'trainer' extra: run `uv add 'stylebot[trainer]'` "
    "(or `pip install 'stylebot[trainer]'`) for tinker-cookbook and its "
    "safetensors/huggingface-hub dependencies."
)

ADAPTER_WEIGHTS = "adapter_model.safetensors"
ADAPTER_CONFIG = "adapter_config.json"

#: Serving-format prefix -> raw-Tinker prefix, applied longest-match-first by
#: `raw_adapter_key`; safe because the planner validates every mapped key and
#: `check_adapter_scaling` refuses the one case where a rename changes meaning.
SERVING_TO_RAW: tuple[tuple[str, str], ...] = (
    ("base_model.model.model.language_model.", "base_model.model.model."),
    ("base_model.model.lm_head.", "base_model.model.model.unembed_tokens."),
)


# ---------------------------------------------------------------------------
# Key inversion — pure, no dependencies, the part worth testing
# ---------------------------------------------------------------------------


def raw_adapter_key(key: str) -> str:
    """One adapter tensor key in raw-Tinker form; unchanged if already raw."""
    for old, new in sorted(SERVING_TO_RAW, key=lambda pair: -len(pair[0])):
        if key.startswith(old):
            return new + key[len(old) :]
    return key


def is_serving_format(keys: Iterable[str]) -> bool:
    """True if any key carries a serving-format prefix, i.e. needs inverting."""
    prefixes = tuple(old for old, _ in SERVING_TO_RAW)
    return any(key.startswith(prefixes) for key in keys)


def check_adapter_scaling(config: dict) -> None:
    """Refuse an adapter whose `lora_alpha != r`: the rename moves where a
    tensor lands, not what it means, only while the scaling factor is 1."""
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
        raise MissingExtra(_EXTRA_HINT) from exc

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
    """The local HF-cache snapshot dir for `base_model`, else the hub name with
    a warning: a local directory bypasses `transformers.AutoConfig`, which may
    not know a just-released architecture."""
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
    """Merge the adapter into `base_model`, writing a merged HF checkpoint via a
    sibling `.partial` directory; an existing `out_dir` is returned untouched.
    `work_dir` holds the key-inverted copy; `merge_fn` is the paid seam."""
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
            raise MissingExtra(_EXTRA_HINT) from exc

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
