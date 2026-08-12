"""Merging a trained LoRA adapter back into its base weights.

The key-inversion logic is pure and always runs; the tests that touch real
`.safetensors` files skip without the ``trainer`` extra (safetensors/torch are
not in the dev group — no 9B downloads in this suite). The merge itself is
never executed: `merge_adapter` takes a `merge_fn` seam and gets a fake, so
everything around the merge is exercised without torch loading a base model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylebot import export

SERVING_KEYS = {
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": 1,
    "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": 2,
    "base_model.model.lm_head.lora_A.weight": 3,
}


# --- key inversion (pure — no extra needed) --------------------------------


def test_language_model_prefix_is_stripped():
    got = export.raw_adapter_key(
        "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight"
    )
    assert got == "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"


def test_lm_head_becomes_unembed_tokens():
    got = export.raw_adapter_key("base_model.model.lm_head.lora_A.weight")
    assert got == "base_model.model.model.unembed_tokens.lora_A.weight"


def test_raw_keys_pass_through_unchanged():
    raw = "base_model.model.model.layers.3.mlp.up_proj.lora_B.weight"
    assert export.raw_adapter_key(raw) == raw


def test_inversion_is_idempotent():
    """Applying it twice must not double-strip — the guard against a re-run
    quietly mangling an already-normalised adapter."""
    once = export.raw_adapter_key(next(iter(SERVING_KEYS)))
    assert export.raw_adapter_key(once) == once


def test_is_serving_format_detects_either_prefix():
    assert export.is_serving_format(SERVING_KEYS)
    assert export.is_serving_format(["base_model.model.lm_head.lora_A.weight"])
    assert not export.is_serving_format(
        ["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    )
    assert not export.is_serving_format([])


# --- the scaling guard -----------------------------------------------------


def test_equal_alpha_and_r_is_accepted():
    export.check_adapter_scaling({"r": 32, "lora_alpha": 32})  # no raise


def test_unequal_scaling_is_refused_and_says_why():
    with pytest.raises(ValueError, match="lora_alpha=64 != r=32"):
        export.check_adapter_scaling({"r": 32, "lora_alpha": 64})


def test_missing_scaling_fields_are_refused():
    with pytest.raises(ValueError, match="cannot be verified"):
        export.check_adapter_scaling({"r": 32})


def test_base_model_of_manifest():
    assert export.base_model_of({"hyperparameters": {"base_model": "Qwen/Qwen3.5-9B"}}) == (
        "Qwen/Qwen3.5-9B"
    )
    with pytest.raises(ValueError, match="hyperparameters.base_model"):
        export.base_model_of({"hyperparameters": {}})


# --- adapter normalisation + merge (needs safetensors) ---------------------


def _adapter(tmp_path, keys, *, r=32, alpha=32):
    """A minimal on-disk PEFT adapter with `keys` as its tensor names."""
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    d = tmp_path / "adapter"
    d.mkdir()
    st.save_file({k: torch.zeros(2, 2) for k in keys}, str(d / export.ADAPTER_WEIGHTS))
    (d / export.ADAPTER_CONFIG).write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "peft_type": "LORA"}), encoding="utf-8"
    )
    return d


def _keys_of(path):
    st = pytest.importorskip("safetensors.torch")
    return set(st.load_file(str(path / export.ADAPTER_WEIGHTS)))


def test_normalize_writes_an_inverted_copy(tmp_path):
    src = _adapter(tmp_path, SERVING_KEYS)
    out = export.normalize_adapter(src, tmp_path / "raw")

    assert out == tmp_path / "raw"
    assert _keys_of(out) == {export.raw_adapter_key(k) for k in SERVING_KEYS}
    assert not export.is_serving_format(_keys_of(out))
    # The config rides along — the merge planner reads it.
    assert json.loads((out / export.ADAPTER_CONFIG).read_text())["r"] == 32
    assert _keys_of(src) == set(SERVING_KEYS), "the source adapter must not be mutated"


def test_normalize_leaves_an_already_raw_adapter_alone(tmp_path):
    raw = {export.raw_adapter_key(k) for k in SERVING_KEYS}
    src = _adapter(tmp_path, raw)
    out = export.normalize_adapter(src, tmp_path / "raw")

    assert out == src, "an already-raw adapter should be used in place, not copied"
    assert not (tmp_path / "raw").exists()


def test_normalize_refuses_a_misscaled_adapter(tmp_path):
    src = _adapter(tmp_path, SERVING_KEYS, r=32, alpha=64)
    with pytest.raises(ValueError, match="wrong scale"):
        export.normalize_adapter(src, tmp_path / "raw")


def test_normalize_names_the_missing_file(tmp_path):
    empty = tmp_path / "adapter"
    empty.mkdir()
    with pytest.raises(ValueError, match=export.ADAPTER_WEIGHTS):
        export.normalize_adapter(empty, tmp_path / "raw")


def _fake_merge(**kw):
    """Stand-in for `build_hf_model`: writes the one file the caller checks."""
    out = Path(kw["output_path"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text('{"model_type": "fake"}', encoding="utf-8")


def test_merge_inverts_keys_then_renames_into_place(tmp_path):
    src = _adapter(tmp_path, SERVING_KEYS)
    seen = {}

    def spy(**kw):
        seen.update(kw)
        seen["adapter_keys"] = _keys_of(Path(kw["adapter_path"]))
        _fake_merge(**kw)

    out = export.merge_adapter(src, tmp_path / "merged", base_model="fake/model", merge_fn=spy)

    assert out == tmp_path / "merged" and (out / "config.json").is_file()
    # The merge saw RAW keys — the whole point of the module.
    assert not export.is_serving_format(seen["adapter_keys"])
    assert not out.with_name(out.name + ".partial").exists(), "partial dir left behind"
    assert not (tmp_path / "merged-adapter-raw").exists(), "scratch adapter not cleaned"


def test_merge_is_idempotent(tmp_path):
    src = _adapter(tmp_path, SERVING_KEYS)
    calls = []

    def counting(**kw):
        calls.append(kw)
        _fake_merge(**kw)

    export.merge_adapter(src, tmp_path / "m", base_model="fake/model", merge_fn=counting)
    export.merge_adapter(src, tmp_path / "m", base_model="fake/model", merge_fn=counting)
    assert len(calls) == 1, "an existing merged model must not be rebuilt"


def test_merge_refuses_a_merge_that_produced_nothing(tmp_path):
    src = _adapter(tmp_path, SERVING_KEYS)

    def silent_failure(**kw):
        Path(kw["output_path"]).mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="no config.json"):
        export.merge_adapter(src, tmp_path / "m", base_model="fake/model", merge_fn=silent_failure)
    # Nothing renamed into place, so a later run retries rather than serving a shell.
    assert not (tmp_path / "m").exists()
