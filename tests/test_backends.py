"""The styler-backend surface as a LIBRARY (`stylebot.backends`).

`BackendOpts` + `build_styler` lived in `bin/ai_style.py` until 2026-09-03, so
the only way to reach them was `CliRunner`, and livingthing reimplemented them
instead of importing them. The copy passed no `max_tokens`, which truncated
every rewrite at the 2000-token floor while the promoted run served a 4500-char
diet. These pin the moved mechanism: plain exceptions (no click), the diet read
off a manifest, and the token budget that follows from it.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from stylebot import backends, infer
from stylebot.backends import BackendOpts

# --- errors are plain, and each kind means one thing ------------------------


def test_build_styler_raises_plain_exceptions_not_click():
    with pytest.raises(ValueError, match="no --backend"):
        backends.build_styler(BackendOpts())
    with pytest.raises(ValueError, match="--sampler-path"):
        backends.build_styler(BackendOpts(backend="tinker"))
    with pytest.raises(ValueError, match="--base-url"):
        backends.build_styler(BackendOpts(backend="openai"))


def test_missing_key_is_a_runtime_error_not_a_usage_one(monkeypatch):
    """An unready environment is not a flag the user got wrong: the CLI maps
    this to exit 1 and the usage errors above to exit 2."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TINKER_API_KEY"):
        backends.build_styler(
            BackendOpts(backend="tinker", sampler_path="tinker://x", base_model="b")
        )
    monkeypatch.delenv("NO_SUCH_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NO_SUCH_KEY"):
        backends.build_styler(
            BackendOpts(backend="openai", base_url="u", model="m", api_key_env="NO_SUCH_KEY")
        )


def test_openai_backend_gets_the_scaled_token_budget(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        infer, "openai_backend", lambda *a, **kw: seen.update(kw) or (lambda *_a, **_k: [])
    )
    backends.build_styler(BackendOpts(backend="openai", base_url="u", model="m",
                                      max_chunk_chars=4500))
    assert seen["max_tokens"] == infer.max_tokens_for(4500)


# --- the served adapter's diet ---------------------------------------------


def _manifest(budgets, modes, **extra):
    return {"data": {"chunk_scale": {"budgets": budgets, "modes": modes}}, **extra}


def test_serving_diet_falls_back_when_a_manifest_predates_chunk_scale():
    assert backends.serving_diet({}) == (infer.STYLE_CHARS_PER_CHUNK, False)
    assert backends.serving_diet(None) == (infer.STYLE_CHARS_PER_CHUNK, False)
    # The caller's corpus policy is the fallback when it has one.
    assert backends.serving_diet(None, default_chunk_chars=1234) == (1234, False)


def test_serving_diet_reads_a_mixed_scale_manifest():
    m = _manifest({"1500": 2400, "3000": 1200, "4500": 900, "None": 451},
                  {"cross-heading": 4500, "None": 451})
    assert backends.serving_diet(m) == (4500, True)


def test_serving_diet_ignores_a_thinly_supported_scale():
    """Over-chunking triggers the identity fallback, so a handful of long pairs
    in an otherwise short corpus must not drag serving out to their length."""
    m = _manifest({"1500": 4000, "4500": 100}, {"cross-heading": 100, "None": 4000})
    assert backends.serving_diet(m) == (1500, False)


def test_serving_diet_needs_a_cross_heading_majority_to_inline():
    minority = _manifest({"4500": 1}, {"cross-heading": 400, "None": 600})
    majority = _manifest({"4500": 1}, {"cross-heading": 600, "None": 400})
    assert backends.serving_diet(minority)[1] is False
    assert backends.serving_diet(majority)[1] is True


# --- manifest -> opts -------------------------------------------------------

_RUN = _manifest(
    {"None": 1799, "1500": 1514, "4500": 445, "3000": 632},
    {"None": 1799, "cross-heading": 2591},
    hyperparameters={"base_model": "Qwen/Qwen3-30B"},
    result={
        "checkpoints": {"sampler_path": "tinker://ckpt"},
        "renderer_name": "qwen3_disable_thinking",
    },
)


def test_opts_from_manifest_fills_the_checkpoint_and_the_diet():
    opts = backends.opts_from_manifest(_RUN, BackendOpts(backend="tinker"))
    assert opts.sampler_path == "tinker://ckpt"
    assert opts.base_model == "Qwen/Qwen3-30B"
    assert opts.renderer_name == "qwen3_disable_thinking"
    assert (opts.chunk_chars(), opts.headings()) == (4500, True)
    # The bug this whole seam exists to kill: a 4500-char diet served on the
    # 2000-token floor truncates every rewrite mid-chunk.
    assert opts.tokens() == infer.max_tokens_for(4500) > infer.DEFAULT_MAX_TOKENS


def test_explicit_flags_beat_the_manifest():
    pinned = BackendOpts(
        backend="tinker", sampler_path="tinker://mine", max_chunk_chars=1500,
        inline_headings=False, max_tokens=99,
    )
    opts = backends.opts_from_manifest(_RUN, pinned)
    assert opts.sampler_path == "tinker://mine"
    assert (opts.chunk_chars(), opts.headings(), opts.tokens()) == (1500, False, 99)
    # ... and the tri-state is what makes that possible: an explicit
    # --no-inline-headings is False, "nobody said" is None.
    assert BackendOpts().inline_headings is None


def test_renderer_falls_back_to_the_hyperparameter_when_the_run_left_none():
    """A dry-run manifest (and three of the blog's real ones) records the
    renderer only under `hyperparameters`. Filling from `result` alone left the
    field None, so serving fell through to the cookbook recommendation for the
    base model rather than the renderer the run actually pinned."""
    dry = _manifest({"1500": 10}, {"None": 10},
                    hyperparameters={"renderer_name": "qwen3_5_disable_thinking"},
                    result={})
    assert backends.manifest_renderer(dry) == "qwen3_5_disable_thinking"
    assert backends.opts_from_manifest(dry).renderer_name == "qwen3_5_disable_thinking"


# --- a manifest is a pin, and a violated pin is silent ----------------------


def test_no_mismatch_when_nothing_disagrees():
    assert backends.manifest_mismatches(_RUN, BackendOpts(backend="tinker")) == []
    assert backends.manifest_mismatches(None, BackendOpts()) == []


def test_style_system_pin_is_checked_only_when_the_manifest_carries_one():
    from stylebot.pairs import system_sha

    stale = json.loads(json.dumps(_RUN))
    stale["data"]["style_system_sha"] = "deadbeefdeadbeef"
    notes = backends.manifest_mismatches(stale, BackendOpts())
    assert len(notes) == 1 and "STYLE_SYSTEM has changed" in notes[0]

    current = json.loads(json.dumps(_RUN))
    current["data"]["style_system_sha"] = system_sha()
    assert backends.manifest_mismatches(current, BackendOpts()) == []
    # A manifest predating the pin answers "cannot say", not "mismatch".
    assert "style_system_sha" not in _RUN["data"]
    assert backends.manifest_mismatches(_RUN, BackendOpts()) == []


def test_renderer_and_base_model_overrides_are_reported():
    pinned = BackendOpts(backend="tinker", renderer_name="llama3", base_model="Qwen/Other")
    notes = backends.manifest_mismatches(_RUN, pinned)
    assert any("renderer mismatch" in n for n in notes)
    assert any("base-model mismatch" in n for n in notes)


def test_opts_from_manifest_logs_the_mismatch_it_did_not_override(caplog):
    pinned = BackendOpts(backend="tinker", renderer_name="llama3")
    with caplog.at_level("WARNING", logger="stylebot.backends"):
        opts = backends.opts_from_manifest(_RUN, pinned)
    # The explicit flag still wins — filling is silent, overriding is not.
    assert opts.renderer_name == "llama3"
    assert any("renderer mismatch" in r.getMessage() for r in caplog.records)


# --- the CLI is only translation -------------------------------------------


def test_run_manifest_threads_the_diet_and_the_token_budget(tmp_path, monkeypatch):
    from stylebot.bin import ai_style

    seen_backend: dict = {}
    seen_rewrite: dict = {}
    monkeypatch.setattr(
        infer, "openai_backend",
        lambda *a, **kw: seen_backend.update(kw) or (lambda *_a, **_k: []),
    )
    monkeypatch.setattr(
        infer, "rewrite_text",
        lambda text, styler, **kw: seen_rewrite.update(kw) or infer.RewriteResult(text=text),
    )
    manifest = tmp_path / "run.json"
    manifest.write_text(json.dumps(_RUN), encoding="utf-8")
    doc = tmp_path / "d.qmd"
    doc.write_text("Some prose.\n", encoding="utf-8")

    result = CliRunner().invoke(
        ai_style.main,
        ["run", str(doc), "--manifest", str(manifest),
         "--backend", "openai", "--base-url", "http://x/v1", "--model", "m"],
    )
    assert result.exit_code == 0, result.output
    assert seen_rewrite["max_chunk_chars"] == 4500
    assert seen_rewrite["inline_headings"] is True
    assert seen_backend["max_tokens"] == infer.max_tokens_for(4500)


# The CLI's own exit-code translation (usage -> 2, environment -> 1) is pinned
# by `test_smoke.test_run_backend_errors_keep_their_exit_codes`; what is tested
# here is that the library raises the two kinds it distinguishes them by.
