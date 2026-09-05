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


# --- the served adapter's diet ---------------------------------------------


def _manifest(budgets, modes, **extra):
    return {"data": {"chunk_scale": {"budgets": budgets, "modes": modes}}, **extra}


def test_serving_diet_falls_back_when_a_manifest_predates_chunk_scale():
    assert backends.serving_diet({}) == (infer.STYLE_CHARS_PER_CHUNK, False)
    assert backends.serving_diet(None) == (infer.STYLE_CHARS_PER_CHUNK, False)
    # The caller's corpus policy is the fallback when it has one.
    assert backends.serving_diet(None, default_chunk_chars=1234) == (1234, False)


def test_serving_diet_reads_a_mixed_scale_manifest():
    m = _manifest(
        {"1500": 2400, "3000": 1200, "4500": 900, "None": 451}, {"cross-heading": 4500, "None": 451}
    )
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
        backend="tinker",
        sampler_path="tinker://mine",
        max_chunk_chars=1500,
        inline_headings=False,
        max_tokens=99,
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
    dry = _manifest(
        {"1500": 10},
        {"None": 10},
        hyperparameters={"renderer_name": "qwen3_5_disable_thinking"},
        result={},
    )
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


def test_an_openai_backend_warns_that_the_renderer_pin_is_uncheckable():
    """Serving a manifest-pinned adapter through an OpenAI-compatible endpoint
    used to look exactly like serving it correctly: both silent. The renderer
    is not *checked* there, it is *unverifiable*, and that is worth a line."""
    notes = backends.manifest_mismatches(_RUN, BackendOpts(backend="openai", model="local-q4"))
    assert len(notes) == 1
    assert "renderer unverifiable" in notes[0]
    assert "qwen3_disable_thinking" in notes[0] and "local-q4" in notes[0]

    # Not a second complaint on top of a real mismatch, and never on tinker.
    both = backends.manifest_mismatches(_RUN, BackendOpts(backend="openai", renderer_name="llama3"))
    assert [n for n in both if "renderer" in n] == [n for n in both if "renderer mismatch" in n]
    assert backends.manifest_mismatches(_RUN, BackendOpts(backend="tinker")) == []


def test_the_warning_is_skipped_when_the_manifest_pins_no_renderer():
    """No pin, nothing to be unable to check — silence, not noise."""
    unpinned = json.loads(json.dumps(_RUN))
    unpinned["result"].pop("renderer_name", None)
    unpinned["hyperparameters"].pop("renderer_name", None)
    assert backends.manifest_mismatches(unpinned, BackendOpts(backend="openai")) == []


# --- `run` as a library: RunOpts and the two passes (D78) -------------------

from stylebot.ai_core import STYLE_SYSTEM  # noqa: E402
from stylebot.backends import RunOpts  # noqa: E402
from stylebot.errors import ConfigError  # noqa: E402


def _identity(messages, num_samples=1):
    return [messages[1]["content"]] * num_samples


def _pairs(tmp_path, n=3):
    pairs = tmp_path / "pairs.jsonl"
    recs = [
        {
            "messages": [
                {"role": "system", "content": STYLE_SYSTEM},
                {"role": "user", "content": f"slop {i}"},
                {"role": "assistant", "content": f"dan {i}"},
            ],
            "meta": {"source": f"p{i}", "capture_id": f"c{i}", "chunk_index": 0},
        }
        for i in range(n)
    ]
    pairs.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return pairs


def _emitted():
    lines: list[tuple[str, str]] = []

    def emit(message, /, *, level="info"):
        lines.append((level, message))

    return lines, emit


def test_run_options_render_both_bundles_with_named_defaults():
    import click

    @click.command()
    @backends.run_options(defaults=[BackendOpts(backend="tinker")])
    def cmd(**kw):
        pass

    flags = {o for p in cmd.params for o in p.opts}
    assert {"--best-of", "--write", "--pairs", "--out", "--limit", "--manifest"} <= flags
    assert {"--backend", "--max-tokens", "--detector-model"} <= flags
    by_name = {p.name: p for p in cmd.params}
    assert by_name["best_of"].default == backends.DEFAULT_BEST_OF == 1
    assert by_name["backend"].default == "tinker"
    assert by_name["manifest_path"].help == backends.MANIFEST_HELP


def test_load_manifest_is_empty_for_none_and_a_config_error_for_bad_json(tmp_path):
    assert backends.load_manifest(None) == {}
    good = tmp_path / "m.json"
    good.write_text('{"run_id": "r1"}', encoding="utf-8")
    assert backends.load_manifest(good) == {"run_id": "r1"}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON") as info:
        backends.load_manifest(bad)
    assert info.value.param_hint == "--manifest"
    with pytest.raises(ConfigError) as info:
        backends.load_manifest(bad, param_hint="--documents-manifest")
    assert info.value.param_hint == "--documents-manifest"


def test_corpus_mode_refuses_the_three_flag_combinations(tmp_path):
    pairs = _pairs(tmp_path)
    assert backends.corpus_mode(RunOpts()) is False
    assert backends.corpus_mode(RunOpts(pairs_path=pairs, out_path=tmp_path / "o.jsonl")) is True
    with pytest.raises(ConfigError, match="not both"):
        backends.corpus_mode(RunOpts(pairs_path=pairs, out_path=tmp_path / "o"), source=pairs)
    with pytest.raises(ConfigError, match="needs --out"):
        backends.corpus_mode(RunOpts(pairs_path=pairs))
    with pytest.raises(ConfigError, match="only apply in corpus mode"):
        backends.corpus_mode(RunOpts(limit=2))
    with pytest.raises(ConfigError, match="only apply in corpus mode"):
        backends.corpus_mode(RunOpts(out_path=tmp_path / "o.jsonl"))


def test_rewrite_corpus_styles_the_selected_pairs_and_reports(tmp_path):
    out = tmp_path / "outputs.jsonl"
    lines, emit = _emitted()
    opts = RunOpts(pairs_path=_pairs(tmp_path), out_path=out, limit=2)
    n = backends.rewrite_corpus(opts, _identity, emit=emit)
    assert n == 2
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["output"] for r in rows] == ["slop 0", "slop 1"]
    assert lines[-1] == ("status", f"rewrote 2 pair(s) -> {out}")
    assert all(level == "status" for level, _ in lines)

    # The caller's stratum policy: a selector over pair records.
    out2 = tmp_path / "outputs2.jsonl"
    n = backends.rewrite_corpus(
        RunOpts(pairs_path=opts.pairs_path, out_path=out2),
        _identity,
        selector=lambda rec: rec["meta"]["source"] == "p2",
    )
    assert n == 1
    assert json.loads(out2.read_text())["meta"]["source"] == "p2"

    with pytest.raises(ConfigError, match="needs --pairs"):
        backends.rewrite_corpus(RunOpts(out_path=out), _identity)


def test_rewrite_document_rewrites_in_place_with_a_backup_and_a_summary(tmp_path):
    src = tmp_path / "d.qmd"
    text = "---\ntitle: T\n---\n\nSome prose to style.\n"
    src.write_text(text, encoding="utf-8")
    lines, emit = _emitted()

    def upper(messages, num_samples=1):
        return [messages[1]["content"].upper()] * num_samples

    result = backends.rewrite_document(
        text, RunOpts(write_in_place=True), BackendOpts(), upper, source=src, emit=emit
    )
    assert result.text == "---\ntitle: T\n---\n\nSOME PROSE TO STYLE.\n"
    assert src.read_text(encoding="utf-8") == result.text
    assert src.with_suffix(".qmd.bak").read_text(encoding="utf-8") == text
    assert lines == [
        ("status", backends.format_rewrite_summary(result)),
        ("status", f"rewrote {src} (backup at {src}.bak)"),
    ]
    assert lines[0][1] == "[1 chunk(s), 1 sample(s)]"
    # Without --write the file is untouched and only the summary is emitted.
    lines.clear()
    backends.rewrite_document(text, RunOpts(), BackendOpts(), upper, source=src, emit=emit)
    assert [level for level, _ in lines] == ["status"]
    with pytest.raises(ConfigError, match="not both"):
        backends.rewrite_document(
            text, RunOpts(pairs_path=src, out_path=tmp_path / "o"), BackendOpts(), upper
        )


def test_rewrite_document_takes_the_rewrite_hook_and_the_anchor_log(tmp_path):
    """The blog's masking defence is a `rewrite` with `rewrite_text`'s
    keywords; a custom hook's decisions are emitted like a scorer's would be."""
    seen: dict = {}
    text = "Read [the notes](./real.qmd) whenever you get the chance.\n"

    def hook(text, styler, **kw):
        seen.update(kw)
        result = infer.rewrite_text(text, styler, **kw)
        result.decisions.append("mask: nothing to mask")
        return result

    def inventor(messages, num_samples=1):
        return ["Read the notes, invented [here](https://fake.example/x)."] * num_samples

    log = tmp_path / "anchors.jsonl"
    lines, emit = _emitted()
    opts = RunOpts(anchor_log=log, strict_anchors=True)
    backend = BackendOpts(max_chunk_chars=900, inline_headings=True)
    result = backends.rewrite_document(
        text, opts, backend, inventor, rewrite=hook, div_is_prose=lambda a: True, emit=emit
    )
    assert seen["max_chunk_chars"] == 900 and seen["inline_headings"] is True
    assert seen["inline_anchors"] is True and seen["div_is_prose"] is not None
    assert result.text == text  # the only candidate lost the link: input kept
    (entry,) = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert entry["source"] is None and entry["source_anchors"] == ["./real.qmd"]
    summary = backends.format_rewrite_summary(result)
    assert "1 kept as input" in summary and "(1 lost, 1 invented)" in summary
    assert ("status", "  mask: nothing to mask") in lines
    assert ("status", f"  1 chunk(s) of anchor accounting -> {log}") in lines
    assert lines[-1] == ("status", summary)


def test_format_rewrite_summary_breaks_rejections_out_by_cause():
    r = infer.RewriteResult(
        text="",
        n_chunks=3,
        n_candidates=12,
        n_kept_input=1,
        n_anchor_rejected=2,
        n_anchor_lost=1,
        n_anchor_invented=2,
        n_truncated=1,
    )
    assert backends.format_rewrite_summary(r) == (
        "[3 chunk(s), 12 sample(s), 1 kept as input, 2 sample(s) rejected for anchor damage "
        "(1 lost, 2 invented), 1 truncated (raise --max-tokens?)]"
    )
    assert backends.format_rewrite_summary(infer.RewriteResult(text="")) == (
        "[0 chunk(s), 0 sample(s)]"
    )
