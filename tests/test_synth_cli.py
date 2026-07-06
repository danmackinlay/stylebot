"""Tests for the shared synth CLI kit (stylebot.bin.synth_cli).

The kit exists so `ai-style synth` and downstream blog wrappers share one
option surface and one command body instead of copy-pasting them. These pin
the override/exclude machinery and the drift-prone body behaviors (dry-run
stubs, the non-zero exit on generation errors).
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from stylebot import synth
from stylebot.bin import synth_cli
from stylebot.bin.ai_style import main as ai_style_main

HUMAN_POST = """---
title: A Human Post
automation: 0
---

This is a sufficiently long human-authored paragraph about graph theory and the
quiet dignity of well-chosen abstractions, long enough to clear the min-chars bar.

Another paragraph, also comfortably past the minimum length, musing on the way
synthetic data both helps and quietly lies to the model that consumes it.
"""


def _make_blog(tmp_path):
    root = tmp_path / "blog"
    (root / "post").mkdir(parents=True)
    (root / "post" / "human.qmd").write_text(HUMAN_POST, encoding="utf-8")
    return root


# -- synth_options machinery --


def test_synth_options_overrides_defaults():
    @click.command()
    @synth_cli.synth_options(merge=True, min_chars=123)
    def cmd(**kw):
        click.echo(f"{kw['merge']} {kw['min_chars']} {kw['drop_list_items']}")

    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0, result.output
    # Overridden defaults apply; untouched options keep the stylebot default.
    assert result.output.strip() == "True 123 False"


def test_synth_options_exclude_removes_option():
    @click.command()
    @synth_cli.synth_options(exclude=("gpt_model", "local_model"))
    def cmd(**kw):
        click.echo(",".join(sorted(kw)))

    runner = CliRunner()
    help_out = runner.invoke(cmd, ["--help"]).output
    assert "--gpt-model" not in help_out
    assert "--merge" in help_out  # the rest of the surface survives
    assert runner.invoke(cmd, ["--gpt-model", "x"]).exit_code != 0
    # And the callback never receives the excluded params.
    ok = runner.invoke(cmd, [])
    assert ok.exit_code == 0
    assert "gpt_model" not in ok.output


def test_synth_options_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown synth option"):
        synth_cli.synth_options(no_such_knob=1)
    with pytest.raises(ValueError, match="unknown synth option"):
        synth_cli.synth_options(exclude=("also_not_a_knob",))


def test_pop_chunk_kwargs_shapes_iter_targets_args():
    kw = {"min_chars": 10, "max_chars": 0, "sort_name": "length", "dry_run": True}
    chunk = synth_cli.pop_chunk_kwargs(kw)
    assert chunk["max_chars"] is None  # 0 -> uncapped
    assert callable(chunk["sort_key"]) and "sort_name" not in chunk
    assert kw == {"dry_run": True}  # generation params stay for run_synth


# -- end-to-end through the rebuilt ai-style synth --


def test_ai_style_synth_dry_run(tmp_path):
    root = _make_blog(tmp_path)
    result = CliRunner().invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "corpus"),
         "--openrouter-model", "test/model", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "2 target chunk(s) from 1 source(s)" in result.output
    assert "would write 2 new pair(s)" in result.output
    assert "openrouter/test/model: 2" in result.output


def test_ai_style_synth_sample_needs_no_data_dir(tmp_path):
    root = _make_blog(tmp_path)
    result = CliRunner().invoke(
        ai_style_main, ["synth", "--blog-root", str(root), "--sample", "1"]
    )
    assert result.exit_code == 0, result.output


def test_inspection_mode_banner_and_generation_flag_pointer(tmp_path):
    # Generation flags alongside --sample/--report are silently inert; the
    # inspection banner must say so and point at the eval pair browser.
    root = _make_blog(tmp_path)
    runner = CliRunner()
    with_flags = runner.invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--sample", "1", "--openrouter-model", "x/y"],
    )
    assert with_flags.exit_code == 0, with_flags.output
    assert "nothing generated" in with_flags.output
    assert "ai-style eval --pairs" in with_flags.output

    without_flags = runner.invoke(
        ai_style_main, ["synth", "--blog-root", str(root), "--sample", "1"]
    )
    assert "nothing generated" in without_flags.output  # banner always
    assert "ai-style eval --pairs" not in without_flags.output  # pointer only on the trap


def test_ai_style_synth_requires_a_generator(tmp_path):
    root = _make_blog(tmp_path)
    result = CliRunner().invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "corpus"), "--dry-run"],
    )
    assert result.exit_code != 0
    assert "no generators selected" in result.output


# -- run_synth body --


def test_run_synth_exits_nonzero_on_generation_errors(tmp_path, capsys):
    targets = synth.iter_targets(blog_root=_make_blog(tmp_path))

    def _boom(_text: str) -> str:
        raise RuntimeError("api down")

    bad = synth.Generator(name="boom", generate=_boom)
    with pytest.raises(SystemExit) as excinfo:
        synth_cli.run_synth(targets, data_dir=tmp_path / "corpus", generators=[bad])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "generation error(s)" in err  # exit summary
    assert "!!" in err and "api down" in err  # surfaced immediately, not just at exit


def test_timeout_reaches_generator_factories(tmp_path, monkeypatch):
    # --timeout must be plumbed into the HTTP client construction; without it
    # the openai SDK waits 600s x retries and a bad upstream stalls in silence.
    root = _make_blog(tmp_path)
    captured = {}

    def fake_factory(*, model, timeout=None, provider_sort=None, **kw):
        captured["timeout"] = timeout
        captured["provider_sort"] = provider_sort
        return synth.Generator(name=f"openrouter/{model}", generate=lambda t: f"slop {t}")

    monkeypatch.setattr(synth, "openrouter_generator", fake_factory)
    result = CliRunner().invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "corpus"),
         "--openrouter-model", "x/y", "--timeout", "7"],
    )
    assert result.exit_code == 0, result.output
    assert captured["timeout"] == 7.0
    assert captured["provider_sort"] == "throughput"  # the default routing preference
    # The heartbeat brackets the run: first and last pair always echo.
    assert "1/2 pairs" in result.output and "2/2 pairs" in result.output

    # --provider-sort none restores OpenRouter's own load-balancing (field omitted).
    result = CliRunner().invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "corpus2"),
         "--openrouter-model", "x/y", "--provider-sort", "none"],
    )
    assert result.exit_code == 0, result.output
    assert captured["provider_sort"] is None


def test_run_synth_defers_data_dir_for_inspection(tmp_path):
    # A callable data_dir must not be invoked in --sample mode.
    targets = synth.iter_targets(blog_root=_make_blog(tmp_path))

    def _explode() -> None:
        raise AssertionError("data_dir resolved during inspection")

    result = synth_cli.run_synth(targets, data_dir=_explode, sample_n=1)
    assert result is None


# -- parallelism + sessions through the shared surface --


def test_max_workers_auto_rule(tmp_path, monkeypatch):
    root = _make_blog(tmp_path)
    captured = {}

    def spy(targets, data_dir, generators, **kw):
        captured.update(kw)
        return synth.SynthResult(planned=0)

    monkeypatch.setattr(synth, "synthesize_pairs", spy)
    monkeypatch.setattr(
        synth, "openrouter_generator",
        lambda *, model, **kw: synth.Generator(name=f"openrouter/{model}", generate=lambda t, history=None: "s"),
    )
    monkeypatch.setattr(
        synth, "local_generator",
        lambda **kw: synth.Generator(name="local-x", generate=lambda t, history=None: "s"),
    )
    runner = CliRunner()
    base = ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "c")]

    # OpenRouter-only -> auto 16.
    assert runner.invoke(ai_style_main, [*base, "--openrouter-model", "x/y"]).exit_code == 0
    assert captured["max_workers"] == 16
    # A local preset in the mix -> auto 1 (local endpoints want sequential).
    assert runner.invoke(ai_style_main, [*base, "--generator", "local"]).exit_code == 0
    assert captured["max_workers"] == 1
    # Explicit always wins.
    assert runner.invoke(ai_style_main, [*base, "--openrouter-model", "x/y", "--max-workers", "3"]).exit_code == 0
    assert captured["max_workers"] == 3


def test_run_synth_sessions_end_to_end(tmp_path):
    import json

    targets = synth.iter_targets(blog_root=_make_blog(tmp_path))
    hist_lens = []

    def gen(text, history=None):
        hist_lens.append(len(history or []))
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 200 * (len(hist_lens))})

    result = synth_cli.run_synth(
        targets,
        data_dir=tmp_path / "corpus",
        generators=[synth.Generator("fake", generate=gen)],
        session_turns=2,
        context_window=8000,  # injected generator: window comes from the flag
    )
    assert result is not None and result.written == 2
    assert hist_lens == [0, 2]
    recs = [json.loads(ln) for ln in (tmp_path / "corpus" / "pairs.jsonl").read_text().splitlines()]
    gens_meta = [r["meta"]["gen"] for r in recs]
    assert [g["session_turn"] for g in gens_meta] == [1, 2]
    assert all(g["context_window"] == 8000 for g in gens_meta)
    assert gens_meta[1]["window_fill"] == round(400 / 8000, 4)
