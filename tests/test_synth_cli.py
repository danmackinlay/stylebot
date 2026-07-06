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
    assert "generation error(s)" in capsys.readouterr().err


def test_run_synth_defers_data_dir_for_inspection(tmp_path):
    # A callable data_dir must not be invoked in --sample mode.
    targets = synth.iter_targets(blog_root=_make_blog(tmp_path))

    def _explode() -> None:
        raise AssertionError("data_dir resolved during inspection")

    result = synth_cli.run_synth(targets, data_dir=_explode, sample_n=1)
    assert result is None
