"""Tests for the shared synth CLI kit (stylebot.synth_cli).

The kit exists so `ai-style synth` and downstream blog wrappers share one
option surface and one command body instead of copy-pasting them. These pin
the defaults/exclude machinery and the drift-prone body behaviors (dry-run
stubs, the domain errors the kit raises and the exit codes `ai-style` maps
them to).

The "-- the drift guard --" block below is the structural half: `synth` is a
41-option command, and every one of its shared flags must reach a consumer.
The bundles are the single source of the surface (each field declares its own
flag via `cli_options.opt`), so a default can no longer drift between the flag
and `run_synth` — the generic machinery is covered in `test_cli_options.py`.
"""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from stylebot import splits as sp
from stylebot import synth, synth_cli
from stylebot import targets as targets_mod
from stylebot.bin.ai_style import main as ai_style_main
from stylebot.cli_options import EMIT_LEVELS


def test_chunk_policy_translates_the_uncapped_cli_spelling():
    # The CLI spells "uncapped" as 0 and the library wants None. The translation
    # lives on the type, so a caller cannot forget to apply it.
    chunk = synth_cli.ChunkPolicy(min_chars=10, max_chars=0)
    assert chunk.min_chars == 10
    assert chunk.max_chars is None  # 0 -> uncapped, normalised in __post_init__


def test_iter_targets_refuses_policy_and_loose_kwargs_together():
    # Two ways to say the same thing, and silently preferring one would make the
    # other's knobs vanish without a word.
    with pytest.raises(ValueError, match="not both"):
        synth.iter_targets(files=[], policy=synth_cli.ChunkPolicy(), min_chars=10)


# -- the drift guard: the bundles are the single source of the surface --


def test_every_bundle_field_has_a_consumer():
    # Membership in a bundle is necessary but not sufficient: the field must
    # actually be READ. Every non-chunk field is read as `<bundle>.<field>` in
    # the kit; scan the AST for exactly that rather than trusting a grep.
    src = Path(synth_cli.__file__).read_text(encoding="utf-8")
    holders = {"gen", "session", "substrate", "inspection", "corpus", "opts"}
    read = {
        node.attr
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in holders
    }
    chunk_fields = {f.name for f in fields(synth_cli.ChunkPolicy)}
    for bundle in synth_cli.DEFAULT_BUNDLES:
        for f in fields(bundle):
            if f.name in chunk_fields:
                continue  # splatted wholesale into iter_targets — checked below
            assert f.name in read, f"{bundle.__name__}.{f.name} is never read"

    # The chunk bundle's consumer proof, in its own module: every ChunkPolicy
    # field is read as `policy.<field>` (or, for the two the type translates
    # itself, on `self`) somewhere in `stylebot.targets`. There is no longer a
    # kwargs dict to check against a signature — the policy IS what iter_targets
    # takes, which is the point of the collapse.
    targets_src = Path(targets_mod.__file__).read_text(encoding="utf-8")
    policy_read = {
        node.attr
        for node in ast.walk(ast.parse(targets_src))
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"policy", "self"}
    }
    for name in chunk_fields:
        assert name in policy_read, f"ChunkPolicy.{name} is never read"


# -- the rotation: Arm / iter_arms / build_generators --


def test_dry_run_stub_names_match_real_generator_names(monkeypatch):
    # The property the old duplicated loops maintained by hand: a dry-run
    # stub's synth_key covariates (name, strategy, effort, prompt id/version)
    # must equal the real factory generator's, or dry-run resume counts lie.
    # Also the standing proof that the real factory still CONSTRUCTS off a
    # dummy key (no network, no request) — the only provider path left after
    # the gpt/local presets went (2026-08-10).
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    arms = synth_cli.iter_arms(
        [synth.resolve_strategy("polish"), synth.resolve_strategy("casual")],
        ["off", "high"],
        ["a/b", "c/d"],
    )
    kw = dict(
        custom_system=None,
        max_tokens=1000,
        timeout=5.0,
        capture_reasoning=False,
        temperature=None,
        top_p=None,
        provider_sort="throughput",
        sticky_provider=True,
        prompt_cache=True,
    )
    stubs = synth_cli.build_generators(arms, dry_run=True, **kw)
    real = synth_cli.build_generators(arms, dry_run=False, **kw)
    key_covariates = [
        (g.name, g.strategy, g.reasoning_effort, g.prompt_id, g.prompt_version) for g in stubs
    ]
    assert key_covariates == [
        (g.name, g.strategy, g.reasoning_effort, g.prompt_id, g.prompt_version) for g in real
    ]
    # The name spelling is the synth_key input: unchanged by the retirement.
    assert {g.name for g in stubs} == {"openrouter/a/b", "openrouter/c/d"}
    # 2 strategies x 2 efforts x 2 models, strategies outermost.
    assert len(stubs) == 8


# -- end-to-end through the rebuilt ai-style synth --


def test_ai_style_synth_dry_run(tmp_path, make_blog):
    root = make_blog(ai=False)
    result = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus"),
            "--openrouter-model",
            "test/model",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 target chunk(s) from 1 source(s)" in result.output
    assert "would write 2 new pair(s)" in result.output
    assert "openrouter/test/model: 2" in result.output


def test_ai_style_synth_requires_a_generator(tmp_path, make_blog):
    root = make_blog(ai=False)
    result = CliRunner().invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "corpus"), "--dry-run"],
    )
    # A flag combination the user must fix is a usage error: exit 2.
    assert result.exit_code == 2, result.output
    assert "no generators selected" in result.output


# -- run_synth body --


def test_run_synth_raises_on_generation_errors(tmp_path, capsys, make_blog):
    # The library signals failure by raising, carrying the result; it must NOT
    # exit the process (it used to `raise SystemExit(1)` from `_report_result`).
    targets = synth.iter_targets(blog_root=make_blog(ai=False))

    def _boom(_text: str) -> str:
        raise RuntimeError("api down")

    bad = synth.Generator(name="boom", generate=_boom)
    with pytest.raises(synth_cli.SynthGenerationError) as excinfo:
        synth_cli.run_synth(
            targets,
            data_dir=tmp_path / "corpus",
            generators=[bad],
            emit=synth_cli.click_emitter(),
        )
    assert excinfo.value.result.errors  # the result survives the failure
    err = capsys.readouterr().err
    assert "generation error(s)" in err  # exit summary
    assert "!!" in err and "api down" in err  # surfaced immediately, not just at exit
    # Attribution: a failed pair is never written, so the message itself must
    # name the config that produced it.
    assert "[boom strategy=polish effort=off]" in err


def test_ai_style_synth_still_exits_1_on_generation_errors(tmp_path, monkeypatch, make_blog):
    # ... and the CLI boundary turns that raise back into the exit code scripts
    # branch on — 1, not click's 2, and with no traceback.
    root = make_blog(ai=False)

    def fake_factory(*, model, **kw):
        def _boom(_text: str) -> str:
            raise RuntimeError("api down")

        return synth.Generator(name=f"openrouter/{model}", generate=_boom)

    monkeypatch.setattr(synth, "openrouter_generator", fake_factory)
    result = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus"),
            "--openrouter-model",
            "x/y",
        ],
    )
    assert result.exit_code == 1, result.output
    assert "generation error(s)" in result.output
    assert "Traceback" not in result.output


def test_run_synth_needs_no_data_dir_for_inspection(tmp_path, make_blog):
    # --sample/--report write nothing, so data_dir=None is legal there: the
    # caller (synth_cmd) checks for inspection *before* resolving, which is why
    # run_synth's data_dir is a plain path rather than a deferred callable.
    targets = synth.iter_targets(blog_root=make_blog(ai=False))
    assert (
        synth_cli.run_synth(targets, data_dir=None, inspection=synth_cli.InspectOpts(sample_n=1))
        is None
    )

    # But a generating run without one is refused, not silently defaulted.
    with pytest.raises(synth_cli.SynthConfigError, match="data-dir is required"):
        synth_cli.run_synth(
            targets,
            data_dir=None,
            generators=[synth.Generator("fake", generate=lambda t: "s")],
        )


def test_role_without_splits_is_a_config_error(make_blog):
    # The guard must run BEFORE load_splits, or --role alone dies inside
    # os.fspath(None) with a TypeError nobody can act on.
    targets = synth.iter_targets(blog_root=make_blog(ai=False))
    with pytest.raises(synth_cli.SynthConfigError, match="--role requires --splits"):
        synth_cli.run_synth(
            targets, data_dir=None, substrate=synth_cli.SubstrateOpts(role="styler")
        )


# -- the output sink: run_synth prints nothing it was not asked to print --


def test_run_synth_without_a_sink_is_silent(tmp_path, capsys, make_blog):
    # The point of the emit seam: a library caller gets a clean stdout and
    # stderr. Not "quieter" — silent, on the paths that used to echo most
    # (substrate line, count, heartbeat, per-error lines, exit summary).
    targets = synth.iter_targets(blog_root=make_blog(ai=False))
    ok = synth.Generator(name="quiet", generate=lambda t, history=None: f"[slop] {t}")
    result = synth_cli.run_synth(targets, data_dir=tmp_path / "corpus", generators=[ok])
    assert result is not None and result.written == 2

    def _boom(_text: str) -> str:
        raise RuntimeError("api down")

    with pytest.raises(synth_cli.SynthGenerationError):
        synth_cli.run_synth(
            targets,
            data_dir=tmp_path / "corpus2",
            generators=[synth.Generator(name="boom", generate=_boom)],
        )
    # And the inspection path, which writes its banner to stderr.
    synth_cli.run_synth(targets, data_dir=None, inspection=synth_cli.InspectOpts(sample_n=1))

    captured = capsys.readouterr()
    assert (captured.out, captured.err) == ("", "")


def test_run_synth_emits_every_line_through_the_sink(tmp_path, make_blog):
    # The other half: nothing bypasses the sink. A recording emitter sees the
    # count line and the summary, and no text reaches the real streams.
    targets = synth.iter_targets(blog_root=make_blog(ai=False))
    seen: list[tuple[str, str]] = []

    def record(message: str, /, *, level: str = "info") -> None:
        seen.append((level, message))

    synth_cli.run_synth(
        targets,
        data_dir=tmp_path / "corpus",
        generators=[synth.Generator(name="quiet", generate=lambda t, history=None: f"[slop] {t}")],
        emit=record,
    )
    levels = {level for level, _ in seen}
    assert levels <= set(EMIT_LEVELS), f"undeclared level(s): {levels - set(EMIT_LEVELS)}"
    text = "\n".join(m for _, m in seen)
    assert "2 target chunk(s) from 1 source(s)" in text
    assert "wrote 2 pair(s)" in text
    # The heartbeat is narration, not the report: it must not land on stdout.
    assert [level for level, m in seen if m.startswith("  ... ")] == ["status", "status"]


def test_click_emitter_routes_streams_and_colour(capsys, monkeypatch):
    emit = synth_cli.click_emitter()
    for level in EMIT_LEVELS:
        emit(f"line-{level}", level=level)
    captured = capsys.readouterr()
    # stdout carries the report (plain + the noticed lines); stderr carries
    # narration, failures and the alarms.
    assert captured.out.split() == ["line-info", "line-note"]
    assert captured.err.split() == ["line-status", "line-error", "line-warn"]

    # The colouring click strips on a pipe, pinned at the call it makes.
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(click, "secho", lambda msg, **kw: calls.append((msg, kw)))
    monkeypatch.setattr(click, "echo", lambda msg, **kw: calls.append((msg, kw)))
    for level in EMIT_LEVELS:
        emit(f"line-{level}", level=level)
    assert calls == [
        ("line-info", {"err": False}),
        ("line-note", {"err": False, "fg": "yellow"}),
        ("line-status", {"err": True}),
        ("line-error", {"err": True}),
        ("line-warn", {"err": True, "fg": "yellow"}),
    ]
    with pytest.raises(ValueError, match="unknown emit level"):
        emit("nope", level="critical")


def test_hygiene_drop_lines_are_yellow_on_stdout(tmp_path, capsys):
    # Regression (a paid run once crashed here on a name not in scope), plus
    # the routing that byte-identity depends on: the drop counts are part of
    # the *report*, so they stay on stdout — only the colour marks them.
    result = synth.SynthResult(planned=3, skipped_degenerate=2, skipped_inflated=1)
    synth_cli._report_result(
        result,
        tmp_path / "pairs.jsonl",
        dry_run=False,
        session_turns=1,
        emit=synth_cli.click_emitter(),
        max_transform_sim=0.95,
        max_length_ratio=3.0,
    )
    captured = capsys.readouterr()
    assert "2 identity pair(s) DROPPED" in captured.out and "0.95" in captured.out
    assert "1 inflated pair(s) DROPPED" in captured.out and "3.0" in captured.out
    # The unattempted-turns alarm is the one warning on stderr.
    assert "never attempted" not in captured.out
    assert "WARNING: 0 planned turn(s)" not in captured.err

    synth_cli._report_result(
        synth.SynthResult(planned=5),
        tmp_path / "pairs.jsonl",
        dry_run=False,
        session_turns=1,
        emit=synth_cli.click_emitter(),
    )
    assert "5 planned turn(s) were never attempted" in capsys.readouterr().err


# -- parallelism + sessions through the shared surface --


def test_max_workers_auto_rule(tmp_path, monkeypatch, make_blog):
    root = make_blog(ai=False)
    captured = {}

    def spy(targets, data_dir, generators, **kw):
        captured.update(kw)
        return synth.SynthResult(planned=0)

    monkeypatch.setattr(synth, "synthesize_pairs", spy)
    monkeypatch.setattr(
        synth,
        "openrouter_generator",
        lambda *, model, **kw: synth.Generator(
            name=f"openrouter/{model}", generate=lambda t, history=None: "s"
        ),
    )
    runner = CliRunner()
    base = ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "c")]

    # OpenRouter-only -> auto 16.
    assert runner.invoke(ai_style_main, [*base, "--openrouter-model", "x/y"]).exit_code == 0
    assert captured["max_workers"] == 16
    # Explicit always wins.
    res = runner.invoke(ai_style_main, [*base, "--openrouter-model", "x/y", "--max-workers", "3"])
    assert res.exit_code == 0
    assert captured["max_workers"] == 3
    # Injected generators — a library caller's own endpoint, which we know
    # nothing about — still get sequential. That is all that is left of the
    # rule now that the gpt/local presets are gone (2026-08-10); the CLI can no
    # longer put a non-OpenRouter generator in the mix.
    synth_cli.run_synth(
        synth.iter_targets(blog_root=root),
        data_dir=tmp_path / "c",
        generators=[synth.Generator("injected", generate=lambda t, history=None: "s")],
    )
    assert captured["max_workers"] == 1


def test_run_synth_sessions_end_to_end(tmp_path, monkeypatch, make_blog):
    import json

    targets = synth.iter_targets(blog_root=make_blog(ai=False))
    hist_lens = []

    def gen(text, history=None):
        hist_lens.append(len(history or []))
        return synth.GenOutput("[slop] " + text, {"prompt_tokens": 200 * (len(hist_lens))})

    # Session windows come from the OpenRouter models registry and nowhere else
    # now: the --context-window manual fallback retired with the presets
    # (2026-08-10), so a generator only gets a window if it is OpenRouter-named.
    monkeypatch.setattr(synth, "openrouter_context_windows", lambda: {"x/y": 8000})
    result = synth_cli.run_synth(
        targets,
        data_dir=tmp_path / "corpus",
        generators=[synth.Generator("openrouter/x/y", generate=gen)],
        session=synth_cli.SessionOpts(session_turns=2),
    )
    assert result is not None and result.written == 2
    assert hist_lens == [0, 2]
    recs = [json.loads(ln) for ln in (tmp_path / "corpus" / "pairs.jsonl").read_text().splitlines()]
    gens_meta = [r["meta"]["gen"] for r in recs]
    assert [g["session_turn"] for g in gens_meta] == [1, 2]
    assert all(g["context_window"] == 8000 for g in gens_meta)
    assert gens_meta[1]["window_fill"] == round(400 / 8000, 4)


# -- prompt rotation (models x strategies cross product) --


def test_strategy_rotation_cross_product(tmp_path, make_blog):
    import json

    # 2 strategies x 1 model: one run spreads targets across both prompts, at
    # no cost multiplier (still one generation per target).
    root = make_blog(ai=False, many=10)
    result = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus"),
            "--openrouter-model",
            "x/y",
            "--slop-strategy",
            "polish",
            "--slop-strategy",
            "casual",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "would write 10 new pair(s)" in result.output  # cross product rotates, not multiplies

    captured = []

    def fake_factory(*, model, strategy, **kw):
        captured.append(strategy)
        return synth.Generator(
            name=f"openrouter/{model}",
            strategy=strategy,
            generate=lambda t, history=None: f"[{strategy}] {t}",
        )

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(synth, "openrouter_generator", fake_factory)
        live = CliRunner().invoke(
            ai_style_main,
            [
                "synth",
                "--blog-root",
                str(root),
                "--data-dir",
                str(tmp_path / "corpus"),
                "--openrouter-model",
                "x/y",
                "--slop-strategy",
                "polish",
                "--slop-strategy",
                "casual",
            ],
        )
    assert live.exit_code == 0, live.output
    assert captured == ["polish", "casual"]  # one generator per strategy
    recs = [json.loads(ln) for ln in (tmp_path / "corpus" / "pairs.jsonl").read_text().splitlines()]
    assert {r["meta"]["slop_strategy"] for r in recs} == {"polish", "casual"}


def test_custom_system_file_rejects_multiple_strategies(tmp_path, make_blog):
    root = make_blog(ai=False)
    prompt = tmp_path / "my-slop.txt"
    prompt.write_text("Rewrite it my way.", encoding="utf-8")
    result = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus"),
            "--openrouter-model",
            "x/y",
            "--slop-system-file",
            str(prompt),
            "--slop-strategy",
            "a",
            "--slop-strategy",
            "b",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "ONE custom prompt" in result.output


def test_plan_sessions_separates_same_model_strategies():
    # Same model under two strategies must NOT merge into one session bucket
    # (grouping is by generator identity), and their session ids differ.
    targets = [
        synth.Target(
            text=f"Long enough paragraph number {i} of prose.",
            source="p.qmd",
            chunk_index=i,
            chunk_total=4,
        )
        for i in range(4)
    ]
    g_polish = synth.Generator(name="openrouter/x", strategy="polish")
    g_cat = synth.Generator(name="openrouter/x", strategy="catalogue")
    assignments = [
        (t, g, "", synth._synth_key(g.name, t.text, strategy=g.strategy))
        for t, g in zip(targets, [g_polish, g_cat, g_polish, g_cat], strict=True)
    ]
    sessions = synth._plan_sessions(assignments, session_turns=2)
    assert len(sessions) == 2
    by_strategy = {s.generator.strategy: s for s in sessions}
    assert set(by_strategy) == {"polish", "catalogue"}
    assert by_strategy["polish"].session_id != by_strategy["catalogue"].session_id


def test_missing_openrouter_key_aborts_before_any_generation(tmp_path, monkeypatch, make_blog):
    # A keyless run must die at generator construction with the actionable
    # config error — never reach the API, never write a partial corpus.
    # Set empty rather than deleted: `get_key` treats blank as unset, and this
    # also survives a test runner launched under direnv with a real key loaded.
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    root = make_blog(ai=False)
    result = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus"),
            "--openrouter-model",
            "qwen/qwen3-8b",
        ],
    )
    # An unready world, not a bad invocation: ClickException -> exit 1.
    assert result.exit_code == 1, result.output
    assert "Missing required secret 'OPENROUTER_API_KEY'" in result.output
    assert not (tmp_path / "corpus" / "pairs.jsonl").exists()


# -- the frozen-eval guard: synth never cuts eval prose unasked ---------------


def _eval_splits(*eval_posts: str, aliases: dict | None = None) -> dict:
    return {
        "schema_version": sp.SCHEMA_VERSION,
        "eval_posts": list(eval_posts),
        **({"aliases": aliases} if aliases else {}),
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }


def _lines(sink: list) -> str:
    return "\n".join(sink)


def _collect(sink: list):
    def emit(message: str, /, *, level: str = "info") -> None:
        sink.append(message)

    return emit


def _stub_gen():
    return synth.Generator(name="stub", generate=lambda t, history=None: f"[slop] {t}")


def test_synth_refuses_eval_posts_from_the_corpus_splits(tmp_path, make_blog):
    """The W0 leak: no --role, no --splits, and eval prose is cut as a target.

    The guard reads the corpus's own splits.json, because the run that leaked
    (blog RUNBOOK §6) passes neither flag.
    """
    root = make_blog(ai=False)
    targets = synth.iter_targets(blog_root=root)
    assert targets and {t.source for t in targets} == {"post/human.qmd"}

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    sp.save_splits(_eval_splits("post/human.qmd"), data_dir / "splits.json")

    sink: list[str] = []
    result = synth_cli.run_synth(
        targets, data_dir=data_dir, generators=[_stub_gen()], emit=_collect(sink)
    )
    assert result is None  # nothing left to synthesize
    assert "eval guard" in _lines(sink) and "post/human.qmd" in _lines(sink)
    assert not (data_dir / "pairs.jsonl").exists()


def test_synth_eval_guard_follows_a_rename(tmp_path, make_blog):
    """A pinned eval post renamed on disk is still refused, via `aliases`."""
    root = make_blog(ai=False)
    targets = synth.iter_targets(blog_root=root)
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    sp.save_splits(
        _eval_splits("post/old.qmd", aliases={"post/human.qmd": "post/old.qmd"}),
        data_dir / "splits.json",
    )
    sink: list[str] = []
    assert (
        synth_cli.run_synth(
            targets, data_dir=data_dir, generators=[_stub_gen()], emit=_collect(sink)
        )
        is None
    )
    assert "eval guard" in _lines(sink)


def test_synth_role_eval_is_the_deliberate_opt_in(tmp_path, make_blog):
    root = make_blog(ai=False)
    targets = synth.iter_targets(blog_root=root)
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    splits_path = sp.save_splits(_eval_splits("post/human.qmd"), data_dir / "splits.json")

    sink: list[str] = []
    result = synth_cli.run_synth(
        targets,
        data_dir=data_dir,
        generators=[_stub_gen()],
        substrate=synth_cli.SubstrateOpts(splits_path=splits_path, role="eval"),
        emit=_collect(sink),
    )
    assert result is not None and result.written == len(targets)
    assert "eval guard" not in _lines(sink)

