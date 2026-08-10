"""Tests for the shared synth CLI kit (stylebot.synth_cli).

The kit exists so `ai-style synth` and downstream blog wrappers share one
option surface and one command body instead of copy-pasting them. These pin
the defaults/exclude machinery and the drift-prone body behaviors (dry-run
stubs, the non-zero exit on generation errors).

The "-- the drift guard --" block below is the structural half: `synth` is a
50-option command, and every one of its shared flags must reach a consumer.
The bundles are the single source of the surface (each field declares its own
flag via `cli_options.opt`), so a default can no longer drift between the flag
and `run_synth` — the generic machinery is covered in `test_cli_options.py`.
"""

from __future__ import annotations

import ast
import inspect as inspect_mod
from dataclasses import fields
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from stylebot import synth, synth_cli
from stylebot.bin.ai_style import main as ai_style_main
from stylebot.cli_options import cli_fields, field_owners


def test_synth_options_repins_defaults():
    @click.command()
    @synth_cli.synth_options(defaults=[synth_cli.ChunkOpts(merge=True, min_chars=123)])
    def cmd(**kw):
        click.echo(f"{kw['merge']} {kw['min_chars']} {kw['drop_list_items']}")

    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0, result.output
    # Re-pinned defaults apply; untouched options keep the stylebot default.
    assert result.output.strip() == "True 123 False"


def test_synth_options_exclude_removes_option():
    # Two names the blog wrapper really excludes (it pins both as policy) —
    # this test used to exclude --gpt-model/--local-model, which no longer
    # exist (the preset path was retired 2026-08-10).
    @click.command()
    @synth_cli.synth_options(exclude=("max_chars", "prose_only"))
    def cmd(**kw):
        click.echo(",".join(sorted(kw)))

    runner = CliRunner()
    help_out = runner.invoke(cmd, ["--help"]).output
    assert "--max-chars" not in help_out
    assert "--merge" in help_out  # the rest of the surface survives
    assert runner.invoke(cmd, ["--max-chars", "10"]).exit_code != 0
    # And the callback never receives the excluded params.
    ok = runner.invoke(cmd, [])
    assert ok.exit_code == 0
    assert "prose_only" not in ok.output


def test_synth_options_rejects_unknown_names():
    # A typo'd default is now caught by the dataclass constructor itself —
    # no bespoke registry lookup, and it names the bundle.
    with pytest.raises(TypeError, match="no_such_knob"):
        synth_cli.synth_options(defaults=[synth_cli.GenOpts(no_such_knob=1)])
    with pytest.raises(ValueError, match="exclude names no CLI field"):
        synth_cli.synth_options(exclude=("also_not_a_knob",))


def test_chunk_opts_iter_targets_kwargs():
    # The single documented selection-side path, with its two translations.
    chunk = synth_cli.ChunkOpts(min_chars=10, max_chars=0, sort_name="length")
    kwargs = chunk.iter_targets_kwargs()
    assert kwargs["min_chars"] == 10
    assert kwargs["max_chars"] is None  # 0 -> uncapped
    assert callable(kwargs["sort_key"]) and "sort_name" not in kwargs


def test_pop_bundles_folds_a_callback_kwargs():
    # The click-callback path: **kw in, bundles out, and the command's own
    # params left behind for the callback itself.
    kw = {"min_chars": 10, "sort_name": "length", "session_turns": 4, "blog_root": "x"}
    opts = synth_cli.pop_bundles(kw)
    assert set(opts) == set(synth_cli.BUNDLE_PARAMS)
    assert opts["chunk"].min_chars == 10 and opts["chunk"].sort_name == "length"
    assert opts["session"].session_turns == 4
    assert opts["gen"] == synth_cli.GenOpts()  # absent names keep their default
    assert kw == {"blog_root": "x"}


# -- the drift guard: the bundles are the single source of the surface --


def test_bundle_field_names_are_unambiguous():
    # Two bundles claiming one name, or a name colliding with run_synth's own
    # parameters, would make routing ambiguous. Import-time in the module; run
    # it here too so the failure names the offender in the test log.
    owners = field_owners(
        synth_cli.DEFAULT_BUNDLES,
        reserved=synth_cli.BUNDLE_PARAMS.keys()
        | {
            "targets",
            "data_dir",
            "count_note",
            "extra_tags",
            "generators",
        },
    )
    assert set(owners) == {f.name for b in synth_cli.DEFAULT_BUNDLES for f in fields(b)}


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
    chunk_fields = {f.name for f in fields(synth_cli.ChunkOpts)}
    for bundle in synth_cli.DEFAULT_BUNDLES:
        for f in fields(bundle):
            if f.name in chunk_fields:
                continue  # splatted wholesale into iter_targets — checked below
            assert f.name in read, f"{bundle.__name__}.{f.name} is never read"

    # The chunk bundle's consumer proof: every key it produces is a real
    # iter_targets parameter, checked against the live signature.
    accepted = set(inspect_mod.signature(synth.iter_targets).parameters)
    produced = set(synth_cli.ChunkOpts().iter_targets_kwargs())
    assert produced <= accepted, f"not iter_targets params: {sorted(produced - accepted)}"
    assert len(produced) == len(chunk_fields)  # sort_name -> sort_key, 1:1


def test_chunk_opts_agree_with_chunk_policy():
    # ChunkOpts is the CLI-side twin of targets.ChunkPolicy; where they share a
    # knob they must share its default, or --min-chars et al. mean one thing on
    # the CLI and another to a library caller building a policy directly.
    from stylebot.targets import ChunkPolicy

    policy, opts = ChunkPolicy(), synth_cli.ChunkOpts()
    shared = {f.name for f in fields(ChunkPolicy)} & {f.name for f in fields(synth_cli.ChunkOpts)}
    assert len(shared) == len(fields(ChunkPolicy)), "a ChunkPolicy knob has no flag"
    for name in shared:
        assert getattr(policy, name) == getattr(opts, name), f"{name} default differs"


def test_run_synth_takes_bundles_only():
    # No targets: run_synth echoes the count and returns before it needs a
    # data-dir or a generator, so this exercises pure bundle plumbing. The
    # loose-keyword form went with the spec table — bundles are the one shape.
    assert (
        synth_cli.run_synth(
            [],
            data_dir=None,
            gen=synth_cli.GenOpts(max_tokens=900),
            session=synth_cli.SessionOpts(session_turns=4),
            substrate=synth_cli.SubstrateOpts(limit=10),
            inspection=synth_cli.InspectOpts(report_title="t"),
            corpus=synth_cli.CorpusOpts(replicate="draw2"),
            count_note="(note)",
            extra_tags=("x",),
            generators=None,
        )
        is None
    )
    with pytest.raises(TypeError, match="unexpected keyword"):
        synth_cli.run_synth([], data_dir=None, max_tokens=900)
    # ChunkOpts is not a run_synth bundle: it is consumed by iter_targets.
    with pytest.raises(TypeError, match="unexpected keyword"):
        synth_cli.run_synth([], data_dir=None, chunk=synth_cli.ChunkOpts())


def test_synth_command_surface_is_exactly_the_bundles_plus_its_own():
    # Pins the 50 (54 until the gpt/local preset path went, 2026-08-10): the
    # shared surface is the bundles' flag-bearing fields and nothing else, and
    # `synth` adds only its own selection/provenance flags.
    params = [p for p in ai_style_main.commands["synth"].params if isinstance(p, click.Option)]
    own = {"blog_root", "data_dir", "field", "max_level", "select_all", "glob", "tags"}
    shared = {f.name for bundle in synth_cli.DEFAULT_BUNDLES for f, _, _ in cli_fields(bundle)}
    assert len(params) == 50
    assert {p.name for p in params} - own == shared
    # Library-only fields stay off the CLI — that is what omitting opt() means.
    assert {"session_budgets", "report_title"} & {p.name for p in params} == set()


def test_synth_help_is_sectioned():
    help_out = CliRunner().invoke(ai_style_main, ["synth", "--help"]).output
    for bundle in synth_cli.DEFAULT_BUNDLES:
        assert f"\n{bundle.cli_section}:\n" in help_out
    # The command's own flags keep a leading, unsectioned block.
    assert help_out.index("\nOptions:\n") < help_out.index(synth_cli.GenOpts.cli_section)


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


def test_ai_style_synth_sample_needs_no_data_dir(tmp_path, make_blog):
    root = make_blog(ai=False)
    result = CliRunner().invoke(ai_style_main, ["synth", "--blog-root", str(root), "--sample", "1"])
    assert result.exit_code == 0, result.output


def test_inspection_mode_banner_and_generation_flag_pointer(tmp_path, make_blog):
    # Generation flags alongside --sample/--report are silently inert; the
    # inspection banner must say so and point at the eval pair browser.
    root = make_blog(ai=False)
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


def test_ai_style_synth_requires_a_generator(tmp_path, make_blog):
    root = make_blog(ai=False)
    result = CliRunner().invoke(
        ai_style_main,
        ["synth", "--blog-root", str(root), "--data-dir", str(tmp_path / "corpus"), "--dry-run"],
    )
    assert result.exit_code != 0
    assert "no generators selected" in result.output


# -- run_synth body --


def test_run_synth_exits_nonzero_on_generation_errors(tmp_path, capsys, make_blog):
    targets = synth.iter_targets(blog_root=make_blog(ai=False))

    def _boom(_text: str) -> str:
        raise RuntimeError("api down")

    bad = synth.Generator(name="boom", generate=_boom)
    with pytest.raises(SystemExit) as excinfo:
        synth_cli.run_synth(targets, data_dir=tmp_path / "corpus", generators=[bad])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "generation error(s)" in err  # exit summary
    assert "!!" in err and "api down" in err  # surfaced immediately, not just at exit
    # Attribution: a failed pair is never written, so the message itself must
    # name the config that produced it.
    assert "[boom strategy=polish effort=off]" in err


def test_timeout_reaches_generator_factories(tmp_path, monkeypatch, make_blog):
    # --timeout must be plumbed into the HTTP client construction; without it
    # the openai SDK waits 600s x retries and a bad upstream stalls in silence.
    root = make_blog(ai=False)
    captured = {}

    def fake_factory(*, model, timeout=None, provider_sort=None, **kw):
        captured["timeout"] = timeout
        captured["provider_sort"] = provider_sort
        return synth.Generator(name=f"openrouter/{model}", generate=lambda t: f"slop {t}")

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
            "--timeout",
            "7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["timeout"] == 7.0
    assert captured["provider_sort"] == "throughput"  # the default routing preference
    # The heartbeat brackets the run: first and last pair always echo.
    assert "1/2 pairs" in result.output and "2/2 pairs" in result.output

    # --provider-sort none restores OpenRouter's own load-balancing (field omitted).
    result = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus2"),
            "--openrouter-model",
            "x/y",
            "--provider-sort",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["provider_sort"] is None


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
    with pytest.raises(click.UsageError, match="data-dir is required"):
        synth_cli.run_synth(
            targets,
            data_dir=None,
            generators=[synth.Generator("fake", generate=lambda t: "s")],
        )


def test_role_without_splits_is_a_usage_error(make_blog):
    # The guard must run BEFORE load_splits, or --role alone dies inside
    # os.fspath(None) with a TypeError nobody can act on.
    targets = synth.iter_targets(blog_root=make_blog(ai=False))
    with pytest.raises(click.UsageError, match="--role requires --splits"):
        synth_cli.run_synth(
            targets, data_dir=None, substrate=synth_cli.SubstrateOpts(role="styler")
        )


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
    assert result.exit_code != 0
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
    assert result.exit_code != 0
    assert "Missing required secret 'OPENROUTER_API_KEY'" in result.output
    assert not (tmp_path / "corpus" / "pairs.jsonl").exists()


def test_reasoning_effort_rotation_cross_product(tmp_path, make_blog):
    import json

    # 2 efforts x 1 strategy x 1 model: effort joins the rotation at no cost
    # multiplier, and each pair records the effort that produced it.
    root = make_blog(ai=False, many=10)
    dry = CliRunner().invoke(
        ai_style_main,
        [
            "synth",
            "--blog-root",
            str(root),
            "--data-dir",
            str(tmp_path / "corpus"),
            "--openrouter-model",
            "x/y",
            "--reasoning-effort",
            "off",
            "--reasoning-effort",
            "high",
            "--dry-run",
        ],
    )
    assert dry.exit_code == 0, dry.output
    assert "would write 10 new pair(s)" in dry.output  # rotation, not multiplication

    captured = []

    def fake_factory(*, model, strategy, reasoning_effort, **kw):
        captured.append(reasoning_effort)
        return synth.Generator(
            name=f"openrouter/{model}",
            strategy=strategy,
            reasoning_effort=reasoning_effort,
            generate=lambda t, history=None: synth.GenOutput(
                f"[slop] {t}", {"reasoning_effort": reasoning_effort}
            ),
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
                "--reasoning-effort",
                "off",
                "--reasoning-effort",
                "high",
            ],
        )
    assert live.exit_code == 0, live.output
    assert captured == ["off", "high"]  # one generator per effort
    recs = [json.loads(ln) for ln in (tmp_path / "corpus" / "pairs.jsonl").read_text().splitlines()]
    assert {r["meta"]["gen"]["reasoning_effort"] for r in recs} == {"off", "high"}
