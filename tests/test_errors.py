"""The one error vocabulary, and the exit codes it buys.

Until 2026-09-03 `bin/ai_style.py` carried six hand-written `except ValueError:`
blocks that all mapped to `click.ClickException` (exit 1), while `synth` — the
only module with a real error hierarchy — mapped its config failures to exit 2.
So `ai-style train` answered "no training pairs left after filtering" with 1 and
`ai-style synth` answered the same class of mistake with 2, and which you got
depended on which `except` clause the command's author had written.

`stylebot.errors` is the vocabulary and `cli_options.ErrorTranslatingGroup` is
the single translation. The contract is enumerated from the live click group
rather than a hand-kept list: a subcommand added tomorrow is covered the day it
is added, and if someone mounts one under a plain `click.Group` these tests fail.
"""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from stylebot.bin import ai_style
from stylebot.cli_options import ErrorTranslatingGroup, translate_errors
from stylebot.errors import ConfigError, MissingExtra, RunError, StylebotError

# ---------------------------------------------------------------------------
# The vocabulary itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,builtin",
    [(ConfigError, ValueError), (RunError, RuntimeError), (MissingExtra, ImportError)],
)
def test_each_class_is_also_the_builtin_it_replaces(cls, builtin):
    """The sweep could not break an existing `except ValueError:` anywhere.

    `stylebot.serve` turns a bad editor selection into an NDJSON error field
    with `except ValueError`, and livingthing's `dan-style` commands catch the
    builtins too. Subclassing is what let ~90 raise sites move in one pass.
    """
    exc = cls("boom")
    assert isinstance(exc, builtin)
    assert isinstance(exc, StylebotError)


def test_missing_extra_is_a_run_error_not_a_config_error():
    """An uninstalled extra is an unready world (exit 1), not a bad flag."""
    assert issubclass(MissingExtra, RunError)
    assert not issubclass(MissingExtra, ConfigError)


def test_synth_errors_are_the_same_vocabulary():
    """`Synth*Error` is a named specialization, no longer a fourth scheme."""
    from stylebot import synth_cli

    assert issubclass(synth_cli.SynthConfigError, ConfigError)
    assert issubclass(synth_cli.SynthRunError, RunError)
    assert issubclass(synth_cli.SynthGenerationError, RunError)


def test_session_error_is_the_same_vocabulary():
    """W11.b's `SessionError` folded in rather than staying a third scheme —
    and keeps the exit code it chose (1, an unready world)."""
    from stylebot.session import SessionError

    assert issubclass(SessionError, RunError)
    assert not issubclass(SessionError, ConfigError)


# ---------------------------------------------------------------------------
# The translation.
# ---------------------------------------------------------------------------


def test_translate_errors_maps_the_two_classes():
    with pytest.raises(click.UsageError) as usage, translate_errors():
        raise ConfigError("fix your flags")
    assert usage.value.exit_code == 2

    with pytest.raises(click.ClickException) as run, translate_errors():
        raise RunError("the world is not ready")
    assert run.value.exit_code == 1


def test_a_param_hint_names_the_flag():
    with pytest.raises(click.BadParameter) as exc, translate_errors():
        raise ConfigError("unknown strategy", param_hint="--slop-strategy")
    assert exc.value.param_hint == "--slop-strategy"


def test_a_quiet_run_error_exits_1_without_re_rendering():
    """`SynthGenerationError` has already printed every error and the summary."""
    with pytest.raises(SystemExit) as exc, translate_errors():
        raise RunError("28 generation error(s)", quiet=True)
    assert exc.value.code == 1


def test_a_bare_builtin_is_not_translated():
    """A violated internal contract is a bug, and gets a traceback, not an exit
    code. That line is why the sweep left `cli_options`' bundle-wiring errors
    and `session`'s `unknown disposition` as plain `ValueError`s."""
    with pytest.raises(ValueError), translate_errors():
        raise ValueError("select_targets: role= requires splits=")


# ---------------------------------------------------------------------------
# The contract, over the live command tree.
# ---------------------------------------------------------------------------


def _leaf_commands(
    group: click.Group, prefix: tuple[str, ...] = ()
) -> list[tuple[tuple, click.Command]]:
    """Every non-group command reachable under `group`, with its argv path.

    Enumerated, never listed: this is what makes "a new subcommand cannot
    silently miss the contract" a test rather than a hope.
    """
    out: list[tuple[tuple, click.Command]] = []
    ctx = click.Context(group)
    for name in group.list_commands(ctx):
        cmd = group.get_command(ctx, name)
        if cmd is None:
            continue
        if isinstance(cmd, click.Group):
            out.extend(_leaf_commands(cmd, (*prefix, name)))
        else:
            out.append(((*prefix, name), cmd))
    return out


LEAVES = _leaf_commands(ai_style.main)


def test_the_tree_is_not_empty_and_covers_both_groups():
    """Guards the enumeration itself: an empty walk would pass vacuously."""
    paths = {p for p, _ in LEAVES}
    assert len(paths) >= 15
    assert ("synth",) in paths and ("train",) in paths  # top level
    assert any(p[0] == "log" for p in paths)  # the `log` subgroup
    assert any(p[0] == "data" for p in paths)  # the hidden migrations subgroup


@pytest.mark.parametrize(
    "path,cmd", LEAVES, ids=lambda v: " ".join(v) if isinstance(v, tuple) else ""
)
def test_every_subcommand_exits_2_on_a_config_error(path, cmd, monkeypatch):
    """The Done criterion for W11.c.

    The command's own parameters are stripped and its body replaced, so this
    measures the error boundary and nothing else — not whether the command
    happens to have a reachable ConfigError today.
    """
    monkeypatch.setattr(cmd, "params", [])
    monkeypatch.setattr(cmd, "callback", lambda: (_ for _ in ()).throw(ConfigError("bad flags")))
    result = CliRunner().invoke(ai_style.main, list(path))
    assert result.exit_code == 2, f"{' '.join(path)}: {result.exit_code}\n{result.output}"
    assert "bad flags" in result.output


@pytest.mark.parametrize(
    "path,cmd", LEAVES, ids=lambda v: " ".join(v) if isinstance(v, tuple) else ""
)
def test_every_subcommand_exits_1_on_a_run_error(path, cmd, monkeypatch):
    """The other half: an unready world stays distinguishable from a bad flag."""
    monkeypatch.setattr(cmd, "params", [])
    monkeypatch.setattr(cmd, "callback", lambda: (_ for _ in ()).throw(RunError("no key")))
    result = CliRunner().invoke(ai_style.main, list(path))
    assert result.exit_code == 1, f"{' '.join(path)}: {result.exit_code}\n{result.output}"
    assert "no key" in result.output


def test_both_groups_translate_so_a_direct_invoke_is_covered_too():
    """`tests/test_ai_style_log.py` invokes `log.main` directly, not through
    `ai-style`, so the `log` group has to carry the translation itself."""
    from stylebot.bin import ai_style_log

    assert isinstance(ai_style.main, ErrorTranslatingGroup)
    assert isinstance(ai_style_log.main, ErrorTranslatingGroup)


# ---------------------------------------------------------------------------
# The symptom the item named, end to end.
# ---------------------------------------------------------------------------


def _corpus(tmp_path):
    """A one-record corpus on `post/x.qmd`, plus a splits file pinning that post
    to the frozen eval stratum — so `train`'s role filter empties it."""
    import json

    from stylebot import config
    from stylebot import splits as sp
    from stylebot.ai_core import STYLE_SYSTEM

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    record = {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": "It is worth noting that terse."},
            {"role": "assistant", "content": "terse."},
        ],
        "meta": {
            "source": "post/x.qmd",
            "captured_at": "2026-09-03T00:00:00+00:00",
            "capture_id": "c1",
            "chunk_index": 0,
            "chunk_total": 1,
        },
    }
    config.pairs_path(data_dir).write_text(json.dumps(record) + "\n", encoding="utf-8")
    splits_path = data_dir / "splits.json"
    sp.save_splits(
        sp.make_splits(
            ["post/x.qmd", "post/y.qmd", "post/z.qmd"],
            eval_candidates=["post/x.qmd"],
            eval_frac=0.34,
        ),
        splits_path,
    )
    return data_dir, splits_path


def test_no_pairs_left_after_filtering_now_exits_2(tmp_path):
    """The exact symptom W11.c named.

    `train`'s emptiness and `synth`'s emptiness are the same class of mistake —
    the caller's own filters — and used to answer 1 and 2 respectively.
    """
    data_dir, splits_path = _corpus(tmp_path)
    result = CliRunner().invoke(
        ai_style.main,
        [
            "train",
            "--data-dir",
            str(data_dir),
            "--dry-run",
            "--splits",
            str(splits_path),
            "--work-dir",
            str(tmp_path / "w"),
            "--manifest-out",
            str(tmp_path / "m.json"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert "no training pairs left after filtering" in result.output


def test_a_malformed_corpus_still_exits_1(tmp_path):
    """The neighbouring failure in the same command stays an unready world: the
    sweep re-classified, it did not flatten everything to one code."""
    import json

    from stylebot import config

    data_dir, splits_path = _corpus(tmp_path)
    config.pairs_path(data_dir).write_text(
        json.dumps({"messages": [], "meta": {}}) + "\n", encoding="utf-8"
    )
    result = CliRunner().invoke(
        ai_style.main,
        [
            "train",
            "--data-dir",
            str(data_dir),
            "--dry-run",
            "--work-dir",
            str(tmp_path / "w"),
            "--manifest-out",
            str(tmp_path / "m.json"),
        ],
    )
    assert result.exit_code == 1, result.output
    assert "fails the pairs contract" in result.output
