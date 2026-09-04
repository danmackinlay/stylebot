"""Tests for the generic option-bundle kit (stylebot.cli_options).

Exercised standalone against a toy bundle, deliberately: this module is the
piece a downstream repo installs and extends, so nothing here may lean on the
synth surface. `tests/test_synth_cli.py` covers stylebot's own use of it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import click
import pytest
from click.testing import CliRunner

from stylebot.cli_options import (
    SectionedCommand,
    bundle_options,
    bundles_from_kwargs,
    cli_fields,
    field_owners,
    opt,
)


@dataclass(frozen=True)
class ToyGen:
    cli_section = "Toy generation"

    model: str = opt("gpt-toy", "--model", show_default=True, help="Which toy model.")
    tags: Sequence[str] = opt((), "--tag", "tags", multiple=True, help="Repeatable tag.")
    verbose: bool = opt(False, "--verbose", is_flag=True, help="Chatter.")
    budgets: Mapping[str, int] | None = None  # library-only: no flag


@dataclass(frozen=True)
class ToyOut:
    # No cli_section: the heading falls back to the class name.
    limit: int | None = opt(None, "--limit", type=int, help="Cap the work.")


TOY_BUNDLES = (ToyGen, ToyOut)


def _toy_cmd(**deco_kw):
    @click.command(cls=SectionedCommand)
    @bundle_options(TOY_BUNDLES, **deco_kw)
    def cmd(**kw):
        click.echo(repr(sorted(kw.items())))

    return cmd


def test_opt_generates_flags_from_field_defaults():
    result = CliRunner().invoke(_toy_cmd(), [])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == (
        "[('limit', None), ('model', 'gpt-toy'), ('tags', ()), ('verbose', False)]"
    )
    # ...and the flags parse, keeping their declared spellings and types.
    parsed = CliRunner().invoke(
        _toy_cmd(), ["--model", "m", "--tag", "a", "--tag", "b", "--verbose", "--limit", "3"]
    )
    assert parsed.exit_code == 0, parsed.output
    assert parsed.output.strip() == (
        "[('limit', 3), ('model', 'm'), ('tags', ('a', 'b')), ('verbose', True)]"
    )


def test_opt_refuses_a_duplicate_default():
    # The whole point: the default is the first positional, declared once.
    with pytest.raises(ValueError, match="default="):
        opt("x", "--x", default="y")


def test_library_only_field_gets_no_flag():
    help_out = CliRunner().invoke(_toy_cmd(), ["--help"]).output
    assert "--budgets" not in help_out
    assert "budgets" not in CliRunner().invoke(_toy_cmd(), []).output
    # It is still a field, just not a CLI one.
    assert "budgets" in {f.name for f in fields(ToyGen)}
    assert "budgets" not in {f.name for f, _, _ in cli_fields(ToyGen)}


def test_defaults_repin_only_what_differs():
    cmd = _toy_cmd(defaults=[ToyGen(model="pinned", tags=("t",))])
    result = CliRunner().invoke(cmd, [])
    assert result.exit_code == 0, result.output
    assert "('model', 'pinned')" in result.output
    assert "('tags', ('t',))" in result.output
    assert "('verbose', False)" in result.output  # untouched keeps its default
    assert "[default: pinned]" in CliRunner().invoke(cmd, ["--help"]).output
    # A re-pinned default is still overridable on the command line.
    assert "('model', 'cli')" in CliRunner().invoke(cmd, ["--model", "cli"]).output


def test_defaults_reject_a_typo_and_a_foreign_bundle():
    # A typo'd field is an ordinary dataclass TypeError — no bespoke registry.
    with pytest.raises(TypeError, match="nope"):
        ToyGen(nope=1)

    @dataclass(frozen=True)
    class Outsider:
        x: int = opt(1, "--x", type=int)

    with pytest.raises(ValueError, match="not one of this command's bundles"):
        _toy_cmd(defaults=[Outsider(x=2)])


def test_defaults_reject_a_library_only_field():
    with pytest.raises(ValueError, match="has no CLI flag"):
        _toy_cmd(defaults=[ToyGen(budgets={"a": 1})])


def test_exclude_drops_the_flag_and_the_callback_param():
    cmd = _toy_cmd(exclude={"limit", "verbose"})
    help_out = CliRunner().invoke(cmd, ["--help"]).output
    assert "--limit" not in help_out and "--verbose" not in help_out
    assert "--model" in help_out  # the rest of the surface survives
    assert CliRunner().invoke(cmd, ["--limit", "3"]).exit_code != 0
    ok = CliRunner().invoke(cmd, [])
    assert ok.exit_code == 0
    assert "limit" not in ok.output and "verbose" not in ok.output


def test_exclude_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="exclude names no CLI field"):
        _toy_cmd(exclude={"budgets"})  # a real field, but library-only
    with pytest.raises(ValueError, match="exclude names no CLI field"):
        _toy_cmd(exclude={"not_a_field"})


def test_field_owners_detects_collisions():
    @dataclass(frozen=True)
    class Clash:
        model: str = opt("other", "--other-model")

    with pytest.raises(ValueError, match="claimed by two bundles"):
        field_owners([ToyGen, Clash])
    with pytest.raises(ValueError, match="reserved parameter name"):
        field_owners(TOY_BUNDLES, reserved={"limit"})
    assert field_owners(TOY_BUNDLES)["limit"] is ToyOut


def test_bundles_from_kwargs_round_trips_a_callback():
    kw = {"model": "m", "tags": ("a",), "verbose": True, "limit": 5, "own_param": 1}
    built = bundles_from_kwargs(kw, {"gen": ToyGen, "out": ToyOut})
    assert built["gen"] == ToyGen(model="m", tags=("a",), verbose=True)
    assert built["out"] == ToyOut(limit=5)
    assert kw == {"own_param": 1}  # the command's own params are left behind
    # Absent names fall back to the declared default, so a wrapper that
    # excluded a flag still gets a complete bundle.
    assert bundles_from_kwargs({}, {"gen": ToyGen})["gen"] == ToyGen()
