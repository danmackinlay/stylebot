"""Dataclass-driven CLI option bundles — generic machinery, no app specifics (D78).

A bundle is a frozen dataclass whose fields are a command's options; `opt()`
marks a field as also a CLI flag, so each default is written once, and a
field without it is library-only:

    @dataclass(frozen=True)
    class GenOpts:
        cli_section = "Generation"                  # --help heading (optional)
        max_tokens: int = opt(800, "--max-tokens", type=int, help="...")
        budgets: Mapping[str, int] | None = None    # library-only, no flag

`bundle_options(bundles, defaults=…, exclude=…)` builds the decorator,
`bundles_from_kwargs()` folds a callback's `**kw` back, `SectionedCommand`
groups `--help` by bundle. `Emitter` / `click_emitter()` are the output half
of the seam; `translate_errors()` / `ErrorTranslatingGroup` are where
`stylebot.errors` become exit codes (D47).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import Field, field, fields
from typing import Any, Protocol

import click

from stylebot.errors import ConfigError, RunError

# Metadata key under which a field carries its click declaration.
CLI = "cli"


# --- the output sink: a library takes an emitter defaulting to `silent`; a
# level names a (stream, emphasis) pair, not a syslog severity.


class Emitter(Protocol):
    """A sink for a library's human-facing output lines."""

    def __call__(self, message: str, /, *, level: str = "info") -> None: ...


#: level -> (write to stderr?, click colour)
EMIT_LEVELS: dict[str, tuple[bool, str | None]] = {
    "info": (False, None),  # the run's report — what you would pipe to a file
    "note": (False, "yellow"),  # a report line that must be noticed
    "status": (True, None),  # progress, banners — narration, not the report
    "error": (True, None),  # the detail of a failure
    "warn": (True, "yellow"),  # an invariant broke
}


def silent(message: str, /, *, level: str = "info") -> None:
    """The default emitter: called as a library, a library says nothing."""


def click_emitter() -> Emitter:
    """An `Emitter` rendering to click's stdout/stderr, per `EMIT_LEVELS`."""

    def emit(message: str, /, *, level: str = "info") -> None:
        try:
            err, fg = EMIT_LEVELS[level]
        except KeyError:
            raise ValueError(
                f"unknown emit level {level!r}; expected one of {sorted(EMIT_LEVELS)}"
            ) from None
        if fg is None:
            # Not `secho(fg=None)`: `click.style` appends a reset sequence even
            # with no styles set, which a tty would show and a pipe would not.
            click.echo(message, err=err)
        else:
            click.secho(message, err=err, fg=fg)

    return emit


def opt(default: Any, /, *decls: str, **click_kw: Any) -> Any:
    """A bundle field that is also a CLI flag: `default` is both defaults
    (never pass `default=` in `click_kw`); the rest goes to `click.option`."""
    if "default" in click_kw:
        raise ValueError(
            f"opt({decls}) got default= in its click kwargs; the first positional "
            "argument is the one default"
        )
    return field(default=default, metadata={CLI: (decls, click_kw)})


def cli_fields(bundle: type) -> list[tuple[Field, tuple[str, ...], dict]]:
    """The flag-bearing fields of `bundle`, in declaration order."""
    return [
        (f, f.metadata[CLI][0], f.metadata[CLI][1]) for f in fields(bundle) if CLI in f.metadata
    ]


def field_owners(bundles: Sequence[type], *, reserved: Iterable[str] = ()) -> dict[str, type]:
    """field name -> owning bundle; raises on a name two bundles claim or one
    in `reserved` (the consuming function's own parameters)."""
    owners: dict[str, type] = {}
    for bundle in bundles:
        for f in fields(bundle):
            if f.name in owners:
                raise ValueError(
                    f"option {f.name!r} is claimed by two bundles "
                    f"({owners[f.name].__name__}, {bundle.__name__}) — routing "
                    "would be ambiguous"
                )
            owners[f.name] = bundle
    clash = sorted(set(reserved) & set(owners))
    if clash:
        raise ValueError(
            f"option(s) {clash} collide with reserved parameter name(s) — rename one side"
        )
    return owners


def section_name(bundle: type) -> str:
    """The `--help` heading for a bundle: its `cli_section`, else its name."""
    return getattr(bundle, "cli_section", None) or bundle.__name__


def _pinned_defaults(instances: Iterable[Any], bundles: Sequence[type]) -> dict[str, Any]:
    """Field values that differ from their declared default, from bundle
    instances passed as `defaults=` — i.e. the caller's re-pinned defaults."""
    out: dict[str, Any] = {}
    for inst in instances:
        cls = type(inst)
        if cls not in bundles:
            raise ValueError(f"{cls.__name__} is not one of this command's bundles")
        for f in fields(cls):
            value = getattr(inst, f.name)
            if value == f.default:
                continue
            if CLI not in f.metadata:
                raise ValueError(
                    f"{cls.__name__}.{f.name} has no CLI flag, so it has no default "
                    "to re-pin; set it on the bundle you pass at call time instead"
                )
            out[f.name] = value
    return out


def last_param(f: Any) -> click.Parameter:
    """The parameter `click.option` just attached (command or bare function): where a
    wrapper that declares a flag of its own sets that flag's `help_section`."""
    return f.params[-1] if isinstance(f, click.Command) else f.__click_params__[-1]


def bundle_options(
    bundles: Sequence[type],
    *,
    defaults: Iterable[Any] = (),
    exclude: Iterable[str] = (),
):
    """Click decorator applying every flag-bearing field of `bundles`;
    `defaults` re-pins flag defaults as bundle instances, `exclude` drops flags
    the wrapper pins itself. `--help` order is bundle order, then field order."""
    bundles = tuple(bundles)
    field_owners(bundles)  # refuse an ambiguous surface before declaring flags
    exclude = set(exclude)
    flaggable = {f.name for b in bundles for f, _, _ in cli_fields(b)}
    unknown = exclude - flaggable
    if unknown:
        raise ValueError(
            f"exclude names no CLI field of these bundles: {sorted(unknown)} "
            f"(known: {sorted(flaggable)})"
        )
    pinned = _pinned_defaults(defaults, bundles)

    def deco(f):
        # Applied in reverse: click reverses __click_params__ when it builds the
        # command, so this yields declaration order in --help.
        for bundle in reversed(bundles):
            section = section_name(bundle)
            for fld, decls, click_kw in reversed(cli_fields(bundle)):
                if fld.name in exclude:
                    continue
                kw = dict(click_kw, default=fld.default)
                if fld.name in pinned:
                    kw["default"] = pinned[fld.name]
                    kw["show_default"] = True
                f = click.option(*decls, **kw)(f)
                last_param(f).help_section = section
        return f

    return deco


def bundles_from_kwargs(kw: dict, bundles: Mapping[str, type]) -> dict[str, Any]:
    """Pop each bundle's fields out of a click callback's `**kw` and build one
    instance per bundle, keyed by the caller's parameter name.

    Fields not present in `kw` (excluded flags, library-only fields) keep their
    declared default; `kw` is left holding only the command's own parameters.
    """
    return {
        name: cls(**{f.name: kw.pop(f.name) for f in fields(cls) if f.name in kw})
        for name, cls in bundles.items()
    }


class SectionedCommand(click.Command):
    """A command whose `--help` groups options under their bundle's
    `help_section`; anything without one lands in a leading "Options" block."""

    def format_options(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        sections: dict[str, list[tuple[str, str]]] = {}
        for param in self.get_params(ctx):
            record = param.get_help_record(ctx)
            if record is not None:
                title = getattr(param, "help_section", None) or "Options"
                sections.setdefault(title, []).append(record)
        for title, records in sections.items():
            with formatter.section(title):
                formatter.write_dl(records)


# --- the error boundary (D47), on the group so every subcommand inherits it.


@contextmanager
def translate_errors():
    """Turn `stylebot.errors` into click exceptions: config 2 (`BadParameter`
    with a `param_hint`, else `UsageError`), run 1 (`ClickException`, or a bare
    exit for a `quiet` one). Anything else propagates: a bug wants a traceback."""
    try:
        yield
    except ConfigError as exc:
        if exc.param_hint:
            raise click.BadParameter(str(exc), param_hint=exc.param_hint) from exc
        raise click.UsageError(str(exc)) from exc
    except RunError as exc:
        if exc.quiet:
            raise SystemExit(1) from exc
        raise click.ClickException(str(exc)) from exc


class ErrorTranslatingGroup(click.Group):
    """A `click.Group` that applies `translate_errors` to its whole subtree
    through `Group.invoke`, the funnel every descendant passes through."""

    def invoke(self, ctx: click.Context):
        with translate_errors():
            return super().invoke(ctx)
