"""Dataclass-driven CLI option bundles — generic machinery, no app specifics.

A *bundle* is a frozen dataclass whose fields are a command's options. `opt()`
marks a field as also being a CLI flag, carrying the click declaration on the
field's metadata, so each default is written exactly once — the dataclass is
the single source of truth. A field declared WITHOUT `opt()` is library-only:
it gets no flag and is set by callers in code.

    @dataclass(frozen=True)
    class GenOpts:
        cli_section = "Generation"                  # --help heading (optional)
        max_tokens: int = opt(800, "--max-tokens", type=int, help="...")
        budgets: Mapping[str, int] | None = None    # library-only, no flag

`bundle_options(bundles, defaults=…, exclude=…)` builds the click decorator,
`bundles_from_kwargs()` folds a callback's `**kw` back into instances, and
`SectionedCommand` groups `--help` by bundle. A downstream repo extends a
surface with `bundle_options([*SOME_BUNDLES, MyOpts], defaults=[GenOpts(...)])`.

`Emitter` / `click_emitter()` are the other half of the same seam, for output
rather than input: a library takes an emitter and defaults it to silence, the
CLI hands it `click_emitter()`. One adapter, so two front-ends (here and
livingthing's `dan-style`) cannot drift in how they render the same run.

`translate_errors()` / `ErrorTranslatingGroup` are the third: the libraries
raise `stylebot.errors`' two classes and never exit, and this is where those
become click exceptions and exit codes.
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


# ---------------------------------------------------------------------------
# The output sink.
#
# A library entry point must not decide where its lines go — a non-CLI caller
# cannot capture or silence a `click.echo` — so it takes an *emitter*
# defaulting to `silent`, and the CLI passes `click_emitter()`. A level names a
# (stream, emphasis) pair, not a syslog severity: `note` is yellow on *stdout*
# because a hygiene-drop count is part of the run's report. `status` and `error`
# render identically today and stay apart because they differ in meaning.
# ---------------------------------------------------------------------------


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
    """A bundle field that is also a CLI flag.

    `default` is the field default AND the flag default (never pass `default=`
    in `click_kw` — that is the duplication this exists to remove); `decls` and
    `click_kw` are handed to `click.option` verbatim.
    """
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
    """field name -> owning bundle.

    Raises if two bundles claim one name, or if a name collides with `reserved`
    (the consuming function's own parameter names) — either way the field could
    not be routed unambiguously.
    """
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
    """Click decorator applying every flag-bearing field of `bundles`.

    `defaults` re-pins flag defaults, given as bundle *instances* — a typo'd
    field name fails in the dataclass constructor. `exclude` drops flags the
    wrapper pins as policy constants (its callback then never sees them, and it
    sets those fields on the bundle itself). `--help` order is bundle order,
    then field order within each bundle.
    """
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
    """A command whose `--help` groups options under their bundle's heading.

    Options carry `help_section` (set by `bundle_options`); anything without one
    — the command's own flags, `--help` — lands in a leading "Options" block.
    """

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


# ---------------------------------------------------------------------------
# The error boundary (D47). Every library module raises `stylebot.errors`'
# vocabulary and never exits; this is where the process consequences are
# decided. On the *group* rather than each callback, so a subcommand added
# tomorrow inherits the contract — `tests/test_errors.py` walks the tree.
# ---------------------------------------------------------------------------


@contextmanager
def translate_errors():
    """Turn `stylebot.errors` into click exceptions: config 2, run 1.

    A `ConfigError` carrying `param_hint` becomes `click.BadParameter` so the
    offending flag is named; without one it is a plain `UsageError`. A
    `RunError` becomes `click.ClickException` — except a `quiet` one, which has
    already printed everything it has to say and only needs the exit code.

    Anything else propagates untouched: a bare `ValueError` from a violated
    internal contract is a bug, and click's traceback is the right report.
    """
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
    """A `click.Group` that applies `translate_errors` to its whole subtree.

    `Group.invoke` is the funnel every descendant command's body passes
    through, so one `try` here covers subgroups (`ai-style log`, `ai-style
    data`) and any command added later, at any depth.
    """

    def invoke(self, ctx: click.Context):
        with translate_errors():
            return super().invoke(ctx)
