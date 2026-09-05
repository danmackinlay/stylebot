"""One error vocabulary for the whole package (D47).

stylebot is library-first, so no library module may exit the process. What it
*can* do is say which **class** of failure it hit, and that class is what a CLI
turns into an exit code:

| class | means | exit |
| --- | --- | --- |
| `ConfigError` | the arguments are wrong or incoherent; the caller must change them | 2 |
| `RunError` | the arguments are fine, the world is not ready | 1 |
| `MissingExtra` | a `RunError` whose unready world is an uninstalled install extra | 1 |

`stylebot.cli_options.translate_errors` is where those become click
exceptions. A violated contract between stylebot's own functions stays a
bare `ValueError`/`RuntimeError`: the line is whether the person at the
terminal can act on the message.
"""

from __future__ import annotations

__all__ = ["ConfigError", "MissingExtra", "RunError", "StylebotError"]


class StylebotError(Exception):
    """Base for every failure stylebot raises on purpose — as distinct from a
    bug, which arrives as a plain builtin exception with a traceback."""


class ConfigError(StylebotError, ValueError):
    """The arguments are wrong or incoherent — the caller must change them;
    exit 2. `param_hint` names the flag to blame."""

    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.param_hint = param_hint


class RunError(StylebotError, RuntimeError):
    """The configuration is fine but the world isn't; exit 1. `quiet=True`
    means the failure has already reported itself in full."""

    def __init__(self, message: str, *, quiet: bool = False) -> None:
        super().__init__(message)
        self.quiet = quiet


class MissingExtra(RunError, ImportError):
    """An optional dependency is not installed: a `RunError` (exit 1) and an
    `ImportError`, so an `except ImportError` guard keeps working."""
