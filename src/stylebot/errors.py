"""One error vocabulary for the whole package (D47).

stylebot is library-first, so no library module may exit the process. What it
*can* do is say which **class** of failure it hit, and that class is what a CLI
turns into an exit code:

| class | means | exit |
| --- | --- | --- |
| `ConfigError` | the arguments are wrong or incoherent; the caller must change them | 2 |
| `RunError` | the arguments are fine, the world is not ready | 1 |
| `MissingExtra` | a `RunError` whose unready world is an uninstalled install extra | 1 |

`stylebot.cli_options.translate_errors` is the one place those become click
exceptions, and `ErrorTranslatingGroup` installs it over a whole command tree so
a new subcommand cannot miss the contract.

**What does not belong here:** a violated contract between stylebot's own
functions stays a bare `ValueError`/`RuntimeError`. The line is *can the person
at the terminal act on the message?* Every class here subclasses the builtin it
replaces, so introducing them could not break an existing `except ValueError:`.
"""

from __future__ import annotations

__all__ = ["ConfigError", "MissingExtra", "RunError", "StylebotError"]


class StylebotError(Exception):
    """Base for every failure stylebot raises on purpose — as distinct from a
    bug, which arrives as a plain builtin exception with a traceback."""


class ConfigError(StylebotError, ValueError):
    """The arguments are wrong or incoherent — the caller must change them.

    `param_hint`, when set, names the flag a CLI wrapper should blame (it
    becomes `click.BadParameter`'s hint). Either way: **exit 2**. Emptiness
    produced by the caller's own filters counts here, not as an unready world.
    """

    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.param_hint = param_hint


class RunError(StylebotError, RuntimeError):
    """The configuration is fine but the world isn't. **Exit 1**.

    A missing API key, an unreachable registry, no open capture session, a
    corpus that fails the record contract. `quiet=True` means the failure has
    already reported itself in full, so the CLI exits 1 without re-rendering it.
    """

    def __init__(self, message: str, *, quiet: bool = False) -> None:
        super().__init__(message)
        self.quiet = quiet


class MissingExtra(RunError, ImportError):
    """An optional dependency is not installed: `uv add 'stylebot[trainer]'`.

    A `RunError` (exit 1) because the invocation is fine — the install is what
    is short — and an `ImportError` so a caller guarding a heavyweight path with
    `except ImportError` keeps working.
    """
