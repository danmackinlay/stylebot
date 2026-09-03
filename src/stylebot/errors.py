"""One error vocabulary for the whole package.

stylebot is library-first: every phase is a typed function that a second caller
(livingthing's `dan-style`, a notebook, the blog build) may drive directly, so
no library module may exit the process. What it *can* do is say which **class**
of failure it hit, and that class is what a CLI turns into an exit code.

There are exactly two classes, and they are the ones `ai-style log` has used
since Phase 1:

| class | means | exit |
| --- | --- | --- |
| `ConfigError` | the arguments are wrong or incoherent; the caller must change them | 2 |
| `RunError` | the arguments are fine, the world is not ready | 1 |
| `MissingExtra` | a `RunError` whose unready world is an uninstalled install extra | 1 |

`stylebot.cli_options.translate_errors` is the one place those become click
exceptions, and `ErrorTranslatingGroup` installs it over a whole command tree so
a new subcommand cannot miss the contract.

**What does *not* belong here.** A violated contract *between stylebot's own
functions* — `select_targets: role= requires splits=`, an option claimed by two
bundles, an unknown `disposition` that no flag can produce — stays a bare
`ValueError`/`RuntimeError`. Those are bugs in the caller's code, not in the
user's invocation, and a traceback is the right report for them. The line is:
**can the person at the terminal act on the message?** If yes it is one of the
classes below; if no it is an assertion with a nicer message.

Every class here also subclasses the builtin it replaces (`ConfigError` is a
`ValueError`, `RunError` a `RuntimeError`, `MissingExtra` also an
`ImportError`), so the sweep that introduced them could not break an existing
`except ValueError:` anywhere in either repo — including the ones inside
`stylebot.serve`, which turn a bad editor selection into an NDJSON error field
rather than an exit code.
"""

from __future__ import annotations

__all__ = ["ConfigError", "MissingExtra", "RunError", "StylebotError"]


class StylebotError(Exception):
    """Base for every failure stylebot raises on purpose.

    Catch this to mean "stylebot decided to stop", as distinct from a bug
    (which arrives as a plain builtin exception, unmapped, with a traceback).
    """


class ConfigError(StylebotError, ValueError):
    """The arguments are wrong or incoherent — the caller must change them.

    `param_hint`, when set, names the flag a CLI wrapper should blame (it
    becomes `click.BadParameter`'s hint); without it the wrapper raises a plain
    usage error. Either way: **exit 2**.

    Emptiness produced by the caller's own filters counts as a config error, not
    an unready world: "no training pairs left after filtering" is fixed by
    changing `--min-quality`/`--role`/`--pairs`, which is the same thing
    `synth`'s "no targets" has always meant.
    """

    def __init__(self, message: str, *, param_hint: str | None = None) -> None:
        super().__init__(message)
        self.param_hint = param_hint


class RunError(StylebotError, RuntimeError):
    """The configuration is fine but the world isn't. **Exit 1**.

    A missing API key, an unreachable model registry, no open capture session,
    a corpus that fails the record contract, a manifest that does not name the
    base model it was trained from.

    `quiet=True` means the failure has *already* reported itself in full — a
    synth run that printed every generation error and its own summary — so the
    CLI should exit 1 without re-rendering the message.
    """

    def __init__(self, message: str, *, quiet: bool = False) -> None:
        super().__init__(message)
        self.quiet = quiet


class MissingExtra(RunError, ImportError):
    """An optional dependency is not installed: `uv add 'stylebot[trainer]'`.

    A `RunError` (exit 1, not 2) because the invocation is fine — the install
    is what is short. Also an `ImportError` so the lazy import sites that raise
    it read as what they are, and so a caller that guards a heavyweight path
    with `except ImportError` keeps working.
    """
