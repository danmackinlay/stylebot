"""Central config + secrets access for stylebot.

The one place that loads `.env` and resolves the corpus location, so every
phase shares the same contract instead of each reinventing `os.environ` reads.

Usage:

    from stylebot import config

    config.load()                       # idempotent; loads .env if present
    key = config.require_key("TOGETHER_API_KEY")   # raises a clear error if unset
    corpus = config.data_dir()          # Path to $STYLEBOT_DATA_DIR or default

Secrets live in `.env` (gitignored) — see `.env.example` for the key names.
Nothing here ever prints or logs a key value.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_LOADED = False


def load() -> None:
    """Load `.env` into the process environment once (idempotent).

    Existing environment variables win over `.env` values (the dotenv
    default), so CI / shell exports override the file.
    """
    global _LOADED
    if _LOADED:
        return
    load_dotenv()
    _LOADED = True


def resolve_data_dir(flag: str | os.PathLike | None = None) -> Path:
    """Resolve the corpus directory by precedence: flag > env > default.

    The single home for the precedence ladder every CLI/library entry shares:

      1. an explicit ``--data-dir`` value (``flag``), if given — wins always;
      2. else ``$STYLEBOT_DATA_DIR`` from the environment / ``.env``;
      3. else the cwd-relative default ``_training_pairs``.

    Convenience commands (the interactive ``ai-style-log``) call this with no
    argument and happily take the env/default. Expensive, stateful commands
    (``train``, ``split``) should instead make ``--data-dir`` a *required*
    option and pass it here, so a run can never silently target the wrong or
    empty corpus — the explicit path is the reproducibility record.
    """
    if flag is not None:
        return Path(flag)
    load()
    return Path(os.environ.get("STYLEBOT_DATA_DIR", "_training_pairs"))


def data_dir() -> Path:
    """Back-compat shorthand for ``resolve_data_dir()`` (env > default)."""
    return resolve_data_dir()


def get_key(name: str) -> str | None:
    """Return an API key from the environment, or None if unset/blank."""
    load()
    val = os.environ.get(name)
    return val or None


def require_key(name: str) -> str:
    """Return an API key, or raise a clear, actionable error if it's missing."""
    val = get_key(name)
    if not val:
        raise RuntimeError(
            f"Missing required secret {name!r}. "
            f"Copy .env.example to .env and set {name}, "
            f"or export {name} in your shell. See _plans/OVERVIEW.md for "
            f"which phase needs which key."
        )
    return val
