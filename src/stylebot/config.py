"""Central config + secrets access for stylebot.

The one place that resolves the corpus location and reads secrets, so every
phase shares the same contract instead of each reinventing `os.environ` reads.

Usage:

    from stylebot import config

    key = config.require_key("TOGETHER_API_KEY")   # raises a clear error if unset
    corpus = config.resolve_data_dir()  # Path to $STYLEBOT_DATA_DIR or default

Everything here reads the *process environment* and nothing else. Getting the
keys in there is the caller's job, and the way Dan does it is direnv:

    direnv exec ~/Source/livingthing uv run ai-style <args>

python-dotenv used to sit under this module and was dropped in 2026-08. Called
bare, `load_dotenv()` anchors its search on the *source tree* — it walks up from
`src/stylebot/` — so a `.env` beside the corpus you were actually running
against was invisible, while a stale one next to the checkout won. direnv keys
off the working directory, which is what anyone typing these commands expects.

Nothing here ever prints or logs a key value.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_data_dir(flag: str | os.PathLike | None = None) -> Path:
    """Resolve the corpus directory by precedence: flag > env > default.

    The single home for the precedence ladder every CLI/library entry shares:

      1. an explicit ``--data-dir`` value (``flag``), if given — wins always;
      2. else ``$STYLEBOT_DATA_DIR`` from the environment;
      3. else the cwd-relative default ``_training_pairs``.

    Convenience commands (the interactive ``ai-style log``) call this with no
    argument and happily take the env/default. Expensive, stateful commands
    (``train``, ``split``) should instead make ``--data-dir`` a *required*
    option and pass it here, so a run can never silently target the wrong or
    empty corpus — the explicit path is the reproducibility record.
    """
    if flag is not None:
        return Path(flag)
    return Path(os.environ.get("STYLEBOT_DATA_DIR", "_training_pairs"))


def get_key(name: str) -> str | None:
    """Return an API key from the environment, or None if unset/blank."""
    val = os.environ.get(name)
    return val or None


def require_key(name: str) -> str:
    """Return an API key, or raise a clear, actionable error if it's missing."""
    val = get_key(name)
    if not val:
        raise RuntimeError(
            f"Missing required secret {name!r}. "
            f"Export {name} in your shell, or run under direnv "
            f"(`direnv exec <dir-with-an-.envrc> uv run ai-style ...`). "
            f"See _plans/OVERVIEW.md for which phase needs which key."
        )
    return val
