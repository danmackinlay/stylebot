"""Central config + secrets access for stylebot.

The one place that resolves the corpus location and reads secrets, so every
phase shares the same contract instead of each reinventing `os.environ` reads.

Usage:

    from stylebot import config

    key = config.require_key("TOGETHER_API_KEY")   # raises a clear error if unset
    corpus = config.resolve_data_dir()  # Path to $STYLEBOT_DATA_DIR or default

There is no dotenv loader: everything here reads the *process environment* and
nothing else, so getting the keys in there is the caller's job. Dan's way is
direnv, which keys off the working directory:

    direnv exec ~/Source/livingthing uv run ai-style <args>

Nothing here ever prints or logs a key value.
"""

from __future__ import annotations

import os
from pathlib import Path

from stylebot import corpus
from stylebot.errors import RunError

#: The corpus directory's name under a project root, when nothing else says.
#: Named rather than spelled inline so a caller that has a *root* rather than a
#: data dir (livingthing's `training_targets.data_dir`, whose default is
#: `<blog-root>/_training_pairs`) composes the same name this module defaults to.
DEFAULT_DATA_DIR_NAME = "_training_pairs"


def resolve_data_dir(flag: str | os.PathLike | None = None) -> Path:
    """Resolve the corpus directory: ``flag`` > ``$STYLEBOT_DATA_DIR`` > the
    cwd-relative default ``_training_pairs``.

    Convenience commands (the interactive ``ai-style log``) call this with no
    argument and happily take the env/default. Expensive, stateful commands
    (``train``, ``make-splits``) should instead make ``--data-dir`` a *required*
    option and pass it here, so a run can never silently target the wrong or
    empty corpus — the explicit path is the reproducibility record.
    """
    if flag is not None:
        return Path(flag)
    return Path(os.environ.get("STYLEBOT_DATA_DIR", DEFAULT_DATA_DIR_NAME))


DEFAULT_PAIRS_NAME = "pairs.jsonl"


def pairs_path(data_dir: str | os.PathLike | None = None) -> Path:
    """Where the corpus itself lives inside a corpus directory.

    The `splits_path` treatment for `<data-dir>/pairs.jsonl`, and for the same
    reason: the join was spelled out at four sites here and four more in
    livingthing, so `$STYLEBOT_DATA_DIR` redirected the capture path (which went
    through `resolve_data_dir`) while the blog's synth and train defaults were
    still hardcoded at `<blog-root>/_training_pairs`. Returned whether or not it
    exists; callers that treat an absent file as an empty corpus test for
    themselves.

    **A corpus is one file or a directory of shards** (`stylebot.corpus`), and
    this is the one place that decides which: `<data-dir>/pairs/` when that
    directory is there, `<data-dir>/pairs.jsonl` otherwise. The sharded form is
    opt-in by existing — nothing creates it but `ai-style data shard-pairs` —
    so an install that never migrates keeps the single file forever and never
    learns that shards are a thing. The directory is named so that
    `pairs_path().parent` is the data dir under BOTH shapes, which is what the
    eight `pairs_path.parent / "captures"`-style call sites rely on.
    """
    root = resolve_data_dir(data_dir)
    sharded = root / corpus.PAIRS_DIR_NAME
    if sharded.is_dir():
        return sharded
    return root / DEFAULT_PAIRS_NAME


DEFAULT_SPLITS_NAME = "splits.json"


def splits_path(data_dir: str | os.PathLike | None = None) -> Path:
    """Where the by-post splits contract lives inside a corpus directory.

    One spelling of `<data-dir>/splits.json`, so the guards that consult it
    (synth's frozen-eval refusal, `log save`'s notice) find the same file
    `ai-style make-splits --out` is pointed at. The path is returned whether or
    not it exists; callers that treat an absent file as "no contract" test for
    themselves.
    """
    return resolve_data_dir(data_dir) / DEFAULT_SPLITS_NAME


def get_key(name: str) -> str | None:
    """Return an API key from the environment, or None if unset/blank."""
    val = os.environ.get(name)
    return val or None


def require_key(name: str) -> str:
    """Return an API key, or raise a clear, actionable error if it's missing."""
    val = get_key(name)
    if not val:
        raise RunError(
            f"Missing required secret {name!r}. "
            f"Export {name} in your shell, or run under direnv "
            f"(`direnv exec <dir-with-an-.envrc> uv run ai-style ...`). "
            f"See _plans/OVERVIEW.md for which phase needs which key."
        )
    return val
