"""Central config + secrets access for stylebot.

The one place that resolves the corpus location and reads secrets, so every
phase shares one contract instead of reinventing `os.environ` reads:

    from stylebot import config

    key = config.require_key("TOGETHER_API_KEY")   # raises if unset
    corpus = config.resolve_data_dir()             # $STYLEBOT_DATA_DIR or default

There is no dotenv loader (D24): everything reads the *process environment*, so
getting keys in there is the caller's job — Dan's way is direnv,
``direnv exec ~/Source/livingthing uv run ai-style <args>``. Nothing here ever
prints or logs a key value.
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

    Convenience commands take the env/default; expensive, stateful ones make
    ``--data-dir`` required, so a run cannot silently target the wrong corpus
    (D45).
    """
    if flag is not None:
        return Path(flag)
    return Path(os.environ.get("STYLEBOT_DATA_DIR", DEFAULT_DATA_DIR_NAME))


DEFAULT_PAIRS_NAME = "pairs.jsonl"


def pairs_path(data_dir: str | os.PathLike | None = None) -> Path:
    """Where the corpus itself lives inside a corpus directory.

    The `splits_path` treatment for `<data-dir>/pairs.jsonl`, and for the same
    reason (D45): a join spelled out at eight sites across two repos let
    `$STYLEBOT_DATA_DIR` redirect some of them and not others. Returned whether
    or not it exists.

    **A corpus is one file or a directory of shards** (D50), and this is the one
    place that decides which: `<data-dir>/pairs/` when that directory is there,
    `<data-dir>/pairs.jsonl` otherwise. Sharding is opt-in by existing, so an
    install that never migrates keeps the single file. The directory is named so
    that `pairs_path().parent` is the data dir under BOTH shapes.
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


DRAWS_DIR_NAME = "draws"
VIEWS_DIR_NAME = "views"


def draws_path(data_dir: str | os.PathLike | None = None) -> Path:
    """`<data-dir>/draws`, the append-only layer (D84), under `resolve_data_dir`'s
    precedence; returned whether or not it exists, like `pairs_path`."""
    return resolve_data_dir(data_dir) / DRAWS_DIR_NAME


def views_root(data_dir: str | os.PathLike | None = None) -> Path:
    """`<data-dir>/views`, where `draws.ensure_view` builds; never committed (D84)."""
    return resolve_data_dir(data_dir) / VIEWS_DIR_NAME


def view_name(policy_sha: str) -> str:
    """The one spelling of a view directory's name: `pairs-<policy_sha>`."""
    return f"pairs-{policy_sha}"


def view_path(data_dir: str | os.PathLike | None, policy_sha: str) -> Path:
    """`<data-dir>/views/pairs-<policy_sha>`: the view of `draws_path(data_dir)`
    under the policy `policy_sha` names (`draws.policy_sha`)."""
    return views_root(data_dir) / view_name(policy_sha)


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
