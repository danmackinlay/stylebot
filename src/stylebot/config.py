"""Central config + secrets access: the one place that resolves the corpus
location (D45) and reads keys from the process environment — no dotenv
loader (D24). Nothing here prints or logs a key value."""

from __future__ import annotations

import os
from pathlib import Path

from stylebot import corpus
from stylebot.errors import ConfigError, RunError

#: The corpus directory's name under a project root, when nothing else says.
DEFAULT_DATA_DIR_NAME = "_training_pairs"


def load_callable(spec: str, *, param_hint: str = "--div-policy"):
    """`module:attr` -> the attribute, or ConfigError naming `param_hint`."""
    import importlib

    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr:
        raise ConfigError(f"{param_hint} wants module:attr, got {spec!r}", param_hint=param_hint)
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"{param_hint} {spec!r}: {exc}", param_hint=param_hint) from exc


def resolve_div_policy(flag: str | None = None):
    """The capture differ's `div_is_prose` (D100) from a `module:attr` flag; None
    means every div is protected. A policy-bearing shell pins it instead (D101)."""
    return load_callable(flag) if flag else None


def resolve_data_dir(flag: str | os.PathLike | None = None) -> Path:
    """Resolve the corpus directory: ``flag`` > ``$STYLEBOT_DATA_DIR`` > the
    cwd-relative default ``_training_pairs`` (D45)."""
    if flag is not None:
        return Path(flag)
    return Path(os.environ.get("STYLEBOT_DATA_DIR", DEFAULT_DATA_DIR_NAME))


DEFAULT_PAIRS_NAME = "pairs.jsonl"


def pairs_path(data_dir: str | os.PathLike | None = None) -> Path:
    """Where the corpus lives inside a corpus directory: `<data-dir>/pairs/`
    when that directory exists, else `<data-dir>/pairs.jsonl` (D45, D50).
    Returned whether or not it exists."""
    root = resolve_data_dir(data_dir)
    sharded = root / corpus.PAIRS_DIR_NAME
    if sharded.is_dir():
        return sharded
    return root / DEFAULT_PAIRS_NAME


DEFAULT_SPLITS_NAME = "splits.json"


def splits_path(data_dir: str | os.PathLike | None = None) -> Path:
    """`<data-dir>/splits.json`, the one spelling; returned whether or not it exists."""
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
