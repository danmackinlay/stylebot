"""Provenance stamps for the committed records: timestamp, git SHA, content hash.

Every artifact that gets committed — a pair record, a run manifest, a classifier
`meta.json` — answers "when, from which tree, over what bytes" with these three.

Stdlib-only and import-cheap on purpose: like `splits`, this must import with
no extras installed (Phase 3 and the eval harness both depend on it).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path


def now_iso() -> str:
    """UTC now as ISO-8601 with microseconds, so adjacent stamps stay distinct."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def git_sha(repo: str | Path | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or "."), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(repo: str | Path | None) -> bool | None:
    """True if the working tree has uncommitted changes; None if not a checkout.

    One `git status --porcelain` over a 1700-post repo is not free, so callers
    stamp this ONCE per run and reuse it, never per record.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or "."), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(out.stdout.strip())


def git_repo_root(path: str | Path | None) -> Path | None:
    """The git work-tree root containing `path` (a file or a directory), or None.

    Walks the real filesystem rather than shelling out, so it is cheap enough to
    call on a default-argument path. A `.git` FILE counts (linked worktrees).
    """
    if path is None:
        return None
    p = Path(path).resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def stylebot_sha() -> str | None:
    """Short SHA of the stylebot checkout this package is imported from; None
    for a wheel install."""
    root = git_repo_root(Path(__file__).parent)
    return git_sha(root) if root is not None else None


def _canonical(value: object) -> object:
    """A JSON-encodable, order-stable view of a policy value: dataclasses
    become field dicts, containers recurse, and a callable becomes its
    `module.qualname` so a predicate like `div_is_prose` hashes stably."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _canonical(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (set, frozenset)):
        return [_canonical(v) for v in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        qual = getattr(value, "__qualname__", None) or repr(value)
        return f"{getattr(value, '__module__', '?')}.{qual}"
    return repr(value)


def policy_digest(policy: object | None) -> str | None:
    """A short, stable digest of a chunking policy — whatever the producer
    holds, so digests compare within a producer; None for no policy."""
    if policy is None:
        return None
    canon = json.dumps(
        _canonical(policy), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


#: The `meta.provenance` contract on a pair record, in stamp order. `blog_*`
#: names the repo the SOURCE PROSE came from — the blog, in the only deployment
#: there is — as distinct from `stylebot_sha`, the tooling tree that cut it.
PROVENANCE_KEYS = (
    "blog_sha",
    "blog_dirty",
    "stylebot_sha",
    "policy_digest",
    "source_automation",
)


def repo_provenance(blog_root: str | Path | None = None) -> dict:
    """The three tree-level provenance fields; two `git` calls, so cache per
    run. `blog_root` may be any path inside the prose repo; every value
    degrades to None rather than raising."""
    root = git_repo_root(blog_root)
    return {
        "blog_sha": git_sha(root) if root is not None else None,
        "blog_dirty": git_dirty(root) if root is not None else None,
        "stylebot_sha": stylebot_sha(),
    }


def pair_provenance(
    repo: Mapping | None = None,
    *,
    policy_digest: str | None = None,
    source_automation: object = None,
) -> dict:
    """Assemble one record's `meta.provenance` — always all five `PROVENANCE_KEYS`.

    A key whose answer is unknown is stamped `None` rather than omitted, so a
    consumer can tell "this record predates the contract" (no `provenance` at
    all) from "this producer could not answer" (present, null).
    """
    repo = repo or {}
    return {
        "blog_sha": repo.get("blog_sha"),
        "blog_dirty": repo.get("blog_dirty"),
        "stylebot_sha": repo.get("stylebot_sha"),
        "policy_digest": policy_digest,
        "source_automation": source_automation,
    }


def file_sha256(path: str | Path) -> str:
    """16-hex content hash of a file (the `source_sha256` convention of
    `eval.write_covariate_table`) — the corpus pin in the manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
