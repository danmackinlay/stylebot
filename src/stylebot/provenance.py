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
    """UTC now as ISO-8601 with microseconds.

    Microsecond precision so records stamped close together (e.g. two sessions
    opened by a script) get distinct values — `ai-style log`'s `save` matches a
    snapshot on ``snapped_at``, and at second resolution two adjacent `open`
    calls are indistinguishable.
    """
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
    """Short SHA of the stylebot checkout this package is imported from.

    None for a wheel install: the SHA is only meaningful for a source checkout,
    and `git_repo_root` walking out of `site-packages` into some unrelated
    enclosing repo would be worse than no answer. The stamp says "which tooling
    tree wrote this record"; the version string is not enough, since the corpus
    changes shape between releases.
    """
    root = git_repo_root(Path(__file__).parent)
    return git_sha(root) if root is not None else None


def _canonical(value: object) -> object:
    """A JSON-encodable, order-stable view of a policy value.

    Dataclasses become field dicts (so `ChunkPolicy` participates whole, and any
    field a later composition adds is picked up without touching this module),
    mappings and sequences recurse, and a **callable becomes its
    `module.qualname`** — which is the point of the whole exercise: the one
    policy field that has silently differed between training and inference is
    `div_is_prose`, a predicate, and `repr()` of a function includes its memory
    address, so it could never have been hashed stably.
    """
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
    """A short, stable digest of a chunking policy — None for no policy.

    **Hash whatever the producer actually holds.** Today that is a
    `targets.ChunkPolicy` for synthetic pairs and the cut knobs (`gap_paras`,
    `max_chars`, …) for captures; the two are different shapes and the digests
    are therefore only comparable *within* a producer, which is the question
    anyone asks of them ("were these pairs cut the same way?"). Taking the
    object rather than a fixed field list means the digest keeps meaning what it
    says once the blog composes its policy in one place.

    Twelve hex chars: this discriminates a handful of hand-chosen diets, and is
    printed in reports next to `_recut_digest`'s eight.
    """
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
    """The three tree-level provenance fields. Two `git` calls — cache per run.

    `blog_root` may be any path inside the prose repo (its root, or the data
    directory that lives in it); the work-tree root is resolved from it. None
    means "not a checkout", and every value degrades to None rather than raising
    — a pair record is worth writing even from a tarball.
    """
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
