"""Provenance stamps for the committed records: timestamp, git SHA, content hash.

Every artifact that gets committed — a pair record, a run manifest, a classifier
`meta.json` — answers "when, from which tree, over what bytes" with these three.

Stdlib-only and import-cheap on purpose: like `splits`, this must import with
no extras installed (Phase 3 and the eval harness both depend on it).
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    """UTC now as ISO-8601 with microseconds.

    Microsecond precision so records stamped close together (e.g. two sessions
    opened by a script) get distinct values — `ai-style log`'s `save` matches a
    snapshot on ``snapped_at``, and at second resolution two adjacent `open`
    calls are indistinguishable.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def git_sha(repo: str | Path | None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo or "."), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def file_sha256(path: str | Path) -> str:
    """16-hex content hash of a file (the `source_sha256` convention of
    `eval.write_covariate_table`) — the corpus pin in the manifest."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
