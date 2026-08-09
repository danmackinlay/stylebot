"""Provenance stamps for the committed records: git SHA + content hash.

Both training paths (`train`'s run manifest, `classify_train`'s artifact
`meta.json`) grew their own copy of the git-SHA helper; this is the one copy.

Stdlib-only and import-cheap on purpose: like `splits`, this must import with
no extras installed (Phase 3 and the eval harness both depend on it).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


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
