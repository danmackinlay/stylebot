"""A corpus is one JSONL file, or a directory of JSONL shards (D50).

`shard_of` routes by producer, then by calendar month, by record content, so
a view `draws.build_view` materialises lands in the shards an append would
have. Nothing here rewrites or locks (D84).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from stylebot.jsonl import dumps_line, iter_jsonl

#: The sharded corpus directory's name inside a data dir, chosen so that
#: `Path(corpus).parent` is the data dir for both shapes.
PAIRS_DIR_NAME = "pairs"

#: Real captures — human editing work, irreplaceable.
CAPTURES_SHARD = "captures.jsonl"

_MONTH_RE = re.compile(r"^(\d{4}-\d{2})")


def is_sharded(corpus: str | Path) -> bool:
    """True when `corpus` names a shard directory rather than a single file."""
    return Path(corpus).is_dir()


def shard_paths(corpus: str | Path) -> list[Path]:
    """The JSONL files a corpus path names, in a stable order: a file names
    itself (absent reads as empty); a directory names every `*.jsonl`, sorted."""
    path = Path(corpus)
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path]


def shard_of(record: dict) -> str:
    """Which shard a record belongs in: `captures.jsonl` for real captures,
    `synth-<YYYY-MM>.jsonl` by `meta.captured_at`, `synth-undated.jsonl` when
    no usable timestamp."""
    meta = record.get("meta")
    if not isinstance(meta, dict) or not meta.get("synthetic"):
        return CAPTURES_SHARD
    stamp = meta.get("captured_at")
    match = _MONTH_RE.match(stamp) if isinstance(stamp, str) else None
    return f"synth-{match.group(1) if match else 'undated'}.jsonl"


def iter_records(corpus: str | Path) -> Iterator[dict]:
    """Yield every record in a corpus, shard by shard in `shard_paths` order."""
    for shard in shard_paths(corpus):
        yield from iter_jsonl(shard)


def read_records(corpus: str | Path) -> list[dict]:
    """`iter_records` collected into a list."""
    return list(iter_records(corpus))


def line_of(record: dict) -> str:
    """The exact bytes a view gives `record` — and so its identity on disk."""
    return dumps_line(record) + "\n"


def content_sha256(corpus: str | Path) -> str | None:
    """A 16-hex content hash of a corpus, for pinning in a run manifest:
    `provenance.file_sha256` for a file, `shards_sha256` for a directory (two
    shardings of the same records hash differently; `record_ids_sha256`
    answers "the same records?")."""
    from stylebot.provenance import file_sha256

    path = Path(corpus)
    if not path.is_dir():
        return file_sha256(path)
    return shards_sha256((shard.name, file_sha256(shard)) for shard in shard_paths(path))


def shards_sha256(named: Iterable[tuple[str, str | None]]) -> str:
    """16 hex over `(shard name, shard digest)` in order: the directory half of
    `content_sha256`, so `draws.build_view` can hash an unwritten view alike."""
    digest = hashlib.sha256()
    for name, sha in named:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((sha or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]
