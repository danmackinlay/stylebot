"""A corpus is one JSONL file, or a directory of JSONL shards (D50).

Both shapes are first-class. **One file** (`<data-dir>/pairs.jsonl`) is the
simplest thing that works and what every README example shows. **A directory**
(`<data-dir>/pairs/`) is what a corpus grows into once it is committed to git:
sharding does not make the corpus smaller, it makes *the file a write touches*
smaller, which is the quantity git multiplies by the number of commits.

`shard_of` routes **by producer, then by calendar month**, by record content,
so a view `draws.build_view` materialises lands in the shards an append would
have. Nothing here rewrites or locks: a corpus is only appended to and a view
is built fresh, so D60's writer lock retired with D84 (W4 step 4).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from stylebot.jsonl import dumps_line, iter_jsonl

#: The sharded corpus directory's name inside a data dir. Chosen so that
#: `Path(corpus).parent` is the data dir for BOTH shapes — `<dd>/pairs.jsonl`
#: and `<dd>/pairs/` — which is what lets `captures/`, `splits.json` and
#: `scores.jsonl` keep being addressed as `pairs_path.parent / ...` with no
#: change at any of the eight call sites that do it.
PAIRS_DIR_NAME = "pairs"

#: Real captures — human editing work, irreplaceable.
CAPTURES_SHARD = "captures.jsonl"

_MONTH_RE = re.compile(r"^(\d{4}-\d{2})")


def is_sharded(corpus: str | Path) -> bool:
    """True when `corpus` names a shard directory rather than a single file."""
    return Path(corpus).is_dir()


def shard_paths(corpus: str | Path) -> list[Path]:
    """The JSONL files a corpus path names, in a stable order.

    A file names itself (whether or not it exists — an absent corpus reads as
    empty, as it always has). A directory names every `*.jsonl` in it, sorted by
    name, which puts `captures.jsonl` first and the synth months in
    chronological order.
    """
    path = Path(corpus)
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path]


def shard_of(record: dict) -> str:
    """Which shard a record belongs in. The routing rule; see the module docstring.

    Real captures go to `captures.jsonl`; synthetic pairs to
    `synth-<YYYY-MM>.jsonl` keyed on `meta.captured_at`. A synthetic record with
    no usable timestamp goes to `synth-undated.jsonl` rather than being guessed
    into a month — a wrong month is indistinguishable from a right one later.
    """
    meta = record.get("meta")
    if not isinstance(meta, dict) or not meta.get("synthetic"):
        return CAPTURES_SHARD
    stamp = meta.get("captured_at")
    match = _MONTH_RE.match(stamp) if isinstance(stamp, str) else None
    return f"synth-{match.group(1) if match else 'undated'}.jsonl"


def shard_path_for(corpus: str | Path, record: dict) -> Path:
    """The concrete file `record` should be appended to.

    For a single-file corpus that is the file itself: the routing rule only
    applies once a corpus has been sharded, so an unsharded install keeps
    writing exactly where it always did.
    """
    path = Path(corpus)
    if not path.is_dir():
        return path
    return path / shard_of(record)


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
    """A 16-hex content hash of a corpus, for pinning in a run manifest.

    For a single file this is exactly `provenance.file_sha256`, so hashes
    recorded before sharding stay comparable. For a directory it hashes each
    shard's name and digest in `shard_paths` order, so two different shardings of
    the same records hash differently — correct, since the manifest identifies
    the bytes that were trained on. `record_ids_sha256` answers "the same
    records?".
    """
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
