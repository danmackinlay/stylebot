"""A corpus is one JSONL file, or a directory of JSONL shards.

Both shapes are first-class and neither is going away:

- **One file** (`<data-dir>/pairs.jsonl`) is what `ai-style` does for anyone who
  is not Dan. It is the simplest thing that works, it is what every example in
  the README shows, and a corpus of a few hundred captures never needs more.
- **A directory** (`<data-dir>/pairs/`) is what a corpus grows into once it is
  committed to git and rewritten by machines. Measured on the blog on
  2026-09-03: 122 versions of a 38 MB `pairs.jsonl` accounted for 1.54 GB of a
  1.8 GB `.git`, because every append rewrote the file whole and git stored each
  version at roughly its compressed size. Sharding does not make the corpus
  smaller; it makes *the file a write touches* smaller, which is the quantity
  git multiplies by the number of commits.

**The routing rule (`shard_of`) is by producer, then by month.** The producer
split is the load-bearing one — the two writers have different rewrite
disciplines. `ai-style log save` read-modify-writes the corpus on every save (to
apply its replace rule), so keeping captures in their own shard means a save
rewrites ~1 MB instead of ~38 MB, and it is the write that happens daily. Synth
only ever appends.

Within synth the split is by calendar month rather than by generator, which the
plan offered as the alternative. Three reasons:

1. **A month shard is sealed when the month ends.** History stops accruing
   versions of data nobody will append to again, so the growing blob is bounded
   by one month of synthesis rather than by the whole corpus. A generator shard
   grows forever: the largest one is already 1911 records, and every future run
   would rewrite a bigger blob than the last.
2. **Generator identity is a facet, not a location.** It is already recorded in
   `meta.gen.model`, where it can be selected on. Encoding it in a filename
   makes a second, weaker copy of that fact: retiring an arm strands a file, a
   model rename forks one group across two shards, and the roster turns over
   several times a year while the calendar does not.
3. **Generator ids are not filenames.** `openrouter/qwen/qwen3.6-35b-a3b` needs
   a lossy sanitiser and a collision rule; `2026-08` needs neither.

The cost, recorded honestly: two synth processes running concurrently either
side of a month boundary append to different shards. `PairsWriter` closes that
by locking one file per corpus and catching up on *every* shard before it
decides, so the W8 no-duplicate-cell guarantee holds across shards too.

Routing is by record content, not by which shard a record was read from, so a
rewrite is idempotent and self-healing: a record filed in the wrong shard moves
to the right one on the next rewrite, and the migration from a single file is
the same function every rewrite already calls.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from stylebot.jsonl import dumps_line, iter_jsonl, write_jsonl_atomic

#: The sharded corpus directory's name inside a data dir. Chosen so that
#: `Path(corpus).parent` is the data dir for BOTH shapes — `<dd>/pairs.jsonl`
#: and `<dd>/pairs/` — which is what lets `captures/`, `splits.json` and
#: `scores.jsonl` keep being addressed as `pairs_path.parent / ...` with no
#: change at any of the eight call sites that do it.
PAIRS_DIR_NAME = "pairs"

#: Real captures — human editing work, irreplaceable, rewritten on every `save`.
CAPTURES_SHARD = "captures.jsonl"

#: Where a line that does not parse as a JSON object is filed. Nothing produces
#: these; the corpus reader preserves them (`keep_undecodable`) rather than
#: silently dropping a corrupt line, so a rewrite needs somewhere to put one.
UNPARSED_SHARD = "unparsed.jsonl"

#: One lock for the whole corpus, whatever its shape. `*.lock` is already
#: gitignored blog-side; it is transient and per-machine.
LOCK_NAME = ".pairs.lock"

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
    if "_raw" in record and set(record.keys()) == {"_raw"}:
        return UNPARSED_SHARD
    meta = record.get("meta")
    if not isinstance(meta, dict):
        return UNPARSED_SHARD
    if not meta.get("synthetic"):
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


def iter_records(corpus: str | Path, *, keep_undecodable: bool = False) -> Iterator[dict]:
    """Yield every record in a corpus, shard by shard in `shard_paths` order."""
    for shard in shard_paths(corpus):
        yield from iter_jsonl(shard, keep_undecodable=keep_undecodable)


def read_records(corpus: str | Path, **kw) -> list[dict]:
    """`iter_records` collected into a list."""
    return list(iter_records(corpus, **kw))


def rewrite(corpus: str | Path, records: Iterable[dict]) -> list[Path]:
    """Rewrite a corpus so it holds exactly `records`. Returns the shards touched.

    The one write path for every read-filter-write caller — `log save`'s replace
    rule, `log tidy`, `remove_pairs_for_source`, `recut`, `dedupe_pairs_file`,
    the migration prunes. On a single-file corpus it is `write_jsonl_atomic` and
    nothing else, so nothing about the unsharded install changes.

    On a sharded corpus every record is routed by `shard_of` and **a shard whose
    bytes would not change is not written at all**. That is not an optimisation,
    it is the property the whole exercise is for: a synth run that appends 200
    September pairs leaves the twelve other shards byte-identical, so
    `git diff --stat` names one file and git stores one small delta.

    A shard that exists and ends up empty is truncated rather than deleted, so a
    prune that empties a month shows up as a diff instead of as a vanished file.
    """
    path = Path(corpus)
    records = list(records)
    if not path.is_dir():
        write_jsonl_atomic(path, records)
        return [path]

    buckets: dict[str, list[dict]] = {}
    for rec in records:
        buckets.setdefault(shard_of(rec), []).append(rec)
    # Shards that exist but got no records must be emptied, not left stale.
    for shard in shard_paths(path):
        buckets.setdefault(shard.name, [])

    touched: list[Path] = []
    for name, recs in sorted(buckets.items()):
        shard = path / name
        wanted = "".join(_line(rec) for rec in recs)
        if shard.exists() and shard.read_text(encoding="utf-8") == wanted:
            continue
        write_jsonl_atomic(shard, recs)
        touched.append(shard)
    return touched


def _line(record: dict) -> str:
    if "_raw" in record and set(record.keys()) == {"_raw"}:
        return record["_raw"] + "\n"
    return dumps_line(record) + "\n"


def shard_corpus(corpus_file: str | Path, out_dir: str | Path) -> dict[str, int]:
    """Migrate a single-file corpus into a shard directory. Returns per-shard counts.

    Reads with `keep_undecodable` so a corrupt line is carried across rather than
    silently dropped, and leaves the source file alone: the caller decides
    whether to remove it, and the round-trip is checkable while both exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = list(iter_jsonl(corpus_file, keep_undecodable=True))
    rewrite(out_dir, records)
    counts: dict[str, int] = {}
    for rec in records:
        name = shard_of(rec)
        counts[name] = counts.get(name, 0) + 1
    return counts


def content_sha256(corpus: str | Path) -> str | None:
    """A 16-hex content hash of a corpus, for pinning in a run manifest.

    For a single file this is exactly `provenance.file_sha256`, so hashes
    recorded before sharding stay comparable with hashes recorded after it for
    the same unsharded corpus. For a directory it hashes each shard's name and
    digest in `shard_paths` order, which makes it insensitive to how the same
    records are distributed only in the sense that it is *not*: two different
    shardings of the same records hash differently, and that is correct — the
    manifest's job is to identify the bytes that were trained on.
    `record_ids_sha256` is the hash that answers "the same records?".
    """
    from stylebot.provenance import file_sha256

    path = Path(corpus)
    if not path.is_dir():
        return file_sha256(path)
    digest = hashlib.sha256()
    for shard in shard_paths(path):
        digest.update(shard.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update((file_sha256(shard) or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


@contextmanager
def corpus_lock(corpus: str | Path):
    """Exclusive advisory lock over a whole corpus, whatever its shape.

    One lock per corpus rather than one per shard, so two writers targeting
    *different* shards still serialise. Without that, the W8 duplicate-cell
    defect would come back in exactly the window sharding introduces: two synth
    processes either side of a month boundary, each appending cells the other
    had already planned. A no-op on platforms without `fcntl`.
    """
    path = Path(corpus)
    data_dir = path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / LOCK_NAME
    try:
        import fcntl
    except ImportError:  # pragma: no cover — non-POSIX
        yield
        return
    with lock_path.open("a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
