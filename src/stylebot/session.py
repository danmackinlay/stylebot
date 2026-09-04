"""Rewrite-session state for `ai-style log`, as a library (D35).

`stylebot.capture` owns the capture *mechanism* — paragraph diffing, the pair
record, the archive, the draw. This module owns everything around it: where
session state lives (`SessionStore`); snapshot persistence — a directory of
the base text plus a `meta.json` sidecar, *archived* rather than deleted on
close, which is what makes `reopen` / `save --redo` possible; symlink
canonicalisation (`canonical_source`), so the blog's `_focus/` harness cannot
key a session under a scratch tree that gets wiped; and `save`'s disposition
rule. Everything takes explicit paths; `bin/ai_style_log.py` is the click
shell. `SessionError` (a `RunError`) means the world is not ready, a
`ValueError` that the arguments contradict (D47). Nothing here prints or exits.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from stylebot import capture, config, corpus, draws
from stylebot.errors import ConfigError, RunError

__all__ = [
    "Disposition",
    "SNAPSHOT_CONTENT",
    "SNAPSHOT_META",
    "CorpusSummary",
    "PairResult",
    "PriorPairs",
    "SaveResult",
    "SessionError",
    "SessionStore",
    "Snapshot",
    "SnapshotEntry",
    "SourceSummary",
    "StateSummary",
    "canonical_source",
    "capture_pair",
    "close_session",
    "corpus_summary",
    "drop_session",
    "humanize_age",
    "load_snapshots",
    "move_snapshot",
    "open_session",
    "partition_prior_pairs",
    "read_pairs",
    "read_snapshot",
    "reopen_session",
    "restore_snapshot",
    "save_session",
    "snapshot_dirs",
    "summarize_state",
    "write_snapshot",
]


# `save`'s three ways of disposing of prior pairs for the same source. "smart"
# is the default: replace this session's interim pairs, keep earlier sessions'.
Disposition = Literal["smart", "replace-all", "append"]


# A snapshot is a DIRECTORY mirroring the source path,
# `snapshots/post/foo.qmd/{content.md, meta.json}`: readable on disk rather
# than JSON-escaped, and `.md` so the blog's `*.qmd` tooling cannot see it.
SNAPSHOT_CONTENT = "content.md"
SNAPSHOT_META = "meta.json"


class SessionError(RunError):
    """The arguments are fine; the world is not ready — no open session, nothing
    changed, no archive to reopen. Exit 1 at the CLI, unlike a contradictory
    invocation (`ValueError` -> exit 2)."""


# ---- Where session state lives -------------------------------------------


@dataclass(frozen=True)
class SessionStore:
    """The on-disk home of session state and the corpus, under one data dir.

    Resolved per invocation, not baked into module constants: an import-time
    constant freezes the environment at first import, which in 2026-08 had
    `log` writing to a cwd-relative `_training_pairs/` instead of the corpus.
    """

    data_dir: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))

    @classmethod
    def resolve(cls, flag: str | Path | None = None) -> SessionStore:
        """Build a store from the `--flag` > `$STYLEBOT_DATA_DIR` > default ladder."""
        return cls(config.resolve_data_dir(flag))

    @property
    def snapshots_dir(self) -> Path:
        """Where open sessions' snapshots live."""
        return self.data_dir / "snapshots"

    @property
    def closed_snapshots_dir(self) -> Path:
        """Where closed sessions' snapshots are retained: a sibling directory, not
        a suffix inside `snapshots_dir`, whose rglob would list them as open."""
        return self.data_dir / "snapshots-closed"

    @property
    def pairs_path(self) -> Path:
        """The pair corpus: `pairs/` or `pairs.jsonl` (`config.pairs_path`)."""
        return config.pairs_path(self.data_dir)

    @property
    def draws_dir(self) -> Path:
        """`<data_dir>/draws`, where a save lands (D84); may not exist yet."""
        return config.draws_path(self.data_dir)

    @property
    def corpus_path(self) -> Path:
        """What `read_pairs` reads: the draws when there are any, else the
        pair corpus (the old layout, until W4 step 5)."""
        return self.draws_dir if self.draws_dir.is_dir() else self.pairs_path

    def snapshot_path(self, source: Path | str) -> Path:
        """Where the snapshot for `source` lives: the source path mirrored under
        the snapshots dir, `post/foo.qmd` -> `snapshots/post/foo.qmd/`."""
        return self.snapshots_dir / str(source)

    def closed_snapshot_path(self, source: Path | str) -> Path:
        """The retained snapshot of `source`'s last closed session: one per
        source, overwritten on each close — redo only wants the latest base."""
        return self.closed_snapshots_dir / str(source)


# ---- Snapshot persistence -------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """A session's recorded 'before' state."""

    source: str
    snapped_at: str
    content: str

    @property
    def chars(self) -> int:
        return len(self.content)


@dataclass(frozen=True)
class SnapshotEntry:
    """One snapshot directory, loaded or not.

    Listings must tolerate an unreadable snapshot (a half-written sidecar, a
    directory someone poked at) rather than aborting the whole command, so the
    failure is carried as data.
    """

    path: Path
    snapshot: Snapshot | None = None
    error: str | None = None

    @property
    def source(self) -> str:
        """The recorded source path, falling back to the snapshot directory."""
        if self.snapshot is not None and self.snapshot.source:
            return self.snapshot.source
        return str(self.path)


def write_snapshot(snap: Path, snapshot: Snapshot) -> None:
    """Write a snapshot directory (base text + sidecar)."""
    snap.mkdir(parents=True, exist_ok=True)
    (snap / SNAPSHOT_CONTENT).write_text(snapshot.content, encoding="utf-8")
    (snap / SNAPSHOT_META).write_text(
        json.dumps(
            {"source": snapshot.source, "snapped_at": snapshot.snapped_at},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def read_snapshot(snap: Path) -> Snapshot:
    """Load a snapshot directory."""
    meta = json.loads((snap / SNAPSHOT_META).read_text(encoding="utf-8"))
    return Snapshot(
        source=meta.get("source", ""),
        snapped_at=meta.get("snapped_at", ""),
        content=(snap / SNAPSHOT_CONTENT).read_text(encoding="utf-8"),
    )


def move_snapshot(src: Path, dst: Path) -> None:
    """Move a snapshot directory, replacing any existing one at `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def snapshot_dirs(root: Path) -> list[Path]:
    """Every snapshot directory under `root`, sorted; `[]` if it doesn't exist.

    A snapshot is identified by its sidecar, so this globs for `meta.json` and
    takes the parent — the open-session reminder and `list` both need it, for
    both the open and the closed tree.
    """
    return sorted(p.parent for p in root.rglob(SNAPSHOT_META)) if root.exists() else []


def load_snapshots(root: Path) -> list[SnapshotEntry]:
    """Every snapshot under `root`, loaded, carrying any read failure as data."""
    entries: list[SnapshotEntry] = []
    for snap in snapshot_dirs(root):
        try:
            entries.append(SnapshotEntry(path=snap, snapshot=read_snapshot(snap)))
        except Exception as exc:  # noqa: BLE001 — a listing must not abort on one bad dir
            entries.append(SnapshotEntry(path=snap, error=str(exc)))
    return entries


def humanize_age(iso_ts: str) -> str:
    """Render an ISO-8601 timestamp as a relative age string ("3h ago")."""
    try:
        ts = datetime.fromisoformat(iso_ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        delta = datetime.now(UTC) - ts
        seconds = max(0, int(delta.total_seconds()))
        if seconds < 60:
            return f"{seconds}s ago"
        if seconds < 3600:
            return f"{seconds // 60}m ago"
        if seconds < 86400:
            return f"{seconds // 3600}h ago"
        if seconds < 86400 * 14:
            return f"{seconds // 86400}d ago"
        return f"{seconds // (86400 * 7)}w ago"
    except Exception:
        return iso_ts


# ---- Symlink canonicalisation ---------------------------------------------


def canonical_source(value: Path | None, *, cwd: Path | None = None) -> Path | None:
    """Normalize a symlinked FILE to the path it points at.

    A session key *is* the source path as given, so a second path to the same
    file opens a second, parallel session. The blog's `_focus/` harness creates
    exactly that: it symlinks a post into a scratch project that is wiped on
    every invocation, so a session keyed there loses its file and a `save` writes
    a transient path into the committed corpus.

    Only symlinks are rewritten, so no existing session key can move. `cwd` is
    the tree the result is made relative to; a target outside it comes back
    absolute.
    """
    if value is None or not value.is_symlink():
        return value
    target = value.resolve()
    try:
        return target.relative_to(Path.cwd() if cwd is None else cwd)
    except ValueError:
        return target


# ---- The corpus file ------------------------------------------------------


def read_pairs(store: SessionStore) -> list[dict]:
    """The live pair records: the identity view of the draws (superseded
    saves excluded, D84), or the pair corpus under the old layout; [] if absent."""
    if store.draws_dir.is_dir():
        return list(draws.view_records(draws.iter_draws(store.draws_dir), draws.ViewPolicy()))
    return corpus.read_records(store.pairs_path)


@dataclass(frozen=True)
class CorpusSummary:
    """Running corpus size and heading-context coverage, for a write's receipt."""

    path: Path
    pairs: int
    with_context: int


def corpus_summary(store: SessionStore) -> CorpusSummary:
    """Count what is in the corpus right now, and where it is."""
    pairs = read_pairs(store)
    with_ctx = sum(
        1 for r in pairs if isinstance(r, dict) and (r.get("meta", {}).get("context") or "").strip()
    )
    return CorpusSummary(path=store.corpus_path, pairs=len(pairs), with_context=with_ctx)


# ---- Sessions -------------------------------------------------------------


def open_session(store: SessionStore, source: Path) -> Snapshot:
    """Record `source`'s current content as the session's 'before' state.

    Sticky: refuses to overwrite an existing session so the original (worst)
    state is preserved across editing sittings.
    """
    snap = store.snapshot_path(source)
    if snap.exists():
        raise SessionError(
            f"session already open for {source}; "
            "`drop` it, or `save` without --keep-open to close it"
        )
    snap.parent.mkdir(parents=True, exist_ok=True)
    snapshot = Snapshot(
        source=str(source),
        snapped_at=capture.now_iso(),
        content=Path(source).read_text(encoding="utf-8"),
    )
    write_snapshot(snap, snapshot)
    return snapshot


def close_session(store: SessionStore, source: Path) -> None:
    """Close a session by archiving its snapshot rather than deleting it.

    Retaining the base text is what makes `reopen` / `save --redo` possible: the
    pairs themselves cannot reconstruct it, since in chunk mode they hold only
    the changed regions. The archive keeps the original `snapped_at`, so a
    re-save after `reopen` lands in the same session and `save`'s smart
    disposition REPLACES the earlier pairs instead of stacking a near-duplicate
    onto the corpus.
    """
    move_snapshot(store.snapshot_path(source), store.closed_snapshot_path(source))


def restore_snapshot(store: SessionStore, source: Path) -> Snapshot:
    """Restore `source`'s archived session to the live snapshot path.

    Raises `FileNotFoundError` when there is no archive to restore. Preserves
    `snapped_at` verbatim — see `close_session` for why that is load-bearing.
    """
    closed = store.closed_snapshot_path(source)
    if not closed.exists():
        raise FileNotFoundError(closed)
    snapshot = read_snapshot(closed)
    snap = store.snapshot_path(source)
    snap.parent.mkdir(parents=True, exist_ok=True)
    move_snapshot(closed, snap)
    return snapshot


def reopen_session(store: SessionStore, source: Path) -> Snapshot:
    """Reopen `source`'s last closed session, restoring its 'before' state."""
    if store.snapshot_path(source).exists():
        raise SessionError(
            f"session already open for {source}; nothing to reopen "
            "(`save` against it, or `drop` it first)"
        )
    try:
        return restore_snapshot(store, source)
    except FileNotFoundError as exc:
        raise SessionError(
            f"no closed session for {source}; run `open` to start a fresh one"
        ) from exc


def drop_session(store: SessionStore, source: Path) -> None:
    """Abandon the open session for `source` without saving any pair."""
    if not store.snapshot_path(source).exists():
        raise SessionError(f"no open session for {source}")
    close_session(store, source)


# ---- save: the disposition rule -------------------------------------------


@dataclass(frozen=True)
class PriorPairs:
    """The corpus, split three ways relative to one source and one session."""

    same_session: list[dict]
    other_session: list[dict]
    other_source: list[dict]

    @property
    def total(self) -> int:
        """How many records the corpus held when the split was taken."""
        return len(self.other_source) + len(self.other_session) + len(self.same_session)

    @property
    def earlier_session_stamps(self) -> list[str]:
        """Sorted `snapped_at` values of the earlier sessions, blanks excluded."""
        stamps = {
            rec.get("meta", {}).get("snapped_at")
            for rec in self.other_session
            if isinstance(rec, dict)
        }
        return sorted(ts for ts in stamps if ts)

    @property
    def sessionless(self) -> int:
        """Prior pairs for this source logged outside a session (`pair`, `synth`):
        `snapped_at=None`, so no session timestamp to sort or age; counted apart
        (sorting the mixed set raised `TypeError`)."""
        return sum(
            1
            for rec in self.other_session
            if not (isinstance(rec, dict) and rec.get("meta", {}).get("snapped_at"))
        )


def partition_prior_pairs(
    records: Sequence[dict], *, source: str, snapped_at: str | None
) -> PriorPairs:
    """Split `records` into this session's, this source's earlier, and everything else."""
    same_session: list[dict] = []
    other_session: list[dict] = []
    other_source: list[dict] = []
    for rec in records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        if meta.get("source") != source:
            other_source.append(rec)
        elif meta.get("snapped_at") == snapped_at:
            same_session.append(rec)
        else:
            other_session.append(rec)
    return PriorPairs(
        same_session=same_session, other_session=other_session, other_source=other_source
    )


@dataclass(frozen=True)
class SaveResult:
    """What one `save` did: the records written and what it did to prior ones."""

    source: Path
    written: list[dict]
    disposition: Disposition
    prior: PriorPairs
    kept_open: bool
    whole: bool
    reopened: Snapshot | None = None

    @property
    def before_chars(self) -> int:
        return sum(rec["meta"]["before_chars"] for rec in self.written)

    @property
    def after_chars(self) -> int:
        return sum(rec["meta"]["after_chars"] for rec in self.written)


def _capture_ids(records: Sequence[dict]) -> tuple[str, ...]:
    """The distinct `capture_id`s of `records`, in first-seen order."""
    seen: dict[str, None] = {}
    for rec in records:
        cid = (rec.get("meta") or {}).get("capture_id") if isinstance(rec, dict) else None
        if cid:
            seen[cid] = None
    return tuple(seen)


def _to_supersede(
    store: SessionStore,
    prior: PriorPairs,
    disposition: Disposition,
    *,
    source: Path,
    say: Callable[[str], None],
) -> list[dict]:
    """The prior records this save supersedes under `disposition`, announced."""
    superseded = prior.same_session if disposition == "smart" else []
    if disposition == "replace-all":
        superseded = prior.same_session + prior.other_session
    if superseded and not store.draws_dir.is_dir():
        say(f"no {store.draws_dir}: prior pairs are kept, not superseded")
        return []
    if disposition == "smart" and superseded:
        say(f"superseded {len(superseded)} interim pair(s) from this session")
    elif superseded:
        earlier = len(prior.other_session)
        note = f" (including {earlier} from earlier session(s))" if earlier else ""
        say(f"superseded {len(superseded)} prior pair(s) for {source}{note}")
    return superseded


def _open_for_save(
    store: SessionStore, source: Path, *, redo: bool, say: Callable[[str], None]
) -> Snapshot | None:
    """Reopen for `redo` when nothing is open; raise when there is no session."""
    snap = store.snapshot_path(source)
    reopened: Snapshot | None = None
    if redo and not snap.exists():
        try:
            reopened = restore_snapshot(store, source)
        except FileNotFoundError as exc:
            raise SessionError(
                f"--redo: no closed session for {source} to reopen; run `open` first"
            ) from exc
        say(
            f"reopened session for {source} (originally opened {humanize_age(reopened.snapped_at)})"
        )
    if not snap.exists():
        hint = "run `open` first"
        if store.closed_snapshot_path(source).exists():
            hint += ", or `save --redo` to reopen the last closed session"
        raise SessionError(f"no open session for {source}; {hint}")
    return reopened


def save_session(
    store: SessionStore,
    source: Path,
    *,
    keep_open: bool = False,
    disposition: Disposition = "smart",
    redo: bool = False,
    tags: Sequence[str] = (),
    whole: bool = False,
    heading_context: bool = True,
    notify: Callable[[str], None] | None = None,
) -> SaveResult:
    """Diff `source` against its session snapshot and land the pair(s) as draws.

    Nothing is rewritten (D84): prior pairs are *superseded* — the new draws
    name their `capture_id`s and a view drops them. **Smart by default**:
    pairs from the *same* session (matching `meta.snapped_at`) are superseded,
    being interim `keep_open` saves that should not stack, while earlier
    sessions' pairs are kept as legitimate prior rewrites. `"replace-all"`
    supersedes cross-session pairs too; `"append"` supersedes nothing. Under
    the old layout (no `draws/`) nothing is superseded and the pairs stack.

    `redo` reopens the last closed session first, so a corrected rewrite is
    re-diffed against the SAME base and lands in the same session. `notify`
    receives the lines describing work done *before* the capture.
    """
    say = notify or (lambda _line: None)
    if disposition not in ("append", "replace-all", "smart"):  # pragma: no cover
        raise ValueError(f"unknown disposition {disposition!r}")
    reopened = _open_for_save(store, source, redo=redo, say=say)

    snapshot = read_snapshot(store.snapshot_path(source))
    before_text = snapshot.content
    after_text = Path(source).read_text(encoding="utf-8")
    if before_text == after_text:
        raise SessionError(f"no changes since session opened on {source}; nothing to save")

    prior = partition_prior_pairs(
        read_pairs(store), source=str(source), snapped_at=snapshot.snapped_at
    )
    superseded = _to_supersede(store, prior, disposition, source=source, say=say)

    try:
        written = capture.capture_pairs(
            before_text,
            after_text,
            pairs_path=store.pairs_path,
            source=str(source),
            snapped_at=snapshot.snapped_at,
            whole=whole,
            extra_meta={"tags": list(tags)} if tags else None,
            heading_context=heading_context,
            supersedes=_capture_ids(superseded),
        )
    except RunError as exc:
        raise SessionError(str(exc)) from exc

    if not keep_open:
        close_session(store, source)

    return SaveResult(
        source=source,
        written=written,
        disposition=disposition,
        prior=prior,
        kept_open=keep_open,
        whole=whole,
        reopened=reopened,
    )


# ---- pair: one-shot capture, no session -----------------------------------


@dataclass(frozen=True)
class PairResult:
    """What one `pair` did."""

    written: list[dict]
    replaced: int
    whole: bool

    @property
    def before_chars(self) -> int:
        return sum(rec["meta"]["before_chars"] for rec in self.written)

    @property
    def after_chars(self) -> int:
        return sum(rec["meta"]["after_chars"] for rec in self.written)


def capture_pair(
    store: SessionStore,
    *,
    before_text: str,
    after_text: str,
    source: str | None = None,
    replace: bool = False,
    tags: Sequence[str] = (),
    whole: bool = False,
    heading_context: bool = True,
    notify: Callable[[str], None] | None = None,
) -> PairResult:
    """One-shot pair capture from two texts, with no session dance.

    Unlike `save_session`, this APPENDS by default: the typical use is a one-off
    pair from a clipboard or a script. `replace` supersedes every prior pair
    for `source` (D84), and requires one. `notify` is `save_session`'s
    pre-capture progress channel.
    """
    say = notify or (lambda _line: None)
    if before_text == after_text:
        raise SessionError("before and after are identical; refusing to log")

    superseded: list[dict] = []
    if replace:
        if source is None:
            # A contradiction in the arguments, not an unready world: the CLI
            # turns this into a UsageError (exit 2), so keep it a ValueError.
            raise ConfigError("--replace requires --source")
        superseded = [r for r in read_pairs(store) if (r.get("meta") or {}).get("source") == source]
        if superseded:
            say(f"superseded {len(superseded)} prior pair(s) for {source}")

    try:
        written = capture.capture_pairs(
            before_text,
            after_text,
            pairs_path=store.pairs_path,
            source=source,
            snapped_at=None,
            whole=whole,
            extra_meta={"tags": list(tags)} if tags else None,
            heading_context=heading_context,
            supersedes=_capture_ids(superseded),
        )
    except RunError as exc:
        raise SessionError(str(exc)) from exc
    return PairResult(written=written, replaced=len(superseded), whole=whole)


# ---- list -----------------------------------------------------------------


@dataclass(frozen=True)
class SourceSummary:
    """One source's footprint in the corpus."""

    source: str
    pairs: int
    saves: int


@dataclass(frozen=True)
class StateSummary:
    """Everything `log list` reports."""

    pairs_path: Path
    pair_count: int
    by_source: list[SourceSummary]
    open_sessions: list[SnapshotEntry]
    closed_sessions: list[SnapshotEntry]


def summarize_state(store: SessionStore) -> StateSummary:
    """Open sessions, saved pairs grouped by source, and the closed archives."""
    records = read_pairs(store)
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        grouped.setdefault(meta.get("source") or "(no source)", []).append(rec)

    by_source: list[SourceSummary] = []
    for src, recs in sorted(grouped.items()):
        capture_ids = {r.get("meta", {}).get("capture_id") for r in recs if isinstance(r, dict)}
        capture_ids.discard(None)
        by_source.append(SourceSummary(source=src, pairs=len(recs), saves=len(capture_ids)))

    return StateSummary(
        pairs_path=store.corpus_path,
        pair_count=len(records),
        by_source=by_source,
        open_sessions=load_snapshots(store.snapshots_dir),
        closed_sessions=load_snapshots(store.closed_snapshots_dir),
    )
