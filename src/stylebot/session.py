"""Rewrite-session state for `ai-style log`, as a library (D35).

`stylebot.capture` owns the capture mechanism; this module owns what
surrounds it: where state lives (`SessionStore`), snapshot persistence
(archived on close, so `reopen` / `save --redo` work), symlink
canonicalisation, and `save`'s disposition rule. `SessionError` (a
`RunError`) means the world is not ready, a `ValueError` that the arguments
contradict (D47). Nothing here prints or exits.
"""

from __future__ import annotations

import difflib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from stylebot import capture, config, draws, markdown, pairs
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
    """The on-disk home of session state and the corpus, under one data dir;
    resolved per invocation, never an import-time constant (D45)."""

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
    def captures_dir(self) -> Path:
        """`<data_dir>/captures`, the archives a save writes and `recut` re-lands (D96)."""
        return self.pairs_path.parent / "captures"

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
    """One snapshot directory, loaded or not: a listing carries an unreadable
    snapshot's failure as data rather than aborting."""

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
    """Every snapshot directory under `root` (identified by its `meta.json`),
    sorted; `[]` if it doesn't exist."""
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
    """Normalize a symlinked FILE to the path it points at, so a scratch-tree
    symlink (the blog's `_focus/`) cannot open a parallel session or write a
    transient path into the corpus. Only symlinks are rewritten; `cwd` is the
    tree the result is made relative to."""
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
    saves excluded, D84); [] while nothing has been saved."""
    if not store.draws_dir.is_dir():
        return []
    return list(draws.view_records(draws.iter_draws(store.draws_dir), draws.ViewPolicy()))


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
    return CorpusSummary(path=store.draws_dir, pairs=len(pairs), with_context=with_ctx)


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
    """Close a session by archiving its snapshot, keeping the base text and the
    original `snapped_at`, so a re-save after `reopen` lands in the same session."""
    move_snapshot(store.snapshot_path(source), store.closed_snapshot_path(source))


def restore_snapshot(store: SessionStore, source: Path) -> Snapshot:
    """Restore `source`'s archived session to the live snapshot path,
    `snapped_at` preserved; raises `FileNotFoundError` with no archive."""
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

    Prior pairs are superseded, never rewritten (D84): by default the same
    session's (interim `keep_open` saves); `"replace-all"` earlier sessions'
    too; `"append"` none. `redo` reopens the last closed session first so the
    re-diff is against the same base. `notify` receives the pre-capture lines.
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
    """One-shot pair capture from two texts, no session: appends by default;
    `replace` supersedes every prior pair for `source` (D84), and requires one."""
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


# ---- recut: re-land an edited archive --------------------------------------


@dataclass(frozen=True)
class PlannedPair:
    """One pair a recut would land: `status` against the archive's live landing
    (`kept` | `changed` | `new`), `flags` the sanity marks a human should read."""

    before: str
    after: str
    context: str
    status: str
    flags: tuple[str, ...]

    @property
    def similarity(self) -> float:
        return difflib.SequenceMatcher(None, self.before, self.after).ratio()


def _pair_flags(before: str, after: str, span_kinds: list[str | None]) -> tuple[str, ...]:
    """`identity` (near-unchanged), `replaced` (little shared text), `non-prose`
    (the whole before side lies in a protected block of the archive), `mixed`
    (a protected block sits inside it), `oversize`."""
    sim = difflib.SequenceMatcher(None, before, after).ratio()
    flags: list[str] = []
    if sim >= 0.98:
        flags.append("identity")
    elif sim < 0.3:
        flags.append("replaced")
    protected = [kind is not None for kind in span_kinds]
    if protected and all(protected):
        flags.append("non-prose")
    elif any(protected):
        flags.append("mixed")
    if len(before) > 4500 or len(after) > 4500:
        flags.append("oversize")
    return tuple(flags)


def plan_recut(
    store: SessionStore,
    archive_id: str,
    *,
    heading_context: bool = True,
    gap_paras: int = 0,
    gap_chars: int = 0,
) -> list[PlannedPair]:
    """What `recut_capture` would land, writing nothing (D96): each planned pair
    against the archive's live landing, flagged for a human sanity check."""
    archive_dir = store.captures_dir / archive_id
    arch = capture.read_archive(archive_dir)
    if arch is None:
        raise SessionError(f"no capture archive {archive_id} under {store.captures_dir}")
    live = {
        pairs.extract_slop(r): pairs.extract_target(r)
        for r in read_pairs(store)
        if capture.archive_id_of(r.get("meta") or {}) == archive_id
    }
    try:
        planned = capture.plan_pairs(
            arch["before_text"],
            arch["after_text"],
            whole=bool(arch.get("whole")),
            heading_context=heading_context,
            coalesce_gap_paras=gap_paras,
            coalesce_gap_chars=gap_chars,
        )
    except RunError as exc:
        raise SessionError(str(exc)) from exc
    kinds = markdown.line_kinds(arch["before_text"], headings=False)
    out: list[PlannedPair] = []
    for before, after, context in planned:
        status = "kept" if live.get(before) == after else ("changed" if before in live else "new")
        start = arch["before_text"].find(before)
        span_kinds = (
            kinds[
                arch["before_text"].count("\n", 0, start) : arch["before_text"].count(
                    "\n", 0, start + len(before)
                )
                + 1
            ]
            if start >= 0
            else []
        )
        out.append(
            PlannedPair(before, after, context, status, _pair_flags(before, after, span_kinds))
        )
    return out


@dataclass(frozen=True)
class RecutResult:
    """What one `recut` did."""

    archive_id: str
    written: list[dict]
    superseded: int


def recut_capture(
    store: SessionStore,
    archive_id: str,
    *,
    tags: Sequence[str] = (),
    heading_context: bool = True,
    gap_paras: int = 0,
    gap_chars: int = 0,
    notify: Callable[[str], None] | None = None,
) -> RecutResult:
    """Re-land `captures/<archive_id>/` after a hand edit (D96): the archive's
    live landings, original or an earlier recut, are superseded."""
    say = notify or (lambda _line: None)
    archive_dir = store.captures_dir / archive_id
    if not archive_dir.is_dir():
        raise SessionError(f"no capture archive {archive_id} under {store.captures_dir}")
    prior = [
        r for r in read_pairs(store) if capture.archive_id_of(r.get("meta") or {}) == archive_id
    ]
    if prior:
        say(f"superseded {len(prior)} prior pair(s) landed from {archive_id}")
    try:
        written = capture.recut_archive(
            archive_dir,
            pairs_path=store.pairs_path,
            supersedes=_capture_ids(prior),
            heading_context=heading_context,
            coalesce_gap_paras=gap_paras,
            coalesce_gap_chars=gap_chars,
            extra_meta={"tags": list(tags)} if tags else None,
        )
    except RunError as exc:
        raise SessionError(str(exc)) from exc
    return RecutResult(archive_id=archive_id, written=written, superseded=len(prior))


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
        pairs_path=store.draws_dir,
        pair_count=len(records),
        by_source=by_source,
        open_sessions=load_snapshots(store.snapshots_dir),
        closed_sessions=load_snapshots(store.closed_snapshots_dir),
    )
