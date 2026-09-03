"""Rewrite-session state for `ai-style log`, as a library.

`stylebot.capture` owns the capture *mechanism* — paragraph diffing, the pair
record, the archive. This module owns everything that used to sit in
`bin/ai_style_log.py` around it and was reachable only through click:

* **where session state lives** (`SessionStore`: the snapshots, closed-snapshots
  and `pairs.jsonl` paths under one data dir);
* **snapshot persistence** — a snapshot is a directory holding the base text as
  an ordinary file plus a `meta.json` sidecar, and closing a session *archives*
  that directory rather than deleting it, which is what makes `reopen` /
  `save --redo` possible;
* **symlink canonicalisation** (`canonical_source`), so the blog's `_focus/`
  render harness cannot key a session under a scratch tree that gets wiped;
* **`save`'s disposition rule** — same-session interim pairs are replaced,
  earlier sessions' pairs are kept, both overridable;
* **`tidy`'s latest-wins rule** — one save per source, newest `captured_at`.

Everything takes explicit paths and parameters, so a second caller (a test, an
editor sidecar, a batch script) can drive a session without going through the
CLI. `bin/ai_style_log.py` is the click shell over this: it resolves the data
dir, translates exceptions into exit codes, and does the printing.

Errors keep the two classes `ai-style` distinguishes. A `SessionError` means the
arguments were fine but the world is not ready (no open session, nothing
changed) — the CLI turns it into `click.ClickException`, exit 1. A `ValueError`
means the arguments themselves contradict — the CLI turns it into
`click.UsageError`, exit 2. Nothing here prints or exits.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from stylebot import capture, config
from stylebot.jsonl import read_jsonl, write_jsonl_atomic

__all__ = [
    "Disposition",
    "SNAPSHOT_CONTENT",
    "SNAPSHOT_META",
    "CorpusSummary",
    "PairResult",
    "PriorPairs",
    "RecutResult",
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
    "recut",
    "reopen_session",
    "restore_snapshot",
    "save_session",
    "snapshot_dirs",
    "summarize_state",
    "tidy_pairs",
    "write_snapshot",
]


# `save`'s three ways of disposing of prior pairs for the same source. "smart"
# is the default: replace this session's interim pairs, keep earlier sessions'.
Disposition = Literal["smart", "replace-all", "append"]


# A snapshot is a DIRECTORY mirroring the source path, holding the base text as
# an ordinary file plus a small sidecar:
#
#     snapshots/post/foo.qmd/content.md
#     snapshots/post/foo.qmd/meta.json   {source, snapped_at}
#
# Same reasoning as the capture archive (see stylebot.capture): the document is
# readable and editable on disk instead of being a JSON-escaped string, and it
# is `.md` so the blog's `*.qmd` tooling — the reformatting pre-commit hook, qmd
# search, the synthesis target walk — cannot see it.
SNAPSHOT_CONTENT = "content.md"
SNAPSHOT_META = "meta.json"


class SessionError(RuntimeError):
    """The arguments are fine; the world is not ready.

    No open session, nothing changed since the snapshot, no archive to reopen.
    The CLI maps this to `click.ClickException` (exit 1), which is what keeps it
    distinguishable from a contradictory invocation (`ValueError` -> exit 2).
    """


# ---- Where session state lives -------------------------------------------


@dataclass(frozen=True)
class SessionStore:
    """The on-disk home of session state and the corpus, under one data dir.

    Resolved per invocation rather than baked into module constants: import-time
    constants freeze whatever the environment looked like when the module was
    first imported, and in 2026-08 that made `log` the one subcommand that could
    quietly write to a cwd-relative `_training_pairs/` instead of the configured
    corpus.
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
        """Where closed sessions' snapshots are retained.

        A sibling directory rather than a suffix inside the snapshots dir,
        because the open-session listings rglob there and would otherwise report
        closed sessions as open.
        """
        return self.data_dir / "snapshots-closed"

    @property
    def pairs_path(self) -> Path:
        """The corpus file itself — what every downstream phase reads."""
        return self.data_dir / "pairs.jsonl"

    def snapshot_path(self, source: Path | str) -> Path:
        """Where the snapshot for `source` lives.

        Mirrors the source path under the snapshots dir as a directory, so
        `post/foo.qmd` -> `_training_pairs/snapshots/post/foo.qmd/`.
        """
        return self.snapshots_dir / str(source)

    def closed_snapshot_path(self, source: Path | str) -> Path:
        """Where the retained snapshot of `source`'s last closed session lives.

        One archive per source, overwritten each time a session closes — the
        redo workflow only ever wants the most recent base.
        """
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

    A session key *is* the source path as given (`SessionStore.snapshot_path`),
    so any second path to the same file opens a second, parallel session. The
    blog's `_focus/` render harness creates exactly that: it symlinks a post into
    a scratch project, and that scratch project is wiped and rebuilt on every
    invocation -- so a session keyed there loses its file the next time a
    different post is focused, and a `save` would write a transient, gitignored
    path into the committed `pairs.jsonl`.

    Only symlinks are rewritten. An ordinary path is returned untouched, so no
    existing session key can move. `cwd` defaults to the process's, and is the
    tree the returned path is made relative to; a target outside it comes back
    absolute, which is still better than a key under a directory that gets wiped.
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
    """Read `pairs.jsonl` into memory. Returns [] if the file is missing.

    Unparseable lines come back as ``{"_raw": line}`` so the rewrite path
    (`stylebot.jsonl.write_jsonl_atomic`) round-trips them instead of silently
    dropping them.
    """
    return read_jsonl(store.pairs_path, keep_undecodable=True)


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
    return CorpusSummary(path=store.pairs_path, pairs=len(pairs), with_context=with_ctx)


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
        """Prior pairs for this source logged outside a session (`pair`, `synth`).

        They carry `snapped_at=None`, so they have no session timestamp to sort
        or age and must be counted separately — sorting the mixed set raised
        `TypeError: '<' not supported between 'str' and 'NoneType'`.
        """
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
    """Diff `source` against its session snapshot and write the pair(s).

    Disposition of prior pairs for this source is **smart by default**: pairs
    from the *same* session (matching `meta.snapped_at`) are replaced — these are
    interim saves from `keep_open` that should not stack — while pairs from
    *earlier sessions* are kept, because they represent legitimate prior rewrites
    of the same file that both belong in the corpus. `"replace-all"` drops
    cross-session pairs too; `"append"` keeps everything, even interim pairs.

    `redo` reopens the last closed session first (a no-op if one is already
    open), so a corrected rewrite is re-diffed against the SAME base and lands in
    the same session, replacing rather than stacking.

    `notify` receives the lines describing work done *before* the capture — the
    reopen, and the disposition of prior pairs. They are emitted through a
    callback rather than returned because that work is already committed to disk
    when the capture raises, and the CLI must report it in order either way.
    """
    say = notify or (lambda _line: None)
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

    snapshot = read_snapshot(snap)
    before_text = snapshot.content
    after_text = Path(source).read_text(encoding="utf-8")
    if before_text == after_text:
        raise SessionError(f"no changes since session opened on {source}; nothing to save")

    session_snapped_at = snapshot.snapped_at
    all_records = read_pairs(store)
    prior = partition_prior_pairs(all_records, source=str(source), snapped_at=session_snapped_at)

    if disposition == "append":
        records_to_keep = all_records
    elif disposition == "replace-all":
        records_to_keep = prior.other_source
    elif disposition == "smart":
        records_to_keep = prior.other_source + prior.other_session
    else:  # pragma: no cover — a typo in a caller, not a user error
        raise ValueError(f"unknown disposition {disposition!r}")

    if len(records_to_keep) != len(all_records):
        write_jsonl_atomic(store.pairs_path, records_to_keep)

    # Report disposition before the save line.
    if disposition == "smart" and prior.same_session:
        say(f"replaced {len(prior.same_session)} interim pair(s) from this session")
    elif disposition == "replace-all":
        total_dropped = len(prior.same_session) + len(prior.other_session)
        if total_dropped:
            extra_note = (
                f" (including {len(prior.other_session)} from earlier session(s))"
                if prior.other_session
                else ""
            )
            say(f"replaced {total_dropped} prior pair(s) for {source}{extra_note}")

    extra: dict = {}
    if tags:
        extra["tags"] = list(tags)
    try:
        written = capture.capture_pairs(
            before_text,
            after_text,
            pairs_path=store.pairs_path,
            source=str(source),
            snapped_at=session_snapped_at,
            whole=whole,
            extra_meta=extra or None,
            heading_context=heading_context,
        )
    except ValueError as exc:
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

    Unlike `save_session`, this APPENDS by default: the typical use is logging
    one-off pairs from a clipboard or a script, where you usually want to add to
    the corpus rather than overwrite. `replace` drops prior pairs for `source`
    first, and requires one.

    `notify` is the same pre-capture progress channel as `save_session`'s: the
    removal is already committed to disk when the capture raises, so it has to
    be reportable either way.
    """
    say = notify or (lambda _line: None)
    if before_text == after_text:
        raise SessionError("before and after are identical; refusing to log")

    replaced = 0
    if replace:
        if source is None:
            # A contradiction in the arguments, not an unready world: the CLI
            # turns this into a UsageError (exit 2), so keep it a ValueError.
            raise ValueError("--replace requires --source")
        replaced = capture.remove_pairs_for_source(store.pairs_path, source)
        if replaced:
            say(f"replaced {replaced} prior pair(s) for {source}")

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
        )
    except ValueError as exc:
        raise SessionError(str(exc)) from exc
    return PairResult(written=written, replaced=replaced, whole=whole)


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
        pairs_path=store.pairs_path,
        pair_count=len(records),
        by_source=by_source,
        open_sessions=load_snapshots(store.snapshots_dir),
        closed_sessions=load_snapshots(store.closed_snapshots_dir),
    )


# ---- tidy: latest save per source wins ------------------------------------


@dataclass(frozen=True)
class TidyResult:
    """What a tidy would do, or did."""

    kept: list[dict]
    removed: int
    empty: bool
    applied: bool


def tidy_pairs(
    store: SessionStore, source: str | None = None, *, dry_run: bool = False
) -> TidyResult:
    """Keep only the latest save per source in `pairs.jsonl`.

    Saves are grouped by source path; the most recent `captured_at` wins, and
    older saves of the same source are removed. With `source` given, only that
    source is tidied. Records with no source (and undecodable lines) are always
    kept — there is nothing to compare them against.

    Normally unnecessary because `save_session` replaces by default; this is the
    retroactive cleanup after a run of `disposition="append"`.
    """
    records = read_pairs(store)
    if not records:
        return TidyResult(kept=[], removed=0, empty=True, applied=False)

    latest_ts: dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        meta = rec.get("meta", {})
        src = meta.get("source")
        ts = meta.get("captured_at", "")
        if src is None:
            continue
        if ts > latest_ts.get(src, ""):
            latest_ts[src] = ts

    kept: list[dict] = []
    removed = 0
    for rec in records:
        if not isinstance(rec, dict) or "meta" not in rec:
            kept.append(rec)
            continue
        meta = rec["meta"]
        src = meta.get("source")
        if src is None or (source is not None and src != source):
            kept.append(rec)
            continue
        if meta.get("captured_at", "") == latest_ts.get(src):
            kept.append(rec)
        else:
            removed += 1

    applied = bool(removed) and not dry_run
    if applied:
        write_jsonl_atomic(store.pairs_path, kept)
    return TidyResult(kept=kept, removed=removed, empty=False, applied=applied)


# ---- recut ----------------------------------------------------------------


@dataclass(frozen=True)
class RecutResult:
    """What a re-cut of the capture archives produced."""

    records: list[dict]
    captures: int
    pairs_before: int
    pairs_after: int
    dry_run: bool

    @property
    def lengths(self) -> list[int]:
        """After-side lengths of the re-cut pairs, sorted."""
        return sorted(len(r["messages"][2]["content"]) for r in self.records)

    @property
    def median_chars(self) -> int:
        lens = self.lengths
        return lens[len(lens) // 2]

    @property
    def p90_chars(self) -> int:
        lens = self.lengths
        return lens[int(0.9 * (len(lens) - 1))]

    @property
    def max_chars(self) -> int:
        return self.lengths[-1]


def recut(
    store: SessionStore,
    *,
    gap_paras: int,
    gap_chars: int,
    max_chars: int | None,
    min_chars: int = 0,
    max_similarity: float | None = None,
    dry_run: bool = False,
) -> RecutResult:
    """Re-cut real pairs from the capture archives at a new gap policy.

    Thin over `capture.recut_archives`, adding the before/after corpus counts and
    the length distribution a caller reports. `capture.iter_archives` refuses
    unmigrated legacy archives with a `ValueError`; that becomes a `SessionError`
    here, because the fix is a command the user has to run, not a traceback.
    """
    pairs_before = len(read_pairs(store))
    try:
        records = capture.recut_archives(
            store.pairs_path,
            coalesce_gap_paras=gap_paras,
            coalesce_gap_chars=gap_chars,
            max_chars=max_chars,
            min_chars=min_chars,
            max_similarity=max_similarity,
            dry_run=dry_run,
        )
    except ValueError as exc:
        raise SessionError(str(exc)) from exc

    captures = len({r["meta"]["capture_id"] for r in records}) if records else 0
    pairs_after = pairs_before if (dry_run or not records) else len(read_pairs(store))
    return RecutResult(
        records=records,
        captures=captures,
        pairs_before=pairs_before,
        pairs_after=pairs_after,
        dry_run=dry_run,
    )
