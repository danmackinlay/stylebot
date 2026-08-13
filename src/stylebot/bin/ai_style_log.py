"""
ai-style log: capture (slop -> Dan rewrite) training pairs for the fine-tune.

Phase 1 of the plan at `_plans/OVERVIEW.md`. Every time Dan rewrites
an AI draft, we want the (before, after) so it can go into the corpus. The
capture is deliberately manual — invoked by hand, no automation, no git hook,
no surprise side effects.

The cycle is one rewrite of one file per session:

    uv run ai-style log open <file>    # record the current text as "before"
    # ... user edits the file by hand, across one or more sittings ...
    uv run ai-style log save <file>    # write pair(s) for what changed, close session

`save` diffs at paragraph granularity and emits one pair per changed region,
not one pair per file. Pairs land in `$STYLEBOT_DATA_DIR/pairs.jsonl` in the
shared chat-completion contract that Phase 2 synthesis also writes and Phase 3
training reads (`stylebot.pairs` validates it; `stylebot.capture` implements
the mechanism, while this module holds the path policy and session workflow).

The full workflow — chunking rules, the `snapped_at` disposition of prior
pairs, the `--replace`/`--append`/`--keep-open`/`--redo` flags, editing
cadence, the one-shot `pair` entry point, and the on-disk layout — is
documented for users in `_training_pairs/README.md`.

Errors follow the one `ai-style` idiom: `click.UsageError` when the flags
themselves are wrong (contradictory or missing a companion — click exits 2),
`click.ClickException` when the flags are fine but the world is not (no open
session, nothing changed — exit 1). Never a bare `sys.exit` after an
`echo(err=True)`: raising leaves the exit code and the "Error:" prefix to
click, which is what makes the two classes distinguishable to a caller.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

from stylebot import capture, config
from stylebot.capture import diff_chunks  # noqa: F401 — callers/tests reach it as log.diff_chunks
from stylebot.jsonl import read_jsonl, write_jsonl_atomic


# Root for all logged state (the corpus), resolved through `stylebot.config` —
# the one home of the `--flag` > `$STYLEBOT_DATA_DIR` > cwd-relative
# `_training_pairs` ladder. `ai-style log` passes no flag: it is the
# convenience command where a mistake is cheap, so it takes the env/default.
#
# Why configurable: the corpus is the project's valuable, *private* asset. The
# stylebot code repo is public, so the corpus must live outside it. The logger
# runs in the blog repo and writes there; downstream phases (synthesis,
# training) run in this repo and point STYLEBOT_DATA_DIR at the same corpus.
# Both default and override are gitignored. See `_training_pairs/README.md`.
#
# Resolved lazily (per use, when a command runs) rather than baked into module
# constants at import time. Import-time constants freeze whatever the
# environment looked like when the module was first imported, which is the
# wrong moment: `log` is imported by the CLI entry point and by tests, and in
# 2026-08 that made it the one subcommand that could quietly write to a
# cwd-relative `_training_pairs/` instead of the configured corpus. Resolving
# per use is a dict lookup — there is nothing to cache.
def _data_dir() -> Path:
    """The corpus root for this invocation."""
    return config.resolve_data_dir()


def _snapshots_dir() -> Path:
    """Where open sessions' snapshots live."""
    return _data_dir() / "snapshots"


def _closed_snapshots_dir() -> Path:
    """Where closed sessions' snapshots are retained.

    Retained so `reopen` / `save --redo` can diff a corrected rewrite against
    the SAME base. A sibling directory rather than a suffix inside the
    snapshots dir, because the open-session listings rglob there and would
    otherwise report closed sessions as open.
    """
    return _data_dir() / "snapshots-closed"


def _pairs_path() -> Path:
    """The corpus file itself — what every downstream phase reads."""
    return _data_dir() / "pairs.jsonl"


# The capture mechanism lives in stylebot.capture; this CLI keeps the path
# policy (the resolvers above) and the session workflow. Call `capture.X`
# directly — the `_now_iso` / `_build_pair_record` aliases here were only ever
# reached through this module by tests, which now import them from their home.


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


def _snapshot_path(source: Path) -> Path:
    """Where the snapshot for `source` lives.

    Mirrors the source path under the snapshots dir as a directory. So
    `post/foo.qmd` -> `_training_pairs/snapshots/post/foo.qmd/`.
    """
    return _snapshots_dir() / str(source)


def _closed_snapshot_path(source: Path) -> Path:
    """Where the retained snapshot of `source`'s last closed session lives.

    Same mirrored layout as `_snapshot_path`, under the closed-snapshots dir.
    One archive per source, overwritten each time a session closes — the redo
    workflow only ever wants the most recent base.
    """
    return _closed_snapshots_dir() / str(source)


def _write_snapshot(snap: Path, *, source: str, snapped_at: str, content: str) -> None:
    """Write a snapshot directory (base text + sidecar)."""
    snap.mkdir(parents=True, exist_ok=True)
    (snap / SNAPSHOT_CONTENT).write_text(content, encoding="utf-8")
    (snap / SNAPSHOT_META).write_text(
        json.dumps({"source": source, "snapped_at": snapped_at}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _read_snapshot(snap: Path) -> dict:
    """Load a snapshot directory into the flat payload shape callers expect."""
    meta = json.loads((snap / SNAPSHOT_META).read_text(encoding="utf-8"))
    return {**meta, "content": (snap / SNAPSHOT_CONTENT).read_text(encoding="utf-8")}


def _move_snapshot(src: Path, dst: Path) -> None:
    """Move a snapshot directory, replacing any existing one at `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))


def _close_snapshot(snap: Path, source: Path) -> None:
    """Close a session by archiving its snapshot rather than deleting it.

    Retaining the base text is what makes `reopen` / `save --redo` possible:
    the pairs themselves cannot reconstruct it, since in chunk mode they hold
    only the changed regions. The archive keeps the original `snapped_at`, so
    a re-save after `reopen` lands in the same session and `save`'s smart
    disposition REPLACES the earlier pairs instead of stacking a near-duplicate
    onto the corpus.
    """
    _move_snapshot(snap, _closed_snapshot_path(source))


def _iter_pairs() -> list[dict]:
    """Read pairs.jsonl into memory. Returns [] if file is missing.

    Unparseable lines come back as ``{"_raw": line}`` so the rewrite path
    (`stylebot.jsonl.write_jsonl_atomic`) round-trips them instead of silently dropping.
    """
    return read_jsonl(_pairs_path(), keep_undecodable=True)


def _corpus_confirmation() -> str:
    """One-line 'it landed where training reads' confirmation for write commands.

    Resolves the data-dir to an ABSOLUTE path (a relative default like
    ``_training_pairs/`` otherwise hides *where* pairs went) and reports the
    running corpus size plus heading-context coverage, so each save visibly
    accumulates in the corpus you expect rather than a stray cwd dir.
    """
    pairs = _iter_pairs()
    with_ctx = sum(1 for r in pairs if (r.get("meta", {}).get("context") or "").strip())
    return (
        f"corpus now: {len(pairs)} pair(s) ({with_ctx} with heading context) "
        f"-> {_pairs_path().resolve()}"
    )


def _capture_and_append(
    *,
    before_text: str,
    after_text: str,
    source: str | None,
    snapped_at: str | None,
    whole: bool,
    extra_meta: dict | None,
    heading_context: bool = True,
) -> list[dict]:
    """`stylebot.capture.capture_pairs` against the module's corpus path."""
    return capture.capture_pairs(
        before_text,
        after_text,
        pairs_path=_pairs_path(),
        source=source,
        snapped_at=snapped_at,
        whole=whole,
        extra_meta=extra_meta,
        heading_context=heading_context,
    )


# ---- CLI ----


def _humanize_age(iso_ts: str) -> str:
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


def _remind_open_sessions(ctx: click.Context) -> None:
    """Print a short reminder of currently-open sessions unless --quiet.

    Called at the end of every command except `list` (which already
    prints the full state). The reminder makes stale sessions visible
    so they don't accumulate silently.
    """
    if ctx.obj and ctx.obj.get("quiet"):
        return
    snapshots_dir = _snapshots_dir()
    if not snapshots_dir.exists():
        return
    snaps = sorted(p.parent for p in snapshots_dir.rglob(SNAPSHOT_META))
    if not snaps:
        return

    entries: list[tuple[str, str]] = []
    for snap in snaps:
        try:
            payload = _read_snapshot(snap)
            source = payload.get("source", str(snap))
            age = _humanize_age(payload.get("snapped_at", ""))
            entries.append((source, age))
        except Exception:
            entries.append((str(snap), "unreadable"))

    width = max(len(src) for src, _ in entries)
    click.echo()
    click.echo(f"Open sessions ({len(entries)}):", err=True)
    for source, age in entries:
        click.echo(f"  {source:<{width}}  (opened {age})", err=True)
    click.echo(
        "  use `ai-style log -q ...` to suppress, `ai-style log drop <file>` to abandon", err=True
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress the reminder of open sessions printed after each command.",
)
@click.pass_context
def main(ctx: click.Context, quiet: bool):
    """Capture (slop -> Dan rewrite) training pairs for ai-style fine-tuning."""
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet


@main.command(name="open")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def open_session(ctx: click.Context, file: Path):
    """Open a rewrite session on FILE: record its current content as the 'before' state.

    Run this *before* you start rewriting an AI-generated draft. The
    session is sticky — `open` refuses to overwrite an existing one so
    the original (worst) state is preserved across editing sittings.
    A later `save` diffs the current contents against this recorded state.
    """
    snap = _snapshot_path(file)
    if snap.exists():
        raise click.ClickException(
            f"session already open for {file}; `drop` it, or `save` without --keep-open to close it"
        )
    snap.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(file),
        "snapped_at": capture.now_iso(),
        "content": file.read_text(encoding="utf-8"),
    }
    _write_snapshot(
        snap, source=payload["source"], snapped_at=payload["snapped_at"], content=payload["content"]
    )
    click.echo(f"opened session for {file} ({len(payload['content'])} chars) -> {snap}")
    _remind_open_sessions(ctx)


def _restore_snapshot(file: Path) -> dict:
    """Restore `file`'s archived session to the live snapshot path.

    Returns the restored payload. Raises `FileNotFoundError` when there is no
    archive to restore. Preserves `snapped_at` verbatim — see `_close_snapshot`
    for why that is the load-bearing part.
    """
    closed = _closed_snapshot_path(file)
    if not closed.exists():
        raise FileNotFoundError(closed)
    payload = _read_snapshot(closed)
    snap = _snapshot_path(file)
    snap.parent.mkdir(parents=True, exist_ok=True)
    _move_snapshot(closed, snap)
    return payload


@main.command(name="reopen")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def reopen_session(ctx: click.Context, file: Path):
    """Reopen FILE's last closed session, restoring its original 'before' state.

    For the do-over case: you saved, then spotted a typo in the rewrite. Fix
    the file, `reopen`, then `save` again — the re-save diffs against the same
    base and replaces the previous pairs rather than logging a near-duplicate.
    `save --redo` does both steps in one.

    Also recovers an accidental `drop`.
    """
    snap = _snapshot_path(file)
    if snap.exists():
        raise click.ClickException(
            f"session already open for {file}; nothing to reopen "
            "(`save` against it, or `drop` it first)"
        )
    try:
        payload = _restore_snapshot(file)
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"no closed session for {file}; run `open` to start a fresh one"
        ) from exc
    # Surface the age: resurrecting a weeks-old base against a much-changed
    # file produces a huge, unrepresentative diff, and that should be visible.
    age = _humanize_age(payload.get("snapped_at", ""))
    click.echo(
        f"reopened session for {file} "
        f"({len(payload.get('content', ''))} chars, originally opened {age})"
    )
    _remind_open_sessions(ctx)


@main.command(name="save")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--tag", "tags", multiple=True, help="Tag the pair(s) for later filtering.")
@click.option(
    "--keep-open",
    is_flag=True,
    help="Leave the session open after saving so the original 'before' state "
    "stays sticky for future saves against the same starting point. Useful "
    "for banking interim progress.",
)
@click.option(
    "--whole",
    is_flag=True,
    help="Emit one pair for the whole file rather than one pair per changed "
    "paragraph region. Escape hatch for cases where the chunk diff loses too "
    "much context (heavy structural rearrangement, paragraph moves).",
)
@click.option(
    "--replace",
    "force_replace",
    is_flag=True,
    help="Force-replace every prior pair for this source, even ones from "
    "different sessions. Use when you genuinely want to abandon earlier "
    "sessions' pairs. Mutually exclusive with --append.",
)
@click.option(
    "--append",
    "force_append",
    is_flag=True,
    help="Force-keep every prior pair for this source, even interim pairs "
    "from the current session. Mutually exclusive with --replace.",
)
@click.option(
    "--redo",
    is_flag=True,
    help="Reopen the last closed session for this file first, then save. The "
    "do-over shorthand: you spotted a typo after saving and want the pair "
    "recaptured against the same base. No-op if a session is already open.",
)
@click.option(
    "--heading-context/--no-heading-context",
    "heading_context",
    default=True,
    help="Prepend the nearest preceding section heading (from the after text) "
    "as fixed context to each changed chunk, identically on both sides, and "
    "record it in meta.context. On by default; --no-heading-context captures "
    "bare bodies instead. Ignored for --whole (no single heading).",
)
@click.pass_context
def save_session(
    ctx: click.Context,
    file: Path,
    tags: tuple[str, ...],
    keep_open: bool,
    whole: bool,
    force_replace: bool,
    force_append: bool,
    redo: bool,
    heading_context: bool,
):
    """Save FILE's rewrite as pair(s) in pairs.jsonl, then close the session.

    By default emits one pair per changed paragraph region (use --whole for
    one whole-file pair).

    Disposition of prior pairs for this source is **smart by default**:
    pairs from the *same* session (matching `meta.snapped_at`) are replaced
    — these are interim saves from `--keep-open` that should not stack —
    while pairs from *earlier sessions* (different `snapped_at`) are kept,
    because they represent legitimate prior rewrites of the same file
    that both belong in the training corpus.

    Override the default with --replace (force full replace, dropping
    cross-session pairs too) or --append (force keep everything, even
    interim pairs from the current session).
    """
    if force_replace and force_append:
        raise click.UsageError("--replace and --append are mutually exclusive")

    snap = _snapshot_path(file)
    if redo and not snap.exists():
        try:
            restored = _restore_snapshot(file)
        except FileNotFoundError as exc:
            raise click.ClickException(
                f"--redo: no closed session for {file} to reopen; run `open` first"
            ) from exc
        click.echo(
            f"reopened session for {file} "
            f"(originally opened {_humanize_age(restored.get('snapped_at', ''))})"
        )
    if not snap.exists():
        hint = "run `open` first"
        if _closed_snapshot_path(file).exists():
            hint += ", or `save --redo` to reopen the last closed session"
        raise click.ClickException(f"no open session for {file}; {hint}")

    payload = _read_snapshot(snap)
    before_text = payload["content"]
    after_text = file.read_text(encoding="utf-8")

    if before_text == after_text:
        raise click.ClickException(f"no changes since session opened on {file}; nothing to save")

    extra: dict = {}
    if tags:
        extra["tags"] = list(tags)

    # ---- Disposition of prior pairs for this source ----
    #
    # Smart default: replace same-session interim pairs, keep cross-session.
    # --replace forces full drop of all prior pairs for this source.
    # --append forces keep of all prior pairs, even same-session.
    session_snapped_at = payload.get("snapped_at")
    all_records = _iter_pairs()
    same_session: list[dict] = []
    other_session: list[dict] = []
    other_source: list[dict] = []
    for rec in all_records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        if meta.get("source") != str(file):
            other_source.append(rec)
        elif meta.get("snapped_at") == session_snapped_at:
            same_session.append(rec)
        else:
            other_session.append(rec)

    if force_append:
        records_to_keep = all_records
        disposition = "append"
    elif force_replace:
        records_to_keep = other_source
        disposition = "replace-all"
    else:
        records_to_keep = other_source + other_session
        disposition = "smart"

    if len(records_to_keep) != len(all_records):
        write_jsonl_atomic(_pairs_path(), records_to_keep)

    # Report disposition before the save line
    if disposition == "smart" and same_session:
        click.echo(f"replaced {len(same_session)} interim pair(s) from this session")
    elif disposition == "replace-all":
        total_dropped = len(same_session) + len(other_session)
        if total_dropped:
            extra_note = (
                f" (including {len(other_session)} from earlier session(s))"
                if other_session
                else ""
            )
            click.echo(f"replaced {total_dropped} prior pair(s) for {file}{extra_note}")

    try:
        written = _capture_and_append(
            before_text=before_text,
            after_text=after_text,
            source=str(file),
            snapped_at=session_snapped_at,
            whole=whole,
            extra_meta=extra or None,
            heading_context=heading_context,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not keep_open:
        _close_snapshot(snap, file)

    total_before = sum(rec["meta"]["before_chars"] for rec in written)
    total_after = sum(rec["meta"]["after_chars"] for rec in written)
    mode = "whole-file" if whole else "chunk-by-chunk"
    status = "session kept open" if keep_open else "session closed"
    click.echo(
        f"saved {len(written)} pair(s) for {file} "
        f"({mode}; {total_before} -> {total_after} chars total; {status})"
    )
    click.echo(_corpus_confirmation())

    # Post-save reporting of preserved cross-session / appended pairs
    if disposition == "smart" and other_session:
        # Pairs logged outside a session (`log pair`, `synth`) carry snapped_at=None;
        # they have no session timestamp to sort or age, so count them separately.
        stamps = {
            rec.get("meta", {}).get("snapped_at") for rec in other_session if isinstance(rec, dict)
        }
        sessions_ts = sorted(ts for ts in stamps if ts)
        sessionless = len(other_session) - sum(
            1
            for rec in other_session
            if isinstance(rec, dict) and rec.get("meta", {}).get("snapped_at")
        )
        parts = []
        if sessions_ts:
            parts.append(
                f"{len(sessions_ts)} earlier session(s) "
                f"[oldest opened {_humanize_age(sessions_ts[0])}]"
            )
        if sessionless:
            parts.append(f"{sessionless} logged outside a session")
        click.echo(f"  kept {len(other_session)} pair(s) from {', '.join(parts)}")
    elif disposition == "append":
        total_kept = len(same_session) + len(other_session)
        if total_kept:
            click.echo(
                f"  appended alongside {total_kept} prior pair(s) for this source "
                f"({len(same_session)} from this session, {len(other_session)} from earlier)"
            )

    _remind_open_sessions(ctx)


@main.command(name="drop")
@click.argument("file", type=click.Path(dir_okay=False, path_type=Path))
@click.pass_context
def drop_session(ctx: click.Context, file: Path):
    """Drop the open session for FILE without saving any pair."""
    snap = _snapshot_path(file)
    if not snap.exists():
        raise click.ClickException(f"no open session for {file}")
    _close_snapshot(snap, file)
    click.echo(f"dropped session for {file} (`reopen` restores it if that was a mistake)")
    _remind_open_sessions(ctx)


@main.command(name="list")
def list_state():
    """List open sessions, saved pairs (grouped by source), and totals."""
    snapshots_dir = _snapshots_dir()
    snaps = (
        sorted(p.parent for p in snapshots_dir.rglob(SNAPSHOT_META))
        if snapshots_dir.exists()
        else []
    )
    records = _iter_pairs()
    pair_count = len(records)

    # Group pairs by source for a quick summary
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        src = meta.get("source") or "(no source)"
        by_source.setdefault(src, []).append(rec)

    click.echo(f"{pair_count} pair(s) across {len(by_source)} source(s) in {_pairs_path()}")
    for src, recs in sorted(by_source.items()):
        capture_ids = {r.get("meta", {}).get("capture_id") for r in recs if isinstance(r, dict)}
        capture_ids.discard(None)
        click.echo(f"  {src}: {len(recs)} pair(s) across {len(capture_ids)} save(s)")

    click.echo(f"{len(snaps)} open session(s)")
    for snap in snaps:
        try:
            payload = _read_snapshot(snap)
            click.echo(
                f"  {payload.get('source', snap)} "
                f"(opened {payload.get('snapped_at', '?')}, "
                f"{len(payload.get('content', ''))} chars)"
            )
        except Exception as exc:
            click.echo(f"  {snap} [unreadable: {exc}]")

    closed_dir = _closed_snapshots_dir()
    closed = (
        sorted(p.parent for p in closed_dir.rglob(SNAPSHOT_META)) if closed_dir.exists() else []
    )
    if closed:
        click.echo(f"{len(closed)} closed session(s), redoable via `reopen` / `save --redo`")
        for snap in closed:
            try:
                payload = _read_snapshot(snap)
                click.echo(
                    f"  {payload.get('source', snap)} "
                    f"(opened {_humanize_age(payload.get('snapped_at', ''))}, "
                    f"{len(payload.get('content', ''))} chars)"
                )
            except Exception as exc:
                click.echo(f"  {snap} [unreadable: {exc}]")


@main.command(name="tidy")
@click.argument("source", required=False)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report what would be removed without modifying pairs.jsonl.",
)
@click.pass_context
def tidy_pairs(ctx: click.Context, source: str | None, dry_run: bool):
    """Keep only the latest save per source in pairs.jsonl.

    Saves are grouped by source path; the most recent captured_at timestamp
    wins. Older saves of the same source are removed. With SOURCE given,
    only tidy that source. With no arguments, tidy every source.

    Normally unnecessary because `save` replaces by default. Useful if
    you've used `save --append` and want to clean up retroactively.
    """
    records = _iter_pairs()
    if not records:
        click.echo("pairs.jsonl is empty; nothing to tidy")
        return

    # Find latest captured_at per source (only among records that have a source)
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
        ts = meta.get("captured_at", "")
        if src is None:
            kept.append(rec)
            continue
        if source is not None and src != source:
            kept.append(rec)
            continue
        if ts == latest_ts.get(src):
            kept.append(rec)
        else:
            removed += 1

    if removed == 0:
        click.echo("nothing to remove; every source already has a single save")
        _remind_open_sessions(ctx)
        return

    click.echo(f"{'would remove' if dry_run else 'removed'} {removed} pair(s), keeping {len(kept)}")
    if not dry_run:
        write_jsonl_atomic(_pairs_path(), kept)
    _remind_open_sessions(ctx)


@main.command(name="recut")
@click.option(
    "--gap-paras",
    default=2,
    show_default=True,
    type=int,
    help="Bridge unchanged runs of at most this many paragraphs.",
)
@click.option(
    "--gap-chars",
    default=800,
    show_default=True,
    type=int,
    help="Bridge unchanged runs of at most this many characters.",
)
@click.option(
    "--max-chars",
    default=8000,
    show_default=True,
    type=int,
    help="Coalesced pairs over this size fall back to the uncoalesced cut (0 disables).",
)
@click.option(
    "--min-chars",
    default=80,
    show_default=True,
    type=int,
    help="Drop re-cut pairs whose after-side is shorter than this.",
)
@click.option("--dry-run", is_flag=True, help="Report the new cut without modifying pairs.jsonl.")
@click.pass_context
def recut_pairs(ctx, gap_paras, gap_chars, max_chars, min_chars, dry_run):
    """Re-cut real pairs from the capture archives at a new gap policy.

    The archives under `captures/` hold the full before/after texts, so the
    corpus can be re-derived at a different cut without recapturing. Bridging
    short unchanged runs keeps a multi-paragraph rewrite as ONE pair instead of
    fragmenting it — the edits that are *about* cross-paragraph consistency are
    exactly the ones the default cut chops up.

    Idempotent: re-cut records reuse their archive's capture_id, so every pair
    from a re-cut capture (original or previous re-cut) is replaced, not
    stacked. Captures with no archive keep their native cut.
    """
    before = len(_iter_pairs())
    try:
        records = capture.recut_archives(
            _pairs_path(),
            coalesce_gap_paras=gap_paras,
            coalesce_gap_chars=gap_chars,
            max_chars=max_chars or None,
            min_chars=min_chars,
            dry_run=dry_run,
        )
    except ValueError as exc:
        # e.g. unmigrated legacy JSON archives — a legible instruction, not a
        # traceback, since the fix is a command the user has to run.
        raise click.ClickException(str(exc)) from exc
    if not records:
        click.echo("no archives to re-cut (captures/ is empty or holds only --whole captures)")
        return
    lens = sorted(len(r["messages"][2]["content"]) for r in records)
    med = lens[len(lens) // 2]
    caps = len({r["meta"]["capture_id"] for r in records})
    verb = "would write" if dry_run else "wrote"
    click.echo(
        f"{verb} {len(records)} pair(s) from {caps} capture(s) "
        f"[median {med} chars, p90 {lens[int(0.9 * (len(lens) - 1))]}, max {lens[-1]}]"
    )
    if not dry_run:
        click.echo(f"pairs.jsonl: {before} -> {len(_iter_pairs())}")
    _remind_open_sessions(ctx)


def _read_source(spec: str) -> str:
    """Read text from a path or '-' for stdin."""
    if spec == "-":
        return sys.stdin.read()
    return Path(spec).read_text(encoding="utf-8")


@main.command()
@click.option(
    "--before",
    "before_spec",
    required=True,
    help="Path to the 'slop' (AI-generated) text, or '-' for stdin.",
)
@click.option(
    "--after",
    "after_spec",
    required=True,
    help="Path to the rewritten text, or '-' for stdin.",
)
@click.option(
    "--source",
    "source",
    default=None,
    help="Optional source label stored in meta (e.g. a file path or 'clipboard').",
)
@click.option("--tag", "tags", multiple=True, help="Tag the pair(s) for later filtering.")
@click.option(
    "--whole",
    is_flag=True,
    help="Emit one pair for the entire input rather than per-chunk.",
)
@click.option(
    "--replace",
    "replace",
    is_flag=True,
    help="If --source is given, remove any prior pairs for that source first. "
    "By default `pair` appends — unlike `save`, which replaces by default.",
)
@click.option(
    "--heading-context/--no-heading-context",
    "heading_context",
    default=True,
    help="Prepend the nearest preceding section heading (from the after text) "
    "as fixed context to each changed chunk, identically on both sides, and "
    "record it in meta.context. On by default; --no-heading-context captures "
    "bare bodies instead. Ignored for --whole (no single heading).",
)
@click.pass_context
def pair(
    ctx: click.Context,
    before_spec: str,
    after_spec: str,
    source: str | None,
    tags: tuple[str, ...],
    whole: bool,
    replace: bool,
    heading_context: bool,
):
    """One-shot pair capture from files or stdin (no session dance).

    Unlike `save`, `pair` defaults to APPEND. The typical use is logging
    one-off pairs from clipboard or scripts, where you usually want to add
    to the corpus rather than overwrite. Pass --replace (with --source) to
    drop prior pairs for that source first.
    """
    if before_spec == "-" and after_spec == "-":
        raise click.UsageError("only one of --before/--after may be '-'")
    before_text = _read_source(before_spec)
    after_text = _read_source(after_spec)

    if before_text == after_text:
        raise click.ClickException("before and after are identical; refusing to log")

    if replace:
        if source is None:
            raise click.UsageError("--replace requires --source")
        removed = capture.remove_pairs_for_source(_pairs_path(), source)
        if removed:
            click.echo(f"replaced {removed} prior pair(s) for {source}")

    extra = {"tags": list(tags)} if tags else None
    try:
        written = _capture_and_append(
            before_text=before_text,
            after_text=after_text,
            source=source,
            snapped_at=None,
            whole=whole,
            extra_meta=extra,
            heading_context=heading_context,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    total_before = sum(rec["meta"]["before_chars"] for rec in written)
    total_after = sum(rec["meta"]["after_chars"] for rec in written)
    mode = "whole-input" if whole else "chunk-by-chunk"
    click.echo(
        f"saved {len(written)} pair(s) ({mode}; {total_before} -> {total_after} chars total)"
    )
    click.echo(_corpus_confirmation())
    _remind_open_sessions(ctx)


if __name__ == "__main__":
    main()
