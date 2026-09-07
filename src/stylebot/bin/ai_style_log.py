"""ai-style log: capture (slop -> rewrite) training pairs, Phase 1.

Deliberately manual — no hook, no surprise side effects. One rewrite of one
file per session:

    uv run ai-style log open <file>    # record the current text as "before"
    # ... edit the file by hand, across one or more sittings ...
    uv run ai-style log save <file>    # write pair(s) for what changed, close

A click shell over `stylebot.session` and `stylebot.capture` (D35); errors
follow the package idiom (D47). The workflow is `_training_pairs/README.md`.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from stylebot import session
from stylebot import splits as splits_mod
from stylebot.capture import diff_chunks  # noqa: F401 — callers/tests reach it as log.diff_chunks
from stylebot.cli_options import (
    ErrorTranslatingGroup,
    SectionedCommand,
    bundle_options,
    bundles_from_kwargs,
    click_emitter,
    opt,
)
from stylebot.session import SessionStore, SnapshotEntry, humanize_age


@dataclass(frozen=True)
class CaptureOpts:
    """The flags `save` and `pair` share — what a captured pair looks like (D78).
    `--replace` is not here: it means different things on the two commands."""

    cli_section = "Capture"

    tags: Sequence[str] = opt(
        (), "--tag", "tags", multiple=True, help="Tag the pair(s) for later filtering."
    )
    whole: bool = opt(
        False,
        "--whole",
        is_flag=True,
        help="Emit ONE pair for the whole input rather than one per changed paragraph region. "
        "Escape hatch for rewrites the chunk diff cuts badly — heavy structural rearrangement, "
        "paragraph moves.",
    )
    heading_context: bool = opt(
        True,
        "--heading-context/--no-heading-context",
        "heading_context",
        help="Prepend the nearest preceding section heading (from the after text) "
        "as fixed context to each changed chunk, identically on both sides, and "
        "record it in meta.context. On by default; --no-heading-context captures "
        "bare bodies instead. Ignored for --whole (no single heading).",
    )


def capture_options(**kw):
    """The shared `save`/`pair` flags. `defaults=`/`exclude=` as elsewhere."""
    return bundle_options([CaptureOpts], **kw)


# Root for all logged state, resolved through `stylebot.config` per command,
# never at import (D45); `log` takes the env/default because a mistake is cheap.
def _store() -> SessionStore:
    """The session/corpus store for this invocation."""
    return SessionStore.resolve()


def _warn_if_eval_post(store: SessionStore, source: Path) -> None:
    """Say so, once, when a capture lands on a frozen-eval post: a notice, not
    `synth`'s refusal (D34), since a real rewrite belongs in the corpus and the
    splits contract keeps it out of training."""
    try:
        splits = splits_mod.load_corpus_splits(store.data_dir)
    except (OSError, ValueError) as exc:  # a broken splits.json must not block a capture
        click.secho(f"note: could not read splits.json ({exc})", fg="yellow", err=True)
        return
    if splits is None or splits_mod.role_of(str(source), splits) != "eval":
        return
    click.secho(
        f"NOTE: {source} is a FROZEN EVAL post — this capture is legitimate (it is your "
        f"own rewrite) but it will never train anything, and it scores the final eval.",
        fg="yellow",
        err=True,
    )


def _echo_snapshots(entries: list[SnapshotEntry], *, age: bool = False) -> None:
    """List snapshot directories one per line, tolerating an unreadable one.

    `age` renders `snapped_at` as a relative age rather than the raw stamp.
    """
    for entry in entries:
        if entry.snapshot is None:
            click.echo(f"  {entry.path} [unreadable: {entry.error}]")
            continue
        stamp = entry.snapshot.snapped_at
        when = humanize_age(stamp) if age else (stamp or "?")
        click.echo(f"  {entry.source} (opened {when}, {entry.snapshot.chars} chars)")


def _canonical_source(ctx, param, value):
    """click callback wrapping `session.canonical_source`."""
    return session.canonical_source(value)


def _corpus_confirmation(store: SessionStore) -> str:
    """One-line confirmation for write commands: the absolute data dir, the
    running corpus size and heading-context coverage."""
    summary = session.corpus_summary(store)
    return (
        f"corpus now: {summary.pairs} pair(s) ({summary.with_context} with heading context) "
        f"-> {summary.path.resolve()}"
    )


def _remind_open_sessions(ctx: click.Context) -> None:
    """Print a short reminder of currently-open sessions unless --quiet, at the
    end of every command except `list`."""
    if ctx.obj and ctx.obj.get("quiet"):
        return
    entries = session.load_snapshots(_store().snapshots_dir)
    if not entries:
        return

    rows = [
        (
            entry.source,
            humanize_age(entry.snapshot.snapped_at) if entry.snapshot else "unreadable",
        )
        for entry in entries
    ]
    width = max(len(src) for src, _ in rows)
    click.echo()
    click.echo(f"Open sessions ({len(rows)}):", err=True)
    for source, age in rows:
        click.echo(f"  {source:<{width}}  (opened {age})", err=True)
    click.echo(
        "  use `ai-style log -q ...` to suppress, `ai-style log drop <file>` to abandon", err=True
    )


@click.group(cls=ErrorTranslatingGroup, context_settings={"help_option_names": ["-h", "--help"]})
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
@click.argument(
    "file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    callback=_canonical_source,
)
@click.pass_context
def open_session(ctx: click.Context, file: Path):
    """Open a rewrite session on FILE: record its current content as the 'before'
    state. Sticky: refuses to overwrite an existing session, so the original
    state survives across sittings."""
    store = _store()
    snapshot = session.open_session(store, file)
    click.echo(f"opened session for {file} ({snapshot.chars} chars) -> {store.snapshot_path(file)}")
    _remind_open_sessions(ctx)


@main.command(name="reopen")
@click.argument(
    "file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    callback=_canonical_source,
)
@click.pass_context
def reopen_session(ctx: click.Context, file: Path):
    """Reopen FILE's last closed session, restoring its 'before' state, so a
    corrected `save` re-diffs against the same base and replaces the previous
    pairs (`save --redo` does both). Also recovers an accidental `drop`."""
    snapshot = session.reopen_session(_store(), file)
    # Surface the age: a weeks-old base gives a huge, unrepresentative diff.
    click.echo(
        f"reopened session for {file} "
        f"({snapshot.chars} chars, originally opened {humanize_age(snapshot.snapped_at)})"
    )
    _remind_open_sessions(ctx)


@main.command(name="save", cls=SectionedCommand)
@click.argument(
    "file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    callback=_canonical_source,
)
@click.option(
    "--keep-open",
    is_flag=True,
    help="Leave the session open after saving so the original 'before' state "
    "stays sticky for future saves against the same starting point. Useful "
    "for banking interim progress.",
)
@click.option(
    "--replace",
    "force_replace",
    is_flag=True,
    help="Supersede every prior pair for this source, even ones from "
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
@capture_options()
@click.pass_context
def save_session(
    ctx: click.Context,
    file: Path,
    keep_open: bool,
    force_replace: bool,
    force_append: bool,
    redo: bool,
    **kw,
):
    """Save FILE's rewrite as draws (D84), then close the session.

    One pair per changed paragraph region (--whole for one pair). Prior pairs
    from the same session are superseded by default; --replace supersedes
    earlier sessions' too, --append keeps everything.
    """
    if force_replace and force_append:
        raise click.UsageError("--replace and --append are mutually exclusive")

    cap: CaptureOpts = bundles_from_kwargs(kw, {"capture": CaptureOpts})["capture"]
    disposition = "append" if force_append else ("replace-all" if force_replace else "smart")

    store = _store()
    result = session.save_session(
        store,
        file,
        keep_open=keep_open,
        disposition=disposition,
        redo=redo,
        tags=cap.tags,
        whole=cap.whole,
        heading_context=cap.heading_context,
        notify=click.echo,
    )

    _warn_if_eval_post(store, file)

    mode = "whole-file" if result.whole else "chunk-by-chunk"
    status = "session kept open" if result.kept_open else "session closed"
    click.echo(
        f"saved {len(result.written)} pair(s) for {file} "
        f"({mode}; {result.before_chars} -> {result.after_chars} chars total; {status})"
    )
    click.echo(_corpus_confirmation(store))

    # Post-save reporting of preserved cross-session / appended pairs
    prior = result.prior
    if result.disposition == "smart" and prior.other_session:
        # Pairs logged outside a session carry snapped_at=None: counted apart.
        stamps = prior.earlier_session_stamps
        parts = []
        if stamps:
            parts.append(
                f"{len(stamps)} earlier session(s) [oldest opened {humanize_age(stamps[0])}]"
            )
        if prior.sessionless:
            parts.append(f"{prior.sessionless} logged outside a session")
        click.echo(f"  kept {len(prior.other_session)} pair(s) from {', '.join(parts)}")
    elif result.disposition == "append":
        total_kept = len(prior.same_session) + len(prior.other_session)
        if total_kept:
            click.echo(
                f"  appended alongside {total_kept} prior pair(s) for this source "
                f"({len(prior.same_session)} from this session, "
                f"{len(prior.other_session)} from earlier)"
            )

    _remind_open_sessions(ctx)


@main.command(name="mine", cls=SectionedCommand)
@click.option(
    "--transcripts",
    "transcripts_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A Claude Code project directory of session JSONLs (~/.claude/projects/<slug>/).",
)
@click.option(
    "--blog-root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The prose checkout the sessions edited; Dan's next commit is the after side.",
)
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where draws and captures/ are appended. Required: this grows the real stratum.",
)
@click.option("--since", default=None, help="Skip sessions whose last edit precedes this ISO date.")
@click.option(
    "--claude-left",
    "claude_left",
    default=None,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A directory of <session>/<path> copies of each file as Claude left it (the blog's "
    "PostToolUse hook); such a window replays nothing.",
)
@click.option(
    "--apply",
    is_flag=True,
    help="Write draws, archives and the turns sidecar (default: report only).",
)
@click.option("--no-turns", is_flag=True, help="With --apply, do not write transcript_turns.jsonl.")
@click.option(
    "--gap-paras",
    default=0,
    show_default=True,
    type=int,
    help="Coalesce gap in paragraphs (the blog uses its recut policy's).",
)
@click.option(
    "--gap-chars", default=0, show_default=True, type=int, help="Coalesce gap in characters."
)
@click.option("--heading-context/--no-heading-context", default=True, help="As for save.")
def mine_transcripts_cmd(
    transcripts_dir,
    blog_root,
    data_dir,
    since,
    claude_left,
    apply,
    no_turns,
    gap_paras,
    gap_chars,
    heading_context,
):
    """Replay Claude Code sessions and capture what Dan changed afterwards (D89).

    Each session's .qmd edits are replayed onto the file as Claude saw it; the
    result is paired with the file at Dan's next commit through the same differ
    `save` uses, keeping only paragraphs Claude touched. Dry-run by default.
    """
    from stylebot import transcripts

    report = transcripts.mine_transcripts(
        transcripts_dir,
        blog_root=blog_root,
        data_dir=data_dir,
        since=since,
        claude_left=claude_left,
        dry_run=not apply,
        write_turns=not no_turns,
        heading_context=heading_context,
        coalesce_gap_paras=gap_paras,
        coalesce_gap_chars=gap_chars,
        emit=click_emitter(),
    )
    click.echo("\n".join(report.lines()))
    if apply:
        click.echo(f"wrote {len(report.written)} pair(s) -> {data_dir}")
    else:
        click.echo("dry run: nothing written (pass --apply)")


@main.command(name="drop")
@click.argument(
    "file",
    type=click.Path(dir_okay=False, path_type=Path),
    callback=_canonical_source,
)
@click.pass_context
def drop_session(ctx: click.Context, file: Path):
    """Drop the open session for FILE without saving any pair."""
    session.drop_session(_store(), file)
    click.echo(f"dropped session for {file} (`reopen` restores it if that was a mistake)")
    _remind_open_sessions(ctx)


@main.command(name="list")
def list_state():
    """List open sessions, saved pairs (grouped by source), and totals."""
    state = session.summarize_state(_store())

    click.echo(
        f"{state.pair_count} pair(s) across {len(state.by_source)} source(s) in {state.pairs_path}"
    )
    for row in state.by_source:
        click.echo(f"  {row.source}: {row.pairs} pair(s) across {row.saves} save(s)")

    click.echo(f"{len(state.open_sessions)} open session(s)")
    _echo_snapshots(state.open_sessions)

    if state.closed_sessions:
        click.echo(
            f"{len(state.closed_sessions)} closed session(s), redoable via `reopen` / `save --redo`"
        )
        # Relative age for closed sessions: the live question is whether a
        # `--redo` would diff against prose that has moved on.
        _echo_snapshots(state.closed_sessions, age=True)


def _read_source(spec: str) -> str:
    """Read text from a path or '-' for stdin."""
    if spec == "-":
        return sys.stdin.read()
    return Path(spec).read_text(encoding="utf-8")


@main.command(cls=SectionedCommand)
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
@click.option(
    "--replace",
    "replace",
    is_flag=True,
    help="If --source is given, supersede every prior pair for that source. "
    "By default `pair` appends — unlike `save`, which supersedes by default.",
)
@capture_options()
@click.pass_context
def pair(
    ctx: click.Context,
    before_spec: str,
    after_spec: str,
    source: str | None,
    replace: bool,
    **kw,
):
    """One-shot pair capture from files or stdin, no session. Appends by
    default; --replace (with --source) supersedes prior pairs for that source."""
    cap: CaptureOpts = bundles_from_kwargs(kw, {"capture": CaptureOpts})["capture"]
    if before_spec == "-" and after_spec == "-":
        raise click.UsageError("only one of --before/--after may be '-'")

    store = _store()
    result = session.capture_pair(
        store,
        before_text=_read_source(before_spec),
        after_text=_read_source(after_spec),
        source=source,
        replace=replace,
        tags=cap.tags,
        whole=cap.whole,
        heading_context=cap.heading_context,
        notify=click.echo,
    )

    mode = "whole-input" if result.whole else "chunk-by-chunk"
    click.echo(
        f"saved {len(result.written)} pair(s) "
        f"({mode}; {result.before_chars} -> {result.after_chars} chars total)"
    )
    click.echo(_corpus_confirmation(store))
    _remind_open_sessions(ctx)


if __name__ == "__main__":
    main()
