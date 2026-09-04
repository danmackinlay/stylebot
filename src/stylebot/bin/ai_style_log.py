"""
ai-style log: capture (slop -> Dan rewrite) training pairs for the fine-tune.

Phase 1 of `_plans/OVERVIEW.md`: every time Dan rewrites an AI draft, keep the
(before, after). Deliberately manual — no automation, no git hook, no surprise
side effects. One rewrite of one file per session:

    uv run ai-style log open <file>    # record the current text as "before"
    # ... user edits the file by hand, across one or more sittings ...
    uv run ai-style log save <file>    # write pair(s) for what changed, close

`save` diffs at paragraph granularity and emits one pair per changed region.
Pairs land in the corpus in the chat-completion contract Phase 2 also writes and
Phase 3 reads.

This module is a **click shell**: it declares flags, resolves the store, calls
`stylebot.session`, and prints. `stylebot.capture` does the diffing and the pair
records; `stylebot.session` owns snapshot persistence, symlink canonicalisation and
`save`'s disposition rule (D35). The user-facing workflow is
`_training_pairs/README.md`.

Errors follow the package idiom (D47): `ConfigError` for wrong flags (exit 2),
`session.SessionError` — a `RunError` — for a world that is not ready (exit 1),
translated by `ErrorTranslatingGroup` rather than per-command `except` clauses.
What stays here is the `click.UsageError`s for flag *combinations*, which only
click can see.
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
    opt,
)
from stylebot.session import SessionStore, SnapshotEntry, humanize_age


@dataclass(frozen=True)
class CaptureOpts:
    """The flags `save` and `pair` share — what a captured pair looks like.

    Both commands land (before, after) as draws (D84) and differ only in
    where the two texts come from: `save` diffs a file against its session
    snapshot, `pair` takes two paths or stdin. *What the resulting record looks
    like* is the same question for both, and these three answer it — so they are
    declared once here rather than twice, which is how the help text for
    `--heading-context` came to be a six-line block maintained in two places.

    `--replace` is deliberately NOT here despite appearing on both: it means
    "supersede prior pairs for this source" in `pair` and "supersede across
    sessions, overriding the smart default" in `save`. One spelling, two
    behaviours — sharing the declaration would imply a symmetry there isn't.
    """

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


# Root for all logged state, resolved through `stylebot.config` — the one home
# of the `--flag` > `$STYLEBOT_DATA_DIR` > cwd-relative ladder (D45). `ai-style
# log` passes no flag: it is the command where a mistake is cheap, so it takes
# the env/default. Resolved lazily, per command: an import-time constant freezes
# whatever the environment looked like when the CLI entry point imported this
# module, which is how `log` once wrote to a cwd-relative `_training_pairs/`.
def _store() -> SessionStore:
    """The session/corpus store for this invocation."""
    return SessionStore.resolve()


def _warn_if_eval_post(store: SessionStore, source: Path) -> None:
    """Say so — loudly, once — when a capture lands on a frozen-eval post.

    Deliberately a NOTICE, not a refusal, and the asymmetry with `synth` is the
    point: a real captured rewrite of an eval post is Dan's own editing work and
    belongs in the corpus, where the splits contract keeps it out of every
    training and fitting stage. What must not happen silently is that the eval
    stratum shrinks — every capture here is prose the styler will never see and
    the frozen eval will score, so the operator should know they just spent one.
    """
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
    """click callback wrapping `session.canonical_source`.

    A symlinked FILE is keyed by the path it points at, so the blog's `_focus/`
    harness cannot open a second, parallel session under a scratch tree that
    gets wiped. See the library function for the full reasoning.
    """
    return session.canonical_source(value)


def _corpus_confirmation(store: SessionStore) -> str:
    """One-line 'it landed where training reads' confirmation for write commands.

    Resolves the data-dir to an ABSOLUTE path (a relative default like
    ``_training_pairs/`` otherwise hides *where* pairs went) and reports the
    running corpus size plus heading-context coverage, so each save visibly
    accumulates in the corpus you expect rather than a stray cwd dir.
    """
    summary = session.corpus_summary(store)
    return (
        f"corpus now: {summary.pairs} pair(s) ({summary.with_context} with heading context) "
        f"-> {summary.path.resolve()}"
    )


def _remind_open_sessions(ctx: click.Context) -> None:
    """Print a short reminder of currently-open sessions unless --quiet.

    Called at the end of every command except `list` (which already
    prints the full state). The reminder makes stale sessions visible
    so they don't accumulate silently.
    """
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


@click.group(
    cls=ErrorTranslatingGroup, context_settings={"help_option_names": ["-h", "--help"]}
)
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
    """Open a rewrite session on FILE: record its current content as the 'before' state.

    Run this *before* you start rewriting an AI-generated draft. The
    session is sticky — `open` refuses to overwrite an existing one so
    the original (worst) state is preserved across editing sittings.
    A later `save` diffs the current contents against this recorded state.
    """
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
    """Reopen FILE's last closed session, restoring its original 'before' state.

    For the do-over case: you saved, then spotted a typo in the rewrite. Fix
    the file, `reopen`, then `save` again — the re-save diffs against the same
    base and replaces the previous pairs rather than logging a near-duplicate.
    `save --redo` does both steps in one.

    Also recovers an accidental `drop`.
    """
    snapshot = session.reopen_session(_store(), file)
    # Surface the age: resurrecting a weeks-old base against a much-changed
    # file produces a huge, unrepresentative diff, and that should be visible.
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

    By default emits one pair per changed paragraph region (use --whole for
    one whole-file pair). Nothing is rewritten: a prior pair is *superseded*
    and a view drops it.

    Disposition of prior pairs for this source is **smart by default**: pairs
    from the *same* session (matching `meta.snapped_at`) are superseded, since
    those are interim `--keep-open` saves that should not stack, while pairs
    from earlier sessions are kept as legitimate prior rewrites. Override with
    --replace (supersede cross-session pairs too) or --append (keep everything).
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
        # Pairs logged outside a session (`log pair`, `synth`) carry snapped_at=None;
        # they have no session timestamp to sort or age, so count them separately.
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
        # Relative age for closed sessions: an absolute stamp answers "which
        # session was this" for one still open, but for a closed one the live
        # question is whether the base is stale enough that a `--redo` would
        # diff against prose that has moved on.
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
    """One-shot pair capture from files or stdin (no session dance).

    Unlike `save`, `pair` defaults to APPEND. The typical use is logging
    one-off pairs from clipboard or scripts, where you usually want to add
    to the corpus rather than overwrite. Pass --replace (with --source) to
    supersede prior pairs for that source.
    """
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
