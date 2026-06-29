"""
ai-style-log: capture (slop -> Dan rewrite) training pairs for the future
ai-style fine-tune.

Phase 1 of the plan at `_plans/ai-style-fine-tune.md`. This module is a
manual logger: every time Dan rewrites an AI draft, we want to capture
the (before, after) so it can later go into the fine-tune corpus. It is
deliberately invoked by hand — no automation, no git hook, no surprise
side effects.

The canonical use is the open/save cycle. Each rewrite of a file is a
**session**: opened explicitly, saved when finished, optionally banked
mid-stream and superseded later.

    uv run ai-style-log open <file>         # begin a rewrite session
    # ... user edits the file by hand, across one or more sittings ...
    uv run ai-style-log save <file>         # write pair(s) for what changed, close session

Every command (except `list`) ends by printing a short reminder of
currently-open sessions to stderr, with their relative age, so stale
sessions become visible. Pass `-q` / `--quiet` at the group level to
suppress the reminder for one invocation:

    uv run ai-style-log -q save <file>      # no reminder appended

The session is **sticky** — `open` refuses to overwrite an existing one,
so the original ("worst") state of the file survives across editing
sittings. Just open once at the start of a rewrite project and save when
you are happy. No need to be in one sitting.

------------------------------------------------------------------------
Chunk-level save (the default)
------------------------------------------------------------------------

`save` does NOT emit a single whole-file pair by default. It diffs the
session's recorded "before" state against the current file at paragraph
granularity and emits one pair *per changed region*, skipping untouched
paragraphs entirely.

Why: a real rewrite typically touches a few paragraphs out of many. A
whole-file pair would force the model to learn "preserve everything
except these spans, which get rewritten" — that is not the lesson we
want. Per-chunk pairs isolate the actual transformation. They also
match the chunk size Phase 2 will use for synthetic pairs, so training
data from both sources is shape-compatible.

The diff is paragraph-level (split on blank lines) + difflib
SequenceMatcher opcodes. **Contiguous changed paragraphs coalesce into
one pair**, regardless of N → M paragraph counts on either side — the
`replace` opcode pairs spans, not paragraphs. The pair only breaks
when an *untouched* paragraph sits between two changed regions.

Examples:

    rewrote whole document       -> 1 pair (entire body -> entire body)
    rewrote paras 2..4 of 5      -> 1 pair (paras 2+3+4 -> the rewrite)
    rewrote paras 2 and 4 of 5   -> 2 pairs (gap at para 3 breaks them)
    rewrote 3 paras into 2       -> 1 pair (3 joined -> 2 joined)
    rewrote 1 para into 5        -> 1 pair (1 -> 5 joined)
    moved a paragraph            -> 0 pairs (pure delete + pure insert)

Pure inserts (paragraphs Dan added with no source) and pure deletes
(slop removed entirely) are skipped — no transformation to learn.

Use `--whole` when the diff would be actively misleading: heavy
structural rearrangement, paragraphs moved around, or sentence-level
edits scattered across every paragraph (where the chunk algorithm
would emit one giant `replace` covering most of the file anyway).

------------------------------------------------------------------------
Worst -> best signal: smart default disposition of prior pairs
------------------------------------------------------------------------

A real rewrite often happens across multiple sittings, sometimes spread
across days, sometimes with `--keep-open` to bank interim progress
mid-session. The training corpus has different requirements for these
two situations:

- **Same-session interim saves** (typically from `--keep-open`): the
  *latest* interim save should win; earlier ones are scaffolding and
  should not pollute training data.
- **Cross-session saves** (you finished a session, did unrelated work,
  later opened a fresh session): both sets of pairs are legitimate
  training data and should both survive in the corpus.

`save` distinguishes these automatically by keying on each pair's
`meta.snapped_at` (the timestamp of the session that produced it,
recorded when `open` ran). On every save:

- Prior pairs whose `meta.snapped_at` matches the current session's
  snapshot timestamp are **replaced** (same-session interim).
- Prior pairs whose `meta.snapped_at` differs are **kept** (cross-session).

You never need a flag in the common cases. Just `open`, edit, `save`,
optionally `--keep-open` for banking. The tool figures out the
disposition.

Explicit override flags for the rare cases:

    --replace        force-drop every prior pair for this source,
                     including cross-session pairs. Use when you
                     genuinely want to abandon earlier sessions.
    --append         force-keep every prior pair for this source,
                     including interim pairs from the current session.
                     Use only if you want multiple interim states
                     preserved (very rare).
    --keep-open      orthogonal to the above: leaves the snapshot in
                     place after saving for further `save` calls
                     against the same starting state.

Typical interim-progress loop (no flags beyond --keep-open needed):

    open foo.qmd                                   # session 1, T0 recorded
    save --keep-open foo.qmd                       # interim pairs from session 1
    # ... more editing ...
    save --keep-open foo.qmd                       # smart: replaces session 1's interim, keeps nothing
    # ... finished ...
    save foo.qmd                                   # final, replaces interim, closes session 1

Multiple de-slop sessions on the same file, separated by unrelated
substantive edits — no flag needed; smart default keeps both:

    open foo.qmd                                   # session 1
    # ... de-slop, T0 -> T1 ...
    save foo.qmd                                   # session 1 pairs written, session closed
    # ... unrelated edits OUTSIDE any session, T1 -> T2 ...
    open foo.qmd                                   # session 2 (different snapped_at)
    # ... de-slop, T2 -> T3 ...
    save foo.qmd                                   # session 2 pairs ADDED; session 1 pairs kept
                                                   # output: "kept 1 pair(s) from 1 earlier session(s)"

Critical: close session 1 before doing unrelated edits. If you used
`save --keep-open` between sessions, session 2's diff would be against
T0 (session 1's snapshot) — folding structural edits into the pair.
The clean close in step 3 is what protects training data from being
polluted with non-style work.

If you ever force-replaced when you didn't mean to (and the prior
session's pairs were not yet backed up), the data is gone.
Conservative practice: back up `$STYLEBOT_DATA_DIR/pairs.jsonl` after each
substantial save. The corpus is gitignored in the public code repo, so its
recovery backstop is the out-of-band private backup, not this repo's git
history (see `_training_pairs/README.md`).

`tidy [source]` is the retroactive cleanup if you used `--append`
recklessly and want to keep only the latest save per source.

------------------------------------------------------------------------
Recommended editing cadence
------------------------------------------------------------------------

The natural unit of capture is "this file, as a rewrite project" —
not "this editing sitting". A typical cycle:

1. `open` once at the start of the rewrite project (typically when
   the file is `automation: 2`). The session is sticky; this marks
   "begin work", not "start a sitting".
2. Edit at your own pace. One sitting, one week — doesn't matter.
3. `save` when you've reached a satisfying state. Default behaviour
   emits per-region pairs, replaces any prior pairs for this file,
   and closes the session.
4. To bank interim progress without closing the session, use
   `save --keep-open`. Subsequent saves replace the interim pairs
   by default.
5. Avoid `open` on files you're writing from scratch (no slop input
   to learn from), tiny mechanical fixes (no transformation signal),
   or per-paragraph (`save` already isolates changed paragraphs).
   Cap simultaneous open sessions at ~3–4 to keep the bookkeeping
   manageable — `list` shows everything pending.
6. Use `--tag` to label saves by provenance (`from-claude-chat`,
   `dan-voice-draft`, `heavy-edit`) for downstream filtering.

The longer-form discussion of cadence lives in
`_training_pairs/README.md`.

------------------------------------------------------------------------
One-shot entry (no session dance)
------------------------------------------------------------------------

For ad-hoc pairs (clipboard paste, ephemeral content, scripted pipelines)
use `pair` with explicit before/after:

    uv run ai-style-log pair --before before.txt --after after.txt
    uv run ai-style-log pair --before - --after after.txt   # stdin
    pbpaste | uv run ai-style-log pair --before - --after rewritten.txt \\
        --source clipboard --tag hand-typed

`pair` chunks by default (same paragraph diff as `save`); pass `--whole`
to keep the whole input as a single pair. `pair` defaults to append (not
replace) since the typical use is logging one-off pairs that should add
to the corpus, not bank-with-supersede; pass `--replace` to remove prior
pairs for a given `--source` first.

------------------------------------------------------------------------
State on disk
------------------------------------------------------------------------

    $STYLEBOT_DATA_DIR/snapshots/<rel-path>.json   # open sessions (gitignored)
    $STYLEBOT_DATA_DIR/pairs.jsonl                 # saved pairs (gitignored)

`STYLEBOT_DATA_DIR` defaults to `_training_pairs` relative to cwd. The whole
directory is gitignored: the corpus is private and lives outside the public
code repo, backed up out-of-band.

The on-disk directory for open sessions is still called `snapshots/`
because the file *is* a snapshot of the source's initial state — that
matches the file's role at the storage layer. The CLI verb is `open`
because that names the user's *action* (opening a session) rather than
the file format.

Pairs are written in Together / OpenAI chat-completion JSONL:

    {"messages": [{"role": "system", "content": STYLE_SYSTEM},
                  {"role": "user", "content": <slop chunk>},
                  {"role": "assistant", "content": <Dan-voice chunk>}],
     "meta": {"source": "post/foo.qmd",
              "captured_at": "...", "snapped_at": "...",
              "capture_id": "<short id>",      # groups chunks from one save
              "chunk_index": 2, "chunk_total": 5,
              "before_chars": ..., "after_chars": ...,
              "before_frontmatter": {...},     # only on chunk 0
              "after_frontmatter": {...},      # only on chunk 0
              "tags": [...]}}

Rewrites to `pairs.jsonl` (default `save`, `pair --replace`, `tidy`) go
through an atomic `.jsonl.tmp` + rename so an interrupted run cannot
corrupt the audit trail. Unparseable lines are preserved as opaque
`_raw` records during rewrites.

------------------------------------------------------------------------
What survives a rename or repo move
------------------------------------------------------------------------

`meta.source` is the relative path at save time. Renaming the source
file later does not invalidate the pair (the text content is what
matters for training), but `save`'s default replace and `tidy` use the
source string to identify "the same file" — so if you rename a file and
then re-save, the old pairs will not be replaced automatically. Use
`tidy <old-path>` to clean those up by hand, or hand-edit the JSONL.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import click

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.lib import read_w_frontmatter_text, split_paragraphs
from stylebot.pairs import build_pair_content

# Root for all logged state (the corpus). Resolved from $STYLEBOT_DATA_DIR if
# set, else cwd-relative `_training_pairs` (the historical default, preserved
# so the logger keeps working unchanged inside the prose working tree).
#
# Why configurable: the corpus is the project's valuable, *private* asset. The
# stylebot code repo is public, so the corpus must live outside it. The logger
# runs in the blog repo and writes there; downstream phases (synthesis,
# training) run in this repo and point STYLEBOT_DATA_DIR at the same corpus.
# Both default and override are gitignored. See `_training_pairs/README.md`.
TRAINING_PAIRS_DIR = Path(os.environ.get("STYLEBOT_DATA_DIR", "_training_pairs"))
SNAPSHOTS_DIR = TRAINING_PAIRS_DIR / "snapshots"
PAIRS_PATH = TRAINING_PAIRS_DIR / "pairs.jsonl"


def _now_iso() -> str:
    # Microsecond precision so sessions opened close together (e.g. via
    # scripts) get distinct snapped_at values. The smart-default matcher
    # in `save` keys on snapped_at to distinguish same-session interim
    # pairs from cross-session pairs; second-resolution collisions used
    # to make two adjacent `open` calls indistinguishable.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _capture_id() -> str:
    """Short id grouping all chunk-pairs from one capture invocation."""
    return uuid.uuid4().hex[:8]


def _snapshot_path(source: Path) -> Path:
    """Where the snapshot for `source` lives.

    Mirrors the source path under SNAPSHOTS_DIR, suffixed with .json. So
    `post/foo.qmd` -> `_training_pairs/snapshots/post/foo.qmd.json`.
    """
    return SNAPSHOTS_DIR / f"{source}.json"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Best-effort split; returns ({}, text) for non-.qmd or plain prose."""
    try:
        meta, body = read_w_frontmatter_text(text)
        return (meta or {}), body
    except Exception:
        return {}, text


# ---- Paragraph-level diffing ----


def _split_paragraphs(text: str) -> list[str]:
    """Blank-line paragraph split — delegates to the shared splitter.

    The canonical implementation lives in `stylebot.lib.split_paragraphs` so
    Phase 1 (this logger) and Phase 2 (`stylebot.synth`) chunk identically.
    """
    return split_paragraphs(text)


# A markdown ATX header line: up to 3 leading spaces, 1-6 hashes, then a space
# and the heading text. Matched against the FIRST line of a paragraph (post
# `_split_paragraphs`) so only standalone header paragraphs count as headings.
_ATX_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s")


def _is_heading_paragraph(para: str) -> bool:
    """True if `para` is a standalone ATX header paragraph.

    A "heading" is a paragraph whose first line is a markdown ATX header
    (`# ...` through `###### ...`). Only the first line is inspected; a header
    glued to body text in the same paragraph (no blank line between, e.g.
    `## H\\ntext`) is NOT treated as a heading here — the heading is already
    inside that chunk, so we must not double-add it as context.
    """
    if not para:
        return False
    first_line = para.splitlines()[0] if "\n" in para else para
    return bool(_ATX_HEADER_RE.match(first_line))


def _nearest_heading(paras: list[str], idx: int) -> str:
    """Return the nearest preceding heading paragraph's text (immediate depth).

    Scans backwards from `idx - 1` through `paras` for the most recent
    standalone header paragraph and returns its text, stripped. If none
    precedes `idx` (the region is in the preamble before any heading),
    returns "" — a heading-less region, which is valid.

    Depth is IMMEDIATE: only the single nearest heading, never a breadcrumb.
    """
    for k in range(idx - 1, -1, -1):
        if _is_heading_paragraph(paras[k]):
            return paras[k].strip()
    return ""


def diff_chunks(
    before: str, after: str, *, heading_context: bool = False
) -> list[tuple[str, str, str]]:
    """Return (before_chunk, after_chunk, context) triples per changed region.

    Algorithm: split both texts on blank lines, run SequenceMatcher on the
    paragraph lists, emit a pair for every `replace` opcode (a contiguous
    changed region). Skip `equal` opcodes (untouched paragraphs) and pure
    `insert` / `delete` opcodes (no transformation signal to learn from).

    Contiguous changed paragraphs end up coalesced into a single pair,
    which is desirable: rewriting two adjacent paragraphs is usually one
    intent, not two independent edits.

    When `heading_context` is True, each triple's `context` is the nearest
    preceding section heading among the AFTER paragraphs (the kept text),
    resolved from the changed region's after-start index `j1`. Preamble
    regions (no heading precedes them) get `context == ""`. When
    `heading_context` is False, `context` is always "" (legacy behaviour).
    """
    before_paras = _split_paragraphs(before)
    after_paras = _split_paragraphs(after)

    matcher = difflib.SequenceMatcher(
        None, before_paras, after_paras, autojunk=False
    )
    triples: list[tuple[str, str, str]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_chunk = "\n\n".join(before_paras[i1:i2])
        after_chunk = "\n\n".join(after_paras[j1:j2])
        if not before_chunk.strip() or not after_chunk.strip():
            # Pure insert (no before content) or pure delete (no after
            # content) — nothing to learn the transformation from.
            continue
        context = _nearest_heading(after_paras, j1) if heading_context else ""
        triples.append((before_chunk, after_chunk, context))
    return triples


# ---- Pair record I/O ----


def _build_pair_record(
    *,
    before_body: str,
    after_body: str,
    source: str | None,
    snapped_at: str | None,
    before_frontmatter: dict | None,
    after_frontmatter: dict | None,
    capture_id: str,
    chunk_index: int,
    chunk_total: int,
    context: str = "",
    extra_meta: dict | None = None,
) -> dict:
    """Build a chat-completion JSONL record from pre-stripped body strings.

    When `context` is non-empty (a section heading), it is prepended
    *verbatim and identically* to both message bodies via
    `stylebot.pairs.build_pair_content` and recorded under `meta.context`
    (with `meta.context_mode = "immediate"`). `before_chars`/`after_chars`
    keep counting the BODY only (excluding the prepended heading), so the
    transform-region size is unchanged by adding context.
    """
    meta: dict = {
        "source": source,
        "captured_at": _now_iso(),
        "capture_id": capture_id,
        "chunk_index": chunk_index,
        "chunk_total": chunk_total,
        "before_chars": len(before_body),
        "after_chars": len(after_body),
    }
    if snapped_at is not None:
        meta["snapped_at"] = snapped_at
    if before_frontmatter:
        meta["before_frontmatter"] = before_frontmatter
    if after_frontmatter:
        meta["after_frontmatter"] = after_frontmatter

    context = (context or "").strip()
    if context:
        meta["context"] = context
        meta["context_mode"] = "immediate"

    if extra_meta:
        meta.update(extra_meta)

    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": build_pair_content(context, before_body)},
            {"role": "assistant", "content": build_pair_content(context, after_body)},
        ],
        "meta": meta,
    }


def _append_pair(record: dict) -> None:
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_pairs() -> list[dict]:
    """Read pairs.jsonl into memory. Returns [] if file is missing."""
    if not PAIRS_PATH.exists():
        return []
    out: list[dict] = []
    with PAIRS_PATH.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # Preserve unparseable lines as raw text wrapped so the
                # rewrite path doesn't silently drop them.
                out.append({"_raw": line})
    return out


def _corpus_confirmation() -> str:
    """One-line 'it landed where training reads' confirmation for write commands.

    Resolves the data-dir to an ABSOLUTE path (a relative default like
    ``_training_pairs/`` otherwise hides *where* pairs went) and reports the
    running corpus size plus heading-context coverage, so each save visibly
    accumulates in the corpus you expect rather than a stray cwd dir.
    """
    pairs = _iter_pairs()
    with_ctx = sum(1 for r in pairs if (r.get("meta", {}).get("context") or "").strip())
    return f"corpus now: {len(pairs)} pair(s) ({with_ctx} with heading context) -> {PAIRS_PATH.resolve()}"


def _write_pairs_atomic(records: list[dict]) -> None:
    """Atomic rewrite of pairs.jsonl from the given record list."""
    PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PAIRS_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for rec in records:
            if "_raw" in rec and set(rec.keys()) == {"_raw"}:
                fp.write(rec["_raw"] + "\n")
            else:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(PAIRS_PATH)


def _remove_pairs_for_source(source: str) -> int:
    """Drop every record whose meta.source matches. Returns count removed."""
    records = _iter_pairs()
    kept: list[dict] = []
    removed = 0
    for rec in records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        if meta.get("source") == source:
            removed += 1
        else:
            kept.append(rec)
    if removed:
        _write_pairs_atomic(kept)
    return removed


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
    """Strip frontmatter, diff, and append one or more pair records.

    Returns the list of records written (each describes one chunk-pair).
    Raises ValueError if nothing changed worth capturing.

    When `heading_context` is True (the default), each per-region pair is
    prefixed with the nearest preceding section heading (from the after body)
    as fixed context. `--whole` mode spans many headings, so it never carries
    context regardless of this flag.
    """
    before_meta, before_body = _split_frontmatter(before_text)
    after_meta, after_body = _split_frontmatter(after_text)

    if whole:
        # A whole-file pair spans many headings — no single heading applies,
        # so it carries no context regardless of `heading_context`.
        chunk_pairs: list[tuple[str, str, str]] = [(before_body, after_body, "")]
    else:
        chunk_pairs = diff_chunks(
            before_body, after_body, heading_context=heading_context
        )

    if not chunk_pairs:
        raise ValueError(
            "no changed text chunks to capture "
            "(only whitespace, frontmatter, pure inserts, or pure deletes changed)"
        )

    cap_id = _capture_id()
    total = len(chunk_pairs)
    written: list[dict] = []
    for i, (b, a, context) in enumerate(chunk_pairs):
        record = _build_pair_record(
            before_body=b,
            after_body=a,
            source=source,
            snapped_at=snapped_at,
            before_frontmatter=before_meta if i == 0 else None,
            after_frontmatter=after_meta if i == 0 else None,
            capture_id=cap_id,
            chunk_index=i,
            chunk_total=total,
            context=context,
            extra_meta=extra_meta,
        )
        _append_pair(record)
        written.append(record)
    return written


# ---- CLI ----


def _humanize_age(iso_ts: str) -> str:
    """Render an ISO-8601 timestamp as a relative age string ("3h ago")."""
    try:
        ts = datetime.fromisoformat(iso_ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
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
    if not SNAPSHOTS_DIR.exists():
        return
    snaps = sorted(SNAPSHOTS_DIR.rglob("*.json"))
    if not snaps:
        return

    entries: list[tuple[str, str]] = []
    for snap in snaps:
        try:
            payload = json.loads(snap.read_text(encoding="utf-8"))
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
    click.echo("  use `ai-style-log -q ...` to suppress, `ai-style-log drop <file>` to abandon", err=True)


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
        click.echo(
            f"session already open for {file}; "
            "`drop` it, or `save` without --keep-open to close it",
            err=True,
        )
        sys.exit(1)
    snap.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(file),
        "snapped_at": _now_iso(),
        "content": _read_text(file),
    }
    snap.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    click.echo(f"opened session for {file} ({len(payload['content'])} chars) -> {snap}")
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
    "--heading-context/--no-heading-context",
    "heading_context",
    default=True,
    help="Prepend the nearest preceding section heading (from the after text) "
    "as fixed context to each changed chunk, identically on both sides, and "
    "record it in meta.context. On by default; --no-heading-context restores "
    "the old bare-body behaviour. Ignored for --whole (no single heading).",
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
        click.echo("--replace and --append are mutually exclusive", err=True)
        sys.exit(2)

    snap = _snapshot_path(file)
    if not snap.exists():
        click.echo(f"no open session for {file}; run `open` first", err=True)
        sys.exit(1)

    payload = json.loads(snap.read_text(encoding="utf-8"))
    before_text = payload["content"]
    after_text = _read_text(file)

    if before_text == after_text:
        click.echo(
            f"no changes since session opened on {file}; nothing to save",
            err=True,
        )
        sys.exit(1)

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
        _write_pairs_atomic(records_to_keep)

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
        click.echo(str(exc), err=True)
        sys.exit(1)

    if not keep_open:
        snap.unlink()

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
        sessions_ts = sorted(
            {rec.get("meta", {}).get("snapped_at") for rec in other_session if isinstance(rec, dict)}
        )
        oldest = sessions_ts[0] if sessions_ts else ""
        click.echo(
            f"  kept {len(other_session)} pair(s) from {len(sessions_ts)} earlier session(s) "
            f"[oldest opened {_humanize_age(oldest)}]"
        )
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
        click.echo(f"no open session for {file}", err=True)
        sys.exit(1)
    snap.unlink()
    click.echo(f"dropped session for {file}")
    _remind_open_sessions(ctx)


@main.command(name="list")
def list_state():
    """List open sessions, saved pairs (grouped by source), and totals."""
    snaps = sorted(SNAPSHOTS_DIR.rglob("*.json")) if SNAPSHOTS_DIR.exists() else []
    records = _iter_pairs()
    pair_count = len(records)

    # Group pairs by source for a quick summary
    by_source: dict[str, list[dict]] = {}
    for rec in records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        src = meta.get("source") or "(no source)"
        by_source.setdefault(src, []).append(rec)

    click.echo(f"{pair_count} pair(s) across {len(by_source)} source(s) in {PAIRS_PATH}")
    for src, recs in sorted(by_source.items()):
        capture_ids = {
            r.get("meta", {}).get("capture_id")
            for r in recs
            if isinstance(r, dict)
        }
        capture_ids.discard(None)
        click.echo(f"  {src}: {len(recs)} pair(s) across {len(capture_ids)} save(s)")

    click.echo(f"{len(snaps)} open session(s)")
    for snap in snaps:
        try:
            payload = json.loads(snap.read_text(encoding="utf-8"))
            click.echo(
                f"  {payload.get('source', snap)} "
                f"(opened {payload.get('snapped_at', '?')}, "
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
        _write_pairs_atomic(kept)
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
    "record it in meta.context. On by default; --no-heading-context restores "
    "the old bare-body behaviour. Ignored for --whole (no single heading).",
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
        click.echo("only one of --before/--after may be '-'", err=True)
        sys.exit(2)
    before_text = _read_source(before_spec)
    after_text = _read_source(after_spec)

    if before_text == after_text:
        click.echo("before and after are identical; refusing to log", err=True)
        sys.exit(1)

    if replace:
        if source is None:
            click.echo("--replace requires --source", err=True)
            sys.exit(2)
        removed = _remove_pairs_for_source(source)
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
        click.echo(str(exc), err=True)
        sys.exit(1)

    total_before = sum(rec["meta"]["before_chars"] for rec in written)
    total_after = sum(rec["meta"]["after_chars"] for rec in written)
    mode = "whole-input" if whole else "chunk-by-chunk"
    click.echo(
        f"saved {len(written)} pair(s) "
        f"({mode}; {total_before} -> {total_after} chars total)"
    )
    click.echo(_corpus_confirmation())
    _remind_open_sessions(ctx)


if __name__ == "__main__":
    main()
