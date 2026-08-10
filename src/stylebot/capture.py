"""Phase-1 capture mechanism: paragraph diffing + pair-record I/O, as a library.

The `ai-style log` CLI (`stylebot.bin.ai_style_log`) is a thin wrapper over
this module; it keeps the *path policy* (its ``$STYLEBOT_DATA_DIR``-derived
defaults) and the interactive session-snapshot workflow. Everything here takes
explicit paths and strings, so a non-CLI producer (an editor integration, a
batch import) can capture pairs without touching the logger's globals.

The captured record shape is the shared `pairs.jsonl` contract
(`stylebot.pairs.validate_pair_record`); heading context follows
`_plans/heading-context.md` via `stylebot.pairs.build_pair_content`.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import shutil
import uuid
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.jsonl import append_jsonl, read_jsonl, write_jsonl_atomic
from stylebot.lib import read_w_frontmatter_text, split_paragraphs
from stylebot.pairs import build_pair_content
from stylebot.provenance import now_iso  # `ai-style log` binds capture.now_iso

logger = logging.getLogger(__name__)


def new_capture_id() -> str:
    """Short id grouping all chunk-pairs from one capture invocation."""
    return uuid.uuid4().hex[:8]


def split_frontmatter_meta(text: str) -> tuple[dict, str]:
    """Best-effort split into (parsed YAML frontmatter, body).

    Returns ``({}, text)`` for non-.qmd or plain prose. The `_meta` suffix is
    load-bearing: `stylebot.infer.split_frontmatter` splits the same text into
    ``(raw frontmatter string, body)``, and the two are not interchangeable.
    """
    try:
        meta, body = read_w_frontmatter_text(text)
        return (meta or {}), body
    except Exception:
        return {}, text


# ---- Paragraph-level diffing ----

# A markdown ATX header line: up to 3 leading spaces, 1-6 hashes, then a space
# and the heading text. Matched against the FIRST line of a paragraph (post
# `split_paragraphs`) so only standalone header paragraphs count as headings.
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


def _coalesce_regions(
    regions: list[tuple[int, int, int, int]],
    after_paras: list[str],
    *,
    gap_paras: int,
    gap_chars: int,
) -> list[tuple[int, int, int, int]]:
    """Merge changed regions separated by only a short unchanged run.

    Each region is `(i1, i2, j1, j2)` — before-span and after-span. Two adjacent
    regions merge when the untouched paragraphs between them are within BOTH
    tolerances, absorbing that gap into the merged pair.

    Why this and not a fixed-width window: a document's untouched prose is text
    Dan did not get to, not text he endorsed. A capture's before-side is
    slop-bearing, so padding pairs out with their untouched neighbours would
    teach the styler to leave slop alone. Bridging only *short* gaps keeps the
    pair essentially all-changed (measured changed-fraction stays ~1.00 at
    2 paragraphs / 800 chars) while letting one multi-paragraph intent stay one
    pair. See `_plans/v2-longer-rewrites.md`.
    """
    if not regions or (gap_paras <= 0 and gap_chars <= 0):
        return regions
    out = [list(regions[0])]
    for i1, i2, j1, j2 in regions[1:]:
        prev_j2 = out[-1][3]
        gap = after_paras[prev_j2:j1]
        gap_len = sum(len(p) + 2 for p in gap)
        if len(gap) <= gap_paras and gap_len <= gap_chars:
            out[-1][1] = i2  # extend before-span across the gap
            out[-1][3] = j2  # extend after-span across the gap
        else:
            out.append([i1, i2, j1, j2])
    return [tuple(r) for r in out]  # type: ignore[misc]


def diff_chunks(
    before: str,
    after: str,
    *,
    heading_context: bool = False,
    coalesce_gap_paras: int = 0,
    coalesce_gap_chars: int = 0,
    max_chars: int | None = None,
) -> list[tuple[str, str, str]]:
    """Return (before_chunk, after_chunk, context) triples per changed region.

    Algorithm: split both texts on blank lines (`stylebot.lib.split_paragraphs`
    — the same chunk shape Phase 2 samples, so real and synthetic pairs are
    mixable), run SequenceMatcher on the paragraph lists, emit a pair for every
    `replace` opcode (a contiguous changed region). Skip `equal` opcodes
    (untouched paragraphs) and pure `insert` / `delete` opcodes (no
    transformation signal to learn from).

    Contiguous changed paragraphs end up coalesced into a single pair,
    which is desirable: rewriting two adjacent paragraphs is usually one
    intent, not two independent edits.

    `coalesce_gap_paras` / `coalesce_gap_chars` extend that coalescing across
    *short unchanged runs*, so an edit spanning several paragraphs with a
    left-alone paragraph in the middle stays ONE pair instead of fragmenting.
    Both default to 0 (legacy behaviour: only contiguous regions merge). A
    merged region exceeding `max_chars` falls back to its uncoalesced parts
    rather than being dropped. Untouched text is never appended to the *ends*
    of a pair — only bridged between two changed regions (see
    `_coalesce_regions` for why).

    When `heading_context` is True, each triple's `context` is the nearest
    preceding section heading among the AFTER paragraphs (the kept text),
    resolved from the changed region's after-start index `j1`. Preamble
    regions (no heading precedes them) get `context == ""`. When
    `heading_context` is False, `context` is always "" (legacy behaviour).
    """
    before_paras = split_paragraphs(before)
    after_paras = split_paragraphs(after)

    matcher = difflib.SequenceMatcher(
        None, before_paras, after_paras, autojunk=False
    )
    raw = [
        (i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]
    regions = _coalesce_regions(
        raw, after_paras, gap_paras=coalesce_gap_paras, gap_chars=coalesce_gap_chars
    )

    def _emit(span: tuple[int, int, int, int]) -> tuple[str, str, str] | None:
        i1, i2, j1, j2 = span
        before_chunk = "\n\n".join(before_paras[i1:i2])
        after_chunk = "\n\n".join(after_paras[j1:j2])
        if not before_chunk.strip() or not after_chunk.strip():
            # Pure insert (no before content) or pure delete (no after
            # content) — nothing to learn the transformation from.
            return None
        context = _nearest_heading(after_paras, j1) if heading_context else ""
        return (before_chunk, after_chunk, context)

    triples: list[tuple[str, str, str]] = []
    for span in regions:
        emitted = _emit(span)
        if emitted is not None and (
            max_chars is None or max(len(emitted[0]), len(emitted[1])) <= max_chars
        ):
            triples.append(emitted)
            continue
        # Oversized (or empty) after coalescing — fall back to the uncoalesced
        # regions inside this span, so a long merge degrades to the legacy cut
        # instead of silently dropping a real edit.
        for part in raw:
            if part[2] >= span[2] and part[3] <= span[3]:
                emitted = _emit(part)
                if emitted is not None:
                    triples.append(emitted)
    return triples


# ---- Pair record I/O ----


def build_pair_record(
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
        "captured_at": now_iso(),
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


def append_pair(pairs_path: Path, record: dict) -> None:
    """Append one record to `pairs_path` (parent dirs created)."""
    append_jsonl(pairs_path, record)


# The atomic rewrite is a JSONL-level operation, not a pair-level one, so it
# lives in `stylebot.jsonl`. The old name stays bound for callers that reach it
# through this module (`stylebot.bin.ai_style_log`).
write_pairs_atomic = write_jsonl_atomic


def remove_pairs_for_source(pairs_path: Path, source: str) -> int:
    """Drop every record whose meta.source matches. Returns count removed."""
    records = read_jsonl(pairs_path, keep_undecodable=True)
    kept: list[dict] = []
    removed = 0
    for rec in records:
        meta = rec.get("meta", {}) if isinstance(rec, dict) else {}
        if meta.get("source") == source:
            removed += 1
        else:
            kept.append(rec)
    if removed:
        write_pairs_atomic(pairs_path, kept)
    return removed


def capture_pairs(
    before_text: str,
    after_text: str,
    *,
    pairs_path: Path,
    source: str | None,
    snapped_at: str | None,
    whole: bool = False,
    extra_meta: dict | None = None,
    heading_context: bool = True,
    coalesce_gap_paras: int = 0,
    coalesce_gap_chars: int = 0,
    max_chars: int | None = None,
) -> list[dict]:
    """Strip frontmatter, diff, and append one or more pair records.

    Returns the list of records written (each describes one chunk-pair).
    Raises ValueError if nothing changed worth capturing.

    When `heading_context` is True (the default), each per-region pair is
    prefixed with the nearest preceding section heading (from the after body)
    as fixed context. `whole` mode spans many headings, so it never carries
    context regardless of this flag.

    `coalesce_gap_paras` / `coalesce_gap_chars` / `max_chars` pass through to
    `diff_chunks`, bridging short unchanged runs so one multi-paragraph edit
    stays one pair. Defaults are the legacy no-bridging behaviour.

    The full before/after texts are also archived beside the corpus
    (`<pairs_dir>/captures/<capture_id>.json`) — the chunk records are a
    lossy VIEW (changed regions only, no document offsets), and the archive
    is what lets the corpus be re-cut at other chunk scales later (longer
    multi-paragraph pairs, discourse context) without re-capturing. Captures
    saved before 2026-07-22 predate the archive and are chunk-scale forever.
    """
    before_meta, before_body = split_frontmatter_meta(before_text)
    after_meta, after_body = split_frontmatter_meta(after_text)

    if whole:
        # A whole-file pair spans many headings — no single heading applies,
        # so it carries no context regardless of `heading_context`.
        chunk_pairs: list[tuple[str, str, str]] = [(before_body, after_body, "")]
    else:
        chunk_pairs = diff_chunks(
            before_body,
            after_body,
            heading_context=heading_context,
            coalesce_gap_paras=coalesce_gap_paras,
            coalesce_gap_chars=coalesce_gap_chars,
            max_chars=max_chars,
        )

    if not chunk_pairs:
        raise ValueError(
            "no changed text chunks to capture "
            "(only whitespace, frontmatter, pure inserts, or pure deletes changed)"
        )

    cap_id = new_capture_id()
    write_archive(
        pairs_path.parent / "captures" / cap_id,
        capture_id=cap_id,
        source=source,
        snapped_at=snapped_at,
        whole=whole,
        before_text=before_text,
        after_text=after_text,
    )
    total = len(chunk_pairs)
    written: list[dict] = []
    for i, (b, a, context) in enumerate(chunk_pairs):
        record = build_pair_record(
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
        append_pair(pairs_path, record)
        written.append(record)
    return written


# ---- The capture archive ----
#
# An archive is a DIRECTORY of ordinary files, not a JSON blob:
#
#     captures/<capture_id>/before.md
#     captures/<capture_id>/after.md
#     captures/<capture_id>/meta.json   {capture_id, source, snapped_at, whole}
#
# The two documents are the point. They used to be JSON-escaped strings on one
# line, which made the archive unreadable, unopenable and undiffable — perverse,
# since the before->after diff IS the training signal. As plain files they can be
# read, `diff`ed, and hand-corrected when a capture comes out wrong; `recut`
# picks the correction up on its next run.
#
# `.md` and not `.qmd` deliberately. Every tool in the blog repo keys on
# `*.qmd` — the pre-commit hook would reformat these files (destroying the
# verbatim record it is their whole job to keep), `qmd` search would index the
# captured slop as if Dan wrote it, and `iter_targets` would sample it as a
# synthesis target. `.md` is invisible to all of them, and matches the existing
# `_drafts/` convention for markdown that is not a post.

ARCHIVE_BEFORE = "before.md"
ARCHIVE_AFTER = "after.md"
ARCHIVE_META = "meta.json"


def write_archive(
    archive_dir: Path,
    *,
    capture_id: str,
    source: str | None,
    snapped_at: str | None,
    whole: bool,
    before_text: str,
    after_text: str,
) -> Path:
    """Write one capture archive directory, atomically.

    Three files means a partial write is now possible where the old single
    `write_text` could only truncate, so build in a sibling temp directory and
    `os.replace` it into place — an archive is either wholly there or absent,
    never half-populated (which `iter_archives` would silently skip).
    """
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_dir.with_name(archive_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    (tmp / ARCHIVE_BEFORE).write_text(before_text, encoding="utf-8")
    (tmp / ARCHIVE_AFTER).write_text(after_text, encoding="utf-8")
    (tmp / ARCHIVE_META).write_text(
        json.dumps(
            {
                "capture_id": capture_id,
                "source": source,
                "snapped_at": snapped_at,
                "whole": whole,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    os.replace(tmp, archive_dir)
    return archive_dir


def read_archive(archive_dir: Path) -> dict | None:
    """Load one archive directory into the flat dict the rest of the module uses.

    Returns None (rather than raising) for an unreadable or incomplete archive,
    matching `iter_archives`' skip-and-continue contract.
    """
    try:
        meta = json.loads((archive_dir / ARCHIVE_META).read_text(encoding="utf-8"))
        return {
            **meta,
            "before_text": (archive_dir / ARCHIVE_BEFORE).read_text(encoding="utf-8"),
            "after_text": (archive_dir / ARCHIVE_AFTER).read_text(encoding="utf-8"),
        }
    except (OSError, json.JSONDecodeError, TypeError):
        return None


# ---- Re-cutting the archive ----


def iter_archives(pairs_path: Path):
    """Yield each capture archive dict beside `pairs_path`, oldest first.

    Sorted by `snapped_at` so a re-cut is deterministic. Yields the flat shape
    (`before_text`/`after_text` alongside the meta keys) that `recut_archives`
    consumes, whichever on-disk layout it came from.

    Legacy `captures/<capture_id>.json` blobs are still read, and warned about.
    The warning is the point: a silent skip here would look exactly like "that
    capture had no changed regions", and the caller would never learn that an
    unmigrated archive had dropped out of the corpus.
    """
    cap_dir = pairs_path.parent / "captures"
    if not cap_dir.is_dir():
        return
    archives = []
    for path in sorted(cap_dir.iterdir()):
        if path.is_dir() and not path.name.endswith(".tmp"):
            arch = read_archive(path)
            if arch is not None:
                archives.append(arch)
        elif path.suffix == ".json":
            logger.warning(
                "legacy JSON capture archive %s — migrate with "
                "scripts/migrate_capture_archives.py",
                path,
            )
            try:
                archives.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
    archives.sort(key=lambda d: (d.get("snapped_at") or "", d.get("capture_id") or ""))
    yield from archives


def recut_archives(
    pairs_path: Path,
    *,
    coalesce_gap_paras: int,
    coalesce_gap_chars: int,
    max_chars: int | None = None,
    heading_context: bool = True,
    min_chars: int = 0,
    dry_run: bool = False,
) -> list[dict]:
    """Re-derive real pairs from the capture archives at a new cut policy.

    The archive holds the full before/after texts, so the corpus can be re-cut
    without recapturing — that is the whole reason it exists. Emitted records
    carry `meta.recut` (the gap policy) and reuse the archive's own
    `capture_id`, so a re-cut is IDEMPOTENT: re-running it with the same policy
    replaces the previous re-cut for that capture rather than stacking
    duplicates.

    Pairs from the *original* (uncoalesced) cut of the same capture are also
    dropped, since they are a strictly lossier view of the same edit. Captures
    with no archive (pre-2026-07-22) are untouched at their native scale —
    there is nothing to re-cut them from.

    Returns the records written (or that *would* be written, under `dry_run`).
    """
    policy = {"gap_paras": coalesce_gap_paras, "gap_chars": coalesce_gap_chars}
    records: list[dict] = []
    recut_ids: set[str] = set()

    for arch in iter_archives(pairs_path):
        cap_id = arch.get("capture_id")
        if not cap_id or arch.get("whole"):
            continue  # whole-file captures are already unchunked
        _, before_body = split_frontmatter_meta(arch.get("before_text") or "")
        _, after_body = split_frontmatter_meta(arch.get("after_text") or "")
        triples = diff_chunks(
            before_body,
            after_body,
            heading_context=heading_context,
            coalesce_gap_paras=coalesce_gap_paras,
            coalesce_gap_chars=coalesce_gap_chars,
            max_chars=max_chars,
        )
        triples = [t for t in triples if len(t[1].strip()) >= min_chars]
        if not triples:
            continue
        recut_ids.add(cap_id)
        for i, (b, a, context) in enumerate(triples):
            records.append(
                build_pair_record(
                    before_body=b,
                    after_body=a,
                    source=arch.get("source"),
                    snapped_at=arch.get("snapped_at"),
                    before_frontmatter=None,
                    after_frontmatter=None,
                    capture_id=cap_id,
                    chunk_index=i,
                    chunk_total=len(triples),
                    context=context,
                    extra_meta={"recut": policy},
                )
            )

    if not dry_run and records:
        kept = [
            rec
            for rec in read_jsonl(pairs_path, keep_undecodable=True)
            if (rec.get("meta", {}) if isinstance(rec, dict) else {}).get("capture_id")
            not in recut_ids
        ]
        write_pairs_atomic(pairs_path, kept + records)
    return records
