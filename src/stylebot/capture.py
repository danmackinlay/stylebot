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
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.jsonl import read_jsonl
from stylebot.lib import read_w_frontmatter_text, split_paragraphs
from stylebot.pairs import build_pair_content


def now_iso() -> str:
    """UTC now as ISO-8601 with microseconds.

    Microsecond precision so sessions opened close together (e.g. via scripts)
    get distinct ``snapped_at`` values — the smart-default matcher in the CLI's
    `save` keys on ``snapped_at``, and second-resolution collisions used to
    make two adjacent `open` calls indistinguishable.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_capture_id() -> str:
    """Short id grouping all chunk-pairs from one capture invocation."""
    return uuid.uuid4().hex[:8]


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Best-effort split; returns ({}, text) for non-.qmd or plain prose."""
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


def diff_chunks(
    before: str, after: str, *, heading_context: bool = False
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
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_pairs_atomic(pairs_path: Path, records: list[dict]) -> None:
    """Atomic rewrite of `pairs_path` from the given record list.

    Records that are pure ``{"_raw": line}`` wrappers (unparseable lines kept
    by `stylebot.jsonl.read_jsonl(keep_undecodable=True)`) are written back
    verbatim, so a rewrite never silently drops a corrupt line.
    """
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pairs_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for rec in records:
            if "_raw" in rec and set(rec.keys()) == {"_raw"}:
                fp.write(rec["_raw"] + "\n")
            else:
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(pairs_path)


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
) -> list[dict]:
    """Strip frontmatter, diff, and append one or more pair records.

    Returns the list of records written (each describes one chunk-pair).
    Raises ValueError if nothing changed worth capturing.

    When `heading_context` is True (the default), each per-region pair is
    prefixed with the nearest preceding section heading (from the after body)
    as fixed context. `whole` mode spans many headings, so it never carries
    context regardless of this flag.
    """
    before_meta, before_body = split_frontmatter(before_text)
    after_meta, after_body = split_frontmatter(after_text)

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

    cap_id = new_capture_id()
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
