"""Phase-1 capture mechanism: paragraph diffing + pair-record I/O, as a library.

The `ai-style log` CLI is a thin wrapper keeping the path policy and the
interactive session-snapshot workflow. Everything here takes explicit paths and
strings, so a non-CLI producer can capture pairs without the logger's globals.
Records follow the shared `pairs.jsonl` contract
(`stylebot.pairs.validate_pair_record`); see `_plans/decisions.md` D14.
"""

from __future__ import annotations

import difflib
import json
import os
import shutil
import uuid
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM
from stylebot.jsonl import append_jsonl, read_jsonl, write_jsonl_atomic
from stylebot.lib import read_w_frontmatter_text, split_paragraphs
from stylebot.markdown import is_heading_line
from stylebot.pairs import build_pair_content
from stylebot.provenance import now_iso  # `ai-style log` binds capture.now_iso


def new_capture_id() -> str:
    """Short id grouping all chunk-pairs from one capture invocation."""
    return uuid.uuid4().hex[:8]


def split_frontmatter_meta(text: str) -> tuple[dict, str]:
    """Best-effort split into (parsed YAML frontmatter, body).

    ``({}, text)`` for non-.qmd or plain prose. The `_meta` suffix is
    load-bearing: `stylebot.infer.split_frontmatter` splits the same text into
    ``(raw frontmatter string, body)``, and the two are not interchangeable.
    """
    try:
        meta, body = read_w_frontmatter_text(text)
        return (meta or {}), body
    except Exception:
        return {}, text


# ---- Paragraph-level diffing ----


def _is_heading_paragraph(para: str) -> bool:
    """True if `para` is a standalone ATX header paragraph.

    Only the first line is inspected. A header glued to body text in the same
    paragraph (no blank line between, e.g. `## H\\ntext`) is NOT a heading here:
    it is already inside that chunk, so adding it as context would duplicate it.
    """
    if not para:
        return False
    first_line = para.splitlines()[0] if "\n" in para else para
    return is_heading_line(first_line)


def _nearest_heading(paras: list[str], idx: int) -> str:
    """Return the nearest preceding heading paragraph's text (immediate depth).

    Scans backwards from `idx - 1`, returning the heading stripped, or "" when
    the region is in the preamble before any heading (valid, not an error).
    Depth is IMMEDIATE: the single nearest heading, never a breadcrumb.
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

    Bridging rather than a fixed-width window, because untouched prose is text
    the author did not get to, not text they endorsed: a capture's before-side is
    slop-bearing, so padding pairs out with untouched neighbours would teach the
    styler to leave slop alone. See `_plans/v2-longer-rewrites.md`.
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

    Split both texts on blank lines (the same chunk shape Phase 2 samples, so
    real and synthetic pairs are mixable), run SequenceMatcher on the paragraph
    lists, and emit a pair per contiguous changed region. `equal` opcodes are
    skipped, and so are pure `insert` / `delete` ones — nothing to learn from.

    `coalesce_gap_paras` / `coalesce_gap_chars` extend coalescing across *short
    unchanged runs*, so an edit spanning several paragraphs with a left-alone
    paragraph in the middle stays ONE pair; both default to 0. A merged region
    exceeding `max_chars` falls back to its uncoalesced parts rather than being
    dropped. Untouched text is never appended to a pair's *ends*, only bridged
    between two changed regions (see `_coalesce_regions`).

    With `heading_context`, each `context` is the nearest preceding heading among
    the AFTER paragraphs; preamble regions get `""`.
    """
    before_paras = split_paragraphs(before)
    after_paras = split_paragraphs(after)

    matcher = difflib.SequenceMatcher(None, before_paras, after_paras, autojunk=False)
    raw = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]
    regions = _coalesce_regions(
        raw, after_paras, gap_paras=coalesce_gap_paras, gap_chars=coalesce_gap_chars
    )

    def _emit(span: tuple[int, int, int, int]) -> tuple[str, str, str] | None:
        i1, i2, j1, j2 = span
        before_chunk = "\n\n".join(before_paras[i1:i2])
        after_chunk = "\n\n".join(after_paras[j1:j2])
        if not before_chunk.strip() or not after_chunk.strip():
            # Pure insert or pure delete — no transformation to learn from.
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
        # regions inside this span, so a long merge degrades to the finer cut
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

    A non-empty `context` is prepended *verbatim and identically* to both message
    bodies. `before_chars`/`after_chars` still count the BODY only, so adding
    context does not change the recorded transform-region size.
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
        write_jsonl_atomic(pairs_path, kept)
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

    Returns the records written, one per chunk-pair. Raises ValueError if nothing
    changed worth capturing. `whole` mode spans many headings, so it never
    carries context regardless of `heading_context`; the gap/`max_chars` args
    pass through to `diff_chunks`.

    The full before/after texts are also archived beside the corpus
    (`<pairs_dir>/captures/<capture_id>/`). The chunk records are a lossy VIEW —
    changed regions only, no document offsets — so the archive is what lets the
    corpus be re-cut at other chunk scales later without re-capturing.
    """
    before_meta, before_body = split_frontmatter_meta(before_text)
    after_meta, after_body = split_frontmatter_meta(after_text)

    if whole:
        # A whole-file pair spans many headings, so no single one applies.
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
        append_jsonl(pairs_path, record)
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
# Plain files rather than JSON-escaped strings, so they can be read, `diff`ed and
# hand-corrected when a capture comes out wrong (`recut` picks the correction up
# next run). The before->after diff IS the training signal; it must stay legible.
#
# `.md` and not `.qmd` deliberately. Every tool in the blog repo keys on `*.qmd`
# — the pre-commit hook would reformat these files (destroying the verbatim
# record it is their whole job to keep), search would index the captured slop as
# if the author wrote it, and `iter_targets` would sample it as a synthesis
# target. `.md` is invisible to all of them.

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

    Three separate files make a partial write possible, so build in a sibling
    temp directory and `os.replace` it into place: an archive is either wholly
    there or absent, never half-populated (which `iter_archives` would skip).
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

    None (rather than raising) for an unreadable or incomplete archive, matching
    `iter_archives`' skip-and-continue contract.
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

    Sorted by `snapped_at` so a re-cut is deterministic, and flattened to the
    shape `recut_archives` consumes.

    Directories only. Support for the superseded `captures/<capture_id>.json`
    blobs was dropped in 0.3, but they are REFUSED rather than skipped: a silent
    skip would look exactly like "that capture had no changed regions", so the
    caller would never learn an archive had dropped out of the corpus. Migrate
    with `scripts/migrate_capture_archives.py`, which still reads the old form.
    """
    cap_dir = pairs_path.parent / "captures"
    if not cap_dir.is_dir():
        return
    archives = []
    legacy: list[Path] = []
    for path in sorted(cap_dir.iterdir()):
        if path.is_dir() and not path.name.endswith(".tmp"):
            arch = read_archive(path)
            if arch is not None:
                archives.append(arch)
        elif path.suffix == ".json":
            legacy.append(path)
    if legacy:
        raise ValueError(
            f"{len(legacy)} legacy JSON capture archive(s) in {cap_dir} "
            f"(e.g. {legacy[0].name}). Support was removed in stylebot 0.3; "
            "migrate them with:\n"
            f"  uv run python scripts/migrate_capture_archives.py {pairs_path.parent} --apply"
        )
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

    Emitted records carry `meta.recut` (the gap policy) and reuse the archive's
    own `capture_id`, making a re-cut IDEMPOTENT: re-running with the same policy
    replaces the previous re-cut rather than stacking duplicates. Pairs from the
    original uncoalesced cut are dropped too, being a strictly lossier view of
    the same edit; captures with no archive are left untouched.

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
        write_jsonl_atomic(pairs_path, kept + records)
    return records
