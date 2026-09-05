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
from collections.abc import Sequence
from pathlib import Path

from stylebot import config, draws
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.errors import RunError
from stylebot.lib import read_w_frontmatter_text, split_paragraphs
from stylebot.markdown import is_heading_line
from stylebot.pairs import PAIR_SCHEMA_VERSION, build_pair_content
from stylebot.provenance import (  # `ai-style log` binds capture.now_iso
    now_iso,
    pair_provenance,
    policy_digest,
    repo_provenance,
)


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


def _automation_of(meta: dict | None) -> int | None:
    """The frontmatter `automation` level, or None if absent/not an int.

    The one blog-shaped key stylebot already names generically — it is
    `lib.is_human_authored`'s default field, and the pair record needs to say
    what the selector was looking at, because the post is editable and the
    corpus is frozen. A caller selecting on some other field gets None here
    rather than a number that means something else.
    """
    value = (meta or {}).get("automation")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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
    styler to leave slop alone (D81).
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

    Split both texts on blank lines (Phase 2's chunk shape, so real and
    synthetic pairs are mixable), run SequenceMatcher on the paragraph lists,
    and emit a pair per contiguous changed region; `equal`, `insert` and
    `delete` opcodes are skipped, having nothing to learn from.

    `coalesce_gap_paras` / `coalesce_gap_chars` extend coalescing across *short
    unchanged runs*, so an edit with a left-alone paragraph in the middle stays
    ONE pair; both default to 0, a merged region over `max_chars` falls back to
    its parts, and untouched text is only bridged between two changed regions
    (`_coalesce_regions`). With `heading_context`, each `context` is the nearest
    preceding heading among the AFTER paragraphs; preamble regions get `""`.
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
    provenance: dict | None = None,
    captured_at: str | None = None,
) -> dict:
    """Build a chat-completion JSONL record from pre-stripped body strings.

    A non-empty `context` is prepended *verbatim and identically* to both message
    bodies. `before_chars`/`after_chars` still count the BODY only, so adding
    context does not change the recorded transform-region size.

    `provenance` is the caller's `provenance.pair_provenance(...)` dict — which
    trees and which cut policy produced the record. Omitted rather than
    null-filled when the caller has none, so an unstamped record stays
    distinguishable from one whose producer could not answer.
    """
    meta: dict = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "source": source,
        "captured_at": captured_at or now_iso(),
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

    if provenance:
        meta["provenance"] = provenance

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
    blog_root: Path | str | None = None,
    supersedes: Sequence[str] = (),
) -> list[dict]:
    """Strip frontmatter, diff, archive the texts and land one draw per chunk.

    Returns the records written, one per chunk-pair; raises if nothing changed
    worth capturing. `whole` mode spans many headings, so it carries no
    context; the gap/`max_chars` args pass through to `diff_chunks`. The full
    texts are archived beside the corpus (`<data_dir>/captures/<capture_id>/`)
    and each record becomes an accepted draw under `<data_dir>/draws/` naming
    the `capture_id`s it `supersedes` (D84, `_land`). Every record carries
    `meta.provenance` (D36); a capture's "policy" is its cut knobs, so
    `policy_digest`s compare within a producer and not across one.
    """
    before_meta, before_body = split_frontmatter_meta(before_text)
    after_meta, after_body = split_frontmatter_meta(after_text)
    knobs = {
        "whole": whole,
        "heading_context": heading_context,
        "gap_paras": coalesce_gap_paras,
        "gap_chars": coalesce_gap_chars,
        "max_chars": max_chars,
    }
    provenance = pair_provenance(
        repo_provenance(blog_root if blog_root is not None else pairs_path.parent),
        policy_digest=policy_digest({"producer": "capture", **knobs}),
        source_automation=_automation_of(after_meta) or _automation_of(before_meta),
    )
    chunk_pairs = _chunk_pairs(before_body, after_body, **knobs)

    cap_id = new_capture_id()
    captured_at = now_iso()
    write_archive(
        pairs_path.parent / "captures" / cap_id,
        capture_id=cap_id,
        source=source,
        snapped_at=snapped_at,
        whole=whole,
        before_text=before_text,
        after_text=after_text,
        captured_at=captured_at,
    )
    records = [
        build_pair_record(
            before_body=b,
            after_body=a,
            source=source,
            snapped_at=snapped_at,
            before_frontmatter=before_meta if i == 0 else None,
            after_frontmatter=after_meta if i == 0 else None,
            capture_id=cap_id,
            chunk_index=i,
            chunk_total=len(chunk_pairs),
            context=context,
            extra_meta=extra_meta,
            provenance=provenance,
            captured_at=captured_at,
        )
        for i, (b, a, context) in enumerate(chunk_pairs)
    ]
    spans = [_span_of(before_text, b) for b, _a, _c in chunk_pairs]
    return _land(pairs_path, records, spans, supersedes=tuple(supersedes) or None)


def _chunk_pairs(
    before_body: str, after_body: str, *, whole: bool, heading_context: bool, **cut
) -> list[tuple[str, str, str]]:
    """The (before, after, context) triples of one capture: the whole body as
    one pair (no single heading applies), else `diff_chunks` under `cut`."""
    if whole:
        return [(before_body, after_body, "")]
    triples = diff_chunks(
        before_body,
        after_body,
        heading_context=heading_context,
        coalesce_gap_paras=cut["gap_paras"],
        coalesce_gap_chars=cut["gap_chars"],
        max_chars=cut["max_chars"],
    )
    if not triples:
        raise RunError(
            "no changed text chunks to capture "
            "(only whitespace, frontmatter, pure inserts, or pure deletes changed)"
        )
    return triples


def _span_of(text: str, chunk: str) -> tuple[int, int] | None:
    """Where `chunk` sits in `text` (the archived `before.md`), when it does."""
    start = text.find(chunk)
    return (start, start + len(chunk)) if start >= 0 else None


def _land(
    pairs_path: Path,
    records: list[dict],
    spans: Sequence[tuple[int, int] | None],
    *,
    supersedes: tuple[str, ...] | None,
) -> list[dict]:
    """Land a capture's records as draws (D84): `draw_from_pair`-shaped, no
    blog sha (the tree is dirty by definition), `span` into `before.md`.

    `draws/` is the only landing site (W4 step 5). An absent one is created,
    not refused: `ai-style log` has a default data dir, so a fresh corpus is
    an ordinary first save. `draws.append` would otherwise write a *file*.
    """
    draws_dir = config.draws_path(pairs_path.parent)
    draws_dir.mkdir(parents=True, exist_ok=True)
    for record, span in zip(records, spans, strict=True):
        draw = draws.draw_from_pair(record)
        draw.blog_dirty, draw.span, draw.supersedes = True, span, supersedes
        draws.append(draws_dir, draw)
    return records


# ---- The capture archive ----
#
#     captures/<capture_id>/{before.md, after.md, meta.json}
#
# A DIRECTORY of ordinary files, not a JSON blob, so a capture can be read and
# `diff`ed when it comes out wrong: the before->after diff IS the training
# signal and must stay legible. `.md` and not `.qmd`, because every tool in the blog repo
# keys on `*.qmd` — the pre-commit hook would reformat the verbatim record,
# search would index captured slop as the author's, and `iter_targets` would
# sample it as a target.

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
    captured_at: str | None = None,
) -> Path:
    """Write one capture archive directory, atomically; `captured_at` is the
    save time, so the order of one session's saves is checkable (W4 step 4).

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
                "captured_at": captured_at,
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


# ---- Reading the archive ----


def iter_archives(pairs_path: Path):
    """Yield each capture archive dict beside `pairs_path`, oldest first.

    Sorted by `snapped_at` so a walk is deterministic, in `read_archive`'s
    flat shape.

    Directories only. Support for the superseded `captures/<capture_id>.json`
    blobs was dropped in 0.3, but they are REFUSED rather than skipped: a silent
    skip would look exactly like "that capture had no changed regions", so the
    caller would never learn an archive had dropped out of the corpus. Their
    migration is spent and gone (D38), so the message names no command.
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
        raise RunError(
            f"{len(legacy)} legacy JSON capture archive(s) in {cap_dir} "
            f"(e.g. {legacy[0].name}). Support was removed in stylebot 0.3 and its "
            "migration retired; lay each out by hand as "
            "captures/<capture_id>/{before.md, after.md, meta.json}, or move it aside."
        )
    archives.sort(key=lambda d: (d.get("snapped_at") or "", d.get("capture_id") or ""))
    yield from archives
