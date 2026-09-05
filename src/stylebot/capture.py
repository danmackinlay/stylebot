"""Phase-1 capture mechanism: paragraph diffing + pair-record I/O, as a
library over explicit paths and strings (D14, D35). Records follow
`stylebot.pairs.validate_pair_record`."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from pathlib import Path

from stylebot import config, draws
from stylebot.errors import RunError
from stylebot.lib import DEFAULT_SOURCE, SourcePolicy, read_w_frontmatter_text, split_paragraphs
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
    """Best-effort split into (parsed YAML frontmatter, body); ``({}, text)``
    for plain prose. Not `infer.split_frontmatter`, which returns the raw string."""
    try:
        meta, body = read_w_frontmatter_text(text)
        return (meta or {}), body
    except Exception:
        return {}, text


# ---- Paragraph-level diffing ----


def _is_heading_paragraph(para: str) -> bool:
    """True if `para` is a standalone ATX header paragraph; a header glued to
    body text is already inside its chunk and is not one."""
    if not para:
        return False
    first_line = para.splitlines()[0] if "\n" in para else para
    return is_heading_line(first_line)


def _nearest_heading(paras: list[str], idx: int) -> str:
    """The nearest preceding heading paragraph's text, stripped — the single
    nearest, never a breadcrumb — or "" in the preamble."""
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
    """Merge changed regions `(i1, i2, j1, j2)` separated by an unchanged run
    within both tolerances (D81)."""
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

    Paragraph-split both texts, run SequenceMatcher, emit a pair per contiguous
    `replace` region. `coalesce_gap_paras` / `coalesce_gap_chars` bridge short
    unchanged runs (D81); a merged region over `max_chars` falls back to its
    parts. With `heading_context`, `context` is the nearest preceding heading
    among the after paragraphs, `""` in the preamble.
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
        # Oversized after coalescing: fall back to the finer cut inside this span.
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
    system: str,
    context: str = "",
    extra_meta: dict | None = None,
    provenance: dict | None = None,
    captured_at: str | None = None,
) -> dict:
    """Build a chat-completion JSONL record from pre-stripped body strings;
    `system` is `messages[0]`, the producer's policy prompt (D85); `context` is
    prepended to both (D14) while `before_chars`/`after_chars` count the body
    only. `provenance` is omitted, not null-filled, when absent."""
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
            {"role": "system", "content": system},
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
    source_policy: SourcePolicy = DEFAULT_SOURCE,
    system_prompt: str | None = None,
) -> list[dict]:
    """Strip frontmatter, diff, archive the texts and land one draw per chunk.

    Returns the records written; raises if nothing changed worth capturing.
    `whole` mode carries no context. The texts are archived under
    `<data_dir>/captures/<capture_id>/` and each record becomes an accepted
    draw naming the `capture_id`s it `supersedes` (D84), carrying
    `meta.provenance` (D36) read under `source_policy`, whose prompt is
    `messages[0]` unless `system_prompt` says otherwise.
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
        source_automation=source_policy.level_of(after_meta) or source_policy.level_of(before_meta),
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
            system=system_prompt or source_policy.system_prompt,
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
    blog sha (the tree is dirty by definition), `span` into `before.md`. An
    absent `draws/` is created: a fresh corpus is an ordinary first save."""
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
# A directory of ordinary files so a capture can be `diff`ed; `.md` and not
# `.qmd`, so no blog tool reformats, indexes or samples the verbatim record.

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
    """Write one capture archive directory atomically (a sibling temp dir,
    `os.replace`d into place); `captured_at` is the save time."""
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
    """Load one archive directory into the module's flat dict; None for an
    unreadable or incomplete archive."""
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
    """Yield each capture archive dict beside `pairs_path`, oldest first by
    `snapped_at`. A legacy `captures/<capture_id>.json` blob is refused, not
    skipped: a skip would read as "no changed regions" (D38)."""
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
