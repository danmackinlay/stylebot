"""Phase-1 capture mechanism: paragraph diffing + pair-record I/O, as a
library over explicit paths and strings (D14, D35). Records follow
`stylebot.pairs.validate_pair_record`."""

from __future__ import annotations

import difflib
import json
import os
import shutil
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from stylebot import config, draws
from stylebot.errors import RunError
from stylebot.lib import DEFAULT_SOURCE, SourcePolicy, read_w_frontmatter_text, split_paragraphs
from stylebot.markdown import DivIsProse, is_heading_line, line_kinds
from stylebot.pairs import PAIR_SCHEMA_VERSION, build_pair_content, extract_slop, extract_target
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

#: What every protected paragraph becomes for the matcher (D100).
_PROTECTED = "\0protected"


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


def _protected_paragraphs(
    body: str, paras: list[str], div_is_prose: DivIsProse | None
) -> list[bool]:
    """Per paragraph of `split_paragraphs(body)`, whether every line of it is
    protected by the segmenter (D100); a heading stays content (D64)."""
    kinds = line_kinds(body, headings=False, div_is_prose=div_is_prose)
    out: list[bool] = []
    pos = 0
    for para in paras:
        start = body.find(para, pos)
        first = body.count("\n", 0, start)
        out.append(all(k is not None for k in kinds[first : first + para.count("\n") + 1]))
        pos = start + len(para)
    return out


def _prose_span(
    span: tuple[int, int, int, int], before_prot: list[bool], after_prot: list[bool]
) -> tuple[int, int, int, int] | None:
    """`span` with protected paragraphs trimmed from both ends of both sides;
    None when one remains inside (a block inserted or deleted amid edited
    prose is no style transform, D100)."""
    i1, i2, j1, j2 = span
    while i1 < i2 and before_prot[i1]:
        i1 += 1
    while i1 < i2 and before_prot[i2 - 1]:
        i2 -= 1
    while j1 < j2 and after_prot[j1]:
        j1 += 1
    while j1 < j2 and after_prot[j2 - 1]:
        j2 -= 1
    if any(before_prot[i1:i2]) or any(after_prot[j1:j2]):
        return None
    return (i1, i2, j1, j2)


def _coalesce_regions(
    regions: list[tuple[int, int, int, int]],
    after_paras: list[str],
    *,
    gap_paras: int,
    gap_chars: int,
    after_prot: list[bool] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Merge changed regions `(i1, i2, j1, j2)` separated by an unchanged run
    within both tolerances (D81), never across a protected paragraph (D100)."""
    if not regions or (gap_paras <= 0 and gap_chars <= 0):
        return regions
    out = [list(regions[0])]
    for i1, i2, j1, j2 in regions[1:]:
        prev_j2 = out[-1][3]
        gap = after_paras[prev_j2:j1]
        gap_len = sum(len(p) + 2 for p in gap)
        bridgeable = not any(after_prot[prev_j2:j1]) if after_prot else True
        if bridgeable and len(gap) <= gap_paras and gap_len <= gap_chars:
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
    div_is_prose: DivIsProse | None = None,
) -> list[tuple[str, str, str]]:
    """Return (before_chunk, after_chunk, context) triples per changed region.

    Paragraph-split both texts, run SequenceMatcher, emit a pair per contiguous
    `replace` region. A paragraph the segmenter protects (`div_is_prose` its
    div policy) never enters a pair: the matcher sees every one as the same
    token, so an edit inside a block is no change and a block aligns across
    prose edits either side of it (D100). `coalesce_gap_paras` /
    `coalesce_gap_chars` bridge short unchanged runs (D81); a merged region
    over `max_chars` falls back to its parts. With `heading_context`, `context`
    is the nearest preceding heading among the after paragraphs, `""` in the
    preamble.
    """
    before_paras = split_paragraphs(before)
    after_paras = split_paragraphs(after)
    before_prot = _protected_paragraphs(before, before_paras, div_is_prose)
    after_prot = _protected_paragraphs(after, after_paras, div_is_prose)
    masked = (
        [_PROTECTED if p else para for para, p in zip(before_paras, before_prot, strict=True)],
        [_PROTECTED if p else para for para, p in zip(after_paras, after_prot, strict=True)],
    )

    matcher = difflib.SequenceMatcher(None, *masked, autojunk=False)
    raw = [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]
    regions = _coalesce_regions(
        raw,
        after_paras,
        gap_paras=coalesce_gap_paras,
        gap_chars=coalesce_gap_chars,
        after_prot=after_prot,
    )

    def _emit(span: tuple[int, int, int, int]) -> tuple[str, str, str] | None:
        prose = _prose_span(span, before_prot, after_prot)
        if prose is None:
            return None
        i1, i2, j1, j2 = prose
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


def novel_after(before_body: str, after_body: str, *, threshold: float = 0.5) -> float | None:
    """The fraction of after-side sentences with no before-side sentence at
    `difflib` ratio >= `threshold` — what Dan added, not restyled (D89);
    None when a side has no sentence."""
    from difflib import SequenceMatcher

    from stylebot.features import sentences

    before, after = sentences(before_body), sentences(after_body)
    if not before or not after:
        return None
    novel = 0
    for a in after:
        matcher = SequenceMatcher(None, a, "")
        best = 0.0
        for b in before:
            matcher.set_seq2(b)
            if matcher.real_quick_ratio() >= threshold and matcher.quick_ratio() >= threshold:
                best = max(best, matcher.ratio())
                if best >= threshold:
                    break
        novel += best < threshold
    return round(novel / len(after), 3)


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
    novelty = novel_after(before_body, after_body)
    if novelty is not None:
        meta["novel_after"] = novelty

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
    keep: Callable[[str, str], bool] | None = None,
    div_is_prose: DivIsProse | None = None,
) -> list[dict]:
    """Strip frontmatter, diff, archive the texts and land one draw per chunk.

    Returns the records written; raises if nothing changed worth capturing
    and writes nothing when `keep` admits no chunk (D89). `whole` mode
    carries no context. The texts are archived under
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
    chunk_pairs = _chunk_pairs(
        before_body, after_body, keep=keep, div_is_prose=div_is_prose, **knobs
    )
    if not chunk_pairs:
        return []

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
    records = _records(
        chunk_pairs,
        capture_id=cap_id,
        source=source,
        snapped_at=snapped_at,
        frontmatter=(before_meta, after_meta),
        system=system_prompt or source_policy.system_prompt,
        extra_meta=extra_meta,
        provenance=_provenance(
            blog_root or pairs_path.parent, knobs, source_policy, before_meta, after_meta
        ),
        captured_at=captured_at,
    )
    spans = [_span_of(before_text, b) for b, _a, _c in chunk_pairs]
    return _land(pairs_path, records, spans, supersedes=tuple(supersedes) or None)


def _provenance(
    root: Path | str, knobs: dict, source_policy: SourcePolicy, before_meta: dict, after_meta: dict
) -> dict:
    """`meta.provenance` (D36): the repo at `root`, the cut knobs' digest, the level."""
    return pair_provenance(
        repo_provenance(root),
        policy_digest=policy_digest({"producer": "capture", **knobs}),
        source_automation=source_policy.level_of(after_meta) or source_policy.level_of(before_meta),
    )


def _records(
    chunk_pairs: Sequence[tuple[str, str, str]],
    *,
    capture_id: str,
    source: str | None,
    snapped_at: str | None,
    frontmatter: tuple[dict, dict],
    system: str,
    extra_meta: dict | None,
    provenance: dict,
    captured_at: str,
) -> list[dict]:
    """One record per chunk; the frontmatter pair rides on chunk 0 only."""
    before_meta, after_meta = frontmatter
    return [
        build_pair_record(
            before_body=b,
            after_body=a,
            source=source,
            snapped_at=snapped_at,
            before_frontmatter=before_meta if i == 0 else None,
            after_frontmatter=after_meta if i == 0 else None,
            capture_id=capture_id,
            chunk_index=i,
            chunk_total=len(chunk_pairs),
            system=system,
            context=context,
            extra_meta=extra_meta,
            provenance=provenance,
            captured_at=captured_at,
        )
        for i, (b, a, context) in enumerate(chunk_pairs)
    ]


def recut_archive(
    archive_dir: Path,
    *,
    pairs_path: Path,
    supersedes: Sequence[str] = (),
    heading_context: bool = True,
    coalesce_gap_paras: int = 0,
    coalesce_gap_chars: int = 0,
    max_chars: int | None = None,
    blog_root: Path | str | None = None,
    extra_meta: dict | None = None,
    source_policy: SourcePolicy = DEFAULT_SOURCE,
    system_prompt: str | None = None,
    div_is_prose: DivIsProse | None = None,
) -> list[dict]:
    """Re-land an edited archive as draws under a fresh `capture_id` (D96):
    its own `whole`, `source` and `snapped_at`, `meta.archive_id` naming it,
    `captured_at` the landing time; nothing under `captures/` is written."""
    arch = read_archive(archive_dir)
    if arch is None:
        raise RunError(f"no readable capture archive at {archive_dir}")
    before_meta, before_body = split_frontmatter_meta(arch["before_text"])
    after_meta, after_body = split_frontmatter_meta(arch["after_text"])
    if before_body == after_body:
        raise RunError(
            f"{archive_dir.name}: before and after are identical after the edit; "
            "retire the capture by a policy clause instead"
        )
    knobs = {
        "whole": bool(arch.get("whole")),
        "heading_context": heading_context,
        "gap_paras": coalesce_gap_paras,
        "gap_chars": coalesce_gap_chars,
        "max_chars": max_chars,
    }
    chunk_pairs = _chunk_pairs(before_body, after_body, div_is_prose=div_is_prose, **knobs)
    records = _records(
        chunk_pairs,
        capture_id=new_capture_id(),
        source=arch.get("source"),
        snapped_at=arch.get("snapped_at"),
        frontmatter=(before_meta, after_meta),
        system=system_prompt or source_policy.system_prompt,
        extra_meta={**(extra_meta or {}), "archive_id": archive_dir.name},
        provenance=_provenance(
            blog_root or pairs_path.parent, knobs, source_policy, before_meta, after_meta
        ),
        captured_at=now_iso(),
    )
    spans = [_span_of(arch["before_text"], b) for b, _a, _c in chunk_pairs]
    return _land(pairs_path, records, spans, supersedes=tuple(supersedes) or None)


def archive_id_of(meta: Mapping) -> str | None:
    """The archive a real record was cut from: `archive_id` after a recut, else `capture_id`."""
    return meta.get("archive_id") or meta.get("capture_id")


def drifted_archives(captures_dir: Path, records: Iterable[dict]) -> list[tuple[str, int]]:
    """`(archive_id, n)`: `n` live records whose bodies are no longer substrings
    of their archive — edited since landing, so `log recut` is due (D96).
    Synthetic records and absent archives are not drift."""
    archives: dict[str, dict | None] = {}
    drift: dict[str, int] = {}
    for rec in records:
        meta = rec.get("meta") or {}
        aid = archive_id_of(meta)
        if meta.get("synthetic") or not aid:
            continue
        if aid not in archives:
            archives[aid] = read_archive(captures_dir / aid)
        arch = archives[aid]
        if arch is None:
            continue
        if (
            extract_slop(rec) not in arch["before_text"]
            or extract_target(rec) not in arch["after_text"]
        ):
            drift[aid] = drift.get(aid, 0) + 1
    return sorted(drift.items())


def plan_pairs(before_text: str, after_text: str, **knobs) -> list[tuple[str, str, str]]:
    """The `(before, after, context)` triples `capture_pairs` would land for two
    texts, frontmatter stripped, writing nothing; `knobs` are its `whole`,
    `heading_context`, `coalesce_gap_paras`, `coalesce_gap_chars`, `max_chars`,
    `div_is_prose`."""
    return _chunk_pairs(
        split_frontmatter_meta(before_text)[1],
        split_frontmatter_meta(after_text)[1],
        whole=knobs.get("whole", False),
        heading_context=knobs.get("heading_context", True),
        gap_paras=knobs.get("coalesce_gap_paras", 0),
        gap_chars=knobs.get("coalesce_gap_chars", 0),
        max_chars=knobs.get("max_chars"),
        div_is_prose=knobs.get("div_is_prose"),
    )


def _chunk_pairs(
    before_body: str,
    after_body: str,
    *,
    whole: bool,
    heading_context: bool,
    keep: Callable[[str, str], bool] | None = None,
    **cut,
) -> list[tuple[str, str, str]]:
    """The (before, after, context) triples of one capture: the whole body as
    one pair (no single heading applies), else `diff_chunks` under `cut`,
    then those `keep(before, after)` admits."""
    if whole:
        triples = [(before_body, after_body, "")]
        return [x for x in triples if keep is None or keep(x[0], x[1])]
    triples = diff_chunks(
        before_body,
        after_body,
        heading_context=heading_context,
        coalesce_gap_paras=cut["gap_paras"],
        coalesce_gap_chars=cut["gap_chars"],
        max_chars=cut["max_chars"],
        div_is_prose=cut.get("div_is_prose"),
    )
    if not triples:
        raise RunError(
            "no changed text chunks to capture "
            "(only whitespace, frontmatter, pure inserts, or pure deletes changed)"
        )
    return [x for x in triples if keep is None or keep(x[0], x[1])]


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
