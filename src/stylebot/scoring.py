"""Corpus-scale scoring: a `pairs.jsonl` in, a signal-keyed `scores.jsonl` out.

One row per `(record_id, field, signal)`, appended, never rewritten, beside the
corpus rather than in a rebuildable view (D39, D84). `load_scores` folds the
rows — and an old whole-record file — back to the shape readers take. Signals
are `stylebot.eval`; the paired verdict over two runs is `stylebot.verdict`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from stylebot.classify import DETECTOR_IDENTITY_KEYS, detect_many
from stylebot.cli_options import Emitter, silent
from stylebot.errors import ConfigError, RunError
from stylebot.eval import (
    SCHEMA_VERSION,
    AsyncJudge,
    Detector,
    Judge,
    _aggregate,
    _vale_unavailable,
    detector_identity,
    null_detector,
    score_candidate,
    vale_score_many,
)
from stylebot.jsonl import dumps_line, read_jsonl, write_jsonl_atomic
from stylebot.pairs import FIELD_EXTRACTORS, GEN_FACET_KEYS, iter_pairs, record_id

if TYPE_CHECKING:
    from stylebot.docscale import DocumentPass
    from stylebot.verdict import EvalOpts, VerdictOpts

# meta keys carried onto each score record, for grouping without the corpus.
_CARRIED_META = (
    "source",
    "synthetic",
    "generator",
    "slop_strategy",
    "chunk_index",
    "transform_sim",
    "reranker",
)
# `meta.gen` covariates (synthetic pairs only) carried onto each row for
# `summarize_scores(by=…)`; real pairs omit them.
_CARRIED_GEN = GEN_FACET_KEYS

#: Judge calls in flight (D59); `verdict.EvalOpts.max_workers` shares it.
DEFAULT_MAX_WORKERS = 8


#: The signals a row can carry: an `eval.score_candidate` record's keys, less
#: `text_chars`, which is the field's and rides on every one of its rows.
SCORE_SIGNALS = ("vale", "judge", "detector", "eyeball")


def explode_score_row(rec: dict) -> list[dict]:
    """A whole-record score row -> its `(id, field, signal)` rows; pure. Also
    reads the pre-D39 shape. No result, no row; a keyed row passes through."""
    if "signal" in rec or "scores" not in rec:
        return [rec]
    base = {"id": rec.get("id"), "meta": rec.get("meta") or {}}
    cells = {name: cell or {} for name, cell in (rec.get("scores") or {}).items()}
    return [
        {**base, "field": name, "signal": s, "text_chars": c.get("text_chars"), "value": c[s]}
        for name, c in cells.items()
        for s in SCORE_SIGNALS
        if c.get(s) is not None
    ]


def _fold_score_rows(rows: Iterable[dict]) -> list[dict]:
    """Rows -> one record per id, first-seen order; last write wins per key,
    which makes an append an upsert (D39)."""
    folded: dict[str, dict] = {}
    for row in rows:
        rid, name, signal = row.get("id"), row.get("field"), row.get("signal")
        if rid is None or not name or not signal:
            continue
        rec = folded.setdefault(str(rid), {"id": rid, "meta": {}, "scores": {}})
        rec["meta"] = row.get("meta") or rec["meta"]
        cell = rec["scores"].setdefault(name, {"text_chars": None} | dict.fromkeys(SCORE_SIGNALS))
        cell["text_chars"], cell[signal] = row.get("text_chars"), row.get("value")
    return list(folded.values())


def existing_score_keys(out_path: str | Path) -> set[tuple[str, str, str]]:
    """Every `(record_id, field, signal)` a scores file already holds."""
    return {
        (str(row["id"]), str(row["field"]), str(row["signal"]))
        for rec in read_jsonl(out_path)
        for row in explode_score_row(rec)
        if row.get("id") is not None and row.get("field") and row.get("signal")
    }


def live_record_ids(pairs_path: str | Path) -> set[str]:
    """Every join key the corpus currently carries — one `record_id` per pair."""
    return {record_id(rec) for rec in iter_pairs(pairs_path)}


def prune_orphan_scores(pairs_path: str | Path, out_path: str | Path) -> int:
    """Drop score rows whose pair is gone from the corpus; return the row count (D63).

    Undecodable lines round-trip verbatim; a decodable row with no `id` is
    pruned. Rewritten atomically, and only when something is dropped.
    """
    records = read_jsonl(out_path, keep_undecodable=True)
    if not records:
        return 0
    live = live_record_ids(pairs_path)
    kept = [
        rec
        for rec in records
        if (set(rec) == {"_raw"}) or (rec.get("id") is not None and str(rec["id"]) in live)
    ]
    pruned = len(records) - len(kept)
    if pruned:
        write_jsonl_atomic(Path(out_path), kept)
    return pruned


# Which instrument wrote these rows (D39): `eval.detector_identity` reads a live
# detector, these read what a scores file records.


def _scores_detector(detector: Detector) -> bool:
    """Does this head produce detector rows, or is it the null default (D39)?
    A chunk-wise wrapper suffixes its scale (D54), so the head precedes the `/`."""
    name = detector_identity(detector).get("name")
    return str(name or "null").split("/")[0] != "null"


def scores_filename(detector: Detector) -> str:
    """The file these rows belong in: `scores-<name>@<git_sha>.jsonl` (D39);
    a null head keeps the plain default, so nothing existing moves."""
    if not _scores_detector(detector):
        return "scores.jsonl"
    ident = detector_identity(detector)
    raw = f"{ident['name']}@{ident['git_sha']}" if ident.get("git_sha") else ident["name"]
    slug = "".join(c if (c.isalnum() or c in "@._-") else "-" for c in str(raw))
    return f"scores-{slug}.jsonl"


def _identity_of_row(scores: dict) -> tuple | None:
    """The detector identity recorded on one score record, as a hashable tuple;
    None when it carries no configured detector (D39)."""
    for signals in (scores or {}).values():
        detector = (signals or {}).get("detector") or {}
        if detector.get("configured") and detector.get("score") is not None:
            return tuple(detector.get(key) for key in DETECTOR_IDENTITY_KEYS)
    return None


def detector_identities(scores: str | Path | Iterable[dict]) -> list[dict]:
    """Every distinct detector identity in a `scores.jsonl`, first-seen order;
    more than one means any mean over the file is a blend (D39)."""
    seen: dict[tuple, dict] = {}
    for rec in load_scores(scores):
        key = _identity_of_row(rec.get("scores") or {})
        if key is not None and key not in seen:
            seen[key] = dict(zip(DETECTOR_IDENTITY_KEYS, key, strict=True))
    return list(seen.values())


def _identity_label(identity: dict) -> str:
    return (
        f"{identity.get('name')}@{identity.get('git_sha')} "
        f"(trained_at={identity.get('trained_at')}, n_pairs={identity.get('n_pairs')})"
    )


def assert_single_detector(
    scores: str | Path | Iterable[dict], *, allow_mixed: bool = False
) -> list[dict]:
    """Refuse a scores file written by more than one detector (D39); return what
    it has. `allow_mixed=True` returns them all."""
    identities = detector_identities(scores)
    if len(identities) > 1 and not allow_mixed:
        labels = "; ".join(_identity_label(i) for i in identities)
        raise ConfigError(
            f"scores mix {len(identities)} detector artifacts — every mean over them is a "
            f"blend of instruments: {labels}. Re-score with one head, or pass "
            f"allow_mixed=True / --allow-mixed-detector to say you meant it"
        )
    return identities


def _guard_detector_continuity(
    out_path: str | Path, detector: Detector, *, allow_mixed: bool
) -> None:
    """Refuse to append rows from a different head than the file already holds."""
    identities = assert_single_detector(out_path, allow_mixed=allow_mixed)
    if not identities or allow_mixed:
        return
    incoming = detector_identity(detector)
    if incoming.get("name") is None and all(v is None for v in incoming.values()):
        return  # an unidentifiable detector cannot be shown to differ
    existing = identities[0]
    # A null detector adds no detector rows, so only a scoring head has to match.
    if incoming == existing or not _scores_detector(detector):
        return
    fresh = Path(out_path).parent / scores_filename(detector)
    raise ConfigError(
        f"{out_path} was scored by {_identity_label(existing)} but this run uses "
        f"{_identity_label(incoming)} — resume skips by signal key, so appending would "
        f"blend two instruments into one mean. Point --out at a fresh file "
        f"({fresh}), or pass --allow-mixed-detector"
    )


@dataclass
class ScoreResult:
    """Outcome of a `score_pairs_file` run."""

    written: int = 0
    skipped_existing: int = 0
    pruned_orphans: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (id, message)


# Records per slice, the granularity at which results reach the file: bigger
# amortises `vale` startup, smaller loses less to a crash.
_SCORE_BATCH_RECORDS = 128


def _score_record(
    record: dict,
    *,
    fields: Sequence[str],
    judge: Judge | AsyncJudge | None,
    detector: Detector,
    vale_config: str | Path | None,
    vale_by_field: dict[str, dict] | None = None,
    detector_by_field: dict[str, dict] | None = None,
    judge_by_field: dict[str, dict | BaseException] | None = None,
) -> dict:
    """Score the requested fields of one pair record -> a whole-record row.

    The `*_by_field` maps carry the slice's pre-pass results; a field missing
    from the judge map raises (an inline call would store a coroutine). A judge
    call that raised is re-raised here, against that record alone.
    """
    meta = record.get("meta") or {}
    scores: dict[str, dict] = {}
    for name in fields:
        text = FIELD_EXTRACTORS[name](record)
        verdict = (judge_by_field or {}).get(name)
        if isinstance(verdict, BaseException):
            raise verdict
        if judge is not None and verdict is None:
            raise RunError(f"record {record_id(record)!r}: no judge verdict for field {name!r}")
        scores[name] = score_candidate(
            text,
            detector=detector,
            vale_config=vale_config,
            vale=(vale_by_field or {}).get(name),
            detector_result=(detector_by_field or {}).get(name),
            judge_result=verdict,
        )
    gen = meta.get("gen") or {}
    carried = {k: meta.get(k) for k in _CARRIED_META if k in meta}
    carried.update({k: gen[k] for k in _CARRIED_GEN if k in gen})
    return {
        "id": record_id(record),
        "meta": carried,
        "scores": scores,
    }


def _slice_texts(
    records: Sequence[dict], fields: Sequence[str]
) -> tuple[list[str], list[tuple[int, str]]]:
    """Every (record, field) text in a slice, plus the slot each came from; an
    extraction that raises is left for `_score_record` to capture per record."""
    texts: list[str] = []
    slots: list[tuple[int, str]] = []
    for i, record in enumerate(records):
        for name in fields:
            try:
                text = FIELD_EXTRACTORS[name](record)
            except Exception:  # noqa: BLE001 - the worker re-hits this and reports it per record
                continue
            slots.append((i, name))
            texts.append(text)
    return texts, slots


def _vale_prepass(
    records: Sequence[dict],
    *,
    fields: Sequence[str],
    vale_config: str | Path | None,
) -> list[dict[str, dict]]:
    """Batch-score every (record, field) text with Vale -> one ``{field:
    vale_record}`` per record, aligned. Never raises; a batch failure degrades."""
    texts, slots = _slice_texts(records, fields)
    per_record: list[dict[str, dict]] = [{} for _ in records]
    if not texts:
        return per_record
    try:
        results = vale_score_many(texts, config_path=vale_config)
    except Exception as exc:  # noqa: BLE001 - Vale is a signal, never a run-ender
        reason = f"vale batch failed: {type(exc).__name__}: {exc}"
        results = [_vale_unavailable(reason) for _ in texts]
    for (i, name), result in zip(slots, results, strict=True):
        per_record[i][name] = result
    return per_record


def _detector_error(detector: Detector, exc: BaseException) -> dict:
    """An unscored detector record that still says which head was asked (D39)."""
    return detector_identity(detector) | {
        "score": None,
        "configured": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _detector_prepass(
    records: Sequence[dict],
    *,
    fields: Sequence[str],
    detector: Detector,
) -> list[dict[str, dict]]:
    """Score every (record, field) text with the detector, Vale-style (D59): one
    `detect_many` per slice, a failing batch retried per text. One dict per record."""
    per_record: list[dict[str, dict]] = [{} for _ in records]
    texts, slots = _slice_texts(records, fields)
    if not texts:
        return per_record
    try:
        results = detect_many(detector, texts)
    except Exception:  # noqa: BLE001 - a bad batch degrades to one text at a time
        results = []
        for text in texts:
            try:
                results.append(detector(text))
            except Exception as exc:  # noqa: BLE001 - and a bad text degrades to a note
                results.append(_detector_error(detector, exc))
    for (i, name), result in zip(slots, results, strict=True):
        per_record[i][name] = result
    return per_record


async def _judge_prepass(
    records: Sequence[dict],
    *,
    fields: Sequence[str],
    judge: Judge | AsyncJudge | None,
    max_workers: int,
) -> list[dict[str, dict | BaseException]]:
    """Judge every (record, field) text, `max_workers` calls in flight (D59); a
    sync judge is serial. A call that raised is carried back as its exception."""
    per_record: list[dict[str, dict | BaseException]] = [{} for _ in records]
    if judge is None:
        return per_record
    texts, slots = _slice_texts(records, fields)
    if not texts:
        return per_record
    gate = asyncio.Semaphore(max(1, max_workers))

    async def _one(text: str) -> dict:
        async with gate:
            verdict = judge(text)
            return await verdict if inspect.isawaitable(verdict) else verdict

    results = await asyncio.gather(*(_one(t) for t in texts), return_exceptions=True)
    for (i, name), result in zip(slots, results, strict=True):
        per_record[i][name] = result
    return per_record


def _is_async_judge(judge: Judge | AsyncJudge) -> bool:
    call = judge if inspect.isfunction(judge) or inspect.ismethod(judge) else type(judge).__call__
    return inspect.iscoroutinefunction(judge) or inspect.iscoroutinefunction(call)


def _guard_judge_concurrency(judge: Judge | AsyncJudge | None, max_workers: int) -> None:
    if judge is None or max_workers <= 1 or _is_async_judge(judge):
        return
    name = getattr(judge, "__name__", type(judge).__name__)
    raise ConfigError(
        f"judge {name!r} is synchronous, so max_workers={max_workers} would run one "
        "call at a time; pass an async judge (eval.openrouter_judge_async) or "
        "max_workers=1."
    )


def _guard_no_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise ConfigError(
        "score_pairs_file owns an event loop and cannot run inside one; "
        "await score_pairs_file_async instead."
    )


async def _aclose_judge(judge: Judge | AsyncJudge | None) -> None:
    aclose = getattr(judge, "aclose", None)
    if aclose is not None:
        await aclose()


def score_pairs_file(pairs_path: str | Path, out_path: str | Path, **kwargs) -> ScoreResult:
    """The sync entry point: `asyncio.run` over `score_pairs_file_async`, whose
    keywords it takes."""
    _guard_no_running_loop()
    return asyncio.run(score_pairs_file_async(pairs_path, out_path, **kwargs))


def _run_signals(judge: Judge | AsyncJudge | None, detector: Detector) -> tuple[str, ...]:
    """The signals this run can produce — the keys it is allowed to skip on."""
    signals = ["vale"]
    if _scores_detector(detector):
        signals.append("detector")
    if judge is not None:
        signals.append("judge")
    return tuple(signals)


def _unscored_records(
    pairs_path: Path,
    keys: set[tuple[str, str, str]],
    *,
    fields: Sequence[str],
    signals: Sequence[str],
    limit: int | None,
) -> tuple[list[dict], int]:
    """The corpus rows this run owes a key, and how many it skipped: work is a
    missing `(record_id, field, signal)`, not a missing record."""
    todo: list[dict] = []
    skipped = 0
    for rec in iter_pairs(pairs_path):
        rid = record_id(rec)
        if all((rid, f, s) in keys for f in fields for s in signals):
            skipped += 1
            continue
        todo.append(rec)
        if limit is not None and len(todo) >= limit:
            break
    return todo, skipped


async def _score_slice(
    fp,
    batch: Sequence[dict],
    result: ScoreResult,
    keys: set[tuple[str, str, str]],
    *,
    fields: Sequence[str],
    signals: Sequence[str],
    judge: Judge | AsyncJudge | None,
    detector: Detector,
    vale_config: str | Path | None,
    max_workers: int,
    progress: Callable[[], None] | None = None,
) -> None:
    """One pre-pass per signal over the slice, then the rows the file still owes:
    a `signals` this run produces that `keys` lacks. A record that raises or will
    not serialise lands in `result.errors`, nothing written."""
    vale_results = _vale_prepass(batch, fields=fields, vale_config=vale_config)
    det_results = _detector_prepass(batch, fields=fields, detector=detector)
    judge_results = await _judge_prepass(batch, fields=fields, judge=judge, max_workers=max_workers)
    for i, rec in enumerate(batch):
        if progress is not None:
            progress()
        try:
            line = _score_record(
                rec,
                fields=fields,
                judge=judge,
                detector=detector,
                vale_config=vale_config,
                vale_by_field=vale_results[i],
                detector_by_field=det_results[i],
                judge_by_field=judge_results[i],
            )
            rows = [
                ((str(row["id"]), row["field"], row["signal"]), dumps_line(row))
                for row in explode_score_row(line)
                if row["signal"] in signals
            ]
        except Exception as exc:  # a field/judge/row error on one record — log, continue
            result.errors.append((record_id(rec), f"{type(exc).__name__}: {exc}"))
            continue
        for key, text in rows:
            if key not in keys:
                keys.add(key)
                fp.write(text + "\n")
        fp.flush()
        result.written += 1


async def score_pairs_file_async(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    fields: Sequence[str] = ("slop", "target"),
    judge: Judge | AsyncJudge | None = None,
    detector: Detector = null_detector,
    vale_config: str | Path | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    limit: int | None = None,
    prune_orphans: bool = True,
    allow_mixed_detector: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> ScoreResult:
    """Score a corpus -> a signal-keyed `scores.jsonl`, batched and resumable.

    One row per `(record_id, field, signal)`; keys already in `out_path` are
    skipped, so a run adds a signal without re-buying one. Per slice the Vale
    and detector pre-passes block the loop, then `max_workers` judge calls go
    in flight (D59). The rows' detector identity must match this run's unless
    `allow_mixed_detector` (D39), checked before anything writes; `out_path` is
    then pruned against the whole corpus (`prune_orphans=False` for another).
    """
    pairs_path = Path(pairs_path)
    out_path = Path(out_path)

    unknown = [f for f in fields if f not in FIELD_EXTRACTORS]
    if unknown:
        raise ConfigError(f"unknown field(s) {unknown}; known: {sorted(FIELD_EXTRACTORS)}")
    _guard_judge_concurrency(judge, max_workers)

    # Guard, then prune, then `keys`: a refused run rewrites nothing, and an
    # orphan must not count as scored.
    _guard_detector_continuity(out_path, detector, allow_mixed=allow_mixed_detector)
    pruned = prune_orphan_scores(pairs_path, out_path) if prune_orphans else 0

    signals = _run_signals(judge, detector)
    keys = existing_score_keys(out_path)
    todo, skipped = _unscored_records(pairs_path, keys, fields=fields, signals=signals, limit=limit)
    result = ScoreResult(skipped_existing=skipped, pruned_orphans=pruned)
    if not todo:
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(todo)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    batch_size = max(1, _SCORE_BATCH_RECORDS)
    try:
        with out_path.open("a", encoding="utf-8") as fp:
            for start in range(0, total, batch_size):
                await _score_slice(
                    fp,
                    todo[start : start + batch_size],
                    result,
                    keys,
                    fields=fields,
                    signals=signals,
                    judge=judge,
                    detector=detector,
                    vale_config=vale_config,
                    max_workers=max_workers,
                    progress=tick,
                )
    finally:
        await _aclose_judge(judge)
    return result


def load_scores(scores: str | Path | Iterable[dict]) -> list[dict]:
    """Load a `scores.jsonl` (or in-memory rows, keyed or pre-D39 whole-record)
    folded to one record per id; tolerant of a missing file."""
    raw = read_jsonl(scores) if isinstance(scores, (str, Path)) else list(scores)
    return _fold_score_rows(row for rec in raw for row in explode_score_row(rec))


def _aggregate_fields(records: list[dict]) -> dict:
    """Per-field aggregates over score records (fields-across-rows = groups)."""
    field_names: list[str] = []
    for rec in records:
        for name in rec.get("scores") or {}:
            if name not in field_names:
                field_names.append(name)
    return {
        name: _aggregate(
            [rec["scores"][name] for rec in records if name in (rec.get("scores") or {})]
        )
        for name in field_names
    }


def split_by_provenance(records: Sequence[dict]) -> dict[str, list[dict]]:
    """Score records split `real` / `synthetic` on `meta.synthetic` (D40)."""
    facets: dict[str, list[dict]] = {"real": [], "synthetic": []}
    for rec in records:
        facets["synthetic" if (rec.get("meta") or {}).get("synthetic") else "real"].append(rec)
    return facets


def summarize_scores(
    scores: str | Path | Iterable[dict],
    *,
    by: str | None = None,
    facet_provenance: bool = True,
) -> dict:
    """Aggregate a `scores.jsonl` (path or records) per field, optionally faceted.

    `by` facets on a carried `meta` key; `facet_provenance` adds
    `by_provenance.real` / `.synthetic` with `headline_facet: "real"` (D40);
    the detector identities ride beside it (D39).
    """
    records = load_scores(scores)
    summary: dict = {
        "schema_version": SCHEMA_VERSION,
        "pairs": len(records),
        "fields": _aggregate_fields(records),
        "detectors": detector_identities(records),
    }
    if facet_provenance:
        facets = split_by_provenance(records)
        summary["by_provenance"] = {
            name: {"pairs": len(rows), "fields": _aggregate_fields(rows)}
            for name, rows in facets.items()
        }
        summary["headline_facet"] = "real"
    if by is not None:
        buckets: dict[str, list[dict]] = {}
        for rec in records:
            key = (rec.get("meta") or {}).get(by)
            buckets.setdefault("null" if key is None else str(key), []).append(rec)
        summary["by"] = {k: _aggregate_fields(v) for k, v in sorted(buckets.items())}
    return summary


def run_eval(
    opts: EvalOpts,
    verdict: VerdictOpts,
    *,
    detector: Detector = null_detector,
    documents: DocumentPass | None = None,
    emit: Emitter = silent,
) -> dict:
    """The `eval` pass (D78): score, summarise, then `verdict.report_eval`.

    `documents` is a finished `docscale.document_pass`: its output replaces the
    corpus, `output` joins the fields, pruning is off (a different corpus), and
    `detector` is wrapped at its serving scale (D54). Returns the summary.
    """
    from stylebot.eval import chunked_detector, openrouter_judge_async
    from stylebot.verdict import format_summary, report_eval

    if opts.pairs_path is None:
        raise ConfigError("no corpus: pass --draws, or a scratch --pairs", param_hint="--pairs")
    pairs_path, out = Path(opts.pairs_path), opts.scores_path()
    fields, prune_orphans = opts.score_fields(), opts.prune_orphans
    if documents is not None:
        pairs_path, fields = documents.pairs_path, documents.fields
        prune_orphans = documents.prune_orphans
        detector = chunked_detector(detector, **documents.chunk_scale)
    judge = openrouter_judge_async(model=opts.judge_model) if opts.judge else None

    def progress(i: int, total: int) -> None:
        if i == 1 or i % 25 == 0 or i == total:
            emit(f"  ... {i}/{total} scored", level="status")

    result = score_pairs_file(
        pairs_path,
        out,
        fields=fields,
        judge=judge,
        detector=detector,
        vale_config=opts.vale_config,
        max_workers=opts.max_workers,
        limit=opts.limit,
        prune_orphans=prune_orphans,
        allow_mixed_detector=verdict.allow_mixed_detector,
        on_progress=progress,
    )
    # Loud: the prune rewrote a file the operator may have committed.
    if result.pruned_orphans:
        emit(
            f"pruned {result.pruned_orphans} orphan score row(s) from {out} "
            f"— their pair is no longer in {pairs_path}",
            level="status",
        )
    emit(
        f"scored {result.written} pair(s) "
        f"({result.skipped_existing} already scored, skipped) -> {out}"
    )
    if result.errors:
        emit(f"{len(result.errors)} scoring error(s):", level="error")
        for rid, msg in result.errors[:10]:
            emit(f"  {rid}: {msg}", level="error")

    summary = summarize_scores(out, by=opts.by)
    for line in format_summary(summary):
        emit(line)
    if opts.by is not None:
        emit(f"by {opts.by}:")
        for facet, fld in summary["by"].items():
            parts = ", ".join(
                f"{fn}=det:{a['mean_detector_score']}/judge:{a['mean_judge_score']}"
                f"/unscored:{a['n_unscored']}"
                for fn, a in fld.items()
            )
            emit(f"  {facet}: {parts}")
    if opts.summary_path is not None:
        Path(opts.summary_path).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        emit(f"wrote summary -> {opts.summary_path}")
    report_eval(out, pairs_path, fields, verdict, opts, emit=emit)
    return summary
