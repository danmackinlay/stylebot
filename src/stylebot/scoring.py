"""Corpus-scale scoring: a `pairs.jsonl` in, an id-keyed `scores.jsonl` out.

One score record per corpus row, keyed by `record_id`, so scores join back to
the corpus, resume by id, and facet on carried `meta`. The per-passage signals
and `score_candidate` are `stylebot.eval`; the paired verdict over two scored
runs is `stylebot.verdict`.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from stylebot.classify import DETECTOR_IDENTITY_KEYS, detect_many
from stylebot.errors import ConfigError
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
from stylebot.jsonl import dumps_line, iter_jsonl, read_jsonl, write_jsonl_atomic
from stylebot.pairs import FIELD_EXTRACTORS, GEN_FACET_KEYS, iter_pairs, record_id

# meta keys carried onto each score record, so scores group / filter / join
# downstream without re-reading the corpus.
_CARRIED_META = (
    "source",
    "synthetic",
    "generator",
    "slop_strategy",
    "chunk_index",
    "transform_sim",
)
# Generation covariates nested under `meta.gen` (synthetic pairs only), flattened
# onto each score record so `summarize_scores(by=…)` can facet by them directly.
# Real pairs lack `meta.gen` and simply omit these (null bucket).
_CARRIED_GEN = GEN_FACET_KEYS


def existing_scored_ids(out_path: str | Path) -> set[str]:
    """Read the `id`s already present in a `scores.jsonl` (for resumable runs)."""
    return {str(rec["id"]) for rec in iter_jsonl(out_path) if rec.get("id")}


def live_record_ids(pairs_path: str | Path) -> set[str]:
    """Every join key the corpus currently carries — one `record_id` per pair."""
    return {record_id(rec) for rec in iter_pairs(pairs_path)}


def prune_orphan_scores(pairs_path: str | Path, out_path: str | Path) -> int:
    """Drop score records whose pair is gone from the corpus; return the count.

    `scores.jsonl` is append-only and joins on `record_id`, so a re-cut strands
    the superseded row; no consumer checks the join, so an orphan counts as a
    pair — it inflates every `summarize_scores` mean and puts a phantom row in
    a committed `covariate_table`. Pruning here keeps all of them consistent.

    Undecodable lines are round-tripped verbatim (`jsonl.write_jsonl_atomic`);
    a decodable record with no `id` has no join key and is pruned. Rewrites
    `out_path` atomically, and only when something is actually dropped.
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


# Which instrument wrote these rows (D39). `eval.detector_identity` reads a
# live detector; these read what a scores file already records.


def _identity_of_row(scores: dict) -> tuple | None:
    """The detector identity recorded on one score record, as a hashable tuple.

    None when the row carries no *configured* detector — an unconfigured or
    pre-contract row is not evidence of a different instrument, so it must not
    trip the mixing guard.
    """
    for signals in (scores or {}).values():
        detector = (signals or {}).get("detector") or {}
        if detector.get("configured") and detector.get("score") is not None:
            return tuple(detector.get(key) for key in DETECTOR_IDENTITY_KEYS)
    return None


def detector_identities(scores: str | Path | Iterable[dict]) -> list[dict]:
    """Every distinct detector identity present in a `scores.jsonl`.

    More than one means the file mixes instruments: some rows were scored by a
    head that no longer exists, and any mean over the file is a blend. Order is
    first-seen, so the head of the list is the file's original instrument.
    """
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
    """Refuse a scores file written by more than one detector; return what it has.

    Everything that *compares* numbers goes through here first (D39).
    `allow_mixed=True` downgrades to "return them all" for the deliberate case:
    a file being migrated, a diagnostic.
    """
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
    # An unconfigured (null) detector adds no detector scores at all, so it can
    # never blend the file's means; only a *scoring* head has to match.
    if incoming == existing or incoming.get("name") in (None, "null"):
        return
    raise ConfigError(
        f"{out_path} was scored by {_identity_label(existing)} but this run uses "
        f"{_identity_label(incoming)} — resume skips by pair id, so appending would "
        f"blend two instruments into one mean. Point --out at a fresh file, or pass "
        f"--allow-mixed-detector"
    )


@dataclass
class ScoreResult:
    """Outcome of a `score_pairs_file` run."""

    written: int = 0
    skipped_existing: int = 0
    pruned_orphans: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (id, message)


# Records per slice of the scoring loop. Each slice takes one pre-pass per
# signal, so this is the granularity at which results reach the file: bigger
# amortises `vale` startup and the judge's concurrency further, smaller keeps a
# crashed run's lost work smaller. `fields` multiplies it into texts per slice.
_SCORE_BATCH_RECORDS = 128


def _score_record(
    record: dict,
    *,
    fields: Sequence[str],
    judge: Judge | None,
    detector: Detector,
    vale_config: str | Path | None,
    vale_by_field: dict[str, dict] | None = None,
    detector_by_field: dict[str, dict] | None = None,
    judge_by_field: dict[str, dict | BaseException] | None = None,
) -> dict:
    """Score the requested fields of one pair record -> a `scores.jsonl` line.

    The three `*_by_field` maps carry the slice's pre-pass results (field name
    -> that signal's record); a field missing from one falls back to scoring
    that signal inline. A judge call that raised is carried as the exception and
    re-raised here, so it fails its own record and no other.
    """
    meta = record.get("meta") or {}
    scores: dict[str, dict] = {}
    for name in fields:
        text = FIELD_EXTRACTORS[name](record)
        verdict = (judge_by_field or {}).get(name)
        if isinstance(verdict, BaseException):
            raise verdict
        scores[name] = score_candidate(
            text,
            judge=judge,
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
    """Every (record, field) text in a slice, plus the slot each came from.

    An extraction that blows up is simply left out: `_score_record` hits the
    same error and `ScoreResult` captures it per record. A pre-pass must never
    be the thing that fails a run.
    """
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
    """Batch-score every (record, field) text with Vale -> per-record results.

    Returns one ``{field: vale_record}`` dict per input record, aligned to
    `records`. Never raises and never fails a record; a Vale batch failure
    degrades the affected texts to ``available: False``.
    """
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
    """Score every (record, field) text with the detector, Vale-style (D59).

    One `classify.detect_many` call per slice. A failing batch retries per text
    on this task and then records the error in the field: a signal is never a
    run-ender. One ``{field: detector_record}`` dict per record, aligned.
    """
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
    """Judge every (record, field) text, `max_workers` calls in flight (D59).

    An async judge is awaited under an `asyncio.Semaphore`; a sync one is serial,
    so the entry points take it only at `max_workers=1`. A call that raised is
    carried back as its exception, re-raised against that record alone.
    """
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


def score_pairs_file(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    fields: Sequence[str] = ("slop", "target"),
    judge: Judge | AsyncJudge | None = None,
    detector: Detector = null_detector,
    vale_config: str | Path | None = None,
    max_workers: int = 8,
    limit: int | None = None,
    prune_orphans: bool = True,
    allow_mixed_detector: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> ScoreResult:
    """Score a `pairs.jsonl` corpus -> an id-keyed `scores.jsonl` (batched).

    The sync entry point: it owns an event loop (`asyncio.run`) for the judge
    pass. Call `score_pairs_file_async` from inside a running loop instead.
    """
    _guard_no_running_loop()
    return asyncio.run(
        score_pairs_file_async(
            pairs_path,
            out_path,
            fields=fields,
            judge=judge,
            detector=detector,
            vale_config=vale_config,
            max_workers=max_workers,
            limit=limit,
            prune_orphans=prune_orphans,
            allow_mixed_detector=allow_mixed_detector,
            on_progress=on_progress,
        )
    )


def _unscored_records(
    pairs_path: Path, seen: set[str], limit: int | None
) -> tuple[list[dict], int]:
    """The corpus rows this run has to score, and how many it skipped."""
    todo: list[dict] = []
    skipped = 0
    for rec in iter_pairs(pairs_path):
        if record_id(rec) in seen:
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
    *,
    fields: Sequence[str],
    judge: Judge | AsyncJudge | None,
    detector: Detector,
    vale_config: str | Path | None,
    max_workers: int,
    progress: Callable[[], None] | None = None,
) -> None:
    """One pre-pass per signal over the slice, then one written line per record.

    Assembling the records afterwards is pure bookkeeping. Appends to `result`
    in place: a written record bumps `written`, a raising one lands in `errors`.
    """
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
        except Exception as exc:  # a field/judge error on one record — log, continue
            result.errors.append((record_id(rec), f"{type(exc).__name__}: {exc}"))
            continue
        fp.write(dumps_line(line) + "\n")
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
    max_workers: int = 8,
    limit: int | None = None,
    prune_orphans: bool = True,
    allow_mixed_detector: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> ScoreResult:
    """Score a `pairs.jsonl` corpus -> an id-keyed `scores.jsonl` (batched).

    For each pair, score the named `fields` and append one score record keyed by
    `record_id`. Idempotent + resumable: ids already in `out_path` are skipped.
    Records go in slices of `_SCORE_BATCH_RECORDS` (`_score_slice`), so a crash
    costs at most the slice in flight. `max_workers` counts judge calls in
    flight — the only concurrency (D59), so a corpus pass wants an async judge.

    On entry `out_path` is pruned of records whose pair no longer exists
    (`prune_orphan_scores`; count in `ScoreResult.pruned_orphans` — surface it,
    this rewrites a file the caller may have committed) against the WHOLE
    corpus, `limit` notwithstanding; pass `prune_orphans=False` when `out_path`
    belongs to a different corpus. The existing rows' detector identity must
    match this run's unless `allow_mixed_detector` (D39).
    """
    pairs_path = Path(pairs_path)
    out_path = Path(out_path)

    unknown = [f for f in fields if f not in FIELD_EXTRACTORS]
    if unknown:
        raise ConfigError(f"unknown field(s) {unknown}; known: {sorted(FIELD_EXTRACTORS)}")
    _guard_judge_concurrency(judge, max_workers)

    # Prune before reading `seen`: an orphan id would otherwise both survive in
    # the file and count as already-scored.
    pruned = prune_orphan_scores(pairs_path, out_path) if prune_orphans else 0

    _guard_detector_continuity(out_path, detector, allow_mixed=allow_mixed_detector)

    todo, skipped = _unscored_records(pairs_path, existing_scored_ids(out_path), limit)
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
                    fields=fields,
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
    """Load score records from a `scores.jsonl` path (or accept them in-memory).

    Tolerant (`stylebot.jsonl`: blank/undecodable lines skipped, missing file →
    empty). An iterable is returned as a list unchanged.
    """
    if isinstance(scores, (str, Path)):
        return read_jsonl(scores)
    return list(scores)


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
    """Score records split `real` / `synthetic` on `meta.synthetic`.

    The headline number comes from the real facet; the synthetic one is
    reported separately, never folded in. Why: D40.
    """
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

    Fields-across-rows are the "groups" the styler is judged by movement across
    (slop vs target vs output). `by` additionally facets on a carried `meta` key
    — `by="slop_strategy"` gives per-strategy means. `facet_provenance`
    (default on) adds `by_provenance.real` / `.synthetic` and names
    `headline_facet: "real"` (D40). The summary always records the file's
    `detector` identities beside the reading (D39).
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
