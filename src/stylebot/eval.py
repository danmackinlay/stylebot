"""Eval harness — offline scoring of candidate prose against the four signals.

The ground-truth scorer every later phase reports against. It runs **offline**:
it scores candidate output and is *not* wired into the trainer, *not* baked into
the served adapter. Spec: `_plans/eval-harness.md`.

Library-first: typed functions over explicit prose strings / file lists. This
module never imports a CLI and never imports `livingthing`.

The four signals (`_plans/eval-harness.md` "The four signals"): **Vale**
(mechanical slop, shelled out to the binary, degrades gracefully if absent), an
**LLM judge** (injectable `Judge`; `openrouter_judge` builds a real one), a
**statistical detector** (pluggable `Detector`, `null_detector` by default), and
**Dan's eyeball** (the non-automatable veto channel, carried as a string).

Guardrail (`_plans/eval-harness.md` "Guardrail"): this harness is **not** a
slop-detection evader. A styler should lower its detector score only as a
*consequence* of writing more like Dan. The judge and eyeball signals exist to
catch reward-hacking (passes-detector-but-still-bad output); a movement in the
detector alone is never the success criterion.

Heavy / networked imports (`openai`) are LAZY — inside the factory — so
importing `stylebot.eval` never requires the SDK or an API key. The harness
runs end-to-end with no key: Vale-only (or Vale-absent), null judge, null
detector.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Protocol

from stylebot.jsonl import dumps_line, iter_jsonl, read_jsonl, write_jsonl_atomic

# The pair-text extractors and the score join key live in `stylebot.pairs`; this
# module imports the four it uses and re-exports nothing: every consumer
# (`infer`, `train`, `classify_train`, the scripts, the tests, livingthing)
# reaches them at `stylebot.pairs` directly, so a facade here would be a second
# import path for one surface.
from stylebot.pairs import FIELD_EXTRACTORS, GEN_FACET_KEYS, iter_pairs, record_id
from stylebot.provenance import file_sha256

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Signal 1 — Vale (mechanical slop)
# ---------------------------------------------------------------------------


# Files handed to a single `vale` invocation. Vale takes many paths at once and
# returns one {filename: [alerts]} map, so batching turns "one process per scored
# text" into "one process per this many". Capped rather than unbounded: the paths
# all land on one argv, and ARG_MAX is the wall.
_VALE_FILES_PER_RUN = 256


def _vale_unavailable(reason: str) -> dict:
    """The graceful-degradation record. Fresh dicts per call — callers mutate."""
    return {
        "available": False,
        "alerts": 0,
        "by_severity": {"error": 0, "warning": 0, "suggestion": 0},
        "raw": [],
        "reason": reason,
    }


def _vale_summarize(alerts: list[dict]) -> dict:
    by_severity = {"error": 0, "warning": 0, "suggestion": 0}
    for alert in alerts:
        sev = str(alert.get("Severity", "")).lower()
        if sev in by_severity:
            by_severity[sev] += 1
    return {
        "available": True,
        "alerts": len(alerts),
        "by_severity": by_severity,
        "raw": alerts,
    }


def _basename(key: str) -> str:
    """Last path segment of a vale JSON key, whichever separator it used.

    Vale keys its output map by the path it was *handed*: absolute when we pass
    absolute paths, relative when relative. We always pass absolute paths inside
    one temp directory whose filenames are unique by construction, so the last
    segment is an unambiguous join key either way.
    """
    return str(key).replace("\\", "/").rsplit("/", 1)[-1]


def _vale_run(texts: Sequence[str], *, config_path: str | Path | None) -> list[dict]:
    """One `vale` invocation over `texts`; one result dict per text, in order."""
    cmd = ["vale", "--output=JSON"]
    if config_path is not None:
        cmd.append(f"--config={Path(config_path)}")

    # Index-derived filenames inside one scratch directory: unique by
    # construction, so no two texts can collide on a key even if two carry
    # identical prose.
    names = [f"{i:08d}.md" for i in range(len(texts))]
    try:
        with tempfile.TemporaryDirectory(
            prefix="stylebot-vale-", ignore_cleanup_errors=True
        ) as tmpdir:
            paths = []
            for name, text in zip(names, texts, strict=True):
                path = Path(tmpdir) / name
                path.write_text(text, encoding="utf-8")
                paths.append(str(path))
            proc = subprocess.run(
                [*cmd, *paths],
                capture_output=True,
                text=True,
                check=False,
            )
    except OSError as exc:  # vale present but unexecutable, or the scratch dir failed
        return [_vale_unavailable(f"vale failed to run: {exc}") for _ in texts]

    out = (proc.stdout or "").strip()
    if not out:
        # No JSON at all — usually a config/runtime error printed on stderr.
        reason = (proc.stderr or "vale produced no output").strip()
        return [_vale_unavailable(reason) for _ in texts]

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return [_vale_unavailable("vale output was not JSON") for _ in texts]

    # Vale emits a runtime-error *object* (e.g. E100 "no config file found")
    # instead of the normal {filename: [alerts]} map. Treat that as unavailable —
    # it is a whole-run failure, so every text in the batch degrades.
    if isinstance(parsed, dict) and parsed.get("Code"):
        reason = str(parsed.get("Text", parsed["Code"])).strip()
        return [_vale_unavailable(reason) for _ in texts]

    by_name: dict[str, list[dict]] = {}
    if isinstance(parsed, dict):
        for key, file_alerts in parsed.items():
            if isinstance(file_alerts, list):
                by_name.setdefault(_basename(key), []).extend(
                    a for a in file_alerts if isinstance(a, dict)
                )
    # A clean file is OMITTED from vale's map, not present with an empty list —
    # so a missing key means zero alerts, not a missing result.
    return [_vale_summarize(by_name.get(name, [])) for name in names]


def vale_score_many(texts: Sequence[str], *, config_path: str | Path | None = None) -> list[dict]:
    """Score many texts with as few `vale` processes as possible.

    The batch form of `vale_score`, and the one the corpus scorer uses: process
    startup dominates a per-text shell-out, and a full corpus run scores ~2 texts
    per pair. Vale accepts many paths per invocation and answers with one
    ``{filename: [alerts]}`` map, so this writes the batch into a single scratch
    directory (index-derived ``*.md`` names — unique by construction, so no two
    texts collide on a key) and chunks it across invocations of at most
    `_VALE_FILES_PER_RUN` files.

    Returns one dict per input text, **in input order**, each shaped exactly as
    `vale_score` returns. Never raises: a whole-run failure (vale absent,
    unexecutable, silent, non-JSON, or reporting its own ``E100``) degrades every
    text in that chunk to ``{"available": False, "reason": …}`` — the list is
    always as long as `texts`.
    """
    texts = list(texts)
    if not texts:
        return []
    if shutil.which("vale") is None:
        return [_vale_unavailable("vale not on PATH") for _ in texts]

    results: list[dict] = []
    for start in range(0, len(texts), _VALE_FILES_PER_RUN):
        results.extend(
            _vale_run(texts[start : start + _VALE_FILES_PER_RUN], config_path=config_path)
        )
    return results


def vale_score(text: str, *, config_path: str | Path | None = None) -> dict:
    """Score `text` with the `vale` binary; return a JSON-able alert summary.

    Shells out to ``vale --output=JSON`` over the text written to a temporary
    ``*.md`` file (NOT stdin). Vale applies glob-scoped styles by matching the
    *filename*, and its synthetic stdin filename matches no glob, so a config
    like ``[*.{md,qmd}] BasedOnStyles = …`` silently never runs over stdin and
    every alert count comes back 0. `config_path` points at a Vale config
    (`.vale.ini`); it is an explicit *parameter* — never a hardcoded blog
    ruleset. If `vale` is not on PATH, or no usable config is available, this
    returns ``{"available": False, ...}`` and never raises, so a run without Vale
    still completes.

    Returns ``{"available", "alerts", "by_severity", "raw"}``, where `alerts` is
    the total (0 when unavailable) and `raw` is vale's per-alert objects.

    A batch of one, over `vale_score_many` — which is what you want whenever you
    have more than one text, since this costs a whole process.
    """
    return vale_score_many([text], config_path=config_path)[0]


# ---------------------------------------------------------------------------
# Signal 2 — LLM-as-judge ("is this Dan-shaped / free of AI-slop tells")
# ---------------------------------------------------------------------------

# Callable contract: prose -> {"score": int, "rationale": str}. Injectable so
# tests pass a fake (no API). `openrouter_judge` builds the real one.
Judge = Callable[[str], dict]

JUDGE_SYSTEM = (
    "You are a discerning editor judging whether a passage reads like Dan "
    "Mackinlay's own prose, or like generic AI-assisted writing ('AI slop'). "
    "Dan's voice is dry, precise, idiosyncratic, and unhedged; it avoids "
    "throat-clearing signposts, indefinite 'you', empty intensifiers, generic "
    "cross-references, and the uniform rhythm of polished filler. "
    "Score the passage from 1 to 5: "
    "1 = unmistakable AI slop; "
    "3 = plausible but bland; "
    "5 = unmistakably Dan-shaped, free of slop tells. "
    "Reply with ONLY a JSON object of the form "
    '{"score": <int 1-5>, "rationale": "<one short sentence>"}. '
    "Do not wrap it in code fences or add any other text."
)


def parse_judge_reply(reply: str) -> dict:
    """Extract ``{"score": int, "rationale": str}`` from a model reply.

    Robust to extra prose around the JSON object and to a missing/invalid
    score (clamped into 1-5; falls back to a mid score with the raw reply as
    rationale if nothing parseable is found).
    """
    obj = _first_json_object(reply)
    if obj is not None:
        score = obj.get("score")
        try:
            score_int = int(score)
        except (TypeError, ValueError):
            score_int = None
        if score_int is not None:
            score_int = max(1, min(5, score_int))
            rationale = obj.get("rationale")
            return {
                "score": score_int,
                "rationale": str(rationale) if rationale is not None else "",
            }

    # Last-ditch: a bare integer somewhere in the reply.
    m = re.search(r"\b([1-5])\b", reply)
    if m:
        return {"score": int(m.group(1)), "rationale": reply.strip()}
    return {"score": 3, "rationale": reply.strip()}


def _first_json_object(text: str) -> dict | None:
    """Return the first brace-balanced JSON object decodable from `text`."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # try the next opening brace
                    if isinstance(obj, dict):
                        return obj
                    break
        start = text.find("{", start + 1)
    return None


def openrouter_judge(
    *,
    model: str = "anthropic/claude-opus-4-8",
    api_key: str | None = None,
    base_url: str | None = None,
    system: str = JUDGE_SYSTEM,
    max_tokens: int = 256,
) -> Judge:
    """Build a `Judge` backed by OpenRouter via the OpenAI-compatible SDK.

    Keys/paths resolve through `stylebot.config`: `OPENROUTER_API_KEY` is
    required (via `config.require_key`); `OPENROUTER_BASE_URL` overrides the
    default endpoint (via `config.get_key`). Nothing here prints a key.
    """
    import openai

    from stylebot import config

    resolved_base = (
        base_url or config.get_key("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    )
    client = openai.OpenAI(
        api_key=api_key or config.require_key("OPENROUTER_API_KEY"),
        base_url=resolved_base,
    )

    def judge(prose: str) -> dict:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prose},
            ],
        )
        reply = resp.choices[0].message.content or ""
        verdict = parse_judge_reply(reply)
        verdict["model"] = model
        return verdict

    return judge


# ---------------------------------------------------------------------------
# Signal 3 — statistical detector (pluggable; null by default)
# ---------------------------------------------------------------------------


# Callable contract: prose -> {"score": float|None, "name": str, ...}, where a
# higher score means *more AI-like*. The default `null_detector` is unconfigured.
class Detector(Protocol):
    def __call__(self, prose: str) -> dict: ...


def null_detector(prose: str) -> dict:
    """The default detector: scores nothing, signals it is unconfigured.

    Returns ``score=None`` so the rest of the harness composes cleanly without
    pretending to a signal it does not have. Wiring a real one is the operator's
    call — see `_plans/eval-harness.md` "The detector decision".
    """
    return {"score": None, "name": "null", "configured": False}


# ---------------------------------------------------------------------------
# Orchestration + schema
# ---------------------------------------------------------------------------


def score_candidate(
    text: str,
    *,
    judge: Judge | None = None,
    detector: Detector = null_detector,
    vale_config: str | Path | None = None,
    eyeball: str | None = None,
    vale: dict | None = None,
) -> dict:
    """Score one candidate passage across the available signals.

    `judge=None` skips the LLM signal (the harness then runs with no API key);
    its field comes back ``None``. `eyeball` is an optional human note carried
    verbatim into the record.

    `vale` injects an already-computed Vale result (from `vale_score_many`)
    instead of spending a process here; leave it `None` — the default — and this
    calls `vale_score` itself exactly as before.

    Returns a stable, JSON-able record keyed
    ``text_chars`` / ``vale`` / ``judge`` / ``detector`` / ``eyeball``.
    """
    return {
        "text_chars": len(text),
        "vale": vale if vale is not None else vale_score(text, config_path=vale_config),
        "judge": judge(text) if judge is not None else None,
        "detector": detector(text),
        "eyeball": eyeball,
    }


def _mean_or_none(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def _aggregate(records: list[dict]) -> dict:
    """Per-group aggregates over a list of `score_candidate` records."""
    judge_scores = [
        r["judge"]["score"]
        for r in records
        if isinstance(r.get("judge"), dict) and r["judge"].get("score") is not None
    ]
    vale_alerts = [r["vale"]["alerts"] for r in records if r.get("vale", {}).get("available")]
    detector_scores = [
        r["detector"]["score"]
        for r in records
        if isinstance(r.get("detector"), dict) and r["detector"].get("score") is not None
    ]
    return {
        "n": len(records),
        "mean_judge_score": _mean_or_none(judge_scores),
        "mean_vale_alerts": _mean_or_none(vale_alerts),
        "mean_detector_score": _mean_or_none(detector_scores),
    }


# ---------------------------------------------------------------------------
# Batched, JSONL-native scoring over a pairs.jsonl corpus
# ---------------------------------------------------------------------------
#
# Eval scores named *fields* per corpus row, emitting one id-keyed score record
# per row so scores join back to the corpus, stay idempotent/resumable, and can
# be faceted. Batching + extraction + aggregation over the primitives above.


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

    `scores.jsonl` is append-only and joins to `pairs.jsonl` on `record_id`, so
    re-cutting a capture or regenerating the corpus strands the superseded row
    there forever. No consumer checks the join, so orphans are counted as pairs:
    `summarize_scores` inflates n and skews every mean, and `covariate_table`
    emits a phantom row per superseded id into a CSV that gets committed.
    Pruning at the source keeps every consumer consistent for free.

    Undecodable lines are round-tripped verbatim rather than dropped (see
    `jsonl.write_jsonl_atomic`); a decodable record with no `id` has no join
    key and is pruned. Rewrites `out_path` atomically, and ONLY when something
    is actually dropped — an already-clean file is left untouched.
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


@dataclass
class ScoreResult:
    """Outcome of a `score_pairs_file` run."""

    written: int = 0
    skipped_existing: int = 0
    pruned_orphans: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (id, message)


# Records per slice of the scoring loop. Each slice takes one batched Vale
# pre-pass then one thread-pool drain, so this is the granularity at which
# results reach the file: bigger amortises `vale` startup further, smaller keeps
# a crashed run's lost work smaller. `fields` multiplies it into texts per batch.
_SCORE_BATCH_RECORDS = 128


def _score_record(
    record: dict,
    *,
    fields: Sequence[str],
    judge: Judge | None,
    detector: Detector,
    vale_config: str | Path | None,
    vale_by_field: dict[str, dict] | None = None,
) -> dict:
    """Score the requested fields of one pair record -> a `scores.jsonl` line.

    `vale_by_field` carries the batched pre-pass's results (field name -> Vale
    record); a field missing from it falls back to scoring Vale inline.
    """
    meta = record.get("meta") or {}
    scores: dict[str, dict] = {}
    for name in fields:
        text = FIELD_EXTRACTORS[name](record)
        scores[name] = score_candidate(
            text,
            judge=judge,
            detector=detector,
            vale_config=vale_config,
            vale=(vale_by_field or {}).get(name),
        )
    gen = meta.get("gen") or {}
    carried = {k: meta.get(k) for k in _CARRIED_META if k in meta}
    carried.update({k: gen[k] for k in _CARRIED_GEN if k in gen})
    return {
        "id": record_id(record),
        "meta": carried,
        "scores": scores,
    }


def _vale_prepass(
    records: Sequence[dict],
    *,
    fields: Sequence[str],
    vale_config: str | Path | None,
) -> list[dict[str, dict]]:
    """Batch-score every (record, field) text with Vale -> per-record results.

    Returns one ``{field: vale_record}`` dict per input record, aligned to
    `records`. Never raises and never fails a record: an extraction that blows up
    is simply left out (the worker thread hits the same error and `ScoreResult`
    captures it per record, as it always did), and a Vale batch failure degrades
    the affected texts to ``available: False``.
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


def score_pairs_file(
    pairs_path: str | Path,
    out_path: str | Path,
    *,
    fields: Sequence[str] = ("slop", "target"),
    judge: Judge | None = None,
    detector: Detector = null_detector,
    vale_config: str | Path | None = None,
    max_workers: int = 8,
    limit: int | None = None,
    prune_orphans: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> ScoreResult:
    """Score a `pairs.jsonl` corpus -> an id-keyed `scores.jsonl` (batched).

    For each pair, score the named `fields` (default the slop and Dan sides) and
    append one score record keyed by `record_id`, so scores join back to the
    corpus. Idempotent + resumable: ids already in `out_path` are skipped.

    Records are processed in slices of `_SCORE_BATCH_RECORDS`. Each slice runs a
    single batched **Vale pre-pass** (`vale_score_many` — one `vale` process per
    ~256 texts rather than one per text, which on the live corpus would be ~9k
    processes and dominated by startup cost), then fans the judge /
    detector out over `max_workers` threads with each record's Vale result
    injected. A single writer appends results as they complete and flushes per
    line, so the file still grows during the run and a crash still costs at most
    the slice in flight. A field/judge error on one record is captured in
    `ScoreResult.errors` and never aborts the run; a Vale batch failure degrades
    to ``available: False`` rather than failing the records.

    On entry, `out_path` is pruned of records whose pair no longer exists
    (`prune_orphan_scores`; count in `ScoreResult.pruned_orphans` — surface it,
    this rewrites a file the caller may have committed). `limit` does not
    narrow that: the live id set is the WHOLE corpus regardless of how many
    pairs this run scores. Set `prune_orphans=False` when `out_path` belongs to
    a *different* corpus, where every record would read as orphaned.
    """
    pairs_path = Path(pairs_path)
    out_path = Path(out_path)

    unknown = [f for f in fields if f not in FIELD_EXTRACTORS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; known: {sorted(FIELD_EXTRACTORS)}")

    # Prune before reading `seen`: an orphan id would otherwise both survive in
    # the file and count as already-scored.
    pruned = prune_orphan_scores(pairs_path, out_path) if prune_orphans else 0

    seen = existing_scored_ids(out_path)
    todo: list[dict] = []
    skipped = 0
    for rec in iter_pairs(pairs_path):
        if record_id(rec) in seen:
            skipped += 1
            continue
        todo.append(rec)
        if limit is not None and len(todo) >= limit:
            break

    result = ScoreResult(skipped_existing=skipped, pruned_orphans=pruned)
    if not todo:
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(todo)
    done = 0
    batch_size = max(1, _SCORE_BATCH_RECORDS)
    with out_path.open("a", encoding="utf-8") as fp:
        for start in range(0, total, batch_size):
            batch = todo[start : start + batch_size]
            # One `vale` run for the whole slice, up front, on this thread.
            vale_results = _vale_prepass(batch, fields=fields, vale_config=vale_config)
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
                futures = {
                    pool.submit(
                        _score_record,
                        rec,
                        fields=fields,
                        judge=judge,
                        detector=detector,
                        vale_config=vale_config,
                        vale_by_field=vale_results[i],
                    ): rec
                    for i, rec in enumerate(batch)
                }
                for fut in as_completed(futures):
                    done += 1
                    if on_progress is not None:
                        on_progress(done, total)
                    rec = futures[fut]
                    try:
                        line = fut.result()
                    except Exception as exc:  # a field/judge error on one record — log, continue
                        result.errors.append((record_id(rec), f"{type(exc).__name__}: {exc}"))
                        continue
                    fp.write(dumps_line(line) + "\n")
                    fp.flush()
                    result.written += 1
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


def summarize_scores(scores: str | Path | Iterable[dict], *, by: str | None = None) -> dict:
    """Aggregate a `scores.jsonl` (path or records) per field, optionally faceted.

    Fields-across-rows are the "groups" the styler is judged by movement across
    (slop vs target vs output). `by` additionally facets by a carried `meta` key
    — `by="slop_strategy"` gives per-strategy means.
    """
    records = load_scores(scores)
    summary: dict = {
        "schema_version": SCHEMA_VERSION,
        "pairs": len(records),
        "fields": _aggregate_fields(records),
    }
    if by is not None:
        buckets: dict[str, list[dict]] = {}
        for rec in records:
            key = (rec.get("meta") or {}).get(by)
            buckets.setdefault("null" if key is None else str(key), []).append(rec)
        summary["by"] = {k: _aggregate_fields(v) for k, v in sorted(buckets.items())}
    return summary


# --- the publishable artifact -------------------------------------------------
# A scores.jsonl nests and is mostly Vale's `raw` alert objects, so it is not a
# shape you can commit next to a blog post. `covariate_table` projects it to one
# tidy row per pair with no prose text — small enough to version, and sufficient
# to redraw every figure.


def _flatten_scores(scores: dict) -> dict:
    """`{"slop": {"detector": {"score": .4}}}` -> `{"slop.detector.score": .4}`.

    Scalars only. Nested payloads (notably Vale's per-alert `raw` list) are
    dropped: they are the bulk of the file and none of its analysable signal.
    """
    flat: dict = {}
    for field_name, signals in (scores or {}).items():
        for signal, value in (signals or {}).items():
            if value is None or isinstance(value, (str, int, float, bool)):
                flat[f"{field_name}.{signal}"] = value
            elif isinstance(value, dict):
                for key, inner in value.items():
                    if inner is None or isinstance(inner, (str, int, float, bool)):
                        flat[f"{field_name}.{signal}.{key}"] = inner
    return flat


def covariate_table(
    scores: str | Path | Iterable[dict], *, drop_constant: bool = True
) -> tuple[list[dict], dict]:
    """Project a `scores.jsonl` to tidy per-pair rows -> `(rows, constants)`.

    Each row is `id` + the carried `meta` covariates + one column per scalar
    signal. No prose text: the passages are ~95% of a corpus's bytes, contribute
    to no figure, and for a synthetic sweep are regenerable besides.

    `drop_constant` hoists columns that never vary into the returned
    `constants` dict rather than repeating them on every row — an embedder id
    or a detector name describes the *run*, not the pair, and that repetition
    is ~40% of the file on a real corpus.
    """
    records = load_scores(scores)
    rows: list[dict] = []
    for rec in records:
        row: dict = {"id": rec.get("id")}
        row.update(rec.get("meta") or {})
        row.update(_flatten_scores(rec.get("scores") or {}))
        rows.append(row)

    # Column order: id, generation covariates in schema order, then the rest
    # alphabetically — design columns before signal columns.
    seen = {k for row in rows for k in row}
    ordered = ["id", *(k for k in GEN_FACET_KEYS if k in seen)]
    ordered += sorted(seen - set(ordered))

    constants: dict = {}
    if drop_constant:
        for col in list(ordered):
            if col == "id":
                continue
            values = {row.get(col) for row in rows}
            if len(values) == 1 and len(rows) > 1:
                constants[col] = values.pop()
                ordered.remove(col)

    # Project every row onto the same column list, so a row missing a key
    # carries it as an explicit None: a rectangular CSV, not a ragged one.
    rows = [{k: row.get(k) for k in ordered} for row in rows]
    return rows, constants


def write_covariate_table(
    scores: str | Path,
    out_path: str | Path,
    *,
    extra: dict | None = None,
) -> dict:
    """Write the tidy table as CSV plus a `<stem>.meta.json` provenance sidecar.

    The sidecar records the hoisted constants, the row/column counts, and a
    digest of the scores file the table came from, so a future reader can tell
    whether a re-run is comparable or merely similar. `extra` carries the
    caller's experiment config (model list, splits role, sample fraction).
    """
    out_path = Path(out_path)
    rows, constants = covariate_table(scores)
    if not rows:
        raise ValueError(f"no score records in {scores} — nothing to export")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    digest = file_sha256(scores) if Path(scores).exists() else None
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "source": str(scores),
        "source_sha256": digest,
        "rows": len(rows),
        "columns": columns,
        "constant_columns": constants,
        **({"experiment": extra} if extra else {}),
    }
    sidecar_path = out_path.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "csv": out_path,
        "sidecar": sidecar_path,
        "rows": len(rows),
        "columns": columns,
        "constants": constants,
    }
