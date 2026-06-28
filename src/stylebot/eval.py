"""Eval harness — offline scoring of candidate prose against the four signals.

This is the ground-truth scorer every later phase reports against. It runs
**offline**: it scores candidate output and is *not* wired into the trainer and
*not* baked into the served adapter. See `_plans/eval-harness.md` for the spec.

Library-first: these are typed functions over explicit prose strings / file
lists. The `ai-style eval` CLI is a thin wrapper the caller builds on top; this
module never imports a CLI and never imports `livingthing`.

The four signals (`_plans/eval-harness.md` "The four signals"):

1. **Vale** — mechanical slop (banned words, indefinite "you", -ize spelling).
   Shelled out to the `vale` binary; degrades gracefully if it is absent.
2. **LLM-as-judge** — "is this Dan-shaped" at the voice level. An injectable
   `Judge` callable; `openrouter_judge` builds a real one on the `openai` SDK.
3. **Statistical detector** — "would an external classifier flag this". A
   pluggable `Detector` seam, defaulting to `null_detector` (no model wired).
   The detector *audition* (Binoculars / RADAR / Ghostbuster / Pangram) is a
   deliberate later decision — see `_plans/eval-harness.md` "The detector
   decision". We do NOT download or run any detector here.
4. **Dan's eyeball** — the veto channel, not automatable: an optional
   `eyeball` string carried alongside the machine scores.

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

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Protocol

from stylebot.pairs import iter_pairs

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Signal 1 — Vale (mechanical slop)
# ---------------------------------------------------------------------------


def vale_score(text: str, *, config_path: str | Path | None = None) -> dict:
    """Score `text` with the `vale` binary; return a JSON-able alert summary.

    Shells out to ``vale --output=JSON`` over the text on stdin. `config_path`
    points at a Vale config (`.vale.ini`); it is an explicit *parameter* — never
    a hardcoded blog ruleset. If `vale` is not on PATH, or no usable config is
    available, this returns ``{"available": False, ...}`` and never raises, so a
    run without Vale still completes.

    Returns::

        {
          "available": bool,
          "alerts": int,                       # total alerts (0 when unavailable)
          "by_severity": {"error": int, "warning": int, "suggestion": int},
          "raw": [ {alert}, ... ],             # vale's per-alert objects
        }
    """
    empty = {
        "available": False,
        "alerts": 0,
        "by_severity": {"error": 0, "warning": 0, "suggestion": 0},
        "raw": [],
    }
    if shutil.which("vale") is None:
        return {**empty, "reason": "vale not on PATH"}

    cmd = ["vale", "--output=JSON"]
    if config_path is not None:
        cmd.append(f"--config={Path(config_path)}")
    # `--ext` tells Vale how to parse stdin (markdown), matching our prose.
    cmd += ["--ext=.md", "-"]

    try:
        proc = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # pragma: no cover - vale present but unexecutable
        return {**empty, "reason": f"vale failed to run: {exc}"}

    out = (proc.stdout or "").strip()
    if not out:
        # No JSON at all — usually a config/runtime error printed on stderr.
        return {**empty, "reason": (proc.stderr or "vale produced no output").strip()}

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return {**empty, "reason": "vale output was not JSON"}

    # Vale emits a runtime-error *object* (e.g. E100 "no config file found")
    # instead of the normal {filename: [alerts]} map. Treat that as unavailable.
    if isinstance(parsed, dict) and parsed.get("Code"):
        return {**empty, "reason": str(parsed.get("Text", parsed["Code"])).strip()}

    alerts: list[dict] = []
    if isinstance(parsed, dict):
        for file_alerts in parsed.values():
            if isinstance(file_alerts, list):
                alerts.extend(a for a in file_alerts if isinstance(a, dict))

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

    Mirrors `synth.openai_generator` (lazy `import openai`,
    `client.chat.completions.create`) but stays independent of `synth.py`.

    Keys/paths resolve through `stylebot.config`: `OPENROUTER_API_KEY` is
    required (via `config.require_key`); `OPENROUTER_BASE_URL` overrides the
    default endpoint (via `config.get_key`). Nothing here prints a key.
    """
    import openai

    from stylebot import config

    resolved_base = base_url or config.get_key("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
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

    Wiring a real statistical detector (Binoculars / RADAR / Ghostbuster /
    Pangram) is a deliberate later decision that is GPU/$$-heavy and needs the
    operator's go-ahead — see `_plans/eval-harness.md` "The detector decision".
    Until that audition is run, the seam stays here returning ``score=None`` so
    the rest of the harness composes cleanly without pretending to a signal it
    does not have.
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
) -> dict:
    """Score one candidate passage across the available signals.

    `judge=None` skips the LLM signal (the harness then runs with no API key);
    its field comes back ``None``. `detector` defaults to `null_detector`.
    `eyeball` is an optional human note carried verbatim into the record.

    Returns a stable, JSON-able record::

        {
          "text_chars": int,
          "vale": {available, alerts, by_severity, raw, ...},
          "judge": {"score": int, "rationale": str, ...} | None,
          "detector": {"score": float|None, "name": str, ...},
          "eyeball": str | None,
        }
    """
    return {
        "text_chars": len(text),
        "vale": vale_score(text, config_path=vale_config),
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
        r["judge"]["score"] for r in records if isinstance(r.get("judge"), dict) and r["judge"].get("score") is not None
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


def evaluate_groups(
    groups: dict[str, list[str]],
    *,
    judge: Judge | None = None,
    detector: Detector = null_detector,
    vale_config: str | Path | None = None,
) -> dict:
    """Score several named groups so a run is judged by MOVEMENT across them.

    The plan's three groups are ``styler-input`` (pre-styling slop),
    ``styler-output`` (the styler's rewrite), and ``pure-Dan`` (human
    reference); a successful styler moves ``styler-output`` toward ``pure-Dan``
    and away from ``styler-input``. Any named groups work — the function does
    not hardcode the blog's three.

    Returns a stable, JSON-able report::

        {
          "schema_version": 1,
          "groups": {
            "<name>": {
              "aggregate": {n, mean_judge_score, mean_vale_alerts,
                            mean_detector_score},
              "records": [ {score_candidate record}, ... ],
            },
            ...
          },
        }
    """
    out: dict = {"schema_version": SCHEMA_VERSION, "groups": {}}
    for name, texts in groups.items():
        records = [
            score_candidate(
                t,
                judge=judge,
                detector=detector,
                vale_config=vale_config,
            )
            for t in texts
        ]
        out["groups"][name] = {
            "aggregate": _aggregate(records),
            "records": records,
        }
    return out


def read_prose_files(paths: Iterable[str | Path]) -> list[str]:
    """Read a group of candidate prose files into a list of strings.

    Generic helper — no blog assumptions, no frontmatter handling, no glob: the
    caller hands in an explicit list of files (each is read whole as UTF-8).
    """
    return [Path(p).read_text(encoding="utf-8") for p in paths]


# ---------------------------------------------------------------------------
# Batched, JSONL-native scoring over a pairs.jsonl corpus
# ---------------------------------------------------------------------------
#
# The pipeline's lingua franca is `pairs.jsonl` (chat-completion records). Rather
# than one-prose-per-file, eval reads a corpus and scores named *fields* per row,
# emitting one id-keyed score record per row so scores join back to the corpus,
# stay idempotent/resumable (like `synth`), and can be faceted (e.g. by slop
# strategy). The per-text primitives above (`score_candidate` etc.) are reused
# verbatim; this layer is just batching + extraction + aggregation.


def pair_body(content: str, context: str | None) -> str:
    """The prose body of a pair message, with the heading-context prefix removed.

    `stylebot.pairs.build_pair_content(context, body)` prepends ``f"{context}\\n\\n"``
    *identically to both sides* of a pair when a heading context is present; this
    is its inverse, so the judge scores the transform region (the body), not the
    shared heading. No context (or content lacking the prefix) returns unchanged.
    """
    if not context:
        return content
    prefix = f"{context}\n\n"
    return content[len(prefix):] if content.startswith(prefix) else content


def _extract_slop(record: dict) -> str:
    return pair_body(record["messages"][1]["content"], (record.get("meta") or {}).get("context"))


def _extract_target(record: dict) -> str:
    return pair_body(record["messages"][2]["content"], (record.get("meta") or {}).get("context"))


# Field name -> (record -> body text). Defaults cover the pairs schema
# (messages[1]=slop/user, messages[2]=Dan/assistant). A Phase-4 styler-output
# JSONL adds an "output" extractor here; nothing else changes.
FIELD_EXTRACTORS: dict[str, Callable[[dict], str]] = {
    "slop": _extract_slop,
    "target": _extract_target,
}

# meta keys carried onto each score record, so scores group / filter / join
# downstream without re-reading the corpus.
_CARRIED_META = ("source", "synthetic", "generator", "slop_strategy", "chunk_index")


def record_id(record: dict) -> str:
    """A stable join key linking a pair's score back to the corpus.

    Prefers `meta.synth_key` (synthetic pairs); else `capture_id:chunk_index`
    (real captures group by `capture_id`); else a hash of `source` + chunk index.
    """
    meta = record.get("meta") or {}
    if meta.get("synth_key"):
        return str(meta["synth_key"])
    capture_id = meta.get("capture_id")
    chunk_index = meta.get("chunk_index")
    if capture_id is not None:
        return f"{capture_id}:{chunk_index}"
    h = hashlib.sha256(f"{meta.get('source')}\x00{chunk_index}".encode("utf-8"))
    return h.hexdigest()[:16]


def existing_scored_ids(out_path: str | Path) -> set[str]:
    """Read the `id`s already present in a `scores.jsonl` (for resumable runs)."""
    out_path = Path(out_path)
    ids: set[str] = set()
    if not out_path.exists():
        return ids
    with out_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id") if isinstance(rec, dict) else None
            if rid:
                ids.add(str(rid))
    return ids


@dataclass
class ScoreResult:
    """Outcome of a `score_pairs_file` run."""

    written: int = 0
    skipped_existing: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)  # (id, message)


def _score_record(
    record: dict,
    *,
    fields: Sequence[str],
    judge: Judge | None,
    detector: Detector,
    vale_config: str | Path | None,
) -> dict:
    """Score the requested fields of one pair record -> a `scores.jsonl` line."""
    meta = record.get("meta") or {}
    scores: dict[str, dict] = {}
    for name in fields:
        text = FIELD_EXTRACTORS[name](record)
        scores[name] = score_candidate(text, judge=judge, detector=detector, vale_config=vale_config)
    return {
        "id": record_id(record),
        "meta": {k: meta.get(k) for k in _CARRIED_META if k in meta},
        "scores": scores,
    }


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
    on_progress: Callable[[int, int], None] | None = None,
) -> ScoreResult:
    """Score a `pairs.jsonl` corpus -> an id-keyed `scores.jsonl` (batched).

    For each pair, score the named `fields` (default the slop and Dan sides) and
    append one score record keyed by `record_id`, so scores join back to the
    corpus. Idempotent + resumable: ids already in `out_path` are skipped, so
    re-running never duplicates and a crash resumes where it stopped.

    Scoring is concurrent (`max_workers` threads — the judge / Vale are blocking
    IO, and the `openai` SDK is sync); a single writer appends results as they
    complete and flushes per line (crash-safe). A field/judge error on one record
    is captured in `ScoreResult.errors` and never aborts the run. Pass
    `judge=None` to run keyless (Vale + null detector only).
    """
    pairs_path = Path(pairs_path)
    out_path = Path(out_path)

    unknown = [f for f in fields if f not in FIELD_EXTRACTORS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; known: {sorted(FIELD_EXTRACTORS)}")

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

    result = ScoreResult(skipped_existing=skipped)
    if not todo:
        return result

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(todo)
    done = 0
    with out_path.open("a", encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {
            pool.submit(
                _score_record, rec, fields=fields, judge=judge, detector=detector, vale_config=vale_config
            ): rec
            for rec in todo
        }
        for fut in as_completed(futures):
            done += 1
            if on_progress is not None:
                on_progress(done, total)
            rec = futures[fut]
            try:
                line = fut.result()
            except Exception as exc:  # a field/judge error on one record — record and continue
                result.errors.append((record_id(rec), f"{type(exc).__name__}: {exc}"))
                continue
            fp.write(json.dumps(line, ensure_ascii=False) + "\n")
            fp.flush()
            result.written += 1
    return result


def _load_score_records(scores: str | Path | Iterable[dict]) -> list[dict]:
    """Load score records from a `scores.jsonl` path or accept them in-memory."""
    if isinstance(scores, (str, Path)):
        path = Path(scores)
        out: list[dict] = []
        if path.exists():
            with path.open(encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        out.append(rec)
        return out
    return list(scores)


def _aggregate_fields(records: list[dict]) -> dict:
    """Per-field aggregates over score records (fields-across-rows = groups)."""
    field_names: list[str] = []
    for rec in records:
        for name in rec.get("scores") or {}:
            if name not in field_names:
                field_names.append(name)
    return {
        name: _aggregate([rec["scores"][name] for rec in records if name in (rec.get("scores") or {})])
        for name in field_names
    }


def summarize_scores(scores: str | Path | Iterable[dict], *, by: str | None = None) -> dict:
    """Aggregate a `scores.jsonl` (path or records) per field, optionally faceted.

    Fields-across-rows are the "groups" the styler is judged by movement across
    (slop vs target vs output). `by` additionally facets by a carried `meta` key
    — `by="slop_strategy"` gives per-strategy means, the experimental-loop view
    that turns "is catalogue slop better than polish" into a number.
    """
    records = _load_score_records(scores)
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
