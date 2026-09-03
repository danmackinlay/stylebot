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
import random
import re
import shutil
import subprocess
import tempfile
import time as _time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Protocol

# click is an unconditional dependency, and `VerdictOpts` below is a
# `cli_options` bundle for the same reason `train.Hyperparameters` is: the
# comparison flags have two CLIs (`ai-style eval` and the blog's mirror) and one
# declaration, so neither can silently drop one.
import click

# `classify` is ML-dependency-free at import (that is its whole design), so
# taking the detector-identity contract from it here costs nothing and keeps one
# definition of "which head scored this".
from stylebot.classify import DETECTOR_IDENTITY_KEYS
from stylebot.cli_options import opt
from stylebot.errors import ConfigError, RunError
from stylebot.jsonl import dumps_line, iter_jsonl, read_jsonl, write_jsonl_atomic

# The pair-text extractors and the score join key live in `stylebot.pairs`; this
# module imports the four it uses and re-exports nothing: every consumer
# (`infer`, `train`, `classify_train`, the scripts, the tests, livingthing)
# reaches them at `stylebot.pairs` directly, so a facade here would be a second
# import path for one surface.
from stylebot.pairs import FIELD_EXTRACTORS, GEN_FACET_KEYS, iter_pairs, record_id
from stylebot.provenance import file_sha256

# 3 (2026-09-03, W10): the summary gained `by_provenance` + `headline_facet` (the
# headline is the REAL facet) and `detectors` (which artifact produced the
# readings). Additive, but the headline convention changed, so consumers keying
# on the version should see it move.
SCHEMA_VERSION = 3

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
            # OpenRouter returns the billed cost on the usage object when asked
            # for it. Without this the judge is the one signal whose price is
            # unknowable after the fact — the detector and Vale are free and
            # local, so "is the judge worth it?" is a question only this field
            # can answer. Recorded per call rather than aggregated: a stratified
            # subsample is the standing recommendation, and a per-call figure is
            # what sizes one.
            extra_body={"usage": {"include": True}},
        )
        reply = resp.choices[0].message.content or ""
        verdict = parse_judge_reply(reply)
        verdict["model"] = model
        usage = getattr(resp, "usage", None)
        if usage is not None:
            verdict["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                # Absent on providers that do not report it, and — per the blog's
                # BYOK note — potentially the 5% fee rather than list price when
                # the arm bills to a bring-your-own key. Treat as an indication.
                "cost": getattr(usage, "cost", None),
            }
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


null_detector.identity = {key: None for key in DETECTOR_IDENTITY_KEYS} | {"name": "null"}


#: Pangram's async task API base. Overridable so the one-shot cross-check can be
#: pointed at a mock or a future path without editing this module.
#:
#: There is no synchronous `/predict` any more: Pangram 3.0 replaced it with
#: `POST /task` -> task id -> `GET /task/{id}` polled until a terminal stage.
#: The old URL this module shipped answered 404, which nothing noticed because
#: the detector had only ever been exercised through its `post=` seam. That is
#: the argument for the live-call test below, not for a better stub.
PANGRAM_URL = "https://text.external-api.pangram.com/task"

#: Terminal stages of a Pangram task; anything else means keep polling.
PANGRAM_DONE = "STAGE_SUCCESS"
PANGRAM_FAILED = "STAGE_FAILED"


def pangram_detector(
    *,
    api_key: str | None = None,
    url: str = PANGRAM_URL,
    model: str = "pangram-4",
    timeout: float = 30.0,
    poll_interval: float = 1.0,
    poll_timeout: float = 180.0,
    post: Callable[[str, dict, dict], dict] | None = None,
    fetch: Callable[[str, dict], dict] | None = None,
) -> Detector:
    """The **optional, one-shot** independent cross-check (`_plans/eval-harness.md`).

    Pangram is a *different model family* with different blind spots, which is
    the entire point: the voice classifier cannot self-certify that a styler
    lowered AI-ness rather than learning its embedding's blind spots. It is a
    paid API (~$0.05/1k words, `PANGRAM_API_KEY`) and it must **never** be the
    default detector and **never** run in a hot loop — best-of-N reranking or a
    whole-corpus pass would spend real money per draw. Use it once, over a
    bounded sample, to confirm the free detector's direction; the judge and
    Dan's eyeball supply the same orthogonality more cheaply the rest of the
    time.

    Polarity matches the house convention: ``score = fraction_ai`` (higher =
    more AI-like), so it drops into `mean_detector_score` unchanged. `post` is
    the transport seam — ``(url, payload, headers) -> response dict`` — so tests
    (and a dry run) exercise everything but the spend.

    Networking is `urllib` from the stdlib rather than the `openai` SDK: this is
    a plain JSON POST to a non-OpenAI endpoint, and eval must stay importable
    with no SDK installed.
    """
    from stylebot import config

    key = api_key or config.require_key("PANGRAM_API_KEY")
    transport = post or _urllib_post_json

    def detect(prose: str) -> dict:
        payload = {"text": prose, "model": model}
        headers = {"Content-Type": "application/json", "x-api-key": key}
        body = transport(url, payload, headers)
        # `POST /task` answers with an id, not a verdict. Poll the task until a
        # terminal stage; `fetch` is the GET seam, so a fake `post` that already
        # returns a scored body (the shape the tests use) still short-circuits
        # here rather than needing a second stub.
        task_id = body.get("task_id") or body.get("id")
        if task_id and "fraction_ai" not in body:
            deadline = _time.monotonic() + poll_timeout
            get = fetch or _urllib_get_json
            while True:
                body = get(f"{url}/{task_id}", headers)
                stage = body.get("stage")
                if stage == PANGRAM_DONE:
                    break
                if stage == PANGRAM_FAILED:
                    raise RunError(f"pangram task {task_id} failed: {body.get('error') or stage}")
                if _time.monotonic() > deadline:
                    raise RunError(
                        f"pangram task {task_id} still {stage} after {poll_timeout:.0f}s"
                    )
                _time.sleep(poll_interval)
        # `fraction_ai` is the documented field; `ai_likelihood` is what some
        # responses call it. Neither present = no signal, said out loud rather
        # than defaulted to 0.0, which would read as "certainly human".
        score = body.get("fraction_ai")
        if score is None:
            score = body.get("ai_likelihood")
        return {
            "score": float(score) if score is not None else None,
            "name": "pangram",
            "configured": True,
            "embed_model": None,
            "git_sha": None,
            "trained_at": None,
            "n_pairs": None,
            "prediction": body.get("prediction"),
        }

    detect.identity = {key_: None for key_ in DETECTOR_IDENTITY_KEYS} | {"name": "pangram"}
    return detect


def _urllib_get_json(url: str, headers: dict, *, timeout: float = 30.0) -> dict:
    import urllib.request

    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read().decode("utf-8"))


def _urllib_post_json(url: str, payload: dict, headers: dict, *, timeout: float = 30.0) -> dict:
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return json.loads(resp.read().decode("utf-8"))


def detector_identity(detector: Detector) -> dict:
    """Which head is this — `DETECTOR_IDENTITY_KEYS`, without spending an embed.

    Reads the ``.identity`` attribute `classify.embedding_classifier_detector`
    attaches. A detector that carries none (a test stub, a hand-rolled
    callable) answers all-None with its ``__name__`` for `name`, so callers get
    a comparable shape from anything satisfying the protocol.
    """
    identity = getattr(detector, "identity", None)
    if isinstance(identity, dict):
        return {key: identity.get(key) for key in DETECTOR_IDENTITY_KEYS}
    fallback = {key: None for key in DETECTOR_IDENTITY_KEYS}
    fallback["name"] = getattr(detector, "__name__", None)
    return fallback


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

    `allow_mixed=True` downgrades to "return them all" for the deliberate case
    (a file being migrated, a diagnostic). Everything that *compares* numbers
    goes through here first, because a refit head moves P(slop) by more than the
    effects being measured — run 4a's headline moved 0.038 while its detector was
    refit the same day.
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
    allow_mixed_detector: bool = False,
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

    Resume is keyed on the pair id and says nothing about *what did the
    scoring*, so appending with a refit head silently yields a file whose means
    blend two instruments — a bigger effect than most of what gets measured
    across them. On entry the existing rows' detector identity is compared with
    this run's; a mismatch raises `ValueError` unless `allow_mixed_detector`.
    """
    pairs_path = Path(pairs_path)
    out_path = Path(out_path)

    unknown = [f for f in fields if f not in FIELD_EXTRACTORS]
    if unknown:
        raise ConfigError(f"unknown field(s) {unknown}; known: {sorted(FIELD_EXTRACTORS)}")

    # Prune before reading `seen`: an orphan id would otherwise both survive in
    # the file and count as already-scored.
    pruned = prune_orphan_scores(pairs_path, out_path) if prune_orphans else 0

    _guard_detector_continuity(out_path, detector, allow_mixed=allow_mixed_detector)

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


def split_by_provenance(records: Sequence[dict]) -> dict[str, list[dict]]:
    """Score records split `real` / `synthetic` on `meta.synthetic`.

    The headline number must come from the real facet. Synthetic pairs are our
    own generators' paraphrases: a verdict computed over them largely measures
    how well the styler undoes the slop model we happen to be running this
    month, and on the frozen eval stratum they are the *majority* of the rows
    (measured 2026-09-03: 62 of 103). Reporting them is fine; letting them into
    the headline is how "the adapter closes 90% of the gap" stops meaning
    anything about Dan's prose.
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
    (slop vs target vs output). `by` additionally facets by a carried `meta` key
    — `by="slop_strategy"` gives per-strategy means.

    `facet_provenance` (default on) adds `by_provenance.real` / `.synthetic` and
    names `headline_facet: "real"`, mirroring what the classifier trainer
    already does with `metrics.by_provenance`. It also records the file's
    `detector` identities, so a summary carries the instrument that produced it
    rather than only the reading.
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


# ---------------------------------------------------------------------------
# The verdict — paired comparison with a cluster bootstrap
# ---------------------------------------------------------------------------
#
# Two adapters scored on the same chunks are compared paired; the question is
# what the CI around the paired mean is allowed to assume. Treating chunks as
# independent assumes 127 independent observations when 77 of them come from one
# post (`notebook/automatic_coding.qmd`, re-measured 2026-09-03: 77 of 103 eval
# pairs, 75%). Chunks within a post share topic, register, and whatever the
# generator did to that post, so the effective sample size is nearer the 6 posts
# than the 103 chunks, and a chunk-level CI is too narrow by roughly the square
# root of the average cluster size. Resampling POSTS — not chunks — is the fix,
# and it is a dozen lines of stdlib.


#: How to pull one scalar out of a scored field. Vale reports `null` when the
#: binary is absent rather than 0, so an unavailable run drops out of the
#: comparison instead of scoring perfect.
SIGNAL_EXTRACTORS: dict[str, Callable[[dict], float | None]] = {
    "detector": lambda signals: (signals.get("detector") or {}).get("score"),
    "judge": lambda signals: (signals.get("judge") or {}).get("score"),
    "vale": lambda signals: (
        (signals.get("vale") or {}).get("alerts")
        if (signals.get("vale") or {}).get("available")
        else None
    ),
}


def paired_delta(
    a: Sequence[float | None],
    b: Sequence[float | None],
    *,
    clusters: Sequence[object] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict | None:
    """Mean of `a - b` with a confidence interval; clustered when asked.

    `a`, `b` and `clusters` are aligned; a position where either value is None
    drops out. With `clusters`, the CI is a **cluster bootstrap**: whole
    clusters are resampled with replacement `n_boot` times and the interval is
    the percentile interval of the resampled means — the honest interval when
    observations inside a cluster are not independent. Without them every
    observation is its own cluster, which is the ordinary paired bootstrap and
    reproduces (to Monte-Carlo error) the normal-approximation interval the
    ad-hoc run-4 script computed.

    Returns None when fewer than two usable differences remain; the interval is
    None when they all fall in one cluster — one post is not an interval.
    """
    clustered = clusters is not None
    if clusters is None:
        clusters = range(len(a))
    diffs: list[float] = []
    labels: list[object] = []
    for x, y, cluster in zip(a, b, clusters, strict=True):
        if x is None or y is None:
            continue
        diffs.append(float(x) - float(y))
        labels.append(cluster)
    if len(diffs) < 2:
        return None

    grouped: dict[object, list[float]] = {}
    for label, diff in zip(labels, diffs, strict=True):
        grouped.setdefault(label, []).append(diff)
    point = mean(diffs)
    base = {
        "mean": point,
        "n": len(diffs),
        "n_clusters": len(grouped),
    }

    if len(grouped) < 2:
        return {**base, "ci_low": None, "ci_high": None, "method": "single-cluster", "n_boot": 0}

    keys = list(grouped)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_boot):
        drawn: list[float] = []
        for _ in keys:
            drawn.extend(grouped[keys[rng.randrange(len(keys))]])
        if drawn:
            means.append(mean(drawn))
    means.sort()
    lo = means[max(0, int(alpha / 2 * len(means)) - 1)]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return {
        **base,
        "ci_low": lo,
        "ci_high": hi,
        "method": "cluster_bootstrap" if clustered else "bootstrap",
        "n_boot": len(means),
        "alpha": alpha,
    }


@dataclass(frozen=True)
class VerdictOpts:
    """The comparison surface, declared once for every CLI that offers it.

    `ai-style eval` and the blog's `dan-style eval` are the two today, and the
    mirror test (`tests/test_dan_style_cli.py`) asserts the second carries every
    flag of the first — so these are a `cli_options` bundle rather than six
    `click.option` lines retyped in two repos, the same reason `Hyperparameters`
    and `BackendOpts` are.
    """

    cli_section = "Verdict / comparison"

    allow_mixed_detector: bool = opt(
        False,
        "--allow-mixed-detector",
        is_flag=True,
        help="Append to a scores file written by a DIFFERENT detector artifact. Off by default: "
        "resume skips by pair id, so a refit head would blend two instruments into one mean.",
    )
    compare_path: Path | None = opt(
        None,
        "--compare",
        "compare_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="A second scores.jsonl to compare against, paired on pair id: prints the mean "
        "difference with a bootstrap CI clustered by POST. Refuses a cross-detector comparison.",
    )
    compare_field: str = opt(
        "output",
        "--compare-field",
        show_default=True,
        help="Which scored field --compare pits against the other file.",
    )
    compare_signal: str = opt(
        "detector",
        "--compare-signal",
        show_default=True,
        type=click.Choice(["detector", "judge", "vale"]),
        help="Which signal --compare reports.",
    )
    cluster_by: str = opt(
        "source",
        "--cluster-by",
        show_default=True,
        help="Meta key the --compare bootstrap resamples (the POST). Empty string = treat every "
        "chunk as independent, which over a concentrated stratum makes the CI too narrow.",
    )
    allow_cross_detector: bool = opt(
        False,
        "--allow-cross-detector",
        is_flag=True,
        help="Let --compare span two detector artifacts. A refit head moves P(slop) on its own, "
        "so the delta then measures the instrument as much as the adapter.",
    )


@dataclass(frozen=True)
class DocumentOpts:
    """The document-scale eval surface — `rewrite_documents_file`'s flags.

    Deliberately only four: the styler is configured entirely by the training
    manifest (`backends.opts_from_manifest`), because the whole point of
    document-scale eval is to exercise the diet the manifest records. A flag
    soup here would let you score a diet no adapter was trained on.
    """

    cli_section = "Document-scale eval"

    documents: bool = opt(
        False,
        "--documents",
        is_flag=True,
        help="Rewrite WHOLE capture-archive documents through the styler instead of scoring "
        "pre-cut chunks, then score the result. This is the only mode that exercises "
        "segmentation, protected blocks and the chunk diet — a diet switch cannot move a "
        "number computed on chunks that were cut before it. PAID: it samples the adapter.",
    )
    documents_manifest: Path | None = opt(
        None,
        "--documents-manifest",
        "documents_manifest",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Training manifest whose adapter and chunk diet --documents serves.",
    )
    documents_out: Path | None = opt(
        None,
        "--documents-out",
        "documents_out",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Where --documents writes its output JSONL. Default: documents.jsonl beside --out.",
    )
    documents_role: str = opt(
        "eval",
        "--documents-role",
        show_default=True,
        help="Restrict --documents to this splits role. 'all' takes every archive; anything but "
        "'eval' is measuring memorisation, since the styler trained on those posts.",
    )


VERDICT_PARAMS: dict[str, type] = {"verdict": VerdictOpts, "documents": DocumentOpts}


def verdict_options(**kw):
    """The shared verdict/comparison + document-scale flags.

    One decorator for both bundles: `ai-style eval` and the blog's `dan-style
    eval` mirror each other flag for flag (pinned by the blog's mirror test), so
    a surface split across two decorators is two chances to add one and forget
    the other. `defaults=`/`exclude=` as elsewhere.
    """
    from stylebot.cli_options import bundle_options

    return bundle_options([VerdictOpts, DocumentOpts], **kw)


def format_summary(summary: dict) -> list[str]:
    """The aggregate as printable lines, **headline facet first**.

    A renderer, not a printer, so `ai-style eval` and the blog's mirror show the
    same thing without either owning the layout. The headline is the REAL facet:
    on the frozen eval stratum most rows are our own generators' paraphrases (62
    of 103, measured 2026-09-03), and a mean over those says how well the styler
    undoes this month's slop model, not how Dan-shaped its prose is.
    """
    lines: list[str] = []

    def block(prefix: str, fields: dict) -> None:
        for name, agg in fields.items():
            lines.append(
                f"{prefix}{name}: n={agg['n']} judge={agg['mean_judge_score']} "
                f"vale_alerts={agg['mean_vale_alerts']} detector={agg['mean_detector_score']}"
            )

    facets = summary.get("by_provenance") or {}
    real, synthetic = facets.get("real") or {}, facets.get("synthetic") or {}
    if real.get("pairs") and synthetic.get("pairs"):
        lines.append(f"HEADLINE — real pairs only ({real['pairs']}):")
        block("  ", real["fields"])
        lines.append(f"synthetic pairs, reported separately ({synthetic['pairs']}):")
        block("  ", synthetic["fields"])
        lines.append(f"all pairs ({summary['pairs']}):")
        block("  ", summary["fields"])
    else:
        block("", summary["fields"])
    for identity in summary.get("detectors") or []:
        lines.append(f"detector: {_identity_label(identity)}")
    return lines


def format_verdict(verdict: dict, a_label: object = "a", b_label: object = "b") -> list[str]:
    """The paired comparison as printable lines, CI and cluster count included."""
    delta = verdict["delta"]
    lines = [
        f"verdict — {verdict['signal']} on '{verdict['field']}', "
        f"{verdict['provenance'] or 'all'} pairs, clustered by {verdict['cluster_by']}",
        f"  a = {a_label}  mean {verdict['mean_a']}",
        f"  b = {b_label}  mean {verdict['mean_b']}",
    ]
    if delta is None:
        lines.append("  not enough shared, scored ids to compare")
        return lines
    ci = (
        f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
        if delta.get("ci_low") is not None
        else "[no interval: one cluster]"
    )
    lines.append(
        f"  paired a-b: {delta['mean']:+.4f}  95% CI {ci}  "
        f"n={delta['n']} over {delta['n_clusters']} cluster(s) [{delta['method']}]"
        f"  -> {verdict['verdict']}"
    )
    if verdict["cross_detector"]:
        lines.append(
            "  WARNING: the two sides were scored by DIFFERENT detector artifacts; "
            "this delta measures the instrument as well as the adapter"
        )
    return lines


def _verdict_label(delta: dict | None, *, lower_is_better: bool) -> str:
    if delta is None or delta.get("ci_low") is None:
        return "undetermined"
    if delta["ci_high"] < 0:
        return "a better" if lower_is_better else "b better"
    if delta["ci_low"] > 0:
        return "b better" if lower_is_better else "a better"
    return "indistinguishable"


def compare_scores(
    scores_a: str | Path | Iterable[dict],
    scores_b: str | Path | Iterable[dict],
    *,
    field: str = "output",
    signal: str = "detector",
    cluster_by: str | None = "source",
    provenance: str | None = "real",
    n_boot: int = 2000,
    seed: int = 0,
    allow_cross_detector: bool = False,
) -> dict:
    """Paired verdict between two scored runs over the same ids.

    Joins on the score `id`, keeps the ids both files carry, and reports the
    paired difference with a bootstrap clustered on `meta[cluster_by]` (the
    POST, by default). `provenance` restricts the comparison to one facet —
    `"real"` by default, because the headline must not be computed over our own
    generators' paraphrases; pass None for all rows.

    **Refuses a cross-detector comparison.** Both files must have been written
    by the same detector artifact; a refit head moves P(slop) on its own, so a
    delta across two heads measures the instrument. `allow_cross_detector=True`
    says you mean it, and the returned dict records that you did.
    """
    if signal not in SIGNAL_EXTRACTORS:
        raise ConfigError(f"unknown signal {signal!r}; known: {sorted(SIGNAL_EXTRACTORS)}")
    rows_a = {str(r["id"]): r for r in load_scores(scores_a) if r.get("id") is not None}
    rows_b = {str(r["id"]): r for r in load_scores(scores_b) if r.get("id") is not None}

    ids_a = assert_single_detector(rows_a.values(), allow_mixed=allow_cross_detector)
    ids_b = assert_single_detector(rows_b.values(), allow_mixed=allow_cross_detector)
    if not allow_cross_detector and ids_a and ids_b and ids_a[0] != ids_b[0]:
        raise ConfigError(
            f"refusing to compare verdicts across detector artifacts: "
            f"{_identity_label(ids_a[0])} vs {_identity_label(ids_b[0])} — a refit head moves "
            f"P(slop) by more than most effects being measured. Re-score both sides with one "
            f"head, or pass allow_cross_detector=True / --allow-cross-detector"
        )

    extract = SIGNAL_EXTRACTORS[signal]
    shared = sorted(set(rows_a) & set(rows_b))
    a_vals: list[float | None] = []
    b_vals: list[float | None] = []
    clusters: list[object] = []
    for rid in shared:
        meta = rows_a[rid].get("meta") or {}
        facet = "synthetic" if meta.get("synthetic") else "real"
        if provenance is not None and facet != provenance:
            continue
        a_vals.append(extract((rows_a[rid].get("scores") or {}).get(field) or {}))
        b_vals.append(extract((rows_b[rid].get("scores") or {}).get(field) or {}))
        clusters.append(meta.get(cluster_by) if cluster_by else rid)

    delta = paired_delta(
        a_vals,
        b_vals,
        clusters=clusters if cluster_by else None,
        n_boot=n_boot,
        seed=seed,
    )
    usable_a = [v for v in a_vals if v is not None]
    usable_b = [v for v in b_vals if v is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "field": field,
        "signal": signal,
        "provenance": provenance,
        "cluster_by": cluster_by,
        "n_shared_ids": len(shared),
        "mean_a": mean(usable_a) if usable_a else None,
        "mean_b": mean(usable_b) if usable_b else None,
        "delta": delta,
        "verdict": _verdict_label(delta, lower_is_better=(signal != "judge")),
        "detector_a": ids_a[0] if ids_a else None,
        "detector_b": ids_b[0] if ids_b else None,
        "cross_detector": bool(allow_cross_detector and ids_a and ids_b and ids_a[0] != ids_b[0]),
    }


# ---------------------------------------------------------------------------
# Document-scale eval — whole posts through `infer.rewrite_text`
# ---------------------------------------------------------------------------
#
# Everything above scores PRE-CUT chunks: `infer.rewrite_pairs_file` hands
# `messages[:2]` straight to the styler, so the chunking that `rewrite_text`
# actually performs at serving time — segmentation, protected blocks, the
# heading-context rule, the cross-heading diet, the do-no-harm guard — is never
# exercised by the verdict. A diet switch like run 4a's (`inline_headings`,
# 4500-char budget) therefore cannot move the number that decides whether to
# promote it, which is the wrong way round.
#
# The material for a document-scale eval already exists: a capture archive holds
# `before.md` (the slop-bearing draft) and `after.md` (Dan's edit) as WHOLE
# documents. Rewriting `before.md` and scoring it against both is the same
# slop → output → Dan movement view, one post at a time.


@dataclass
class DocumentRewriteResult:
    """Outcome of a `rewrite_documents_file` run."""

    written: int = 0
    skipped_existing: int = 0
    n_chunks: int = 0
    n_kept_input: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def iter_document_pairs(pairs_path: str | Path, *, splits: dict | None = None, role: str = "eval"):
    """Whole-document (before, after) captures, optionally restricted to a role.

    Yields `{"capture_id", "source", "before", "after"}` per capture archive
    beside `pairs_path`. With `splits`, only archives whose post resolves to
    `role` are yielded — `role="eval"` being the frozen stratum, the only one a
    verdict may be computed on. `role=None` yields every archive.
    """
    from stylebot.capture import iter_archives

    for archive in iter_archives(Path(pairs_path)):
        source = archive.get("source")
        if splits is not None and role is not None:
            from stylebot import splits as splits_mod

            if splits_mod.role_of(source or "?", splits) != role:
                continue
        yield {
            "capture_id": archive.get("capture_id"),
            "source": source,
            "before": archive.get("before_text") or "",
            "after": archive.get("after_text") or "",
        }


def rewrite_documents_file(
    pairs_path: str | Path,
    out_path: str | Path,
    styler,
    *,
    splits: dict | None = None,
    role: str = "eval",
    limit: int | None = None,
    rewrite: Callable[..., object] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    **rewrite_kwargs,
) -> DocumentRewriteResult:
    """Rewrite whole eval-stratum documents; emit a scorable output JSONL.

    Each record is pair-shaped — `messages` = (frozen system, the before
    document, the after document) plus a top-level `output` — so
    `score_pairs_file(fields=("slop", "target", "output"))` scores it with no
    change, and `report.render_scores_report` renders it. `meta.capture_id` is
    prefixed `doc:` so a document row's `record_id` can never be mistaken for
    the chunk-level pair cut from the same capture.

    `rewrite` is the seam onto `infer.rewrite_text` — imported lazily, injected
    by tests as a stub so the whole path is exercised with no sampling spend.
    Remaining keywords (`max_chunk_chars`, `inline_headings`, `best_of`,
    `scorer`, `div_is_prose`, …) go straight through to it, which is the point:
    THIS is where a diet switch gets exercised.

    Resumable by record id. The output belongs to a different corpus than
    `pairs.jsonl`, so score it with `prune_orphans=False`.
    """
    from stylebot.ai_core import STYLE_SYSTEM

    if rewrite is None:
        from stylebot.infer import rewrite_text as rewrite

    out_path = Path(out_path)
    done = {record_id(rec) for rec in iter_pairs(out_path) if rec.get("messages")}

    docs = []
    for doc in iter_document_pairs(pairs_path, splits=splits, role=role):
        if not doc["before"].strip() or not doc["after"].strip():
            continue
        if f"doc:{doc['capture_id']}:0" in done:
            continue
        docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break

    result = DocumentRewriteResult(skipped_existing=len(done))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        for i, doc in enumerate(docs, start=1):
            try:
                rewritten = rewrite(doc["before"], styler, **rewrite_kwargs)
            except Exception as exc:  # noqa: BLE001 - one bad post must not end the pass
                result.errors.append((str(doc["capture_id"]), f"{type(exc).__name__}: {exc}"))
                continue
            text = getattr(rewritten, "text", rewritten)
            result.n_chunks += int(getattr(rewritten, "n_chunks", 0) or 0)
            result.n_kept_input += int(getattr(rewritten, "n_kept_input", 0) or 0)
            fp.write(
                dumps_line(
                    {
                        "messages": [
                            {"role": "system", "content": STYLE_SYSTEM},
                            {"role": "user", "content": doc["before"]},
                            {"role": "assistant", "content": doc["after"]},
                        ],
                        "output": text,
                        "meta": {
                            "source": doc["source"],
                            "capture_id": f"doc:{doc['capture_id']}",
                            "chunk_index": 0,
                            "synthetic": False,
                            "scale": "document",
                        },
                    }
                )
                + "\n"
            )
            fp.flush()
            result.written += 1
            if on_progress is not None:
                on_progress(i, len(docs))
    return result


def verdict_dir(runs_dir: str | Path, run_id: str) -> Path:
    """Where a run's verdict outputs live: `<runs_dir>/<run_id>/`.

    Run 4a's verdict was computed in `_tmp/eval/` and never committed, so the
    manifest records a training run whose evidence is on one laptop. Verdicts
    belong beside the manifest that names the adapter they judge.
    """
    path = Path(runs_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        raise RunError(f"no score records in {scores} — nothing to export")

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
