"""The eval harness's four signals, and one-passage scoring over them.

Offline ground truth: it scores candidate output and is not wired into the
trainer, not baked into the served adapter. Library-first — typed functions
over prose strings — and it never imports a CLI, nor `livingthing`.

The signals (`_plans/eval-harness.md` "The four signals"): **Vale** (mechanical
slop, shelled out to the binary, degrades gracefully if it is absent), an **LLM
judge** (injectable `Judge`; `openrouter_judge` builds a real one), a
**statistical detector** (pluggable `Detector`, `null_detector` by default), and
**Dan's eyeball** (the non-automatable veto channel, carried as a string).

Guardrail (`_plans/eval-harness.md` "Guardrail"): this harness is **not** a
slop-detection evader. A styler should lower its detector score only as a
*consequence* of writing more like Dan. The judge and eyeball exist to catch
reward-hacking; a movement in the detector alone is never the success criterion.

Networked imports (`openai`) are lazy, so importing this module requires no SDK
and no key. Corpus-scale scoring is `stylebot.scoring`, the paired verdict
`stylebot.verdict`, document-scale rewriting `stylebot.docscale`, the tidy CSV
`stylebot.covariates` — all re-exported here for compatibility (D62).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time as _time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Protocol

# `classify` is ML-dependency-free at import, so taking the detector-identity
# contract from it keeps one definition of "which head scored this".
from stylebot.classify import DETECTOR_IDENTITY_KEYS
from stylebot.errors import RunError

# Bumped to 3 on 2026-09-03: the summary gained `by_provenance` +
# `headline_facet` (D40) and `detectors` (D39). Additive, but the headline
# convention changed, so consumers keying on the version should see it move.
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
    startup dominates a per-text shell-out. The batch is written into one
    scratch directory under index-derived ``*.md`` names — unique by
    construction, so no two texts collide on a key in vale's
    ``{filename: [alerts]}`` map — and chunked across invocations of at most
    `_VALE_FILES_PER_RUN` files.

    One dict per input text, **in input order**, shaped as `vale_score` returns.
    Never raises: a whole-run failure (vale absent, unexecutable, silent,
    non-JSON, or reporting its own ``E100``) degrades every text in that chunk
    to ``{"available": False, "reason": …}``.
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
    ``*.md`` file, **not stdin**: vale scopes styles by matching the *filename*,
    and its synthetic stdin name matches no glob, so a config like
    ``[*.{md,qmd}] BasedOnStyles = …`` silently returns 0 alerts for everything.
    `config_path` is an explicit parameter — never a hardcoded blog ruleset.

    Returns ``{"available", "alerts", "by_severity", "raw"}``; `alerts` is the
    total (0 when unavailable) and `raw` is vale's per-alert objects. Never
    raises: no binary or no usable config gives ``available: False``.

    A batch of one over `vale_score_many`, which is what you want for more than
    one text — this costs a whole process.
    """
    return vale_score_many([text], config_path=config_path)[0]


# ---------------------------------------------------------------------------
# Signal 2 — LLM-as-judge ("is this Dan-shaped / free of AI-slop tells")
# ---------------------------------------------------------------------------

# Callable contract: prose -> {"score": int, "rationale": str}. Injectable so
# tests pass a fake (no API). `openrouter_judge` builds the blocking one and
# `openrouter_judge_async` the awaited one, which a corpus pass takes (D59).
Judge = Callable[[str], dict]
AsyncJudge = Callable[[str], Awaitable[dict]]

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


def _openrouter_client_kwargs(api_key: str | None, base_url: str | None) -> dict:
    """Key + endpoint for either OpenRouter client, resolved via `stylebot.config`.

    `OPENROUTER_API_KEY` is required (`config.require_key`);
    `OPENROUTER_BASE_URL` overrides the default endpoint. Nothing prints a key.
    """
    from stylebot import config

    return {
        "api_key": api_key or config.require_key("OPENROUTER_API_KEY"),
        "base_url": base_url
        or config.get_key("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1",
    }


def _judge_request(model: str, system: str, max_tokens: int, prose: str) -> dict:
    return {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prose},
        ],
        # OpenRouter returns the billed cost on the usage object when asked for
        # it. Without this the judge is the one signal whose price is unknowable
        # after the fact — the detector and Vale are free and local. Per call
        # rather than aggregated: a per-call figure is what sizes a subsample.
        "extra_body": {"usage": {"include": True}},
    }


def _judge_verdict(resp, model: str) -> dict:
    verdict = parse_judge_reply(resp.choices[0].message.content or "")
    verdict["model"] = model
    usage = getattr(resp, "usage", None)
    if usage is not None:
        verdict["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            # Absent on providers that do not report it, and — per the blog's
            # BYOK note — potentially the 5% fee rather than list price when the
            # arm bills to a bring-your-own key. Treat as an indication.
            "cost": getattr(usage, "cost", None),
        }
    return verdict


def openrouter_judge(
    *,
    model: str = "anthropic/claude-opus-4-8",
    api_key: str | None = None,
    base_url: str | None = None,
    system: str = JUDGE_SYSTEM,
    max_tokens: int = 256,
) -> Judge:
    """Build a blocking `Judge` backed by OpenRouter (OpenAI-compatible SDK).

    For single-passage callers (`score_candidate`). A corpus pass wants
    `openrouter_judge_async`, which is the same request awaited: `score_pairs_file`
    takes a sync judge only at `max_workers=1`, one call at a time (D59).
    """
    import openai

    client = openai.OpenAI(**_openrouter_client_kwargs(api_key, base_url))

    def judge(prose: str) -> dict:
        resp = client.chat.completions.create(**_judge_request(model, system, max_tokens, prose))
        return _judge_verdict(resp, model)

    return judge


def openrouter_judge_async(
    *,
    model: str = "anthropic/claude-opus-4-8",
    api_key: str | None = None,
    base_url: str | None = None,
    system: str = JUDGE_SYSTEM,
    max_tokens: int = 256,
) -> AsyncJudge:
    """`openrouter_judge` on `openai.AsyncOpenAI` — the judge a corpus pass takes.

    `score_pairs_file` overlaps these awaits `max_workers` at a time, the
    harness's only concurrency. The client is built on the first await, so it
    binds that loop, and `judge.aclose()` releases it when the pass ends.
    """
    import openai

    kwargs = _openrouter_client_kwargs(api_key, base_url)
    client = None

    async def judge(prose: str) -> dict:
        nonlocal client
        if client is None:
            client = openai.AsyncOpenAI(**kwargs)
        resp = await client.chat.completions.create(
            **_judge_request(model, system, max_tokens, prose)
        )
        return _judge_verdict(resp, model)

    async def aclose() -> None:
        nonlocal client
        if client is not None:
            await client.close()
            client = None

    judge.aclose = aclose
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
#: pointed at a mock or a future path without editing this module. There is no
#: synchronous `/predict`: Pangram 3.0 is `POST /task` -> task id ->
#: `GET /task/{id}`, polled until a terminal stage (findings.md, 2026-09-03).
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
    """The **optional, one-shot** independent cross-check (D7).

    A paid API (~$0.05/1k words, `PANGRAM_API_KEY`) that must never be the
    default detector and never run in a hot loop: use it once, over a bounded
    sample, to confirm the free detector's direction. What that bought and why
    it stays one-shot: findings.md, "Pangram cross-check".

    Polarity matches the house convention — ``score = fraction_ai``, higher =
    more AI-like — so it drops into `mean_detector_score` unchanged. `post` and
    `fetch` are the transport seams (`(url, payload, headers) -> dict` and
    `(url, headers) -> dict`), so tests exercise everything but the spend.
    Networking is stdlib `urllib`: a plain JSON POST to a non-OpenAI endpoint,
    and eval must stay importable with no SDK installed.
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


def _chunked_name(
    identity: dict, max_chunk_chars: int, inline_headings: bool, div_is_prose=None
) -> str:
    """How a chunk-wise detector names the scale and div policy it ran under.

    Scale, and a `div_is_prose` saying what is prose at all, each make a second
    instrument from one artifact; `DETECTOR_IDENTITY_KEYS` has a field for
    neither, so both ride on `name` — the policy as `_canonical` writes it.
    """
    from stylebot.provenance import _canonical

    suffix = f"/chunked:{max_chunk_chars}" + (":inline-headings" if inline_headings else "")
    if div_is_prose is not None:
        suffix += f":divs={_canonical(div_is_prose)}"
    return f"{identity.get('name') or 'detector'}{suffix}"


def chunked_detector(
    detector: Detector,
    *,
    max_chunk_chars: int,
    inline_headings: bool = False,
    div_is_prose=None,
) -> Detector:
    """Score a WHOLE DOCUMENT at the scale the head was fit on, and aggregate.

    Wraps any `Detector`: a passage is segmented the way serving segments it
    (`infer.prose_chunks` — same budget, diet and protected blocks), each prose
    chunk is scored alone, and the passage's score is the **unweighted mean over
    its chunks**, with ``n_chunks`` and ``sd`` (the within-passage sample
    standard deviation) beside it. Chunks are **not** aligned across `slop` /
    `target` / `output`, so a comparison is between two unaligned means. A
    passage with no prose chunks scores ``None``, the convention every other
    absent signal follows. The identity is the wrapped head's, with the scale
    and any `div_is_prose` appended to ``name``, so D39's mixing guard refuses
    to blend two instruments' readings into one mean. Why all of that: D54.
    """
    from stylebot.infer import prose_chunks

    identity = detector_identity(detector)
    identity["name"] = _chunked_name(identity, max_chunk_chars, inline_headings, div_is_prose)
    inner_batch = getattr(detector, "batch_detect", None)

    def _cut(prose: str) -> list[str]:
        return prose_chunks(
            prose,
            max_chunk_chars=max_chunk_chars,
            inline_headings=inline_headings,
            div_is_prose=div_is_prose,
        )

    def _aggregate_chunks(results: Sequence[dict], n_chunks: int) -> dict:
        scores = [r.get("score") for r in results if r.get("score") is not None]
        configured = any(r.get("configured") for r in results)
        if not scores:
            return {
                "score": None,
                "p_dan": None,
                "configured": configured,
                "n_chunks": n_chunks,
                "sd": None,
                **identity,
            }
        mean_score = mean(scores)
        return {
            "score": mean_score,
            "p_dan": 1.0 - mean_score,
            "configured": configured,
            "n_chunks": len(scores),
            "sd": (stdev(scores) if len(scores) > 1 else 0.0),
            **identity,
        }

    def detect(prose: str) -> dict:
        chunks = _cut(prose)
        return _aggregate_chunks([detector(chunk) for chunk in chunks], len(chunks))

    detect.identity = dict(identity)

    if inner_batch is not None:
        # One passage is ~17 embeds, so this is the wrapper that most needs a
        # batch path: flatten, one encode, regroup through the same aggregation
        # the per-passage path uses, so the two agree by construction.
        def batch_detect(passages: Sequence[str]) -> list[dict]:
            cut = [_cut(p) for p in passages]
            flat = [chunk for chunks in cut for chunk in chunks]
            results = inner_batch(flat)
            out: list[dict] = []
            at = 0
            for chunks in cut:
                out.append(_aggregate_chunks(results[at : at + len(chunks)], len(chunks)))
                at += len(chunks)
            return out

        detect.batch_detect = batch_detect
    return detect


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
    detector_result: dict | None = None,
    judge_result: dict | None = None,
) -> dict:
    """Score one candidate passage across the available signals.

    `judge=None` skips the LLM signal (the harness then runs with no API key);
    its field comes back ``None``. `eyeball` is an optional human note carried
    verbatim into the record.

    Each signal has an injection slot for an already-computed result, which is
    how `score_pairs_file` feeds a whole slice's pre-passes back in: `vale`
    (from `vale_score_many`), `detector_result` (from `classify.detect_many`),
    `judge_result` (from the awaited judge pass). Leave them `None` — the
    default — and this computes each signal itself, one passage at a time.

    Returns a stable, JSON-able record keyed
    ``text_chars`` / ``vale`` / ``judge`` / ``detector`` / ``eyeball``.
    """
    if judge_result is None and judge is not None:
        judge_result = judge(text)
    return {
        "text_chars": len(text),
        "vale": vale if vale is not None else vale_score(text, config_path=vale_config),
        "judge": judge_result,
        "detector": detector_result if detector_result is not None else detector(text),
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


# --- compatibility re-exports for the modules this one split into (D62) ------
# They import from here, so this direction is lazy or it is a cycle. New code
# imports the owner; this keeps `stylebot.eval.X` working for the blog mirror.

_SPLIT_OUT = {
    "stylebot.scoring": (
        "ScoreResult",
        "assert_single_detector",
        "detector_identities",
        "existing_scored_ids",
        "live_record_ids",
        "load_scores",
        "prune_orphan_scores",
        "score_pairs_file",
        "score_pairs_file_async",
        "split_by_provenance",
        "summarize_scores",
    ),
    "stylebot.verdict": (
        "SIGNAL_EXTRACTORS",
        "VERDICT_PARAMS",
        "DocumentOpts",
        "VerdictOpts",
        "compare_scores",
        "format_summary",
        "format_verdict",
        "paired_delta",
        "verdict_options",
    ),
    "stylebot.docscale": (
        "DocumentRewriteResult",
        "iter_document_pairs",
        "rewrite_documents_file",
        "verdict_dir",
    ),
    "stylebot.covariates": ("covariate_table", "write_covariate_table"),
}
_OWNER = {name: module for module, names in _SPLIT_OUT.items() for name in names}


def __getattr__(name: str):
    module = _OWNER.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_OWNER])
