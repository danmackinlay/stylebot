"""The eval harness's four signals, and one-passage scoring over them (D7).

Offline ground truth over prose strings, never wired into the trainer: Vale
(mechanical slop, degrades gracefully when absent), an LLM judge (injectable
`Judge`; `openrouter_judge` builds one), a statistical detector (pluggable
`Detector`, `null_detector` by default) and the eyeball (a human note). A
detector movement alone is never the success criterion (D7). Networked
imports are lazy. Corpus-scale scoring is `stylebot.scoring`, the verdict
`stylebot.verdict`, document scale `stylebot.docscale`, the tidy CSV
`stylebot.covariates`, all re-exported here (D62).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Protocol

# One definition of "which head scored this"; `classify` is import-light.
from stylebot.classify import DETECTOR_IDENTITY_KEYS
from stylebot.errors import ConfigError, MissingExtra, RunError

#: 3 -> 4 when a judge refusal became `None` and `n_unscored` joined the summary (D87).
SCHEMA_VERSION = 4

#: The judge's OpenRouter model id: `openrouter_judge*` and `verdict.EvalOpts` share it.
DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4-8"

# ---------------------------------------------------------------------------
# Signal 1 — Vale (mechanical slop)
# ---------------------------------------------------------------------------


# Files handed to a single `vale` invocation; the paths share one argv, and
# ARG_MAX is the wall.
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
    """Last path segment of a vale JSON key, whichever separator it used; the
    scratch filenames are unique, so it is the join key."""
    return str(key).replace("\\", "/").rsplit("/", 1)[-1]


def _vale_run(texts: Sequence[str], *, config_path: str | Path | None) -> list[dict]:
    """One `vale` invocation over `texts`; one result dict per text, in order."""
    cmd = ["vale", "--output=JSON"]
    if config_path is not None:
        cmd.append(f"--config={Path(config_path)}")

    # Index-derived filenames: unique even when two texts carry identical prose.
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

    # A runtime-error object (e.g. E100) instead of the map: the whole batch degrades.
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
    # A clean file is omitted from vale's map: a missing key is zero alerts.
    return [_vale_summarize(by_name.get(name, [])) for name in names]


def vale_score_many(texts: Sequence[str], *, config_path: str | Path | None = None) -> list[dict]:
    """Score many texts with as few `vale` processes as possible.

    The batch form of `vale_score`: one scratch directory, at most
    `_VALE_FILES_PER_RUN` files per invocation. Returns one dict per text, in
    input order, shaped as `vale_score` returns. Never raises: a whole-run
    failure degrades every text in that chunk to ``{"available": False, …}``.
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
    """Score `text` with the `vale` binary; a batch of one over `vale_score_many`.

    The text goes through a temporary ``*.md`` file, not stdin: vale scopes
    styles by filename, and stdin matches no glob. Returns ``{"available",
    "alerts", "by_severity", "raw"}``; never raises — no binary or no usable
    config gives ``available: False``.
    """
    return vale_score_many([text], config_path=config_path)[0]


# ---------------------------------------------------------------------------
# Signal 2 — LLM-as-judge ("is this Dan-shaped / free of AI-slop tells")
# ---------------------------------------------------------------------------

# Callable contract: prose -> {"score": int, "rationale": str}; a corpus pass
# takes the awaited form (D59).
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

    Tolerates prose around the JSON object and clamps the score into 1-5. A
    reply with no numeric ``score`` is ``{"score": None, "parse_error": True,
    "rationale": reply}``: unscored, never a digit fished out of the prose.
    """
    obj = _first_json_object(reply)
    try:
        score = int(obj.get("score")) if obj is not None else None
    except (TypeError, ValueError):
        score = None
    if score is None:
        return {"score": None, "parse_error": True, "rationale": reply}
    rationale = obj.get("rationale")
    return {
        "score": max(1, min(5, score)),
        "rationale": str(rationale) if rationale is not None else "",
    }


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
    """Key + endpoint for either OpenRouter client, via `stylebot.config`;
    nothing prints a key."""
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
        # Billed cost per call, so the judge's price is knowable after the fact.
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
            # Absent on providers that do not report it; the BYOK fee, not
            # list price, on a bring-your-own key (D66). An indication.
            "cost": getattr(usage, "cost", None),
        }
    return verdict


def openrouter_judge(
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    system: str = JUDGE_SYSTEM,
    max_tokens: int = 256,
) -> Judge:
    """Build a blocking `Judge` backed by OpenRouter, for single-passage callers;
    a corpus pass wants `openrouter_judge_async` (D59)."""
    import openai

    client = openai.OpenAI(**_openrouter_client_kwargs(api_key, base_url))

    def judge(prose: str) -> dict:
        resp = client.chat.completions.create(**_judge_request(model, system, max_tokens, prose))
        return _judge_verdict(resp, model)

    return judge


def openrouter_judge_async(
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    system: str = JUDGE_SYSTEM,
    max_tokens: int = 256,
) -> AsyncJudge:
    """`openrouter_judge` on `openai.AsyncOpenAI` — the judge a corpus pass takes
    (D59). The client binds the loop of its first await; `aclose()` releases it."""
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


# Callable contract: prose -> {"score": float|None, "name": str, ...}; higher
# is more AI-like.
class Detector(Protocol):
    def __call__(self, prose: str) -> dict: ...


def null_detector(prose: str) -> dict:
    """The default detector: ``score=None``, signalling it is unconfigured (D7)."""
    return {"score": None, "name": "null", "configured": False}


null_detector.identity = {key: None for key in DETECTOR_IDENTITY_KEYS} | {"name": "null"}


def detector_identity(detector: Detector) -> dict:
    """Which head is this — `DETECTOR_IDENTITY_KEYS` from the ``.identity``
    attribute, without spending an embed; a bare callable answers all-None
    with its ``__name__``."""
    identity = getattr(detector, "identity", None)
    if isinstance(identity, dict):
        return {key: identity.get(key) for key in DETECTOR_IDENTITY_KEYS}
    fallback = {key: None for key in DETECTOR_IDENTITY_KEYS}
    fallback["name"] = getattr(detector, "__name__", None)
    return fallback


def _chunked_name(
    identity: dict, max_chunk_chars: int, inline_headings: bool, div_is_prose=None
) -> str:
    """How a chunk-wise detector names the scale and div policy it ran under:
    each makes a second instrument from one artifact, so both ride on `name`."""
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
    html_comments: bool = False,
) -> Detector:
    """Score a whole document at the scale the head was fit on, and aggregate (D54).

    Segments as serving does (`infer.prose_chunks`), scores each prose chunk,
    and returns the unweighted mean with ``n_chunks`` and ``sd``; no prose
    chunks scores ``None``. The identity is the wrapped head's with the scale
    and any `div_is_prose` appended to ``name`` (D39).
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
            html_comments=html_comments,
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
        # Flatten, one encode, regroup through the per-passage aggregation.
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
# Signal 3, hosted kinds — Sapling, the cache, the one factory (D106)
# ---------------------------------------------------------------------------

#: Every `--detector KIND`; `build_detector` is the factory behind the flag.
DETECTOR_KINDS = ("voice-clf", "sapling")

SAPLING_URL = "https://api.sapling.ai/api/v1/aidetect"
#: Pinned: a version bump is a new instrument (new `name`, so new thresholds).
SAPLING_VERSION = "20251027"
#: Sapling's per-request caps: texts in a batch, and their combined length.
SAPLING_BATCH = 10
SAPLING_MAX_CHARS = 200_000
#: Sapling's own floor; a shorter text abstains unsent (`score: None`).
SAPLING_MIN_CHARS = 300


def _urllib_post_json(url: str, payload: dict, headers: dict, *, timeout: float = 30.0) -> dict:
    """One JSON POST over stdlib urllib; an HTTP or transport failure is a
    `RunError` naming the status and the body's head, never a silent score."""
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:  # noqa: S310 - fixed https
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:200]
        raise RunError(f"{url}: HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RunError(f"{url}: {exc}") from exc


def _sapling_batches(texts: Sequence[str]) -> Iterable[list[int]]:
    """Index batches under Sapling's per-request caps, in input order."""
    batch: list[int] = []
    chars = 0
    for i, text in enumerate(texts):
        if batch and (len(batch) >= SAPLING_BATCH or chars + len(text) > SAPLING_MAX_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(i)
        chars += len(text)
    if batch:
        yield batch


def sapling_detector(
    *,
    api_key: str | None = None,
    version: str = SAPLING_VERSION,
    min_chars: int = SAPLING_MIN_CHARS,
    timeout: float = 30.0,
    post: Callable[[str, dict, dict], dict] | None = None,
) -> Detector:
    """Sapling's hosted human-vs-AI detector as a `Detector` (D106): the batch
    form, ``score`` its 0-human…1-AI number unchanged (the house polarity). A
    text under `min_chars` abstains and is never sent. `post` is the transport
    seam; ``.stats`` counts the spend. Needs `SAPLING_API_KEY` unless given."""
    from stylebot import config

    key = api_key or config.require_key("SAPLING_API_KEY")

    def _post(url: str, payload: dict, headers: dict) -> dict:
        return _urllib_post_json(url, payload, headers, timeout=timeout)

    transport = post or _post
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    identity = dict.fromkeys(DETECTOR_IDENTITY_KEYS) | {"name": f"sapling-{version}"}
    stats = {"requests": 0, "texts_scored": 0, "chars_sent": 0}

    def _result(score: float | None) -> dict:
        p_dan = None if score is None else 1.0 - score
        return {"score": score, "p_dan": p_dan, "configured": True, **identity}

    def batch_detect(texts: Sequence[str]) -> list[dict]:
        texts = list(texts)
        out = [_result(None) for _ in texts]
        sendable = [i for i, t in enumerate(texts) if len(t) >= min_chars]
        for batch in _sapling_batches([texts[i] for i in sendable]):
            idx = [sendable[j] for j in batch]
            payload = {"texts": [texts[i] for i in idx], "sent_scores": False, "version": version}
            body = transport(SAPLING_URL, payload, headers)
            results = body.get("results") if isinstance(body, dict) else None
            if not isinstance(results, list) or len(results) != len(idx):
                raise RunError(f"sapling: expected {len(idx)} results, got {str(body)[:200]!r}")
            stats["requests"] += 1
            stats["texts_scored"] += len(idx)
            stats["chars_sent"] += sum(len(texts[i]) for i in idx)
            for i, item in zip(idx, results, strict=True):
                score = item.get("score") if isinstance(item, dict) else None
                out[i] = _result(None if score is None else float(score))
        return out

    def detect(prose: str) -> dict:
        return batch_detect([prose])[0]

    detect.identity = dict(identity)
    detect.batch_detect = batch_detect
    detect.stats = stats
    return detect


def cached_detector(detector: Detector) -> Detector:
    """Memoise `detector` by text hash, in process (D106): a rescore of an
    unchanged paragraph costs nothing; misses go through in one batch, deduped.
    Keeps ``.identity``; ``.stats`` is the inner's plus ``cache_hits``."""
    import hashlib

    memo: dict[str, dict] = {}
    inner_batch = getattr(detector, "batch_detect", None)
    stats = getattr(detector, "stats", None)
    if not isinstance(stats, dict):
        stats = {}
    stats["cache_hits"] = 0

    def batch_detect(texts: Sequence[str]) -> list[dict]:
        keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        todo: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key not in memo:
                todo.setdefault(key, text)
        if todo:
            fresh = list(todo.values())
            results = inner_batch(fresh) if inner_batch else [detector(t) for t in fresh]
            memo.update(zip(todo, results, strict=True))
        stats["cache_hits"] += len(keys) - len(todo)  # every text answered without a send
        return [dict(memo[k]) for k in keys]

    def detect(prose: str) -> dict:
        return batch_detect([prose])[0]

    detect.identity = detector_identity(detector)
    detect.batch_detect = batch_detect
    detect.stats = stats
    return detect


def build_detector(kind: str, model_dir: str | Path | None = None, **kw) -> Detector:
    """The factory behind `--detector KIND` (D106): ``voice-clf`` rebuilds the
    artifact at `model_dir`, ``sapling`` dials the hosted API; `kw` reaches
    the kind's constructor (a test's ``post=``)."""
    if kind == "voice-clf":
        if model_dir is None:
            raise ConfigError("--detector voice-clf needs --detector-model (head.json + meta.json)")
        from stylebot import classify

        try:
            return classify.sklearn_detector(model_dir, **kw)
        except ImportError as exc:
            raise MissingExtra(f"--detector-model needs the embedder installed: {exc}") from exc
    if kind == "sapling":
        return sapling_detector(**kw)
    raise ConfigError(f"unknown detector kind {kind!r} (expected one of {DETECTOR_KINDS})")


def arm_detectors(kinds: Sequence[str], model_dir: str | Path | None = None) -> dict[str, Detector]:
    """`build_detector` over `kinds`, keyed by identity name in the given order
    (the sidecar's default is the first); hosted kinds sit behind
    `cached_detector`. A repeated kind is a `ConfigError`."""
    if len(set(kinds)) != len(kinds):
        raise ConfigError(f"--detector repeats a kind: {list(kinds)}")
    armed: dict[str, Detector] = {}
    for kind in kinds:
        detector = build_detector(kind, model_dir)
        if kind != "voice-clf":
            detector = cached_detector(detector)
        armed[detector_identity(detector)["name"]] = detector
    return armed


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

    `judge=None` skips the LLM signal (its field is ``None``); `eyeball` is a
    human note carried verbatim. `vale`, `detector_result` and `judge_result`
    inject a pre-computed result. Returns a JSON-able record keyed
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
    """Per-group aggregates over `score_candidate` records; ``n_unscored`` is
    judge refusals, kept out of the mean (a keyless run is unjudged, not unscored)."""
    judged = [r["judge"] for r in records if isinstance(r.get("judge"), dict)]
    judge_scores = [j["score"] for j in judged if j.get("score") is not None]
    vale_alerts = [r["vale"]["alerts"] for r in records if (r.get("vale") or {}).get("available")]
    detector_scores = [
        r["detector"]["score"]
        for r in records
        if isinstance(r.get("detector"), dict) and r["detector"].get("score") is not None
    ]
    return {
        "n": len(records),
        "n_unscored": len(judged) - len(judge_scores),
        "mean_judge_score": _mean_or_none(judge_scores),
        "mean_vale_alerts": _mean_or_none(vale_alerts),
        "mean_detector_score": _mean_or_none(detector_scores),
    }


# --- compatibility re-exports for the modules this one split into (D62); lazy
# because they import from here.

_SPLIT_OUT = {
    "stylebot.scoring": (
        "ScoreResult",
        "assert_single_detector",
        "detector_identities",
        "live_record_ids",
        "load_scores",
        "prune_orphan_scores",
        "score_pairs_file",
        "score_pairs_file_async",
        "split_by_provenance",
        "summarize_scores",
        "run_eval",
    ),
    "stylebot.verdict": (
        "SIGNAL_EXTRACTORS",
        "EVAL_PARAMS",
        "VERDICT_PARAMS",
        "DocumentOpts",
        "EvalOpts",
        "VerdictOpts",
        "compare_scores",
        "eval_options",
        "format_summary",
        "format_verdict",
        "paired_delta",
        "report_eval",
        "verdict_options",
    ),
    "stylebot.docscale": (
        "DocumentPass",
        "DocumentRewriteResult",
        "document_pass",
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
