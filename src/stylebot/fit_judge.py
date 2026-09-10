"""Signal 5's judge — a paired forced choice at a boundary (D97).

`(before, first, second, after) -> {"choice": 0|1|2, "rationale": str}`, where
1/2 index the spans **as presented** and 0 is a tie. The cases are
`stylebot.fit`; `judge_fit_pass` is the `ai-style judge-fit` body (D78).

Two refusals, both deliberate: this never runs at inference — it is an eval
signal and a source of Phase-7 preference labels, not a reranker — and it never
scores voice, which `eval.JUDGE_SYSTEM` already does. Design and the first
verdict: `_plans/fit-judge.md`, `findings.md` "The fit judge (coherence)".
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import click

from stylebot import config
from stylebot.cli_options import Emitter, bundle_options, opt, silent
from stylebot.draws import ViewOpts
from stylebot.errors import ConfigError
from stylebot.eval import DEFAULT_JUDGE_MODEL, _first_json_object, _openrouter_client_kwargs
from stylebot.fit import NEIGHBOUR_SOURCES, Cut, FitCase, fits_from_rewrites
from stylebot.jsonl import dumps_line, iter_jsonl
from stylebot.scoring import DEFAULT_MAX_WORKERS

FitJudge = Callable[[str, str, str, str], dict | Awaitable[dict]]

FIT_JUDGE_SYSTEM = (
    # FROZEN, "seam" and all: findings.md's numbers were produced by this exact
    # string, and an instrument that drifts silently is worse than a stale word.
    "You are judging COHERENCE at a seam in a piece of writing, not writing quality "
    "and not voice. You are shown a paragraph BEFORE, then two candidate versions of "
    "the passage that sits between them, then a paragraph AFTER. BEFORE and AFTER are "
    "fixed and identical for both candidates; only the middle differs. "
    "Choose the candidate that fits its neighbours better. Consider only: pronouns and "
    "definite references that resolve to something already named; terms used "
    "consistently with the surrounding text; a metaphor or running example carried "
    "forward rather than restarted or dropped; continuity of tense, person and "
    "register; no restating of what BEFORE already said and no contradiction of it; "
    "and a hand-off that AFTER follows from. "
    "Explicitly IGNORE which candidate is better written, wittier, more concise, or "
    "more in any particular author's voice — those are scored elsewhere, and a "
    "candidate may fit its neighbours better while being the weaker prose. "
    "If neither fits better — the differences are real but coherence-neutral — say tie. "
    "Do not use tie merely because the choice is hard. "
    'Reply with ONLY a JSON object of the form {"choice": <0, 1 or 2>, '
    '"rationale": "<one short sentence naming the specific link that survives or breaks>"}, '
    "where 1 = the first candidate, 2 = the second, 0 = tie. "
    "Do not wrap it in code fences or add any other text."
)


def build_fit_prompt(before: str, first: str, second: str, after: str) -> str:
    """The user message: fixed neighbours, two candidates, explicit labels."""
    return (
        "=== BEFORE (fixed) ===\n"
        f"{before.strip() or '(nothing — this passage opens its section)'}\n\n"
        "=== CANDIDATE 1 ===\n"
        f"{first.strip()}\n\n"
        "=== CANDIDATE 2 ===\n"
        f"{second.strip()}\n\n"
        "=== AFTER (fixed) ===\n"
        f"{after.strip() or '(nothing — this passage closes its section)'}"
    )


def parse_fit_reply(reply: str) -> dict:
    """Extract `{"choice": 0|1|2, "rationale": str}` from a model reply.

    An unparseable reply falls back to **tie**, never to a side: a coin-flip
    fallback would inject noise straight into the counts the verdict is read
    off, where `eval`'s mid-score fallback costs only a little accuracy.
    """
    obj = _first_json_object(reply)
    if obj is not None:
        try:
            choice = int(obj.get("choice"))
        except (TypeError, ValueError):
            choice = None
        if choice in (0, 1, 2):
            rationale = obj.get("rationale")
            return {"choice": choice, "rationale": str(rationale or "")}
    m = re.search(r"\b([012])\b", reply)
    if m:
        return {"choice": int(m.group(1)), "rationale": reply.strip()}
    return {"choice": 0, "rationale": reply.strip(), "unparsed": True}


def openrouter_fit_judge(
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    system: str = FIT_JUDGE_SYSTEM,
    max_tokens: int = 256,
) -> FitJudge:
    """An async `FitJudge` over OpenRouter — the corpus pass awaits (D59). The
    client binds the loop of its first await."""
    import openai

    kwargs = _openrouter_client_kwargs(api_key, base_url)
    client = None

    async def judge(before: str, first: str, second: str, after: str) -> dict:
        nonlocal client
        if client is None:
            client = openai.AsyncOpenAI(**kwargs)
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": build_fit_prompt(before, first, second, after)},
            ],
            extra_body={"usage": {"include": True}},
        )
        verdict = parse_fit_reply(resp.choices[0].message.content or "")
        verdict["model"] = model
        usage = getattr(resp, "usage", None)
        if usage is not None:
            verdict["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "cost": getattr(usage, "cost", None),
            }
        return verdict

    return judge


# ---------------------------------------------------------------------------
# Running the comparison
# ---------------------------------------------------------------------------


def a_first(fit_id: str, *, salt: str = "") -> bool:
    """Whether arm A is presented FIRST for this case.

    Deterministic from the id, not `random`: a resumed run must present a case
    the way it did before, or the recorded order stops describing the call that
    produced the verdict. `salt` re-randomises a whole comparison.
    """
    return hashlib.sha256(f"{salt}\x00{fit_id}".encode()).digest()[0] & 1 == 0


@dataclass
class FitResult:
    n_judged: int = 0
    skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)


def existing_fit_ids(path: str | Path) -> set[str]:
    """The case ids already judged in `path` — the resume point."""
    return {r["id"] for r in iter_jsonl(path) if isinstance(r.get("id"), str)}


def _fit_record(
    case: FitCase, span_a: str, span_b: str, verdict: dict, *, arm_a: str, arm_b: str, salt: str
) -> dict:
    """One `fits.jsonl` row, `choice` normalised to the ARM: no consumer should
    have to un-shuffle, and the presented order rides along so the judge's
    position bias stays measurable afterwards."""
    a_is_first = a_first(case.id, salt=salt)
    slot = verdict.get("choice", 0)
    first_arm, second_arm = (arm_a, arm_b) if a_is_first else (arm_b, arm_a)
    rec = {
        "id": case.id,
        "source": case.source,
        "index": case.index,
        "arms": [arm_a, arm_b],
        "presented": [first_arm, second_arm],
        "choice": "tie" if slot == 0 else (first_arm if slot == 1 else second_arm),
        "choice_slot": slot,
        "rationale": verdict.get("rationale", ""),
        "interior": case.is_interior,
        "before": case.before,
        "after": case.after,
        "spans": {arm_a: span_a, arm_b: span_b},
    }
    for k in ("model", "usage", "unparsed"):
        if k in verdict:
            rec[k] = verdict[k]
    return rec


async def judge_fits_async(
    cases: Sequence[tuple[FitCase, str, str]],
    out_path: str | Path,
    *,
    judge: FitJudge,
    arm_a: str = "a",
    arm_b: str = "b",
    salt: str = "",
    max_workers: int = DEFAULT_MAX_WORKERS,
    limit: int | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> FitResult:
    """Judge `(case, span_a, span_b)` triples -> an id-keyed `fits.jsonl`.

    Resumable and **position-randomised per case**, both here rather than in
    the caller because both are cheap and easy to omit. `max_workers` calls are
    in flight (D59); a sync judge is serial. Rows are drained in submission
    order and flushed as they land, and a judge that raised costs its own case
    rather than the run, since the file is the resume point.
    """
    out_path = Path(out_path)
    todo = [t for t in cases if t[0].id not in existing_fit_ids(out_path)]
    result = FitResult(skipped_existing=len(cases) - len(todo))
    if limit is not None:
        todo = todo[:limit]
    if not todo:
        return result

    gate = asyncio.Semaphore(max(1, max_workers))

    async def one(triple: tuple[FitCase, str, str]) -> dict:
        case, span_a, span_b = triple
        first, second = (span_a, span_b) if a_first(case.id, salt=salt) else (span_b, span_a)
        async with gate:
            verdict = judge(case.before, first, second, case.after)
            if inspect.isawaitable(verdict):
                verdict = await verdict
        return _fit_record(case, span_a, span_b, verdict, arm_a=arm_a, arm_b=arm_b, salt=salt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tasks = [asyncio.ensure_future(one(t)) for t in todo]
    with out_path.open("a", encoding="utf-8") as fp:
        for done, (triple, task) in enumerate(zip(todo, tasks, strict=True), start=1):
            try:
                rec = await task
            except Exception as exc:  # noqa: BLE001 — one bad case, not a dead run
                result.errors.append(f"{triple[0].id}: {exc}")
                continue
            fp.write(dumps_line(rec) + "\n")
            fp.flush()
            result.n_judged += 1
            if on_progress:
                on_progress(done, len(todo))
    return result


def judge_fits(cases: Sequence[tuple[FitCase, str, str]], out_path: str | Path, **kw) -> FitResult:
    """The sync entry point: `asyncio.run` over `judge_fits_async`."""
    return asyncio.run(judge_fits_async(cases, out_path, **kw))


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k/n, which is the right tool at this n: well
    under 100 decided cases, with a proportion near 0.5."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize_fits(records: Iterable[dict], *, arm_a: str = "a", arm_b: str = "b") -> dict:
    """Win/loss/tie with a Wilson CI on the A-win rate **among decided cases**.

    A tie is a judgment ("these differ, but not about coherence"), not a
    half-win, so it is reported and excluded rather than split — folding it in
    drags every comparison toward 0.5 in proportion to the judge's caution.
    `n_interior` is printed because a verdict resting on one-sided cases
    measures a weaker constraint than the design claims.
    """
    a = b = tie = n = interior = unparsed = 0
    cost = 0.0
    for r in records:
        n += 1
        choice = r.get("choice")
        a += choice == arm_a
        b += choice == arm_b
        tie += choice not in (arm_a, arm_b)
        interior += bool(r.get("interior"))
        unparsed += bool(r.get("unparsed"))
        usage = r.get("usage") or {}
        if isinstance(usage.get("cost"), int | float):
            cost += float(usage["cost"])
    decided = a + b
    lo, hi = _wilson(a, decided)
    return {
        "schema_version": 1,
        "arms": [arm_a, arm_b],
        "n": n,
        "n_interior": interior,
        "n_decided": decided,
        f"{arm_a}_wins": a,
        f"{arm_b}_wins": b,
        "ties": tie,
        "a_win_rate": (a / decided) if decided else None,
        "a_win_rate_ci95": [lo, hi] if decided else None,
        # An interval excluding 0.5 is a difference; anything else is a wash,
        # however lopsided the raw counts look.
        "decisive": bool(decided and (lo > 0.5 or hi < 0.5)),
        "n_unparsed": unparsed,
        "judge_cost": round(cost, 6) if cost else None,
    }


# ---------------------------------------------------------------------------
# The `judge-fit` pass (D78)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitOpts:
    """The `judge-fit` surface: which two arms, cut how, judged by whom."""

    cli_section = "Fit judge"

    a_path: Path | None = opt(
        None,
        "--a",
        "a_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Arm A: an output-bearing rewrite JSONL (`ai-style run --pairs --out`).",
    )
    b_path: Path | None = opt(
        None,
        "--b",
        "b_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="Arm B: the same corpus rewritten by the other adapter (--a again for a control).",
    )
    field_a: str = opt(
        "output",
        "--field-a",
        show_default=True,
        type=click.Choice(sorted(("output", "target", "slop"))),
        help="Which text of arm A's records is the span. The POSITIVE CONTROL is "
        "--field-a target --field-b slop on one file, and it runs with every verdict: a "
        "judge that cannot pick Dan over the slop he replaced has no resolution (D97).",
    )
    field_b: str = opt(
        "output",
        "--field-b",
        show_default=True,
        type=click.Choice(sorted(("output", "target", "slop"))),
        help="Which text of arm B's records is the span.",
    )
    arm_a: str = opt("a", "--arm-a", show_default=True, help="Label for arm A in the output.")
    arm_b: str = opt("b", "--arm-b", show_default=True, help="Label for arm B in the output.")
    pairs_path: Path | None = opt(
        None,
        "--pairs",
        "pairs_path",
        type=click.Path(exists=True, path_type=Path),
        help="A scratch corpus in place of --draws: the PRIMARY neighbour source, since a "
        "synthetic pair's chunk_index is its position in the source post.",
    )
    captures_dir: Path | None = opt(
        None,
        "--captures",
        "captures_dir",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Capture archives [default: the captures/ beside --draws]. The fallback for real "
        "records, whose chunk_index orders changed regions, not document positions (D96).",
    )
    blog_root: Path | None = opt(
        None,
        "--blog-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Live posts — the last-resort neighbour source, matched on the Dan target. The "
        "prose has drifted since capture, so it places a minority of records.",
    )
    max_chunk_chars: int = opt(
        4500,
        "--max-chunk-chars",
        show_default=True,
        type=int,
        help="Cut neighbours at the SERVING diet (blog-side: chunk_diet).",
    )
    inline_headings: bool = opt(
        True,
        "--inline-headings/--no-inline-headings",
        show_default=True,
        help="Match the adapter's heading diet (cross-heading = inline).",
    )
    html_comments: bool = opt(
        False,
        "--html-comments/--no-html-comments",
        show_default=True,
        help="Protect whole-line HTML comments, if the served diet did (D93).",
    )
    judge_model: str = opt(
        DEFAULT_JUDGE_MODEL,
        "--judge-model",
        show_default=True,
        help="OpenRouter model id for the fit judge.",
    )
    out_path: Path | None = opt(
        None,
        "--out",
        "out_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="fits.jsonl to append [default: beside --a]. Resumable: judged ids are skipped.",
    )
    summary_path: Path | None = opt(
        None,
        "--summary",
        "summary_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Also write the win/loss/tie summary JSON here.",
    )
    salt: str = opt(
        "",
        "--salt",
        help="Re-randomise presentation order for a position-bias replication, "
        "without renaming any case.",
    )
    limit: int | None = opt(
        None, "--limit", type=int, help="Cap cases judged (smoke test / cost control)."
    )
    max_workers: int = opt(
        DEFAULT_MAX_WORKERS, "--max-workers", show_default=True, type=int, help="Judges in flight."
    )
    dry_run: bool = opt(
        False,
        "--dry-run",
        is_flag=True,
        help="Build the stratum, report what it holds, and stop — no key, no spend. Run it "
        "first: a stratum that quietly shrank reads exactly like a small one.",
    )
    #: Library-only: an injected judge, in place of `--judge-model`'s.
    judge: FitJudge | None = None

    def cut(self) -> Cut:
        return Cut(
            max_chars=self.max_chunk_chars,
            inline_headings=self.inline_headings,
            html_comments=self.html_comments,
        )


FIT_PARAMS: dict[str, type] = {"fit": FitOpts, "view": ViewOpts}


def fit_options(**kw):
    """The whole `judge-fit` surface, one decorator."""
    return bundle_options(list(FIT_PARAMS.values()), **kw)


def _report_stratum(rep, triples, emit: Emitter) -> None:
    """The stratum banner, provenance always printed: `sibling` neighbours are
    exact and `post` ones drifted, so a stratum leaning on the weak sources is
    a weaker verdict and the reader should not have to ask."""
    interior = sum(1 for t in triples if t[0].is_interior)
    emit(
        f"stratum: {rep.n_built} cases ({interior} interior) from {rep.n_joined} joined "
        f"records [A {rep.n_a}, B {rep.n_b}, only-A {rep.only_in_a}, only-B {rep.only_in_b}]",
        level="status",
    )
    if rep.by_neighbour_source:
        counts = ", ".join(
            f"{k} {rep.by_neighbour_source[k]}"
            for k in NEIGHBOUR_SOURCES
            if k in rep.by_neighbour_source
        )
        emit(f"  neighbours: {counts}", level="status")
    for label, ids in (
        ("no neighbours recoverable", rep.unplaceable),
        ("empty arm output", rep.empty_output),
    ):
        if ids:
            emit(f"  dropped {len(ids)} ({label}): {', '.join(ids[:5])}…", level="status")


def judge_fit_pass(opts: FitOpts, view: ViewOpts, *, emit: Emitter = silent) -> str:
    """The `judge-fit` body: build the stratum, judge it, return the summary JSON."""
    if opts.a_path is None or opts.b_path is None:
        raise ConfigError("judge-fit needs both --a and --b", param_hint="--a")
    corpus = view.corpus_or_view(opts.pairs_path, required=True)
    captures = opts.captures_dir
    if captures is None and view.draws_dir is not None:
        captures = view.draws_dir.parent / "captures"
    triples, rep = fits_from_rewrites(
        opts.a_path,
        opts.b_path,
        corpus_path=corpus,
        cut=opts.cut(),
        captures_dir=captures,
        blog_root=opts.blog_root,
        field_a=opts.field_a,
        field_b=opts.field_b,
    )
    _report_stratum(rep, triples, emit)
    if not triples:
        raise ConfigError("no judgeable cases — check the corpus and --max-chunk-chars")
    if opts.dry_run:
        emit("dry run: no judge called", level="status")
        return ""

    judge = opts.judge
    if judge is None:
        config.require_key("OPENROUTER_API_KEY")
        judge = openrouter_fit_judge(model=opts.judge_model)
    out = opts.out_path or (opts.a_path.parent / "fits.jsonl")

    def progress(i: int, total: int) -> None:
        if i == 1 or i % 25 == 0 or i == total:
            emit(f"  ... {i}/{total} judged", level="status")

    result = judge_fits(
        triples,
        out,
        judge=judge,
        arm_a=opts.arm_a,
        arm_b=opts.arm_b,
        salt=opts.salt,
        limit=opts.limit,
        max_workers=opts.max_workers,
        on_progress=progress,
    )
    emit(
        f"judged {result.n_judged} (skipped {result.skipped_existing} already in {out})",
        level="status",
    )
    for err in result.errors[:5]:
        emit(f"  error: {err}", level="error")
    summary = summarize_fits(iter_jsonl(out), arm_a=opts.arm_a, arm_b=opts.arm_b)
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if opts.summary_path:
        opts.summary_path.parent.mkdir(parents=True, exist_ok=True)
        opts.summary_path.write_text(text + "\n", encoding="utf-8")
        emit(f"summary -> {opts.summary_path}", level="status")
    return text
