"""Phase 4: inference — rewrite prose through the trained styler.

Library-first (`ai-style run` is the thin CLI): a `Styler` is any callable

    styler(messages, num_samples=1) -> list[str]

taking chat messages (`[{role, content}, ...]`) and returning candidate
assistant texts. Two bundled backends:

- `tinker_backend(...)` — samples the Tinker checkpoint pinned in a training
  manifest, rendered with the SAME cookbook renderer training used (no
  train/serve chat-template skew possible). Needs the ``trainer`` extra +
  TINKER_API_KEY. The v1 default: zero serving infrastructure.
- `openai_backend(...)` — any OpenAI-compatible endpoint (Fireworks,
  `mlx_lm.server`, Osaurus, vLLM). The local/hosted exit. NB the served
  model's chat template must match training's `qwen3_5_disable_thinking`
  rendering — verify parity by scoring the same inputs through both backends
  before trusting a new endpoint (see phase-4 plan, MLX slice).

`rewrite_text` chunks at inference the way training saw data: blank-line
paragraph regions, grouped to ~STYLE_CHARS_PER_CHUNK chars, with chunk
boundaries never inside a code fence and the original inter-chunk whitespace
reassembled verbatim. A leading YAML frontmatter block is never sent to the
model. `STYLE_SYSTEM` travels verbatim (`stylebot.ai_core`) — the adapter was
trained on it; a different system prompt is out-of-distribution.

Best-of-N: pass `best_of > 1` and a `scorer` (lower = better, e.g. the voice
classifier via `detector_scorer`). The scorer also arms the do-no-harm guard:
the input chunk competes, so a rewrite ships only if scored strictly better.
Small N is Goodhart-gentle (KL ~ log N; see eval-harness.md "Eval vs reward").
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from stylebot.ai_core import STYLE_SYSTEM

logger = logging.getLogger(__name__)

# Default inference chunk budget (chars). MUST match the caller's TRAINING
# chunk scale: at out-of-distribution lengths the adapter falls back to
# copying the input (observed 2026-07-22 — a 5.4KB single-chunk rewrite came
# back byte-identical, while the same model transforms pair-scale chunks
# well). The old ~8k figure from ai-style-fine-tune.md predates the corpus
# chunking policy; blog policy trains on <=1500-char merged paragraphs, so
# dan-style run passes its own MERGE_MAX_CHARS here.
STYLE_CHARS_PER_CHUNK = 1_500

DEFAULT_TEMPERATURE = 0.3
# Output budget per chunk — a FLOOR, not the whole story. It must clear the
# chunk budget with room to spare: a ~4500-char cross-heading chunk is ~1200
# tokens, and truncation mid-rewrite is worse than no rewrite. The backends take
# max_tokens fixed at construction, so it is the CLI (`ai-style run` / `serve`)
# that scales it off --max-chunk-chars through `max_tokens_for` whenever
# --max-tokens is not given explicitly.
DEFAULT_MAX_TOKENS = 2_000
# Tokens allowed per character of input chunk (chars/4 for the rewrite, plus
# generous headroom since the styler may legitimately expand a terse passage).
_TOKENS_PER_CHAR = 0.75


def max_tokens_for(max_chunk_chars: int) -> int:
    """Output-token budget matching a chunk budget (never below the default)."""
    return max(DEFAULT_MAX_TOKENS, int(max_chunk_chars * _TOKENS_PER_CHAR))

# A styler: (messages, num_samples) -> candidate assistant texts.
Styler = Callable[..., list[str]]

_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCE_RE = re.compile(r"^(```+|~~~+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Chunking — match training granularity, never slice a fence
# ---------------------------------------------------------------------------


def split_frontmatter(text: str) -> tuple[str, str]:
    """`(frontmatter, body)` — the YAML block (with trailing newline) is
    protected verbatim; empty string when absent."""
    m = _FRONTMATTER_RE.match(text)
    return (m.group(0), text[m.end():]) if m else ("", text)


def _blocks_with_separators(body: str) -> tuple[list[str], list[str]]:
    """Split body into blocks on blank-line runs OUTSIDE code fences.

    Returns `(blocks, seps)` with `len(seps) == len(blocks) - 1`;
    `blocks[0] + seps[0] + blocks[1] + ...` reproduces the body exactly.
    Leading/trailing blank runs stay attached to their neighbouring block.
    """
    if not body:
        return [], []
    # Segment lines into alternating runs: content (any line inside a fence,
    # or non-blank outside) vs separator (blank lines outside fences).
    runs: list[tuple[bool, list[str]]] = []  # (is_sep, lines)
    fence: str | None = None
    for line in body.split("\n"):
        opener = _FENCE_RE.match(line)
        is_sep = fence is None and not line.strip()
        if runs and runs[-1][0] == is_sep:
            runs[-1][1].append(line)
        else:
            runs.append((is_sep, [line]))
        if not is_sep:
            if fence is None and opener:
                fence = opener.group(1)
            elif fence is not None and line.strip().startswith(fence):
                fence = None
    # Fold a leading separator into the first block, a trailing one into the last.
    if runs and runs[0][0]:
        if len(runs) == 1:
            return ["\n".join(runs[0][1])], []
        runs[1] = (False, runs[0][1] + runs[1][1])
        runs = runs[1:]
    if runs and runs[-1][0]:
        runs[-2] = (False, runs[-2][1] + runs[-1][1])
        runs = runs[:-1]
    blocks = ["\n".join(lines) for is_sep, lines in runs if not is_sep]
    # A separator string carries its boundary newlines, so plain concatenation
    # (block + sep + block + ...) reproduces the body byte-for-byte.
    seps = ["\n" + "\n".join(lines) + "\n" for is_sep, lines in runs if is_sep]
    return blocks, seps


_HEADING_RE = re.compile(r"^#{1,6}\s")


def _segment(
    body: str, *, max_chars: int, inline_headings: bool = False
) -> tuple[list[tuple[str, str]], list[str]]:
    """Segment the body the way TRAINING saw data.

    Returns `(items, seps)` where each item is `(kind, text)`, kind in
    {"prose", "heading", "fence"}; concatenating item texts interleaved with
    seps reproduces the body exactly. Headings and code fences are protected
    (training pairs never contain them as content — headings ride along as
    `meta.context`); adjacent prose blocks are grouped to <= max_chars.

    `inline_headings` matches the v2 cross-heading diet: headings become
    groupable content, so a chunk spans sections exactly as the training pairs
    do. Code fences stay protected either way — they are excluded from the
    corpus in both modes, so the styler has never been trained to reproduce
    one. Grouped heading items are relabelled "prose" (they are chunk content
    now); a heading that ends up alone keeps kind "heading" so callers can
    still skip a bare trailing header rather than ask for a one-line rewrite.
    """
    blocks, seps = _blocks_with_separators(body)
    items: list[tuple[str, str]] = []
    iseps: list[str] = []
    for idx, block in enumerate(blocks):
        if idx:
            iseps.append(seps[idx - 1])
        lines = block.split("\n")
        n_head = 0
        while n_head < len(lines) and _HEADING_RE.match(lines[n_head]):
            n_head += 1
        if n_head and n_head < len(lines):
            # A heading glued to prose with no blank line: split it off.
            items.append(("heading", "\n".join(lines[:n_head])))
            iseps.append("\n")
            rest = "\n".join(lines[n_head:])
            items.append(("fence" if _FENCE_RE.match(rest) else "prose", rest))
        elif n_head:
            items.append(("heading", block))
        elif _FENCE_RE.match(block):
            items.append(("fence", block))
        else:
            items.append(("prose", block))
    # Group ADJACENT groupable items within the budget (never across a
    # protected item). Under inline_headings, headings are groupable too, so a
    # chunk spans sections the way a cross-heading training pair does.
    groupable = {"prose", "heading"} if inline_headings else {"prose"}

    out_items: list[tuple[str, str]] = []
    out_seps: list[str] = []
    for i, item in enumerate(items):
        sep = iseps[i - 1] if i else None
        if (
            sep is not None
            and out_items
            and item[0] in groupable
            and out_items[-1][0] in groupable
            and len(out_items[-1][1]) + len(sep) + len(item[1]) <= max_chars
        ):
            # A group containing anything is chunk content, hence "prose".
            out_items[-1] = ("prose", out_items[-1][1] + sep + item[1])
        else:
            if sep is not None:
                out_seps.append(sep)
            out_items.append(item)
    return out_items, out_seps


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


@dataclass
class RewriteResult:
    text: str
    n_chunks: int = 0
    n_candidates: int = 0  # total samples drawn (n_chunks * best_of)
    n_kept_input: int = 0  # chunks where no candidate beat the input (guard)
    n_anchor_rejected: int = 0  # samples disqualified for losing links/citations
    chunk_chars: list[tuple[int, int]] = field(default_factory=list)  # (in, out)
    decisions: list[str] = field(default_factory=list)  # one human line per chunk


# Content anchors — the mechanically-checkable information a style rewrite
# must never lose: markdown link destinations, bare URLs, citation keys. The
# run-1 failure was exactly this (a rewrite deleted a link + a citation while
# scoring WELL on voice), so anchor integrity is checked before any scoring.
_ANCHOR_RES = (
    re.compile(r"\]\(([^)\s]+)"),          # markdown link/image destination
    re.compile(r"(?<!\()https?://[^\s)\]]+"),  # bare URL
    re.compile(r"@[A-Za-z][A-Za-z0-9_:-]*"),   # Quarto/pandoc citation key
)


def content_anchors(text: str) -> dict[str, int]:
    """Multiset of link targets / URLs / citation keys in `text`."""
    counts: dict[str, int] = {}
    for pattern in _ANCHOR_RES:
        for m in pattern.finditer(text):
            key = m.group(1) if m.groups() else m.group(0)
            counts[key] = counts.get(key, 0) + 1
    return counts


def missing_anchors(source: str, candidate: str) -> list[str]:
    """Anchors present in `source` that `candidate` lost (count decreased)."""
    have = content_anchors(candidate)
    return [
        key for key, n in content_anchors(source).items()
        if have.get(key, 0) < n
    ]


def detector_scorer(detector: Callable[[str], dict]) -> Callable[[str], float]:
    """Best-of-N scorer from the voice classifier: lower = more Dan
    (`P(slop)`; blank text scores worst). `detector` is the
    `stylebot.classify` seam."""

    def score(text: str) -> float:
        return detector(text)["score"] if text.strip() else 1.0

    return score


def rewrite_candidates(
    text: str,
    styler: Styler,
    *,
    context: str | None = None,
    best_of: int = 4,
    scorer: Callable[[str], float] | None = None,
    max_chunk_chars: int = STYLE_CHARS_PER_CHUNK,
    inline_headings: bool = False,
) -> dict:
    """All candidates for ONE prose chunk, none auto-chosen — the editor seam.

    Where `rewrite_text` decides (do-no-harm guard, anchors disqualify), this
    returns the full slate for a human to choose from: anchor violations FLAG
    (`anchors_lost`) rather than disqualify, and the input's own score rides
    along so the caller can show the delta. Serves the `rewrite` op of
    `stylebot.serve` (see `_plans/slice-vscode-rewrite.md`).

    Single-chunk contract: `text` must be prose at training chunk scale —
    frontmatter, headings, or fences in the selection, or prose over
    `max_chunk_chars` (which `_segment` would split), raise ValueError with a
    message fit to surface in an editor.

    Under `inline_headings` (the v2 cross-heading diet) a selection MAY span
    headings — that is the editor's "rewrite section" mode, and it is exactly
    what the adapter was trained on. Fences still raise: they are absent from
    the corpus in both diets, so the styler has never learned to reproduce one
    and would be free to mangle it. The size limit still applies, since a
    selection `_segment` would split into several chunks is not one rewrite.

    Returns `{"input_score": float|None, "n_sampled": int,
              "candidates": [{"text", "score", "anchors_lost", "identity"},
                             ...]}`
    in sampled order (the caller owns presentation order — and should log it;
    see the presentation-bias note in the plan). Duplicate sample texts are
    collapsed (low temperature mode-collapses freely — `n_sampled` keeps the
    draw count honest) and a candidate byte-equal to the input is flagged
    `identity: true` — the styler declining to change already-Dan prose is a
    do-no-harm answer, not a candidate worth presenting as new.
    """
    from stylebot.eval import pair_body
    from stylebot.pairs import build_pair_content

    frontmatter, body = split_frontmatter(text)
    if frontmatter:
        raise ValueError("selection includes YAML frontmatter — select prose only")
    items, _seps = _segment(body, max_chars=max_chunk_chars, inline_headings=inline_headings)
    kinds = [kind for kind, chunk in items if chunk.strip()]
    # Headings are content under the cross-heading diet; fences never are.
    allowed = {"prose", "heading"} if inline_headings else {"prose"}
    if any(k not in allowed for k in kinds):
        bad = next(k for k in kinds if k not in allowed)
        raise ValueError(f"selection contains a {bad} — select prose only")
    if len(kinds) != 1:
        raise ValueError(
            f"selection is {len(body)} chars — over the {max_chunk_chars}-char "
            "chunk budget; select less (a paragraph or two)"
        )
    chunk = body.strip("\n")
    messages = [
        {"role": "system", "content": STYLE_SYSTEM},
        {"role": "user", "content": build_pair_content(context, chunk)},
    ]
    sampled = styler(messages, num_samples=best_of)
    cleaned = [pair_body(c, context).strip() for c in sampled if c and c.strip()]
    seen: set[str] = set()
    deduped = [c for c in cleaned if c and not (c in seen or seen.add(c))]
    candidates = [
        {
            "text": c,
            "score": scorer(c) if scorer is not None else None,
            "anchors_lost": missing_anchors(chunk, c),
            "identity": c == chunk,
        }
        for c in deduped
    ]
    return {
        "input_score": scorer(chunk) if scorer is not None else None,
        "n_sampled": len(sampled),
        "candidates": candidates,
    }


def rewrite_text(
    text: str,
    styler: Styler,
    *,
    max_chunk_chars: int = STYLE_CHARS_PER_CHUNK,
    best_of: int = 1,
    scorer: Callable[[str], float] | None = None,
    on_decision: Callable[..., None] | None = None,
    inline_headings: bool = False,
) -> RewriteResult:
    """Rewrite prose chunk-by-chunk through the styler, mirroring training.

    Frontmatter, headings, and code fences pass through verbatim (training
    pairs never contain them as content); each prose chunk is sent with the
    nearest preceding heading prepended exactly as `build_pair_content` did
    at capture time, and the heading prefix is stripped from the sample
    (`pair_body`) before reassembly. Original whitespace is reassembled
    verbatim; an empty/blank styler answer falls back to the input chunk
    (never destroy prose on a bad sample).

    `inline_headings` selects the v2 cross-heading diet: headings become chunk
    content (rewritable, and no longer emitted as `context`), so chunks span
    sections. Set it to match the promoted adapter's training diet — the
    manifest records that under `data.chunk_scale`.

    With a `scorer` (lower = better, e.g. `detector_scorer`), the INPUT chunk
    competes with the candidates — the do-no-harm guard: a chunk is only
    replaced by a rewrite the scorer rates strictly better, so best-of-N can
    conclude "leave it alone". Each chunk's decision lands in
    `result.decisions` for the CLI to surface.

    `on_decision(chunk, context, candidates, scores, chosen_index,
    kept_input)` is called once per prose chunk after its decision — the
    preference-data seam (best-of choices are chosen-vs-rejected pairs on
    real inputs). `candidates` are the cleaned sample bodies as considered,
    `scores` aligns with it (None for unscored/anchor-rejected samples), and
    `chosen_index` indexes `candidates` (None when the input was kept).
    """
    from stylebot.eval import pair_body
    from stylebot.pairs import build_pair_content

    frontmatter, body = split_frontmatter(text)
    items, seps = _segment(body, max_chars=max_chunk_chars, inline_headings=inline_headings)
    result = RewriteResult(text="")
    out_chunks: list[str] = []
    context: str | None = None
    for kind, chunk in items:
        if kind == "heading":
            context = chunk.split("\n")[-1]
            out_chunks.append(chunk)
            continue
        if kind == "fence" or not chunk.strip():
            out_chunks.append(chunk)
            continue
        result.n_chunks += 1
        n = result.n_chunks
        messages = [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": build_pair_content(context, chunk)},
        ]
        candidates = styler(messages, num_samples=best_of)
        result.n_candidates += len(candidates)
        raw = [pair_body(c, context).strip() for c in candidates if c and c.strip()]
        raw = [c for c in raw if c]
        # Anchor integrity: a candidate that lost a link/URL/citation the
        # input had is disqualified BEFORE any voice scoring — style rewrites
        # must not delete information, however Dan they sound.
        lost: list[str] = []
        keep = []
        keep_idx: list[int] = []  # positions of `keep` entries within `raw`
        for i, c in enumerate(raw):
            gone = missing_anchors(chunk, c)
            if gone:
                lost.extend(gone)
                result.n_anchor_rejected += 1
            else:
                keep.append(c)
                keep_idx.append(i)
        raw_scores: list[float | None] = [None] * len(raw)
        chosen_idx: int | None = None
        if not keep:
            out = chunk
            result.n_kept_input += 1
            if lost:
                uniq = sorted(set(lost))
                result.decisions.append(
                    f"chunk {n}: kept input (all {len(raw)} sample(s) lost anchors: "
                    f"{', '.join(uniq[:3])}{'…' if len(uniq) > 3 else ''})"
                )
            else:
                logger.warning("styler returned nothing for a %d-char chunk; kept input", len(chunk))
                result.decisions.append(f"chunk {n}: kept input (no usable sample)")
        elif scorer is None:
            out = keep[0]
            chosen_idx = keep_idx[0]
            result.decisions.append(f"chunk {n}: sample 1/{len(keep)}")
        else:
            input_score = scorer(chunk.strip())
            scores = [scorer(c) for c in keep]
            for i, s in zip(keep_idx, scores):
                raw_scores[i] = s
            best = min(range(len(scores)), key=scores.__getitem__)
            if scores[best] < input_score:
                out = keep[best]
                chosen_idx = keep_idx[best]
                result.decisions.append(
                    f"chunk {n}: sample {best + 1}/{len(keep)} "
                    f"(score {scores[best]:.2f} < input {input_score:.2f})"
                )
            else:
                out = chunk
                result.n_kept_input += 1
                result.decisions.append(
                    f"chunk {n}: kept input (best sample {min(scores):.2f} "
                    f">= input {input_score:.2f})"
                )
        if on_decision is not None:
            on_decision(chunk, context, list(raw), raw_scores, chosen_idx,
                        chosen_idx is None)
        if out is not chunk:
            # Backends strip sampled text; restore the chunk's edge newline
            # runs so reassembly (and EOF newlines) stay byte-faithful.
            lead = chunk[: len(chunk) - len(chunk.lstrip("\n"))]
            trail = chunk[len(chunk.rstrip("\n")):]
            out = lead + out.strip("\n") + trail
        result.chunk_chars.append((len(chunk), len(out)))
        out_chunks.append(out)
    pieces = [frontmatter]
    for i, chunk in enumerate(out_chunks):
        if i:
            pieces.append(seps[i - 1])
        pieces.append(chunk)
    result.text = "".join(pieces)
    return result


def rewrite_pairs_file(
    pairs_path: str | Path,
    out_path: str | Path,
    styler: Styler,
    *,
    limit: int | None = None,
    selector: Callable[[dict], bool] | None = None,
    best_of: int = 1,
    scorer: Callable[[str], float] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    """Style the slop side of a pairs corpus into an output JSONL for eval.

    Each output line is the pair record plus a top-level `"output"` field (the
    styler's rewrite of `messages[1]` as stored, heading context included —
    the eval extractor strips it like every other field). Resumable: records
    whose id is already in `out_path` are skipped. Returns the number written.
    """
    from stylebot.eval import record_id
    from stylebot.pairs import iter_pairs

    out_path = Path(out_path)
    done = {rec.get("id") or record_id(rec) for rec in iter_pairs(out_path)}
    todo = []
    for rec in iter_pairs(pairs_path):
        if selector is not None and not selector(rec):
            continue
        if record_id(rec) in done:
            continue
        todo.append(rec)
        if limit is not None and len(todo) >= limit:
            break

    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        for i, rec in enumerate(todo):
            candidates = styler(rec["messages"][:2], num_samples=best_of)
            keep = [c for c in candidates if c and c.strip()]
            if not keep:
                logger.warning("styler returned nothing for %s; skipped", record_id(rec))
                continue
            if len(keep) == 1 or scorer is None:
                output = keep[0]
            else:
                output = min(keep, key=scorer)
            fp.write(json.dumps({**rec, "output": output}, ensure_ascii=False) + "\n")
            fp.flush()
            written += 1
            if on_progress is not None:
                on_progress(i + 1, len(todo))
    return written


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def tinker_backend(
    sampler_path: str,
    base_model: str,
    *,
    renderer_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Styler:
    """Sample the trained checkpoint via the Tinker sampling client.

    Uses the SAME cookbook renderer as training (default: the model's
    recommended one — pass the manifest's `renderer_name` to be exact), so no
    chat-template skew is possible. Needs the ``trainer`` extra and
    TINKER_API_KEY; ~$0.60/1M sampled tokens for a 9B-class model.
    """
    try:
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:  # pragma: no cover - needs the extra
        raise ImportError(
            "tinker_backend needs the 'trainer' extra: uv add 'stylebot[trainer]'"
        ) from exc

    from stylebot.train import _patch_pyqwest_tls_trust

    _patch_pyqwest_tls_trust()
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(
        renderer_name or model_info.get_recommended_renderer_name(base_model), tokenizer
    )
    params = types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    client = tinker.ServiceClient().create_sampling_client(model_path=sampler_path)

    def styler(messages: list[dict], num_samples: int = 1) -> list[str]:
        prompt = renderer.build_generation_prompt(list(messages))
        response = client.sample(
            prompt, num_samples=num_samples, sampling_params=params
        ).result()
        outs = []
        for seq in response.sequences:
            message, _ok = renderer.parse_response(seq.tokens)
            outs.append((message.get("content") or "").strip())
        return outs

    return styler


def openai_backend(
    base_url: str,
    model: str,
    *,
    api_key: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Styler:
    """Any OpenAI-compatible endpoint: Fireworks, `mlx_lm.server`, Osaurus,
    vLLM. The served model's chat template must match training's rendering —
    score a parity sample against `tinker_backend` before trusting a new
    endpoint. `num_samples` is drawn as separate requests (the `n` parameter
    is not universally supported by local servers)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "unused")

    def styler(messages: list[dict], num_samples: int = 1) -> list[str]:
        outs = []
        for _ in range(num_samples):
            resp = client.chat.completions.create(
                model=model, messages=list(messages),
                temperature=temperature, max_tokens=max_tokens,
            )
            outs.append((resp.choices[0].message.content or "").strip())
        return outs

    return styler
