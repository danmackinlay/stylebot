"""Phase 4: inference — rewrite prose through the trained styler.

A `Styler` is the `Protocol` below, `styler(messages, num_samples=1) ->
list[str]` (D48); the bundled backends are `tinker_backend` and
`openai_backend`. `rewrite_text` chunks the way training saw data — prose
grouped to a chunk budget, never inside a protected block, whitespace
reassembled verbatim, frontmatter never sent — under `system`, the caller's
prompt (a run's manifest pins its text; D64). Best-of-N with a `scorer`
(lower = better) also arms the do-no-harm guard: the input chunk competes
(D20, D7).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from stylebot import markdown
from stylebot.ai_core import style_system
from stylebot.errors import ConfigError, MissingExtra
from stylebot.eval import detector_identity
from stylebot.jsonl import dumps_line

logger = logging.getLogger(__name__)

# Default inference chunk budget (chars); must match the training scale (D15).
STYLE_CHARS_PER_CHUNK = 1_500

DEFAULT_TEMPERATURE = 0.3
# Output budget per chunk, a floor: the backends fix max_tokens at
# construction, so the CLI scales it off the chunk budget via `max_tokens_for`.
DEFAULT_MAX_TOKENS = 2_000
# Output tokens per input char: chars/4 for the rewrite, plus headroom to expand.
_TOKENS_PER_CHAR = 0.75


def max_tokens_for(max_chunk_chars: int) -> int:
    """Output-token budget matching a chunk budget (never below the default)."""
    return max(DEFAULT_MAX_TOKENS, int(max_chunk_chars * _TOKENS_PER_CHAR))


@runtime_checkable
class Styler(Protocol):
    """The trained rewriter, as everything downstream of it sees one (D48):
    it types the call, not the tokens — parity is `backends.manifest_mismatches`
    and a measured sample (findings.md, "MLX local-serving parity")."""

    def __call__(self, messages: list[dict], num_samples: int = 1) -> list[str]:
        """Candidate assistant texts for one chat-message list."""
        ...


_FRONTMATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)


# --- Chunking — match training granularity, never slice a protected block ---


def split_frontmatter(text: str) -> tuple[str, str]:
    """`(frontmatter, body)` — the YAML block (with trailing newline) is
    protected verbatim; empty string when absent."""
    m = _FRONTMATTER_RE.match(text)
    return (m.group(0), text[m.end() :]) if m else ("", text)


def _blocks_with_separators(
    body: str, kinds: list[str | None]
) -> tuple[list[str], list[str], list[list[str | None]]]:
    """Split body into blocks on blank, unprotected line runs (D26, D27).

    `kinds` is `markdown.line_kinds(body)`, one label per line of
    `body.split("\\n")`. Returns `(blocks, seps, block_kinds)` with
    `len(seps) == len(blocks) - 1`; `blocks[0] + seps[0] + blocks[1] + ...`
    reproduces the body exactly, and `block_kinds[i]` is the kind list of
    `blocks[i].split("\\n")`.
    """
    if not body:
        return [], [], []
    # Alternating runs: content vs separator (blank, unprotected lines).
    runs: list[tuple[bool, list[str], list[str | None]]] = []  # (is_sep, lines, kinds)
    for line, kind in zip(body.split("\n"), kinds, strict=True):
        is_sep = kind is None and not line.strip()
        if runs and runs[-1][0] == is_sep:
            runs[-1][1].append(line)
            runs[-1][2].append(kind)
        else:
            runs.append((is_sep, [line], [kind]))
    # Fold a leading separator into the first block, a trailing one into the last.
    if runs and runs[0][0]:
        if len(runs) == 1:
            return ["\n".join(runs[0][1])], [], [runs[0][2]]
        runs[1] = (False, runs[0][1] + runs[1][1], runs[0][2] + runs[1][2])
        runs = runs[1:]
    if runs and runs[-1][0]:
        runs[-2] = (False, runs[-2][1] + runs[-1][1], runs[-2][2] + runs[-1][2])
        runs = runs[:-1]
    blocks = ["\n".join(lines) for is_sep, lines, _ in runs if not is_sep]
    block_kinds = [ks for is_sep, _, ks in runs if not is_sep]
    # A separator carries its boundary newlines, so concatenation is byte-exact.
    seps = ["\n" + "\n".join(lines) + "\n" for is_sep, lines, _ in runs if is_sep]
    return blocks, seps, block_kinds


def _segment(
    body: str,
    *,
    max_chars: int,
    inline_headings: bool = False,
    div_is_prose: markdown.DivIsProse | None = None,
    html_comments: bool = False,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Segment the body the way training saw data (D27); `html_comments`
    protects whole-line comments, off unless the caller's training did (D93).

    Returns `(items, seps)`, each item `(kind, text)` with kind in
    {"prose", "heading"} | `markdown.PROTECTED_KINDS`; items interleaved with
    seps reproduce the body exactly. Adjacent prose is grouped to <= max_chars;
    under `inline_headings` headings are groupable content, relabelled "prose"
    when grouped, while a lone heading keeps kind "heading".
    """
    kinds = markdown.line_kinds(body, div_is_prose=div_is_prose, html_comments=html_comments)
    blocks, seps, block_kinds = _blocks_with_separators(body, kinds)
    items: list[tuple[str, str]] = []
    iseps: list[str] = []
    for idx, (block, mask) in enumerate(zip(blocks, block_kinds, strict=True)):
        if idx:
            iseps.append(seps[idx - 1])
        # Run-length split on equal line kind; joining runs with "\n"
        # reproduces the block, so reassembly stays byte-exact.
        lines = block.split("\n")
        start = 0
        for i in range(1, len(lines) + 1):
            if i < len(lines) and mask[i] == mask[start]:
                continue
            if start:
                iseps.append("\n")
            items.append((mask[start] or "prose", "\n".join(lines[start:i])))
            start = i
    # Group adjacent groupable items within the budget, never across a protected one.
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


def prose_chunks(
    text: str,
    *,
    max_chunk_chars: int = STYLE_CHARS_PER_CHUNK,
    inline_headings: bool = False,
    div_is_prose: markdown.DivIsProse | None = None,
    html_comments: bool = False,
) -> list[str]:
    """The prose chunks `rewrite_text` would send to the styler: `_segment`'s
    prose items, frontmatter dropped. Public for `eval.chunked_detector` (D54)."""
    _, body = split_frontmatter(text)
    items, _ = _segment(
        body,
        max_chars=max_chunk_chars,
        inline_headings=inline_headings,
        div_is_prose=div_is_prose,
        html_comments=html_comments,
    )
    return [
        chunk
        for kind, chunk in items
        if kind not in markdown.PROTECTED_KINDS and kind != "heading" and chunk.strip()
    ]


# --- Rewrite ---


@dataclass
class RewriteResult:
    text: str
    n_chunks: int = 0
    n_candidates: int = 0  # total samples drawn (n_chunks * best_of)
    n_kept_input: int = 0  # chunks where no candidate beat the input (guard)
    n_anchor_rejected: int = 0  # samples disqualified for anchor damage, either cause
    n_anchor_lost: int = 0  # ... of those, samples that DROPPED a link/citation
    n_anchor_invented: int = 0  # ... of those, samples that CONJURED one (can overlap)
    n_truncated: int = 0  # samples disqualified for collapsing in length
    chunk_chars: list[tuple[int, int]] = field(default_factory=list)  # (in, out)
    decisions: list[str] = field(default_factory=list)  # one human line per chunk
    # Per-chunk anchor accounting, one entry per prose chunk (`write_anchor_audit`).
    anchor_audit: list[dict] = field(default_factory=list)


# A rewrite shorter than this fraction of its input stopped; a floor, not a
# band, set below the corpus's length-ratio tail (findings.md).
MIN_REWRITE_LENGTH_RATIO = 0.3


def is_truncated(
    source: str, candidate: str, *, min_ratio: float = MIN_REWRITE_LENGTH_RATIO
) -> bool:
    """True if `candidate` is too short to be a rewrite of `source` rather than
    a cut-off one. Empty source is never truncated (nothing to collapse from)."""
    if not source.strip():
        return False
    return len(candidate.strip()) < min_ratio * len(source.strip())


# Content anchors — link destinations, bare URLs, citation keys — checked
# before any scoring, since a rewrite can delete them and still score well (D32).
_ANCHOR_RES = (
    re.compile(r"\]\(([^)\s]+)"),  # markdown link/image destination
    re.compile(r"(?<!\()https?://[^\s)\]]*[^\s)\].,;:!?]"),  # bare URL, sans trailing punctuation
    re.compile(r"(?<![\w.\\])@[A-Za-z][A-Za-z0-9_:-]*"),  # citation key, not an email
)

# Inline markup the anchor guard protects because segmentation cannot (D27).
# The math pattern is tight on purpose: a false anchor rejects a good rewrite,
# a missed one costs nothing.
_INLINE_ANCHOR_RES = (
    re.compile(r"(?<!`)(`+)(?P<anchor>[^`\n]+?)\1(?!`)"),  # inline code span
    re.compile(r"(?<![\\$\w])\$(?![\s$])(?P<anchor>[^$\n]*?[^\s$])\$(?!\$)"),  # inline math
)


def content_anchors(text: str, *, inline: bool = False) -> dict[str, int]:
    """Multiset of link targets / URLs / citation keys in `text`; with
    `inline`, code spans and inline math too — a stricter guard (findings.md),
    opt-in where it decides and always on where it only flags (D27, D20)."""
    counts: dict[str, int] = {}
    patterns = _ANCHOR_RES + _INLINE_ANCHOR_RES if inline else _ANCHOR_RES
    for pattern in patterns:
        for m in pattern.finditer(text):
            if "anchor" in pattern.groupindex:
                key = m.group("anchor")
            else:
                key = m.group(1) if m.groups() else m.group(0)
            counts[key] = counts.get(key, 0) + 1
    return counts


def anchor_delta(
    source: str, candidate: str, *, inline: bool = False
) -> tuple[list[str], list[str]]:
    """`(lost, gained)` — anchors `candidate` dropped, and ones it conjured.
    Lost is a count decrease, gained is key absence (D32)."""
    have = content_anchors(candidate, inline=inline)
    want = content_anchors(source, inline=inline)
    lost = [key for key, n in want.items() if have.get(key, 0) < n]
    gained = [key for key in have if key not in want]
    return lost, gained


def invented_anchors(source: str, candidate: str, *, inline: bool = False) -> list[str]:
    """Anchors `candidate` carries that `source` never had at all."""
    return anchor_delta(source, candidate, inline=inline)[1]


def record_anchor_delta(record: dict, *, inline: bool = False) -> tuple[list[str], list[str]]:
    """`anchor_delta` over a pair record: `(lost, gained)` from its slop side to
    its target side."""
    from stylebot.pairs import extract_slop, extract_target

    return anchor_delta(extract_slop(record), extract_target(record), inline=inline)


def is_anchor_faithful(
    record: dict, *, lose: bool = True, gain: bool = True, inline: bool = False
) -> bool:
    """Pair-level `selector` for `train.assemble_training_corpus`: keep pairs
    whose target neither invents nor drops an anchor (D32); `lose` and `gain`
    select which halves bind. Wrap a keyword variant in a named function, not
    a `partial`: only `selector.__name__` reaches the manifest."""
    lost, gained = record_anchor_delta(record, inline=inline)
    return not (lose and lost) and not (gain and gained)


def _anchor_summary(keys: list[str]) -> str:
    """The first three distinct anchors, elided."""
    uniq = sorted(set(keys))
    return f"{', '.join(uniq[:3])}{'…' if len(uniq) > 3 else ''}"


def write_anchor_audit(
    path: str | Path, result: RewriteResult, *, source: str | None = None
) -> int:
    """Append a run's per-chunk anchor accounting to a JSONL sidecar (D32, D33):
    one line per prose chunk, anchor keys and lengths but never candidate
    text. Returns the number of lines written."""
    from stylebot.jsonl import append_jsonl
    from stylebot.provenance import now_iso

    stamp = now_iso()
    for entry in result.anchor_audit:
        append_jsonl(path, {"at": stamp, "source": source, **entry})
    return len(result.anchor_audit)


def detector_scorer(detector: Callable[[str], dict]) -> Callable[[str], float]:
    """Best-of-N scorer from the voice classifier: `P(slop)`, lower = better,
    blank text worst; the detector's identity rides along as `.identity` (D39)."""

    def score(text: str) -> float:
        return detector(text)["score"] if text.strip() else 1.0

    score.identity = detector_identity(detector)
    return score


def reranker_identity(scorer: object) -> dict | None:
    """`meta.reranker` for an output row: the instrument that chose the output
    (D39). A scorer chooses at any `best_of`; None means nothing chose."""
    return detector_identity(scorer) if scorer is not None else None


def rewrite_candidates(
    text: str,
    styler: Styler,
    *,
    context: str | None = None,
    best_of: int = 4,
    scorer: Callable[[str], float] | None = None,
    max_chunk_chars: int = STYLE_CHARS_PER_CHUNK,
    inline_headings: bool = False,
    div_is_prose: markdown.DivIsProse | None = None,
    system: str | None = None,
    html_comments: bool = False,
) -> dict:
    """All candidates for one prose chunk, none auto-chosen — the editor seam (D20).
    Anchor violations flag rather than disqualify; `text` must be prose at chunk
    scale (frontmatter, a heading, a protected block, a `:::` line or an
    over-budget chunk raises; D27, D76). Returns `{"input_score", "n_sampled",
    "candidates": [{"text", "score", "anchors_lost", "anchors_gained",
    "truncated", "identity"}, ...]}` in sampled order, duplicates collapsed."""
    from stylebot.pairs import build_pair_content, pair_body

    # Cross-heading pairs record no `meta.context`: a prefix training never saw (D27).
    if inline_headings:
        context = None
    frontmatter, body = split_frontmatter(text)
    if frontmatter:
        raise ConfigError("selection includes YAML frontmatter — select prose only")
    # Half a div resolves to no span (D76), so its fence line would reach the styler.
    if any(markdown.is_div_fence_line(line) for line in body.splitlines()):
        raise ConfigError("selection contains a div — select prose only")
    items, _seps = _segment(
        body,
        max_chars=max_chunk_chars,
        inline_headings=inline_headings,
        div_is_prose=div_is_prose,
        html_comments=html_comments,
    )
    kinds = [kind for kind, chunk in items if chunk.strip()]
    # Nothing selected is its own failure, not an over-budget selection.
    if not kinds:
        raise ConfigError("selection is empty — select some prose")
    # Headings are content under the cross-heading diet; fences never are.
    allowed = {"prose", "heading"} if inline_headings else {"prose"}
    if any(k not in allowed for k in kinds):
        bad = next(k for k in kinds if k not in allowed)
        raise ConfigError(f"selection contains a {bad} — select prose only")
    if len(kinds) != 1:
        raise ConfigError(
            f"selection is {len(body)} chars — over the {max_chunk_chars}-char "
            "chunk budget; select less (a paragraph or two)"
        )
    chunk = body.strip("\n")
    messages = [
        {"role": "system", "content": system or style_system()},
        {"role": "user", "content": build_pair_content(context, chunk)},
    ]
    sampled = styler(messages, num_samples=best_of)
    cleaned = [pair_body(c, context).strip() for c in sampled if c and c.strip()]
    seen: set[str] = set()
    deduped = [c for c in cleaned if c and not (c in seen or seen.add(c))]
    candidates = []
    for c in deduped:
        # Always inline here: this seam only flags, and a human picks.
        lost, gained = anchor_delta(chunk, c, inline=True)
        candidates.append(
            {
                "text": c,
                "score": scorer(c) if scorer is not None else None,
                "anchors_lost": lost,
                # Invention is the likelier failure and the hardest to catch by eye.
                "anchors_gained": gained,
                # Flagged, not filtered, like the anchor fields.
                "truncated": is_truncated(chunk, c),
                "identity": c == chunk,
            }
        )
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
    inline_anchors: bool = False,
    div_is_prose: markdown.DivIsProse | None = None,
    system: str | None = None,
    html_comments: bool = False,
) -> RewriteResult:
    """Rewrite prose chunk-by-chunk through the styler, mirroring training.

    Frontmatter, headings and protected blocks pass through (D27); a chunk
    carries the nearest heading as `context` unless `inline_headings`. A blank
    answer keeps the input; `system` is `messages[0]`. Candidates are
    disqualified truncated first, then for anchor damage (D32; `inline_anchors`
    widens the guard). With a `scorer` the input competes (D20) and
    `on_decision(chunk, context, candidates, scores, chosen_index, kept_input)`
    fires per chunk, `chosen_index` None when the input was kept.
    """
    from stylebot.pairs import build_pair_content, pair_body

    frontmatter, body = split_frontmatter(text)
    items, seps = _segment(
        body,
        max_chars=max_chunk_chars,
        inline_headings=inline_headings,
        div_is_prose=div_is_prose,
        html_comments=html_comments,
    )
    result = RewriteResult(text="")
    out_chunks: list[str] = []
    context: str | None = None
    for kind, chunk in items:
        if kind == "heading":
            # Under the cross-heading diet a heading is content, not `context`.
            if not inline_headings:
                context = chunk.split("\n")[-1]
            out_chunks.append(chunk)
            continue
        # A protected block does not reset `context`: `targets._split_sections`
        # never sees one, so a section's heading survives it.
        if kind in markdown.PROTECTED_KINDS or not chunk.strip():
            out_chunks.append(chunk)
            continue
        result.n_chunks += 1
        n = result.n_chunks
        messages = [
            {"role": "system", "content": system or style_system()},
            {"role": "user", "content": build_pair_content(context, chunk)},
        ]
        candidates = styler(messages, num_samples=best_of)
        result.n_candidates += len(candidates)
        raw = [pair_body(c, context).strip() for c in candidates if c and c.strip()]
        raw = [c for c in raw if c]
        # Truncation first (it CAUSES anchor loss), then anchor integrity — D32.
        lost: list[str] = []
        invented: list[str] = []
        n_lost = n_invented = 0  # SAMPLES, not anchors, for this chunk's decision line
        cut: list[float] = []  # length ratios of the truncated samples
        keep = []
        keep_idx: list[int] = []  # positions of `keep` entries within `raw`
        audit: list[dict] = []
        for i, c in enumerate(raw):
            if is_truncated(chunk, c):
                cut.append(len(c.strip()) / max(1, len(chunk.strip())))
                result.n_truncated += 1
                # No anchor verdict: a cut sample's delta measures the budget.
                audit.append({"truncated": True, "chars": len(c.strip())})
                continue
            gone, new = anchor_delta(chunk, c, inline=inline_anchors)
            audit.append({"chars": len(c.strip()), "lost": gone, "gained": new})
            if gone or new:
                lost.extend(gone)
                invented.extend(new)
                n_lost += bool(gone)
                n_invented += bool(new)
                result.n_anchor_rejected += 1
                result.n_anchor_lost += bool(gone)
                result.n_anchor_invented += bool(new)
            else:
                keep.append(c)
                keep_idx.append(i)
        result.anchor_audit.append(
            {
                "chunk": n,
                "chunk_chars": len(chunk.strip()),
                "inline": inline_anchors,
                "source_anchors": sorted(content_anchors(chunk, inline=inline_anchors)),
                "candidates": audit,
            }
        )
        raw_scores: list[float | None] = [None] * len(raw)
        chosen_idx: int | None = None
        if not keep:
            out = chunk
            result.n_kept_input += 1
            if cut and not lost:
                # Truncation and anchor loss want different fixes: say which.
                result.decisions.append(
                    f"chunk {n}: kept input (all {len(raw)} sample(s) truncated, "
                    f"shortest {min(cut):.0%} of input — raise --max-tokens?)"
                )
            elif lost or invented:
                # Side by side, never merged: they want different fixes (D32).
                bits = []
                if lost:
                    bits.append(f"{n_lost} sample(s) lost anchors: {_anchor_summary(lost)}")
                if invented:
                    bits.append(
                        f"{n_invented} sample(s) invented anchors: {_anchor_summary(invented)}"
                    )
                also = f", {len(cut)} truncated" if cut else ""
                result.decisions.append(f"chunk {n}: kept input ({'; '.join(bits)}{also})")
            else:
                logger.warning(
                    "styler returned nothing for a %d-char chunk; kept input", len(chunk)
                )
                result.decisions.append(f"chunk {n}: kept input (no usable sample)")
        elif scorer is None:
            out = keep[0]
            chosen_idx = keep_idx[0]
            result.decisions.append(f"chunk {n}: sample 1/{len(keep)}")
        else:
            input_score = scorer(chunk.strip())
            scores = [scorer(c) for c in keep]
            for i, s in zip(keep_idx, scores, strict=True):
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
            on_decision(chunk, context, list(raw), raw_scores, chosen_idx, chosen_idx is None)
        if out is not chunk:
            # Backends strip sampled text; restore the chunk's edge whitespace.
            lead = chunk[: len(chunk) - len(chunk.lstrip())]
            trail = chunk[len(chunk.rstrip()) :]
            out = lead + out.strip() + trail
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
    """Style the slop side of a pairs corpus into an output JSONL for eval: each
    record plus `"output"` (the rewrite of `messages[1]` as stored) and
    `meta.reranker`. Resumable by id. Returns the number written."""
    from stylebot.pairs import iter_pairs, record_id

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
    reranker = reranker_identity(scorer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        for i, rec in enumerate(todo):
            candidates = styler(rec["messages"][:2], num_samples=best_of)
            keep = [c for c in candidates if c and c.strip()]
            if not keep:
                logger.warning("styler returned nothing for %s; skipped", record_id(rec))
                continue
            output = keep[0] if len(keep) == 1 or scorer is None else min(keep, key=scorer)
            meta = {**(rec.get("meta") or {}), "reranker": reranker}
            fp.write(dumps_line({**rec, "output": output, "meta": meta}) + "\n")
            fp.flush()
            written += 1
            if on_progress is not None:
                on_progress(i + 1, len(todo))
    return written


# --- Backends ---


def _message_text(content: object) -> str:
    """Flatten a renderer `Message.content` — `str | list[ContentPart]` — to
    text; a thinking part is re-wrapped, not dropped (D75)."""
    if isinstance(content, str):
        return content
    out: list[str] = []
    for part in content or ():
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            out.append(part.get("text") or "")
        elif part.get("type") == "thinking":
            out.append(f"<think>{part.get('thinking') or ''}</think>")
    return "".join(out)


def tinker_backend(
    sampler_path: str,
    base_model: str,
    *,
    renderer_name: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Styler:
    """Sample the trained checkpoint via the Tinker sampling client, under the
    cookbook renderer training used (pass the manifest's `renderer_name`).
    Needs the ``trainer`` extra + TINKER_API_KEY (D19)."""
    try:
        import tinker
        from tinker import types
        from tinker_cookbook import model_info, renderers
        from tinker_cookbook.tokenizer_utils import get_tokenizer
    except ImportError as exc:  # pragma: no cover - needs the extra
        raise MissingExtra(
            "tinker_backend needs the 'trainer' extra: uv add 'stylebot[trainer]'"
        ) from exc

    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(
        renderer_name or model_info.get_recommended_renderer_name(base_model), tokenizer
    )
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    client = tinker.ServiceClient().create_sampling_client(model_path=sampler_path)

    def styler(messages: list[dict], num_samples: int = 1) -> list[str]:
        prompt = renderer.build_generation_prompt(list(messages))
        response = client.sample(prompt, num_samples=num_samples, sampling_params=params).result()
        outs = []
        for seq in response.sequences:
            message, _ok = renderer.parse_response(seq.tokens)
            outs.append(_message_text(message.get("content")).strip())
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
    """Any OpenAI-compatible endpoint; its chat template is uncheckable (D48),
    so score a parity sample first. `num_samples` is drawn as separate requests."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key or "unused")

    def styler(messages: list[dict], num_samples: int = 1) -> list[str]:
        outs = []
        for _ in range(num_samples):
            resp = client.chat.completions.create(
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outs.append((resp.choices[0].message.content or "").strip())
        return outs

    return styler
