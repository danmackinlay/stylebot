"""Phase 4: inference — rewrite prose through the trained styler.

Library-first (`ai-style run` is the thin CLI): a `Styler` is any callable

    styler(messages, num_samples=1) -> list[str]

taking chat messages (`[{role, content}, ...]`) and returning candidate
assistant texts. Two bundled backends:

- `tinker_backend(...)` — samples the Tinker checkpoint pinned in a training
  manifest, rendered with the SAME cookbook renderer training used (no
  train/serve chat-template skew possible). Needs the ``trainer`` extra +
  TINKER_API_KEY.
- `openai_backend(...)` — any OpenAI-compatible endpoint (Fireworks,
  `mlx_lm.server`, Osaurus, vLLM). NB the served model's chat template must
  match training's `qwen3_5_disable_thinking` rendering — verify parity by
  scoring the same inputs through both backends before trusting a new endpoint
  (the MLX exit did exactly that: `_plans/findings.md` "MLX local-serving
  parity" for the numbers and gates, `_plans/decisions.md` D8 for the caveats).

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

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from stylebot import markdown
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.jsonl import dumps_line

logger = logging.getLogger(__name__)

# Default inference chunk budget (chars). MUST match the caller's TRAINING
# chunk scale: at out-of-distribution lengths the adapter stops transforming
# and returns the input verbatim. Blog policy trains on <=1500-char merged
# paragraphs, so dan-style run passes its own MERGE_MAX_CHARS here.
STYLE_CHARS_PER_CHUNK = 1_500

DEFAULT_TEMPERATURE = 0.3
# Output budget per chunk — a FLOOR. It must clear the chunk budget with room
# to spare (a ~4500-char cross-heading chunk is ~1200 tokens) because truncation
# mid-rewrite is worse than no rewrite. The backends fix max_tokens at
# construction, so the CLI (`ai-style run` / `serve`) scales it off
# --max-chunk-chars through `max_tokens_for` unless --max-tokens is explicit.
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


# ---------------------------------------------------------------------------
# Chunking — match training granularity, never slice a protected block
# ---------------------------------------------------------------------------


def split_frontmatter(text: str) -> tuple[str, str]:
    """`(frontmatter, body)` — the YAML block (with trailing newline) is
    protected verbatim; empty string when absent."""
    m = _FRONTMATTER_RE.match(text)
    return (m.group(0), text[m.end() :]) if m else ("", text)


def _blocks_with_separators(
    body: str, kinds: list[str | None]
) -> tuple[list[str], list[str], list[list[str | None]]]:
    """Split body into blocks on blank-line runs that are OUTSIDE everything.

    `kinds` is `markdown.line_kinds(body)` — one label per line of
    `body.split("\\n")`, so the indices line up exactly. A line separates blocks
    only when it is blank *and* unprotected: a blank line inside a `:::` div or
    a display-math block belongs to that block.

    Returns `(blocks, seps, block_kinds)` with `len(seps) == len(blocks) - 1`;
    `blocks[0] + seps[0] + blocks[1] + ...` reproduces the body exactly.
    Leading/trailing blank runs stay attached to their neighbouring block, and
    `block_kinds[i]` is the per-line kind list of `blocks[i].split("\\n")`.

    The fence STATE MACHINE runs alongside the mask and is not redundant with
    it: `markdown.CODE_FENCE_RE` needs a closer, so an unclosed fence yields no
    span at all, and only the state machine's run-to-EOF behaviour keeps a
    runaway fence out of the styler's hands.
    """
    if not body:
        return [], [], []
    # Segment lines into alternating runs: content (any line inside a fence or
    # other protected block, or non-blank outside) vs separator (blank,
    # unprotected lines).
    runs: list[tuple[bool, list[str], list[str | None]]] = []  # (is_sep, lines, kinds)
    fence: str | None = None
    for line, mask in zip(body.split("\n"), kinds, strict=True):
        opener = markdown.FENCE_OPEN_RE.match(line)
        in_fence = fence is not None or opener is not None
        kind = mask if mask is not None else (markdown.FENCE if in_fence else None)
        is_sep = not in_fence and kind is None and not line.strip()
        if runs and runs[-1][0] == is_sep:
            runs[-1][1].append(line)
            runs[-1][2].append(kind)
        else:
            runs.append((is_sep, [line], [kind]))
        if not is_sep:
            if fence is None and opener:
                fence = opener.group("fence")
            elif fence is not None and line.strip().startswith(fence):
                fence = None
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
    # A separator string carries its boundary newlines, so plain concatenation
    # (block + sep + block + ...) reproduces the body byte-for-byte.
    seps = ["\n" + "\n".join(lines) + "\n" for is_sep, lines, _ in runs if is_sep]
    return blocks, seps, block_kinds


def _segment(
    body: str,
    *,
    max_chars: int,
    inline_headings: bool = False,
    div_is_prose: markdown.DivIsProse | None = None,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Segment the body the way TRAINING saw data.

    Returns `(items, seps)` where each item is `(kind, text)`, kind in
    {"prose", "heading"} | `markdown.PROTECTED_KINDS`; concatenating item texts
    interleaved with seps reproduces the body exactly. Everything the training
    pipeline drops before chunking — fenced code, display math, `:::` divs,
    blockquotes — is protected here too, per construct rather than under one
    flat "protected" label, so `rewrite_candidates` can tell an editor *which*
    thing it refused. Headings are protected as well (training pairs never
    contain them as content — they ride along as `meta.context`); adjacent prose
    blocks are grouped to <= max_chars.

    `inline_headings` matches the cross-heading diet: headings become groupable
    content, so a chunk spans sections exactly as the training pairs do. The
    other protected kinds stay protected either way — they are excluded from the
    corpus in both modes, so the styler has never been trained to reproduce one.
    Grouped heading items are relabelled "prose" (they are chunk content now); a
    heading that ends up alone keeps kind "heading", so callers can skip a bare
    header.
    """
    kinds = markdown.line_kinds(body, div_is_prose=div_is_prose)
    blocks, seps, block_kinds = _blocks_with_separators(body, kinds)
    items: list[tuple[str, str]] = []
    iseps: list[str] = []
    for idx, (block, mask) in enumerate(zip(blocks, block_kinds, strict=True)):
        if idx:
            iseps.append(seps[idx - 1])
        # Uniform run-length split on equal line kind. This is what replaced the
        # bespoke leading-heading split, and it is why a heading glued to prose,
        # a blockquote with no blank line after it, and a fence riding on the
        # end of a paragraph all come apart the same way. Joining consecutive
        # runs with "\n" reproduces the block, so reassembly stays byte-exact.
        lines = block.split("\n")
        start = 0
        for i in range(1, len(lines) + 1):
            if i < len(lines) and mask[i] == mask[start]:
                continue
            if start:
                iseps.append("\n")
            items.append((mask[start] or "prose", "\n".join(lines[start:i])))
            start = i
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
    n_anchor_rejected: int = 0  # samples disqualified for anchor damage, either cause
    n_anchor_lost: int = 0  # ... of those, samples that DROPPED a link/citation
    n_anchor_invented: int = 0  # ... of those, samples that CONJURED one (can overlap)
    n_truncated: int = 0  # samples disqualified for collapsing in length
    chunk_chars: list[tuple[int, int]] = field(default_factory=list)  # (in, out)
    decisions: list[str] = field(default_factory=list)  # one human line per chunk
    # Per-chunk anchor accounting, one entry per prose chunk — see
    # `write_anchor_audit`. Held rather than printed because the headline counts
    # answer "did this run go wrong" and this answers "how does the model behave".
    anchor_audit: list[dict] = field(default_factory=list)


# A rewrite shorter than this fraction of its input did not compress — it
# stopped. Checked BEFORE the anchor guard, because a truncated sample loses
# whatever anchors sat past the cut and would otherwise be filed as anchor
# damage, contaminating every "which constructs does this model break" number
# with "this model ran out of budget" (`_plans/placeholder-design-2026-08.md`,
# "Truncation, misattributed to the mask").
#
# 0.3, not the 0.9 that finding used: that was a copy-edit task, where output
# length tracks input closely enough for a 10% drop to be suspicious. A style
# rewrite is not that. Measured over the corpus, len(target)/len(slop) spans
# 0.18 to 9.76 — it leans compressive only ON AVERAGE (mean 0.81, median 0.79),
# and 13.6% of real rewrites EXPAND, 1.5% by more than half again. A 0.9 floor
# would call 70% of genuine Dan rewrites truncated.
#
# That two-sidedness is also why this is a floor and not a band: length alone
# cannot tell an expansive rewrite from a rambling one, so a ceiling here would
# be guessing. Collapse is the only end where length is diagnostic on its own.
# 0.3 sits below the 0.1th percentile — it flags 2 of 4,590 real rewrites.
MIN_REWRITE_LENGTH_RATIO = 0.3


def is_truncated(
    source: str, candidate: str, *, min_ratio: float = MIN_REWRITE_LENGTH_RATIO
) -> bool:
    """True if `candidate` is too short to be a rewrite of `source` rather than
    a cut-off one. Empty source is never truncated (nothing to collapse from)."""
    if not source.strip():
        return False
    return len(candidate.strip()) < min_ratio * len(source.strip())


# Content anchors — the mechanically-checkable information a style rewrite
# must never lose: markdown link destinations, bare URLs, citation keys. A
# rewrite can delete these while scoring WELL on voice, so anchor integrity is
# checked before any scoring, never after it.
_ANCHOR_RES = (
    re.compile(r"\]\(([^)\s]+)"),  # markdown link/image destination
    re.compile(r"(?<!\()https?://[^\s)\]]+"),  # bare URL
    re.compile(r"@[A-Za-z][A-Za-z0-9_:-]*"),  # Quarto/pandoc citation key
)

# Inline markup the styler must carry through verbatim: a code span or a piece
# of inline math is not prose and has no voice, but unlike a fenced block it
# lives INSIDE a chunk, so segmentation cannot protect it and only the anchor
# guard can. Both are single-line by construction: with DOTALL the code pattern
# happily swallows a whole fenced block.
#
# The math pattern is deliberately tight. The obvious `\$([^$\n]+?)\$` matches
# "it costs $5 and $10" and calls "5 and " an anchor, so every price in a
# paragraph becomes a false positive; requiring non-space at both ends kills it.
# Double-backtick spans are left unmatched on purpose — conservative, since a
# missed anchor costs nothing while a false one rejects a good rewrite.
_INLINE_ANCHOR_RES = (
    re.compile(r"(?<!`)(`+)(?P<anchor>[^`\n]+?)\1(?!`)"),  # inline code span
    re.compile(r"(?<![\\$\w])\$(?![\s$])(?P<anchor>[^$\n]*?[^\s$])\$(?!\$)"),  # inline math
)


def content_anchors(text: str, *, inline: bool = False) -> dict[str, int]:
    """Multiset of link targets / URLs / citation keys in `text`.

    With `inline`, code spans and inline math count too. That is a much stricter
    guard — measured over the 4624-pair corpus it takes the share of pairs where
    a real Dan rewrite would be disqualified from 6.4% to 19.1% — so it is
    opt-in for `rewrite_text` (which decides) and always on for
    `rewrite_candidates` (which only flags, and a human decides).
    """
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

    The guard was one-directional until 2026-08-12 and checked only the half the
    corpus does NOT teach: measured over the 4,595-pair corpus, 115 pairs (3.0%
    of the anchor-bearing ones) hand the styler a target that dropped an anchor,
    while 876 (19.1% of the corpus) hand it a target carrying anchors its source
    never had — half of those external URLs whose destination no model can know.
    Reverse-edit synthesis produces that directly: a generator that drops a link
    while "improving" a paragraph becomes, after the swap, an example reading
    *given text without links, produce text with these links*
    (`_plans/anchor-integrity.md`, stylebot#1).

    The two halves use deliberately different rules, because the harms are not
    mirror images:

    - LOST is a count DECREASE. Each occurrence anchors a separate claim, so
      dropping one of two citations of the same key still leaves a sentence
      standing on nothing.
    - GAINED is key ABSENCE, not a count increase. What this guards against is a
      destination the model invented and cannot have verified; repeating a URL
      the source supplied invents nothing, because it points somewhere the
      source already vouched for. A rewrite that splits one sentence in two and
      re-cites the same link is a legitimate edit, and a count rule would reject
      it.
    """
    have = content_anchors(candidate, inline=inline)
    want = content_anchors(source, inline=inline)
    lost = [key for key, n in want.items() if have.get(key, 0) < n]
    gained = [key for key in have if key not in want]
    return lost, gained


def missing_anchors(source: str, candidate: str, *, inline: bool = False) -> list[str]:
    """Anchors present in `source` that `candidate` lost (count decreased)."""
    return anchor_delta(source, candidate, inline=inline)[0]


def invented_anchors(source: str, candidate: str, *, inline: bool = False) -> list[str]:
    """Anchors `candidate` carries that `source` never had at all."""
    return anchor_delta(source, candidate, inline=inline)[1]


def invents_anchors(record: dict, *, inline: bool = False) -> bool:
    """True if a pair record's Dan-side target conjures anchors its slop lacks.

    The corpus-side face of `anchor_delta`: such a pair is an explicit lesson in
    fabricating links, so it is what the synth write-gate drops
    (`--drop-invented-anchors`) and what `is_anchor_faithful` filters out of a
    training assembly.
    """
    from stylebot.pairs import extract_slop, extract_target

    return bool(invented_anchors(extract_slop(record), extract_target(record), inline=inline))


def is_anchor_faithful(record: dict) -> bool:
    """Pair-level `selector` for `train.assemble_training_corpus`: keep pairs
    whose target invents no anchor.

    Named rather than a lambda because `assemble_training_corpus` records
    `selector.__name__` in the run manifest — a run filtered this way should say
    so. The corpus is a frozen record, so the 876 offending pairs stay on disk
    as evidence and are excluded from the training SIGNAL here instead of being
    migrated away (`_plans/anchor-integrity.md`, step 2).
    """
    return not invents_anchors(record)


def _anchor_summary(keys: list[str]) -> str:
    """The first three distinct anchors, elided — a decision line names enough
    to recognise the damage, not enough to reproduce it."""
    uniq = sorted(set(keys))
    return f"{', '.join(uniq[:3])}{'…' if len(uniq) > 3 else ''}"


def write_anchor_audit(
    path: str | Path, result: RewriteResult, *, source: str | None = None
) -> int:
    """Append a run's per-chunk anchor accounting to a JSONL sidecar.

    Step 0 of `_plans/anchor-integrity.md`, and the reason it comes first: the
    anchor counters were computed, printed to stderr and then discarded on every
    run, so the question the masking work is gated on — *how often does an
    UNMASKED candidate lose or invent an anchor?* — could never be answered
    without instrumenting a run specially. It is now answerable by joining these
    lines: the denominator is every candidate not marked `truncated`, the
    numerator is those with a non-empty `lost` or `gained`.

    Append-only and in the idiom of the reasoning sidecar: one line per prose
    chunk, carrying anchor KEYS and lengths but never candidate text, so the log
    stays joinable without becoming a second copy of the draft. Returns the
    number of lines written.
    """
    from stylebot.jsonl import append_jsonl
    from stylebot.provenance import now_iso

    stamp = now_iso()
    for entry in result.anchor_audit:
        append_jsonl(path, {"at": stamp, "source": source, **entry})
    return len(result.anchor_audit)


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
    div_is_prose: markdown.DivIsProse | None = None,
) -> dict:
    """All candidates for ONE prose chunk, none auto-chosen — the editor seam.

    Where `rewrite_text` decides (do-no-harm guard, anchors disqualify), this
    returns the full slate for a human to choose from: anchor violations FLAG
    (`anchors_lost`, `anchors_gained`) rather than disqualify, and the input's
    own score rides along so the caller can show the delta. Serves the `rewrite`
    op of `stylebot.serve` (see `_plans/decisions.md` D20).

    Single-chunk contract: `text` must be prose at training chunk scale —
    frontmatter, headings, or a protected block in the selection, or prose over
    `max_chunk_chars` (which `_segment` would split), raise ValueError with a
    message fit to surface in an editor, naming the construct
    ("selection contains a blockquote — select prose only"). An empty or
    whitespace-only selection gets its own message rather than the budget one:
    nothing selected is not too much selected.

    Under `inline_headings` (the cross-heading diet) a selection MAY span
    headings. Protected blocks still raise: they are absent from the corpus in
    both diets, so the styler has never learned to reproduce one and would be
    free to mangle it — and a styler quietly rewording somebody else's quoted
    words is the worst outcome this package has.

    Returns `{"input_score": float|None, "n_sampled": int,
              "candidates": [{"text", "score", "anchors_lost", "anchors_gained",
                              "truncated", "identity"}, ...]}`
    in sampled order (the caller owns presentation order — and should log it;
    see the presentation-bias note in the plan). Duplicate sample texts are
    collapsed, so `n_sampled` keeps the draw count honest, and a candidate
    byte-equal to the input is flagged `identity: true`.
    """
    from stylebot.pairs import build_pair_content, pair_body

    frontmatter, body = split_frontmatter(text)
    if frontmatter:
        raise ValueError("selection includes YAML frontmatter — select prose only")
    items, _seps = _segment(
        body,
        max_chars=max_chunk_chars,
        inline_headings=inline_headings,
        div_is_prose=div_is_prose,
    )
    kinds = [kind for kind, chunk in items if chunk.strip()]
    # Nothing selected is its own failure: it is not an over-budget selection,
    # and saying so (the old fall-through) told the user something false.
    if not kinds:
        raise ValueError("selection is empty — select some prose")
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
    candidates = []
    for c in deduped:
        # Always inline here: this seam only FLAGS, and a human picks. It is
        # precisely where a strict mask-preservation check earns its keep.
        lost, gained = anchor_delta(chunk, c, inline=True)
        candidates.append(
            {
                "text": c,
                "score": scorer(c) if scorer is not None else None,
                "anchors_lost": lost,
                # Invention is the likelier failure by ~6x in what the styler was
                # trained on, and the one an editor is least able to catch by
                # eye: a plausible URL reads exactly like a real one.
                "anchors_gained": gained,
                # Flagged, not filtered, like the anchor fields: a human picking
                # from the slate should be told "this one is cut off" in those
                # words rather than left to read it as a very terse rewrite.
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
) -> RewriteResult:
    """Rewrite prose chunk-by-chunk through the styler, mirroring training.

    Frontmatter, headings, and every protected block — fenced code, display
    math, `:::` divs, blockquotes — pass through verbatim, matching what
    `lib.editable_prose` drops before a target is ever cut (training pairs
    never contain them as content); each prose chunk is sent with the
    nearest preceding heading prepended exactly as `build_pair_content` did
    at capture time, and the heading prefix is stripped from the sample
    (`pair_body`) before reassembly. Original whitespace is reassembled
    verbatim; an empty/blank styler answer falls back to the input chunk
    (never destroy prose on a bad sample).

    `inline_headings` selects the cross-heading diet: headings become chunk
    content (rewritable, and no longer emitted as `context`), so chunks span
    sections. Set it to match the promoted adapter's training diet — the
    manifest records that under `data.chunk_scale`.

    A sample that collapsed below `MIN_REWRITE_LENGTH_RATIO` of the input is
    disqualified as TRUNCATED before anchors are consulted, and counted
    separately (`n_truncated`). Truncation causes anchor loss, so checking it
    second would file every cut-off sample as anchor damage; and it is the
    failure anchors cannot see at all when a chunk carries no link to lose.

    The anchor guard runs in BOTH directions: a candidate is disqualified for
    dropping a link/URL/citation the input had, and equally for inventing one
    the input never named (`anchor_delta`). `n_anchor_rejected` counts the
    disqualifications; `n_anchor_lost` / `n_anchor_invented` say which cause,
    and can overlap when one sample does both. Per-chunk detail lands in
    `result.anchor_audit` for `write_anchor_audit`.

    `inline_anchors` widens the anchor-integrity guard to inline code spans and
    inline math. Default off: measured across the 4624-pair corpus it triples
    the share of pairs where a genuine Dan rewrite is disqualified (6.4% ->
    19.1%), and a disqualified chunk is kept verbatim, so enabling it by default
    would trade a rare corruption for a common no-op.

    With a `scorer` (lower = better, e.g. `detector_scorer`), the INPUT chunk
    competes with the candidates — the do-no-harm guard: a chunk is only
    replaced by a rewrite the scorer rates strictly better. Each chunk's
    decision lands in `result.decisions` for the CLI to surface.

    `on_decision(chunk, context, candidates, scores, chosen_index, kept_input)`
    is called once per prose chunk after its decision — the preference-data
    seam. `candidates` are the cleaned sample bodies as considered, `scores`
    aligns with it (None for unscored/anchor-rejected samples), and
    `chosen_index` indexes `candidates` (None when the input was kept).
    """
    from stylebot.pairs import build_pair_content, pair_body

    frontmatter, body = split_frontmatter(text)
    items, seps = _segment(
        body,
        max_chars=max_chunk_chars,
        inline_headings=inline_headings,
        div_is_prose=div_is_prose,
    )
    result = RewriteResult(text="")
    out_chunks: list[str] = []
    context: str | None = None
    for kind, chunk in items:
        if kind == "heading":
            # Under the cross-heading diet a heading is chunk CONTENT, and those
            # pairs record no `meta.context` at all — so setting `context` here
            # would prepend a prefix to every following chunk that training
            # never saw. Defect A4: prospective today (the corpus has zero
            # cross-heading pairs) and live the moment the blog's
            # CROSS_HEADING policy produces its first batch.
            if not inline_headings:
                context = chunk.split("\n")[-1]
            out_chunks.append(chunk)
            continue
        # A protected block passes through and — deliberately — does NOT reset
        # `context`. `targets._split_sections` runs AFTER `editable_prose`, so
        # protected blocks are invisible to it and never break a section; the
        # heading a chunk was captured under must survive one in between.
        if kind in markdown.PROTECTED_KINDS or not chunk.strip():
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
        # Two disqualifications, in this order, both BEFORE any voice scoring.
        #
        # 1. TRUNCATION. A sample that collapsed in length did not compress, it
        #    stopped. It is checked first because it *causes* the second: a cut
        #    sample loses whatever anchors sat past the cut, so testing anchors
        #    first would file every truncation as anchor damage and make the
        #    anchor counts mean nothing. Truncation is also the case anchors
        #    CANNOT catch on their own — a chunk carrying no link or citation
        #    has no anchor to lose, so a 30%-length sample of it would sail
        #    through the guard and ship.
        # 2. ANCHOR INTEGRITY, in BOTH directions. A candidate that lost a
        #    link/URL/citation the input had must not ship, however Dan it
        #    sounds — and neither must one that invented a destination the input
        #    never named, which is the failure the training corpus actively
        #    teaches (see `anchor_delta`).
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
                # Recorded as truncation with no anchor verdict at all: a cut
                # sample's anchor delta measures the token budget, not the
                # model's link handling, and averaging the two together is
                # exactly what the audit exists to avoid.
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
                # Say truncation plainly rather than dressing it as anchor loss —
                # the two want different fixes (a bigger --max-tokens vs a better
                # sample), and conflating them sends the operator after the wrong one.
                result.decisions.append(
                    f"chunk {n}: kept input (all {len(raw)} sample(s) truncated, "
                    f"shortest {min(cut):.0%} of input — raise --max-tokens?)"
                )
            elif lost or invented:
                # Loss and invention want different fixes (a better sample vs a
                # model that stops making things up), and a chunk can suffer
                # both, so they are reported side by side rather than merged
                # into one "anchor damage" count.
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
            # Backends strip sampled text; restore the chunk's edge newline
            # runs so reassembly (and EOF newlines) stay byte-faithful.
            lead = chunk[: len(chunk) - len(chunk.lstrip("\n"))]
            trail = chunk[len(chunk.rstrip("\n")) :]
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        for i, rec in enumerate(todo):
            candidates = styler(rec["messages"][:2], num_samples=best_of)
            keep = [c for c in candidates if c and c.strip()]
            if not keep:
                logger.warning("styler returned nothing for %s; skipped", record_id(rec))
                continue
            output = keep[0] if len(keep) == 1 or scorer is None else min(keep, key=scorer)
            fp.write(dumps_line({**rec, "output": output}) + "\n")
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
    chat-template skew is possible. Needs the ``trainer`` extra + TINKER_API_KEY.
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
                model=model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            outs.append((resp.choices[0].message.content or "").strip())
        return outs

    return styler
