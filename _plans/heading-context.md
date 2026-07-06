# Design — heading context for training pairs

## Status — 🔧 BUILT (2026-06-28)

Implemented in **both** producers on the shared contract, default behaviour
gated so nothing changed until opted in:

- **Contract:** `stylebot.pairs.build_pair_content(context, body)` + `meta.context`
  / `meta.context_mode`; `validate_pair_record` enforces the heading is the
  verbatim prefix of both sides. Existing 254 pairs (no `meta.context`) stay valid.
- **Phase 2:** `Target.context`; `iter_targets(heading_context="immediate")` (a
  unified section-aware path); `synthesize_pairs` generates slop from the body
  only, `synth_key` includes context, `context_dropout` (deterministic) keeps a
  fraction heading-less; CLI `--heading-context` / `--context-dropout`;
  `stylebot.report` shows the heading. Blog policy: `HEADING_CONTEXT="immediate"`,
  `CONTEXT_DROPOUT=0.1` in `livingthing.training_targets`.
- **Phase 1:** `ai-style-log` `diff_chunks(..., heading_context=)` resolves the
  nearest preceding heading; default **ON** (`--no-heading-context` to opt out);
  `--whole`/preamble carry none; forward-only.
- Tested: cross-producer pairs are byte-identical for the same heading. 85% of
  real-blog passages carry a heading.

**Deferred (the "Open choices" below):** breadcrumb depth (only `immediate`
shipped), title/subtitle as context for preamble, and tuning `context_dropout`.
The design below is the original spec, kept for rationale.

## Motivation

Dan leans on section headings to set context for the paragraphs beneath them, so
many target passages are under-determined in isolation. Measured on the live
blog: **85% of merged passages sit under a heading** (15% preamble), and a large
share open with a back-referent the heading supplies — e.g. heading
*"It's not what you know it's who you know"* → *"It is hard to find an
explanation for this kind of behaviour in terms of miasmas…"*.

We want the heading attached as **fixed context** — identical on both sides of
the pair, never rewritten — so the styler learns to *restyle the body
conditioned on the heading*, matching how Dan writes and how he'll feed real
drafts at inference. This spans both pair producers: Phase 2 (synthetic) and
Phase 1 (real capture). They MUST emit the same shape.

## The shared contract (the seam — lock this first)

A pair may carry an optional **`context`** string: the section heading, or the
ancestor heading breadcrumb, verbatim from the source. It is prepended to **both**
message bodies, byte-identical, keeping the frozen 3-message schema:

```
user.content      = build_pair_content(context, slop_or_before)
assistant.content = build_pair_content(context, dan_or_after)
# build_pair_content(ctx, body) = f"{ctx}\n\n{body}" if ctx else body
```

- `meta.context` = the context string (omit/empty when none).
- `meta.context_mode` ∈ `{"immediate", "breadcrumb"}` for provenance.
- `before_chars`/`after_chars` keep counting the **body** (the transform region).

**Implementation of the lock:** add `build_pair_content(context, body) -> str` to
`stylebot.pairs` (which already owns the schema/validator). Both
`stylebot.bin.ai_style_log` and `stylebot.synth` import it, so the two producers
cannot diverge. Optionally extend `validate_pair_record` with a soft invariant:
*if `meta.context` is non-empty, both user and assistant content must start with
it* — catches accidental heading-rewriting.

Why prepend-to-content (not a 4th message): preserves the
`[system, user, assistant]` triple that `validate_pairs_file` and Phase-3
training depend on. The heading is real leading markdown; `STYLE_SYSTEM` already
says preserve headings verbatim, so identical context teaches "copy the heading,
restyle the body." `validate_pairs_file` stays valid unchanged.

## Phase 2 — synthetic pairs (`stylebot.synth`)

1. `Target` gains `context: str = ""` (body stays in `.text`).
2. New extractor `_split_sections_with_context(text, *, mode) -> list[(context, body)]`
   (supersedes `_split_sections`): walk the heading stack; per section return
   `(context, body)` where context is the immediate heading (`mode="immediate"`)
   or the `\n`-joined ancestor breadcrumb (`mode="breadcrumb"`); preamble →
   `context=""`. Non-merge path attaches the nearest preceding heading per
   paragraph the same way.
3. `iter_targets` gains `heading_context: "none" | "immediate" | "breadcrumb" =
   "none"` (default `none` ⇒ existing tests/CLI unchanged). When set, populate
   `Target.context`.
4. `synthesize_pairs`: generate slop from `target.text` **only** (the body —
   guarantees the heading is never paraphrased), then assemble both messages via
   `build_pair_content`. Record `meta.context`/`context_mode`. **`synth_key` must
   include the context** (`hash(generator, context, body)`) so changing context
   regenerates and resume stays correct.
5. Optional `context_dropout: float` (deterministic by `hash(body) % 100`) keeps
   a fraction heading-less so the model doesn't *require* a heading; the natural
   15% preamble already provides some.
6. CLI `ai-style synth --heading-context [none|immediate|breadcrumb]`; `report.py`
   shows the context (prefix/column) so heading+passage units are eyeball-able.
7. Blog policy (`livingthing.training_targets`): add `HEADING_CONTEXT =
   "breadcrumb"` (or `"immediate"`), wired by `dan-style synth`.

## Phase 1 — real capture (`stylebot.bin.ai_style_log`)

Add the **same** context to captured edit pairs (faithfulness + consistency with
Phase 2):

1. `diff_chunks(before, after)` → return the after-paragraph start index per pair
   (e.g. yield `(before_chunk, after_chunk, j1)`), so the caller can map each
   changed region to its nearest preceding heading among `after_paras`. Add
   `_nearest_heading(paras, idx, *, mode)`.
2. `_build_pair_record` gains `context: str`; prepend via `build_pair_content`;
   record `meta.context`/`context_mode`. Thread through `_capture_and_append`.
   `--whole` mode has no single heading → no context (unchanged).
3. Compute context from the **after** body (the kept text); if the before body's
   nearest heading differs, prefer after's (the edit may have moved a heading) —
   record both only if they diverge (rare).
4. Flag `ai-style-log save|pair --heading-context/--no-heading-context`.

### Back-compat ↔ faithfulness trade-off

- **Existing 254 pairs** were captured context-less. **Forward-only**: leave them
  as-is; they're distinguishable by the absence of `meta.context`. A corpus mixing
  heading-prefixed and bare pairs is fine — Phase-3 training handles both, and the
  mix is itself useful augmentation. No risky retro-fit (source files have since
  changed; re-deriving context by text-matching is fragile).
- `STYLE_SYSTEM` (frozen) is **untouched** — context lives in user/assistant
  content, so the frozen-prompt invalidation rule does not fire. The only
  inconsistency is structural (old bare vs new prefixed), softened by `meta`.
- **Default recommendation: ON** for new captures (the daily logger is the main
  real-pair source going forward and the corpus is still young), with `--no-…` to
  opt out and `meta.context` versioning. Alternative if Dan prefers zero change to
  his daily flow: default OFF, opt in per save.

## Rollout & parallelization

1. **Step 0 — lock the contract** (`stylebot.pairs.build_pair_content` + the
   `meta.context` convention + optional validator invariant + tests). Small,
   shared, blocks both. **Do this first.**
2. Phase 2 (touches `synth.py`, `report.py`, `bin/ai_style.py`, blog policy) and
   Phase 1 (touches `bin/ai_style_log.py`) edit **disjoint files** → safe to run
   **in parallel** once Step 0 lands. Recommended: land Step 0 + Phase 2 first
   (the report validates the design and the active curation work is here), then
   Phase 1 — or **dispatch an agent for Phase 1** against the locked contract
   concurrently. Parallel is only safe *after* Step 0; otherwise the two diverge
   on the seam.

## Open choices (need Dan's call)

- **Breadcrumb vs immediate** heading (recommend `breadcrumb` for faithfulness;
  compare via the report before committing the corpus).
- **Title/subtitle as context for the 15% preamble** (recommend OFF initially —
  avoids injecting structure not in the body).
- **`context_dropout`** fraction (recommend ~0.1, or rely on the natural 15%).
- **Phase-1 default** ON (faithfulness) vs OFF (zero change to daily flow).

## Verification (no API spend)

- Phase 2: unit tests for `_split_sections_with_context` (immediate/breadcrumb,
  preamble), pair assembly (context byte-identical both sides; slop generated from
  body only), `synth_key` includes context, validator invariant; `--dry-run
  --report` on the real blog shows heading+passage; `iter_targets` defaults
  (`heading_context="none"`) leave the current 56 tests green.
- Phase 1: tests for `_nearest_heading` mapping, context in captured pairs, and
  back-compat (old context-less pairs still validate; new pairs validate *with*
  context); `validate_pairs_file` clean on a mixed file.
