# v2 · Longer rewrites — scale curriculum + discourse coherence

Dan's v2 notes (2026-07-22): paragraph-by-paragraph rewriting loses
coherence between paragraphs, and 1500 chars feels like ~1/3 of the length
we actually want. This doc records what blocks longer rewrites, the
groundwork ALREADY LAID so the data problem stops compounding, and the
path to a v2 run.

> **Status 2026-08-04 — mechanism shipped, generation not yet run.**
> The diagnosis below was half wrong and the section "Why the model is
> stuck at 1500" is corrected in place; read "The section wall" first.
> Decisions taken: mixed scale curriculum (1500/3000/4500), headings
> inline and **rewritable**, v1 kept alongside v2. What exists now:
> `cross_heading` chunking (`targets.py`), gap-coalescing + `ai-style log
> recut` (`capture.py`), amended `STYLE_SYSTEM`, `data.chunk_scale` in the
> manifest, `inline_headings` through inference and the editor seam. What
> does not: any v2 pair has yet been generated, and no v2 adapter trained.

## The section wall (the actual blocker)

**Raising `MERGE_MAX_CHARS` does almost nothing**, because `_split_sections`
means a chunk can never cross a heading, and Dan's sections are short.
Measured over `notebook/` + `post/` (2026-08-04): section prose is median
**553** chars, p90 1729; only **13.6%** of sections exceed 1500 and **0.8%**
exceed 4500. Run through the real chunker with blog policy, tripling the
budget moves the median target the wrong way:

| budget | section-bounded | cross-heading |
|---|---|---|
| 1500 | 2088 chunks, median 626 | 1442 chunks, median 1278 |
| 3000 | 1782 chunks, median 588 | 805 chunks, median 2603 |
| 4500 | 1727 chunks, median 578 | 615 chunks, median 3224 |

So the fix is structural, not numeric: pack across section boundaries with
heading lines as chunk **content**. Cross-heading packing at the *existing*
1500 budget already doubles the effective scale. This is why the synthetic
median target was 662 chars against a 1500-char budget — it was hitting the
section wall, not the budget.

Consequence, and the reason `STYLE_SYSTEM` changed: an inline heading is
text the styler rewrites, so the frozen prompt's "preserve … headings …
verbatim" clause contradicted the data. It now says headings may be
restyled but must keep their level and meaning. Every v1 record was
re-stamped (`scripts/restamp_style_system.py`, system message only — user
and assistant sides untouched, verified). A restyled heading changes its
slug and can break inbound `other.qmd#slug` links, which the per-chunk
anchor guard cannot see; `workflow_style.check_heading_anchors` warns.

## Why the model is stuck at 1500

Not a model limitation — a diet limitation, twice over:

- **The corpus is short.** 4,169 of 4,500 pairs are synthetic, chunked at
  blog policy `MERGE_MAX_CHARS = 1500`. Real captured pairs are unbounded
  (contiguous changed regions; observed median 693 chars, p90 1866, max
  4143) but are only 331 records. At out-of-distribution lengths the
  adapter falls back to copying the input (observed 2026-07-22: a 5.4KB
  single chunk came back byte-identical), so the inference budget is
  *pinned* to the training scale — raising `--max-chunk-chars` without
  retraining just buys identity fallbacks.
- **Every pair is an island.** Training pairs carry a heading as context
  and nothing else; the model has never seen "the previous paragraph said
  X, now rewrite this one" — cross-paragraph coherence was never in the
  objective.

## The logging problem (was real; groundwork now laid)

The capture pipeline destroyed document-scale alignment at birth, in two
places:

1. `diff_chunks` emits only the CHANGED regions — unchanged paragraphs
   between them are dropped, and no paragraph indices/offsets are stored.
   `chunk_index` orders the changed regions; it does not place them.
2. `ai-style log save` deleted the session snapshot (the full before-text)
   on success. The 109 pre-2026-07-22 captures are chunk-scale forever.

**Groundwork shipped 2026-07-22 — capture archive:** `capture_pairs` now
writes the full before/after texts to
`_training_pairs/captures/<capture_id>.json` beside the corpus. The chunk
records become a lossy *view*; the archive is the re-cuttable raw. Any
future chunking policy (3–4 merged paragraphs, whole sections, pairs with
preceding-paragraph context) can be applied retroactively to everything
captured from now on, without recapturing. Cost: ~one blog post of text
per capture. Commit them like `pairs.jsonl`.

**Groundwork shipped same day — preference records:** editor decisions
(voice-marker ≥0.0.4) now log `offsets` plus `para_before`/`para_after`
(the flanking prose paragraphs, read before the edit mutates the buffer).
The working copy is uncommitted, so this discourse context is
unrecoverable later — logging it now means the preference corpus supports
coherence-aware objectives when we get there.

## Path to v2 (design, not schedule)

1. **Synth at scale** — the easy 80%: source posts are on disk, and
   `--merge-max-chars` was already a per-batch flag. The missing piece —
   the chunk scale was not RECORDED per pair, so a mixed-scale corpus
   would have been un-ablatable — shipped 2026-07-22: `Target` carries
   `merge_max_chars` and synth stamps it into `meta.merge_max_chars`
   (None = unmerged; pre-facet records lack the key and were all 1500).
   A v2 batch is now `dan-style synth --merge-max-chars 4500` (cross-heading
   is blog-policy default; `meta.chunk_mode` joins `meta.merge_max_chars` as
   the second facet, and mode matters more than budget so the ablation needs
   both). NOT YET RUN. Sized with the real selector: 2426 / 1244 / 917
   targets at 1500 / 3000 / 4500, ~8.9M target chars over the whole
   curriculum, ~$39 of generation against v1's $15.27.
   **Do the scratch audition first** (RUNBOOK §4): `max_length_ratio` (3.0)
   and `max_transform_sim` (0.95) were calibrated on 1500-char chunks.
   `DEFAULT_SLOP_MAX_TOKENS` (8192) already covers a 4500-char output, and
   `_plan_sessions` budgets sessions from estimated prompt tokens, so
   `SESSION_TURNS` self-limits at the larger scale — verify, do not assume.
2. **Real pairs at scale** — SHIPPED as gap-coalescing, not windowing.
   `ai-style log recut --gap-paras 2 --gap-chars 800` replays the archive,
   bridging *short unchanged runs* so one multi-paragraph rewrite stays one
   pair rather than fragmenting. Measured over the 41 archives: 87 pairs,
   median 1191 chars, p90 4230 (against 130 / 845 / 1938 uncoalesced),
   idempotent via `capture_id`, oversized merges falling back to the legacy
   cut at `MAX_CHUNK_CHARS`.

   **A fixed-width window would have been wrong**, and the first draft of
   this plan proposed one. Untouched prose in a capture is text Dan did not
   get to, not text he endorsed — a capture's before-side is slop-bearing,
   so padding pairs out with their neighbours teaches the styler to leave
   slop alone. It measures badly too: changed-fraction 0.41 at a 4500-char
   window, against 0.96 (p10 0.86, none below 0.5) under coalescing. The
   do-no-harm lesson keeps coming from where it already did — the ~19% of
   synthetic pairs rewritten to 85% the same.

   Note the asymmetry this creates, deliberately: on the **synthetic** side
   long cross-heading windows are fine, because the target is Dan's
   published prose and all of it is endorsed. On the **capture** side pairs
   stay changed-region-focused.

   Also dead: the theory that `diff_chunks` was discarding Dan's
   paragraph-count-changing edits by skipping pure `insert`/`delete`
   opcodes. Measured across all 41 archives: **1** dropped opcode.
   `SequenceMatcher` folds reflowed paragraphs into `replace`.

   (The 331 legacy pairs stay in the diet at their native scale; curriculum
   mixes.)
3. **Train a scale-mixed run** — 1500 + 3000 + 4500 pairs, one adapter.
   NOT YET RUN. Watch the run-3 lesson: longer chunks give the model more
   room to compress and delete; the anchor-integrity eval
   (deletion_report) and the promotion field are the gate.
   `DEFAULT_MAX_LENGTH` (8192 tokens) needs no change — a 4500-char pair is
   ~2400 tokens both sides.
4. **Inference follows the diet** — SHIPPED. `build_manifest` records
   `data.chunk_scale` (per-budget and per-mode pair counts, plus
   `max_target_chars`); `style_file.chunk_diet` reads it and returns
   `(max_chunk_chars, inline_headings)`. Two one-sided rules, because
   over-chunking hits the identity fallback while under-chunking only
   costs coherence: a budget needs >=10% corpus support before serving
   uses it, and inline headings need a cross-heading *majority*. Manifests
   predating the key fall back to `MERGE_MAX_CHARS`, section-bounded —
   which is what runs 1-3 actually trained on.
5. **Coherence proper (further out)** — condition on the previous
   *rewritten* paragraph: new pair format (context = preceding target
   paragraph, reconstructable for every archived capture), sliding-window
   inference in `rewrite_text`, and the editor's `para_before` records as
   preference data. Strictly after the scale curriculum — longer chunks
   alone (3–4 paragraphs rewritten as one unit) may buy most of the
   perceived coherence, since intra-chunk coherence is free.

## Resolved (2026-08-04)

- **Is 4500 the right v2 scale?** Mixed curriculum instead: 1500 / 3000 /
  4500 in one adapter, kept separable by `merge_max_chars` + `chunk_mode`
  so "did scale help" stays answerable. v1 stays in the corpus alongside.
- **Should the editor grow a "rewrite section" mode?** Yes, and
  `rewrite_candidates` now allows it: under `inline_headings` a selection
  may span headings. Fences still raise — they are absent from the corpus
  in both diets, so the styler has never learned to reproduce one. The
  size limit still applies; a selection `_segment` would split is not one
  rewrite. The VS Code extension has not been updated to offer it yet.

## Still open

- Step 5 (previous-rewritten-paragraph conditioning) is untouched, and may
  turn out unnecessary — intra-chunk coherence is free, so cross-heading
  chunks alone may buy most of the perceived coherence. Decide after the
  v2 eval, not before.
- The detector is still paragraph-scale, and the editor thresholds
  `[0.51, 0.55, 0.59, 0.65]` are calibrated on paragraph scores. Retrain it
  on the mixed corpus so candidate reranking sees long passages in
  distribution, then re-check paragraph scores BEFORE touching thresholds.
