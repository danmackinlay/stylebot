# v2 · Longer rewrites — scale curriculum + discourse coherence

Dan's v2 notes (2026-07-22): paragraph-by-paragraph rewriting loses
coherence between paragraphs, and 1500 chars feels like ~1/3 of the length
we actually want. This doc records what blocks longer rewrites, the
groundwork ALREADY LAID so the data problem stops compounding, and the
path to a v2 run. Nothing here is scheduled; the groundwork is done, the
rest is design.

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

1. **Synth at scale** — the easy 80%: synthetic pairs need no new
   groundwork (source posts are on disk). A v2 corpus batch generates at
   e.g. `MERGE_MAX_CHARS = 4500` alongside the existing 1500 diet, with
   the chunk scale recorded per-pair (a `scale` facet in `meta.gen`, like
   the generator facet) so the manifest can ablate scale with a one-line
   selector — the qwen3-32b lesson, applied in advance. Costs scale
   linearly with tokens (~3× the 1500 batch).
2. **Real pairs at scale** — re-cut the capture archive with a
   paragraph-window chunker once enough archived captures accrue. (The
   331 legacy pairs stay in the diet at their native scale; curriculum
   mixes.)
3. **Train a scale-mixed run** — 1500 + 4500 pairs, one adapter. Watch
   the run-3 lesson: longer chunks give the model more room to compress
   and delete; the anchor-integrity eval (deletion_report) and the
   promotion field are the gate.
4. **Inference follows the diet** — `default_chunk_chars` rises to match
   whatever scale the promoted manifest trained on; record the chunk
   scale in the manifest (`hyperparameters` or `data`) so serving can
   read it instead of hard-coding.
5. **Coherence proper (further out)** — condition on the previous
   *rewritten* paragraph: new pair format (context = preceding target
   paragraph, reconstructable for every archived capture), sliding-window
   inference in `rewrite_text`, and the editor's `para_before` records as
   preference data. Strictly after the scale curriculum — longer chunks
   alone (3–4 paragraphs rewritten as one unit) may buy most of the
   perceived coherence, since intra-chunk coherence is free.

## Open questions for Dan

- Is 4500 the right v2 scale, or should the mix include whole-section
  pairs (`--whole` captures exist but are rare)?
- Should the editor rewrite command grow a "rewrite section" mode once
  the adapter can handle it (multi-chunk selection, per-chunk sequential
  decisions vs one long chunk)?
