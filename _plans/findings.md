# Findings — measured results

Empirical results from the stylebot pipeline, collected here as the per-phase
design plans were retired. **This file exists because these numbers are not
re-derivable from git.** Reproducing any of them costs real money and
wall-clock; the code that produced them records the method, never the result.

Each entry: date, what was measured, sample size, the number.

---

## Corpus and target selection

**Blog frontmatter partition** (2026-06-27, live blog). `automation:` levels —
`0` (pure human) 1591 posts, `1` 58, `2` 61. Posts under `post/` (~73) and
`notebook/` (~1600).

**Target selection funnel** (2026-06-28, live blog, through the real chunker).
Naive `automation:0`-only ≈ **27.4k** junky chunks → prose-only + generic
hygiene ≈ **19.5k** → plus the blog's `quality>6`/not-draft selector ≈ **10k**
clean targets → merged into passages ≈ **3.6k**, median ~620 chars. The hygiene
passes discard roughly 87% of the naive extraction.

**Heading coverage** (2026-06-28, live blog). **85%** of merged passages sit
under a section heading; 15% are preamble. This is what justified attaching the
heading as fixed context, byte-identical on both sides of every pair.

**Section length** (2026-08-04, over `notebook/` + `post/`). Section prose is
median **553** chars, p90 **1729**; only **13.6%** of sections exceed 1500
chars and **0.8%** exceed 4500. Dan writes short sections, which is why raising
the merge budget alone does almost nothing.

**Chunk yield by budget and mode** (2026-08-04, real chunker, blog policy):

| budget | section-bounded | cross-heading |
|---|---|---|
| 1500 | 2088 chunks, median 626 | 1442 chunks, median 1278 |
| 3000 | 1782 chunks, median 588 | 805 chunks, median 2603 |
| 4500 | 1727 chunks, median 578 | 615 chunks, median 3224 |

Tripling the budget moves the section-bounded median *down*. Cross-heading
packing at the existing 1500 budget already doubles effective scale. So
`chunk_mode` dominates `merge_max_chars` — section-bounded packing hits the
section wall long before the byte budget, and two runs differing only in
`merge_max_chars` produce near-identical diets.

**Corpus composition over time.** 254 records (2026-06-27, all real) → 331 real
/ 3,231 synthetic (2026-07-21) → 4,169 of 4,500 pairs synthetic (2026-07-22) →
**4624 records, 4169 synthetic** (2026-08-10 audit).

**Real captured pairs are unbounded but few** (2026-07-22): observed median 693
chars, p90 1866, max 4143 — across only 331 records.

**Real rewrites compress** (no sample size recorded): Dan's own edits compress
~**1.5×** commonly, **2×** sometimes, **3×** rarely. So the real corpus supplies
no *inflated* slop, and inflation has to be asked for deliberately — the reason
a `verbose` slop strategy exists. The `--max-length-ratio 3.0` hygiene gate sits
at the top of that observed range.

**What has actually varied across the synthetic corpus** (2026-08-10, over the
4169 synthetic pairs):

| covariate | distinct values |
|---|---|
| `gen.model` | 9 (incl. 3 since dropped from the roster) |
| `slop_strategy` | 6 — polish 943, measured 896, casual 783, engaging 567, linkedin 520, verbose 460 |
| `gen.reasoning_effort` | 2 — `off` 2632, `high` 1537 |
| `gen.max_tokens` | 2 — 8192 (4130), 16384 (39) |
| `gen.replicate` | `verbose1` on 460, unset on 3709 |
| `gen.session_id` | 190 sessions, 99 distinct turn indices |
| `meta.context_mode` | `immediate` 3212, unset 957 (context dropout working) |
| `gen.temperature`, `gen.top_p` | **1 each — always `None`** |
| `gen.provider_sort` | **1 — always `throughput`** |
| `gen.context_window` | 5, all from the OpenRouter registry, never from a flag |
| `meta.merge_max_chars`, `meta.chunk_mode` | **`None` on all 4624** — v2 scale curriculum not yet run |
| `meta.tags` | 1 — `["synthetic","paraphrase","livingthing"]` on every pair |

---

## Generation experiments

**Reasoning-effort sweep** (2026-07-20). **1241 pairs, six models**, `off` vs
`medium`. The detector gap was **flat — 0.088 vs 0.092** — for roughly **3× the
wall clock**. The cheap end won: the default flipped to `off`, and effort is now
swept *up* on suspicion rather than defaulted high. Caveat: the sweep measured
`off` vs `medium` only — **`high`, the then-default, was never measured**.

**A provider ignored `effort=off`** (2026-07-20): qwen3-32b emitted ~700
reasoning tokens anyway. The *requested* effort is recorded regardless of what
the provider honours; `finish_reason`/`completion_tokens` are what expose this.

**Session depth is quadratic in input tokens** (2026-07-06). Each stateless call
re-sends the history, so a 32k-token session bills **~25×** the input tokens of
the same pairs generated statelessly. Whether *dollars* follow depends on the
provider's cache-read discount: measured live, qwen-via-Groq and sub-1024-token
OpenAI prompts returned `cached_tokens=0`, i.e. genuinely quadratic dollars.

**Prompt caching is unexploitable on stateless corpus runs** (2026-07-07): the
only shared prefix is the slop system prompt, which sits under every provider's
~1024-token cache minimum.

**Window position does not affect slop** (2026-07-21 sweep): drift correlation
**r = −0.05**. This retired session-window-fill as a treatment; it survives only
as a recorded covariate.

**Silent-drop post-mortem** (2026-07-21): a `--session-turns 128` run silently
discarded **46%** of planned work when the 32k token budget bound before the
turn count did. Leftover turns now reflow into fresh sessions.

**Re-key epoch double-weighting** (2026-07): a config re-key regenerated
already-covered targets, doubling their training weight — **~1,970 doubled
targets**, an imbalance correlated with walk order. `--skip-covered` is the fix.

**Rendezvous vs modulo arm assignment** (2026-08-10; 20k targets, 6 arms;
fraction of the corpus reassigned by a roster change, "min" = theoretical floor):

| roster change | modulo | rendezvous | min |
|---|---|---|---|
| drop 1 arm | 83.2% | **16.7%** | 16.7% |
| add 1 arm | 85.8% | **14.4%** | 14.3% |
| swap 1 arm | 16.9% | 28.7% | ~33% |
| reorder only | **100.0%** | **0.0%** | 0% |

Balance is comparable (5.8% modulo vs 4.3% rendezvous spread). Modulo made the
winner a *position*, so a pure reorder of the model list moved the whole corpus.

**The switch was free when taken** (2026-08-10): the roster had already drifted
(3 models held pairs but were no longer rostered), so a dry run against the live
corpus reported "0 already present" — resume overlap was zero either way.

---

## Data hygiene / QA

**Identity pairs** (2026-07-21 corpus QA): **386** pairs where the generator
returned its input essentially unchanged — all from the qwen3-8b @ `effort=off`
cell, median `transform_sim` **0.98**, length-ratio p10 **1.00**. Plus
stragglers ≥0.95 elsewhere. These teach the styler to copy; now gated at birth
by `--max-transform-sim 0.95`.

**Inflation tail** (2026-07-21): qwen3-32b produced **334** florid pairs at
**4.4–6.8×** length inflation. Gated by `--max-length-ratio 3.0`.

**Near-copies are common and are good styler data** (2026-07-21): **~19%** of
synthetic pairs have `transform_sim > 0.85`. They are label noise for the
*detector* (dropped there) but teach the styler restraint — leave alone prose
that is already fine.

**Capture diffing loses almost nothing** (2026-08-04, across all 41 archives):
the theory that `diff_chunks` was discarding paragraph-count-changing edits by
skipping pure insert/delete opcodes died on measurement — **1** dropped opcode
in total. `SequenceMatcher` folds reflowed paragraphs into `replace`.

**Gap-coalescing vs fixed-width windowing** (2026-08-04, over the 41 capture
archives). Coalescing short unchanged runs: **87 pairs, median 1191 chars, p90
4230** (against 130 pairs / 845 / 1938 uncoalesced). A fixed-width window would
have had changed-fraction **0.41** at 4500 chars, against **0.96** (p10 0.86,
none below 0.5) under coalescing. The bridging tolerance was calibrated at
**2 paragraphs / 800 chars** — the widest bridge at which changed-fraction still
holds near 1.0.

---

## The voice classifier (the detector signal)

**Backbone bake-off** (2026-06-30), ranked by content-matched pairwise accuracy
and AUC, split by POST:

| backbone | pairwise_acc | AUC |
|---|---|---|
| **StyleDistance** (`StyleDistance/styledistance`, 768-d) — winner | **0.78** | **0.72** |
| LUAR-MUD | 0.76 | 0.71 |
| mxbai (semantic baseline) | 0.75 | **0.62** |
| Wegmann CISR | 0.73 | 0.67 |

The AUC column is the tell: mxbai's topic-dominance collapses on the absolute
"is this slop" question while a content-independent *style* embedding holds.
This is what made both Pangram and a contrastive fine-tune unnecessary for v1.

**Score distribution is narrow** (2026-07-06, 264 pairs, both sides). The whole
distribution lives in ≈[0.32, 0.68]:

| percentile | 5 | 25 | 50 | 75 | 95 |
|---|---|---|---|---|---|
| Dan side | .32 | .40 | .46 | .51 | .62 |
| slop side | .40 | .47 | .55 | .59 | .68 |

Hence warped, quantile-calibrated editor buckets `[0.51, 0.55, 0.59, 0.65]`
(Dan-p75 / slop-p50 / slop-p75 / ≈slop-p95) rather than equal-width bins, which
would have wasted most buckets.

**Editor latency is a non-issue** (2026-07-06, StyleDistance on Dan's Mac, MPS):

| | latency |
|---|---|
| cold model load (once, at sidecar startup) | 5.0 s |
| warm: 1 paragraph | 7 ms |
| warm: whole 30-paragraph notebook | **34 ms** |
| head dot-product → `P(slop)` | 28 µs |

Re-scoring a whole document in ~34 ms is fast enough for debounced keystrokes,
not just on-save.

---

## Training runs

**Base model rate**: `Qwen/Qwen3.5-9B` at **$1.463/1M train tokens**
(2026-07-21). Tinker sampling for inference: ~**$0.60/1M tokens**.

**Pre-run dry run** (2026-07-21, corpus sha `a38610bc71290d87`, styler selector
active): 1,666 train / 148 val pairs (116 real / 1,698 synthetic; 235
styler-role identity pairs dropped), ~1.12M train tokens ≈ **$1.64/epoch**.

**Run 1** (`20260721-run1`): corpus sha `0838ae470afef6c3`, val NLL **1.018**,
cost **$1.97**.

**Run 1 smoke** (2026-07-22, 40 val pairs, detector `P(slop)` means):
slop 0.507 → **adapter 0.422** → target 0.352. The prompt-only baseline scored
**0.552** — *worse than the untouched input*. The adapter beat that baseline on
**35/40** pairs.

**Run 2** (2026-07-22, post-surgery corpus — qwen3-32b cell and the >3×
inflation tail excised): corpus sha `ea9f5981f8074cb3`, val NLL **0.952**
(run 1: 1.018), cost **$1.49**. Same-40-val-pairs comparison: voice metrics
marginally softer (detector 0.425 vs 0.415, judge 3.48 vs 3.70) but the
over-compression tail retreats (p10 output/slop **0.76 vs 0.60**) and content
fidelity improves (anchor losses **1 vs 2**). Fidelity was chosen over voice
score; run 2 is the serving default.

---

## Inference and serving

**Full eval against baseline** (2026-07-22, 40 val pairs, three signals
agreeing):

| signal | slop | output | target | prompt-only baseline |
|---|---|---|---|---|
| detector P(slop) | 0.507 | **0.414** | 0.352 | 0.552 |
| judge (1–5) | 2.40 | **3.45** | 4.33 | — |
| Vale alerts | 3.3 | **1.28** | 1.2 | — |

**Out-of-distribution length makes the styler copy its input** (2026-07-22). A
**5.4KB single-chunk** rewrite came back **byte-identical** — the adapter falls
back to identity at lengths it never trained on. The same post chunked at
training scale moved whole-body detector P(slop) **0.589 → 0.281**. The
inference chunk budget is therefore pinned to the training chunk scale: raising
`--max-chunk-chars` without retraining buys identity fallbacks, not longer
rewrites.

**MLX local-serving parity** (2026-07-22, run 2, 20 val pairs @ temp 0.3):
detector output means **tinker 0.4627 vs local 4-bit 0.4529**, Δ **−0.0098**,
inside the 0.03 gate. Quantization drift stayed within the 0.05 gate, so no
8-bit build was needed. Zero template artifacts; real-post round-trip
structure-clean.

---

## Cost: forecast vs actual

The original plan budgeted **$180–290** for synthetic pair generation (model
matrix, Sonnet-weighted; a cheaper Haiku variant at $100–170), inside a total
upfront envelope of ~$200–330.

Actual v1 corpus generation came in at **$15.27** — an order of magnitude under
forecast. Training was likewise cheap: run 1 $1.97, run 2 $1.49 against a
$6–15 estimate. The v2 scale curriculum is sized at ~8.9M target chars over
2426 / 1244 / 917 targets at 1500 / 3000 / 4500, ≈ **$39** of generation.

---

## Surface size

**`ai-style synth` flag audit** (2026-08-10): **54** click options at audit
time, against **19** for the next-largest subcommand. Classified as 14
load-bearing / 24 policy constants / 16 retirement candidates; retiring the
`gpt`/`local` preset path removed 4, leaving **50**. The full classification and
the remaining open proposals are in [`synth-flag-audit.md`](synth-flag-audit.md).
