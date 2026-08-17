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
4624 records, 4169 synthetic (2026-08-10 audit) → **4595 records, 4140
synthetic** (2026-08-11, after the A5 prune below). Distinct
(source, target-body) cells among the synthetic pairs: 3710 → **3686**.

**Inline markup is a fifth of the target side** (2026-08-11, over the
then-4624-pair corpus): **749,629 of 3,649,320 target-side characters —
20.5%** — is link syntax, inline code, citation keys and inline math. None of
it carries style signal, which is what makes the symmetric-masking experiment
(`decisions.md`) worth its ~$15. Per-pair incidence, same measurement: links
**77.3%** of pairs, inline code **20.3%**, citations **17.9%**, inline math
**15.8%** — against `〈MASKED_*〉` at **0.5%** (22 pairs, all on the slop side).
Raw inline markdown is the distribution; sentinels are not.

**`:::` div census** (2026-08-11, over 1751 `.qmd` files): **4139 blocks,
535,609 chars**, 3.78% of blog bytes.

| class | blocks | chars | verdict |
|---|---|---|---|
| figure | 2586 | 365,117 | markup — correctly excluded |
| `{#refs}` | 1200 | 18,221 | markup — correctly excluded |
| `callout-*` | 309 | 123,609 | **prose** — wrongly excluded |
| `{.notes}` | 14 | 18,787 | **prose** — wrongly excluded |

So ~27% of div bytes is Dan-voice prose being discarded by protecting the
container rather than the marker. Opting the prose classes in (blog-side
policy, via `div_is_prose`) would move section-bounded targets 3775 → 3824
(139 new, 90 recut, ~120k new chars) and cross-heading 2451 → 2482 (250 new,
219 recut, ~313k new chars).

**Corpus-stability floor** (2026-08-11, `tests/test_corpus_stability.py` against
the live blog): **3503 section-bounded / 396 cross-heading**, of 3686 distinct
corpus target bodies. The ~5% section-bounded gap is ordinary blog prose drift
since generation. Cross-heading is a **near-null instrument** — the corpus was
generated entirely section-bounded and `CROSS_HEADING` was promoted afterwards,
so the cells barely overlap; it earns its place only by exercising the
cross-heading path over real prose, which no fixture does.

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
| `meta.merge_max_chars`, `meta.chunk_mode` | **absent from all 4624** — the keys, not just the values; corrected 2026-08-11, next entry |
| `meta.tags` | 1 — `["synthetic","paraphrase","livingthing"]` on every pair |

**The chunk-scale facet is absent from the corpus** (2026-08-11, `iter_pairs`
over the live `_training_pairs/pairs.jsonl`; 4595 records, 4140 synthetic):

| `meta` key | present on synthetic pairs |
|---|---|
| `merge_max_chars` | **0 / 4140** |
| `chunk_mode` | **0 / 4140** |
| `context`, `context_mode` | 3186 / 4140 |
| `synth_key`, `gen` | 4140 / 4140 |

**Absent, not null** — the keys were never written, so this is not a chunker
that returned `None`. `synth._build_record` does stamp both (the "Chunk-scale
facet" comment), but that code postdates the whole corpus: `merge_max_chars`
landed 2026-07-23 and `chunk_mode` 2026-08-04, while every synthetic pair is
stamped `captured_at` **2026-07-21 (2870) or 2026-07-22 (1270)** — one two-day
batch, generated before the facet existed.

**Consequence: the v2 chunk-scale ablation cannot be run on this corpus.** The
plan in [`v2-longer-rewrites.md`](v2-longer-rewrites.md) groups by exactly the
covariate that is missing, so a mixed-scale run has nothing to contrast against
— only pairs generated from now on carry it. Nothing is damaged and no pair is
wrong; a covariate is simply not there, and it cannot be back-filled, because
the target text alone does not say what budget produced it.

**What the corpus was generated under is inferred, not recorded.** The live
blog policy re-derives **3681 of 3686** distinct corpus cells (99.86%) from the
blog at the pinned generation-time revision, which is strong evidence that the
policy — section-bounded at 1500 — has not materially moved since. That is an
inference from agreement, not provenance: label it as such wherever it is used.

The same shape of gap, smaller: **954 of 4140** synthetic pairs predate heading
context (D14, forward-only by decision), so any analysis faceting on
`meta.context` covers **77%** of the synthetic corpus, not all of it.

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

**Unanchored `$$` mangled sentences inside training targets** (defect A5,
2026-08-11). `MATH_BLOCK_RE` was not line-anchored, so a mid-sentence `$$k$$`
was excised as if it were a display block and the remnants joined:
`high-$$d$$ behaviour` → `high- behaviour` **in a training target**. **145 of
1751** blog files contain a non-standalone `$$`. Blast radius: **24 target
texts, 29 pairs (0.70% of 4169), 15 posts** — corpus 4624 → **4595** after a
one-off `scripts/prune_stale_synth_pairs.py` (deleted in 0.3; it had run
against the only corpus that exists). Stability floors moved with it,
3527 → 3503 section-bounded and 401 → 396 cross-heading. (The plan predicted
19 texts / 24 pairs; the measured text count matches the stability delta
exactly, so the estimate was low.) The other four defects in the A-series
(A1–A4) had **zero** corpus impact: all 6214 targets byte-identical.

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

## The LLM judge — measured 2026-08-17, and it is the weak signal

Prompted by "check that this judge is worth it; it costs an insane amount"
(Dan). Measured on the run-4 eval: 160 opus-4-8 calls over 40 eval chunks
(slop, Dan, and both arms' output). **Two corrections and one design fix.**

**Cost is not the problem — the estimate was wrong by an order of magnitude.**
122,103 prompt + 11,685 completion tokens over 160 calls: **~$2.71 at
claude-opus list**, median prompt **672 tokens**. A full 127-chunk four-field
pass is ~$8.60. OpenRouter reported **$0.0000**, because the Anthropic arm bills
through BYOK. The judge prompt is short; the intuition that it was expensive
came from never having measured it — `openrouter_judge` recorded no usage at
all until this run, which is why the blog's `scores.jsonl` has 270 scored rows
and **zero** judge scores, and why the ρ=0.09 judge-vs-detector figure quoted in
the blog post cannot be reproduced from anything on disk.

**The problem is that it cannot see real slop.** Pairwise, does the signal rank
Dan's side above the slop side of the same pair?

| stratum | judge | detector |
|---|---|---|
| **real** captured pairs (n=20) | **3/20 (15%), 14 ties** | 15/20 (75%) |
| synthetic pairs (n=20) | 16/20 (80%), 4 ties | 20/20 (100%) |

The judge's apparent competence is carried entirely by synthetic slop, which is
florid and easy. On the real slop Dan actually cleans up it is **worse than
chance**, and ties on 14 of 20 — the 1–5 integer scale has no resolution at the
distances that matter. Aggregates still look sane (slop 3.25 → output 4.08 →
Dan 4.20), which is how this went unnoticed: the means move in the right
direction while the per-pair judgements are noise.

**It cannot arbitrate between two adapters.** Paired 4a−4b judge difference
+0.025, 95% CI [−0.124, +0.174] at n=40; resolving a 0.10-point difference
would need **n ≈ 88**. Spearman against the detector is **−0.193** over 160
texts — correctly signed (judge high = Dan, detector high = slop) but weak,
consistent with the post's "near enough independent" claim.

**What to do:** keep the judge as a cheap *aggregate* sanity check on a
subsample (40 chunks is ample for a mean, useless for a comparison), never as a
tie-breaker between runs. The design fix worth trying before spending more on
it is a **pairwise A/B judge** — "which of these two is more Dan?" — since the
ties, not the cost, are what destroy the signal. Until then the detector is the
instrument and the judge is a smoke alarm.

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

**Run 4, the anchor-filter ablation** (2026-08-17, blog-side; two arms on the
recut 8,319-pair corpus, first mixed-scale diet). **4a** = `styler_pair_ok`
with symmetric anchor filtering (3,944 train / 446 val, $5.15, test/nll 0.7630);
**4b** = `styler_pair_ok_keep_anchor_damage`, same but keeping the 188 pairs
whose Dan side invents or deletes an anchor (4,182 / 396, $5.36, test/nll
0.7457). *Those two val losses are not comparable* — the arms hold out
different pairs. The verdict is on the frozen eval stratum, 127 chunks, paired:

| signal | slop in | 4a out | 4b out | Dan | paired 4a−4b (95% CI) |
|---|---:|---:|---:|---:|---|
| detector P(slop) | 0.5823 | **0.2125** | 0.2221 | 0.1744 | −0.0096 [−0.0335, +0.0143] |
| Vale alerts | 2.12 | 1.66 | 1.67 | 0.91 | −0.008 [−0.091, +0.076] |
| judge (n=40) | 3.25 | 4.08 | 4.05 | 4.20 | +0.025 [−0.124, +0.174] |
| outputs damaging an anchor | — | 7/127 | 6/127 | — | 6 lost + 2 invented vs 6 + 0 |

**Every signal says indistinguishable.** The filter costs 15% of the scarce
real pairs (31 of 172) and buys nothing measurable at n=127 — including on
anchor fidelity at inference, the thing it exists to move. The pre-registered
rule (keep the filter unless the unfiltered arm *wins* on the detector with the
judge agreeing) therefore keeps it, but on a tie rather than on evidence; the
honest summary is that anchor-damaged pairs are ~4% of the diet and behave like
noise at this corpus size. Both arms close ~90% of the slop→Dan detector gap
(4a 90.7%, 4b 88.3%) against runs 1–2's ~55%, but **the detector was refit the
same day**, so that is not a like-for-like comparison with the run-1/2 table
above.

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

**Inline anchors would reject a fifth of good rewrites** (2026-08-11, all 4624
pairs, run in the guard's inference direction). Share of pairs where a genuine
Dan rewrite would be **disqualified** by the anchor guard:

| anchors checked | disqualified | share |
|---|---|---|
| today (links, fences, headings) | 294 | 6.4% |
| + inline code | 552 | 11.9% |
| + inline math | 623 | 13.5% |
| + both | 878 | **19.0%** |

Tripling the rate at which every candidate is rejected and the input kept
verbatim is a bad trade for `run`, which must pick one. Hence inline anchors
default **off** in `rewrite_text` (`--strict-anchors` opts in) and are **always
on** in `rewrite_candidates`, where they only flag and a human decides.

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

Actual v1 corpus generation came in at **$15.27** in billed credits for 4169
synthetic pairs — **~$0.0037 a pair**, an order of magnitude under forecast.
That unit rate is the number to size any re-synthesis against. Training was
likewise cheap: run 1 $1.97, run 2 $1.49 against a
$6–15 estimate. The v2 scale curriculum is sized at ~8.9M target chars over
2426 / 1244 / 917 targets at 1500 / 3000 / 4500, ≈ **$39** of generation.

---

## Surface size

**`ai-style synth` flag audit** (2026-08-10): **54** click options at audit
time, against **19** for the next-largest subcommand. Classified as 14
load-bearing / 24 policy constants / 16 retirement candidates; retiring the
`gpt`/`local` preset path removed 4, leaving **50**. The full classification and
the remaining open proposals are in [`synth-flag-audit.md`](synth-flag-audit.md).

**Segmenter implementations: 7 → 3** (2026-08-11). Five copies inside stylebot
were merged into `stylebot.markdown`. The three that remain are all pinned:
`stylebot.markdown` is the source of truth, and the TypeScript port
(`tests/test_marker_segmentation.py`) and `livingthing.qmd_core`
(`tests/fixtures/segmenter_conformance.*`) are each held to a golden fixture.
Neither can be deleted — one is a different language, one must run with
stylebot uninstalled (`decisions.md` D26/D28/D30).

**Test count: 370 → 474** (2026-08-11, `uv run pytest -q`; 2 skipped, the
corpus-stability pair, which need `LIVINGTHING_ROOT`).

**The scores report's controls had never worked.** The page emits
`<table class=headline>` *before* the data table, so the script's bare
`querySelector('tbody')` bound every sort/filter/facet handler to the two-row
summary: sorts silently reordered the headline and the text filter threw on its
`.txt`-less cells. Pre-existing, found while consolidating `report.py`'s two
drifted renderers into one; fixed by giving the data table `id=rows`. Worth
recording because nothing failed loudly — a dead control looks like a control
you did not click hard enough.

## Placeholder retention across models (why shape matters, and where)

Measured 2026-08-11 in **livingthing**, during a model-roster bench for
`ai-preen`; full write-up in
[`placeholder-design-2026-08.md`](placeholder-design-2026-08.md). Different task
(copy-edit, not restyle) and different code (livingthing's `pandoc_mask`), so
these are **not** stylebot measurements — but the question is ours the moment
the deferred symmetric-masking experiment is run, because placeholder *shape* is
a free parameter of that experiment.

12 link-bearing chunks, ~122 placeholders per cell; retention = returned verbatim:

| scheme | gemini-3.1-flash-lite | qwen3.6-35b-a3b | glm-4.7-flash |
|---|---|---|---|
| `〈URL_0000〉` Unicode brackets | 100% | 45.9% | 0% |
| `https://example.com/u/<sha1>` URL-shaped | 100% | **68.9%** | 6.0% |
| `[[URL_0000]]` plain ASCII | 100% | 37.3% | 12.9% |

**The mechanism is not character exoticism.** The ASCII variant — the obvious
"fix" for weird Unicode — came *last*, worse than the Unicode it was meant to
replace. What predicts survival is whether the token looks like the kind of
thing that belongs in that slot: a URL-shaped string between `](` and `)` reads
as a URL and gets copied; `〈URL_0000〉` and `[[URL_0000]]` both read as an
identifier the editor has been handed, and editing models treat identifiers as
fair game. Distinctiveness helps your regex and hurts the model's copying.

**Why this lands hardest on us: retention is a model property, and we ship open
weights.** The spread is 100% / 37–69% / 0–13% across three models on identical
inputs. The closed frontier model was insensitive to scheme — for it, shape
design is wasted effort. The failing model was beyond saving by any sigil. The
band where scheme choice *changes the outcome* is the middle one, and that is
where open models sit: `qwen3.6-35b-a3b` moved 31 points on shape alone, and our
own styler is a Qwen fine-tune served through MLX/Fireworks. A masking scheme
validated against a frontier API would tell us nothing about the thing we ship.

**Our fine-tune has no learned affinity to fall back on.** D27 measured
`〈MASKED_*〉` at **0.5%** of corpus pairs, slop-side only — the adapter has
essentially never seen a sentinel as content, so it has learned no special
handling for that shape. The general finding therefore transfers to it rather
than being overridden by fine-tuning, which is the optimistic reading; the
pessimistic one is that it has learned nothing that would *help* either.

**Length-ratio distribution of a real rewrite** (measured here, 2026-08-11, over
the live 4,595-pair corpus; `len(target)/len(slop)` where the slop side exceeds
40 chars, n=4,590):

| p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.31 | 0.34 | 0.39 | 0.61 | 0.79 | 0.93 | 1.25 | 2.16 | 9.76 |

Mean 0.81, min 0.18. The transform leans compressive **on average only**: 13.6%
of real rewrites are longer than their input and 1.5% more than half again. This
is what sets `infer.MIN_REWRITE_LENGTH_RATIO = 0.3` (below the 0.1th percentile;
flags 2 of 4,590) and why the truncation guard is a floor with no ceiling —
length cannot distinguish an expansive rewrite from a rambling one. The source
document's 0.9 threshold, correct for a copy-edit task, would have called 70% of
genuine rewrites truncated.

**Anchor invention is a property of the arm, not of the passage** (measured
2026-08-12 over the 4,140 synthetic pairs, `infer.invented_anchors` on
`extract_slop` → `extract_target`; rate is per pair whose *target* carries an
anchor). "Invention" here means the generator dropped a link while producing
slop, so the reverse-edit swap made Dan's side look like it conjured one:

| arm | pairs | rate | in the live roster? |
|---|---:|---:|---|
| deepseek/deepseek-chat-v3-0324 | 41 | 74.2% | no (left 2026-07-22) |
| nvidia/nemotron-3-nano-30b-a3b | 731 | 60.4% | no (left 2026-07-22) |
| meta-llama/llama-3.3-70b-instruct | 75 | 51.5% | **yes** |
| qwen/qwen3-8b | 151 | 38.0% | no |
| mistralai/mistral-small-3.2-24b | 21 | 36.8% | no |
| qwen/qwen3.5-9b | 553 | 37.9% | **yes** |
| qwen/qwen3.6-35b-a3b | 427 | 24.4% | **yes** |
| google/gemma-4-26b-a4b-it | 1,082 | 5.7% | **yes** |
| anthropic/claude-sonnet-4.6 | 1,059 | 1.3% | **yes** |

A 46× spread across arms, and the two highest-volume arms are the two cleanest —
which is why the corpus-wide rate (18.6%) understates what a gate costs on the
*current* roster: llama-3.3-70b and qwen3.5-9b are still in it, so an unretried
`--drop-invented-anchors` run would drop on the order of a fifth of its cells.

Two secondary effects, both arguing for `--gate-retries` over a bare gate:

- **It is mostly per-sample, not per-passage.** Of the 454 targets seen by more
  than one arm, 148 lost an anchor under at least one — but only 29 lost one
  under *every* arm that saw them. ~80% of rejections are a redraw away from
  clearing.
- **A bare gate biases coverage against link-dense prose**, which is exactly the
  prose anchor fidelity matters most for — **but the effect saturates**, which is
  what makes the gate affordable on v2. P(rejected) by exact anchor count:

  | anchors | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | all arms | 16.1% | 20.1% | 25.1% | 27.9% | 35.5% | 29.5% | 29.4% | 39.2% | 33.3% | 30.9% |
  | live roster | 8.9% | 11.5% | 15.5% | 17.9% | 22.3% | 15.7% | 20.4% | 20.8% | 8.1% | 12.3% |

  It roughly doubles from 1 to 5 anchors and then **plateaus** rather than
  compounding toward certainty — consistent with the arm finding above: a model
  that carries links through carries all of them, and one that strips them
  strips them regardless of how many there are.

**Why this matters for the v2 scale curriculum.** Cross-heading packing raises
anchor density sharply — measured over the live blog through the real chunker:

| config | chunks | median chars | median anchors | 5+ anchors |
|---|---:|---:|---:|---:|
| v1 section-bounded @1500 | 8,811 | 443 | 2 | 16.8% |
| v2 cross-heading @1500 | 4,840 | 1,268 | 4 | 46.6% |
| v2 cross-heading @3000 | 2,783 | 2,333 | 7 | 66.3% |
| v2 cross-heading @4500 | 2,207 | 2,388 | 8 | 70.9% |

Median anchors per target quadruples 1500-section → 4500-cross-heading. Because
the rejection curve plateaus, the expected gate cost on v2 lands in the same
15–22% band (live roster) as on v1 rather than compounding — so
`--drop-invented-anchors --gate-retries N` is affordable at v2 scale. It is not
free: budget the redraws.

**Gate precision.** Spot-checking the cleanest arm's 11 rejections found no false
positives. The one systematic near-miss is pandoc citation locators: `[@X: 45]`
keys as `@X:` while a bare `[@X]` keys as `@X`, so a target adding a page number
reads as invention — 22 of 2,062 instances, 19 pairs flagged solely by it. Kept
as rejections deliberately: a page number the model cannot know is the same
fabrication failure in miniature.

**Chunk budgets overlap, so the scale arms are nested rather than disjoint**
(measured 2026-08-12 against the first real v2 batches, live blog through the
real chunker and selector). Targets per cross-heading budget, and how many are
unique to it: 1500 → 2,491 / 2,366 (95.0%); 3000 → 1,282 / 1,077 (84.0%);
4500 → 937 / 761 (81.2%). Pairwise overlaps 1500∩3000 = 100, 1500∩4500 = 71,
3000∩4500 = 151, all three = 46 — a section shorter than the smallest budget is
packed identically at every budget. Since `_synth_key` hashes target text and
not the budget, such a passage is one cell stamped by whichever batch reached it
first: of the corpus pairs whose target belongs to both the 1500 and 4500 sets,
57 carry `merge_max_chars: 1500` and 2 carry `4500` (the 2 only because the 1500
batch was interrupted at 1,985 of 2,491). Interpretation and the three ways to
analyse it honestly:
[`v2-longer-rewrites.md`](v2-longer-rewrites.md), "What `meta.merge_max_chars`
actually means".

**Anchor survival collapses at v2 chunk scale — the measurement that unblocked
masking** (2026-08-12, first gated run at the 3000-char cross-heading budget;
source `lost_draws.jsonl` + the live corpus).

| arm | draws rejected on the residue |
|---|---:|
| qwen/qwen3.6-35b-a3b | **100%** (0 written of 105) |
| qwen/qwen3.5-9b | 99% |
| google/gemma-4-26b-a4b-it | 98% |
| deepseek/deepseek-v4-flash-0731 | 78% |

Rejections are **wholesale, not marginal**: across 1,215 anchor rejections,
median **5** anchors deleted per rejected draw, mean 6.2, max **29**. Redrawing
does not help — 106 cells redrawn, 212 extra draws spent, **2 recovered (1.9%)**.
The slop prompts already instruct verbatim link preservation, so this is
instruction-following collapsing under length and density, not a missing prompt.

**This corrects the "rejection plateaus" reading recorded above.** A chunk passes
only if *every* anchor survives, so P(pass) ≈ s^k in the anchor count; at k=10 a
model needs per-anchor fidelity ≈0.97 to pass three runs in four. Cross-heading
packing moves median k from 2 (v1, 600 chars) to 8 (3000) to 11 (4500), so the
requirement is **exponential in k, not asymptotic**. The earlier plateau was
measured on v1's short chunks where high-k cases were rare and unrepresentative;
it should not be used to size a v2 run.

**The residue self-selects for difficulty**, which is why re-running never
converges. At the 3000 budget, 1,181 of 1,282 cells are covered (92.1%); cells
already written carry median 8 anchors with 51.0% at 8+, while the 101 still
missing carry median **10** with **72.3%** at 8+. Each run creams off what passes
and leaves a harder pool.

**Sentinel exposure in the corpus** (2026-08-12, 8,264 pairs): `〈MASKED_*〉`
appears on the slop side of **21** pairs and on the target side of **zero**. The
styler has never been asked to *emit* a sentinel that `STYLE_SYSTEM` has always
promised it would preserve — the gap that makes inference-side masking
out-of-distribution today (D33).
