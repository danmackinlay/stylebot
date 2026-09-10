# Findings — measured results

Empirical results from the stylebot pipeline. **This file exists because these
numbers are not re-derivable from git**: reproducing any costs money and
wall-clock, and the code records the method, never the result. **Shape (P4):**
one table per measurement, ≤ 12 lines of reading, each carrying its **date**, the
**instrument**, and the **decision** it fed; a superseded one keeps its table and
gets a one-line strike. These are dated records of what was true then and must
never be edited to match today (D52).

## The view reproduces the corpus — 2026-09-04 (W4 step 1)

`STYLEBOT_SWEEP_BLOG_ROOT=~/Source/livingthing uv run pytest tests/test_draws.py
-s -k reproduces_the_blog`: `draws.draws_from_pairs` over the blog's
`_training_pairs/pairs/` (three shards), `draws.build_view` under the identity
`ViewPolicy()` into a temp dir, both sides hashed by `corpus.content_sha256`.
No spend.

| | `pairs/` (source) | the view |
|---|---|---|
| records | 8,157 | 8,157 |
| distinct `record_id` | 8,157 | 8,157 (same set) |
| `content_sha256` | `d49ba7ad60774ea6` | **`d49ba7ad60774ea6`** |
| shards | captures, synth-2026-07, synth-2026-08 | the same three, same order |
| clauses fired / `policy_sha` | — | none / `7c57099c31be0a0c` |

Byte-identical: every record round-trips through `json.loads` and
`corpus.line_of` unchanged, and routing by `corpus.shard_of` in read order
lands each in the shard it came from. So the view layer costs nothing today
and the `dan-style status` sha is the one number that must hold through step
4. The identity policy's sha names today's corpus as a view; the blog's
`view_policy()` (S1b) must hash to a policy under which no clause fires, or
the done-criterion fails by design. Decided: D84.

## The backfill's recovery — 2026-09-05 (W4 step 3)

`uv run ai-style data backfill-draws _training_pairs --side-file
lost_draws.jsonl --side-file gate_drops.jsonl --resolve
livingthing.corpus_provenance:resolve_record --superseded <SUPERSEDED_CAPTURES>
--apply` in the blog worktree at `5ed9df8572`, reading `_training_pairs/pairs/`
(`d49ba7ad60774ea6`) and the two reject logs; 9.0 s wall (dry run 8.4 s). No spend.

| | count |
|---|---|
| pair records → accepted draws | 8,157 |
| resolved at a pinned revision (`corpus_provenance.PINS`) | 7,741: 3,132 verbatim, 4,609 gapped |
| resolved by the history walk (4 sources, 210 commits read) | 7: 3 verbatim, 4 gapped |
| unresolved synthetic (no sha, no span) | 2: `83f5f1989128f20b`, `e58c7711ba3a8806`, both `notebook/fish.qmd`, legacy, 2026-07-21 |
| real records (no sha, `blog_dirty`) | 407: 174 with a span into `before.md` (171 verbatim, 3 gapped), 233 with no archive |
| tombstones by outcome | 6,429: `gate:skipped_invented_anchors` 5,769, `gate:skipped_degenerate` 565, `gate:skipped_inflated` 93, `error:RuntimeError` 2 |
| superseded (`SUPERSEDED_CAPTURES`, 8 sessions) | 11 draws |
| duplicate draw ids | 7, all real: 6 interim/successor pairs, 1 unmapped (`d89d26a4`/`f608a742`) |
| `draws/` | 3 shards, 14,586 lines, 73 MB (`pairs/`: 37 MB) |
| `dan-style build-view` under `view_policy()` | `content_sha256=d49ba7ad60774ea6`; excluded `outcome: 6429`, nothing else |

"Gapped" means the cutter had dropped a protected block (display math, a fence)
from inside the cell, so the target is not a contiguous substring: the span is
the bounding region its lines occur in, in order. The design's "five
unreachable cells" were counted by re-cutting; by text presence three of them
sit in a committed blob and two `fish.qmd` cells sit in none. `draws/` is 2.0×
`pairs/`, not the estimated 1.6×: an accepted draw carries its record whole
plus `target_text` and `raw_output` again. The seven duplicate ids are a real
draw's cell lacking its capture id, a key-shape question for step 4. The
labelled replicate (`gen.replicate = "verbose1"`, 374 records) crashed
`_lift_pair`'s `int()`; fixed on this branch. Decided: D84 (amended, step 3).

## The cull's baseline — 2026-09-04

Stylebot `src/` before and after X2, the first ship-shape cull, by
`tests/test_budget.py`'s `measure()` over every module; the last row is
`rg -c "legacy|compat|deprecated|shim"` over both `src` trees. No spend.

| | before X2 | after X2 |
|---|---|---|
| `src/` lines (37 modules) | 18,056 | 17,745 |
| code lines | 11,477 | 11,250 |
| docstring + comment lines | 4,687 | 4,630 |
| prose / code | 0.4084 | **0.4116** |
| CLI layer (`bin/*.py`, `cli_options.py`, `synth_cli.py`) | 3,995 | 3,995 |
| `report.py` / `eval.py` / `infer.py` | 885 / 754 / 863 | 692 / 645 / 854 |
| `rg -c`, stylebot / blog | 25 / 24 | 25 / 24 |

The cull removed 311 lines, 227 of them code, and **raised the ratio**: what
went was code-heavy (two HTML templates, a poll loop), so the prose explaining
the rest is now a larger share of it. Nothing in the blog called any of the
three, so its count did not move, nor did the CLI layer. The `rg -c` hits are
unchanged because none of the deleted lines said "legacy", "compat",
"deprecated" or "shim". `report.py` left the D58 allowlist at 692 lines and
`infer.py`'s row fell to 854; `PROSE_CEILING` (4,688) still holds at 4,630
while `RATIO_PIN` (0.4072) no longer does, which is what the count-first pin
was picked for — a code diet must not fail the budget. Nine tests went with
their subject. Decided: D7 (the cross-check goes), D58 (the budgets it is
measured against).

## Threading audit — 2026-09-03/04: the eval pool bought nothing; batching bought 1.7x

Every threading construct in stylebot and the blog, against D56's rule. No spend:
the 34 already-paid-for documents in `runs/docscale-20260903/documents.jsonl` plus
the live corpus. Apple silicon, torch 2.13, 6 torch threads, interleaved repeats.

| measurement | `max_workers=8` | `max_workers=1` |
|---|---|---|
| 1,690 chunks, detector inside the pool | 24.25 s, 24.71 s | 23.96 s, 23.69 s |
| same, batched pre-pass | 12.58 s, 14.85 s | 13.84 s, 15.53 s |
| 1,024 passage-scale embeds, per-text | 15.65 s, 12.90 s | 12.85 s, 12.58 s |
| same, batched | 6.93 s, 8.00 s | 7.35 s, 8.33 s |
| 102 judge calls at a faked 1 s, `ThreadPoolExecutor` | 16.04 s | 103.33 s |
| 102 judge calls, `asyncio.Semaphore` | **13.98 s** | — |

Eight threads were **not faster than one** for the detector, embedding being
CPU-bound. Batching is the win — bare encode over the 1,690 chunks is 22.22 s one
at a time against 14.24 s at `batch_size=32` (**1.56x**; 64 does not help),
~**1.7x** end to end — while the judge is what a gate is *for*, at **6.4x**, and
the semaphore beats the pool by 1.15x because the pool gated *records* where it
gates *calls*. **Batching is not bit-identical**: a re-score matches to 1.86e-07,
four orders below the 0.022 delta measured, but will not diff clean.
Decided: D56 (the rule), D59; D55 and D57 superseded.

## `ai-style --help` import cost — 2026-09-03

`ai-style --help`, median of 12 subprocess runs on Dan's laptop. Interpreter
floor 9.0 ms; `import click` alone 18.7 ms.

| | median |
|---|---|
| before (this branch's starting point) | 49.8 ms |
| after (asyncio deferred in `synth`) | **43.7 ms** |
| hypothetical: `synth`/`synth_cli` also gone | 38.4 ms |

A/B'd on the finished tree (15 runs each), asyncio alone is **4.9 ms**. W11.i's
premise — ~30 ms of a 70 ms `--help` — reproduces in neither term. Decided: D49.

## The sidecar `segment` op — 2026-09-03

One NDJSON round trip per document change, through the real `Sidecar` client
against a stub-detector `serve_loop`, 50 timed trips each, warm.

| document | chars | segments | median | p95 |
|---|---|---|---|---|
| `notebook/music_loopers.qmd` (corpus median) | 5,428 | 9 | **0.30 ms** | 0.38 ms |
| `digest/2026-04-24.qmd` (p90) | 17,662 | 22 | 0.65 ms | 0.71 ms |
| `notebook/automatic_coding.qmd` | 30,496 | 67 | 1.07 ms | 1.10 ms |
| `notebook/local_llm_mac.qmd` (largest in the repo) | 116,774 | 212 | **4.20 ms** | 4.36 ms |

Nearly all of it is the segmentation itself (0.28 / 0.50 / 0.90 / 3.71 ms
in-process), so the hop costs tens of microseconds — unmeasurable against the
34 ms `score` already spends and the 500 ms debounce. D44.

## Document-scale eval on the serving diet — 2026-09-04 (F4): the levels move, the deltas do not

`div_is_prose` now crosses the document-scale seam, so both halves — the rewrite
and the scoring — use the diet `dan-style run`/`serve` use. Fresh run-id, so all
34 documents were re-sampled (~$0.30).
`direnv exec . uv run dan-style eval --documents --run-id docscale-20260904`,
against `runs/docscale-20260904/scores.jsonl` (pins: `runs/docscale-20260904.json`).

| | 2026-09-03 (`div_is_prose` absent) | **2026-09-04 (serving diet)** |
|---|---|---|
| chunks scored, slop / target / output | 563 / 564 / 563 | **608 / 609 / 607** |
| chunks rewritten (kept verbatim) | 563 (7) | 608 (10) |
| mean P(slop), slop / target / output | 0.3844 / 0.3622 / 0.3658 | **0.3620 / 0.3381 / 0.3457** |
| target − slop | −0.0222 [−0.0452, −0.0139] | **−0.0239 [−0.0429, −0.0150]** |
| output − slop | −0.0186 [−0.0234, −0.0118] | **−0.0162 [−0.0220, −0.0057]** |
| output − target | +0.0036 [−0.0055, +0.0288] | +0.0077 [−0.0053, +0.0333] |
| documents where Dan / styler < draft | 25 / 34, 25 / 34 | 25 / 34, 24 / 34 |
| mean within-document sd (slop) | 0.2642 | 0.2781 |

Same 34 documents over 7 posts, same detector (`c0e2ac85be`), 95% CIs over the 7
post clusters. **The correction is real and its effect on the verdict is a null**:
all three levels drop 0.020–0.024 because freeing callout prose adds ~45 chunks
of Dan's own writing to every field, the AI draft's included, while the paired
deltas move inside their own interval widths. D54; supersedes the entry below.

## Document-scale eval, chunk-wise — 2026-09-03: it separates

~~Superseded 2026-09-04 by `docscale-20260904` above, which re-ran it on the
serving `div_is_prose`.~~ Same 34 documents, detector and command; **only the
scoring changed**, to the mean over serving-scale chunks. No new sampling.

| | whole-document embedding | **chunk-wise mean** |
|---|---|---|
| mean P(slop), slop (the AI draft) | 0.4880 | **0.3844** |
| mean P(slop), target (Dan's edit) | 0.4951 | **0.3622** |
| mean P(slop), output (the styler) | 0.4914 | **0.3658** |
| min / max, all three fields | 0.1107 / 0.7921 (identical) | 0.16–0.26 / 0.50–0.56 |
| documents where Dan < draft | 2 / 34 | **25 / 34 = 0.735** |
| documents where styler < draft | 0 / 34 | **25 / 34 = 0.735** |
| target − slop (clustered by post) | +0.0071 (−0.0017, +0.0291) | **−0.0222 (−0.0452, −0.0139)** |
| output − slop | +0.0034 (+0.0000, +0.0148) | **−0.0186 (−0.0234, −0.0118)** |
| output − target | −0.0037 (−0.0172, +0.0022) | +0.0036 (−0.0055, +0.0288) |

The direction is now right and both intervals exclude zero; the identical
min/max is gone — the tell that the embedding read the document rather than the
writing — and 0.735 against the head's own chunk-scale 0.784 says the instrument
is inside its range. Chunk counts 563 / 564 / 563 coincide, **not** align. D54.

## Document-scale eval, whole-document — 2026-09-03 (W10 item 7): the instrument has no resolution there

~~Superseded the same day by the chunk-wise re-score above.~~ `rewrite_text` over
whole eval posts rather than pre-cut chunks: 34 of 35 eval-role archives, run 4a,
refit detector, ~$0.27. The 35th is D75.

| field | n | mean P(slop) | sd | min | max |
|---|---:|---:|---:|---:|---:|
| slop (the AI draft) | 34 | 0.4880 | 0.2260 | 0.1107 | 0.7921 |
| target (Dan's edit) | 34 | 0.4951 | 0.2224 | 0.1107 | 0.7921 |
| output (the styler) | 34 | 0.4914 | 0.2229 | 0.1107 | 0.7921 |

**The verdict is that there is no verdict, and that is the result.** The min and
max are byte-identical across all three fields and the styler scored *closer to
slop* than the draft in 34 of 34: over 25k chars the shared 90-odd percent of two
variants dominates the embedding. Decided: D54.

## Pangram cross-check — 2026-09-03, PARTIAL (W10 item 6)

The one-shot independent-detector check, against the eval_v2 stratum. **The
shipped endpoint was dead** (404: Pangram 3.0 moved to an async task API, and the
detector had only been exercised through its `post=` stub, which cannot tell you
a URL is wrong), and **the account ran out of credit** (402 after 92 of 149), so
the table is that sample, resumable by id.

| | |
|---|---|
| complete pairs (both sides scored) | 92, across 11 posts |
| mean P(AI), slop side / Dan's side | 0.8735 / **0.6300** |
| mean delta | **+0.2434** |
| pairwise, all pairs / excluding ties | 36 / 92 = 0.391 / **36 / 41 = 0.878** |
| wins / losses / **ties** | 36 / 5 / **51** |
| answers exactly on a rail (1.0 or 0.0) | 153 / 184 = 83% |
| Dan's own prose scored exactly 1.0 | **47 / 92** |

**The direction holds** — where Pangram separates the sides at all it agrees with
the voice classifier 88% of the time — **but it is blunt on this corpus**, which
is the more useful result: 83% saturation has no gradient and would veto half of
Dan's real prose. A one-shot check, never a signal (D7, D39).

## Corpus and target selection — 2026-06-27/28, live blog

**Blog frontmatter partition.** `automation:` levels — `0` (pure
human) 1591 posts, `1` 58, `2` 61; posts under `post/` (~73) and `notebook/`
(~1600) — `is_human_authored`'s defaults (D41). **Target selection funnel**
(real chunker): naive `automation:0`-only ≈ **27.4k** junky chunks → prose-only +
hygiene ≈ **19.5k** → plus `quality>6` ≈ **10k** → merged into passages ≈
**3.6k**, median ~620 chars, so hygiene discards ~87%. **85%** of merged passages
sit under a heading (D14).

### Section length — 2026-08-04, over `notebook/` + `post/`

Section prose is median **553** chars, p90 **1729**; only **13.6%** of sections
exceed 1500 chars and **0.8%** exceed 4500. Dan writes short sections, which is
why raising the merge budget alone does almost nothing: the corpus's synthetic
median target was **662** chars against a 1500-char budget. Decided: D15, D64.

### Chunk yield by budget and mode — 2026-08-04, real chunker, blog policy

| budget | section-bounded | cross-heading |
|---|---|---|
| 1500 | 2088 chunks, median 626 | 1442 chunks, median 1278 |
| 3000 | 1782 chunks, median 588 | 805 chunks, median 2603 |
| 4500 | 1727 chunks, median 578 | 615 chunks, median 3224 |

Tripling the budget moves the section-bounded median *down*, while cross-heading
packing at the existing 1500 budget already doubles effective scale. So
`chunk_mode` dominates `merge_max_chars`, and packing headings in as content is
the only route to v2 scale. Decided: D15, D64.

### Corpus composition over time — a dated series, not a running total

254 records (2026-06-27, all real) → 331 real / 3,231 synthetic (2026-07-21) →
4,169 of 4,500 synthetic (2026-07-22) → 4,624 / 4,169 (2026-08-10) → **4,595 /
4,140** (2026-08-11, after the A5 prune) → 8,319 (2026-08-17, run 4's input) →
8,265 → 8,244 (2026-09-03, after W8's 21 duplicate cells) → **8,182, 398 real /
7,784 synthetic over 497 posts** (2026-09-03, after the eval-contamination
prune). Distinct (source, target-body) synthetic cells: 3,710 → 3,686.
*This series stops here on purpose* (D52). Decided: D36, D53.

### Inline markup is a fifth of the target side — 2026-08-11, over the 4,624-pair corpus

**749,629 of 3,649,320 target-side characters — 20.5%** — is link syntax, inline
code, citation keys and inline math, none of which carries style signal. Per-pair
incidence: links **77.3%** of pairs, inline code **20.3%**, citations **17.9%**,
inline math **15.8%** — against `〈MASKED_*〉` at **0.5%** (22 pairs, all slop
side). Raw inline markdown is the distribution; sentinels are not. Decided: D27
(asymmetric masking refuted), and it is what makes the deferred
symmetric-masking experiment worth its ~$15.

### `:::` div census — 2026-08-11, over 1,751 `.qmd`: 4,139 blocks, 535,609 chars (3.78% of blog bytes)

| class | blocks | chars | verdict |
|---|---|---|---|
| figure | 2586 | 365,117 | markup — correctly excluded |
| `{#refs}` | 1200 | 18,221 | markup — correctly excluded |
| `callout-*` | 309 | 123,609 | **prose** — wrongly excluded |
| `{.notes}` | 14 | 18,787 | **prose** — wrongly excluded |

~27% of div bytes is Dan-voice prose discarded by protecting the container rather
than the marker; opting the prose classes in (via `div_is_prose`) moves
section-bounded targets 3775 → 3824 and cross-heading 2451 → 2482. Decided: D29
(the parameter), D44 (the editor could not see them).

### Corpus-stability floor — 2026-08-11, re-measured 2026-09-03

`tests/test_corpus_stability.py` against the live blog, re-cutting each arm at
its own pinned revision under its own policy and its own budget.

| arm | 2026-08-11 | 2026-09-03 (each arm pinned properly) |
|---|---|---|
| section-bounded | 3503 / 3686 | 3023 / 3028 = **99.83%** |
| cross-heading | 396 / 3686 | 417 → **4181 / 4194 = 99.69%** |

The old cross-heading value read 11% of its stratum and could catch only a
catastrophic break: one pin for both arms charged v2 cells for three weeks of
prose drift, one policy ignored that `div_is_prose` postdates v1, and one budget
charged two thirds of the corpus for a cut nobody made. Both floors are now
near-equalities, able to fail for a reason. D37.

### Real captures and how they differ from synthesis — 2026-07-22 (D18, D32)

**Unbounded but few** (2026-07-22): observed median 693 chars, p90 1866, max
4143, across only 331 records. **Real rewrites compress** (no sample size
recorded) ~**1.5×** commonly, **3×** rarely — so the real corpus supplies no
*inflated* slop and inflation must be asked for deliberately (the `verbose`
strategy); `--max-length-ratio 3.0` sits at the top of that range. The **109**
captures logged before the archive shipped (2026-07-23) have no `before.md` /
`after.md` to recut, so they are chunk-scale forever.

### What has actually varied across the synthetic corpus — 2026-08-10, over 4,169 pairs

| covariate | distinct values |
|---|---|
| `gen.model` | 9 (incl. 3 since dropped from the roster) |
| `slop_strategy` | 6 — polish 943, measured 896, casual 783, engaging 567, linkedin 520, verbose 460 |
| `gen.reasoning_effort` | 2 — `off` 2632, `high` 1537 |
| `gen.session_id` | 190 sessions, 99 distinct turn indices |
| `meta.context_mode` | `immediate` 3212, unset 957 (context dropout working) |
| `gen.temperature`, `gen.top_p`, `gen.provider_sort` | **1 each — never varied** |
| `gen.context_window` | 5, all from the OpenRouter registry, never from a flag |
| `meta.merge_max_chars`, `meta.chunk_mode` | **absent from all 4,624** — the keys, not the values |
| `gen.max_tokens` | 2 — `8192` (4130), `16384` (39) |
| `gen.replicate` | `verbose1` on 460 pairs, unset on 3709 |
| `meta.tags` | 1 — `["synthetic","paraphrase","livingthing"]` on every pair |

Decided: D21 (retire the unused preset path), D11 (covariates, not keys), D83.

### The chunk-scale facet is absent from the corpus — 2026-08-11, 4,595 records / 4,140 synthetic

`merge_max_chars` and `chunk_mode` are present on **0 / 4,140** synthetic pairs;
`context`/`context_mode` on 3,186; `synth_key`/`gen` on all. **Absent, not
null** — every synthetic pair is stamped `captured_at` 2026-07-21 or 2026-07-22,
one two-day batch, while the keys landed 2026-07-23 and 2026-08-04. So the v2
chunk-scale ablation cannot be run on this corpus and cannot be back-filled, the
target text not saying what budget produced it; **954 of 4,140** also predate
heading context (D14). And what the corpus was generated under is **inferred**:
blog policy re-derives 3,681 of 3,686 cells (99.86%) at the pinned revision —
agreement, not provenance. Decided: D36.

## Generation experiments

**Reasoning-effort sweep** (2026-07-20). **1,241 pairs, six models**, `off` vs
`medium`: the detector gap was **flat — 0.088 vs 0.092** — for roughly **3× the
wall clock**, so the cheap end won; `high`, the then-default, was never measured.
**A provider ignored `effort=off`** the same day, so the *requested* effort is
recorded and `completion_tokens` exposes the lie. D10.

**Session depth is quadratic in input tokens** (2026-07-06): each stateless call
re-sends the history, so a 32k-token session bills **~25×** the input tokens, and
`cached_tokens=0` came back live — genuinely quadratic dollars. **Prompt caching
is unexploitable here** (2026-07-07): the only shared prefix is the slop system
prompt, under every provider's ~1024-token minimum. **Window position does not
affect slop** (2026-07-21): drift correlation **r = −0.05**. **Silent-drop
post-mortem** (same day): a `--session-turns 128` run discarded **46%** of planned
work when the token budget bound first. **Re-key epoch double-weighting**
(2026-07): a re-key doubled the weight of **~1,970** targets; `--skip-covered`
fixes it. Decided: D13, D11.

### Rendezvous vs modulo arm assignment — 2026-08-10; 20k targets, 6 arms

Fraction of the corpus reassigned by a roster change ("min" = theoretical floor):

| roster change | modulo | rendezvous | min |
|---|---|---|---|
| drop 1 arm | 83.2% | **16.7%** | 16.7% |
| add 1 arm | 85.8% | **14.4%** | 14.3% |
| swap 1 arm | 16.9% | 28.7% | ~33% |
| reorder only | **100.0%** | **0.0%** | 0% |

Balance is comparable (5.8% modulo vs 4.3% rendezvous spread). Modulo made the
winner a *position*, so a pure reorder moved the whole corpus. The switch was
free when taken: the roster had already drifted, so resume overlap was zero
either way. D12.

## Data hygiene / QA

**Identity pairs** (2026-07-21): **386** pairs where the generator returned its
input unchanged, all from the qwen3-8b @ `effort=off` cell, median 0.98; gated by
`--max-transform-sim 0.95`. **Inflation tail**: qwen3-32b produced **334** florid
pairs at **4.4–6.8×**, gated by `--max-length-ratio 3.0`. **Near-copies are good
styler data**: **~19%** exceed 0.85 — label noise for the detector, restraint for
the styler. D18.

**Unanchored `$$` mangled sentences inside training targets** (defect A5,
2026-08-11): `high-$$d$$ behaviour` → `high- behaviour` **in a training target**.
**145 of 1,751** blog files contain a non-standalone `$$`; blast radius **24
target texts, 29 pairs (0.70%), 15 posts** — corpus 4,624 → **4,595** after a
one-off prune (deleted in 0.3). The other four A-series defects had **zero**
corpus impact. D29, D38.

**Capture diffing loses almost nothing** (2026-08-04, all 41 archives): the
theory that `diff_chunks` discarded paragraph-count-changing edits died on
measurement — **1** dropped opcode. **Gap-coalescing vs fixed-width windowing**:
coalescing gives **87 pairs, median 1191 chars, p90 4230** (uncoalesced: 130 /
845 / 1938) and changed-fraction **0.96** (p10 0.86, none below 0.5) against
**0.41** under a 4500-char window; bridging tolerance **2 paragraphs / 800
chars**. D81.

### Two constructs the corpus no longer holds — 2026-09-04 (F12g/j)

`migrations.find_inventing` over `_training_pairs/pairs/`, and a `:::` depth scan
over `lib.gather_qmd_files(base=~/Source/livingthing)`. Read-only, no spend.

| scan | over | found |
|---|---|---|
| synthetic pairs inventing an anchor | 8,156 records | **0** |
| the same scan under `--include-real` | 8,156 records | **49** real, all Dan's own cross-references |
| an unclosed `:::` opener | 1,758 blog `.qmd` | **0** |

The first says the cull is spent (D73) and the second says why `--include-real`
was never its default (D65) — those 49 are the author wiring a note into his own
graph, not a generator hallucinating. The third is why the segmenter's "an
unclosed opener yields no span at all" stays as it is (D76).

### The synth walker's gap is empty — 2026-09-03

stylebot's `iter_targets` predicate and the blog's `gather_qmd_files` agree on
all **1,757** `.qmd` in the live tree: the two skip-lists are disjoint, and no
`.qmd` has ever lived under `renv`/`packrat`/`rsconnect`/`node_modules`. Exactly
one file ever fell in the gap (`tests/fixtures/segmenter_conformance.qmd`, since
moved under `fixtures/hard/`) and `make_selector` rejected it anyway. Asserted by `tests/test_qmd_walker.py`. D46.

### `scores.jsonl` orphan rot — 2026-09-03 (W6 audit)

The top-level `_training_pairs/scores.jsonl` was last written 2026-07-07 for a
corpus that no longer exists: **83 of its 270 rows are orphans** — rows whose
`record_id` no longer joins to any pair. The file is append-only and joins on
`record_id`, a re-cut or prune strands the superseded row, and no consumer checks
the join, so an orphan counts as a pair — inflating every `summarize_scores` mean
and putting a phantom row in a committed `covariate_table`. `--prune-orphans`
fixes the file, not the reason it goes stale, which is why verdicts worth keeping
go under `runs/<run_id>/` with a manifest pinning corpus, splits and detector. No
command is recorded for the count. Decided: D63, D42.

## The LLM judge — measured 2026-08-17, and it is the weak signal

Prompted by "check that this judge is worth it; it costs an insane amount".
Measured on the run-4 eval: 160 opus-4-8 calls over 40 eval chunks.

| stratum | judge pairwise | detector pairwise |
|---|---|---|
| **real** captured pairs (n=20) | **3/20 (15%), 14 ties** | 15/20 (75%) |
| synthetic pairs (n=20) | 16/20 (80%), 4 ties | 20/20 (100%) |

**Cost is not the problem — the estimate was wrong by an order of magnitude.**
122,103 prompt + 11,685 completion tokens over 160 calls: **~$2.71 at claude-opus
list**, a full 127-chunk four-field pass ~$8.60, and OpenRouter reported $0.0000
because the Anthropic arm bills through BYOK (D66). **The problem is that it
cannot see real slop**: its competence is carried entirely by synthetic slop,
while on real slop it is worse than chance and ties 14 of 20 — the 1–5 integer
scale has no resolution at the distances that matter, and the aggregates still
look sane (3.25 → 4.08 → 4.20), which is how it went unnoticed. **It cannot
arbitrate two adapters** (4a−4b +0.025, CI [−0.124, +0.174] at n=40; resolving
0.10 needs n ≈ 88). A smoke alarm, never a tie-breaker. D40, D7.

## Register — the labels are not free (2026-08-17)

Prompted by "my style has many registers and the model is struggling to hit a
median register". A single SFT LoRA on a multimodal target learns the conditional
mean, so a median register is expected, not a defect of any run; the cheap fix
would be a register label, if labels were free. K-means over StyleDistance
embeddings of **2,500 distinct Dan-side chunks** (≥400 chars): silhouette **0.232**
at k=2, 0.231 at k=3, **0.117** at k=4, 0.090 at k=6, and reading the exemplars
what separates is **topic, not register**. The same backbone still supports a
*supervised* Dan-vs-slop direction at 0.73 real pairwise: supervision picks the
direction, k-means picks the biggest one. Decided: D22.

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
"is this slop" question while a content-independent *style* embedding holds —
which made Pangram and a contrastive fine-tune unnecessary for v1. **The score
distribution is narrow** (2026-07-06, 264 pairs): all of it in ≈[0.32, 0.68], Dan
p5/p25/p50/p75/p95 .32/.40/.46/.51/.62 against slop .40/.47/.55/.59/.68, hence
the warped editor buckets `[0.51, 0.55, 0.59, 0.65]` = Dan-p75 (below it,
unmarked, sits three-quarters of Dan's own prose), slop-p50, slop-p75,
≈slop-p95 — the rule that re-derives the edges after a refit. **Editor latency
is a non-issue** (same day, MPS): 5.0 s cold load once, 7 ms warm for one
paragraph (28 µs of it the head's dot-product), **34 ms** for a 30-paragraph
notebook. D7, D20, D44.

## The alias seam — 2026-09-03 (D34), re-measured 2026-09-04 (F3)

`uv run python` over `stylebot.splits` and a **tmp copy** of the live
`splits.json` (eval_v2, `splits_sha256=05b495adac68cdc4`):

| post, as `meta.source` keys it | renamed to (blog `77a1b398aa`) | role the new path hashes to |
|---|---|---|
| `notebook/empowerment.qmd` | `notebook/informational_empowerment.qmd` | styler |
| `notebook/learning_from_madness_of_crowds.qmd` | `notebook/wisdom_from_madness_of_crowds.qmd` | detector |

Both are pinned eval posts, and nothing caught the move: the STALE warning is
about pairs and the old paths still carried theirs, `role_of` is open-world, and
the canonical run passes no `--role` — so the next `synth` would have cut
frozen-eval prose. **The same seam one layer up, 2026-09-04**: `role_of`
canonicalised the query then tested it against the RAW pinned list while
`extend_eval` appended the caller's spelling, so extending with the renamed post
left **both** spellings answering `styler`. Decided: D34.

## Eval-stratum composition — re-measured 2026-09-03 (W10)

The frozen stratum against the live 8,244-pair corpus reads **6 posts / 103
pairs / 41 real / 62 synthetic** through `iter_pairs` and **89 / 41 / 48**
through the detector's `transform_sim > 0.85` gate — both right, answering
different questions. **It is one post wearing five hats**: `automatic_coding.qmd`
carries **77 of the 103 (75%)**, so run 4's "127 chunks, paired" is 6 clusters.
Pairs by source-post `automation`, alias-resolved:

| role | automation 0 | 1 | 2 | of which REAL: 0 / 1 / 2 |
|---|---:|---:|---:|---|
| eval | 4 | 79 | 20 | 4 / 17 / 20 |
| detector | 2,596 | 517 | 139 | 11 / 30 / 138 |
| styler | 4,112 | 569 | 208 | 11 / 33 / 134 |

**168 of the detector pool's 179 real pairs (94%) sit on `automation: 1/2`
posts** — the post is marked AI-touched *because* Dan captured himself de-slopping
it. Capping at `automation: 0` leaves the `real` facet 11 pairs; capping only the
**synthetic** ones drops 488 of 3,252 (15%) and costs no real pair (D41). **What
an eval re-cut would buy, and take:**

| eval_v2 | posts | pairs | real | top-post share | detector real left | styler real left |
|---|---:|---:|---:|---:|---:|---:|
| v1 (then) | 6 | 103 | 41 | 75% | 179 | 178 |
| + top 4 styler-role | 10 | 190 | 128 | 41% | 179 | 91 |
| **+ top 6 styler-role (recommended)** | **12** | **245** | **149** | **31%** | **179** | **70** |
| + all 30 styler-role | 36 | 544 | 219 | 14% | 179 | 0 |
| + all 46 real-pair posts (maximal) | 52 | 1,025 | 398 | 8% | **0** | **0** |

The maximal cut zeroes both other roles' real strata. Decided: D42.

## eval_v2 — applied 2026-09-03 (W10 item 4 / D42)

`dan-style extend-eval --apply` against the post-prune corpus (8,182 records).
These differ from the sizing table above, which predates the prune.

| stratum, as `check-splits` reports it | posts | pairs | real | synthetic | top-post share |
|---|---:|---:|---:|---:|---:|
| eval v1 (`splits_sha256=c8c319578707bb5b`) | 6 | 41 | 41 | 0 | 37% |
| **eval v2 (`splits_sha256=05b495adac68cdc4`)** | **12** | **177** | **149** | **28** | **22%** |
| styler | 290 → 284 | 4,138 → 4,002 | 178 → **70** | 3,960 → 3,932 | 3% |
| detector | 198 → 198 | 2,709 → 2,709 | 179 → **179** | 2,530 → 2,530 | 2% |

**The whole cost falls on the styler's real pairs, by construction** — all six
added posts were styler-role, so the detector pool does not move by a pair, and
real prose goes from 4.3% of the styler's pairs to 1.7%. `ai_search.qmd` brings
28 synthetic pairs, so the stratum is 16% synthetic again (D53's open half), and
document-scale eval goes from 3 documents (86,820 chars) to **35** (**1,561,740**
chars — of 86 capture archives, those on eval-role posts), ≈$0.015 → ≈$0.27.
**The refit moves more than the effects it measures:**

| | old | new |
|---|---|---|
| `git_sha` / `trained_at` | `da335af66d` / 2026-08-17 | `c0e2ac85be` / 2026-09-03 |
| `n_pairs` (real / synthetic) | 2,751 (215 / 2,536) | 2,341 (179 / 2,162) |
| `splits_sha256` / `split.eval_version` | absent | `05b495adac68cdc4` / 2 |
| CV pairwise / AUC | 0.9710 / 0.9652 | 0.9739 / 0.9666 |
| **real facet** pairwise / AUC | **0.7337** / 0.6580 | **0.7840** / 0.6927 |
| paired slop − Dan Δ on the *same* 41 v1 pairs | **0.1466** | **0.1724** |

The pool shrank for reasons that are *not* eval_v2, and the real facet rising on
36 fewer pairs is inside the ±0.05 fold spread. **A refit alone shifts Δ by 0.026,
against run 4a's headline 0.038**, so `compare_scores` refuses the pair. The
**chunk-level** interval narrows as n grows (0.137 → 0.081) while the **cluster**
interval *widens* (0.076 → 0.099) — at 6 clusters the bootstrap resampled
post-means that happened to agree. Decided: D39, D42.

## What a stale-corpus prune would cost and buy — measured 2026-09-03 (W12)

`ai-style data prune-stale`, dry run, against the live 8,244-record corpus re-cut
under the blog's composed `chunk_policy()`. **Nothing was pruned.**

**Orphans: 603 of 7,222 distinct synthetic cells (8.3%), carrying 672 pairs
(8.6%) across 124 posts.** Zero real captures, structurally: a capture's target
comes from a paragraph diff, never from `iter_targets`, so the 398 real pairs are
UNCHECKED rather than clean. Why cells stopped reproducing:

| cause | section-bounded | cross-heading | cells |
|---|---:|---:|---:|
| W2's fence rule | 0 | 0 | **0** |
| `COMMENT_MARKER` in `ignore_markers` | 3 | 24 | 27 |
| `div_is_prose` applied to cells cut before it existed | 70 | 0 | 70 |
| **prose Dan has edited since generation** | 227 | 265 | **492** |
| irreducible residue (synthesis ran against an uncommitted tree) | 5 | 12 | 17 |

**W12's premise is wrong about its own cause**: the two files whose spans W2
moved carry zero cells, and four fifths of the orphans are Dan having edited the
post since. The 70 `div_is_prose` cells sit on the **frozen** section-bounded arm,
so pruning those 349 pairs is permanent loss. **Eval contamination: 62 synthetic
pairs, all `automatic_coding.qmd`**, 41 real captures spared, 25 in both sets — so
the two prunes drop 709, not 734, and the dilution it was wanted for barely moves
(4.06 → 4.04). D53.

## Corpus blob cost — measured 2026-09-03 (W6)

W6 cited "1.56 GB of blobs under `_training_pairs/`… in a repo whose `.git` is
1.8 GB", which reads as the corpus being ~85% of the repository. That is
**logical** blob size. On disk (`%(objectsize:disk)`) the same 124 versions cost
**49.5 MiB** — an append-only JSONL deltifies almost perfectly — so everything
under `_training_pairs/` is 52 MiB of 1,107 MiB of packed blob content, **4.7%**.
Meanwhile `.git/subtree-cache` is **683 MB**, a `git subtree` scratch directory
and not history: deleting it reclaims 13× what a filter-repo pass would, with no
rewrite and no force-push. Decided: D51, D50.

## Training runs — 2026-07-21 to 2026-08-17 (D19, D39, D42)

**Base model rate**: `Qwen/Qwen3.5-9B` at **$1.463/1M train tokens** (2026-07-21),
Tinker sampling ~**$0.60/1M**; the same day's dry run put 1,666 train / 148 val
pairs at ~1.12M tokens ≈ **$1.64/epoch**.

| run | corpus sha | val NLL | cost | note |
|---|---|---|---|---|
| 1 (`20260721-run1`) | `0838ae470afef6c3` | 1.018 | $1.97 | |
| 2 (2026-07-22, post-surgery) | `ea9f5981f8074cb3` | 0.952 | $1.49 | served 2026-07-22 → 2026-08-21 |
| 4a (2026-08-17) | recut 8,319-pair | test 0.7630 | $5.15 | `styler_pair_ok`, 3,944 / 446 |
| 4b (2026-08-17) | same | test 0.7457 | $5.36 | keeps 188 anchor-damaged, 4,182 / 396 |

Run 1 smoke (2026-07-22, 40 val pairs): slop 0.507 → **adapter 0.422** → target
0.352, against a prompt-only baseline of **0.552** — *worse than the untouched
input* — which the adapter beat on 35/40. Run 2 chose fidelity over voice score:
softer metrics (0.425 vs 0.415), but the over-compression tail retreats (p10
output/slop 0.76 vs 0.60). **Run 4's verdict, 127 chunks, paired:**

| signal | slop in | 4a out | 4b out | Dan | paired 4a−4b (95% CI) |
|---|---:|---:|---:|---:|---|
| detector P(slop) | 0.5823 | **0.2125** | 0.2221 | 0.1744 | −0.0096 [−0.0335, +0.0143] |
| Vale alerts | 2.12 | 1.66 | 1.67 | 0.91 | −0.008 [−0.091, +0.076] |
| judge (n=40) | 3.25 | 4.08 | 4.05 | 4.20 | +0.025 [−0.124, +0.174] |
| outputs damaging an anchor | — | 7/127 | 6/127 | — | 6 lost + 2 invented vs 6 + 0 |

Every signal says the two arms are indistinguishable: the filter costs 15% of the
scarce real pairs (31 of 172) and buys nothing measurable at n=127, including on
anchor fidelity at inference. The pre-registered rule keeps it on a tie.

### Run 4's pre-flight inputs — 2026-08-17, from the retired blog plan

Read on 2026-08-17 from `pairs.jsonl`, `dan-style train --dry-run` and the two
run manifests, by the blog's `_plans/styler-run4-preflight-2026-08.md` (a P7
stub since 2026-09-04; `git show 367f170565:` there). The inputs the verdict
above rests on that appear nowhere else:

| what | value |
|---|---|
| `pairs.jsonl` before the recut | 8,335 pairs — 468 captured / 7,867 synthetic |
| `--dry-run`, pre-recut, no anchor filter | 4,905 (188 real / 4,717 synthetic), ~$5.89 |
| assembled post-recut, unfiltered (4b's arm) | 4,578 (172 / 4,406) |
| assembled post-recut, anchor-filtered + rostered (4a's arm) | 4,390 (141 / 4,249), ~$5.15 |
| 4a eval, interrupted at 48 of 127 chunks (Tinker 402) | detector 0.2899 in → 0.2346 out (Dan 0.2007); Vale/chunk 1.04 → 1.00 (0.83) |
| `preferences.jsonl` | 13 records |

The 48-chunk row was a smoke test of the scoring path — unpaired, no judge, no
4b — and the 127-chunk verdict above supersedes it. The filter's pre-recut cost
(29 of 188 real) became 31 of 172 after the recut, because bridged captures
carry more anchors. Thirteen preference records are why DPO stayed out of
scope. Decided: D40 (the verdict's interval), D43 (the by-passage weighting
the arm split exposed).

### Run 4's verdict, re-analysed 2026-09-03 (W10) — the gain over run 2 does not survive

Same score files, re-read through `eval.compare_scores` with the POST cluster
bootstrap and the real/synthetic facet (127 chunks: 65 real / 62 synthetic, 85 of
them from `automatic_coding.qmd`):

| comparison | facet | paired Δ | 95% CI | verdict |
|---|---|---:|---|---|
| 4a − 4b | all 127 | −0.0096 | [−0.0168, −0.0017] | (see caveat) |
| 4a − 4b | **real (65)** | +0.0011 | [−0.0168, +0.0116] | indistinguishable |
| 4a − run 2 | all 127 | −0.0376 | **[−0.0507, +0.0021]** | indistinguishable |
| 4a − run 2 | **real (65)** | **−0.0033** | [−0.0143, +0.0033] | indistinguishable |

~~The earlier headline — "paired 4a − run 2 = −0.0376, 95% CI [−0.0616, −0.0136],
excludes zero" — was a chunk-level normal-approximation interval.~~ Clustered by
post it straddles zero, and on the *real* pairs alone the effect is eleven times
smaller: 4a's promotion rests on treating 127 chunks from 6 posts as independent
and on counting our own generators' paraphrases. Not a reason to un-promote it,
but "a real gain" is not what this data supports. Decided: D40, D42.

### Effective passage weight — measured 2026-09-03 (W9)

The corpus is cut at four diets over the same prose, so a run walks each stretch
several times under a manifest claiming one epoch. Over all 8,244 records,
**5,325 distinct targets are strict substrings of another target in the same
post**, carrying **6,157 pairs (74.7%)**. Reproducing run 4a's selection:

| | count | mean |
| --- | --- | --- |
| train pairs | 3,981 | |
| distinct target bodies | 3,533 | **1.13** pairs each |
| passages (source + containment) | **982** | **4.05** pairs each, max **15** |
| val pairs / passages | 392 / 95 | 4.13 |

The multiplicity is the **budget sweep**, not generator replicates (653 of 678
multi-pair passages span ≥2 budgets), and it re-weights the scarce side — real
pairs are **3.09% of train pairs but 12.2% of train passages** — which the by-post
val split cannot show, since val inflates identically. Capping at one pair per
passage leaves 982 pairs; at 2, 1,660; at 3, 2,212. **No run has been re-cut.** D43.

## The fit judge (coherence) — first verdict 2026-08-21

`ai-style judge-fit` over the frozen 127-chunk eval stratum, both arms' existing
rewrites, neighbours held fixed and identical across arms, presentation order
randomised per case:

| comparison | n | decided | A wins | B wins | ties | A-win rate (Wilson 95%) | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| run 4a vs run 2 | 62 | 18 | 8 | 10 | 44 | 0.44 [0.246, 0.663] | **wash** |
| run 4a vs run 2, interior only | 41 | 10 | 4 | 6 | 31 | 0.40 [0.168, 0.687] | wash |
| **control**: Dan vs slop | 62 | 43 | 31 | 12 | 19 | **0.72 [0.573, 0.833]** | **decisive** |
| control, interior only | 41 | 28 | 22 | 6 | 13 | 0.79 [0.605, 0.898] | decisive |

The control is the instrument's own check: a judge that ties everything produces
the top row whether or not the arms differ. The tie rate is the sharper tell —
71% between the adapters against 31% in the control — so the wash is a property
of the adapters, not of the judge. Position bias is not driving it: order is a
hash of the case id, so arm A led 23 times of 62 in both runs, and in the control
`slop` led 39 times and still lost 12–31. Run 4a closes more of the detector's
slop→Dan gap than run 2 while being indistinguishable at boundaries; those agree
once the detector is read as the paragraph-scale voice head it is.

The stratum placed 62 of 127 (41 interior): 43 by corpus sibling, 7 by capture
archive, 12 by matching the live post. The 65 dropped are overwhelmingly
pre-2026-07-22 captures, which have no archive. Decided: D97.

## Inference and serving

**Full eval against baseline** (2026-07-22, 40 val pairs, three signals agree):

| signal | slop | output | target | prompt-only baseline |
|---|---|---|---|---|
| detector P(slop) | 0.507 | **0.414** | 0.352 | 0.552 |
| judge (1–5) | 2.40 | **3.45** | 4.33 | — |
| Vale alerts | 3.3 | **1.28** | 1.2 | — |

**Out-of-distribution length makes the styler copy its input** (2026-07-22): a
**5.4KB single-chunk** rewrite came back **byte-identical**, while the same post
chunked at training scale moved whole-body P(slop) **0.589 → 0.281** — so the
inference budget is pinned to the training chunk scale (D15).

**Inline anchors would reject a fifth of good rewrites** (2026-08-11, all 4,624
pairs, inference direction) — share of pairs where a genuine Dan rewrite would be
disqualified: today (links, fences, headings) ~~**6.4%**~~ (re-measured under C13,
below); + inline code 11.9%; + inline math 13.5%; **+ both 19.0%**. So they
default **off** in `run` and are **always on** in `rewrite_candidates` (D27, D20).
**MLX local-serving parity** (2026-07-22, run 2, 20 val pairs @ temp 0.3):
detector output means **tinker 0.4627 vs local 4-bit 0.4529**, Δ **−0.0098**,
inside the 0.03 gate; drift within the 0.05 gate, zero template artifacts. D8, D48.

### Anchor keys stop at trailing punctuation and email local parts — 2026-09-04

C13: the bare-URL pattern in `infer._ANCHOR_RES` now drops trailing `.,;:!?)`
(CommonMark's autolink rule) and the citation pattern needs `(?<![\w.\\])@`.
Instrument: `infer.is_anchor_faithful(record)` — the guard `run` applies,
symmetric, inline off — and `(…, gain=False)` for the loss half, a keyless
`uv run python` loop over `livingthing/_training_pairs/pairs/*.jsonl`, run on
the old patterns and the new; "real" is a record with no `meta.gen`.

| stratum | pairs | loss-only, old → new | symmetric, old → new |
|---|---:|---:|---:|
| real captures | 407 | 47 (11.5%) → **44 (10.8%)** | 85 (20.9%) → **83 (20.4%)** |
| synthetic | 7,750 | 298 → 300 (3.9%) | 298 → 300 (3.9%) |
| all pairs | 8,157 | 345 → 344 (4.2%) | 383 → 383 (4.7%) |

Four real verdicts flip. Three rescues are the `@` edit, not the URL one — the
old key was `@k` out of `maj@k` / `Pass@k`; no real pair turns on URL punctuation
today. The one new failure is right: Dan's target reads `of@Agrawal2019Economics`,
which pandoc will not cite either. The three new synthetic failures are D32's
reverse-edit artifact — Dan wrote `\@miteron`, the generator un-escaped it, so the
slop side carries a key the target never had; `\@` is excluded because a
pandoc-escaped `@` cannot cite. The struck 6.4% was loss-only over all strata of a
4,624-pair corpus, not this denominator. D32.

## Cost: forecast vs actual — v1, through 2026-08-17

The plan budgeted **$180–290** for synthetic generation inside a ~$200–330
envelope. Actual v1: **$15.27** billed for 4,169 pairs — **~$0.0037 a pair**, an
order of magnitude under forecast, and the unit rate to size any re-synthesis
against. Training likewise: run 1 $1.97, run 2 $1.49 against a $6–15 estimate.
The v2 curriculum is ~8.9M target chars, ≈ **$39**. Decided: D66.

## Anchors and placeholders: v2 post-gate distributions — 2026-08-17

`MAX_LENGTH_RATIO` (3.0) and `MAX_TRANSFORM_SIM` (0.95) were calibrated on
1500-char chunks and had not been re-checked at 4500. Faceting the written pairs
of the first v2 batches by `meta.merge_max_chars` (cross-heading; 4,532 stamped
pairs, 2,629 / 1,096 / 807 per budget); no command was recorded.

| budget | `transform_sim` p90 | length-ratio p90 | pairs at sim ≥ 0.95 | pairs above 3× | median anchors per target |
|---|---:|---:|---:|---:|---:|
| 1500 | 0.90 | 1.91 | 11 | 0 | 4 |
| 3000 | 0.91 | 1.67 | 6 | 0 | 7 |
| 4500 | 0.92 | 1.66 | 1 | 0 | 10 |

Neither threshold behaves differently with scale, and nothing piles up against
either boundary. **These are post-gate survivors**, so this bounds the *shape* of
what got through, not the rejection rate (`skipped_degenerate` has that). It let
the run-4 pre-flight skip the 4500 audition, and is why the two hygiene gates
stay as calibrated (D32's siblings).

### Anchor delta over the v1 corpus — 2026-08-11/12

`infer.anchor_delta` over the 4,595-pair corpus through the production
extractors (`pairs.extract_slop` / `extract_target`); the measurement stylebot#1
was briefed on, and the one D32 cites.

| anchor-bearing pairs: 3,775 | pairs | share |
|---|---:|---:|
| identical anchor multiset | 2,803 | 74.3% |
| target lost anchors only | 117 | 3.1% |
| target gained anchors only | 678 | 18.0% |
| both | 177 | 4.7% |

855 pairs (18.6% of the corpus) carry invented anchors, 2,062 instances: 49.9%
external URLs, 37.1% internal site links, 12.9% citation keys. Invention
outnumbers loss ~7:1, and the issue thread's reordering worry is 51 pairs (1.1%)
against 876 fabricating in the first count — its "53% of chunks carry 2+ links"
is exposure, not incidence. Decided: D32.

### Anchor invention is a property of the arm, not of the passage — 2026-08-12

`infer.invented_anchors` over the 4,140 synthetic pairs; rate is per pair whose
*target* carries an anchor. "Invention" means the generator dropped a link while
producing slop, so the reverse-edit swap made Dan's side look like it conjured one.

| arm | pairs | rate | in the live roster? |
|---|---:|---:|---|
| deepseek/deepseek-chat-v3-0324 | 41 | 74.2% | no |
| nvidia/nemotron-3-nano-30b-a3b | 731 | 60.4% | no |
| meta-llama/llama-3.3-70b-instruct | 75 | 51.5% | **yes** |
| qwen/qwen3-8b | 151 | 38.0% | no |
| qwen/qwen3.5-9b | 553 | 37.9% | **yes** |
| qwen/qwen3.6-35b-a3b | 427 | 24.4% | **yes** |
| google/gemma-4-26b-a4b-it | 1,082 | 5.7% | **yes** |
| anthropic/claude-sonnet-4.6 | 1,059 | 1.3% | **yes** |

A 46× spread, and the two highest-volume arms are the two cleanest — so the
corpus-wide 18.6% understates what a gate costs on the *current* roster. Two
secondary effects argue for `--gate-retries` over a bare gate. **It is mostly
per-sample**: of 454 targets seen by more than one arm, 148 lost an anchor under
at least one but only 29 under *every* arm. **A bare gate biases coverage against
link-dense prose**: P(rejected) runs 16.1% at one anchor to 35.5% at five.
**Gate precision**: no false positives in the cleanest arm's 11 rejections; the
systematic near-miss is pandoc citation locators, kept deliberately. D32.

### Anchor provenance by stratum — 2026-08-12

From `prune_anchor_inventing_pairs`'s own dry-run `kinds` breakdown, run twice —
default, then `--include-real` — to split the strata. Both tools are now deleted
(D73); `infer.invented_anchors` is the surviving predicate.

| stratum | external URL | internal cross-reference | citation key |
|---|---:|---:|---:|
| synthetic | **51.4%** | — | — |
| real captures | **11.5%** | **53.8%** | **26.9%** |

The synthetic side's inventions are a generator artifact of the reverse-edit
swap; the real side is Dan wiring a note into his own graph while editing. On the
v1 corpus the cull ran 4,595 → 3,790 synthetic-only, 3,740 with `--include-real`,
and 657 of 3,685 targets went back to uncovered. Decided: D65, D32.

### Anchor survival collapses at v2 chunk scale — 2026-08-12

First gated run at the 3000-char cross-heading budget; source `lost_draws.jsonl`
plus the live corpus. Draws rejected on the residue: qwen3.6-35b-a3b **100%**
(0 written of 105), qwen3.5-9b 99%, gemma-4-26b 98%, deepseek-v4-flash 78%.
Rejections are **wholesale, not marginal** — across 1,215 anchor rejections,
median **5** anchors deleted per rejected draw, mean 6.2, max **29** — and
redrawing does not help: 106 cells redrawn, 212 extra draws, **2 recovered**.

| config | chunks | median chars | median anchors | 5+ anchors |
|---|---:|---:|---:|---:|
| v1 section-bounded @1500 | 8,811 | 443 | 2 | 16.8% |
| v2 cross-heading @1500 | 4,840 | 1,268 | 4 | 46.6% |
| v2 cross-heading @3000 | 2,783 | 2,333 | 7 | 66.3% |
| v2 cross-heading @4500 | 2,207 | 2,388 | 8 | 70.9% |

~~This corrects the "rejection plateaus" reading above.~~ A chunk passes only if
*every* anchor survives, so P(pass) ≈ s^k, and cross-heading packing moves median
k from 2 to 8 to 11: **exponential in k, not asymptotic** (the earlier plateau was
measured on v1's short chunks). **The residue self-selects for difficulty** — at
3000, written cells carry median 8 anchors, the 101 still missing median **10**.
The slop prompts already instruct verbatim link preservation, so this is
instruction-following collapsing under length and density. D33.

### Chunk budgets overlap, so the scale arms are nested — 2026-08-12

Targets per cross-heading budget, and how many are unique to it: 1500 → 2,491 /
2,366 (95.0%); 3000 → 1,282 / 1,077 (84.0%); 4500 → 937 / 761 (81.2%). Pairwise
overlaps 1500∩3000 = 100, 1500∩4500 = 71, 3000∩4500 = 151, all three = 46 — a
section shorter than the smallest budget is packed identically at every budget,
and since `_synth_key` hashes target text and not the budget, such a passage is
one cell stamped by whichever batch reached it first: of the targets in both the
1500 and 4500 sets, 57 carry `merge_max_chars: 1500` and 2 carry `4500`, the
1500 batch having been interrupted at 1,985 of 2,491 — so the stamp is
contaminated by run order at the ~2–16% level. How to analyse it honestly: D82.
Decided: D43, D82.

### Length-ratio distribution of a real rewrite — 2026-08-11, over the 4,595-pair corpus

`len(target)/len(slop)` where the slop side exceeds 40 chars, n=4,590:

| p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 0.31 | 0.34 | 0.39 | 0.61 | 0.79 | 0.93 | 1.25 | 2.16 | 9.76 |

Mean 0.81, min 0.18. The transform leans compressive **on average only**: 13.6%
of real rewrites are longer than their input. This sets
`infer.MIN_REWRITE_LENGTH_RATIO = 0.3` (below p0.1; flags 2 of 4,590) and is why
the truncation guard is a floor with no ceiling — length cannot distinguish an
expansive rewrite from a rambling one. The source document's 0.9 threshold, right
for a copy-edit task, would have called 70% of real rewrites truncated. D27.

### Placeholder retention across models — 2026-08-11

Measured in **livingthing** during an `ai-preen` roster bench, so **not** a
stylebot measurement: different task, different code. Full write-up in
[`placeholder-design-2026-08.md`](placeholder-design-2026-08.md). 12 link-bearing
chunks, ~122 placeholders each; retention = returned verbatim.

| scheme | gemini-3.1-flash-lite | qwen3.6-35b-a3b | glm-4.7-flash |
|---|---|---|---|
| `〈URL_0000〉` Unicode brackets | 100% | 45.9% | 0% |
| `https://example.com/u/<sha1>` URL-shaped | 100% | **68.9%** | 6.0% |
| `[[URL_0000]]` plain ASCII | 100% | 37.3% | 12.9% |

**Not character exoticism** — the ASCII "fix" came last; what predicts survival
is whether the token looks like what belongs in that slot. **Retention is a model
property**: 100% / 37–69% / 0–13%, so the scheme only moves the middle band,
where open weights sit. D27, D33, D64.

**Sentinel exposure in the corpus** (2026-08-12, 8,264 pairs): `〈MASKED_*〉` is on
the slop side of **21** pairs and the target side of **zero** — the styler has
never been asked to *emit* the sentinel `STYLE_SYSTEM` promises to preserve,
which is what makes inference-side masking out-of-distribution. D33.

## The fence closer — measured 2026-09-04 (F15)

`STYLEBOT_SWEEP_BLOG_ROOT=~/Source/livingthing uv run pytest
tests/test_segment_hard.py`, over 1,758 blog `.qmd` at blog `f36033d713`, run
with the old `CODE_FENCE_RE` and the new one in the same pass.

| post | fence span before | after | prose returned |
|---|---|---|---|
| `notebook/blogdown.qmd` | 5612–7229 | 5612–5710, 7037–7229 | **1,327 chars**, three `###` headings and their paragraphs |
| `notebook/quarto_websites.qmd` | 21660–22610 | 21660–21871, 22499–22610 | 628 chars |

**The other 1,756 posts are span-identical under both named configurations**,
and the sweep's invariants now pass everywhere (2 files broke them before).
Re-cutting these two under the blog's `chunk_policy()` moves 9 of their 28
targets, so `prune-stale` would newly orphan pairs cut there — 79 of the corpus
come from these two posts, and D53 does not prune. Decided: D69.

## The hard-post fixture set — 2026-09-03 (D68), re-measured 2026-09-04 (F15)

`uv run pytest tests/test_segment_hard.py` against the opt-in
`STYLEBOT_SWEEP_BLOG_ROOT=~/Source/livingthing uv run pytest
tests/test_segment_hard.py`, on Dan's laptop. The selection itself is recorded in
`tests/fixtures/hard/manifest.json` at blog `25cf42e5c5`.

| | value |
|---|---|
| fast path, fixtures only, every `pytest` run | **0.09 s** |
| the same `violations()` invariants over the whole blog | **3.0 s** |
| blog `.qmd` the sweep reads | 1,757 |
| fixture posts: at selection → after F15 | 13 → **11** |
| `tests/fixtures/hard/` on disk, against D68's re-evaluate-if | 64 KB vs ~140 KB |
| `KNOWN_DEFECTS`: at selection → after D69 | 3 → **0** |
| edge-case features the set greedily covers | 22 |

The ranking is `3 × (distinct protected kinds) + Σ min(count, 10) per kind +
10 × Σ log2(1757 / posts_with_feature)` over those 22 features, and the set is a
greedy cover of them rather than the top N by score — which is why it holds three
constructs whose sole corpus instance no hand-picked dozen would have caught.
Two fixtures left on 2026-09-04 because their only construct, an unclosed `:::`
opener, went to zero corpus-wide when the blog fixed its asymmetric div fences.
Decided: D68.

## Surface size

**`ai-style synth` flag audit** (2026-08-10): **54** click options against **19**
for the next-largest subcommand — 14 load-bearing / 24 policy constants / 16
retirement candidates; retiring the `gpt`/`local` preset path left **50**, and
the 2026-08-13 close (four more retired) left **49**.
[`synth-flag-audit.md`](retired/synth-flag-audit.md). Decided: D21, D83.

**Segmenter implementations: 7 → 3** (2026-08-11), then **3 → 1** (2026-09-03).
Five copies inside stylebot merged into `stylebot.markdown`; each survivor was
claimed undeletable on grounds that turned out to be packaging, not segmentation.
Measured before each deletion: the blog copy was **byte-identical on all 1,758
`.qmd`**; the TypeScript port matched on **1,728 of 1,757**, all 29 differences
corrections — plus **385 callout paragraphs, 96,796 chars across 148 files** it
could not see, `div_is_prose` being a blog-side callable. D26, D28/D30, D44.

## The view drops the superseded captures — 2026-09-05 (W4 step 5, D84)

`uv run dan-style status --data-dir ~/Source/livingthing/_training_pairs` over
the live `draws/` (14,586 draws, 407 real), before and after `view_policy()`
stopped passing `keep_superseded=True` and started passing
`superseded_captures` from `SUPERSEDED_CAPTURES`.

| | before | after |
|---|---|---|
| records rendering | 8,157 | **8,145** |
| excluded | `outcome 6,429` | `outcome 6,429, superseded 12` |
| duplicate `draw_id` among kept draws | 7 | **0** |
| `policy_sha` | `42c1ceefe1fd0e4c` → `22b9921a76d4e7f9` | **`4e349d3405787782`** |
| `content_sha256` | `d49ba7ad60774ea6` | **`e3d9809bddc9a616`** |
| splits eval / styler / detector | 151 / 4009 / 2709 | **148 / 4006 / 2703** |

The 12 are 11 draws the backfill stamped `superseded_by` plus the 9th interim
session (`d89d26a4` → `f608a742`), which the backfill predated; all real, so
the synthetic counts do not move and `test_corpus_stability`'s floors are
untouched. The 7 duplicates were backfill artefacts — it keyed a real draw on
`(source, blog_sha, span)`, so an interim save and its successor editing one
paragraph collided; the live capture path keys on `pairs.record_id`
(`capture_id:chunk_index`) and never did. The live `policy_sha` is
`dan-style status`'s to print (D52). Decided: D84.

## Three C13 draws leave the view by policy clause — 2026-09-05 (D84)

`uv run dan-style status` in `~/Source/livingthing`, before and after
`training_targets.EXCLUDED_DRAW_IDS` named the three accepted synthetic draws
whose target carries a backslash-escaped handle the generator un-escaped
(round-2 C13; the synth gate now refuses new ones). Owner decision, Dan.

| | before | after |
|---|---|---|
| records rendering | 8,145 | **8,142** |
| excluded | `outcome 6,429, superseded 12` | `excluded_draw_ids 3, outcome 6,429, superseded 12` |
| `policy_sha` / `content_sha256` | `d02d152e18e406f8` / `e3d9809bddc9a616` | **`5257549f211dda24` / `9c57ef6777e5a63e`** |
| `test_corpus_stability` floors | 2998/3005, 4108/4130 | **2997/3004, 4106/4128** |

All three were reproducible cells, so the floors fell by exactly the cells
excluded, the case the test's message names as a deliberate corpus edit. The
live shas are `dan-style status`'s to print (D52). Decided: D84's clause route.

## The live scores file's join keys — 2026-09-05 (W4 step 5 item 5)

`uv run python` over a scratch **copy** of `~/Source/livingthing/_training_pairs/scores.jsonl`
(270 lines, untracked — the blog's `.gitignore:91`) and the live
`_training_pairs/draws/*.jsonl` (14,579 distinct `draw_id`). Read-only, no spend.

| | rows | judge verdicts |
|---|---|---|
| whole-record rows in the file | 270 | 0 |
| whose `id` matches a `draw_id` | 187 | 0 |
| whose `id` does not (`<capture_id>:<chunk_index>`, e.g. `35f4ae46:6`) | 83 | 0 |
| the same file exploded to `(record_id, field, signal)` rows | 1,080 | 0 |

Keying on `draw_id` would strand the 83: real captures that the retired backfill
keyed on the span rather than the draw. Every row carries `record_id` and always
has, so `(record_id, field, signal)` joins all 270 and strands none. **The file
holds no judge verdicts at all** — nor do the three run-local score files under
`runs/docscale-2026090[34]/` (34 rows each, 0 judge), which corrects the premise
that the 83 carried paid verdicts: what the key protects is 1,080 vale and
detector rows. Decided: D39.

## The eval stratum is synthetic-free, and the clause keeps it so — 2026-09-05 (W4 step 5 item 6)

`uv run python` over the live `~/Source/livingthing/_training_pairs/draws/*.jsonl`
(14,579 draws, `draws_sha256` `6f8081ba2cc917fd`), grouped by
`splits.role_of(source)` against that corpus's `splits.json`. Read-only, no spend.

| role | real accepted | synthetic accepted | synthetic rejected |
|---|---|---|---|
| eval | 151 | 0 | 69 |
| styler | 77 | 4,677 | 3,901 |
| detector | 179 | 3,073 | 2,459 |

No synthetic draw is accepted on an eval post: the retired `prune-stale
--no-orphans` deleted them before the backfill, and nothing has run `synth` over
an eval post since. So `ViewPolicy.eval_sources` excludes **0 records today** —
the 69 are gate-drops the `outcome` clause already takes, and the clause is
ordered after it so they stay counted as rejects. Built against the blog's live
policy, `content_sha256` is `e3d9809bddc9a616` with the clause and without it,
8,145 records either way; `policy_sha` moves, as a new field must. The 14
`eval_posts` entries are 12 sources after `splits.canonical_source`. What the
clause buys is the next `synth` run over an eval post, which would otherwise
contaminate the held-out stratum silently. Decided: D84.

## Syntactic proxies by stratum — 2026-09-07 (D88, `argued-slop-arm.md` stage 0)

`uv run dan-style features --by slop_strategy --separability` in `~/Source/livingthing`,
over the view `pairs-5257549f211dda24` (8,142 records, 395 real). Slop side, per-sentence
rates pooled per stratum; AUC is `features.separability`'s out-of-fold logistic head on
the punctuation-free vector, folds grouped by post. Read-only, keyless.

| stratum | N | words/sent | sent-len var | commas | subordinators | em-dash | semicolon | connectives | concessions | AUC vs real |
|---|---|---|---|---|---|---|---|---|---|---|
| real | 395 | 23.4 | 207 | 1.38 | 0.26 | 0.37 | 0.14 | 0.021 | 0.026 | — |
| casual | 1451 | 17.2 | 99 | 0.92 | 0.16 | 0.14 | 0.04 | 0.005 | 0.041 | 0.776 |
| engaging | 1228 | 18.1 | 140 | 0.82 | 0.13 | 0.20 | 0.04 | 0.004 | 0.039 | 0.793 |
| linkedin | 790 | 12.9 | 84 | 0.51 | 0.08 | 0.10 | 0.04 | 0.005 | 0.017 | 0.920 |
| measured | 1595 | 20.5 | 116 | 1.21 | 0.25 | 0.12 | 0.08 | 0.015 | 0.090 | **0.730** |
| polish | 1538 | 18.6 | 123 | 1.06 | 0.19 | 0.14 | 0.07 | 0.005 | 0.065 | 0.750 |
| verbose | 1145 | 25.3 | 178 | 1.43 | 0.32 | 0.15 | 0.08 | 0.011 | 0.066 | **0.722** |

The 2026-09-03 table in the plan regenerates within rounding, which settles that it was
pooled rates, not medians (D88). `--same-sources` (531 synthetic records on the 54 real
posts) moves no arm by more than 0.04 on any rate. Sentence-length variance is the widest
gap that punctuation does not explain: real 207 against 84–178. The stage-1 bar is AUC
≤ 0.70 and below `measured`'s 0.730; `verbose` at 0.722 is the arm to beat on the
separability criterion, `measured` on the rates. Decided: D88.

## The `argued` audition — 2026-09-07 (`argued-slop-arm.md` stage 1, D88)

`direnv exec . uv run dan-style synth --data-dir _tmp/argued-audition[-v2] --slop-strategy
argued --session-turns 1 --sample-frac 0.05 --limit 120` in `~/Source/livingthing` (the same
120 chunks over 103 posts both times, five roster arms), then `dan-style features --pairs
<scratch view> --by slop_strategy --separability` and `dan-style eval --pairs <scratch view>`.
Slop side, pooled rates; the `real` and `measured` rows are the stage-0 table's. Spend: $0.07.

| arm | N | words/sent | sent-len var | commas | subordinators | connectives | concessions | AUC vs real | len ratio (median) | gate drops | det gap (slop − Dan) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 395 | 23.4 | 207 | 1.38 | 0.26 | 0.021 | 0.026 | — | — | — | — |
| measured (baseline) | 1595 | 20.5 | 116 | 1.21 | 0.25 | 0.015 | 0.090 | 0.730 | — | — | — |
| argued v1 | 110 | 36.6 | 489 | 2.11 | 1.04 | 0.188 | 0.292 | 0.962 | 1.50 | 33% (42/54 anchors) | 0.73 |
| argued v2 | 115 | 22.5 | 254 | 1.26 | 0.52 | 0.004 | 0.108 | 0.831 | 1.05 | 29% (27/47 identity) | 0.50 |

v1 is the cartoonish failure the plan names: "the alternative a reader would first reach
for" became a staged objector — 118 "one might" in 110 pairs, 7 openings of "While one
might" — and the 1.5× ceiling was the median. v2 forbids the objector, argues once a
paragraph and asks for mixed sentence lengths: length, words/sentence, commas and openings
land (no opening recurs more than twice), the detector gap halves toward Dan, and the
connectives go to near zero while concessions stay 4× real ("though" 38, "while" 34).
Subordination and concession are elicitable by prompt; the justificatory connective is
not, in either direction that keeps the rest sane — the arm produces *hedging* structure,
which real captures do not carry more of than synth, and not the scaffolding Dan strips
(the plan's "Risks"). Match fails on connectives and separability fails in both versions
(0.83 against the 0.70 bar and `measured`'s 0.73), which is the plan's stage-2 trigger.
Not decided: stage 2 is a new mechanism and Dan's call.

## The justificatory phrase, counted where it lives — 2026-09-07 (`argued-slop-arm.md` "Risks")

`uv run python` over the live view and the two scratch audition views, two regexes:
*self-ref* = "previous/earlier version|earlier draft|original text/version/draft|as stated
above…"; *justif* = "exactly/precisely the reason|this/that/which is why|the reason is/that|
precisely because|for exactly this reason". Per 1,000 sentences (`features.sentences`).

| stratum | records | sentences | self-ref | justif | records with any self-ref |
|---|---|---|---|---|---|
| real captures, slop side | 395 | 3,675 | 0.5 | 7.6 | 2 |
| real captures, Dan's side | 395 | 3,876 | 0.5 | 3.1 | 2 |
| `measured` | 1,595 | 15,788 | 0.5 | 0.9 | 8 |
| argued v1 | 110 | 783 | 0.0 | 37.0 | 0 |
| argued v2 | 115 | 899 | 0.0 | 3.3 | 0 |

The "exactly the reason" family is the arm's target and is bracketed, not hit: v1 at
5× real as a tic, v2 at half. Dan's edit halves it, never removes it, so the styler's
job on that structure is compression to ~3 per thousand, the number the risk watch
should hold. "A previous version of this text asserted" is nearly absent from the
captures themselves (2 of 395): it lives in Claude's chat-turn account of an edit, which
is `transcript-mining.md`'s sidecar and no synth call's output. Decided: nothing; cited
by the plan's "Risks" and the next-steps note of 2026-09-07.

## Separability on the style backbone — 2026-09-07 (D88, `--embed`)

`uv run dan-style features --pairs <v1 view> --pairs <v2 view> --by slop_strategy
--separability --embed` in `~/Source/livingthing`, and again with `--same-sources`
(synthetic restricted to the 54 posts the real captures sit on). The same post-grouped
logistic head as the proxy AUC, on `StyleDistance/styledistance` embeddings of the slop
side. Keyless, one local encoding pass over 8,367 texts.

| arm | AUC, proxies (all posts) | AUC, embed (all posts) | N same posts | AUC, embed (same posts) |
|---|---|---|---|---|
| measured | 0.730 | 0.999 | 118 | 0.997 |
| verbose | 0.722 | 0.997 | 85 | 0.993 |
| polish | 0.750 | 0.996 | 92 | 0.989 |
| casual | 0.776 | 0.996 | 92 | 0.983 |
| engaging | 0.793 | 0.998 | 83 | 0.996 |
| linkedin | 0.920 | 0.997 | 61 | 0.989 |
| argued v1 | 0.962 | 0.992 | — | — |
| argued v2 | 0.831 | 0.965 | 17 | 0.957 |

On the backbone every synthetic arm is near-perfectly separable from the real captures, and
restricting to the same posts moves nothing, so the gap is not topic. The syntactic proxies
understate it by a wide margin; they measure one dimension of a difference the embedding sees
whole. argued v2 is the least separable arm on both instruments, by a little. Two readings
stay open: the real captures were written by Claude inside an editing session and the synth
by five models through OpenRouter, so part of the signal may be model identity rather than
regime; and the head sees 768 dimensions from 395 positives, so the absolute number is an
upper bound. Either way the fifth signal's instrument is the embedding, with the proxies as
its readable explanation. Decided: D88 amended in spirit; a re-fit of `verbose`/`argued`
against this instrument is the next audition's bar.

## The first mined captures enter the corpus — 2026-09-07 (D89)

`uv run ai-style log mine … --gap-paras 2 --gap-chars 800 --apply` in `~/Source/livingthing`,
then `dan-style build-view`, `ai-style validate-pairs <view>` (0 problems, 0 duplicates) and
`dan-style status`. The dry run's window counts are in the blog's private sizing note.

| | before | after |
|---|---|---|
| records in the view | 8,142 | **8,170** |
| real pairs, eval / styler / detector | 151 / 77 / 179 | 151 / **85** / **187** |
| capture archives | 93 | **115** (22 mined windows) |
| `draws_sha256` / `content_sha256` | `6f8081ba2cc917fd` / `9c57ef6777e5a63e` | **`79837ac841b56a40` / `70757f0f2658bae9`** |
| `novel_after` of the mined pairs, median / q3 / p90 | — | 0.14 / 0.39 / 0.60 |

28 pairs where the dry run counted 31: three later windows' targets were already captured by
an earlier window in the same pass. None land on an eval post. The `transcript_turns.jsonl`
sidecar holds 22 lines and is gitignored. Decided: D89; the `novel_after` clause in the blog's
`styler_pair_ok` is the same commit series.

## Detector refit on the grown corpus — 2026-09-07

`uv run dan-style train-clf` in `~/Source/livingthing`, after the 28 mined captures landed;
keyless, StyleDistance backbone, the committed `splits.json`. Before is the 2026-09-03 head
(`c0e2ac85be`), read from the artifact's `meta.json` in git.

| | before (2026-09-03) | after (2026-09-07) |
|---|---|---|
| detector-pool pairs, real / synthetic | 179 / 2,162 | **187** / 2,162 |
| 8-fold CV pairwise acc / AUC, all | 0.974 / 0.967 | 0.972 / 0.973 |
| 8-fold CV pairwise acc / AUC, real | 0.784 / 0.693 | 0.722 / 0.683 |
| styler-holdout pairwise acc / AUC, real (272 pairs) | — | 0.777 / 0.753 |
| styler-holdout, synthetic (5,633) | — | 0.996 / 0.991 |
| shipped C | — | 10.0 (nested CV) |

The real-pair CV number moved down within its own spread (std 0.058 before), on eight more
real pairs; the holdout, the reward-relevant number, is the one to watch from here and has no
before. Synthetic separation is saturated either way, which is the embedding result seen from
the detector's side ("Separability on the style backbone"). Decided: nothing; the artifact is
committed with the corpus that produced it.

## The `argued2` probe — 2026-09-07 (D92, `argued-slop-arm.md` stage 2)

`brief_then_write` over 40 chunks of the v2 audition's 5% sample (36 posts), brief by
deepseek-v4-flash, write by claude-sonnet-4.6 and qwen3.6-35b, objections sampled from the
69 user turns in `transcript_turns.jsonl`; scratch data-dir, then `dan-style features --pairs
<v2 view> --pairs <probe view> --by slop_strategy --separability --embed`. Spend: $0.02.

| arm | N | words/sent | sent-len var | commas | subordinators | connectives | concessions | justif/1k sent | AUC proxies | AUC embed | drops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 423 | 23.5 | 210 | 1.38 | 0.27 | 0.022 | 0.027 | 7.6 | — | — | — |
| measured | 1595 | 20.5 | 116 | 1.21 | 0.25 | 0.015 | 0.090 | 0.9 | 0.732 | 0.998 | — |
| argued v2 | 115 | 22.5 | 254 | 1.26 | 0.52 | 0.004 | 0.108 | 3.3 | 0.837 | 0.962 | 29% |
| argued2 | 35 | 24.9 | 225 | 1.29 | 0.57 | **0.019** | 0.083 | 3.8 | 0.813 | 0.978 | 40% (qwen 52%, identity) |

The brief did what stage 2 predicted: argued2 is the first arm to land the connective rate
(86% of real) and the closest on sentence-length variance, so a write call with something to
argue from argues. It still fails the criteria that matter: the proxy AUC stays above
`measured`, and on the backbone it is no closer than v2 (0.978 against 0.962, N=35, so within
noise, but not the direction), which is D92's retirement condition. The eyeball file shows why:
the objection is narrated ("this objection fails because"), so the arm stages the collaborator
the way v1 staged a reader, and qwen returns the input on half its draws — the brief is beyond
it. Concessions run 3× real in every argued form; the justificatory phrase never passes half.
Decided: D92's verdict — the real stratum is the route; the composite stays library-only.

## A fresh cut reproduces 117 of 124 archived captures — 2026-09-10 (D96)

`capture.plan_pairs` over every `captures/<id>/` with draws in
`~/Source/livingthing/_training_pairs/draws/captures.jsonl` (228 capture ids; 124 archived, the
other 104 predate the archive), `whole` from `meta.json`, each stored draw matched on its
before-side text.

| gap policy | archives fully matched | stored chunks unmatched | after side differs | fresh chunks with no draw |
|---|---|---|---|---|
| 0 / 0 (`log save`'s default) | 94 | 35 | 6 | 146 |
| 2 / 800 (the retired `recut`'s) | 102 | 81 | 3 | 54 |
| better of the two, per archive | 117 | 7 archives | 5 | — |

No single policy re-cuts the stratum, so a build-time re-cut would change the view (D96's first
alternative). The 28 that match only at 2/800 are not the mined captures (3 of 22 mined are among
them). Two archives (`7d805943`, `f4702b94`) cut differently under both, and five (`0ceb874d`,
`250dae35`, `4821f388`, `aed11f2b`, `b761eb17`) match on the before side while `after.md` carries
prose the draw does not: edited after landing, under a manifest regenerated on 2026-09-04 that
pinned the drift. Decided: D96.

## Captured after-paragraphs against today's posts — 2026-09-10 (`capture-ux.md`)

Every live real record in the blog's view (`draws.live_draws` under
`training_targets.view_policy()`), its Dan-side chunk located in the post's
current text by best `difflib` ratio over a same-length paragraph window.

| state of the captured after-paragraph in today's post | pairs | share |
|---|---:|---:|
| unrecognisable, best match below 0.5 | 132 | 31% |
| rewritten, 0.5 to 0.85 | 86 | 20% |
| typo-scale change, 0.95 or above | 80 | 19% |
| identical | 64 | 15% |
| light edit, 0.85 to 0.95 | 60 | 14% |
| post deleted | 10 | 2% |

Half the stratum has moved beyond recognition or been rewritten since capture,
so an after side that followed the post would replace a style transform with the
post's history in half of all pairs, and the typo band cannot be told from a
small change of claim. Decided: the archive is the pair's moment and corrections
happen inside it; the moving-post proposal is withdrawn (`capture-ux.md`, no D).

## The Tinker roster, priced — 2026-08-25 (recovered 2026-09-10; D99)

Tinker's live model table read on 2026-08-25, prices per 1M tokens; the "run 4a" column is the
train price against run 4a's actual 3,518,325 train tokens (`runs/` manifest). LR is the
cookbook's `hyperparam_utils.get_lr` calibration where one exists.

| model | shape | train | run 4a | sample | calibrated LR |
|---|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-9B` (incumbent) | small dense, Hybrid, 64K | $1.463 | $5.15 | $1.995 | 4.730e-4 |
| `Qwen/Qwen3.6-35B-A3B` | medium MoE, Hybrid, 64K | $1.177 | $4.14 | $1.335 | 4.991e-4 |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` | medium MoE, Hybrid, 64K | $0.44* | $1.55* | $0.495* | none (`NotImplementedError`) |
| `openai/gpt-oss-20b` | small MoE, Reasoning, 32K | $0.396 | $1.39 | $0.45 | none |
| `Qwen/Qwen3.8-27B` | medium dense, Hybrid, 64K | $4.103 | $14.44 | $5.595 | 4.649e-4 |

\* a limited-time 50% discount; the durable Nemotron price is $0.88 / $3.10 / $0.99.

Training cost is not the constraint at this corpus size: the dearest row is $14 a run. Renderers:
`tinker-cookbook` 0.5.2 raised `KeyError` on Qwen3.8-27B and Nemotron; 0.5.5 resolves all four
(the floor was raised on main 2026-09-10). `Qwen3.6-27B`, the dense step-up D2 had named, retired
2026-09-02. Decided: D99.

## Protected blocks in the live real pairs — 2026-09-10 (D100)

Every live real record in the blog's view (`draws.live_draws` under
`training_targets.view_policy()`), its before side located in its archive's
`before.md` and read against `markdown.line_kinds(headings=False)`; the second
column under the blog's `div_is_prose`, the first under the segmenter's default.

| before side of the pair | default | blog div policy |
|---|---:|---:|
| no archive (landed before D96's archive) | 233 | 233 |
| not found in its archive (edited since landing) | 3 | 3 |
| prose | 167 | 167 |
| protected lines glued into a prose paragraph, no blank line | 9 | 13 |
| a wholly protected paragraph among prose | 14 | 12 |
| wholly inside a protected block | 6 | 4 |

The differ split on blank lines only (`lib.split_paragraphs`) and never asked
the segmenter, so 20 of 199 located pairs carry a code fence, display math, a
blockquote or a div; `0ceb874d:5` is bare Python with its fence markers outside
the chunk, invisible to any record-only clause. The two pairs the policies
disagree on are callout prose. Decided: D100.

## The affected archives under the fixed differ — 2026-09-10 (D100)

`capture.plan_pairs` over the 11 archives above, default knobs, the old differ
(`main` at `d3bb9e7`) against the fix; "trimmed" is a new pair that is a
substring of an old one, "gone" an old pair no new pair lies inside.

| archive | old | new | same | trimmed | gone | of which wholly protected |
|---|---:|---:|---:|---:|---:|---:|
| `0ceb874d` | 16 | 16 | 11 | 5 | 2 | 2 |
| `13d37eaf` | 3 | 2 | 2 | 0 | 1 | 1 |
| `4afc6010` | 6 | 6 | 6 | 0 | 0 | 0 |
| `603a9fef` | 1 | 0 | 0 | 0 | 1 | 1 |
| `a48d2c5d` | 1 | 1 | 0 | 1 | 0 | 0 |
| `abba8682` | 7 | 7 | 6 | 1 | 0 | 0 |
| `aed11f2b` | 6 | 6 | 1 | 5 | 2 | 2 |
| `c48abc7b` | 2 | 1 | 1 | 0 | 1 | 0 |
| `ca32f59c` | 17 | 18 | 16 | 2 | 0 | 0 |
| `e4721c01` | 12 | 12 | 12 | 0 | 0 | 0 |
| `f33f75d5` | 1 | 0 | 0 | 0 | 1 | 1 |
| total | 72 | 69 | 55 | 14 | 8 | 7 |

Every new pair is a substring of its `before.md`. The one gone pair that was
not wholly protected (`c48abc7b`) had a blockquote Dan un-quoted and rewrote
around: the interior-drop rule. `603a9fef` and `f33f75d5` are a callout and a
blockquote edit with nothing else changed, so a recut of them lands nothing and
their live pairs retire by `excluded_draw_ids`. Decided: D100.

## The styler selector on the real stratum — 2026-09-10

`livingthing.training_targets.styler_pair_ok` over every live real record in the blog's view
(`draws.live_draws` under `view_policy()`), each drop attributed to its clause in the selector's
order; "hand" is a `log save`, "mined" a D89 capture (`meta.mined`).

| clause | pairs | hand | mined | after chars |
|---|---:|---:|---:|---:|
| `novel_after` > 0.6 | 4 | 2 | 2 | 9,739 |
| identity, char similarity ≥ 0.95 | 30 | 20 | 10 | 34,466 |
| anchor lost | 38 | 35 | 3 | 42,365 |
| anchor invented | 41 | 37 | 4 | 86,689 |
| anchor lost and invented | 11 | 9 | 2 | 20,519 |
| **dropped, any clause** | **121** | 102 | 19 | **181,573** |
| live real, total | 432 | 404 | 28 | 587,604 |

The selector drops 28% of the real pairs and 31% of their characters; the symmetric anchor rule
alone takes 90 pairs, 21% of the stratum, against the 15% its docstring measured on 2026-08-17.
What Dan "invents" is mostly his own notebook: 51 of 81 added anchors are `./page.qmd` wiki-links
and 19 are citations, cross-referencing while he edits; the lost-and-invented rows include a
spelling normalisation of a link target, which the rule reads as one anchor lost and another
invented. Whether 21% of the scarce side is worth its purity is the question
`styler_pair_ok_keep_anchor_damage` exists to answer and has not been run; under the D96 premise
it is the next ablation. Decided: nothing yet; input to the next retrain.

