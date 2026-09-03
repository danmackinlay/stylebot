# Findings — measured results

Empirical results from the stylebot pipeline, collected here as the per-phase
design plans were retired. **This file exists because these numbers are not
re-derivable from git.** Reproducing any of them costs real money and
wall-clock; the code that produced them records the method, never the result.

Each entry: date, what was measured, sample size, the number.

---

## Threading audit — 2026-09-03: the eval pool bought nothing; batching bought 1.7x

Every threading construct in stylebot and the blog, audited against D56's rule.
All local, no spend: the benchmark corpus is the 34 already-paid-for rewritten
documents in `_training_pairs/runs/docscale-20260903/documents.jsonl` and the
committed live pairs corpus. Apple silicon, torch 2.13, sentence-transformers
5.7, `StyleDistance/styledistance`, `torch.get_num_threads() == 6`. Every timing
is an interleaved repeat (`8, 1, 8, 1`), and the spread between repeats is
reported rather than averaged away — an effect that does not clear its own noise
at this size is a null result, not a reason for a bigger sample.

**1. The detector in the thread pool.** `score_pairs_file` over the 34
documents, chunk-wise (1,690 serving-scale chunks, 4500/inline-headings), no
judge, no Vale — the detector is the whole cost:

| | `max_workers=8` | `max_workers=1` |
|---|---|---|
| before (detector inside the pool) | 24.25 s, 24.71 s | 23.96 s, 23.69 s |
| after (batched pre-pass) | 12.58 s, 14.85 s | 13.84 s, 15.53 s |

Eight threads were **not faster than one**, before or after — indistinguishable
within a ±10% repeat spread, and marginally *slower* in three of four pairings.
Embedding is CPU-bound, so the GIL never went anywhere, and since D55 the lock
around `model.encode` made that explicit. The pool over the detector was pure
cost for its entire life.

**2. Batching, the alternative.** Bare encode over the same 1,690 chunks, no
harness: **22.22 s** as batches of one (13.2 ms/chunk) against **14.24 s** in
one `encode(batch_size=32)` call (8.4 ms/chunk) — **1.56x**. Larger batches do
not help (`batch_size=64`: 14.59 s), so 32 is already past the knee.

End to end through `score_pairs_file` the win is a little larger, because the
pre-pass also removes 1,690 Python-level detector calls: 24.25 s -> 12.58 s at
the same eight workers, **~1.7x**.

**3. The ordinary corpus path**, which is the common case (`dan-style eval`, no
`--documents`): passage-scale detector, 512 live pairs x {slop, target} = 1,024
embeds, no judge, no Vale:

| | `max_workers=8` | `max_workers=1` |
|---|---|---|
| per-text (before) | 15.65 s, 12.90 s | 12.85 s, 12.58 s |
| batched (after) | 6.93 s, 8.00 s | 7.35 s, 8.33 s |

**~1.7x** again, and the same verdict on the pool: at 512 records eight workers
never beat one. (A full-corpus run — 8,148 records, collected before the sample
cap was set and not repeated — put it at 223.3 s against 158.8 s, the same 1.4x
once the fixed ~9 s model load is inside the number. It adds nothing the 512
sample did not already resolve.)

**4. The judge, which is what the pool is actually for.** The same 34 records x
3 fields = 102 judge calls, latency simulated at 1 s per call so no money is
spent — the production `score_pairs_file` path, only the judge faked:

| `max_workers` | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| wall clock | 103.33 s | 28.08 s | 16.08 s | 10.03 s |

**6.4x at the default eight**, near-linear. Network wait is the one thing this
pool overlaps that is worth overlapping, and the same number is the reasoning
for the blog's `library_archive` download pool, which cannot be measured
without a keyed external service.

**5. Batching is not bit-identical.** Re-scoring the 34 documents through the
batched path reproduces the committed `scores-chunkwise.jsonl` to a maximum
absolute difference of **1.86e-07** in `detector.score` (204 of 204 float fields
differ, all at that scale). A transformer pads to its batch and float32 addition
is not associative. Four orders of magnitude below the target−slop delta of
0.022 this instrument is used to measure, and below its reporting precision —
but a re-score of a pre-2026-09-03 file will not diff clean.

Decisions: D56 (the rule), D57 (what changed).

---

## Document-scale eval, chunk-wise — run 2026-09-03 (later the same day): it separates

The fix for the entry below. Same 34 rewritten documents, same detector
(`c0e2ac85be`), same command — **only the scoring changed**: each field is now
segmented the way serving segments it (`infer.prose_chunks`, the promoted diet:
4500 chars, cross-heading), every prose chunk is scored on its own, and the
document's number is the mean over its chunks (`eval.chunked_detector`). No new
sampling: the rewrite is resumable by record id, so re-scoring is free.

| | whole-document embedding | **chunk-wise mean** |
|---|---|---|
| mean P(slop), slop (the AI draft) | 0.4880 | **0.3844** |
| mean P(slop), target (Dan's edit) | 0.4951 | **0.3622** |
| mean P(slop), output (the styler) | 0.4914 | **0.3658** |
| min / max, slop | 0.1107 / 0.7921 | 0.2627 / 0.5591 |
| min / max, target | 0.1107 / 0.7921 | 0.1631 / 0.4968 |
| min / max, output | 0.1107 / 0.7921 | 0.2323 / 0.4969 |
| documents where Dan < draft | 2 / 34 = 0.059 | **25 / 34 = 0.735** |
| documents where styler < draft | 0 / 34 = 0.000 | **25 / 34 = 0.735** |

Paired deltas, bootstrap CI **clustered by post** (34 documents over only 7
posts, so the cluster count is the honest n):

| | whole-document | **chunk-wise** |
|---|---|---|
| target − slop | +0.0071 (−0.0017, +0.0291) | **−0.0222 (−0.0452, −0.0139)** |
| output − slop | +0.0034 (+0.0000, +0.0148) | **−0.0186 (−0.0234, −0.0118)** |
| output − target | −0.0037 (−0.0172, +0.0022) | +0.0036 (−0.0055, +0.0288) |

**Three things the numbers say.**

1. **The direction is now right and the interval excludes zero.** Dan's edit
   scores below the AI draft, and the styler's output does too. Whole-document,
   both signs were *positive* — the styler's output scored closer to slop than
   the draft in 34 of 34 documents. The identical min/max across all three
   fields is gone, which was the tell that the embedding was reading the
   document rather than the writing.
2. **0.735 is the document-scale pairwise, against the head's own chunk-scale
   real-facet 0.784.** Close enough to say the instrument is being used inside
   its range, which is exactly the claim the entry below could not make.
3. **The styler cannot be told apart from Dan's own edit at this scale**
   (output − target CI spans zero, and only 7 clusters). That is not a win
   claim; it is the resolution this measurement has. It is also the first
   document-scale reading of the promoted diet that means anything.

Chunk counts: 563 / 564 / 563 for slop / target / output. The near-equality is a
coincidence of these documents, not an alignment — **the chunks are deliberately
not paired across fields.** An edit is not a rewrite: Dan moves paragraph
boundaries, so field *i*'s chunk *k* is often not field *j*'s chunk *k*. Each
field is segmented and aggregated on its own and the comparison is between two
unaligned means. A chunk-index join would be false precision.

Each score also carries `n_chunks` and `sd`, the within-document spread (mean sd
over the slop side: **0.2642** — three times the between-document sd of 0.0812).
That number is itself worth having: documents are far more varied inside
themselves than they are from each other, which is the mechanical reason a
single embedding of one told you nothing.

**Caveat that survives.** StyleDistance truncates at its max sequence length, so
a 4500-char chunk is read to roughly its first half. The same truncation applied
when the head was fit on the corpus's 4500-budget cells, so this is consistent,
not a new bias — but a diet wider than the backbone's window is only ever scored
on its head.

### The segfault was not the libomp double-load — and isolating it is how we know

`--documents` was expected to be the `index-blog` failure: two OpenMP runtimes
(Tinker SDK + sentence-transformers) in one process. It is not. Moving the
scoring pass into its own child — no sampling SDK anywhere near it — the child
**still crashed**, deterministically, exit 139/138, before writing a record.

The cause is **concurrent forward passes through one `SentenceTransformer`**.
`score_pairs_file` fans the detector out over eight worker threads; at
whole-document scale that was three embeds per record and survived, and
chunk-wise it is ~17 per field and does not. `--max-workers 1` completed
cleanly, which located it. The fix is a lock around `model.encode` in
`classify.sentence_transformers_batch_embed_fn` — the one encode call-site, so
it also covers best-of-N reranking and the serve sidecar. With the lock, a
single process holding **both** the Tinker SDK and the embedder scored the
sample fine (measured), which is the direct refutation of the double-load story.

The subprocess isolation stays, for the reason it earned rather than the one it
was built for: **it is what made the crash legible**. In-process it was exit 139
into a half-written file with nothing said, laundered to 0 by a `| tail`; from a
child it is a `RunError` naming the status. Same posture the
`index-blog-pipeline` skill takes about its own guards — the guarantee is
loudness; the double-load defence is defence-in-depth against a mechanism this
repo has still never reproduced.

**Live run, unpiped: `EXIT=0`.** 34 documents scored, 0 errors, 0 rewritten (the
resume skipped all 34, so no sampling spend). `43f0e688` still fails to rewrite
with `AttributeError: 'list' object has no attribute 'strip'` — unchanged, still
undiagnosed, still caught by the per-document guard.

---

## Document-scale eval — run 2026-09-03 (W10 item 7): the instrument has no resolution there

The first time `rewrite_text` has been run over whole eval posts rather than
pre-cut chunks. 34 of 35 eval-role capture archives rewritten by run 4a
(`20260817-170037`) at its own diet — 563 chunks, 7 kept verbatim — then scored
with the refit detector (`c0e2ac85be`). ~$0.27.

**The verdict is that there is no verdict, and that is the result.**

| field | n | mean P(slop) | sd | min | max |
|---|---:|---:|---:|---:|---:|
| slop (the AI draft) | 34 | 0.4880 | 0.2260 | 0.1107 | 0.7921 |
| target (Dan's edit) | 34 | 0.4951 | 0.2224 | 0.1107 | 0.7921 |
| output (the styler) | 34 | 0.4914 | 0.2229 | 0.1107 | 0.7921 |

**The min and max are byte-identical across all three fields**, the means differ
by 0.007, and the styler's output scored *closer to slop* than the draft did in
34 of 34 documents — by 0.0034, which is noise. The detector is responding to
the document, not to who wrote it.

The mechanism is not mysterious. A capture's three variants are the *same
document* differing in a few percent of their characters — an edit, not a
rewrite — and the styler changes a comparable fraction. The detector is a
StyleDistance embedding fit on chunks of ~1.2k chars; over 25k chars the shared
90-odd percent dominates the embedding and the differences wash out. Its
chunk-scale real-facet pairwise is 0.784; here it is indistinguishable from a
coin.

**Consequences.**
1. **A document-scale verdict must score chunk-wise and aggregate**, not embed
   the whole document. The rewrite half of `rewrite_documents_file` is right and
   worth keeping — it is the only thing that exercises a diet switch end to end
   — but the scoring half needs to segment the before/after/output the same way
   serving does and pair the chunks up. Until then `--documents` produces output
   worth reading by eye and a number worth ignoring.
2. **This does not invalidate the chunk-level verdicts.** It bounds where the
   detector may be pointed, which nothing had established: it was built,
   validated and always used at chunk scale, and its first use outside that
   scale silently returned 0.49 for everything.
3. It is a third instance of the day's pattern — Pangram's dead URL, the
   `--documents` segfault below, and this — where the first *live* run of a
   built-and-tested path found something no stub could.

### The run also segfaults, and a pipe hid it

`dan-style eval --documents` exits **139 (SIGSEGV)** after the rewrite completes
and before scoring, reproducibly. Both the Tinker SDK (for sampling) and
sentence-transformers/torch (for the detector) load OpenMP into one process —
the libomp double-load the blog's `index-blog-pipeline` skill documents, which
is why `index-blog` already runs its `map` step in a torch-free subprocess.

Two things made it invisible. The rewrite writes `documents.jsonl` and flushes
per record, so its output looked complete. And the run was piped through `tail`,
so the shell reported **tail's** exit status, not the pipeline's — `$?` was 0.
The scores file was created and left empty, and nothing said why.

Scoring the same file in a torch-only process wrote all 34 records with 0
errors, which is how the table above exists. **The fix is the same as
index-blog's**: run the sampling pass and the scoring pass in separate
processes. Do not add `pipefail`-dependent reasoning to the RUNBOOK and call it
fixed — the segfault is real whether or not the exit code is visible.

### One document still fails to rewrite

`43f0e688`: `AttributeError: 'list' object has no attribute 'strip'` inside
`rewrite_text`. Uncaught by any test, caught by `rewrite_documents_file`'s
per-document guard, so 34 of 35 completed. Not diagnosed.


## Pangram cross-check — run 2026-09-03, PARTIAL (W10 item 6)

The one-shot independent-detector check `eval-harness.md` has described since
the first draft. Run against the eval_v2 stratum. **Two things went wrong before
a number came out, and both are the finding.**

**The shipped endpoint was dead.** `PANGRAM_URL` was
`https://text.api.pangram.com/predict`, which answers **404**: Pangram 3.0
replaced the synchronous endpoint with an async task API (`POST /task` -> id ->
`GET /task/{id}` polled to `STAGE_SUCCESS`). Nothing had caught it because the
detector had only ever been exercised through its `post=` transport seam,
returning an already-scored body. A stub that returns the shape you expect
cannot tell you the URL is wrong. The fix added a `fetch=` seam for the poll
loop and two tests over it, but the honest lesson is that "built and tested,
not run" was worth less than it sounded.

**The account ran out of credit mid-run**: HTTP **402 Payment Required** after
92 of 149 records. The numbers below are that 92-record sample, 11 of the 12
eval posts. Re-running is a top-up away and is resumable — scores are id-keyed,
so it costs only the missing 57.

| | |
|---|---|
| complete pairs (both sides scored) | 92, across 11 posts |
| mean P(AI), slop side | 0.8735 |
| mean P(AI), Dan's side | **0.6300** |
| mean delta | **+0.2434** |
| pairwise, all pairs | 36 / 92 = 0.391 |
| pairwise, **excluding ties** | **36 / 41 = 0.878** |
| wins / losses / **ties** | 36 / 5 / **51** |

**The direction holds.** When Pangram distinguishes the two sides at all it
agrees with the voice classifier 88% of the time, against the voice-clf's own
real-facet 0.784. That is the question the check was for, and the answer is yes.

**But it is a blunt instrument on this corpus, and that is the more useful
result.** 153 of 184 answers (83%) sit exactly on a rail — 121 at 1.0, 32 at
0.0 — which is why 51 of 92 pairs tie. Most starkly: **47 of the 92 chunks of
Dan's own human-written prose scored exactly 1.0**, "certainly AI". Read that
with the caveats it deserves — these are ~200-word chunks of technical prose
with citations and code-adjacent register, the hardest case for a general
detector, and a general detector has no notion of a particular human's voice.
It is nonetheless a 51% false-positive rate on the ground truth we care about.

**Verdict: this stays a one-shot instrument check and does not become a
signal**, which is what `eval-harness.md` already concluded on cost grounds and
what D-series decisions assume. It cannot serve as the orthogonal channel for
best-of-N or a reward: at an 83% saturation rate it has no gradient to give, and
it would veto half of Dan's real prose. The judge and the eyeball remain the
orthogonality. What it does buy, once, is the confirmation above: our detector's
ranking is not an artefact of its own embedding.


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
Continued: 8,319 (2026-08-17, run 4's input, after the v2 scale batches) →
8,265 → 8,244 (2026-09-03, after W8's 21 duplicate cells and 5 pending capture
records) → **8,182, 398 real / 7,784 synthetic over 497 posts** (2026-09-03,
after the 62-pair eval-contamination prune). *This series stops here on
purpose.* It is a record of what was measured on each date, not a running
total — for the current figure run `dan-style status`.

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

## Register — the labels are not free (2026-08-17)

Prompted by "my style has many registers and the model is struggling to hit a
median register... I don't really know how to control for that variable".

**A single SFT LoRA on a multimodal target learns the conditional mean**, so a
median register is the expected outcome rather than a defect of any run. The
cheap fix would be conditioning on a register label, the way heading context is
already conditioned on (D14) — if labels could be had for free.

They cannot. K-means over StyleDistance embeddings of **2,500 distinct Dan-side
chunks** (≥400 chars): silhouette **0.232** at k=2, 0.231 at k=3, **0.117** at
k=4, 0.090 at k=6. And reading the cluster exemplars, what separates is
**topic, not register** — LaTeX/tooling in one, political economy and academic
prose in another, link-list pages in a third. Subject matter dominates the
variance at this granularity, so unsupervised labels would condition the styler
on what a passage is *about*.

**This does not mean no register axis exists.** The same backbone supports a
*supervised* Dan-vs-slop direction at 0.73 real pairwise accuracy despite topic
dominating the same variance — supervision picks the direction, k-means picks
the biggest one. So the honest next step is a few hundred hand-labelled chunks
("brisk technical" / "whimsical" / "academic"), a logistic head over them, and
either a conditioning tag or a steering vector with λ as the slider. What is
ruled out is getting the labels for nothing.

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

## Eval-stratum composition — re-measured 2026-09-03 (W10)

The frozen eval stratum, against the live 8,244-pair corpus:

| view | posts | pairs | real | synthetic |
|---|---:|---:|---:|---:|
| the corpus (`iter_pairs`, every record) | 6 | **103** | **41** | 62 |
| the detector dataset (after the `transform_sim > 0.85` gate) | 6 | **89** | **41** | 48 |

Both numbers are right and they answer different questions: `check-splits`
prints the second because it reports through `classify_train.assemble_dataset`.
14 synthetic eval pairs are near-identity and never reach the classifier.

**The stratum is one post wearing five hats.** Per pinned post, alias-resolved:

| post | pairs | real | `automation` |
|---|---:|---:|---:|
| `notebook/automatic_coding.qmd` | **77 (75%)** | 15 | 1 |
| `notebook/adhd.qmd` | 10 | 10 | 2 |
| `notebook/empowerment.qmd` (now `informational_empowerment`) | 6 | 6 | 2 |
| `notebook/learning_from_madness_of_crowds.qmd` (now `wisdom_from_…`) | 4 | 4 | 0 |
| `notebook/ai_economics_knowledge.qmd` | 4 | 4 | 2 |
| `notebook/multi_agent_reinforcement_learning.qmd` | 2 | 2 | 1 |

Run 4's verdict called this "127 chunks, paired" and gave it a chunk-level
interval. It is 6 clusters (D40). `MIN_POSTS_PER_ROLE = 5` passes at 6 — the
plan's "passes silently at 4" was counting spellings before W0's alias map
folded the two renamed posts back onto their identities.

**Real captures live on AI-touched posts, by construction.** Pairs by source-post
`automation`, alias-resolved, 2026-09-03:

| role | automation 0 | 1 | 2 | of which REAL: 0 / 1 / 2 |
|---|---:|---:|---:|---|
| eval | 4 | 79 | 20 | 4 / 17 / 20 |
| detector | 2,596 | 517 | 139 | 11 / 30 / 138 |
| styler | 4,112 | 569 | 208 | 11 / 33 / 134 |

**168 of the detector pool's 179 real pairs (94%) sit on `automation: 1/2`
posts** — the post is marked AI-touched *because* Dan captured himself
de-slopping it. Capping detector targets at `automation: 0` outright, as the QA
list proposed, would leave the honest `real` facet with 11 pairs. Capping only
the **synthetic** ones drops 488 of 3,252 detector pairs (15%) and costs no real
pair at all. That is what shipped (D41).

**What an eval re-cut would buy, and take** (D42). 52 posts carry ≥1 real pair;
6 are eval, 16 detector, 30 styler.

| eval_v2 | posts | pairs | real | top-post pair share | detector real left | styler real left |
|---|---:|---:|---:|---:|---:|---:|
| v1 (today) | 6 | 103 | 41 | 75% | 179 | 178 |
| + top 4 styler-role | 10 | 190 | 128 | 41% | 179 | 91 |
| **+ top 6 styler-role (recommended)** | **12** | **245** | **149** | **31%** | **179** | **70** |
| + all 30 styler-role | 36 | 544 | 219 | 14% | 179 | 0 |
| + all 46 real-pair posts (maximal) | 52 | 1,025 | 398 | 8% | **0** | **0** |

The maximal cut is the one the QA list describes and it zeroes both other roles'
real strata. The sized row is the one that shipped — see below.

## eval_v2 — applied 2026-09-03 (W10 item 4 / D42)

The sized extension above, applied with `dan-style extend-eval --apply` against
the post-prune corpus (8,182 records; the 62 synthetic eval pairs had gone in
blog `c0e2ac85be`). The re-measured numbers differ from the table above because
that table was computed before the prune: the stratum was already at 41 real /
0 synthetic and 37% concentration, not 75%.

| stratum, as `check-splits` reports it | posts | pairs | real | synthetic | top-post share |
|---|---:|---:|---:|---:|---:|
| eval v1 (`splits_sha256=c8c319578707bb5b`) | 6 | 41 | 41 | 0 | 37% (`automatic_coding`) |
| **eval v2 (`splits_sha256=05b495adac68cdc4`)** | **12** | **177** | **149** | **28** | **22%** (`ai_search`) |
| styler | 290 → 284 | 4,138 → 4,002 | 178 → **70** | 3,960 → 3,932 | 3% |
| detector | 198 → 198 | 2,709 → 2,709 | 179 → **179** | 2,530 → 2,530 | 2% |

**The whole cost falls on the styler's real pairs, by construction**: all six
added posts were styler-role, so the detector pool does not move by a single
pair. The styler keeps 70 real pairs out of 4,002 — real prose was 4.3% of its
pairs and is now 1.7%, which is a real thinning but not the failure mode D42
refused (the maximal cut, which takes the detector's real facet to zero).

**Two things the extension brings back that the W12 prune had just removed.**
`notebook/ai_search.qmd` carries 28 synthetic pairs, so the frozen stratum is no
longer 0% synthetic; it is 16%, well under the `EVAL PURITY` threshold and
reported separately by `headline_facet: "real"`, but the clean-stratum invariant
W12 established now needs one more prune pass to restore. And document-scale
eval goes from **3 documents / 86,820 chars to 35 documents / 1,561,740 chars**
(`iter_document_pairs(role="eval")`) — 18× the prose, so the ≈$0.015 sizing in
`eval-harness.md` becomes ≈$0.27.

**The detector was refit against the new partition** (`dan-style train-clf`,
local, ~4 min). Identity and honest numbers, old vs new:

| | old | new |
|---|---|---|
| `git_sha` | `da335af66d` | `c0e2ac85be` |
| `trained_at` | 2026-08-17T18:41:14Z | 2026-09-03T08:37:11Z |
| `n_pairs` (real / synthetic) | 2,751 (215 / 2,536) | 2,341 (179 / 2,162) |
| `n_posts` | 198 | 183 |
| `splits_sha256` / `split.eval_version` | absent / absent | `05b495adac68cdc4` / 2 |
| `pair_ok` | absent | `detector_pair_ok(<=0)` |
| CV pairwise / AUC | 0.9710 / 0.9652 | 0.9739 / 0.9666 |
| **real facet** pairwise / AUC | **0.7337** / 0.6580 | **0.7840** / 0.6927 |
| styler-holdout real pairwise (n) | 0.8182 (176) | 0.8000 (70) |

The pool shrank for reasons that are *not* eval_v2 — the W12 prune and D41's
`automation` gate on synthetic detector targets — since eval_v2 took nothing
from the detector. The real facet rising 0.734 → 0.784 on 36 fewer real pairs is
within the ±0.05 fold spread and should not be read as an improvement.

**What the verdict did.** The instrument check that needs no spend: score both
sides of the frozen stratum with the new head and take the paired
slop − Dan `P(slop)` difference (`paired_delta`), real facet, clustered by post.

| stratum | n | clusters | mean Δ | cluster CI | width | chunk-level width |
|---|---:|---:|---:|---|---:|---:|
| v1 | 41 | 6 | 0.1724 | [0.1232, 0.1987] | 0.0755 | 0.1367 |
| v2 | 149 | 12 | 0.1560 | [0.1013, 0.2007] | 0.0994 | 0.0806 |

The **chunk-level** interval narrows as n grows (0.137 → 0.081, near the √3.6
the extra pairs buy). The **cluster** interval *widens* (0.076 → 0.099), and
that is the point rather than a regression: at 6 clusters the percentile
bootstrap resamples six post-means that happened to agree, so it reported an
interval *narrower* than the chunk-level one — an inversion that only ever
means "too few clusters to say". At 12 clusters the ordering is the expected
one (clustering widens), and the six new posts show genuinely lower separation,
which also pulls the point estimate down 0.172 → 0.156. v1's number was mildly
optimistic and its interval was not measuring what it claimed.

**And the instrument moves more than the effects it measures.** The same 41 v1
pairs read Δ = **0.1466** under the old head and **0.1724** under the new one —
a 0.026 shift from a refit alone, against run 4a's headline movement of 0.038.
`compare_scores` refuses the pair outright (D39), which is the correct answer.

## What a stale-corpus prune would cost and buy — measured 2026-09-03 (W12)

`ai-style data prune-stale`, dry run, against the live 8,244-record corpus
(398 real / 7,846 synthetic / 497 posts) re-cut under
`livingthing.training_targets.chunk_policy()` — the one composed policy W2 left
behind. **Nothing was pruned.** These are the numbers the decision needs.

**Orphans: 603 of 7,222 distinct synthetic cells (8.3%), carrying 672 pairs
(8.6% of the synthetic corpus) across 124 posts.** By diet: section-bounded
@1500 **349**, cross-heading @1500 **172**, @3000 **81**, @4500 **70**. The
heaviest posts are `notebook/latex.qmd` (52), `quarto_websites.qmd` (46),
`note_taking.qmd` (44), `ai_search.qmd` (33), `optionality_as_end.qmd` (26).

**Zero real captures, and that is structural rather than lucky.** A capture's
target comes from a paragraph diff of an edit archive, never from
`iter_targets`, so asking whether the chunker reproduces it is a category error.
The measurement reports the 398 real pairs as UNCHECKED, not as clean. Zero
orphans from a missing source, too, so no post has been renamed out from under
its pairs since W0's alias map landed.

**Why the cells stopped reproducing.** Re-cut at each stability arm's pinned
revision, so the prose is a frozen git blob and only policy can move the number
(cells, not pairs; the two arms overlap by 3 cells, hence 606 against 603):

| cause | section-bounded | cross-heading | cells |
|---|---:|---:|---:|
| W2's fence rule (unclosed fences, the four-backtick closer) | 0 | 0 | **0** |
| `COMMENT_MARKER` in `ignore_markers` | 3 | 24 | 27 |
| `div_is_prose` applied to cells cut before it existed | 70 | 0 | 70 |
| **prose Dan has edited since generation** | 227 | 265 | **492** |
| irreducible residue (synthesis ran against an uncommitted tree) | 5 | 12 | 17 |
| total against the working tree | 305 | 301 | 606 |

**So W12's premise is wrong about its own cause.** The item was written expecting
W2's segmentation change to have orphaned cells; it orphaned **none**. The two
files whose protected spans W2 moved (`notebook/quarto.qmd`,
`notebook/learning_with_ai.qmd`) carry **zero corpus cells** between them. Four
fifths of the orphans (492 of 606) are Dan having edited the post since 2026-07
or 2026-08 — the corpus being a frozen record of prose that moved, which is not
damage and not a reason to delete a pair. The 97 that a *policy* change caused
are dominated by `div_is_prose` (70) charged against the **frozen**
section-bounded arm: `CROSS_HEADING` is live policy, so no synth run will ever
refill a section-bounded cell, and pruning those 349 pairs is permanent loss.

**Eval contamination: 62 synthetic pairs, all from
`notebook/automatic_coding.qmd`**, 41 real captures spared. This reconciles
exactly with W10's corpus-view row (103 pairs / 41 real / 62 synthetic); W10's
other figure, 48, is the same stratum after the detector's `transform_sim > 0.85`
gate, which 14 near-identity synthetic pairs do not survive. By generator:
sonnet-4.6 **18**, gemma-4-26b **15**, qwen3.5-9b **10**, deepseek-v4-flash
**7**, qwen3.6-35b **5**, llama-3.3-70b **4**, nemotron-3-nano **3**. **25 pairs
are in both sets**, so the two prunes together drop 709, not 734.

| corpus | total | real | synthetic | posts |
|---|---:|---:|---:|---:|
| now | 8,244 | 398 | 7,846 | 497 |
| minus orphans | 7,572 | 398 | 7,174 | 491 |
| minus eval synthetic | 8,182 | 398 | 7,784 | 497 |
| minus both | 7,535 | 398 | 7,137 | 491 |

**And it does not touch the dilution W9 found, which was the reason to want it.**
Styler-role pairs under `styler_pair_ok` with W9's containment grouping (no val
split, so these are larger than W9's 3,981 train pairs by the held-out posts):

| corpus | pairs | passages | mean pairs/passage | real % of pairs | real % of passages |
|---|---:|---:|---:|---:|---:|
| now | 4,373 | 1,077 | 4.06 | 3.16% | 12.53% |
| minus orphans | 4,016 | 993 | 4.04 | 3.44% | 13.60% |
| minus eval synthetic | 4,373 | 1,077 | 4.06 | 3.16% | 12.53% |
| minus both | 4,016 | 993 | 4.04 | 3.44% | 13.60% |

The orphan prune removes 357 styler pairs and 84 whole passages and moves the
mean effective weight **4.06 → 4.04**. It cannot do better: orphans are spread
across 124 posts roughly in proportion to the corpus, so it thins passages and
pairs together. Dilution is what `per_passage` and `pair_weight` are for (D43),
and a prune is not a substitute for either. The eval prune is invisible here by
construction — eval-role pairs never enter the styler set; it protects the
*verdict*, not the training diet.

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
score, and run 2 served from 2026-07-22 until run 4a was promoted on 2026-08-21.
(What serves *now* is not recorded in this file, or in any other: ask
`dan-style status`, or read the newest promoted manifest in the blog's
`_training_pairs/runs/`.)

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

**Against the serving default, though, run 4 is a real gain.** Run 2 sampled
over the same 127 chunks with the same detector: output **0.2501**, closing
81.4% of the gap, against 4a's 0.2125 / 90.7%. Paired **4a − run 2 = −0.0376,
95% CI [−0.0616, −0.0136]** — excludes zero. 4b − run 2 is −0.0280,
CI [−0.0576, +0.0016], which does not. Vale −0.087 [−0.194, +0.021] and anchor
damage 7/127 vs run 2's 6/127 (4a loses more, invents fewer: 6+2 against 4+5),
median output/input length 0.99 vs 1.00 — so the gain is not bought with
deletion, which is what killed run 1.

### Run 4's verdict, re-analysed 2026-09-03 (W10) — the gain over run 2 does not survive

Same score files (`_tmp/eval-run4/scores-full-*.jsonl`, 127 chunks from the
2026-08-17 corpus: 65 real / 62 synthetic, 85 of the 127 from
`notebook/automatic_coding.qmd`), re-read through `eval.compare_scores` with the
POST cluster bootstrap and the real/synthetic facet:

| comparison | facet | paired Δ | 95% CI | verdict |
|---|---|---:|---|---|
| 4a − 4b | all 127 | −0.0096 | [−0.0168, −0.0017] | (see caveat) |
| 4a − 4b | **real (65)** | +0.0011 | [−0.0168, +0.0116] | indistinguishable |
| 4a − run 2 | all 127 | −0.0376 | **[−0.0507, +0.0021]** | indistinguishable |
| 4a − run 2 | **real (65)** | **−0.0033** | [−0.0143, +0.0033] | indistinguishable |

**The headline above — "paired 4a − run 2 = −0.0376, 95% CI [−0.0616, −0.0136]
— excludes zero" — was computed with a chunk-level normal-approximation
interval.** Clustered by post the same point estimate straddles zero, and on the
*real* pairs alone the effect is **eleven times smaller** and nowhere near
significant. Run 4a's promotion over run 2 therefore rests on (a) treating 127
chunks from 6 posts as 127 independent observations and (b) counting our own
generators' paraphrases in the headline. Neither is a reason to un-promote it —
4a is not *worse*, its length and anchor behaviour are fine, and it was trained
on a corpus run 2 never saw — but "a real gain" is not what this data supports.

**Caveat on the 4a − 4b row.** A percentile bootstrap over **6** clusters is
coarse and mildly anti-conservative; a tight interval there means the six
per-post differences happened to agree in sign, not that the effect is
established. The real-facet row, where the same comparison is indistinguishable,
is the one to believe. Both point at D42: the stratum needs more posts before
any of these intervals mean much.

**Every signal says the two run-4 arms are indistinguishable.** The filter costs 15% of the scarce
real pairs (31 of 172) and buys nothing measurable at n=127 — including on
anchor fidelity at inference, the thing it exists to move. The pre-registered
rule (keep the filter unless the unfiltered arm *wins* on the detector with the
judge agreeing) therefore keeps it, but on a tie rather than on evidence; the
honest summary is that anchor-damaged pairs are ~4% of the diet and behave like
noise at this corpus size. Both arms close ~90% of the slop→Dan detector gap
(4a 90.7%, 4b 88.3%) against runs 1–2's ~55%, but **the detector was refit the
same day**, so that is not a like-for-like comparison with the run-1/2 table
above.

**Effective passage weight — measured 2026-09-03 (W9).** The corpus is cut at
four diets over the same prose (`meta.merge_max_chars` null / 1500 / 3000 /
4500), so a training run walks each stretch of prose several times under a
manifest claiming one epoch. Over all 8,244 records: **5,325 distinct targets
are strict substrings of another target in the same post**, carrying **6,157
pairs (74.7%)**; 883 targets carry more than one pair; **92.4%** of the corpus
sits in some containment-or-repeat group. Reproducing run 4a's selection
(`styler_pair_ok`, `splits.json`, `val_frac=0.1`, `seed=0`) against today's
corpus:

| | count | mean |
| --- | --- | --- |
| train pairs | 3,981 | |
| distinct target bodies | 3,533 | **1.13** pairs each |
| passages (source + containment) | **982** | **4.05** pairs each, max **15** |
| val pairs / passages | 392 / 95 | 4.13 |

Two readings. (1) The multiplicity is the budget sweep, not generator
replicates: **653 of the 678 multi-pair passages span ≥2 budgets**, and pairs
per *distinct body* is only 1.13. (2) It re-weights the scarce side — real
captures were cut once, synthetic pairs went through the sweep, so real pairs
are **3.09% of train pairs but 12.2% of train passages**. The by-post val split
cannot show this: val inflates identically (mean 4.13), so held-out loss is
computed over the same skew. Capping at one pair per passage would leave 982
pairs; at 2, 1,660; at 3, 2,212. Mechanism and accounting landed (D43);
**no run has been re-cut or re-weighted** — that costs a paid retrain.

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

**Segmenter implementations: 3 → 1** (2026-09-03). Both "cannot be deleted"
turned out to be claims about packaging and process boundaries rather than about
segmentation, and both fell the same day. `livingthing.qmd_core.segment_for_edit`
became a thin call to `stylebot.markdown.protected_spans` once bare stylebot
became a hard blog dependency (W3, D28 superseded); the TypeScript port became a
`segment` NDJSON op once the extension — which already owns the sidecar's
lifecycle and goes dormant without it — was allowed to ask over the wire it
already holds open (W4, D44). Measured before each deletion: the blog copy was
byte-identical on all 1,758 `.qmd`, and the TypeScript port matched on
**1,728 of 1,757**, with all 29 differences corrections (28 a heading glued to
its paragraph, 1 an unclosed fence) — plus 385 callout paragraphs, 96,796 chars,
that the port could not see at all because `div_is_prose` is a blog-side
callable. Both golden fixtures went with them; what pins segmentation now is
`stylebot.markdown`'s own tests plus `tests/test_serve.py` and
`tests/test_protection_parity.py` over `tests/fixtures/protection_parity.qmd`.

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
