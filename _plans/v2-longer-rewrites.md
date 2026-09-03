# v2 · Longer rewrites — scale curriculum + discourse coherence

Dan's v2 notes (2026-07-22): paragraph-by-paragraph rewriting loses
coherence between paragraphs, and 1500 chars feels like ~1/3 of the length
we actually want. This doc records what blocks longer rewrites, the
groundwork ALREADY LAID so the data problem stops compounding, and the
path to a v2 run.

> **Status 2026-09-03 — the v2 adapter trained, and the scale question is now
> also a weighting question.** Run 4 trained the mixed diet later on 2026-08-17
> (two arms, 4a anchor-filtered and 4b the ablation); which of them serves, and
> on what corpus, is `dan-style status`, not a sentence here. Two things this
> doc did not anticipate: the 2026-09-03 re-analysis clustered run 4's paired
> interval by post and the headline gain over the previous run stopped being
> distinguishable from zero ([`findings.md`](findings.md)); and the budget sweep
> that produced the curriculum is *also* the reason one stretch of prose carries
> ~4 training pairs, which up-weights the synthetic side 4× against real
> captures ([`decisions.md`](decisions.md) D43). A scale facet and a
> multiplicity cap are the same knob seen from two sides. **Counts below are as
> measured on their date and are not maintained.**
>
> **Status 2026-08-17 — generation RAN; no v2 adapter trained yet.** The blog
> corpus holds 4,532 pairs stamped `chunk_mode: cross-heading` at
> `merge_max_chars` 1500 / 3000 / 4500 (2,629 / 1,096 / 807), median anchors per
> target 4 / 7 / 10. That clears the "blocked — no labelled baseline arm" note
> below, which is kept for the reasoning about *what the labels mean* (still
> live: the stamp records which batch first produced a target's text). The
> mixed-scale training run is the blog's run 4,
> `_plans/styler-run4-preflight-2026-08.md`. The two thresholds this doc's
> RUNBOOK counterpart asked to re-audition at 4500 look scale-stable among
> written pairs (`transform_sim` p90 0.90/0.91/0.92, length ratio p90
> 1.91/1.67/1.66) — post-gate survivors, so a bound on shape, not on rejection
> rate.
>
> **Status 2026-08-04 — mechanism shipped, generation not yet run.**
> The diagnosis below was half wrong and the section "Why the model is
> stuck at 1500" is corrected in place; read "The section wall" first.
> Decisions taken: mixed scale curriculum (1500/3000/4500), headings
> inline and **rewritable**, v1 kept alongside v2. What exists now:
> `cross_heading` chunking (`targets.py`), gap-coalescing + `ai-style log
> recut` (`capture.py`), amended `STYLE_SYSTEM`, `data.chunk_scale` in the
> manifest, `inline_headings` through inference and the editor seam. What
> does not: any v2 pair has yet been generated, and no v2 adapter trained.

> **Blocked 2026-08-11 — the ablation has no labelled baseline arm.** Measured
> over the live 4595-pair corpus: `meta.merge_max_chars` and `meta.chunk_mode`
> are present on **0 of 4140** synthetic pairs — *absent*, not `None`.
> `_build_record` stamps them, but that code landed 2026-07-23 and 2026-08-04
> while the whole synthetic corpus was generated 2026-07-21/22. So "did scale
> help" cannot be answered by faceting this corpus: the v1 arm carries no scale
> label to contrast a v2 arm against. Nothing below is wrong and no pair is
> damaged — a covariate is missing, not data. Counts and method in
> [`findings.md`](findings.md).
>
> **Unblocked by** any synthesis batch generated from now on: the stamping
> already ships, so a v2 run labels itself and needs no new mechanism. What is
> *not* available is retrofitting. An un-stamped pair cannot be assigned to a
> scale after the fact, because the target text alone does not say what budget
> produced it — a 600-char target is equally consistent with a 1500 and a 4500
> budget, both hitting the section wall. Mixing pre- and post-facet pairs
> therefore yields a corpus with an unlabelled stratum, not a three-way
> comparison.
>
> What v1 *was* generated under is **inferred**, not recorded: today's live
> policy re-derives 3681 of 3686 distinct corpus cells (99.86%) from the blog at
> the pinned generation-time revision, so section-bounded @ 1500 is near-certain
> — but that is agreement, not provenance, and any write-up should label it so.
> The cheapest way to make v1 a real arm is to re-synthesise a labelled slice at
> 1500 alongside the 3000/4500 batches rather than to assert the label.
>
> **Second reason to re-synthesise that slice, added 2026-08-12.**
> `--drop-invented-anchors` now defaults on (D32), so every batch generated from
> here is anchor-gated while the legacy 4,595 pairs are not. Contrasting a fresh
> v2 arm against the *existing* corpus as the 1500 baseline would vary two things
> at once — scale and anchor gating — and the scale question is the one this
> ablation exists to answer. Generate all three scales fresh under identical gate
> settings. The same hazard applies to `infer.is_anchor_faithful` at training
> assembly: it removes ~18.6% of the legacy corpus, so apply it to every arm or
> to none. Gate cost at v2 density is measured and bounded (`findings.md`,
> "Why this matters for the v2 scale curriculum") — the rejection curve
> plateaus, so budget redraws, not a rewrite of the plan.

## What `meta.merge_max_chars` actually means (read before faceting on it)

Measured 2026-08-12, against the first real v2 batches. **The stamp records
which batch first produced a target's text, not which design cell the pair
belongs to.** Those differ, and the gap is order- and interruption-dependent.

`_synth_key` hashes the target **text** (plus generator, strategy, effort,
prompt id, heading context, replicate) and deliberately **not** the budget —
"key on content, record circumstance". So when two budgets cut the same passage
to byte-identical text, that is *one* cell, generated once, stamped by whichever
batch reached it first. Which happens often, because of the section wall below:
a section shorter than 1500 chars is packed identically at every budget.

Measured over the live blog with the real chunker and selector:

| | targets | unique to that budget |
|---|---:|---:|
| 1500 cross-heading | 2,491 | 2,366 (95.0%) |
| 3000 cross-heading | 1,282 | 1,077 (84.0%) |
| 4500 cross-heading | 937 | 761 (81.2%) |

Overlaps: 1500∩3000 = 100, 1500∩4500 = 71, 3000∩4500 = 151, all three = 46. In
the live corpus, of the pairs whose target text belongs to *both* the 1500 and
4500 sets, **57 carry `merge_max_chars: 1500` and 2 carry `4500`** — the 1500
batch ran first and claimed nearly all of them. The 2 exist only because that
batch was interrupted (1,985 of 2,491) and a few coincident cells fell through
to the next one. **So where a run stops changes the labelling.**

**This is not a bug and must not be "fixed" by keying on the budget.** That
would mint a separate pair per budget for identical text, multiplying the weight
of exactly the short passages the section wall already over-represents — the
failure `--skip-covered` exists to prevent. The dedup is right; only the label's
meaning needs stating.

**Consequences for the ablation:**

- Faceting the corpus by `merge_max_chars` contrasts *short targets against long
  ones*, which is adjacent to the scale question but is not a randomised
  contrast, and it is contaminated by run order at the ~2–16% level.
- For a clean contrast, restrict to targets **unique** to one budget (the third
  column above): 2,366 / 1,077 / 761, disjoint by construction and immune to
  ordering.
- Or sidestep the faceting entirely — treat the corpus as one mixed-scale diet,
  as the curriculum intends, and put the scale question to the *adapter* by
  evaluating it on length-stratified inputs. That measures what the run is
  actually for, and needs no clean strata in the training data at all.

Whichever is chosen, say which in the write-up. The v1 arm's label was inferred
rather than recorded (above); this one is recorded but means something narrower
than its name suggests.

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
re-stamped by a one-off `scripts/restamp_style_system.py` (system message
only — user and assistant sides untouched, verified; the script was deleted
in 0.3 once the only corpus that exists had been through it). A restyled heading changes its
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
writes the full before/after texts to `_training_pairs/captures/<capture_id>/`
beside the corpus. The chunk records become a lossy *view*; the archive is
the re-cuttable raw. Any future chunking policy (3–4 merged paragraphs,
whole sections, pairs with preceding-paragraph context) can be applied
retroactively to everything captured from now on, without recapturing.
Cost: ~one blog post of text per capture. Commit them like `pairs.jsonl`.

**Re-shaped 2026-08-04 — the archive is files, not a blob.** It was one
JSON file per capture with the documents as escaped strings, which made it
unreadable, unopenable and undiffable — perverse, given the before→after
diff *is* the training signal, and it meant a botched capture could not be
repaired. Now a directory of `before.md` / `after.md` / `meta.json`, so
`diff` works and a capture can be hand-corrected before the next `recut`.
Session snapshots got the same treatment
(`snapshots{,-closed}/<source>/{content.md, meta.json}`).

`.md` and not `.qmd` is load-bearing, since the archive lives inside the
blog repo: `*.qmd` there is reformatted by the pre-commit hook (which would
destroy the verbatim record the archive exists to keep), indexed by `qmd`
search, and sampled by `iter_targets` as a synthesis target — and a
verbatim frontmatter copy passes every selector gate, so captured slop
would be trained on as human prose. `ai-style data migrate-captures` converts
the old blobs (verified byte-identical `recut` output across the migration; it
lived at `scripts/migrate_capture_archives.py` until 2026-09-03, when the
`scripts/` carve-out was replaced by the hidden `data` group — `decisions.md`
D38). **Compatibility dropped in 0.3:** `iter_archives` read legacy
blobs — warning each time — for one release; it now reads directories only
and *raises* on a leftover `captures/*.json`, naming the migration command.
Raising, not skipping: a silent skip is indistinguishable from "that
capture had no changed regions", so an archive could drop out of the corpus
unnoticed. The migration script parses the blobs itself and still works.

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
   (None = unmerged; pre-facet records lack the key — that is *every* record
   in today's corpus, and "they were all 1500" is inference from re-derivation,
   not a stamp: see the blocked note at the top).
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
