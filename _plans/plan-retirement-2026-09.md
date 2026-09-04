<!-- Wave-1 work file for ship-shape-2026-09.md slice T1. Load-bearing parts: each plan section §x.4 "Unique numbers" (move to findings.md), §x.5 durable contracts (the stub points at these symbols), §x.6 inbound references (none may dangle), §x.7 the draft stub. DELETED, not stubbed, in the T1 commit — git holds it. -->

# Evidence for retiring five stylebot plans to P7 stubs — 2026-09-04

Read-only survey of `/Users/dan/Source/stylebot` at HEAD `880a579`
(`880a5795c975d2989296c51dbc9e6799dc51fa8a`, branch `main`, clean) and of the
blog checkout `/Users/dan/Source/livingthing`, so that a later agent can rewrite
`_plans/eval-harness.md`, `v2-longer-rewrites.md`, `synth-flag-audit.md`,
`anchor-integrity.md` and `vscode-marker.md` as ≤ 40-line stubs in the shape of
`livingthing/_plans/stylebot-seam-qa-2026-09.md` (what it was / what shipped,
commit range / what was refuted, D-numbers / what stayed open, successor) and a
reviewer can check every claim. For each plan: the git history and the
`git show` pointer; the commits that shipped its work; the D-numbers it became
and the claims that were refuted; every number in the file with an `rg` verdict
against `_plans/findings.md` (and `decisions*.md`), the ones that live nowhere
else listed verbatim under **Unique numbers**; the durable contracts still true
only in the file, each with the code symbol that now embodies it; every inbound
reference with its sentence and what to do about it; and a draft stub. Nothing
in either repo was edited. A closing paragraph checks whether
`masking-placement.md` and `placeholder-design-2026-08.md` are really still
open. Section headings and test counts were checked against the tree, not the
plan text (`tests/test_eval.py` 49 tests, `test_eval_integrity.py` 35,
`test_serve.py` 21, `test_infer.py` 72, `test_synth.py` 47, by `def test_`).

Conventions below: **✓ findings** = the number is in `_plans/findings.md`
(section named); **✓ Dn** = it is in that decision entry; **✗ unique** = it is
in no other layer. Line numbers are as of HEAD.

---

## 1. `_plans/eval-harness.md` — 408 lines, "✅ BUILT"

### 1.1 Git history and pointer

25 commits touch the file, `5c34b21` (2026-06-26, Phase 0 scaffolding) →
`a3b094a` (2026-09-04). **Last-touch sha `a3b094a`**; the stub's pointer is
`git show a3b094a:_plans/eval-harness.md` (408 lines). Dated status blocks
inside the file: 2026-06-28 (as built), 2026-06-30 (detector decision),
2026-08-17 (judge demoted, `fc39abf`), 2026-09-03 (integrity audit, `69aef7b`,
`e952e86`, `2c3384d`), 2026-09-04 (`dba257b`, `a3b094a`).

### 1.2 Commits that shipped its work

Range **`054b2ec..7b11756`** (2026-06-28 → 2026-09-04); the load-bearing ones:

| sha | date | what |
|---|---|---|
| `054b2ec` | 06-28 | batched eval, OpenRouter judge, slop-strategy facet |
| `9915e89` | 06-29 | Vale signal fixed (glob-scoped styles via temp file) |
| `74a470a` | 06-30 | `stylebot.classify` linear-head detector seam |
| `b470e59` `e3f155f` `aa69518` | 07-06 | trainer into stylebot (`[classifier]`), provenance facet, 3-role by-POST `splits.json` + nested-CV `select_C` |
| `27c910d` `b509b3d` | 07-20/21 | `export-table`, cost/latency covariates on score rows |
| `1f6d604` | 08-10 | `scores.jsonl` never carries orphans; run/serve read the chunk diet |
| `61073be` | 08-11 | Vale batched into one invocation per slice |
| `460d6b3` `bd882a3` | 08-17 | judge records cost; `run` corpus mode emits `output`-bearing JSONL |
| `69aef7b` | 09-03 | detector identity on rows, clustered CIs, real-facet headline (D39–D41) |
| `e952e86` | 09-03 | `extend-eval` (D42) |
| `d0e5d10` | 09-03 | Pangram moved to an async task API; shipped URL was 404 |
| `2c3384d` | 09-03 | document verdict = mean over serving-scale chunks (D54) |
| `dba257b` `a3b094a` | 09-04 | detector out of the pool; one mechanism, the judge awaits (D57→D59) |
| `b54055d` | 09-04 | `eval` split into `signals`(in `eval.py`)/`scoring`/`verdict`/`docscale` (D62) |
| `5d5f33f` `f437494` `7b11756` | 09-04 | callout policy pinned across the docscale seam; chunked detector's name carries its div policy; judge path refuses concurrency it cannot deliver (F19c) |

Blog side: `docscale-20260903` / `docscale-20260904` manifests
(`33bdcd2031`, `66e5f0f117`, `649d6f43a1`).

### 1.3 D-numbers it became; claims refuted

Became: **D7** (detector = trained Dan-vs-slop classifier; eval-vs-reward,
Goodhart), **D17** (val by POST), **D18**, **D34** (alias-resolved role),
**D39** (score carries its instrument), **D40** (cluster by post; real facet
headline), **D41** (automation cap on synthetic only), **D42** (add-only eval
stratum; manifest pins), **D54** (document verdict chunk-wise), **D55→D59**,
**D56**, **D57→D59**, **D59** (judge awaits, rest batches), **D62** (lazy
re-export map), **D63** (orphan rows pruned where read), **D66** (judge bills
through BYOK).

Refuted or gone stale, in the file's own words vs. the record:

- "Pangram — **built 2026-09-03, not yet run**" → run 2026-09-03, PARTIAL: the
  shipped endpoint was dead (404) and the account hit 402 after 92 of 149 pairs
  (findings "Pangram cross-check"; fix `d0e5d10`). The sizing in the file
  (298 calls, ~$3.00) was never the number; the measured one is 92 pairs.
- The run-4a headline the file's 2026-08-17 layer reported against was a
  chunk-level interval; clustered by post it straddles zero (D40, findings
  "Run 4's verdict, re-analysed").
- LLM judge as a signal → demoted to a sanity check (findings "The LLM judge";
  already recorded in the file at `fc39abf`).
- First document-scale run scored whole documents and had no resolution
  (0.49 on all three fields) → D54, chunk-wise (already in the file).
- Done-criteria say `tests/test_eval.py` "(15 tests)" and
  `tests/test_eval_integrity.py` "(25 …)": today 49 and 35. Dated counts, D52.
- Outputs say the summary is `schema_version: 3`; Done-criteria say
  `schema_version: 2`. Code: `eval.SCHEMA_VERSION = 3` (`eval.py:45`).
- "inline anchors triple the all-candidates-rejected rate: 6.4% → 19.1%" —
  findings and D27 say **19.0%**.
- "The scoring pass … ran in two [processes] for a day" — D55/D57 superseded
  the same day; `ScoreJob` child and the pool are deleted (`a3b094a`).
- "Planned — synthetic↔real distribution match (a fifth capability)" is still
  unbuilt: no `stylebot.features` module exists (`ls src/stylebot/`), and
  `argued-slop-arm.md` names it as its stage 0.

### 1.4 Numbers in the file, and where they live

| number(s) in the file | verdict |
|---|---|
| judge 3/20 real, 14 ties; cannot resolve two adapters at n=40; ~$2.71 for 160 calls | ✓ findings "The LLM judge" |
| backbone table 0.78/0.72, 0.76/0.71, 0.75/0.62, 0.73/0.67 | ✓ findings "The voice classifier" (identical table) |
| held-out by-POST pairwise 0.78 / AUC 0.72 | ✓ findings, ✓ D7 |
| real-facet 0.734 | ✓ findings "eval_v2" (0.7337) |
| 254 Phase-1 pairs | ✓ findings "Corpus composition over time" |
| eval_v2: `05b495adac68cdc4`, 12 posts, 177 pairs, 149 real / 28 synthetic, largest post 22% | ✓ findings "eval_v2 — applied" |
| 77 of eval_v1's 103 pairs from one post; 6 → 12 clusters; 62 of 103 synthetic | ✓ findings "Eval-stratum composition" |
| whole-document 0.49 on all three fields, identical min/max, 25k chars | ✓ findings "Document-scale eval, whole-document" |
| ≈ $0.27 (eval_v2) vs ≈ $0.015 (eval_v1); 3 → 35 documents | ✓ findings "eval_v2" |
| judge 6.4x at eight workers, 1 s/call | ✓ findings "Threading audit" |
| 6.4% → 19.1% (inline anchors) | ✓ findings "Inference and serving" / D27, as **19.0%** |
| default judge model `anthropic/claude-opus-4-8`; scores 1–5 | code `eval.py:330,355` |
| `--max-workers` is the judge's concurrency only | D59, `scoring.score_pairs_file` |
| Pangram `~$0.05/1k words` | ✗ unique (price, not a measurement) |
| 149 pairs / 298 calls; median target ~1.2k chars ≈ 200 words; ~60k words ≈ $3.00; ~$0.80 under eval_v1's 41 | ✗ unique — a **forecast**, superseded by the partial run |
| "≥48 from `automation:2`" (of the 254) | ✗ unique |
| "a dedicated validation stratum at 28-post scale would starve every other role" | ✗ unique (2026-07-06 design-time count) |
| 35 documents, **1,561,740 chars**; 86 capture archives exist; eval_v1: 3 documents, **86,820 chars** | ✗ unique (findings has "3 → 35" and the dollars, not the chars or the 86) |
| ~350 prose chunks, ~390k output tokens, Tinker ~$0.60/1M, "the better part of an hour" | ✗ unique — forecast; the measured pass scored 563/564/563 then 608/609/607 chunks (findings) |
| 15 tests / 25 tests | stale counts, drop |

**Unique numbers (verbatim, for findings.md if kept at all):**

- "the corpus median target is ~1.2k chars ≈ 200 words" — sizing, 2026-09-03.
- "Under eval_v2 that is 149 pairs / 298 calls … ~60k words ≈ $3.00 at $0.05/1k
  words (it was ~$0.80 under eval_v1's 41)" — forecast for the Pangram pass.
- "the real-slop reference (the 254 Phase-1 pairs, ≥48 from `automation:2`)" —
  2026-06-30.
- "a dedicated validation stratum at 28-post scale" — 2026-07-06.
- "Measured 2026-09-03 under eval_v2: 35 documents, 1,561,740 chars (86 capture
  archives exist; under eval_v1 only 3 sat on eval-role posts, for 86,820
  chars)."
- "1,561,740 chars is ~350 prose chunks and ~390k output tokens, so at Tinker's
  ~$0.60/1M that is ≈ $0.27 and the better part of an hour".

Recommendation: the 1,561,740 / 86,820 / 86-archive line belongs beside
findings "eval_v2 — applied 2026-09-03" (it is the measured input size of the
document-scale pass); the forecasts can be dropped — the measured passes
supersede them.

### 1.5 Durable contracts still true, and the symbol that embodies each

| statement (only in this file) | symbol now |
|---|---|
| Four signals; keyless by default — absent signals are explicit nulls, never faked | `stylebot.eval` module docstring (cites this plan by section), `eval.null_detector`, `eval.vale_score`, `eval.openrouter_judge` |
| `Detector` protocol: `prose -> {score=P(slop), p_dan, name}`; polarity = P(slop) so it composes with `mean_detector_score` | `classify.sklearn_detector`, `eval.Detector`, `classify.detect_many` |
| One shared by-POST partition, three roles, eval frozen, styler/detector by hash | `stylebot.splits` docstring (cites "the split contract"), `splits.make_splits`, `splits.role_of`, `splits.extend_eval` |
| Nested-CV `C` inside each training side; no fourth split for hyperparameters | `classify_train.select_C`, `classify_train.C_GRID` (`3aa29ac` extended it past 10.0) |
| Provenance facet, headline = real; `meta.n_pairs_real/synthetic` | `scoring.summarize_scores` (`by_provenance`, `headline_facet`), `classify_train` metrics `by_provenance` |
| A score carries its instrument; resume refuses a different head; compare refuses two | `classify.DETECTOR_IDENTITY_KEYS`, `scoring.assert_single_detector`, `ai-style eval --allow-mixed-detector`, `verdict.compare_scores(allow_cross_detector=)` |
| Paired verdict bootstraps over POSTs, real facet by default | `verdict.paired_delta(clusters=)`, `verdict.compare_scores(cluster_by="source", provenance="real")` |
| `scores.jsonl` is id-keyed, resumable, orphan-free; `--no-prune-orphans` for a foreign corpus | `scoring.prune_orphan_scores`, `scoring.live_record_ids`, `--prune-orphans/--no-prune-orphans` (`ai_style.py:369`) |
| Document-scale: rewrite whole, score chunk-wise, unaligned across fields; verdicts under `runs/<run_id>/` | `docscale.rewrite_documents_file`, `docscale.iter_document_pairs`, `docscale.verdict_dir`, `eval.chunked_detector`, `--documents` / `--documents-manifest` / `--documents-role` |
| Pangram is opt-in, one-shot, never in a hot loop; missing `fraction_ai` is `None` | `eval.pangram_detector(post=)` |
| Judge is the only network wait; semaphore-gated | `eval.openrouter_judge_async`, `scoring.score_pairs_file(max_workers)` |
| Guardrail: detector movement alone is never the success criterion | `eval` docstring "Guardrail", `infer.py:24`, D7 |
| Report renders any score field, no spend | `report.render_scores_report`, `--report` |
| Pair-schema primitives live in `pairs` | `pairs.record_id`, `pairs.pair_body`, `pairs.FIELD_EXTRACTORS`, `pairs.iter_pairs` |

Everything in the table is already in code; the stub needs only the pointers.
The one "why" not yet in a D-entry is the **fifth-capability spec** (feature
extractor over synthetic vs real-slop strata, seam = `meta.synthetic` facet over
the existing aggregation, consuming `meta.gen`) — ~8 lines that should move to
`argued-slop-arm.md`, which already claims it as stage 0.

### 1.6 Inbound references (each with its sentence)

- `_plans/OVERVIEW.md:17` index row — "the four signals, the detector decision,
  the split contract". Update the description to "retired stub".
- `_plans/OVERVIEW.md:192` phase map — "`stylebot.eval`, `ai-style eval` —
  [eval-harness.md] ✅ built …". Link survives; fine.
- `_plans/OVERVIEW.md:259` — "Spec in [eval-harness.md] §'Planned'."; `:315` —
  "See `eval-harness.md` §'Planned'." **Dangles** once the section goes; repoint
  to `argued-slop-arm.md` after moving the spec there.
- `_plans/OVERVIEW.md:295` — "with the eval-harness Goodhart guards" (prose;
  repoint to D7).
- `_plans/decisions.md:243` (D40) — "the verdict path must import with no
  extras (`_plans/eval-harness.md`)". Repoint to `tests/test_eval.py` /
  `stylebot.verdict` (the constraint is a test, not a plan).
- `_plans/decisions-archive.md:260` (D17) — "the split contract in
  `eval-harness.md`"; `:704` (deferred experiment) — "the reward-hacking
  failure [eval-harness.md] warns about in its own guardrail section". Repoint
  to `stylebot.splits` docstring and D7.
- `_plans/argued-slop-arm.md:97` — "[eval-harness.md] §'Planned', scoped to
  what this slice …"; `:109` — "the seam `eval-harness.md` names". Both cite
  the fifth-capability spec; move the spec into that file and drop the links.
- `src/stylebot/eval.py:7` — "The signals (`_plans/eval-harness.md` 'The four
  signals')"; `:13` — "Guardrail (`_plans/eval-harness.md` 'Guardrail')";
  `:407` — "see `_plans/eval-harness.md` 'The detector decision'". Section
  names will not exist; per P1 cite **D7** instead.
- `src/stylebot/splits.py:3` — "The reward-safety rule (see
  `_plans/eval-harness.md` 'the split contract')". Cite D17/D42.
- `src/stylebot/infer.py:24` — "Small N is Goodhart-gentle (KL ~ log N;
  eval-harness.md 'Eval vs reward')". Cite D7.
- `src/stylebot/classify_train.py:548` — "(see `_plans/eval-harness.md` …)".
  Cite D17/D42.
- `livingthing/src/livingthing/styler.py:385` — "the frozen list cannot drift
  (`eval-harness.md`)". Cite D42.
- `livingthing/_plans/styler-run4-preflight-2026-08.md:23,222,279,301` —
  historical citations in a blog plan that is itself a P7 candidate (its status
  is "W4 half-done"); plain text, not links. Leave.

### 1.7 Draft stub

```markdown
# Eval harness — retired 2026-09-04 (P7)

**What it was.** The offline scorer every other phase reports against: Vale,
an LLM judge, a statistical detector and Dan's eyeball over the fields of a
`pairs.jsonl`, id-keyed, resumable, keyless by default. Written 2026-06-26,
rewritten at each measurement; full text `git show a3b094a:_plans/eval-harness.md`.

**What shipped** — stylebot `054b2ec..7b11756` (2026-06-28 → 2026-09-04):
`stylebot.eval` / `scoring` / `verdict` / `docscale`, `ai-style eval`,
`extend-eval`, `export-table`. The detector became the trained voice
classifier (`classify.sklearn_detector`, D7) behind one 3-role by-POST
`splits.json` (`splits`, D17/D42) with nested-CV `C` (`classify_train.select_C`).
A score row carries its instrument (`classify.DETECTOR_IDENTITY_KEYS`, D39); a
verdict bootstraps over posts on the real facet (`verdict.compare_scores`,
D40/D41); the stratum is add-only (`splits.extend_eval`, D42); a document is
rewritten whole and scored chunk-wise (`docscale`, `eval.chunked_detector`,
D54); the judge is the only concurrency (`scoring.score_pairs_file`, D59).
Pangram is a one-shot opt-in (`eval.pangram_detector`). Numbers:
`findings.md` "The voice classifier", "The LLM judge", "Eval-stratum
composition", "eval_v2", "Document-scale eval", "Pangram cross-check",
"Threading audit".

**What it refuted.** The judge cannot see real slop (3/20, 14 ties) and never
arbitrates two adapters; run 4a's "real gain" was a chunk-level interval and
straddles zero clustered by post (D40); a whole-document embedding has no
resolution (D54); "built and tested, not run" Pangram had a dead endpoint (404)
and ran partially; a thread pool around the detector bought nothing (D55, D57
→ D59). Its Goodhart guardrail — detector movement alone is never success —
stands, in `stylebot.eval`'s docstring and D7.

**What stayed open.** The fifth capability, a synthetic↔real distribution
match over the `meta.synthetic` facet, is unbuilt; its spec is stage 0 of
[`argued-slop-arm.md`](argued-slop-arm.md). D53's open half: `extend-eval`
does not prune the synthetic pairs of the posts it freezes.
```

(34 lines.)

---

## 2. `_plans/v2-longer-rewrites.md` — 327 lines, trained as the blog's run 4

### 2.1 Git history and pointer

10 commits, `f5abafc` (2026-07-23) → `ff0e994` (2026-09-03). **Last-touch
`ff0e994`**; pointer `git show ff0e994:_plans/v2-longer-rewrites.md` (327
lines). Status layers inside: 2026-08-04 (`36a0ecb`), 2026-08-11 blocked
(`f95de12`), 2026-08-12 (`e2ea109`, `8c486b8`), 2026-08-17 generation ran
(`896fc8f`), 2026-09-03 (`ff0e994`).

### 2.2 Commits that shipped its work

Range **`f5abafc..7cfe445`** (2026-07-23 → 2026-09-03) in stylebot:

| sha | date | what |
|---|---|---|
| `f5abafc` | 07-23 | capture archive (full before/after) + `offsets`/`para_before`/`para_after` in preference records |
| `7428223` | 07-23 | `Target.merge_max_chars` → `meta.merge_max_chars` |
| `36a0ecb` | 08-04 | cross-heading chunks, gap-coalesced captures, `ai-style log recut`, `STYLE_SYSTEM` amended (D64), `data.chunk_scale` in the manifest, `inline_headings` through inference and the editor seam |
| `1c47374` | 08-09 | archives as directories of `.md`, not JSON blobs |
| `4230ea7` | 08-09 | `max_tokens` wired to the chunk budget |
| `1f6d604` `b8f172d` | 08-10 | `run`/`serve` read the chunk diet from the manifest (`backends.serving_diet`) |
| `650f361` | 08-11 | legacy blob archives dropped; `iter_archives` raises |
| `d39c72a` | 08-11 | `--skip-covered` on a normalised coverage key |
| `2388335` | 08-17 | recut: null-edit floor, min-chars default |
| `9cb8ce3` | 09-03 | `migrate-captures` under `ai-style data` (D38) |
| `7cfe445` | 09-03 | passage = stretch of prose; `pair_weight`; `sort_key` deleted (D43) |

Blog side: v2 batches generated 2026-08-12/13 under the gate (`8733c43e09`
GATE_RETRIES=2, `a07bfce754`, `21221dcbe8`); run 4a `8cbb626696` and 4b
`d38532f48c` (2026-08-17), `9f5f6d39bd` promoted:false; recuts `14beb8e791`,
`d585e1b723`. RUNBOOK §6b "The v2 scale curriculum (paid, RAN 2026-08)".

### 2.3 D-numbers; claims refuted

Became: **D15** (chunk scale; section wall), **D64** (`STYLE_SYSTEM` amended
once, re-stamped), **D43** (passage weight; a scale facet and a multiplicity
cap are one knob), **D38** (`migrate-captures`), **D36** (the absent facet →
provenance stamping), **D11** (content-only keys → what the stamp means),
**D32/D33** (anchor gate confounds the ablation; survival collapses at v2
scale), **D40** (run 4's gain does not survive), **D14** (heading context).

Refuted, mostly by the file itself in place, so the stub needs only the list:

- First diagnosis "raise `MERGE_MAX_CHARS`" → the section wall (median section
  553 chars); packing across headings is the only route (D15/D64).
- First draft's fixed-width window → gap-coalescing (changed-fraction 0.96 vs
  0.41; findings "Capture diffing").
- "`diff_chunks` discards paragraph-count-changing edits" → 1 dropped opcode
  over 41 archives.
- "Did scale help" by faceting on `merge_max_chars` → the stamp records the
  first batch, arms are nested (findings "Chunk budgets overlap"); the 2026-09
  answer is D43's multiplicity cap, not a facet.
- 2026-08-12 note "the rejection curve plateaus, so budget redraws" → D33:
  exponential in anchor count, the plateau was v1's short chunks.
- "no v2 adapter trained yet" → run 4a/4b, 2026-08-17; and its headline gain
  over run 2 straddles zero clustered by post (D40).
- "`scripts/restamp_style_system.py`" and "`scripts/migrate_capture_archives.py`"
  — both gone; D38/D64.

Still open (the file's own list, checked): step 5 (previous-rewritten-paragraph
conditioning) untouched; the detector is still paragraph-scale and the editor
thresholds `[0.51, 0.55, 0.59, 0.65]` are unchanged in the blog's
`.vscode/settings.json:206` although the head was refit 2026-08-17 and
2026-09-03 (findings "eval_v2"); the VS Code extension still does not offer a
"rewrite section" mode (the seam `rewrite_candidates` allows it).

### 2.4 Numbers, and where they live

| number(s) | verdict |
|---|---|
| 4,532 stamped pairs; 2,629 / 1,096 / 807 per budget | ✓ findings "Anchors and placeholders: v2 post-gate distributions" |
| `transform_sim` p90 0.90/0.91/0.92; length-ratio p90 1.91/1.67/1.66 | ✓ same table |
| facet on 0 of 4140 synthetic; 4595 corpus | ✓ findings "The chunk-scale facet is absent" |
| 3681 of 3686 cells (99.86%) re-derived | ✓ same |
| `is_anchor_faithful` removes ~18.6% | ✓ D32, findings |
| targets/unique table 2,491/2,366 (95.0%), 1,282/1,077 (84.0%), 937/761 (81.2%); overlaps 100/71/151/46 | ✓ findings "Chunk budgets overlap" |
| section median 553, p90 1729, 13.6% > 1500, 0.8% > 4500 | ✓ findings "Section length" |
| chunk-yield table (2088/626 … 615/3224) | ✓ findings "Chunk yield by budget and mode" |
| 4,169 of 4,500 synthetic; 331 real; median 693, p90 1866, max 4143 | ✓ findings "Corpus composition", "Real captures" |
| 5.4KB single chunk byte-identical | ✓ findings "Inference and serving", D15 |
| 41 archives → 87 pairs, median 1191, p90 4230; changed-fraction 0.96 vs 0.41; 2 paras / 800 chars | ✓ findings "Capture diffing loses almost nothing" |
| ~19% of synthetic pairs rewritten to 85% the same | ✓ findings "Data hygiene", D18 |
| 1 dropped opcode | ✓ findings |
| ~8.9M target chars, ~$39 vs v1's $15.27 | ✓ findings "Cost: forecast vs actual" |
| `max_length_ratio` 3.0, `max_transform_sim` 0.95 | ✓ findings; blog `training_targets.py:314,322` |
| `DEFAULT_SLOP_MAX_TOKENS` 8192; `DEFAULT_MAX_LENGTH` 8192 | code `generators.py:173`, `train.py:68` |
| ≥10% corpus support; cross-heading majority | code `backends.MIN_SCALE_SUPPORT = 0.10`, `backends.serving_diet` |
| thresholds `[0.51, 0.55, 0.59, 0.65]` | ✓ findings "The voice classifier" |
| median anchors per target **4 / 7 / 10** at 1500/3000/4500 | ✗ unique (findings' 4/7/8 are per *chunk*, a different table) |
| "**57 carry `merge_max_chars: 1500` and 2 carry `4500`**"; "that batch was interrupted (**1,985 of 2,491**)" | ✗ unique |
| "contaminated by run order at the **~2–16%** level" | ✗ unique |
| "the synthetic median target was **662** chars against a 1500-char budget" | ✗ unique (the findings hit is `17,662`) |
| "The **109** pre-2026-07-22 captures are chunk-scale forever" | ✗ unique |
| uncoalesced **130 / 845 / 1938** (pairs / median / p90) | ✗ unique |
| changed-fraction "(**p10 0.86, none below 0.5**)" | ✗ unique |
| sized targets **2426 / 1244 / 917** at 1500/3000/4500 (2026-08-04 sizing) | ✗ unique (findings has the later 2,491/1,282/937) |
| "a 4500-char pair is **~2400 tokens** both sides" | ✗ unique (arithmetic) |

**Unique numbers (verbatim):**

- "median anchors per target 4 / 7 / 10" (2026-08-17, over the 4,532 stamped
  pairs).
- "of the pairs whose target text belongs to *both* the 1500 and 4500 sets, 57
  carry `merge_max_chars: 1500` and 2 carry `4500` … that batch was interrupted
  (1,985 of 2,491)" (2026-08-12).
- "contaminated by run order at the ~2–16% level" (2026-08-12).
- "the synthetic median target was 662 chars against a 1500-char budget"
  (2026-08-04).
- "The 109 pre-2026-07-22 captures are chunk-scale forever."
- "87 pairs, median 1191 chars, p90 4230 (against 130 / 845 / 1938 uncoalesced)"
  (2026-08-04, 41 archives) — findings has the first triple only.
- "changed-fraction 0.41 at a 4500-char window, against 0.96 (p10 0.86, none
  below 0.5) under coalescing".
- "Sized with the real selector: 2426 / 1244 / 917 targets at 1500 / 3000 /
  4500" (2026-08-04).

Recommendation: the 57/2, 1,985/2,491 and 4/7/10 lines extend findings "Chunk
budgets overlap" (same date, same instrument); 130/845/1938 and p10 0.86 extend
"Capture diffing loses almost nothing"; 662 and 109 extend "Section length" /
"Real captures"; the 2426/1244/917 sizing can go (superseded by the measured
2,491/1,282/937).

### 2.5 Durable contracts still true, and the symbol

| statement | symbol now |
|---|---|
| Cross-heading packing with headings as rewritable content; section-bounded kept as the v1 arm | `targets.ChunkPolicy` (`cross_heading`), `--cross-heading/--no-cross-heading` (`targets.py`), `infer.prose_chunks(inline_headings=)` |
| `STYLE_SYSTEM`: headings may be restyled, level and meaning kept | `ai_core.STYLE_SYSTEM` (`ai_core.py:10-19`, comment cites D64) |
| `meta.merge_max_chars` / `meta.chunk_mode` stamped per pair; `None` = unmerged; **the stamp records which batch first produced the text, not the design cell; do not key on the budget** | `synth.py:122` (docstring cites this plan), `targets.Target.merge_max_chars`, `synth._synth_key`, `pairs.py` |
| Three honest ways to analyse the arms (budget-unique targets; treat as one mixed diet and evaluate on length-stratified inputs; say which) | **nowhere else** — findings "Chunk budgets overlap" defers to this file ("How to analyse it honestly"); move to D43 or that findings entry |
| Capture archive = `captures/<id>/{before.md,after.md,meta.json}`; `.md` not `.qmd` is load-bearing; legacy blobs raise, naming the migration | `capture.capture_pairs`, `capture.iter_archives`, `ai-style data migrate-captures` (D38) |
| Gap-coalescing, not windowing: capture pairs stay changed-region-focused because untouched prose is not endorsed; synthetic windows may be long because all of the target is Dan's | `capture.diff_chunks(coalesce_gap_paras, coalesce_gap_chars)` (`capture.py:111` cites this plan), `session.recut`, `ai-style log recut --gap-paras --gap-chars` — **no D-entry holds this "why"**; the stub or a new D-entry should |
| Preference records carry `offsets`, `para_before`, `para_after` | `vscode-voice-marker/src/sidecar.ts:62-67`, serve `prefer` op |
| Serving follows the diet: largest budget with ≥10% support; inline headings on a cross-heading majority; pre-key manifests fall back section-bounded | `backends.serving_diet`, `train.build_manifest` (`data.chunk_scale`), blog `voice_classifier.py:142` |
| Editor selection may span headings under `inline_headings`; fences still raise; the size limit applies | `infer.rewrite_candidates` |
| A restyled heading can break inbound `#slug` links; warned blog-side | `livingthing.workflow_style.check_heading_anchors` |

### 2.6 Inbound references

- `_plans/OVERVIEW.md:18` index row — "the scale curriculum — mechanism
  shipped, generation run 2026-08, trained as the blog's run 4". Update.
- `_plans/OVERVIEW.md:226` — "per-budget counts are in [v2-longer-rewrites.md]
  and …". Counts are D52-banned in docs anyway; repoint to `dan-style status`
  / findings "v2 post-gate distributions".
- `_plans/findings.md:729` — "How to analyse it honestly:
  [v2-longer-rewrites.md]. Decided: D43." **Dangles** unless the three-option
  paragraph moves; put it in D43 or in that findings entry and drop the link.
- `_plans/anchor-integrity.md:285` is not a reference; `v2-longer-rewrites.md:285`
  mentions "the anchor-integrity eval (deletion_report)" — prose.
- `src/stylebot/synth.py:122` — "(`_plans/v2-longer-rewrites.md`)" on the
  stamp's meaning. Cite D11/D43 once the paragraph moves.
- `src/stylebot/capture.py:111` — "teaches the styler to leave slop alone. See
  `_plans/v2-longer-rewrites.md`." Needs a D-number to cite (see 2.5).
- `livingthing/src/livingthing/training_targets.py:183` — comment
  "# `stylebot/_plans/v2-longer-rewrites.md`." Repoint to D15/D64.
- `livingthing/_plans/styler-run4-preflight-2026-08.md:22,331,362` — historical
  plain-text citations (":331 — `v2-longer-rewrites.md`'s status block and its
  'blocked 2026-08-11 — 0 of…'"; ":362 — restrict to budget-unique targets
  (2,366 / 1,077 /…)"). Leave; that plan is itself a P7 candidate.

### 2.7 Draft stub

```markdown
# v2 · Longer rewrites — retired 2026-09-04 (P7)

**What it was.** Dan's 2026-07-22 note: paragraph-by-paragraph rewriting loses
coherence and 1500 chars is a third of the length wanted. The plan diagnosed
why the styler was pinned at 1500, laid the groundwork so the data problem
stopped compounding, and set the path to a scale-mixed adapter. Full text
`git show ff0e994:_plans/v2-longer-rewrites.md`.

**What shipped** — stylebot `f5abafc..7cfe445` (2026-07-23 → 2026-09-03):
the capture archive (`capture.capture_pairs`, directories of `.md`) and
`para_before` in preference records; `meta.merge_max_chars` / `meta.chunk_mode`
on every pair (`targets.Target`); cross-heading chunking with headings as
content (`targets.ChunkPolicy`, `infer.prose_chunks`) and the one amendment
ever made to `STYLE_SYSTEM` (D64); gap-coalesced recuts (`capture.diff_chunks`,
`ai-style log recut`); serving that reads the diet off the manifest
(`backends.serving_diet`); `per_passage` / `pair_weight` (D43). Blog: the
1500/3000/4500 curriculum generated 2026-08-12/13 under the anchor gate, run 4a
and 4b trained 2026-08-17 (`8cbb626696`, `d38532f48c`); which serves is
`dan-style status`. Numbers: `findings.md` "Section length", "Chunk yield by
budget and mode", "Capture diffing", "v2 post-gate distributions", "Chunk
budgets overlap", "Effective passage weight", "Run 4's verdict, re-analysed".

**What it refuted.** Raising the byte budget does nothing — Dan's sections are
short, so packing across headings is the only route (D15); a fixed-width
capture window teaches the styler to leave slop alone, and `diff_chunks` drops
one opcode in 41 archives; the budget stamp names the first batch that reached
a passage, not a design cell, so "did scale help" cannot be a facet — the
arms are nested and the same sweep is a 4× up-weight of synthetic prose (D43);
run 4a's gain over run 2 straddles zero clustered by post (D40); the anchor
rejection curve does not plateau at v2 density (D33).

**What stayed open.** Step 5, conditioning on the previous rewritten paragraph
— decide after a v2 eval, not before. The detector is paragraph-scale and the
editor thresholds `[0.51, 0.55, 0.59, 0.65]` predate two refits; the
extension does not yet offer "rewrite section". Choosing a weighting policy
is Dan's call (D43); the masking work is
[`masking-placement.md`](masking-placement.md).
```

(37 lines.)

---

## 3. `_plans/synth-flag-audit.md` — 271 lines, closed 2026-08-13

### 3.1 Git history and pointer

5 commits, `2f90476` (2026-08-09, written alongside the option-bundle refactor)
→ `7cfe445` (2026-09-03, a one-line note that `sort_name` was deleted).
**Last-touch `7cfe445`**; pointer `git show 7cfe445:_plans/synth-flag-audit.md`
(271 lines).

### 3.2 Commits that shipped its work

Range **`cb385ea..684d0e8`** (2026-08-10 → 2026-08-13), plus two tails:

| sha | date | what |
|---|---|---|
| `2f90476` | 08-09 | option bundles — one list instead of three (the surface the audit counted) |
| `6aa65ad` | 08-10 | dataclass-driven bundles in `stylebot.cli_options` (`opt()`) |
| `cb385ea` | 08-10 | retire the `gpt`/`local` preset path: `--generator`, `--gpt-model`, `--local-model`, `--context-window` (54 → 50) |
| `684d0e8` | 08-13 | retire `--sort`, `--report-max-rows`, `--all`, `--tag`; close the audit (→ 49) |
| `493a67c` | 08-13 | drop two dead wrappers; `openai_generator` requires its model |
| `7cfe445` | 09-03 | `ChunkPolicy.sort_name` deleted outright (D43) |

Today: `uv run ai-style synth --help` lists 50 option lines including
`--help`, i.e. **49** options — matches the file's closing count. Retired
flags confirmed absent from `src/` (`rg` for `"--sort`, `"--all"`,
`"--generator"`, `local-model`, `gpt-model`: no declarations); kept flags
confirmed present in `synth_cli.py` (`--temperature`, `--top-p`,
`--capture-reasoning`, `--assign-seed`, `--sample-frac`, `--splits`, `--role`,
`--per-generator`, `--slop-system-file`).

### 3.3 D-numbers; claims refuted

Became: **D21** (preset path retired; `--temperature`/`--top-p` kept on absent
evidence), **D43** (`sort_name` deleted), **D11**/**D12** (why `synth_key`
inputs are the retirement boundary). The 2026-08-13 decisions (retire four,
keep `--capture-reasoning` and the substrate cluster, defer `--assign-seed`)
have **no D-entry**; D21 covers only the 08-10 half and says "the remaining
open proposals: synth-flag-audit.md", which is stale — none remain.

Refuted / stale:

- The audit's class C was a retirement list of 16; 8 were retired, 7 kept, 1
  deferred. The "biggest single win" cluster 11–14 was declined because the
  masking work (D33) needs `--role styler` — which is still unbuilt (see the
  masking paragraph at the end), so the bet is still open.
- "RUNBOOK:219 claims every pair is stamped with `meta.merge_max_chars` … both
  are `None` on all 4624" — true when written, false since the 2026-08-12 v2
  batches (4,532 stamped pairs, findings).
- "`_OPTION_SPECS`" was already replaced by `cli_options.opt()` bundles when
  the file was written; the file says so itself.
- `phase-1-pair-capture.md` / `heading-context.md` drift — resolved by their
  retirement 2026-08-10.

### 3.4 Numbers, and where they live

| number(s) | verdict |
|---|---|
| 54 options vs 19 next-largest; 14 / 24 / 16 classes; → 50 | ✓ findings "Surface size" |
| → 49 (2026-08-13) | OVERVIEW:279 only ("49 options"); ✗ not in findings — add one clause to "Surface size" |
| 47 from `_OPTION_SPECS` + 7 on `synth_cmd` | ✗ unique |
| 4624 records / 4169 synthetic | ✓ findings "What has actually varied" |
| covariate table: 9 models; 6 strategies (943/896/783/567/520/460); effort off 2632 / high 1537; 190 sessions / 99 turn indices; context_mode 3212 / 957; temperature, top_p, provider_sort 1 each; context_window 5 from the registry; merge_max_chars/chunk_mode absent | ✓ findings "What has actually varied" |
| `gen.max_tokens` 2 — 8192 (4130), 16384 (39) | ✗ unique |
| `gen.replicate` `verbose1` on 460, unset on 3709 | ✗ unique (findings has "verbose 460" as a strategy count only) |
| `meta.tags` 1 — `["synthetic","paraphrase","livingthing"]` on every pair | ✗ unique |
| RUNBOOK types 11 distinct synth flags; blog wrapper 7-name exclude list, 13 overrides | ✗ unique (method evidence, 2026-08-10) |
| policy pins MIN_CHARS 200, MERGE, DROP_LIST_ITEMS, HEADING_CONTEXT immediate, CONTEXT_DROPOUT 0.1, CROSS_HEADING, MAX_TRANSFORM_SIM 0.95, MAX_LENGTH_RATIO 3.0, STUB_MARKER, STOP_HEADERS | code: blog `training_targets.py:96-322` |
| `--max-chars` default 8000; `report_max_rows` 2000 | code `targets.MAX_CHUNK_CHARS = 8000`, `synth_cli.InspectOpts.report_max_rows = 2000` |
| "All 4169 synthetic pairs were OpenRouter-served" | ✓ D21 |
| 805 pairs culled (mentioned via D32) | ✓ D32 |

**Unique numbers (verbatim):**

- "54 click options (47 from the shared `synth_cli._OPTION_SPECS` table + 7
  declared on `synth_cmd` itself …)".
- "`ai-style synth` is now a **50**-option command" → "a **49**-option command"
  (2026-08-13).
- "`gen.max_tokens` | 2 — `8192` (4130), `16384` (39)".
- "`gen.replicate` | `verbose1` on 460 pairs, unset on 3709".
- "`meta.tags` | **1 — `["synthetic","paraphrase","livingthing"]` on every pair**".
- "The blog RUNBOOK … types **11 distinct synth flags**"; "`synth_options(exclude=…,
  **overrides)` with a 7-name exclude list and 13 overrides".

Recommendation: the `max_tokens`, `replicate` and `tags` rows extend findings
"What has actually varied across the synthetic corpus" (same date, same
instrument); "49" extends "Surface size"; the 47+7 / 11 / 7 / 13 method counts
can stay in git.

### 3.5 Durable contracts still true, and the symbol

| statement | symbol now |
|---|---|
| `synth_key` = (generator, strategy, effort, prompt_id, context, target text[, replicate]); sampling and session knobs deliberately excluded; retire only non-key flags, and hardcode the current value | `synth._synth_key`; D11 |
| Recorded covariates are the analysis surface | `pairs.GEN_BASE_KEYS` / `GEN_EXTRA_KEYS` / `GEN_REPLICATE_KEYS` / `GEN_SESSION_KEYS` (`e95b12c`) |
| One option surface, declared once, each field carrying its flag | `cli_options.opt`, `synth_cli.InspectOpts` / `GenOpts` / `SessionOpts` / `SubstrateOpts` / `CorpusOpts`; blog `synth_options(exclude=…)` validates names at import |
| Retired flags survive as library knobs at their old values | `InspectOpts.report_max_rows = 2000` (`synth_cli.py:161` comment cites "synth-flag-audit C2"), `run_synth(extra_tags=)`; `sort_name` gone |
| `--slop-system-file` is the seam keeping blog prompt text out of stylebot | `synth_cli.py:229` |
| `--temperature` / `--top-p` are recorded-but-unkeyed so sampling can be swept without re-keying | `synth_cli.GenOpts` (`:271-282`); D21 |
| `--assign-seed` is a key input (arm → generator name) | `synth_cli.CorpusOpts.assign_seed` (`:401`); D12 |

The one "why" not in a D-entry: the 2026-08-13 keep/defer reasoning
(`--capture-reasoning` has no substitute; the substrate cluster is the
`--role styler` case masking needs; `--assign-seed` deferred as a key input).
Three lines; the stub can carry them, or D21 can be amended.

### 3.6 Inbound references

- `_plans/OVERVIEW.md:19` index row — "the `synth` option surface, as-audited
  evidence + open retirement proposals". Stale ("open"); update.
- `_plans/OVERVIEW.md:279` — "~~`synth` option-surface retirements~~ — closed
  2026-08-13. … `--assign-seed` deferred as a `synth_key` input
  ([synth-flag-audit.md])". Link survives to the stub.
- `_plans/findings.md:816` — "[synth-flag-audit.md]. Decided: D21." Survives.
- `_plans/decisions-archive.md:309` (D21) — "Full audit and the remaining open
  proposals: [synth-flag-audit.md]." Stale; amend to "closed 2026-08-13".
- `src/stylebot/synth_cli.py:162` — comment "lost its `--report-max-rows` flag
  in 0.3.1 (synth-flag-audit C2)". Plain text; fine, or cite the stub.

### 3.7 Draft stub

```markdown
# `ai-style synth` flag audit — retired 2026-09-04 (P7)

**What it was.** A 2026-08-09/10 classification of the 54 click options
`synth` exposed (the next-largest subcommand had 19) into load-bearing (14),
policy constants (24) and retirement candidates (16), against four kinds of
evidence: `synth_key` inputs, the covariates actually varied across 4,169
synthetic pairs, the blog RUNBOOK's typed flags, and the blog policy pin. Its
rule: retire only non-key flags, and hardcode the value they supplied. Full
text `git show 7cfe445:_plans/synth-flag-audit.md`.

**What shipped** — stylebot `cb385ea..684d0e8` (2026-08-10 → 08-13). The
`gpt`/`local` preset path went with `--generator`, `--gpt-model`,
`--local-model`, `--context-window` (D21); `--sort`, `--report-max-rows`,
`--all`, `--tag` went 2026-08-13, each surviving as a library knob at its old
value (`synth_cli.InspectOpts.report_max_rows`, `run_synth(extra_tags=)`;
`ChunkPolicy.sort_name` deleted outright 2026-09-03, D43). No cell re-keyed.
`synth` is a 49-option command; the surface is declared once in
`stylebot.cli_options` bundles. Numbers: `findings.md` "Surface size", "What
has actually varied across the synthetic corpus".

**What it refuted.** Eight of its sixteen candidates were kept or deferred:
`--temperature` / `--top-p` are unset on every pair but designed as
recorded-unkeyed covariates, and absence of evidence is not evidence (D21);
`--capture-reasoning` has no substitute as a diagnostic; the substrate cluster
`--splits` / `--role` / `--sample-frac` / `--sample-salt` is the `--role styler`
case the masking work (D33) needs; `--assign-seed` is a `synth_key` input and
outside the rule. Its "RUNBOOK claims stamped `merge_max_chars`; none are"
finding was true on 2026-08-10 and false from the 2026-08-12 v2 batches on.

**What stayed open.** Nothing in it. The substrate cluster's justification
rests on the masking work, which is
[`masking-placement.md`](masking-placement.md) and still unbuilt.
```

(32 lines.)

---

## 4. `_plans/anchor-integrity.md` — 234 lines, shipped as D32

### 4.1 Git history and pointer

6 commits, `1b08a2f` (2026-08-12, briefing written 2026-08-11 for stylebot#1)
→ `493a67c` (2026-08-13). **Last-touch `493a67c`**; pointer
`git show 493a67c:_plans/anchor-integrity.md` (234 lines).

### 4.2 Commits that shipped its work

Core range **`51e043f..d99123e`** (2026-08-12, steps 0–2 in two commits), with
the history it corrected and the tail that closed it:

| sha | date | what |
|---|---|---|
| `a72a4b2` | 07-22 | the original one-directional anchor guard (`missing_anchors`) |
| `105447c` | 08-11 | truncation told apart from anchor loss (`MIN_REWRITE_LENGTH_RATIO`) |
| `51e043f` | 08-12 | step 1: `infer.anchor_delta` — invention checked, not just loss; step 0: `--anchor-log` / `write_anchor_audit` |
| `d99123e` | 08-12 | step 2: `--drop-invented-anchors` default on; `--gate-retries` redraws instead of dropping |
| `c8ff2f8` `50653d5` | 08-12 | gate drops persisted (`lost_draws.jsonl`); gate messages blame the generator |
| `8b3591c` | 08-12 | one-off cull script (blog ran it: `dacf57892f`, 805 pairs) |
| `493a67c` | 08-13 | `missing_anchors` and a second dead wrapper deleted |
| `d8fc6e1` | 08-17 | `is_anchor_faithful` symmetric (`lose=`, `gain=`) |
| `9cb8ce3` | 09-03 | the cull becomes `ai-style data prune-invented-anchors` (D38) |
| `2bfc7b4` | 09-04 | … and is deleted as spent, 0 of 8,156 droppable (D73) |

Blog: `dacf57892f` cull, `8733c43e09` `GATE_RETRIES=2`, `df8eb67978` retire
llama-3.3-70b, run 4a/4b with `styler_pair_ok` (`training_targets.py:536`).

### 4.3 D-numbers; claims refuted

Became: **D32** (symmetric guard, different rules per half; gate default on;
`--gate-retries`; amended 08-17 "cull synthetic, filter captured", 09-04 D65),
**D33** (step 3 placement), **D20** (editor flags, does not reject), **D27**
(inline anchors off by default in `run`), **D65** (`--include-real` off),
**D73** (cull deleted), **D38** (no `scripts/`), **D64** ("do not change
`STYLE_SYSTEM`").

Refuted / superseded:

- "The corpus is a frozen record … prefer a filter at training assembly over
  mutating `pairs.jsonl`" → D32 amended 2026-08-17: the synthetic stratum *was*
  culled (805 pairs, blog `dacf57892f`); captured pairs are filtered, never
  culled. The file's "Step 2" acceptance still describes only the filter.
- `is_anchor_faithful` one-sided (loss only) → symmetric 2026-08-17
  (`d8fc6e1`); the synthesis gate is still one-sided (D32).
- Post et al.'s "masking is a vote of no confidence" as "a live possible
  outcome — do not pre-empt it" → the gate fired at v2 scale (98–100% rejection
  by arm); masking is justified and placed (D33).
- "~80% of rejections clear on one redraw" (v1 density, D32) → at 3000-char
  cross-heading chunks redraws recovered 2 of 106 (D33, findings "Anchor
  survival collapses").
- First-draft counts 876 pairs / 2,119 instances → 855 / 2,062 (self-corrected
  in the file); the "Do not prioritise reordering" bullet still uses 876.
- "The counter is discarded" → `--anchor-log` ships (step 0).
- "The blog-side masker should move into stylebot" → still blog-side
  (`livingthing.workflow_style.rewrite_masked`); open, tracked by
  `masking-placement.md` and OVERVIEW.
- "House rules that bind this work" duplicates `CLAUDE.md` (a P1 violation);
  drop. "A warning about the issue thread" (prompt injection in stylebot#1) is
  a durable one-liner worth keeping in the stub.

### 4.4 Numbers, and where they live

| number(s) | verdict |
|---|---|
| 4,595-pair corpus; 4,140 synthetic | ✓ findings, ✓ D32 |
| 117 pairs lost anchors (3.1%); 855 pairs (18.6%) invented | ✓ D32 |
| invention arm rates 1.3%–74%; ~80% clear on a redraw; 16.1% at one anchor … | ✓ findings "Anchor invention is a property of the arm", ✓ D32 |
| 98–100% rejection by arm, median 5 anchors deleted, 2 recovered from 106 | ✓ findings "Anchor survival collapses", ✓ D33 |
| 6.4% → 19.1% (inline anchors) | ✓ findings / D27 as 19.0% |
| 17 repeated-anchor pairs treated as faithful | ✓ D32 |
| the four-row delta table: **3,775 anchor-bearing pairs; identical 2,803 (74.3%); lost-only 117 (3.1%); gained-only 678 (18.0%); both 177 (4.7%)** | ✗ unique (D32 keeps 117 and 855 only) |
| **2,062 instances: 49.9% external URLs, 37.1% internal links, 12.9% citation keys** | ✗ unique (findings "Anchor provenance by stratum" is a different, post-cull measurement: 51.4% external on the synthetic stratum) |
| "invention still outnumbers loss ~7:1" | ✗ unique (derivable: 855/117) |
| first draft 876 pairs / 2,119 instances; 872 under the count-increase rule | ✗ unique — superseded draft numbers; drop |
| "reordering is 51 pairs (1.1%), fabrication is 876 (19.1%) … ~17×"; "53% of chunks carry 2+ links" | ✗ unique (from the issue thread) |
| Pangram etc. — none | — |

**Unique numbers (verbatim):**

- "Measured over the 4,595-pair corpus through the production path
  (`pairs.extract_slop` / `extract_target` → `infer.anchor_delta`): identical
  anchor multiset 2,803 (74.3% of the 3,775 anchor-bearing pairs); target lost
  anchors only 117 (3.1%); target gained anchors only 678 (18.0%); both 177
  (4.7%)."
- "855 pairs (18.6% of the corpus) carry invented anchors, 2,062 instances:
  49.9% external URLs, 37.1% internal site links, 12.9% citation keys."
- "reordering is 51 pairs (1.1%), fabrication is 876 (19.1%). Fabrication is
  ~17× more prevalent. The '53% of chunks carry 2+ links' figure is *exposure*,
  not incidence."

Recommendation: the first two are the measurement D32 cites ("117 pairs …
855 …") and belong as one table in findings under "Anchors and placeholders",
dated 2026-08-11/12, instrument `infer.anchor_delta` over the production
extractors, decided D32. The third is the priority argument from the issue
thread; one line in the same entry or nothing.

### 4.5 Durable contracts still true, and the symbol

| statement | symbol now |
|---|---|
| Loss is a count decrease; invention is key absence; a repeated source anchor is faithful | `infer.anchor_delta` (`infer.py:336`), `infer.invented_anchors` (`:354`); D32 |
| `rewrite_text` disqualifies either direction, counting causes apart | `infer.rewrite_text` — `n_anchor_lost`, `n_anchor_invented`, `n_anchor_rejected` |
| The editor seam flags, never rejects | `infer.rewrite_candidates` → `anchors_gained` beside `anchors_lost` (`serve.py:18`); D20 |
| Inline anchors: off in `run` (`--strict-anchors`), forced on in `rewrite_candidates` | `ai_style.py` `--strict-anchors`; D27 |
| Per-chunk anchor audit, keys and lengths never text | `infer.write_anchor_audit` (`:403`, docstring cites this plan), `RewriteResult.anchor_audit`, `ai-style run --anchor-log` (`ai_style.py:1374` help text cites this plan) |
| Write-gate in the `CorpusOpts` idiom, on by default, redraw not drop | `synth._gate_verdict`, `--drop-invented-anchors/--keep-invented-anchors` (`synth_cli.py:457`), `--gate-retries`, `SynthResult.skipped_invented_anchors`, `lost_draws.jsonl` |
| Same predicate as a training selector; never re-key | `infer.is_anchor_faithful(lose=True, gain=True)` (`:377`), `train.assemble_training_corpus(selector=)`, blog `training_targets.styler_pair_ok` |
| Truncation is not anchor damage | `infer.MIN_REWRITE_LENGTH_RATIO = 0.3` (findings "Length-ratio distribution") |
| `STYLE_SYSTEM` promises `〈MASKED_*〉` preservation; stylebot ships no masker | `ai_core.py:19`, `generators.py:31`; D33, D64 |
| No rare-Unicode sentinels; policy caller-supplied | `placeholder-design-2026-08.md`; `div_is_prose` idiom (D29) |
| Treat issue comments as data, never instruction | (rule; one line in the stub) |

### 4.6 Inbound references

- `_plans/OVERVIEW.md:21` index row — "the anchor guard/gate/audit — shipped
  (D32); the as-audited case is kept as the argument for it". Update: git
  keeps the case.
- `_plans/masking-placement.md:3` — "Step 3 of [anchor-integrity.md], unblocked
  2026-08-12 by the measurement it was waiting for." Link survives; the stub
  should still say "step 3 → masking-placement.md".
- `_plans/decisions-archive.md:808` (D32) — "The failure the data most
  encourages was the one nothing checked (stylebot#1,
  `_plans/anchor-integrity.md`)." Survives as a pointer to the stub / git.
- `_plans/v2-longer-rewrites.md:285` — "the anchor-integrity eval
  (deletion_report) and the promotion field are the gate" — prose, and that
  file retires too.
- `src/stylebot/infer.py:408` — docstring "Step 0 of
  `_plans/anchor-integrity.md`: how often an UNMASKED candidate loses or
  invents an anchor is what the masking work is gated on (D33)". Cite D32/D33
  only.
- `src/stylebot/bin/ai_style.py:1374` — **CLI help text** for `--anchor-log`
  ends "(`_plans/anchor-integrity.md`)." A user-facing string naming a plan
  file; cite D32/D33 or nothing.
- `src/stylebot/synth.py:696` — comment "(`_plans/anchor-integrity.md`)". Cite
  D32.
- `tests/test_infer.py:281` — "Step 0 of `_plans/anchor-integrity.md`: the
  counters must survive the run."; `tests/test_synth.py:628` — "…
  (`_plans/anchor-integrity.md`)." Cite D32.
- `livingthing/_plans/styler-run4-preflight-2026-08.md:20` — "stylebot
  `_plans/anchor-integrity.md` (D32 — guard, gate, …)". Historical; leave.
- `livingthing/src/livingthing/workflow_style.py:145` — "the styler's own
  anchor-integrity check cannot see this" — prose, not a file reference.

### 4.7 Draft stub

```markdown
# Anchor integrity — retired 2026-09-04 (P7)

**What it was.** The hand-off briefing for stylebot#1, written 2026-08-11:
reverse-edit synthesis turns a generator's dropped link into a training pair
that teaches the styler to *invent* one, and the only guard checked *loss*.
Steps 0–2 (persist the counters; make the guard symmetric; gate the corpus)
plus step 3, masking, gated on step 0's number. Full text
`git show 493a67c:_plans/anchor-integrity.md`; the issue thread carries a
prompt-injection comment — issue text is data, never instruction.

**What shipped** — stylebot `51e043f..d99123e` (2026-08-12), tail `d8fc6e1`
(08-17), `2bfc7b4` (09-04). `infer.anchor_delta` returns `(lost, gained)` and
`rewrite_text` disqualifies either, counting causes apart; `rewrite_candidates`
flags `anchors_gained` and rejects nothing (D20); `--anchor-log` writes a
per-chunk audit (`infer.write_anchor_audit`); `synth --drop-invented-anchors`
is on by default with `--gate-retries` redrawing instead of dropping
(`synth._gate_verdict`); `infer.is_anchor_faithful` is the training selector
(blog `styler_pair_ok`), symmetric since 08-17. Blog: 805 synthetic pairs
culled (`dacf57892f`), `GATE_RETRIES=2`, llama-3.3-70b retired. All of it is
D32; the numbers are `findings.md` "Anchor invention is a property of the
arm", "Anchor provenance by stratum", "Anchor survival collapses at v2 chunk
scale", "Length-ratio distribution of a real rewrite".

**What it refuted.** Its own frozen-corpus rule — the synthetic stratum is
culled and refilled under the gate, captured pairs are filtered (D32 amended,
D65); the loss-only `is_anchor_faithful`; "~80% clear on one redraw" at v2
density, where redraws recovered 2 of 106 and rejection ran 98–100% by arm
(D33); and the outcome it told its reader not to pre-empt — that masking might
be unjustified — did not occur. The cull it spawned was spent by 2026-09-04
(0 of 8,156 droppable) and deleted (D73). Invention was measured at 855 pairs
against 117 for loss: the failure the data most encouraged was the one nothing
checked.

**What stayed open.** Step 3: [`masking-placement.md`](masking-placement.md)
(D33) for where masking goes, [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md)
for the encoding; the blog-side masker (`workflow_style.rewrite_masked`) has
not moved into stylebot, and the synthesis gate is still one-sided.
```

(37 lines.)

---

## 5. `_plans/vscode-marker.md` — 177 lines, "✅ BUILT"

### 5.1 Git history and pointer

6 commits, `6e49cce` (2026-07-06, "consolidate plans in stylebot") →
`ff0e994` (2026-09-03). **Last-touch `ff0e994`**; pointer
`git show ff0e994:_plans/vscode-marker.md` (177 lines). Status layers:
2026-07-06 (design + threshold calibration), 2026-09-03 (`bb50a86`, segmenter
removed; `ff0e994`, counts out).

### 5.2 Commits that shipped its work

Range **`4daf001..bb50a86`** (2026-07-06 → 2026-09-03):

| sha | date | what |
|---|---|---|
| `4daf001` | 07-06 | `ai-style serve` sidecar + `vscode-voice-marker/` 0.0.1 |
| `03d7cec` `d5dadd8` | 07-06 | `.vsix` packaging, MIT licence |
| `2210f36` `7f57286` | 07-22 | `rewrite` / `prefer` ops; 0.0.2, 0.0.3 (candidates + diff preview) |
| `f5abafc` | 07-23 | 0.0.4: `offsets`, `para_before`/`para_after` in preference records |
| `474e215` | 07-28 | 0.0.5: per-request timeouts; a wedged sidecar restarts |
| `3d4169b` `b932f78` `66baf75` `5f05840` | 08-10 | marker polish; cross-language fixture pinned the TS port (D28) |
| `bb50a86` | 09-03 | 0.0.6: `segment` op in the sidecar; TS port, its test and fixture deleted (D44) |
| `dba257b` `a3b094a` | 09-04 | `detect_many` moves to `classify`; `serve` stops importing `eval` (D59) |
| `eb98117` | 09-04 | a half-selected `:::` div is refused like a whole one (D76, via `rewrite_candidates`) |

Extension is at `"version": "0.0.6"` (`vscode-voice-marker/package.json:5`);
`src/` holds `decorations.ts`, `extension.ts`, `sidecar.ts` — no `segment.ts`.
Blog: thresholds in `.vscode/settings.json:206`; install/skill
`.claude/skills/voice-marker-install/SKILL.md` (says 0.0.6 needs a
post-2026-09-03 sidecar).

### 5.3 D-numbers; claims refuted

Became: **D20** (human picks; NDJSON not LSP; presentation and selection bias
of the preference corpus), **D44** (editor asks the sidecar where the prose
is), **D7** (the detector), **D28/D30 superseded** (one segmenter),
**D29** (`div_is_prose` as the sidecar's policy), **D59** (`detect_many` in
`classify`), **D76** (unclosed/half-selected div refused).

Refuted / stale:

- "prepend the nearest heading" — wrong, not deferred (file says so; the
  classifier trained on `pair_body`).
- A TypeScript segmenter kept in parity by fixture (D28) → deleted; measured
  1,728 identical / 29 corrections, plus 385 callout paragraphs it could not
  see (D44, findings "Segmenter implementations").
- The latency worry → non-issue (5.0 s cold, 34 ms/doc; the `segment` hop
  0.30–4.20 ms).
- "The shipped head was fit on all Dan's posts (`meta.split.mode = 'fit_all'`)"
  → the current head is fit under `splits.json` (`splits_sha256` /
  `split.eval_version` in the artifact, findings "eval_v2"); the "new drafts
  are the honest case" caveat is weaker than written. `fit_all` appears nowhere
  in findings.
- "recalibrate after retraining the head" — the head was refit 2026-08-17 and
  2026-09-03; the blog thresholds are unchanged. Open.
- Two done-criteria are still unchecked: the eyeball check of the repackaged
  0.0.6 `.vsix` in the Extension Dev Host, and "the loop closes" (a flagged
  paragraph rewritten and captured by `ai-style log` with no extra steps).

### 5.4 Numbers, and where they live

| number(s) | verdict |
|---|---|
| cold load 5.0 s; 30-paragraph notebook 34 ms | ✓ findings "The voice classifier" |
| `segment` hop table (5,428/9/0.30 ms … 116,774/212/4.20 ms; in-process 0.28/0.50/0.90/3.71) | ✓ findings "The sidecar `segment` op" |
| 264 pairs; distribution in ≈[0.32, 0.68]; Dan p5/p50/p95 .32/.46/.62; slop .40/.55/.68 | ✓ findings "The voice classifier" |
| thresholds `[0.51, 0.55, 0.59, 0.65]` | ✓ findings |
| AUC 0.72 / pairwise 0.78 | ✓ findings, D7 |
| 1,728 identical / 29 corrections; 385 callout paragraphs | ✓ findings "Segmenter implementations", D44 |
| ~30 lines a side | ✓ D20 |
| 500 ms debounce | ✓ findings |
| warm: **1 paragraph 7 ms**; **head dot-product 28 µs** | ✗ unique |
| Dan **p25 .40 / p75 .51**; slop **p25 .47 / p75 .59** | ✗ unique (findings keeps p5/p50/p95 only) |
| edge rationale: Dan-p75, slop-p50, slop-p75, ≈slop-p95 | ✗ unique (the mapping, derivable from the row above) |
| real-model smoke: banned-word slop **0.61** vs a Dan sentence **0.48** | ✗ unique |
| alternative B: **110–440 MB** bundle; "no 240 MB model" | ✗ unique (sizing of an unbuilt alternative) |
| `voiceMarker.timeouts` defaults (info 60000, segment 30000, score 30000, rewrite 180000, prefer 15000) | code `package.json:104` |

**Unique numbers (verbatim):**

- "warm: 1 paragraph | 7 ms"; "head dot-product → `P(slop)` | 28 µs"
  (2026-07-06, StyleDistance on Dan's Mac, MPS).
- "| percentile | 5 | 25 | 50 | 75 | 95 | / | Dan side | .32 | .40 | .46 | .51
  | .62 | / | slop side | .40 | .47 | .55 | .59 | .68 |" (264 pairs, 2026-07-06).
- "Shipped edges `[0.51, 0.55, 0.59, 0.65]` = Dan-p75 (below: unmarked, where
  3⁄4 of Dan's own prose sits), slop-p50, slop-p75, ≈slop-p95 (hot)."
- "real-model smoke test scores banned-word slop 0.61 vs a Dan sentence 0.48"
  (2026-07-06).
- "a 110–440 MB bundle" (ONNX alternative B, estimate).

Recommendation: fold the 7 ms / 28 µs and the p25/p75 columns plus the edge
rationale into findings "The voice classifier" (same date, same instrument;
the entry already carries the p5/p50/p95 and the thresholds — the rationale is
what makes the thresholds re-derivable after a refit). The smoke and the
bundle estimate can stay in git.

### 5.5 Durable contracts still true, and the symbol

| statement | symbol now |
|---|---|
| Protocol: NDJSON over stdin/stdout; ops `info` / `score` / `segment` / `rewrite` / `prefer`; malformed → `error`, never a crash | `serve.serve_loop` (docstring `serve.py:8-20`), `ai-style serve --detector-model DIR` |
| Scores over the wire, not vectors; parity with `ai-style eval` | `classify.sklearn_detector`, `classify.detect_many` |
| No heading prefix — the classifier trained on body text | `pairs.pair_body` |
| One segmenter; prose = maximal run of `line_kinds` prose; frontmatter dropped; nearest heading as `context`; div policy is the sidecar's | `serve.prose_paragraphs` over `stylebot.markdown` (`serve.py:63`); `dan-style serve` passes `div_is_prose`; D44 |
| Offsets are UTF-16 code units, converted server-side | `serve._utf16_mapper` (`serve.py:43`), `sidecar.ts:20` |
| Pinned under both policies on the parity fixture | `tests/test_serve.py` (21 tests) on `tests/fixtures/protection_parity.qmd`; blog `tests/test_dan_style_cli.py` (callout paragraph) |
| Extension owns the lifecycle, dormant without `voiceMarker.command`; activates on `markdown` + `quarto` | `extension.ts`, `sidecar.ts`; `package.json:20-21,55` |
| Settings: `command`, `cwd`, `thresholds`, `minChars`, `debounceMs`, `enabled`, `bestOf`, `timeouts` | `package.json:55-104`; blog `.vscode/settings.json:206` |
| Graded tint + overviewRuler + hover, ~5 warped buckets from calibrated quantiles | `decorations.ts`; findings "The voice classifier" |
| Soft signal; eyeball is the arbiter; short paragraphs skipped below `minChars` | D7 guardrail, D20 |
| The marker is the front-end of the capture loop | `ai-style log` tasks (`_training_pairs/README.md` blog-side) |
| Parked alternative B (in-extension ONNX) is the list D44 names | D44 "re-evaluate if" cites this file's alternative B |

### 5.6 Inbound references

- `_plans/OVERVIEW.md:20` index row — "the editor marker design, thresholds,
  honesty caveats"; `:194` phase map — "✅ built — sidecar + extension … the
  extension segments through the sidecar's `segment` op … (D44)". Update the
  index description; the map link survives.
- `_plans/decisions.md:374` (D44) — "**Re-evaluate if** the marker ever ships
  as a `.vsix` with no sidecar — this entry is then the list of what a
  TypeScript segmenter must get right (`_plans/vscode-marker.md`, alternative
  B)." The stub must keep a one-line "alternative B" or D44 loses its pointer.
- `README.md:132` — "Design and caveats: [_plans/vscode-marker.md]"; `:194` —
  "[_plans/vscode-marker.md] §'Threshold calibration'". The section name will
  vanish; repoint `:194` to findings "The voice classifier".
- `vscode-voice-marker/README.md:6` — "caveats: [../_plans/vscode-marker.md]".
  Survives.
- `vscode-voice-marker/src/extension.ts:7` — "See stylebot/_plans/vscode-marker.md."
  Survives; or cite D20/D44.
- `src/stylebot/serve.py:6` — "a live editor marker viable
  (`_plans/vscode-marker.md`; NDJSON not LSP, D20)". Already cites D20; fine.
- `livingthing/_training_pairs/RUNBOOK.md:157` — "(`stylebot/_plans/vscode-marker.md`)".
  Survives as a path.
- `livingthing/.claude/skills/voice-marker-install/SKILL.md:19` — "…
  retraining, see `stylebot/_plans/vscode-marker.md`)" about recalibrating
  thresholds. Repoint to findings "The voice classifier" (where the quantiles
  and edges live) once the rationale moves there.

### 5.7 Draft stub

```markdown
# VS Code paragraph marker — retired 2026-09-04 (P7)

**What it was.** Mark paragraphs dan-or-bot live in the editor with the
trained voice classifier — a graded `P(slop)` tint, ruler mark and hover,
"Rewrite in Dan's Voice" on a selection — as the front-end of the capture
loop, not a gadget. Designed and calibrated 2026-07-06. Full text
`git show ff0e994:_plans/vscode-marker.md`.

**What shipped** — stylebot `4daf001..bb50a86` (2026-07-06 → 2026-09-03):
`ai-style serve` (`serve.serve_loop`: NDJSON over stdin/stdout, ops `info` /
`score` / `segment` / `rewrite` / `prefer`, malformed input answered, never a
crash) wrapping `classify.sklearn_detector`; the extension
`vscode-voice-marker/` at 0.0.6 (`extension.ts`, `sidecar.ts`,
`decorations.ts`), dormant unless `voiceMarker.command` is set, sidecar
lifecycle with capped-backoff restart and per-op timeouts, preference records
carrying `offsets` and `para_before`/`para_after`. Segmentation is one
implementation, `serve.prose_paragraphs` over `stylebot.markdown` with the
sidecar's `div_is_prose`, UTF-16 offsets converted server-side, pinned by
`tests/test_serve.py` under both policies (D44). Scores over the wire, not
vectors; bare body text, no heading prefix (`pairs.pair_body`). The blog
contributes the artifact, `dan-style serve`, and the thresholds
`[0.51, 0.55, 0.59, 0.65]` in its `.vscode/settings.json`. Numbers:
`findings.md` "The voice classifier", "The sidecar `segment` op", "Segmenter
implementations".

**What it refuted.** Latency was the only worry and is nothing (5 s once, 34
ms a document, the `segment` hop under 5 ms on the largest post). Prepending a
heading was wrong, not deferred. The TypeScript segmenter kept in parity by a
fixture (D28) had drifted on 29 posts and could not see 385 callout
paragraphs; it is gone (D44). LSP has no decoration concept (D20). The
preference corpus is biased by presentation and selection, and logged so
(D20). The head is no longer `fit_all` — it is fit under `splits.json` — so
"fresh drafts are the honest case" is weaker than written.

**What stayed open.** Two done-criteria never got their eyeball: the
repackaged 0.0.6 `.vsix` in the Dev Host, and the loop closing through
`ai-style log`. The thresholds predate two refits of the head and want
recalibrating from `scores.jsonl` quantiles; "rewrite section" is allowed by
`rewrite_candidates` and not yet offered. Alternative B — in-extension ONNX,
no sidecar — is parked; D44 is what a port must get right.
```

(40 lines.)

---

## 6. Are `masking-placement.md` and `placeholder-design-2026-08.md` really open?

Yes, both. stylebot ships no masker: `rg -n -i mask src/stylebot/` finds only
the frozen prompt clause (`ai_core.py:19`, `generators.py:31`), the per-line
*kind* mask in `markdown.line_kinds` / `infer._segment` (a different sense),
and comments citing D33; `train.py` has no masked-fraction transform (the only
dropout in the tree is `synth._effective_context`'s heading dropout), and
`is_anchor_faithful` / `_gate_verdict` operate on raw anchors. The masker that
exists is blog-side and inference-only — `livingthing.workflow_style.rewrite_masked`
with `_to_masked_names` / `_sentinel_guard` over `pandoc_mask` — exactly the
generic/policy inversion `anchor-integrity.md` and D33 name. Of
`masking-placement.md`'s ordered work (0: refit the classifier on masked text;
1: synthesis masking with multiset validation; 2: the masked fraction at
assembly; 3: inference in-distribution) none has a commit, and
`voice_classifier.py` / `training_targets.py` contain no "mask" at all.
`OVERVIEW.md:22-23` and its open-slice bullet (":236-248") list both as open;
the OVERVIEW bullet still says "store mostly-masked", which `891df4e` and D33
reversed to "canonical and unmasked on disk, mask at consumption" — worth one
edit there. So P7 does not apply: nothing shipped. What P1 would say is that
`masking-placement.md`'s evidence tables are already in findings ("Anchor
survival collapses at v2 chunk scale", "Placeholder retention across models")
and its decision is D33, so the file could shrink to its work list; and that
`placeholder-design-2026-08.md` has a twin in the blog
(`livingthing/_plans/placeholder-design-for-parsers-2026-08.md`) whose numbers
are in stylebot's findings — a duplication to note, not to fix here.

---

## 7. Cross-cutting notes for the stub-writer

- Every stub above fits the model's four parts and ≤ 40 lines; each names its
  `git show <sha>:_plans/<file>` pointer, a commit range, D-numbers and a
  successor. Where a range spans the whole repo (eval-harness) the stub names
  the bounding commits and the stub-writer should keep the table in §1.2 out
  of the stub — git holds it.
- Numbers marked ✗ unique that are worth keeping go to `findings.md` under the
  entry named in each "Recommendation"; P4 wants them as rows on the existing
  dated table, not new sections. Forecasts and first-draft counts are named so
  they can be dropped deliberately.
- Three "why"s live only in the retiring files and have no D-entry: the
  gap-coalescing-not-windowing rule (`capture.py:111` cites the plan), the
  three honest ways to analyse the nested scale arms (`findings.md:729` cites
  the plan), and the 2026-08-13 keep/defer reasoning for synth flags. Each is
  ≤ 5 lines; D43, D18-adjacent, and D21 are the natural homes.
- Ten source docstrings/comments/help strings cite a plan file by section
  (`eval.py:7,13,407`, `splits.py:3`, `infer.py:24,408`,
  `classify_train.py:548`, `synth.py:122,696`, `capture.py:111`,
  `ai_style.py:1374`, plus `tests/test_infer.py:281`, `tests/test_synth.py:628`,
  blog `styler.py:385`, `training_targets.py:183`). P1 says a docstring cites a
  D-number; the repoint targets are listed per file above.
- Nothing here needs `git show` of a sha older than the last-touch sha; each
  file's history is linear and the last touch is the fullest version.
