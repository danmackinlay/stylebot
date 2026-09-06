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
composition", "eval_v2" (with the document-scale input size), "Document-scale
eval", "Pangram cross-check", "Threading audit".

**What it refuted.** The judge cannot see real slop (3/20, 14 ties) and never
arbitrates two adapters; run 4a's "real gain" was a chunk-level interval and
straddles zero clustered by post (D40); a whole-document embedding has no
resolution (D54); "built and tested, not run" Pangram had a dead endpoint (404)
and ran partially; a thread pool around the detector bought nothing (D55, D57
→ D59). Its Goodhart guardrail — detector movement alone is never success —
stands, in `stylebot.eval`'s docstring and D7.

**What stayed open.** The fifth capability, a synthetic↔real distribution
match over the `meta.synthetic` facet, is unbuilt; its spec is stage 0 of
[`argued-slop-arm.md`](../argued-slop-arm.md). D53's open half: `extend-eval`
does not prune the synthetic pairs of the posts it freezes.
