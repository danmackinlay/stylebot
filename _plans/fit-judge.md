# FitCase judge — a coherence signal for incremental rewrites

**Status: BUILT 2026-08-21 in an unmerged worktree, with the first verdict in —
a wash, and a positive control saying the instrument is not the reason. Ported
to main 2026-09-10 under D97.** Numbers in [`findings.md`](findings.md); code is
`stylebot.fit` (the cases) + `stylebot.fit_judge` (the comparison) + `ai-style
judge-fit` + `tests/test_fit.py`. What follows is the design as built; the two places
reality corrected it are marked **AS BUILT**. Opened after Dan's dissatisfaction
with styling *and coherence*, and the observation that the harness has no
instrument that could register a coherence failure at all.

## The gap

The four signals all score a passage **in isolation**:

| signal | unit | sees neighbours? |
|---|---|---|
| detector (`stylebot.classify`) | paragraph-scale embedding + logistic head | no |
| LLM judge (`eval.JUDGE_SYSTEM`) | one passage → `{score 1-5, rationale}` | no |
| Vale | line-scale rules | no |
| eyeball | the `output` field, passthrough | only if Dan opens the file |

So a rewrite that reads like Dan but restarts a metaphor the previous paragraph
was still running, or strands a pronoun whose referent it renamed, scores
**clean on every instrument**. Coherence is not scored badly; it is outside the
range. Three signals, one axis: local voice.

This is the measurement gap under
[`retired/v2-longer-rewrites.md`](retired/v2-longer-rewrites.md) step 5 (condition
on neighbouring prose), whose successor is
[`coherence-context.md`](coherence-context.md). Building that without this first
buys an improvement no number in the harness can see.

## Why NOT a whole-document judge

The obvious instrument — feed the whole rewritten post, ask for 1–5 — was
considered and rejected. Reasons, in order of weight:

1. **It is not paired.** A document score cannot be attributed to any chunk
   decision, so it cannot rank two adapters the way the paired detector
   comparison in [`findings.md`](findings.md) does.
2. **n collapses.** The frozen eval stratum is 127 chunks across a handful of
   posts. Per-document scoring turns n=127 into single digits, against an
   effect size the paired detector CI already puts at ~0.04.
3. **It measures a mode Dan does not work in.** The live workflow is *not*
   whole-document rewriting. It is: select a span in VS Code, rewrite it,
   accept or reject (`vscode-voice-marker`, `dan-style serve`). The unit of
   work is a span with fixed prose on both sides of it.

## The design

**Unit of judgment is a boundary, not a document.** Input is
`(para_before, span, para_after)` with **two variants of the span**; output is a
forced choice plus a one-sentence reason. Forced choice rather than an absolute
1–5: LLM Likert is noisy, and "does this fit its neighbours" is inherently
comparative. The question asks only about fit — referents resolved, register
continuous, no metaphor restarted, no term silently redefined — and explicitly
**not** about voice, which the existing judge already scores.

Three properties fall out of the workflow, and two of them correct the
autoregressive framing this work started from:

- **The neighbours are Dan's own prose, not styler output.** In the editor path
  the context on both sides is fixed, human, already in voice. The exposure-bias
  hazard (train on gold prefixes, serve on the model's own degraded output)
  largely does not apply here. It still applies to whole-file `ai-style run`.
- **It is two-sided.** An incremental rewrite must fit a paragraph that already
  exists *downstream*. A sliding-window / autoregressive design conditions on
  `para_before` only, and would therefore address half of the real constraint.
  **Step 5's context slot should be two-sided.**
- **The data is already being logged.** `vscode-voice-marker/src/extension.ts`
  (≥0.0.4) records `para_before` / `para_after` on every preference record,
  beside `chunk`, `candidates`, `scores`, `chosen`, `kept_input`. Every editor
  accept/reject Dan has made is already a labelled fit case — for eval now,
  and for preference training later (Phase 7).

**It stays offline.** A fit judgment is ~3 paragraphs of prompt and an
OpenRouter round-trip. The rewrite path is already best-of-N reranked by the
detector at ~34 ms/document; an LLM call in the keystroke path would be felt.
The fit judge is an eval signal and a source of preference labels, never an
inference-time reranker.

## Two strata, and only one of them yields a verdict

- **The frozen fit stratum** — the verdict stratum. Constructed from
  `splits.json` eval posts: cut each at the *serving* diet (`chunk_diet`), and
  for each prose chunk keep the flanking prose as `para_before`/`para_after`.
  Deterministic, re-derivable, adapter-independent. This is what an A/B verdict
  is read off.
- **The logged preference records** — tracking, not verdicts. Whatever Dan
  happened to edit, self-selected and drifting with his habits. Useful for
  "is this getting better in practice" and as preference data; useless as a
  controlled contrast.

## Inputs / outputs

**Inputs**
- The corpus (`--draws` plus the view policy, or an explicit view directory) and
  `splits.json`, both under the blog's `_training_pairs/` — the stratum source.
- Blog post `.qmd` under `--blog-root` — the flanking prose comes from the
  source document, not from the corpus (a pair record does not carry its
  neighbours).
- Two rewrite JSONLs to contrast (`ai-style run` corpus-mode `output` field),
  or one plus the input chunk for a do-no-harm reading.
- `OPENROUTER_API_KEY` for the live judge; a fake judge for tests.

**Outputs**
- `fits.jsonl` — one id-keyed record per boundary: `id` (same join key as
  `scores.jsonl`), `para_before`, `span_a`, `span_b`, `para_after`, `choice`
  (`a` | `b` | `tie`), `rationale`, and the arm labels. Append-only,
  resumable, orphan-pruning — the `score_pairs_file` contract, reused.
- A paired summary: win/loss/tie counts and a CI, in the shape the run-4 table
  in [`findings.md`](findings.md) already uses.

**Position order must be randomised per record and recorded**, or the judge's
position bias is folded into the verdict. This is the one thing that makes the
number worthless if skipped.

## AS BUILT — two corrections

**1. Neighbours come from the corpus, not from matching text in the post.**
The design said "locate the span by matching its Dan target verbatim", on the
reasoning that a synthetic pair's target *is* published prose. Measured: that
places **22 of 127** records. Posts drift after generation, and a merged
cross-heading chunk depends on separators a later edit does not preserve. What
works is a **sibling index** — for a synthetic pair, `chunk_index` is the chunk's
position in its source post, so the adjacent index IS the adjacent prose, exactly
and with no matching. Three sources, best first, and which one was used is
recorded on every case (`FitCase.neighbour_source`):

1. `sibling` — the adjacent chunk's Dan target from the built view. Exact.
   **Synthetic pairs only**: `capture.diff_chunks` emits only *changed regions*
   with no offsets, so consecutive capture indices can sit pages apart, and
   treating them as adjacent would fabricate the thing being measured.
2. `archive` — content-matched in `captures/<id>/after.md`, the document as it
   stood at capture time. Covers captures made since 2026-07-22.
3. `post` — content-matched in today's file. Last resort; it drifts.

Result: 62 placed (43/7/12), 65 dropped — almost all pre-archive captures, which
are unrecoverable by any route.

**2. A positive control is part of the instrument, not an optional extra.**
Absent from the design, and without it the first verdict would have been
uninterpretable: a judge that ties everything produces "wash" whether or not the
arms differ. `--field-a target --field-b slop` runs Dan's own prose against the
slop it replaced *at the same boundaries*, and must come out decisive. It does
(31–12), and the tie rate separates cleanly (31% control vs 71% adapters). **Run
the control alongside every verdict**, not once.

## Done-criteria

- [x] FitCase-stratum constructor — two of them, in the end: `fits_from_document`
      (deterministic document cut, for a frozen stratum built from posts) and
      `fits_from_rewrites` (the three-source recovery above, for contrasting
      two arms' existing rewrites). Both tested against a fixture where a fence,
      display math and a blockquote sit between two paragraphs — none may become
      a neighbour — and where a heading walls off adjacency.
- [x] `FitJudge` interface (`(before, first, second, after) -> {choice, rationale}`)
      + `openrouter_fit_judge`, mirroring `Judge` / `openrouter_judge`. Keyless
      tests with a fake. An unparseable reply falls back to **tie**, never a
      side: a coin-flip fallback would inject noise straight into the counts.
- [x] `ai-style judge-fit` — its own subcommand, not an `eval` flag, because
      the unit is a boundary and `score_pairs_file`'s record is a pair.
      Resumable, position-randomised (deterministically, from the case id, so a
      resumed run presents a case the way it did before), `--dry-run` to price
      and inspect the stratum with no key and no spend.
- [x] First verdict: run 4a vs run 2 over the frozen stratum — **wash**
      (8–10, 44 ties, CI [0.246, 0.663]) with the control decisive
      ([`findings.md`](findings.md)).

## Why the first verdict matters more than usual

Run 4a was promoted 2026-08-21 on a paired detector gain whose CI excludes zero
— but Dan's side-by-side eyeball audition read it as a **wash**. His eye is
currently the only instrument disagreeing with the harness, and it neither
scales nor leaves a record. The fit judge is an attempt to make what his eye is
doing legible and repeatable.

**Answered 2026-08-21: wash, and not because the judge is blind.** It agrees
with Dan against the detector. The control settles the ambiguity the design
left open — the judge has resolution and did not use it, so coherence genuinely
did not move between run 2 and run 4a. Neighbour conditioning
([`coherence-context.md`](coherence-context.md)) is the remaining lever, and it
now has a number to beat: 8–10 with 44 ties, against a control that separates
31–12.

**Ported to main 2026-09-10 (D97).** The port is the same instrument on today's
seams: `fit.py` keeps the case constructors and `fit_judge.py` the comparison
and summary (P2's 800 lines), the judge awaits rather than fanning out to
threads (D56/D59), the corpus is `--draws` plus a view policy rather than a
`pairs.jsonl` (D84), and the archive id is `meta.archive_id or capture_id`
(D96). Still open, both small: the frozen document-cut stratum
(`fits_from_document` exists; nothing builds it from `splits.json`), and
replaying the editor's `para_before`/`para_after` records as a tracking series.
