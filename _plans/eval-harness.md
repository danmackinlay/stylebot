# Eval harness — 🔧 BUILT (library + CLI; detector audition pending)

How we know the styler works. Runs **offline** — it scores candidate output and
is *not* wired into the trainer or baked into the served adapter. Built early and
in parallel: it's the ground truth every later phase reports against, and it
needs only sample prose, not a trained model.

**Status (2026-06-28):** `stylebot.eval` ships the four-signal scorer as typed
library functions + the thin `ai-style eval` CLI. Runs keyless by default (Vale
+ null judge/detector); `--judge` scores voice via OpenRouter. Three of four
signals are live; the **statistical-detector audition is the one remaining
decision** (deliberately deferred — GPU/$$-heavy, needs the operator). See
"Status detail" at the bottom.

## Inputs

- **A `pairs.jsonl` corpus** (the pipeline lingua franca) — eval scores the
  *fields* of each pair: `slop` (`messages[1]`) and `target` (Dan, `messages[2]`),
  with the shared heading-context prefix stripped so only the transform region is
  judged. (One-prose-per-file is gone — we never have that.) A Phase-4 styler run
  scores the same way once it emits an `output`-bearing JSONL (add an `"output"`
  field extractor then).
- `PANGRAM_API_KEY` (only if the paid detector wins the audition below).
- `OPENROUTER_API_KEY` for the LLM judge (one key, many models).
- Vale + an optional Vale config (a parameter — never a hardcoded blog ruleset).

Interface: `stylebot.eval` functions over `pairs.jsonl` (`score_pairs_file` →
id-keyed `scores.jsonl`; `summarize_scores` for the aggregate view); `ai-style
eval` is the thin CLI. Paths resolved via `stylebot.config`.

## The four signals

1. **Vale** — mechanical slop (banned words, indefinite "you", -ize spelling).
2. **LLM-as-judge** — "is this Dan-shaped" at the voice level.
3. **Statistical detector** — "would an external classifier flag this".
4. **Dan's eyeball** — the veto channel (not automatable; reported alongside).

## The detector decision (the one real choice)

Run an **audition before paying**: test open-weights detectors —
[Binoculars](https://github.com/ahans30/Binoculars),
[Ghostbuster](https://github.com/vivek3141/ghostbuster),
[RADAR](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B), OpenAI's
deprecated RoBERTa — against [Pangram](https://www.pangram.com/) on the small
test corpus, in the regime that matters: **single-paragraph, lightly-humanised
text**. Prior: none of the free ones will be good enough at *this* regime, but
the audition is one afternoon.

- If an open-weights detector tracks Pangram well enough → use it, skip the spend.
- If Pangram wins → make it cheap via **distillation**: ~$50 once to label
  ~10,000 paragraphs, then train a small local classifier (DistilBERT-class,
  <100ms) on those labels. That classifier becomes a free reward signal
  callable anywhere on our side (eval, best-of-N at inference, a future DPO
  loop) — and never touches the served adapter. One spend, no more API calls.

### Pangram — captured, NOT implemented (cost discipline)

Pangram is the paid reference the audition benchmarks against. Captured here so
the integration is *designed* before any spend; **do not wire it yet.**

- **API surface** (`pip install pangram-sdk`, key `PANGRAM_API_KEY`):
  - Realtime: `Pangram().predict(text)` → async task under the hood; returns
    `stage`, `fraction_ai`, `fraction_ai_assisted`, `fraction_human`,
    `num_ai_segments`, per-window `windows[{label, ai_assistance_score,
    confidence}]`, `prediction_short`, optional `dashboard_link`. (REST:
    `POST /task` → returns id → poll `GET /task/<id>` until `STAGE_SUCCESS`.)
  - **Bulk API** for async throughput (`submit_bulk(items=[{id, text}])` →
    `wait_for_bulk(id)` → `get_bulk_results[_page](id)`. Async, paged). Pick it
    for *throughput/convenience on a big one-shot job, not for cost* — see below.
  - (Also a plagiarism endpoint — irrelevant to us.)
  - Docs: <https://docs.pangram.com/quickstart>.
- **Price: realtime $0.05 / 1,000 words; bulk only ~20% cheaper (~$0.04/1k) —
  not transformative.** So batching changes throughput, *not* the calculus: it's
  still *metered per call* and **vastly pricier than slop generation** — a
  ~120-word passage costs ~$0.006 to *detect* (bulk ~$0.005) versus ~$0.0002–0.001
  to *generate* on the cheap bulk models, and ∞× a free local detector. The
  asymmetry that bites: a generator is paid **once per pair**; a
  detector-as-reward is paid **every time you score**.
- **Therefore use it sparingly, in exactly two bounded one-shot ways — never in a
  hot loop:**
  1. **Audition reference** (do this — negligible): label only the ~30-paragraph
     test corpus once to benchmark the free detectors → ~**$0.20**. This is the
     gate decision; nothing downstream depends on it being cheap.
  2. **Distillation labels** (only *if* Pangram wins the audition *and* a detector
     signal is actually wanted): one **Bulk API** job over ~10k paragraphs
     (~1.2–1.5M words) → ~**$50–60 once** (bulk's ~20% off the ~$60–75 realtime
     cost — convenience, not a game-changer), then train the free local classifier
     above. After that, **zero Pangram calls** — the local model is the reward
     signal everywhere.
- **Never:** Pangram as the live per-candidate `Detector` in `score_candidate`,
  in best-of-N at inference, or per training step. Those are unbounded recurring
  spend on a signal we can distil to a fixed one-time cost.
- **Integration seam when/if the time comes:** a `pangram_detector(...)` factory
  returning the existing `Detector` callable (`prose -> {score, name}`), mapping
  `score = fraction_ai`, gated behind `PANGRAM_API_KEY`, **opt-in only** (never
  the default; default stays `null_detector`). It is for the one-shot labeling
  passes above, not for `evaluate_groups`' default path.

## Outputs

- **`scores.jsonl`** — one **id-keyed** record per pair (`id` = `synth_key` or
  `capture_id:chunk_index`), carrying a `meta` subset (`source`, `synthetic`,
  `generator`, `slop_strategy`, …) and `scores: {<field>: {vale, judge, detector,
  …}}`. The id is the join key back to the corpus — so scores feed Phase-3
  filtering/weighting and Phase-4 best-of-N without re-deriving anything. The run
  is **idempotent/resumable** (scored ids skipped), like `synth`.
- **A summary** (`summarize_scores`, `schema_version: 2`) aggregating per field
  across rows (fields-across-rows = the movement view: slop vs target vs output),
  optionally faceted `by` a meta key — `by="slop_strategy"` turns "is `catalogue`
  slop better than `polish`?" into per-strategy mean judge/vale/detector scores.

## Done-criteria

- [x] All four signals runnable from one entrypoint — Vale (live, optional),
      LLM-judge (live via OpenRouter), detector (seam + `null_detector`), eyeball
      (passthrough field). `stylebot.eval` + `tests/test_eval.py` (15 tests, no
      API/network/GPU). Runs keyless.
- [ ] The detector audition completed and decision recorded here. **← the one
      remaining track.** Seam is `Detector` (a `prose -> {score, name}` callable);
      drop the winner in as the default for `score_candidate(detector=...)`.
- [x] Scores emitted in a stable, **id-keyed** schema (`scores.jsonl` +
      `schema_version: 2` summary) that Phase 3/4 can cite and join back to pairs.
- [x] A documented entrypoint: `ai-style eval --pairs PATH.jsonl [--field
      slop|target] [--judge --judge-model …] [--vale-config …] [--max-workers N]
      [--out scores.jsonl] [--summary summary.json] [--by slop_strategy] [--limit N]
      [--report scores.html] [--sample N]`. Batched, concurrent, resumable.
- [x] A read-only **scores visualiser** (`stylebot.report.render_scores_report` /
      `--report`): self-contained HTML joining `pairs.jsonl` + `scores.jsonl` by id —
      slop↔Dan text + judge score + rationale per pair, sortable by the slop→Dan
      delta, filterable by strategy, under a per-strategy headline. Generic over
      score *fields* (a Phase-4 `output` field renders with no change). Reuses the
      targets report's `_CSS` / `_histogram_svg` / escaping. No spend (re-renders
      from `scores.jsonl`).

## Status detail (as built 2026-06-28)

- **Public API** (`stylebot.eval`): per-text primitives `vale_score`,
  `openrouter_judge` (factory) / `Judge` / `JUDGE_SYSTEM` / `parse_judge_reply`,
  `Detector` protocol / `null_detector`, `score_candidate`; **batched layer**
  `score_pairs_file` (corpus → id-keyed `scores.jsonl`, concurrent + resumable) /
  `summarize_scores(by=…)` / `record_id` / `pair_body` / `FIELD_EXTRACTORS`. The
  shared reader is `stylebot.pairs.iter_pairs`. (`evaluate_groups` /
  `read_prose_files` remain as in-memory conveniences, no longer the CLI path.)
- **Keyless by default.** No judge ⇒ `judge: null`; Vale absent ⇒
  `{"available": false}`; detector unconfigured ⇒ `{"score": null}`. The harness
  always completes; signals it lacks are explicit nulls, never faked.
- **Judge via OpenRouter** (`OPENROUTER_API_KEY`, optional `OPENROUTER_BASE_URL`),
  default model `anthropic/claude-opus-4-8`, scores 1–5 + rationale; injectable so
  tests pass a fake. Mirrors `synth`'s OpenRouter wiring (one key, many models).
- **Next:** run the detector audition (Binoculars / Ghostbuster / RADAR vs
  Pangram on single-paragraph lightly-humanised text) and record the choice here.

## Guardrail (policy, not optional)

This is **not** a slop-detection evader (see post Non-goals). The detector is
useful only because its tripwires — uniform rhythm, hedging, signposting,
generic vocab — are the same patterns the slop catalogue targets. A styler that
lowers its detector score does so *as a consequence* of writing more like Dan.
Watch for reward-hacking (passes-detector-but-still-bad output); the LLM-judge
and eyeball signals exist to catch exactly that.
