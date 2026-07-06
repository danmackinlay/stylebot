# Eval harness — 🔧 BUILT (library + CLI; trained detector wired)

How we know the styler works. Runs **offline** — it scores candidate output and
is *not* wired into the trainer or baked into the served adapter. Built early and
in parallel: it's the ground truth every later phase reports against, and it
needs only sample prose, not a trained model.

**Status (2026-06-30):** `stylebot.eval` ships the four-signal scorer as typed
library functions + the thin `ai-style eval` CLI. Runs keyless by default (Vale
+ null judge/detector); `--judge` scores voice via OpenRouter. **All four signals
are now live:** the statistical detector is the **trained Dan-voice classifier**
(`stylebot.classify` seam + the livingthing-trained head), wired via
`ai-style eval --detector-model PATH` / `dan-style eval`. It replaced the
"audition a general AI-detector vs Pangram" plan — see "The detector decision"
below. Keyless + free per pair.

## Inputs

- **A `pairs.jsonl` corpus** (the pipeline lingua franca) — eval scores the
  *fields* of each pair: `slop` (`messages[1]`) and `target` (Dan, `messages[2]`),
  with the shared heading-context prefix stripped so only the transform region is
  judged. (One-prose-per-file is gone — we never have that.) A Phase-4 styler run
  scores the same way once it emits an `output`-bearing JSONL (add an `"output"`
  field extractor then).
- A **trained detector artifact** (`_models/voice-clf/`, livingthing-side) for
  the detector signal — keyless. `PANGRAM_API_KEY` only if you ever want Pangram
  as an *optional* independent cross-check (see "The detector decision").
- `OPENROUTER_API_KEY` for the LLM judge (one key, many models).
- Vale + an optional Vale config (a parameter — never a hardcoded blog ruleset).

Interface: `stylebot.eval` functions over `pairs.jsonl` (`score_pairs_file` →
id-keyed `scores.jsonl`; `summarize_scores` for the aggregate view); `ai-style
eval` is the thin CLI. Paths resolved via `stylebot.config`.

## The four signals

1. **Vale** — mechanical slop (banned words, indefinite "you", -ize spelling).
2. **LLM-as-judge** — "is this Dan-shaped" at the voice level.
3. **Trained detector** — "is this Dan vs AI-slop", scored by the trained
   voice classifier (`P(slop)`, higher = more AI-like). Keyless, free per pair.
4. **Dan's eyeball** — the veto channel (not automatable; reported alongside).

## The detector decision (settled 2026-06-30 — trained voice classifier)

**Decision: the detector is a trained Dan-vs-AI-slop classifier, not a
general-purpose AI-detector.** We already hold content-matched labels — every
pair is a `(slop, Dan)` couple over the *same* passage — so we can score the
question we actually care about ("does this read like Dan?") directly, instead of
borrowing someone else's human-vs-AI-in-general detector (Pangram) and paying per
call. It is **keyless, free per pair, reproducible**, and drops straight into the
existing `Detector` seam.

**How it's built** (both halves are stylebot mechanism, split by dependency
weight): the **runtime** is `stylebot/classify.py` — a frozen **style** embedding +
a logistic head, scored by a pure-Python dot product (dep-free at import,
enforced by a test); the **trainer** is `stylebot/classify_train.py` (dataset
assembly, the POST-split methodology, artifact I/O) behind the
`stylebot[classifier]` extra, with the generic CLI `ai-style train-clf`.
Author-specific *policy* — backbone pin, free-positives selector, path
defaults — lives in livingthing (`voice_classifier.py` / `dan-style train-clf`,
a thin delegate). The
backbone was a **measured** choice — a bake-off over the content-matched pairs
ranked candidates by content-matched pairwise accuracy + AUC (split by POST):

| backbone | pairwise_acc | AUC |
|---|---|---|
| **StyleDistance** (`StyleDistance/styledistance`, 768-d) — **winner** | **0.78** | **0.72** |
| LUAR-MUD | 0.76 | 0.71 |
| mxbai (semantic baseline) | 0.75 | **0.62** |
| Wegmann CISR | 0.73 | 0.67 |

The AUC column is the tell: mxbai's topic-dominance collapses on the absolute "is
this slop" question; a content-independent *style* embedding holds. Polarity is
`score = P(slop)` so it composes with `mean_detector_score` unchanged; the factory
also returns `p_dan` for reward callers.

### Eval vs reward — and the leakage-safe split contract

The detector plays two roles with different safety requirements:

- **Eval signal (measure only).** A clean **by-POST** train/test split
  (`GroupShuffleSplit`) gives an honest, in-distribution estimate. Since eval only
  *measures*, that is sufficient — this is the keyless cross-check for the paid
  judge. ✅ built.
- **Reward signal (optimise against — Phase-3 weighting, Phase-4 best-of-N).**
  Here a split is *necessary but not sufficient*. Splitting removes
  **data-leakage** circularity (the detector hasn't memorised the styler's posts);
  it does **not** remove **Goodhart / proxy over-optimisation** — optimising a
  policy against a frozen proxy pushes its outputs into the proxy's blind spots
  *regardless of how the proxy was trained* (Gao et al., reward-model
  over-optimisation). The guard against *that* is a signal with **independent
  failure modes** — the LLM judge and Dan's eyeball — not a fresh split of the
  same model. Best-of-N with small N is gentle (KL ≈ log N), so it's fairly safe;
  DPO/RL needs the orthogonal channel in the loop.

**The split contract (load-bearing, materialised):** one **shared by-POST
partition** governs all three stages — styler-train, detector-train, and the
frozen eval — so the detector never trains on the posts the styler trains on or
that eval scores. It lives in one canonical file, `splits.json`
(`ai-style make-splits`; the blog's is committed at
`_training_pairs/splits.json`), with three roles: **eval** (a frozen pinned
list, sampled from posts with ≥1 *real* pair), **styler**, and **detector** —
the latter two assigned by a deterministic hash rule so new posts flow in with
no file churn and eval can never drift. `ai-style train-clf --splits FILE`
(auto-used by `dan-style train-clf` when the file exists) fits on the detector pool
only, never even embeds the eval posts, reports a bonus styler-posts holdout
metric, and records role counts + the danger report in `meta.split`. The
ad-hoc `--holdout-frac/--holdout-posts` flags remain for experiments; the
splits file is the contract. (The default fit-all artifact is for *measurement*
of an independent styler, and `meta.split` says so.)

**Regularization is nested, not hand-tuned:** C defaults to selection by inner
`GroupKFold` *within each training side* (`select_C`, grid `C_GRID`), so the
reported CV metric covers the tuning step; `--C` is an explicit override, and
sweeping it against the printed number is exactly the tuning-on-the-test-set
this design removes. Residual caveat: the *backbone* bake-off already selected
StyleDistance on this corpus's CV metric, so the headline carries mild
winner's-curse inflation — the frozen eval posts are where that gets settled
honestly, once, at final-eval time. **There is deliberately no fourth
materialised split for hyperparameters** — nested CV inside the detector pool
serves that role; a dedicated validation stratum at 28-post scale would starve
every other role for nothing.

**Synth-augmented training (the provenance facet):** synthetic pairs
(`meta.synthetic`) can enlarge the training corpus, but "how well do we detect
our own paraphrase generator" must never masquerade as the honest number. On a
mixed corpus both eval modes automatically facet the metric by provenance —
`metrics.by_provenance.real` / `.synthetic` in `meta.json`, echoed by the CLIs;
`meta.n_pairs_real`/`n_pairs_synthetic` record the composition. The **real**
facet is the headline when synth pairs are in the fit.

### Pangram — optional independent cross-check only (was: the planned signal)

Pangram is no longer the path; it survives only as an **optional, one-shot,
independent** sanity check — useful precisely because it's a *different model
family* with different blind spots, so it's the one thing the voice-clf (and the
judge) can't self-certify: run it once before trusting the detector as a *reward*
to confirm the styler lowers AI-ness independently rather than gaming our
embedding. It is a paid API (~$0.05/1k words, `PANGRAM_API_KEY`); if ever wired,
do it as a bounded one-shot pass via an opt-in `pangram_detector(...)` factory
(`score = fraction_ai`) — **never** the default, **never** in a hot loop. The
judge + eyeball already supply the orthogonality more cheaply, so Pangram is a
nice-to-have, not a dependency.

## Planned — synthetic↔real distribution match (a fifth capability)

The four signals score *individual passages* and aggregate scalar means. They cannot
yet answer the question Phase-2 needs: **does synthetic slop match the distribution of
real Claude output Dan actually cleans up?** (Eyeballing showed synthetic slop missing
real mannerisms — e.g. incoherent mixed metaphors.) There is no feature/mannerism
extraction and no distributional comparison (KL / Wasserstein / frequency tables).

Planned (its own phase; deferred): a feature extractor (hedge rate, signposting
density, sentence-length variance, the `_voices/slop_patterns.md` families, optionally
an AI-detector score) run over **two strata** — synthetic pairs vs the **real-slop
reference** (the 254 Phase-1 pairs, ≥48 from `automation:2`; plus `automation:1/2`
prose) — and a side-by-side distribution view per strategy/covariate. **Seam:** the
`meta.synthetic` facet (real = `synthetic` falsy/absent) over the existing aggregation;
it consumes the `meta.gen` covariates Phase-2 now records. See
[`phase-2-synthetic-pairs.md`](phase-2-synthetic-pairs.md) → Experiment 3.

## Outputs

- **`scores.jsonl`** — one **id-keyed** record per pair (`id` = `synth_key` or
  `capture_id:chunk_index`), carrying a `meta` subset (`source`, `synthetic`,
  `generator`, `slop_strategy`, the flattened `meta.gen` covariates, …) and
  `scores: {<field>: {vale, judge, detector,
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
- [x] The detector decision made + the signal wired. The trained voice classifier
      (StyleDistance backbone, bake-off-selected) is the detector; `Detector` seam
      unchanged (`prose -> {score=P(slop), p_dan, name}`), served via
      `stylebot.classify` and `ai-style eval --detector-model PATH` /
      `dan-style eval`. Held-out by-POST pairwise 0.78 / AUC 0.72; the
      `--holdout-frac`/`--holdout-posts` split makes it reward-safe (see above).
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
  shared reader is `stylebot.pairs.iter_pairs`. (The legacy in-memory
  conveniences `evaluate_groups` / `read_prose_files` were removed — the batched
  JSONL path is the one way to score.)
- **Keyless by default.** No judge ⇒ `judge: null`; Vale absent ⇒
  `{"available": false}`; detector unconfigured ⇒ `{"score": null}`. The harness
  always completes; signals it lacks are explicit nulls, never faked.
- **Judge via OpenRouter** (`OPENROUTER_API_KEY`, optional `OPENROUTER_BASE_URL`),
  default model `anthropic/claude-opus-4-8`, scores 1–5 + rationale; injectable so
  tests pass a fake. Mirrors `synth`'s OpenRouter wiring (one key, many models).
- **Detector (built 2026-06-30; trainer generalised 2026-07-06):** the trained
  voice classifier is the 4th signal, wired keyless. See "The detector decision"
  above; the generic trainer is `stylebot.classify_train` (`ai-style train-clf`,
  `stylebot[classifier]` extra); blog policy + the artifact live in livingthing
  (`voice_classifier.py` / `_models/voice-clf/`).

## Guardrail (policy, not optional)

This is **not** a slop-detection evader (see post Non-goals). The detector is
useful only because its tripwires — uniform rhythm, hedging, signposting,
generic vocab — are the same patterns the slop catalogue targets. A styler that
lowers its detector score should do so *as a consequence* of writing more like Dan.

The reward-hacking guard is structural, not aspirational: a held-out by-POST split
stops the detector rewarding *memorised* posts, but the deeper risk when you
optimise against the detector is **Goodhart** (the styler finds inputs that score
"Dan" without being Dan) — which the split does not touch. The defence is signals
with **independent failure modes**: the LLM-judge and Dan's eyeball must confirm
the detector's movement is real. Detector movement alone is never the success
criterion. (See "Eval vs reward" above.)
