# Next steps — what to ship

**Purpose.** The forward-looking operational overlay: what's built and the next
moves in priority order, with concrete commands. The authoritative spec is
[`OVERVIEW.md`](OVERVIEW.md) + the per-phase files + `../CLAUDE.md`; this is the
"you are here, go there next" layer. The mechanism/policy codebase is clean (the
QA declutter landed 2026-06-29) — focus here on functionality, not housekeeping.

## Current state (built + verified)

All green: `uv run pytest -q` = **143 passing** (stylebot, incl. the dep-free
`classify` seam + the `classify_train` trainer tests); the blog runners
`uv run python tests/test_training_targets.py` (15/15) +
`tests/test_voice_classifier.py` (4/4 policy tests); `ruff` clean. The QA declutter
(2026-06-29) removed dead blog-build code + the direct-Anthropic generator/dep
(hosted models go via OpenRouter).

- **Phase 0/1** — scaffolding + `ai-style-log` (daily pair capture) shipped;
  heading context added.
- **Phase 2** — `ai-style synth` / blog `dan-style synth` built + curated. Slop is
  a **knob** (`STRATEGIES` / `--slop-strategy` / `--slop-system-file`) and
  **multi-source** via `openrouter_generator` (one `OPENROUTER_API_KEY`). The
  paid at-scale run is **not yet done** — it's the active step below.
- **Eval** — `stylebot.eval` + `ai-style eval` (batched, JSONL-native): scores a
  `pairs.jsonl` → id-keyed `scores.jsonl`, `summarize_scores(by=…)`, and a scores
  **HTML report** (`--report`). Live OpenRouter judge wiring **verified**. The 4th
  signal — the **trained voice classifier** — is now built (StyleDistance backbone;
  `ai-style eval --detector-model` / `dan-style train-clf`), so all four signals run.
- **Deferred (planned, not built):** `meta.weight`, Phase 3 (LoRA training),
  Phase 4 (inference CLI).

## Next move — the two unblocked tracks (run in parallel)

### A. The Phase-2 experimental generation loop (the active step)

Not a one-shot paid run — an iterate-and-promote loop. Per slop strategy:
generate a small batch into a **scratch** dir, eyeball, score, and promote a
winner into the real corpus. Generating with real models and judging both need
the key, so run via `direnv exec` (see Environment).

```sh
cd ~/Source/stylebot
# 1. generate a small batch for one strategy into a throwaway dir:
direnv exec . uv run ai-style synth --blog-root ~/Source/livingthing \
  --data-dir /tmp/slop-experiments --slop-strategy catalogue \
  --openrouter-model anthropic/claude-opus-4-8 --openrouter-model openai/gpt-4o \
  --limit 40 --report /tmp/exp-targets.html
# 2. score it + compare flavours + eyeball the pairs:
direnv exec . uv run ai-style eval --pairs /tmp/slop-experiments/pairs.jsonl \
  --judge --by slop_strategy --report /tmp/exp-scores.html
# 3. repeat for other strategies into the SAME scratch dir (synth_key carries the
#    strategy, so they accumulate without colliding), compare the per-strategy Δ.
```

**Promote the winner** into the real corpus (the blog path is the production run):

```sh
cd ~/Source/livingthing
direnv exec . uv run dan-style synth --dry-run --report /tmp/corpus.html   # vet first
direnv exec . uv run dan-style synth --limit 3000 --slop-strategy <winner>  # spends $$
```

Corpus lives at `~/Source/livingthing/_training_pairs/pairs.jsonl` (private,
gitignored). Gate it: `stylebot.pairs.validate_pairs_file(path)` must return `[]`.

**Watch for (the judge-calibration caveat):** in this real regime the "slop" is a
paraphrase of Dan's *own* prose, so slop and Dan scores sit closer than the easy
1-vs-4 split seen on contrived fixtures. The slop→Dan **delta** is the signal for
ranking strategies. If the judge bunches everything near 3, sharpen `JUDGE_SYSTEM`
(`stylebot.eval`) or move to a **pairwise** "which is more Dan-shaped" comparison.

### B. The detector — BUILT (trained voice classifier)

Was "audition a general AI-detector vs Pangram"; **settled 2026-06-30** by
training a Dan-vs-slop classifier on the content-matched pairs instead (cheaper,
keyless, on-target). Full write-up: [`eval-harness.md`](eval-harness.md) "The
detector decision". State:
- **Built + wired:** StyleDistance backbone (bake-off winner, 0.78/0.72 held-out).
  Runtime `stylebot.classify` (dep-free); generic trainer `stylebot.classify_train`
  / `ai-style train-clf` (`stylebot[classifier]` extra); blog policy wrapper
  livingthing `voice_classifier.py` / `dan-style train-clf`; artifact `_models/voice-clf/`.
  Score it via `ai-style eval --pairs … --detector-model _models/voice-clf` or
  `dan-style eval`. Keyless, free per pair.
- **Reward-safety — the shared splits contract (built 2026-07-06):** the canonical
  three-role partition (frozen **eval** / **styler** / **detector**) lives at
  livingthing `_training_pairs/splits.json` (made once via `ai-style make-splits`;
  eval pinned from real-capture posts, the rest hash-assigned so new posts flow in
  stably). `dan-style train-clf` uses it automatically: head fit on the detector
  pool only, eval posts never embedded, a styler-posts holdout metric reported,
  role counts + **danger report** (dangerously-small strata warnings) recorded in
  `meta.split`. C is selected by **nested group-CV** (`--C` = explicit override —
  don't sweep it against the printed metric). First run: detector-pool CV
  0.787/0.742, styler-holdout 0.830/0.726 (88 unseen pairs), head C=10.0. Phase 3
  must train the styler on the **styler**-role posts and final-eval on **eval**.
  Keep the judge + eyeball as the orthogonal anti-Goodhart guard regardless.
- **Synth-augmented retraining:** more `ai-style synth` pairs are valid training
  data (content-matched, by-POST split covers them via `meta.source`). On a mixed
  corpus the metrics auto-facet by provenance (`metrics.by_provenance.real`/
  `.synthetic`); the **real** facet is the honest number — a synth-heavy corpus
  must not grade itself on detecting its own paraphrase generator. Diversify the
  slop side (model rotation + strategies) against generator-tic shortcuts.
- **Optional:** a one-shot Pangram cross-check before trusting the reward (~$0.20),
  never in a hot loop. Not a dependency.

## The data-gated tail (after enough corpus exists)

- **Phase 3 — LoRA training** ([`phase-3-training.md`](phase-3-training.md)):
  `stylebot.train.run_training` over `pairs.jsonl` (required `--data-dir`).
  Needs corpus volume. This is where `meta.weight` / synthetic down-weighting and
  the train/val split get decided — the id-keyed `scores.jsonl` is the input that
  makes per-pair weighting/filtering possible (an open question, see below).
- **Phase 4 — inference CLI** ([`phase-4-inference-cli.md`](phase-4-inference-cli.md)):
  `ai-style run <file>`. Needs a trained adapter. The eval scores report is
  **already generic over fields** — a styler run scored on `slop`/`output`/`target`
  renders the movement-toward-Dan view by adding one `FIELD_EXTRACTORS` entry.

## Open decisions / things to watch (from OVERVIEW "Open questions")

- **Generation covariate sweeps (the active Phase-2 step)** — reasoning effort
  (default high), prompt, model and sampling are now recorded as `meta.gen` and
  facetable (`ai-style eval --by <covariate>`). Run the reasoning sweep + prompt
  ablation (incl. the blog's `_voices/slop_patterns.md` via `--slop-system-file`);
  see `phase-2-synthetic-pairs.md` → Experiments.
- **Synthetic↔real distribution match (deferred eval phase)** — no distributional
  comparison exists yet; build a feature/mannerism extractor faceted by
  `meta.synthetic` to see which real Claude mannerisms synthetic slop misses.
- **`meta.weight`** — down-weight synthetic vs real pairs at train time? What
  weight? The scores join makes it derivable; decide in Phase 3.
- **Single- vs multi-source slop ablation** — worth publishing? (`SLOP_MODELS` +
  `meta.generator`/`meta.gen` make this measurable now.)
- **Base model** — Qwen3 8B first, 70B if headroom allows.
- **Detector** — settled: trained voice classifier (track B). The open follow-up
  is the reward-time held-out split + the Goodhart guard, not the choice of signal.
- **Heading-context depth** — only `immediate` shipped; `breadcrumb` is designed
  but unbuilt (see `heading-context.md`).

## Environment & resume

- **Both repos checked out side by side** (`~/Source/stylebot`, `~/Source/livingthing`);
  the blog has an editable path dep on `../stylebot`.
- **Keys** in each repo's gitignored `.env` (`OPENROUTER_API_KEY`; optionally
  `OPENAI_API_KEY`/`LOCAL_LLM_*`; `PANGRAM_API_KEY` only for an optional one-shot
  Pangram cross-check). **The key is only in env via `direnv exec uv run …`** — a
  bare `uv run` won't see it. Eval/synth run **keyless** by default (no judge / no
  paid generator), so tests and dry-runs need nothing.
- **Verify:** `cd ~/Source/stylebot && uv run pytest -q && uv run ruff check .`;
  `cd ~/Source/livingthing && uv run python tests/test_training_targets.py`.
- **Inspect with no spend:** `ai-style synth … --dry-run --report`, and
  `ai-style eval --pairs … --report` (keyless re-render over an existing
  `scores.jsonl` is a no-op pass).

## Guardrails (do not violate)

- **Never commit** the corpus (`_training_pairs/`, `pairs.jsonl`) or `.env`/`.envrc`.
  Sanity-check `git status` before every commit. stylebot is public.
- **`STYLE_SYSTEM` (`stylebot.ai_core`) is frozen** — changing it invalidates the
  whole corpus.
- **stylebot never imports `livingthing`** — mechanism stays generic; blog-specific
  policy lives only in `livingthing.training_targets` / `train_targets`.
- **PRs earn their place with a check** (a passing test or an eval number), per
  OVERVIEW "How we work". Plan edits commit straight to the branch.
