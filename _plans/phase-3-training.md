# Phase 3 · LoRA training — 🔧 BUILT (2026-07-21); first paid run pending

Run LoRA SFT on a small open-weights model so the style lives in weights, not
the prompt. Data-gated: needs enough pairs, not more engineering.

## As built (2026-07-21)

`stylebot.train` ships `assemble_training_corpus` (validate → styler-role
filter → policy hooks → deterministic by-POST val split), `run_training`
(manifest → Tinker cookbook supervised recipe → PEFT-adapter export), and the
thin `ai-style train` / `dan-style train` CLIs; `tests/test_train.py` covers
assembly, the pinned data policy, and the paid path through the `runner` seam.
Deps live behind the `trainer` extra (`uv add 'stylebot[trainer]'`).

Decisions taken at build time (superseding the older pins below):

- **Base model `Qwen/Qwen3.5-9B`** ($1.463/1M train tokens). The Qwen3-8B pin
  and the "escalate to 70B" option predate Tinker's 2026-06-12 retirements
  (all Llama-3.x, Qwen3-32B/30B); current step-up candidates are Qwen3.6-27B
  (dense) or Qwen3.6-35B-A3B (MoE — also the local-serving-friendly shape).
- **Thinking disabled** (`qwen3_5_disable_thinking` renderer): the pairs carry
  no reasoning traces and the styler is a pipe; a thinking A/B is a cheap
  follow-up run, not the default.
- **Manifest home: livingthing `_training_pairs/runs/<run-id>.json`**
  (committed, beside the corpus it hash-pins); scratch in `_tmp/train/<id>/`,
  adapter at `_models/styler/<id>/` (both gitignored).
- **Val split**: 10% of styler POSTs, seeded shuffle — not the cookbook's
  row-shuffle `test_size`, which would leak val posts' sibling chunks.
- `meta.weight` stays a hook (`pair_weight`/`per_target` callables), uniform
  by default for run 1.

**Data hygiene (2026-07-21 corpus QA)** — sharpening, not repealing, the
keep-near-copies policy below. QA found 386 *identity* pairs (qwen3-8b @
effort=off returned the input: median `transform_sim` 0.98, length-ratio p10
1.00) plus stragglers ≥0.95 elsewhere; identity ≠ near-copy — they teach the
styler to copy. Two gates now exist:

- **Training side**: livingthing's `train_styler.styler_pair_ok` selector
  (drop the qwen3-8b@off cell + any pair ≥0.95) via the `selector` hook;
  recorded in the manifest's `filters`/`dropped`. qwen3-32b's 334 florid pairs
  (4.4–6.8× inflation) stay in run 1 — directionally-real compressions; facet
  a v2 ablation by generator if eval frowns.
- **Synth side**: `synthesize_pairs(max_transform_sim=…)` /
  `--max-transform-sim` drops degenerate generations at birth, loudly
  (`SynthResult.skipped_degenerate`); blog policy pins 0.95
  (`training_targets.MAX_TRANSFORM_SIM`). Follow-ups for the promote step:
  audition `qwen/qwen3.5-9b` as a slop generator (has the anti-degeneration
  sampler knobs qwen3-8b lacks); reconsider qwen3-32b's roster spot (it
  ignores `reasoning: {enabled: false}`).

Dry-run against the settled-ish corpus (2026-07-21, sha `a38610bc71290d87`),
selector active: 1,666 train / 148 val pairs (116 real / 1,698 synthetic; 235
styler-role identity pairs dropped), ~1.12M train tokens ≈ **$1.64/epoch**.
The first paid run awaits corpus settlement (the coverage run still plans
~480 new pairs) — precondition 2 below.

## Inputs

- `--data-dir` (**required, no default**): the `pairs.jsonl` to train on
  (Phase 1 real + Phase 2 synthetic, mixed). Expensive + stateful, so the path
  is always explicit — it's the reproducibility record (see OVERVIEW
  "Interfaces"). No `--blog-root`: training reads the corpus, not raw prose.
- `TINKER_API_KEY` (trainer) / `FIREWORKS_API_KEY` (serving). (`TOGETHER_API_KEY`
  only for the parked Together training fallback — see livingthing decisions #D1.)
- Base model: start **Qwen3 8B**; escalate to 70B only if 8B lacks headroom.

Interface: `stylebot.train.run_training(...)` over explicit params; `ai-style
train` is the thin CLI. The blog build can import the function directly.

## Method / decisions (from the post)

- **Train on Tinker** (Thinking Machines — GA, account in hand): managed
  distributed training (no GPU to own/rent), per-token (~$0.40/1M, ~$6/8B run),
  weight download → HF → MLX. SFT v1 uses the cookbook's supervised recipe; the
  *same* platform graduates to DPO/preference (Phase 7) via its low-level
  primitives (`forward_backward`/`optim_step`/`sample`/`save_state`) + RL/preference
  cookbook — no migration. Full rationale: livingthing decisions #D1 + #D5.
- **Serve on Fireworks**: scales-to-zero (`--min-replica-count 0`), ideal for a
  personal CLI; accepts the Tinker-exported LoRA. **Fallback:** local MLX on the
  Mac (download → merge → `mlx_lm.server`/Osaurus), then Modal. (Together train+serve
  is parked — dedicated-endpoint-only, wrong idle cost shape.)
- **The shared partition is materialised** (built 2026-07-06): consume
  `stylebot.splits` / the blog's `_training_pairs/splits.json` — Phase 3 trains
  the styler on the **styler**-role posts only (`splits.role_of(meta.source)`),
  the detector is already fit on the **detector** pool, and the frozen **eval**
  posts are for final eval alone (see `eval-harness.md` "the split contract").
- **`meta.weight` via the detector (if used):** the voice classifier's `p_dan`
  on the Dan side can derive a per-pair weight — legitimate under the splits
  contract above (the detector never saw the styler posts). Absolute-probability
  use wants calibration; ranking does not.

## Data policy — pinned 2026-07-21 (decisions an implementer must not re-derive)

- **KEEP near-copy pairs for the styler.** ~19% of synthetic pairs have
  `meta.transform_sim > 0.85` (the model barely changed Dan's text). They are
  label noise for the *detector* — `classify_train.assemble_dataset` drops
  them, and that filter must NOT be copied here — but they are good styler
  data: they teach the model to leave alone prose that is already fine.
  Restraint is half the job.
- **Per-target multiplicity is a policy hook, not an accident.** The corpus
  aims at ~1 synthetic pair per passage (`synth --skip-covered`), but re-key
  epochs and deliberate variants (`--per-generator`, `--replicate`) can stack
  several pairs on one target. Assembly should expose a per-target cap/weight
  (selector-style callable, like every other stylebot policy knob) so
  duplicated targets don't silently up-weight their passages.
- **Covariates available for mix/filter/weight decisions** (all per-pair):
  `meta.synthetic` (real/synthetic strata — 331 real / 3,231 synthetic as of
  2026-07-21), `meta.transform_sim`, `meta.gen.reasoning_effort` (two epochs:
  ~1,970 `high`, the rest `off` — equivalent per the effort sweep, but
  recorded), `meta.gen.model` / `slop_strategy` / `prompt_id`, `meta.gen.replicate`,
  `meta.context` (heading context, prepended identically to BOTH sides —
  train on messages as-is, never strip it from one side only).
- **`STYLE_SYSTEM` is frozen** (`stylebot.ai_core`) and is `messages[0]` of
  every pair; `validate_pairs_file` enforces it. Train with it verbatim —
  inference (Phase 4) sends the same string, so it must be the trained-in
  system prompt.

## Handing this phase to an implementing agent

Start a FRESH session in the stylebot repo; do not import chat history. The
contract stack to read, in order: `CLAUDE.md` → this file →
`phase-1-pair-capture.md` (the pairs schema) → `eval-harness.md` ("the split
contract"). Preconditions to verify before any paid run:

1. `uv sync && uv run pytest -q` green; `TINKER_API_KEY` present in `.env`
   (copy `.env.example`; never commit either).
2. The corpus coverage run has FINISHED (`dan-style synth --skip-covered ...`
   writes nothing new) — the manifest pins a content hash of `pairs.jsonl`,
   so train on a settled corpus, not a moving one.
3. `uv run python -c "from stylebot.pairs import validate_pairs_file; ..."`
   passes on the real corpus, and `splits.json` loads (`stylebot.splits`).

Architecture rules that bind this phase: library-first
(`stylebot.train.run_training(...)`, `ai-style train` a thin wrapper, and a
`dan-style train` mirror in livingthing carrying blog paths/policy); explicit
`--data-dir`, no default; the corpus and adapter weights never enter this
repo — only the manifest is committed. Tinker usage should follow the Tinker
cookbook's SFT recipe (fetch current docs; do not code the API from memory).

## Outputs

- A trained LoRA adapter (Tinker-trained → HF export; served on Fireworks or local MLX).
- A pinned **training manifest** committed to git (NOT the weights, NOT the
  data): base model, hyperparameters, pairs.jsonl content hash + record count,
  train/val split, dataset filters (e.g. synthetic weight), timestamp, cost.
  This is the reproducibility record — `_plans/runs/` or similar.

## Done-criteria

- [x] Input `pairs.jsonl` passes `stylebot.pairs.validate_pairs_file` before
      the run (don't spend money training on a malformed corpus). Enforced in
      code: `assemble_training_corpus` refuses a malformed corpus.
- [x] One reproducible training run end-to-end (data → adapter), manifest
      committed: livingthing `_training_pairs/runs/20260721-run1.json`
      (corpus sha `0838ae470afef6c3`, val NLL 1.018, $1.97).
- [x] Held-out val split; eval numbers on it (2026-07-22 smoke, 40 val
      pairs, detector `P(slop)` means): slop 0.507 → **adapter 0.422** →
      target 0.352; prompt-only baseline 0.552 (worse than the input).
- [x] The adapter beats the prompt-only baseline on the detector signal on
      **35/40** pairs; fluency intact by eyeball (it also reproduces the
      sentence-per-line discipline). Judge cross-check deferred to Phase-4
      eval.
- [x] Documented entrypoint: `dan-style train` (RUNBOOK §8); a rerun from a
      manifest is the same command with the manifest's hyperparameter flags.

## Risks

- 8B may lack headroom to learn the style without losing fluency → try 70B.
- Synthetic/real distribution mismatch → adjust mix/weights, add real pairs.
- Fine-tuning may be a dead end → fallback is a stronger prompt vs a frontier
  model; the labelled corpus remains useful regardless.
