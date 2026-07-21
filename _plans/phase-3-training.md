# Phase 3 · LoRA training — 📋 PLANNED

Run LoRA SFT on a small open-weights model so the style lives in weights, not
the prompt. Data-gated: needs enough pairs, not more engineering.

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

- [ ] Input `pairs.jsonl` passes `stylebot.pairs.validate_pairs_file` before
      the run (don't spend money training on a malformed corpus).
- [ ] One reproducible training run end-to-end (data → adapter), manifest
      committed.
- [ ] Held-out val split; report the eval-harness numbers (Phase E) on it.
- [ ] The adapter beats the prompt-only baseline on ≥1 eval signal without
      tanking general fluency (the "8B headroom" risk).
- [ ] Documented entrypoint to launch a run from a manifest.

## Risks

- 8B may lack headroom to learn the style without losing fluency → try 70B.
- Synthetic/real distribution mismatch → adjust mix/weights, add real pairs.
- Fine-tuning may be a dead end → fallback is a stronger prompt vs a frontier
  model; the labelled corpus remains useful regardless.
