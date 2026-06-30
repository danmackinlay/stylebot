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
- **`meta.weight` via the detector (if used):** the trained voice classifier's
  `p_dan` on the Dan side can derive a per-pair weight, but only under the **shared
  by-POST split** — train the detector on a `--holdout` partition that excludes the
  posts this run trains on, or the weight is fit on its own training data (leakage).
  Coordinate one partition across styler-train / detector-train / eval (see
  `eval-harness.md` "the split contract"). Absolute-probability use wants
  calibration; ranking does not.

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
