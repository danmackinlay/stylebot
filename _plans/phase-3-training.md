# Phase 3 · LoRA training — 📋 PLANNED

Run LoRA SFT on a small open-weights model so the style lives in weights, not
the prompt. Data-gated: needs enough pairs, not more engineering.

## Inputs

- `$STYLEBOT_DATA_DIR/pairs.jsonl` (Phase 1 real + Phase 2 synthetic, mixed).
- `TOGETHER_API_KEY` (default trainer) / `FIREWORKS_API_KEY` / `TINKER_API_KEY`.
- Base model: start **Qwen3 8B**; escalate to 70B only if 8B lacks headroom.

## Method / decisions (from the post)

- **Train on Together** (~$0.48/1M tokens for ≤16B LoRA SFT, ~$10/run — cheap
  enough to iterate). Together makes the LoRA easy to *extract*.
- **Switch on zero-data-retention** on Together (defaults to retaining).
- **Serve on Fireworks**: it scales-to-zero, ideal for a personal CLI; Together
  only offers a $6.49/hr always-on dedicated endpoint (a forgot-to-turn-it-off
  hazard). Fireworks accepts uploaded LoRAs, so a Together-trained adapter can
  be served there.
- **Tinker** is a fancier option whose low-level primitives (`forward_backward`,
  `sample`, `optim_step`) would enable a future DPO/RL loop instead of vanilla
  SFT. Park unless preference-tuning becomes the plan.

## Outputs

- A trained LoRA adapter (downloaded from Together and/or hosted on Fireworks).
- A pinned **training manifest** committed to git (NOT the weights, NOT the
  data): base model, hyperparameters, pairs.jsonl content hash + record count,
  train/val split, dataset filters (e.g. synthetic weight), timestamp, cost.
  This is the reproducibility record — `_plans/runs/` or similar.

## Done-criteria

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
