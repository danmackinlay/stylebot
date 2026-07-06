# Plan: `ai-style` — a fine-tuned prose styler

## Context

The current `ai-preen` workflow runs Claude (via OpenRouter) as a copy-editor with the `/dan-voice` prompt and the Vale ruleset. The prompt-only baseline is heavily iterated and still produces slop — Dan describes the output as "shit." Adding more prompt-engineering has diminishing returns.

The goal is to build a specialist fine-tuned model that does *one thing*: rewrite prose into Dan's voice. It will not replace `ai-preen` or the day-to-day Claude agentic stack. It will be invoked explicitly as a separate CLI (`uv run ai-style …`) with its own options, sharing the existing chunking / masking / OpenAI-compatible-endpoint plumbing.

Public-facing argument and citations: [`notebook/fine_tuning_danbot.qmd`](../notebook/fine_tuning_danbot.qmd). Design-decision audit trail (why each non-obvious choice landed): [`_plans/ai-style-fine-tune-decisions.md`](./ai-style-fine-tune-decisions.md). Phase 1 user-facing docs: [`_training_pairs/README.md`](../_training_pairs/README.md).

## Recommended approach (default path)

**Train a LoRA adapter on Qwen3 8B with Tinker, then serve scale-to-zero on Fireworks; fall back to local MLX (Mac) and then Modal.** Training data is a hybrid pair set: synthetic pairs from the `automation: 0` corpus generated across a model matrix (Phase 2), augmented over time by real (slop → Dan rewrite) pairs from the Phase 1 logger. The Mac local-host path (Osaurus / MLX) is both the offline exit and the realistic serving fallback.

**`automation:` frontmatter as corpus partition**:

- `automation: 0` (1616 files) → "ground truth Dan-voice." Sample for the *target* side of synthetic pairs.
- `automation: 2` (29 files) → existing slop examples. A manual rewrite from `automation: 2 → automation: 0` is a free real pair.

Why Tinker + Fireworks (and the local/Modal fallbacks): decisions.md#D1. Why Tinker for training specifically: decisions.md#D5. Why Qwen3 8B: decisions.md#D2.

## Phases

### Phase 1: Pair-logging infrastructure ✓ shipped

Implemented as `uv run ai-style-log`. User-facing workflow docs in [`_training_pairs/README.md`](../_training_pairs/README.md); implementation reference in the module docstring of [`src/livingthing/bin/ai_style_log.py`](../src/livingthing/bin/ai_style_log.py). The system prompt for pair records is the `STYLE_SYSTEM` constant in [`src/livingthing/ai_core.py`](../src/livingthing/ai_core.py) — single source of truth shared with Phase 2 / Phase 5.

### Phase 2: Synthetic pair generation (~1–2 days, ~$100–250 in API calls)

Framing: "build a Dan-ifier," not a "de-Claude-er." Source-agnostic style transfer where the target is fixed (Dan's voice) and the source side spans the matrix the styler will actually see. Rationale and citations: [`notebook/fine_tuning_danbot.qmd`](../notebook/fine_tuning_danbot.qmd) (the "Whose slop?" section). Audit trail: decisions.md#D3.

**Alignment principles** (apply to all methods below):

- Slop distribution = "what a model emits when asked to write/expand/rewrite," **not** "what it does when asked to copy-edit." Light copy-edits don't surface the signature tics.
- Diverse sources, constant target. The system prompt and all docs say "rewrite in Dan's voice," never "undo Claude."
- Train on raw markdown only; mask-preservation deferred (decisions.md#D4).
- Chunk size matches inference: `STYLE_CHARS_PER_CHUNK ≈ 8_000` (~1.5–4k tokens), respecting paragraph/section boundaries so code fences don't get sliced.
- Chat template matches inference: Together-style `{"messages": [...]}` with the shared `STYLE_SYSTEM` constant.

**Method M1 — "paraphrase across a generator matrix"** (the workhorse, ~$120–180):

Sample paragraph-or-section chunks from `automation: 0`; for each, call a generator with "Rewrite this passage to be clearer and more polished, keeping the same information and length" (no mention of slop — we want natural house-style). Route across a matrix weighted to Dan's real inference distribution: ~60–70% Claude split across Haiku / Sonnet / Opus at current revisions; ~30–40% across GPT, Gemini, and an open model (Qwen / Hermes / Llama via OpenRouter). Optional: a small slice from the *base, un-fine-tuned* Qwen3 8B over skeletal notes — the most on-distribution "styler's own slop." Pair = `(model_paraphrase, dan_original)`. Filter: length delta > 40% or n-gram overlap suspiciously low → discard; keep pairs where Vale fires more on input than output. Target: 2–4k pairs.

**Method M2 — "expand skeleton into prose"** (~$30–50, 500–1000 pairs): reduce Dan's paragraph to a skeletal bullet outline (deterministic or via cheap LLM call), then ask a generator (same matrix) to expand back to a paragraph. Mirrors the "flesh out these notes" workflow where slop pain peaks; also makes the styler useful on Dan's own rough notes.

**Method M3 — "amplified slop"** (~$20–40, 300–500 pairs): explicitly prompt a generator to lean into patterns from [`_voices/slop_patterns.md`](../_voices/slop_patterns.md): "open with a throat-clear+colon, use structural signposting, hedge frequently." The most source-agnostic method by construction — the slop is synthesised from the human-curated pattern catalogue, not from one model's habits.

**Method M4 — mask-preservation supplement (deferred)**: not included upfront. Re-derive from M1 pairs only if Phase 4 eval shows degradation on masked inputs (decisions.md#D4). Marginal cost ~$10–15.

**Method M5 — bias toward styler use cases**: sample chunks from notebook drafts and post-introductions (short, dense prose where voice matters most), not just long reference-style notebook sections that read more like docs.

**Method M6 — detector-in-the-loop slop calibration** (optional): score paraphrases with the **trained voice classifier** (Phase 4a; `train-voice-clf`); if `P(slop)` is below threshold, re-prompt for harder sloppification. Produces synthetic pairs with calibrated AI-ness on the input side. Cost: ~$0 (keyless local detector).

**Output**: `_training_pairs/synthetic.jsonl` in Together chat format with `STYLE_SYSTEM` as the system message. Pre-train sanity: eyeball 30 random pairs (slop side obviously sloppy, target side obviously Dan, markdown preserved); distribution stats (length / edit-distance / Vale-warning delta); confirm Phase-1 real pairs fit the same format. Phase-1 real pairs mix in unweighted at first; upweight via duplication if they look much cleaner than synthetic.

### Phase 3: Fine-tune on Tinker (~$10–50, half a day plus wall-clock)

Run the cookbook's supervised SFT recipe (`tinker_cookbook.supervised` / "102: Your First SFT") on `Qwen/Qwen3-8B` (or current equivalent) — no training loop to write for the simple case. Default hyperparameters; 3 epochs; LoRA rank ~16. Estimated cost: 10k pairs × 500 tokens × 3 epochs ≈ 15M tokens × ~$0.40/M ≈ **$6** for 8B (see Models & Pricing for live rates; ~5× for a 70B-class base). Download the adapter (`tinker_cookbook.weights.download` → HF/PEFT) and save it + the config JSON to `_training_pairs/runs/<timestamp>/`; upload to Fireworks for serving. Same platform graduates to DPO/preference (Phase 7) with no migration.

### Phase 4: Evaluation harness (~1 day, ~$60–80 setup + ~$1/eval)

Build `uv run ai-style-eval` so future fine-tunes compare apples-to-apples:

- Hold out ~200 pairs (10% of training data) before training.
- Run held-out *input* through three configurations: (a) prompt-only baseline (current `/dan-voice` on Claude), (b) fine-tuned styler, (c) raw base model (sanity check that the LoRA is doing work).
- **Cross-source generalisation slice.** Hold out a slice of non-Claude slop inputs (GPT, Gemini, open-model paraphrases) and confirm the styler de-slops them too. Includes the open-model self-cleanup case (feed raw Qwen3.6 output, check the Qwen-based styler cleans up its family's prose). If it only works on Claude, that's a named, measurable failure — not a production surprise.
- **Four eval channels**:
  - **Vale**: slop warnings per 1k words. Deterministic, automatable, runs every iteration.
  - **LLM-as-judge**: side-by-side blind comparison ("which output is more like a human writer with this style sheet?"). ~$2/batch.
  - **Trained voice classifier** (Phase 4a; `train-voice-clf eval` / `ai-style eval --detector-model`): per-passage `P(slop)`, keyless and free. (Optional: a one-shot **Pangram** cross-check as an *independent* family — see decisions.md#D7.)
  - **Eyeball**: Dan reads 20 random samples and votes. The veto channel.
- Save eval results to `_training_pairs/runs/<timestamp>/eval.json`.

Also: an optional one-off Pangram pass over `automation: 0` (~$5) could surface (i) AI-tainted passages to drop and (ii) boundary passages where Dan-style and AI-style blur — *upweight* those in M1 sampling. Not required; the voice classifier's `P(slop)` over the corpus does the same triage keyless.

### Phase 4a: the voice classifier (BUILT 2026-06-30) — the local reward signal

**Enables M6, the `--best-of N` flag in Phase 5, and Phase 7 — with no gating spend.** Earlier this slot was a ~$50–60 Pangram-distillation job; superseded. We already hold content-matched `(slop, Dan)` labels, so we trained a Dan-vs-slop classifier *directly* on the pairs: a frozen **StyleDistance** style embedding + a logistic head (bake-off winner, 0.78 pairwise / 0.72 AUC, beating the mxbai baseline), scored by a pure-Python dot product in `stylebot.classify`. Keyless, free per call, callable anywhere (eval, best-of-N, a future DPO loop). Trainer: `stylebot.classify_train` / `ai-style train-clf` (generic mechanism, behind the `stylebot[classifier]` extra); `train-voice-clf` is the blog wrapper adding path defaults + the free-positives policy; artifact at `_models/voice-clf/`. **Reward-safety:** train with `--holdout-frac/--holdout-posts` (shared by-POST split) so it isn't fit on the styler's posts, and keep R_judge + eyeball as the orthogonal anti-Goodhart guard. Full rationale: decisions.md#D7.

### Phase 5: `uv run ai-style` CLI (~1 day)

New command, *not* hooked into post-render or `ai-preen`. Pure manual invocation.

- New module `src/livingthing/workflow_style.py`, mirroring `workflow_preen.py` but: one pass only (style transfer, no summary/quality/edit passes); shared `STYLE_SYSTEM` constant from `ai_core.py` (Phase 2 byte-identical); `STYLE_MODEL` / `STYLE_MODEL_BASE_URL` env vars (default Fireworks deployment ID; point at Osaurus for local); `date-ai-style` frontmatter timestamp; `STYLE_CHARS_PER_CHUNK ≈ 8_000` matching training; raw markdown by default, `--mask` flag for defense-in-depth.
- Wrap calls with retry-on-`503-DEPLOYMENT_SCALING_UP` so Fireworks' first-after-idle wake (tens of seconds for an 8B-class adapter) doesn't surface to the user.
- Reuse: `qmd_core.py`, `pandoc_mask.py`, `ai_core.call_edit` (with model overridden), `format_sentences` post-step, Vale check.
- Flags: `--model {together|fireworks|local|baseline}` for quick A/B; `--dry-run` for diff preview; `--file <path>` or stdin/stdout. No auto-commit; no batch frontmatter-gated runs.
- **`--best-of N`** (optional, requires Phase 4a): generate N candidates at varying temperatures, score each with the trained voice classifier (`p_dan`), return the most Dan-like. Best-of-5 at temp=0.3/0.5/0.7/0.9/1.0 costs ~5× inference (still cents on Fireworks scale-to-zero). Small N = gentle optimisation pressure (KL ≈ log N), so over-optimisation risk is low; still emit the score to stderr and spot-check with the judge/eyeball. Skip if Phase 4 shows single-shot is good enough.

### Phase 6 (optional): Local Mac hosting

Documented escape hatch for offline / privacy / zero-marginal-cost runs, or as backstop against Fireworks pricing changes:

1. Download the trained adapter from Tinker (`tinker_cookbook.weights.download` / `merge_tinker_adapter_to_hf_model` → HF/PEFT).
2. Merge LoRA into base with `peft.PeftModel.merge_and_unload()`.
3. Convert to MLX with `mlx-lm.convert` (Qwen3 8B fits comfortably on his Mac; quantize to 4-bit if RAM matters).
4. Serve via `mlx_lm.server --model <merged>` *or* via Osaurus by registering the MLX model in its model directory.
5. Point `STYLE_MODEL_BASE_URL=http://localhost:1337` and `STYLE_MODEL=<merged>` in env.

### Phase 7 (optional, deferred): Multi-signal preference-tuning

If Phase 5's SFT-only styler plateaus, move to DPO with a multi-signal reward (blocks reward hacking on any single channel):

- **R_voice**: trained voice-classifier `p_dan` (from Phase 4a; higher = more Dan). **Must** be trained on a held-out by-POST split disjoint from the DPO data, else the reward is fit on its own training set. It is the *optimised* channel, so it carries the Goodhart risk — the other three are its independent-failure-mode guards.
- **R_vale**: Vale slop warnings per 1k words (fewer is better).
- **R_judge**: LLM-as-judge "is this Dan-shaped" score (higher is better) — the key orthogonal channel (different *kind* of signal, not just different rows).
- **R_similarity**: embedding cosine to Dan's nearest corpus paragraph by topic (guardrail against generic-de-slopping).

Reward = weighted sum (or strict "must pass all four"). The multi-signal design is the structural anti-reward-hack: a held-out split removes data-leakage, but only orthogonal *kinds* of signal (judge, Vale, eyeball) catch a policy over-optimising R_voice into the embedding's blind spots. **Tinker** is the platform — GA, Dan has an account, and its cookbook ships DPO/preference/RL primitives (`tinker_cookbook.preference`/`.rl`) the multi-signal reward maps straight onto (see decisions.md#D5). Budget: ~$50–100 in Tinker compute. Skip entirely if Phase 4 eval shows SFT is already good enough.

## Alternative paths (when to deviate)

- **OpenAI fine-tune (gpt-4.1-mini)** — switch only for the absolute lowest-ops path. Loses local-host exit; ~$125 all-in. decisions.md#D2.
- **Larger Mac-runnable base** (Qwen3.6-35B-A3B, Nemotron-Cascade-2-30B-A3B, Qwen3.5-122B-A10B, Llama 3.3 70B) — switch if Phase 4 eval shows ceiling effects on 8B. Same pipeline, ~5× cost. decisions.md#D2.
- **DiffusionGemma** — speculative; revisit only if autoregressive styler plateaus on *structure preservation* specifically. decisions.md#D8.

(Training-platform deviations — Together fallback, Unsloth/DIY-GPU — are no longer live forks; they're parked in decisions.md#D1/#D5/#D6 now that Tinker is the default trainer.)

## Budget summary

| Item | Cost |
|---|---|
| Synthetic pairs M1+M2+M3 (matrix, Sonnet-weighted) | $180–290 |
| → cheaper variant: Haiku-weighted matrix | $100–170 |
| Tinker LoRA training, Qwen3 8B (3 epochs) | $6–15 |
| Voice classifier (Phase 4a) — local, keyless | **$0** (was ~$50–60 Pangram distillation) |
| Optional one-shot Pangram independent cross-check | $0–5 (optional) |
| Eval LLM-as-judge | $5–10 |
| Inference (~10 files/week, Fireworks scale-to-zero) | <$5/month |
| *Optional Phase 7 DPO on Tinker* | *+$50–100* |
| *Optional larger base (35B / 70B etc)* | *+$35–80* |

**Default upfront (8B, Sonnet-weighted matrix, with Phase 4a)**: ~$200–330 (down ~$55–65 — the voice classifier replaced the Pangram-distillation enabler at $0). Haiku variant: ~$110–210. Larger base + Phase 7 upper bound: ~$300–470. All inside the "smart spend" envelope. Synthetic pairs and the trained voice classifier amortise across all subsequent training runs, hyperparameter sweeps, and the optional Phase 7.

## Critical files to modify or add

- **New**: `src/livingthing/workflow_style.py` (mirrors `workflow_preen.py`)
- **New**: `src/livingthing/bin/ai_style.py` (CLI entry, mirrors `bin/ai_preen.py`)
- **New**: `src/livingthing/bin/ai_style_pairs.py` (synthetic-pair generator for Phase 2)
- **New**: `src/livingthing/bin/ai_style_eval.py` (eval harness for Phase 4)
- **New**: `_training_pairs/` already exists from Phase 1; add `synthetic.jsonl` (generated) and `runs/<timestamp>/` (per-train artifacts)
- **Modify**: [`src/livingthing/ai_core.py`](../src/livingthing/ai_core.py) — make `call_edit` accept a per-call model override (already env-configurable; new tool wants per-call for A/B).
- **Modify**: [`pyproject.toml`](../pyproject.toml) — add `ai-style`, `ai-style-pairs`, `ai-style-eval` script entries; add `peft`, `together` as optional `[style]` deps.
- **Modify**: `CLAUDE.md` — add the new commands to the CLI table.
- **Reuse (no changes)**: [`pandoc_mask.py`](../src/livingthing/pandoc_mask.py), [`qmd_core.py`](../src/livingthing/qmd_core.py), [`format_sentences.py`](../src/livingthing/format_sentences.py), Vale config.

## Verification plan

1. After Phase 2: spot-check 10 random pairs in `_training_pairs/synthetic.jsonl`. Slop side obviously sloppy; target side obviously Dan; markdown structure preserved identically on both sides.
2. After Phase 3: run the eval harness (Phase 4); fine-tuned model produces fewer Vale warnings per 1k words than the prompt-only baseline.
3. After Phase 5: `uv run ai-style --file post/<recent-post>.qmd --dry-run` and read the diff. Output should trip fewer Vale warnings; preserve all code/math/links exactly on raw markdown; preserve `〈MASKED_*〉` sentinels under `--mask` (if it doesn't, that's the signal to add M4); not hedge or signpost.
4. LLM-as-judge on a held-out batch of 20 inputs: target ≥60% preference for `ai-style` output over `ai-preen` output. If <60%, escalate to larger base or revisit the pair dataset.
5. Optional Phase 6: local-hosted model produces byte-identical output to the Fireworks-hosted one on the same inputs (modulo sampling temperature).

## Open questions to revisit during execution

- Whether to include `polish:` / `novelty:` frontmatter as conditioning tokens during training (could let the styler emit different polish levels at inference). Probably out of scope for v1.
- Whether to train two adapters (one for blog-post register, one for notebook register) or one model that handles both. Default v1: one model; revisit if Phase 4 eval shows register-specific weaknesses.
