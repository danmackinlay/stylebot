# Design decisions: `ai-style` fine-tuned prose styler

Companion to [`ai-style-fine-tune.md`](./ai-style-fine-tune.md). Carries the *why* for each non-obvious choice so any of them can be re-evaluated later without re-reading the original planning conversation. Each entry: **decision → alternatives considered → deciding constraint → re-evaluate-if**.

## D1: Train on Tinker, serve on Fireworks (fallback: local MLX → Modal)

**Decision (consolidated 2026-06-30)**: train the LoRA on **Tinker**; serve scale-to-zero on **Fireworks**; fall back to **local MLX** on the Mac (the realistic fallback) and **Modal** as a distant container fallback. One trainer, one server, two named fallbacks — the old multi-provider matrix is gone.

**Why these two** (the deciding constraints are unchanged): a personal CLI run on a few files/week needs **scale-to-zero serving** (no idle burn) *and* **adapter download** (to keep the Mac-local exit + cross-platform serving open). Tinker now satisfies the training side cleanly — it's GA (account in hand), managed (no GPU to own/rent), and documents weight download → HF → MLX (`tinker_cookbook.weights` / `merge_tinker_adapter_to_hf_model`); same platform graduates SFT → DPO/preference for Phase 7 (see D5). Fireworks accepts externally-trained LoRA uploads and supports `--min-replica-count 0` (~5m–1h idle window; first request after idle returns `503 DEPLOYMENT_SCALING_UP` for the client to retry while the 8B-class adapter wakes — tens of seconds).

**Fallbacks**: download the Tinker adapter → merge → MLX → `mlx_lm.server`/Osaurus for fully-local, zero-marginal-cost serving (privacy/offline). Modal (scale-to-zero containers) is the cloud fallback if Fireworks pricing/availability ever changes.

**Parked alternatives** (one line each, so they don't sprawl): *Together* train+serve — dedicated endpoints only (~$4.7k/mo idle), wrong cost shape; kept only as a training fallback if Tinker's adapter→Fireworks handoff disappoints (Together's `GET /finetune/{id}/download` is long-confirmed). *Fireworks-train* — no documented weight download (email support), loses the Mac exit. *Modal/RunPod DIY + Unsloth/axolotl* — superseded by managed Tinker (see D6).

**Re-evaluate if**: Tinker's adapter→Fireworks/MLX handoff disappoints on the first run (→ Together training fallback); or Fireworks changes its scale-to-zero terms (→ Modal/local).

## D2: Qwen3 8B as the primary base

**Decision**: Qwen3 8B Instruct as the base model for the LoRA SFT.

**Alternatives considered**:

- **Qwen3.6-35B-A3B** — Dan's current agentic daily-driver on the Mac (~22GB MXFP4). In Together's catalog at specialty-tier pricing.
- **Nemotron-Cascade-2-30B-A3B** — long-context Mamba-MoE hybrid.
- **Qwen3.5-122B-A10B** — still single-Mac runnable via JANG quants.
- **Llama 3.3 70B** — the canonical "next size up." (Illustrative; Dan is genuinely model-family-agnostic.)
- **OpenAI fine-tune (gpt-4.1-mini)** — closed weights, ~$125 all-in, loses local-host exit, GPT base voice may leak through.

**Deciding constraint**: must run locally on the M-series Mac (MLX or GGUF) for the Phase 6 exit; cheap enough to retrain iteratively (~$7/run for 8B vs ~$43 for 70B); large enough that a LoRA can meaningfully move the style needle without obliterating general competence. 8B hits all three. The 35B+ candidates fit on the Mac but cost ~5× per training run and slow iteration speed for v1.

**Re-evaluate if**: Phase 4 eval shows ceiling effects (the 8B styler still hedges or over-structures after training). The upgrade should pick whichever base the eval reveals as bottleneck-limited, not the placeholder "70B."

## D3: Source-agnostic framing ("Dan-ifier" not "de-Claude-er")

**Decision**: train on a *matrix* of slop sources, not just Claude. The system prompt and all docs say "rewrite in Dan's voice," never "undo Claude." Target = constant Dan-voice; sources = matrix.

**Alternatives considered**:

- **Claude-only training mix.** Simpler, cheaper, and Claude is Dan's real inference distribution. But two problems: (a) the styler is itself an open model (Qwen3 8B) so at generation time its prior is open-model-shaped — if the training distribution has no open-model slop, the styler never learns to clean up after its own family [@Antoun2023Text]; (b) "Claude" is not one distribution — Haiku / Sonnet / Opus drift across revisions; pinning to one version overfits a moving target.

**Deciding constraint**: the styler must work on (i) Claude drafts (the common case), (ii) GPT/Gemini paste-ins, (iii) the user's local open-model output, (iv) the user's own rough notes. Source-agnostic framing covers all four without architectural changes. AI slop is two layers — a large shared core (cross-model-convergent: throat-clearing, "it's worth noting," tricolon, signposting, hedging, bullet sprawl — produced by RLHF + overlapping instruction-tune data + shared web corpus in Claude, GPT, Gemini, Qwen, Llama alike [@Attar2026Systematic]) plus a thin model-specific layer. Training to undo the shared core transfers across sources because the target is constant.

**Citations** (these also appear inline in the notebook's "Whose slop?" section):

- [@Attar2026Systematic] — 284 features × 27 LLMs; empirical grounding for the two-layer slop model.
- [@Antoun2023Text] — cross-model detection study; cross-LLM transfer is partial, supports the "include open-model slop on the input side" argument.
- [@Paneru2026Please] — AI→human style transfer; closest published analog to this whole project, and explicitly argues for multi-generator training data.
- [@Soto2025Attacks] — author fingerprints persist under paraphrase; supports the "target-author framing" (Dan-ifier).

**Re-evaluate if**: Phase 4 cross-source slice shows the styler generalises *too* much (loses Claude-specific fixes that mattered) or *too little* (cross-source slice fails badly — training mix leaned too Claude-heavy).

## D4: Raw markdown only; defer mask-preservation

**Decision**: train on raw markdown (code fences, math, links inline). Do not include masked variants upfront (M4 method deferred).

**Alternatives considered**:

- **Include masked-variant pairs from the start** (M4 method). Doubles the dataset, adds complexity to the synthetic-pair generator, and prematurely couples the styler to `ai-preen`'s masking scheme.

**Deciding constraint**: mask preservation is strictly easier than markdown preservation. Mask sentinels (`〈MASKED_*〉`) are random-looking copy-through tokens; an 8B base model that reliably preserves them won't blink. Markdown structure (code fences, math, links) is where the real edit hazards live. An 8B that reliably preserves raw markdown will trivially preserve mask sentinels. Also aligns with the stated direction of eventually dropping masking from `ai-preen` entirely.

**Re-evaluate if**: Phase 4 eval shows masked-input runs degrade (lost mask tokens, prose drift on masked chunks). Marginal cost to add M4 supplement: ~$10–15 (masking pass over existing pairs + one retrain run).

## D5: Tinker — the training platform (SFT v1 + Phase-7 preference/RL)

**Decision**: Tinker is *the* trainer (D1 makes the call; this entry holds the detail). One platform from the simple first SFT run through any later preference/RL phase — no migration.

**Context** (refreshed 2026-06-30; [docs](https://tinker-docs.thinkingmachines.ai/tinker/)): Thinking Machines' managed training API — **GA, self-serve, account in hand** (the old wait-list is gone). You write a CPU-side loop (data + loss); they run distributed training. LoRA-only ("LoRA Without Regret" argues parity, esp. for RL). Models 1B–1T+ dense + MoE (Qwen3/3.5/3.6 incl. 397B, Kimi K2.6, Nemotron, DeepSeek-V3.1, GPT-OSS, VLMs) — live catalog/rates on Models & Pricing (~$0.40/1M train tokens for an 8B-class model, last checked). Weight download → HF → MLX (`tinker_cookbook.weights` / `merge_tinker_adapter_to_hf_model`), so the Phase-6 local exit holds.

**Why it carries Phase 7**: low-level primitives (`forward_backward`/`optim_step`/`sample`/`save_state`) **plus a cookbook with first-class DPO/preference/RL abstractions** (`tinker_cookbook.preference`, `tinker_cookbook.rl`, a DPO Guide + RLHF Pipeline). The Phase-7 multi-signal reward (R_voice/R_vale/R_judge/R_similarity) maps straight onto those. (SFT v1 uses the cookbook's supervised recipe — no loop-writing needed for the simple case.)

**Re-evaluate if**: Tinker's adapter export disappoints on the first run → Together is the parked training fallback (D1).

## D6: Unsloth — not needed (managed Tinker removes the rationale)

**Decision**: not Unsloth. Its whole value is fast/cheap LoRA on a GPU *you* hold (own or rent) — its old slot here was "the second round: rent GPUs for 10+ hyperparameter sweeps, marginal cost → free." **Managed Tinker removes that rationale**: there's no GPU to rent, and per-token training is cheap enough to iterate. Unsloth's headline 2×/70% speedups are **CUDA/Triton-only**, so it doesn't even serve the Mac-local path — that's **MLX** (`mlx_lm`), not Unsloth. And the convergence-debugging case resolves to Tinker's *own* cookbook (per-model renderers fix chat-template/masking; `hyperparam_utils` fixes LR), not a second stack.

**Reference-only value**: their notebooks/blog are a decent cross-check for known-good LoRA recipes (correct chat template, prompt-masking, target modules, LR) if a Tinker run misbehaves. A bookmark, not a dependency.

**Re-evaluate if**: we ever want to train an open base Tinker doesn't host, on hardware we own — and that hardware is NVIDIA, not the Mac.

## D7: the detector — train a Dan-vs-slop classifier directly (not Pangram, not distillation)

**Decision (revised 2026-06-30, BUILT)**: train a Dan-vs-AI-slop classifier *directly* on the content-matched `(slop, Dan)` pairs — a frozen **StyleDistance** style embedding + a logistic head, scored by a pure-Python dot product (`stylebot.classify`; generic trainer `stylebot.classify_train` / `ai-style train-clf`, blog wrapper `dan-style train-clf`). Use it everywhere downstream (eval, best-of-N, a future DPO loop). Keyless, free per call, no gating spend. The earlier decision (spend ~$50 distilling Pangram into a local classifier) is **superseded** — we already hold the labels for the *narrower, more on-target* question, so there is no reason to borrow a general human-vs-AI detector.

**Why direct beats distillation**: Pangram answers "human vs AI in general"; we care about "Dan vs slop". Every synthetic/edit pair is a `(slop, Dan)` couple over the *same content*, which is a supervised training set for exactly the narrow classifier — already in the corpus. A bake-off (StyleDistance vs LUAR vs Wegmann-CISR vs the mxbai semantic baseline, ranked by content-matched pairwise accuracy split by POST) picked StyleDistance at 0.78/0.72, beating mxbai's 0.75/**0.62** — confirming *style*, not topic, is the axis, and that the frozen style embedding is good enough to skip both Pangram and a fine-tune (Option D) for now.

**Alternatives considered**:

- **Distill Pangram** (the prior decision): ~$50 one-time, but borrows the wrong question and adds an external dependency for the labels. Dropped.
- **Live Pangram API calls** in the hot loop: rate/latency/cost-bound. Never.
- **Skip a statistical signal entirely** (Vale + judge + eyeball only): loses the cheap, reproducible voice score. The trained classifier gives it for free.

**The reward-hacking reasoning, corrected** (this is the subtle part). Two distinct risks, conflated in the old note:
1. **Data-leakage circularity** — the detector having seen the styler's training rows. *Fixed by a held-out by-POST split* (`dan-style train-clf --holdout-frac/--holdout-posts`); use one shared partition across styler-train / detector-train / eval.
2. **Goodhart / proxy over-optimisation** — optimising the styler against a frozen proxy pushes its outputs into the proxy's blind spots *regardless of how the proxy was trained* (Gao et al.). A fresh split does **not** fix this. The defence is signals with **independent failure modes** — R_judge + eyeball (a different *kind* of signal), not a different split of the same model.

The old "galaxy-brained reframe" (Goodharting Pangram ≈ de-slopping toward Dan) is *partly* true and weakens risk #1, but it does not retire risk #2 — so the multi-signal reward (R_judge/R_vale/R_similarity alongside R_voice) stays load-bearing, not just suspenders. Best-of-N with small N is gentle (KL ≈ log N); DPO/RL needs the orthogonal channel in the loop.

**Citations**:

- [@Emi2024Technical] — Pangram technical report. Transformer-based supervised classifier trained on ~1M human/AI documents with hard-negative mining and active learning.
- [@Jabarian2025Artificial] — Chicago Booth BFI Working Paper 2025-116. Independent benchmark: "Pangram is the only detector that meets a stringent policy cap (FPR ≤ 0.005) without compromising the ability to accurately detect AI text."

**Re-evaluate if**: the frozen-embedding head plateaus on the boundary cases that matter most (lightly-humanised hybrids) — then escalate to **Option D** (contrastively fine-tune a style encoder on our own pairs, which is how StyleDistance itself was trained), not back to Pangram. Add a one-shot Pangram cross-check only if an *external, independent* opinion is wanted before trusting the detector as a reward.

## D8: DiffusionGemma (speculative, not v1)

**Decision**: not v1. Revisit only as a Phase-7-ish detour if a specific failure mode appears.

**DiffusionGemma context**: structure-preserving constrained rewrite is exactly what discrete-diffusion LMs are built for — bi-directional canvas attention conditions on both sides of a span, so "keep the markdown/identifiers, change the register" is the *native* operation rather than something coaxed out of an autoregressive model. Prior art for diffusion text style transfer: [Lyu et al. 2023, StylePTB](https://aclanthology.org/2023.repl4nlp-1.6); [DDOT infilling, arXiv:2506.13579](https://arxiv.org/abs/2506.13579). Tunable via [Unsloth's DiffusionGemma support](https://unsloth.ai/docs/models/diffusiongemma), Google's Hackable Diffusion (JAX), NVIDIA NeMo, and the `ddm-sft` mask-and-denoise + LoRA recipe. Dan already runs [`OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8`](https://huggingface.co/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8) locally (see [`local_llm_mac.qmd#models-diffusion`](../notebook/local_llm_mac.qmd#models-diffusion)).

**Deciding constraints**:

- Output quality trails standard Gemma 4 (Google's own admission).
- 26B / ~4B-active is bigger than the Qwen3-8B default; harder to retrain cheaply.
- Diffusion serving (schedule-owned sampling, denoising-step budget) doesn't fit the Together-train → Fireworks-scale-to-zero autoregressive economics this plan is built on. Would need a different serving stack.

**Re-evaluate if**: the autoregressive styler plateaus on *structure preservation* specifically (lost code fences, mangled math, dropped link syntax) — the one axis where the diffusion prior should genuinely help. Not the right reach for "still hedges too much" or "voice isn't sharp enough."
