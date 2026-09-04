# Design decisions

The *why* behind each non-obvious choice, so any of them can be re-evaluated
later without re-reading the original planning conversation — and the record of
what we believed and got wrong. Git preserves what changed; it does not preserve
why an alternative was rejected.

Entries D1–D8 keep their original **decision → alternatives → deciding
constraint → re-evaluate-if** shape. Entries D9–D25 are compressed from the
per-phase design plans that were retired once their work shipped: what we
thought, what happened, what we concluded. D26 onward return to the full shape.
Measurements live in [`findings.md`](findings.md).

**This file and [`findings.md`](findings.md) are the two that keep their
numbers.** Everywhere else in the repo, a corpus count or a run id is a claim
about the present that git cannot keep true, and D52 removed them in favour of
`ai-style status`. Here they are dated records of what was measured or believed
on a day, and editing one to match today would destroy the thing the file is
for.

Everything numbered `D…` was **decided and acted on**. A `P…` entry is a
*proposal*, written in the same shape so that accepting one is an edit rather
than a rewrite, and followed by its own verdict entry once one lands. None are
open as of 2026-09-03: P1, the last, was rejected. It had lived under an "Open
proposals" heading placed mid-file, which silently swallowed D31 onward as its
subsections; the heading is gone and P1 now files beside its verdict.

## D1: Train on Tinker, serve on Fireworks (fallback: local MLX → Modal)

**Decision (consolidated 2026-06-30)**: train the LoRA on **Tinker**; serve scale-to-zero on **Fireworks**; fall back to **local MLX** on the Mac (the realistic fallback) and **Modal** as a distant container fallback. One trainer, one server, two named fallbacks — the old multi-provider matrix is gone.

**Why these two** (the deciding constraints are unchanged): a personal CLI run on a few files/week needs **scale-to-zero serving** (no idle burn) *and* **adapter download** (to keep the Mac-local exit + cross-platform serving open). Tinker now satisfies the training side cleanly — it's GA (account in hand), managed (no GPU to own/rent), and documents weight download → HF → MLX (`tinker_cookbook.weights` / `merge_tinker_adapter_to_hf_model`); same platform graduates SFT → DPO/preference for Phase 7 (see D5). Fireworks accepts externally-trained LoRA uploads and supports `--min-replica-count 0` (~5m–1h idle window; first request after idle returns `503 DEPLOYMENT_SCALING_UP` for the client to retry while the 8B-class adapter wakes — tens of seconds).

**Fallbacks**: download the Tinker adapter → merge → MLX → `mlx_lm.server`/Osaurus for fully-local, zero-marginal-cost serving (privacy/offline). Modal (scale-to-zero containers) is the cloud fallback if Fireworks pricing/availability ever changes.

**Parked alternatives** (one line each, so they don't sprawl): *Together* train+serve — dedicated endpoints only (~$4.7k/mo idle), wrong cost shape; kept only as a training fallback if Tinker's adapter→Fireworks handoff disappoints (Together's `GET /finetune/{id}/download` is long-confirmed). *Fireworks-train* — no documented weight download (email support), loses the Mac exit. *Modal/RunPod DIY + Unsloth/axolotl* — superseded by managed Tinker (see D6).

**Re-evaluate if**: Tinker's adapter→Fireworks/MLX handoff disappoints on the first run (→ Together training fallback); or Fireworks changes its scale-to-zero terms (→ Modal/local).

**Phase-3 build note (2026-07-21)**: Tinker's current export docs document the PEFT-adapter path (`weights.download` + `weights.build_lora_adapter` → `adapter_config.json` + `adapter_model.safetensors`) and show serving via **vLLM/SGLang only** — neither Fireworks nor MLX appears in their deployment tutorials. The adapter is standard PEFT, so both handoffs *should* work, but they are now a **Phase-4 verification item**, not a documented path. Also: cookbook checkpoints default to a 7-day TTL (the final checkpoint is kept indefinitely); `ai-style train` downloads the adapter right after the run regardless.

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

**Superseded (2026-07-21, Phase-3 build)**: the primary base is now **`Qwen/Qwen3.5-9B`** — Tinker's 2026-06-12 retirements removed Qwen3-32B/30B and all Llama-3.x (so "70B" no longer exists there), Qwen3-8B survives but is last-gen, and Qwen3.5-9B is the current small dense at $1.463/1M train tokens (~$1.6/epoch at corpus scale — the ~3× rate over Qwen3-8B is immaterial). Step-up candidates if 9B lacks headroom: **Qwen3.6-27B** (dense) or **Qwen3.6-35B-A3B** (MoE — priced by active params and the friendliest shape for the Mac-local exit, per the alternatives list above). Trained no-think (`qwen3_5_disable_thinking`): the pairs carry no reasoning traces; a thinking A/B is a cheap follow-up run.

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

**DiffusionGemma context**: structure-preserving constrained rewrite is exactly what discrete-diffusion LMs are built for — bi-directional canvas attention conditions on both sides of a span, so "keep the markdown/identifiers, change the register" is the *native* operation rather than something coaxed out of an autoregressive model. Prior art for diffusion text style transfer: [Lyu et al. 2023, StylePTB](https://aclanthology.org/2023.repl4nlp-1.6); [DDOT infilling, arXiv:2506.13579](https://arxiv.org/abs/2506.13579). Tunable via [Unsloth's DiffusionGemma support](https://unsloth.ai/docs/models/diffusiongemma), Google's Hackable Diffusion (JAX), NVIDIA NeMo, and the `ddm-sft` mask-and-denoise + LoRA recipe. Dan already runs [`OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8`](https://huggingface.co/OsaurusAI/diffusiongemma-26B-A4B-it-MXFP8) locally (see [*Local LLMs on the Mac* §"Diffusion models"](https://danmackinlay.name/notebook/local_llm_mac.html#models-diffusion)).

**Deciding constraints**:

- Output quality trails standard Gemma 4 (Google's own admission).
- 26B / ~4B-active is bigger than the Qwen3-8B default; harder to retrain cheaply.
- Diffusion serving (schedule-owned sampling, denoising-step budget) doesn't fit the Together-train → Fireworks-scale-to-zero autoregressive economics this plan is built on. Would need a different serving stack.

**Re-evaluate if**: the autoregressive styler plateaus on *structure preservation* specifically (lost code fences, mangled math, dropped link syntax) — the one axis where the diffusion prior should genuinely help. Not the right reach for "still hedges too much" or "voice isn't sharp enough."

**MLX exit verified (2026-07-22, run 2)**: `bin/styler-mlx` merge→4-bit; served template string-identical to training under `enable_thinking:false`; 20-pair parity vs tinker mean Δ −0.010 (<0.03 gate), quantization drift same (<0.05, no 8-bit needed). Caveats: cookbook `build_hf_model` needs the serving-PEFT keys inverted to raw Tinker naming (shim in the script) and a local base-model dir to dodge transformers-4.57's missing `qwen3_5` arch.

---

The entries below are compressed from retired design plans. Each records a
belief we held, a wrong turn we took, or an alternative we rejected.

## D9: slop is generated by asking for polish, never by asking for slop

The `catalogue` slop strategy asked models outright for exaggerated
stereotypical-LLM tics, quoting the tics verbatim in the prompt. Removed
2026-07-07: samplers converge hard on quoted tokens, so every output opened
identically and the slop read as cartoonish — unlike the real drafting
distribution. The rule survives in the prompt code as *describe the texture,
never quote the tics*; this is the only record of where it came from. The
strategies that survived (`polish`, `engaging`, `casual`, `measured`, and later
`linkedin`/`verbose`) all ask for a *neutral improvement* and let the tics
arrive on their own. Old pairs stay resolvable via their data-dir's
`prompts.jsonl`; a bad variate is purged by filtering covariates, never by key
surgery.

The same principle was in the plan from the start and we drifted off it: slop is
"what a model emits when asked to write or expand", not "what it does when asked
to copy-edit". Light copy-edits don't surface the signature tics.

## D10: we assumed real AI prose is written at high reasoning effort — it isn't

The generator default was `reasoning-effort: high` on the belief that the slop
Dan actually cleans up comes from models thinking hard. The sweep measured it
false: the detector gap was flat for ~3× the wall clock (numbers in
`findings.md`). Default flipped to `off` on 2026-07-20; effort is now swept
*up* on suspicion rather than defaulted high. A related surprise from the same
sweep: a provider can silently ignore `effort=off` and reason anyway, so we
record the *requested* effort and let `completion_tokens` expose the lie.

## D11: `synth_key` is a cell identity, not a completion marker

A key means "this exact (model, strategy, effort, prompt, context, target) cell
already has a pair", never "this target is done". Any change on a design axis
mints new keys and *appends* alongside the old pairs; variants coexist and the
corpus is append-only. Consequences we kept re-deriving until they were written
down: changing a default re-keys flag-less runs (a budget decision, not data
damage); purging a bad variate is a covariate filter, never key surgery.

**Rejected: hashing position.** Determinism is content-anchored — hash the text,
never the index. Folding a `chunk_index` into the key breaks superset stability
the moment a paragraph is inserted above it.

**Two accepted weak spots**, taken knowingly: merge-mode packing lets an edit
early in a section re-key later chunks whose own text didn't change; and one
cell = one sample, so a genuine second draw at temperature needs an explicit
`--replicate` label rather than a re-run.

## D12: rendezvous hashing, not modulo, for generator-arm assignment

"Hash the content" left open *what you hash it against*. `hash(salt, text) % n`
made `n` a list length and the winner a *position*, so any change to the roster's
size or order reshuffled the corpus — a pure reorder of the model list moved
100% of it (table in `findings.md`). Replaced 2026-08-10 by scoring every arm as
`sha256(salt, arm_identity, text)` and taking the argmax.

**The identity is the arm, not the model.** The rotation is models × strategies
× efforts, so 6 models × 6 strategies is 36 arms sharing 6 model names; keying on
the name alone would score siblings identically, hand one of them every target,
and starve the rest.

## D13: session folding in keys — right, then wrong

Folding `session_id:turn` into the key was correct while window position was a
candidate *treatment*: the fold is what let the same text be resampled at
different context fills. The sweep then retired that treatment (drift r = −0.05),
leaving pure cost — any target-list drift re-chunked sessions and re-keyed nearly
every turn, so a multi-turn corpus build never deduped across runs and accrued
correlated near-duplicates. Dropped 2026-07-21; window fill survives as a
recorded covariate.

Landed with it, from the same day's post-mortem: a deep-session run had silently
discarded 46% of its planned work when the token budget bound before the turn
count. Turns a session can't run now reflow into fresh sessions, and the exit
summary reports any turn neither written, skipped, nor errored.

## D14: heading context is prepended to content, not carried as a fourth message

A pair may carry the section heading as fixed context, byte-identical on both
sides, so the styler learns to restyle the body *conditioned on* the heading.
Rejected: adding a fourth message — prepending preserves the
`[system, user, assistant]` triple the validator and the trainer depend on, and
the heading is real leading markdown anyway.

**Only `immediate` shipped.** The original design recommended the full ancestor
`breadcrumb` for faithfulness; only the immediate heading was ever built, and
the corpus is `immediate` throughout. Title/subtitle as context for the 15%
preamble was recommended off and never built.

**Forward-only.** The 254 pre-existing context-less pairs were left alone rather
than retro-fitted: source files had already changed, so re-deriving context by
text-matching would have been fragile. A corpus mixing prefixed and bare pairs
is fine, and the mix is itself useful augmentation.

## D15: the training/inference chunk scale was set an order of magnitude too high

The original plan pinned `STYLE_CHARS_PER_CHUNK ≈ 8000` and matched training to
it. Reality settled at 1500, and the reason is that Dan's *sections* are short —
a chunk can never cross a heading, so raising the budget alone does nothing.
The surprise that made this concrete: a 5.4KB single-chunk rewrite came back
byte-identical. Out-of-distribution length makes the styler copy its input, so
the inference budget is pinned to the training diet and read off the manifest,
not chosen by the operator.

## D16: synthetic-pair methods we planned and never built

Only M1, "paraphrase across a generator matrix", shipped. Retired unbuilt:
**M2** expand-a-skeleton-back-into-prose (mirrors the "flesh out these notes"
workflow where slop pain peaks); **M3** amplified slop from the curated pattern
catalogue (superseded by D9 — this is exactly the strategy that failed);
**M5** biasing sampling toward short dense prose where voice matters most;
**M6** detector-in-the-loop re-prompting when `P(slop)` came back too low.
M2, M5 and M6 remain cheap and unexplored; M3 is dead on evidence.

## D17: val split by POST, not by row

The Tinker cookbook's row-shuffle `test_size` would have leaked val posts'
sibling chunks into training — the same post's paragraphs are far from
independent. The val split is 10% of styler *posts*, seeded shuffle. The same
by-POST discipline is what makes the detector reward-safe (see D7 and the split
contract in `eval-harness.md`).

## D18: keep near-copy pairs for the styler, drop them for the detector

~19% of synthetic pairs barely changed Dan's text. They are label noise for the
detector and its dataset assembly drops them — but that filter must never be
copied into styler training, where they are the pairs that teach the model to
leave alone prose that is already fine. Restraint is half the job.

## D19: serving v1 is the Tinker sampling client, not Fireworks

The plan was Fireworks-first (scale-to-zero, adapter upload). Serving v1 instead
samples the manifest's checkpoint directly through Tinker: no new accounts, no
deploy, the checkpoint URI is pinned in the committed manifest, and cost is
per-token with zero idle spend, so scale-to-zero is moot. Fireworks and local
MLX became backends behind the same seam rather than the default path. Tinker's
own docs only ever demonstrated vLLM/SGLang export, so both handoffs were
verification items, not documented paths — MLX has since been verified (D8),
Fireworks has not.

## D20: in the editor, the human picks; the CLI's guard does not

The CLI's do-no-harm guard chooses a candidate for you. The editor's whole point
is that Dan chooses, so `rewrite` returns all candidates with none auto-selected,
and anchor violations **flag** (`anchors_lost`) rather than disqualify — sometimes
the deletion is the edit you wanted, and the flag says which. Escape aborts
without logging a preference: an abort is not a judgement.

Rejected for the sidecar protocol: LSP, which has no decoration concept at all.
NDJSON over stdin/stdout is ~30 lines a side with no framing dependency.

**Two honesty caveats recorded for whoever consumes the preference corpus.**
*Presentation bias*: candidates are shown sorted by detector score, so human
picks are anchored toward the detector's ranking — `scores` and `chooser` are
logged so it stays reconstructable. *Selection bias*: rewrites get requested
where the marker glows red, so the corpus over-samples high-P(slop) prose.
Probably the right bias, but not a random sample.

## D21: the `gpt`/`local` direct-API preset path, retired

All 4169 synthetic pairs were OpenRouter-served and every recorded
`context_window` came from the OpenRouter registry, so the preset path had never
produced a pair. Retired 2026-08-10 with its four flags. `openai_generator`
stays — `openrouter_generator` is built on it. Kept against the same evidence:
`--temperature` and `--top-p`, unset on every pair, because their importance is
*untested* rather than disproven and they were designed as recorded-but-unkeyed
covariates precisely so sampling can be swept without re-keying. Full audit and
the remaining open proposals: [`synth-flag-audit.md`](synth-flag-audit.md).

## D22: register conditioning, parked

Two ideas from the original plan, neither pursued: emitting `polish:`/`novelty:`
frontmatter as conditioning tokens so the styler could produce different polish
levels at inference; and training two adapters, one for blog-post register and
one for notebook register. One model handles both today. Revisit only if eval
shows register-specific weakness.

## D23: `--seed` was plumbed to the val split but not to the epoch shuffle

`shuffle_seed` was hardcoded to `0` at its construction site, so the flag reached
the by-POST val split and stopped there: any two runs differing only in `--seed`
trained on identical epoch order. Fixed 2026-08-09. Nothing downstream was
invalidated — the seed had never been varied in a landed run — but a seed-variance
comparison run before that date would have measured pure noise.

## D24: no dotenv loader; direnv is the environment mechanism

Bare `load_dotenv()` anchors its upward search on the *source* tree, walking up
from `src/stylebot/`, so a `.env` sitting beside the corpus was invisible while a
stale one beside the checkout won — the failure is silent and reads as a missing
key. python-dotenv was dropped 2026-08 rather than pinned to an explicit path:
keys reach the process as ordinary environment variables, and direnv
(`direnv exec ~/Source/livingthing uv run ai-style …`) already scopes them to the
directory you run from.

## D25: the tinker SDK's TLS trust is patched, not configured

tinker 0.23.x routes HTTP through pyqwest (reqwest/rustls) constructed with
`tls_include_system_certs=False`, whose bundled roots reject the Google Trust
Services chain `tinker.thinkingmachines.dev` serves — `invalid peer certificate:
UnknownIssuer` on the very first auth-token call, first hit 2026-07-22. There is
no supported env or config knob, so `stylebot.tinker_tls` rebuilds the transport
with system certs and patches the SDK's private transport factory. Best-effort by
design: if the internals move it warns by name rather than failing the run.

**Retired 2026-08-11**: tinker 0.25.0 ships the same fix upstream — its
`_default_pyqwest_transport` passes `tls_include_system_certs=True` itself, and
guards the *call* in `try/except TypeError` so an older pyqwest that rejects the
kwarg still constructs. That guard is the one thing our version lacked. The floor
moved to `tinker>=0.25.0` and `stylebot.tinker_tls` was deleted along with its
call sites in `train.py`, `infer.py` and livingthing's `bin/styler-mlx`. Verified
by rebuilding the SDK's own default transport with no patch applied and getting a
live handshake against `tinker.thinkingmachines.dev` — the same probe that
demonstrated the bug at 0.23.2, now returning HTTP 200. Re-open only if the floor
is ever walked back below 0.25.0.

---

D26–D30 come from the 2026-08-11 segmentation harmonisation, which found the
same markdown segmenter written seven times and pulled the surviving copies into
one contract. They use the full **decision → alternatives → deciding constraint
→ re-evaluate-if** shape.

## D26: no shared markdown package (yet) — `stylebot.markdown`, not an `mdcore` release

**Decision (2026-08-11)**: merge the duplicated segmenters *inside* stylebot into
`stylebot.markdown` and stop there. Do not extract a third repo.

**Alternatives considered**:

- **Extract `mdcore`** — a standalone package both stylebot and livingthing
  depend on. The obvious end state, and the one the duplication argues for.
- **Vendor a copy** into livingthing with a sync script. A build step, and it
  fails silently when nobody runs it.

**Deciding constraint**: of the **seven** implementations found — five inside
stylebot, `livingthing.qmd_core` (which duplicated the scan locally as well),
and the marker's TypeScript port — a Python package unifies **exactly one**.
Five were merged into `stylebot.markdown` and needed no package to do it; the TS
port cannot consume Python at all. So the release buys unifying ~130 lines
between two sibling checkouts, against a third repo, a release cadence, a
version to pin, and a new version-skew failure mode where the two repos are
built against different `mdcore`s. The conformance fixture (D28) is needed
*regardless*, because the TS port exists — so the package's marginal gain over
the fixture is small. Extraction stays cheap if it ever earns its place:
`markdown.py` is ~350 lines of stdlib-only code with no stylebot imports.

**Re-evaluate if**: a **third** Python consumer appears; the surface passes
~500 lines or grows inline parsing (emphasis, links, tables — where a real
CommonMark dependency starts to look better than either option); or there is a
reason to publish it for people who are not us.

## D27: exclusion, not masking, for block constructs; the anchor guard for inline

**Decision (2026-08-11)**: protected *blocks* are cut out of the text the styler
sees (see the protection principle in `stylebot.markdown`). Protected *inline*
constructs — code spans, inline math — stay in the prose and are defended after
the fact by the anchor guard, not by sentinel masking.

**Alternatives considered**:

- **Sentinel masking at inference** — replace `` `x` ``/`$x$` with `〈MASKED_*〉`
  before sending, restore after. This is what livingthing's opt-in `--mask`
  does.
- **Always-on inline anchors** — keep the constructs raw but disqualify any
  candidate that loses one.

**Deciding constraint**: *asymmetric* masking — masking at inference over an
adapter trained on raw text — is out of distribution by measurement. Over the
4624-pair corpus, raw inline markdown is thoroughly in-distribution (links
77.3% of pairs, inline code 20.3%, citations 17.9%, inline math 15.8%) while
`〈MASKED_*〉` occurs in **0.5%** — and in those, only on the slop side. The
adapter has essentially never seen a sentinel as content. Always-on inline
anchors fail on a different measurement: they would disqualify a genuine
Dan-grade rewrite on **19.0%** of pairs against today's 6.4% (table in
[`findings.md`](findings.md)), i.e. triple the rate at which `run` rejects every
candidate and keeps the input verbatim. So inline anchors are **off by default
in `rewrite_text`** (behind `--strict-anchors`) and **always on in
`rewrite_candidates`**, where a human sees the flag and decides.

This refutes asymmetric masking. It does **not** refute *symmetric* masking —
masking both sides at training and at inference — which remains open; see the
deferred experiment below. Note D4 already decided "raw markdown only"; this is
the same call, now with numbers behind it.

**Re-evaluate if**: the symmetric-masking experiment below is run and wins on
the judge signal; or a future corpus is generated with masking on, which would
move the distribution this decision rests on.

**Reinforced 2026-08-11** by an independent measurement (see the shape
constraints on the deferred experiment below, and `findings.md`): placeholder
retention is strongly model-dependent, and weakest in exactly the open-weight
band we serve from. That raises the cost of any masking scheme relative to
exclusion, which cannot be got wrong by a model at all — an excluded block is
never in the model's hands to mangle.

### D27 amendment (2026-09-03, W2): exclusion has a packing consequence, and it is priced, declared, and not paid yet

Exclusion is not a local edit. `targets._post_targets` deletes protected blocks
**and then packs paragraphs**, so a passage either side of a figure becomes one
target — a passage the document does not contain. `infer._segment` cannot ever
present it: it breaks a chunk at every protected item. The adapter is trained on
"A+B (fence gone)" and served "A", then "B".

**Measured 2026-09-03**, over the blog's 467 selected posts and the 7,846-pair
synthetic corpus, which is what turns this from a curiosity into the dominant
train/serve skew in the system:

| | training (`_post_targets`, 4500 cross-heading) | serving (`_segment`, same diet) |
| --- | --- | --- |
| chunks over the same prose | 933 | 4,278 |
| median chars | 4,027 | **434** |
| mean chars | 3,267 | 919 |

Same prose, cut into 2 pieces per post for training and 9 for serving. 4,646
protected items are scattered through those 467 posts (1,424 divs, 1,313
blockquotes, 1,114 fences, 795 display-math blocks) and 466 of the 467 posts
carry at least one. Group the served chunks across them and the median goes to
3,558 — i.e. essentially all of the gap is this. D15 says out-of-distribution
length makes the styler return its input verbatim, and run 4a was trained at
4,500 cross-heading.

**Alternatives considered, and both were priced before choosing.**

- **A: `_segment` groups prose across protected items, reassembling the block in
  place on output.** *Impossible, and the reason is structural rather than
  effortful.* The group is sent as A+B with the block deleted; the model returns
  one text; there is no anchor in it that says where A ended. The only
  reassemblies available are splitting the model's output on a paragraph-count
  heuristic — which silently relocates a figure relative to the prose that
  introduces it the first time a rewrite merges two paragraphs — or sending the
  block inline and asking the model to echo it, which is exactly what the
  protection principle forbids and what D27 above refutes on measurement.
  Rejected on correctness, not cost.
- **B: `_post_targets` stops packing across a deleted block.** *Correct, and
  expensive.* Implemented and measured: as `ChunkPolicy.pack_across_protected`,
  defaulting to True (today's behaviour). Flipping it, priced against the two
  corpus-stability arms as W5 re-measured them on the same day:

  | arm | today | with B |
  | --- | --- | --- |
  | section-bounded (frozen) | 3,023 / 3,028 | 2,116 / 3,028 |
  | cross-heading | 4,181 / 4,194 | 964 / 4,194 |

  **4,124 distinct cells orphaned, 57% of the synthetic corpus.** Orphaned, not
  deleted: `synth_key` keys on target text, so those pairs stay trainable and the
  *new* cells simply go unfilled until a synth run pays for them. The
  section-bounded 907 cannot be refilled at all, since `CROSS_HEADING` is live
  policy.

**Deciding constraint**: B is the right shape and its bill is a paid synth run
plus a re-measure of both corpus-stability floors, which is Dan's call and not a
refactor's. A is unavailable at any price. So the asymmetry **stands, as a
declared parameter with a default**, and the reason it is tolerable while the
other three A1-class findings were not is a real difference in kind: the other
three put constructs in the model's hands that the corpus never contained
(callout prose rewritten but never trained, a heading prefix training never saw,
a runaway fence protected on one side only). This one puts nothing new in the
model's hands. It is a **scale** mismatch, of exactly the kind the mixed-scale
curriculum (1500/3000/4500) exists to absorb — and the measurement above says the
curriculum is aimed at the wrong end of the distribution, which is a training-diet
question rather than a correctness one.

`tests/test_protection_parity.py` now pins both halves: that the two sides agree
under `pack_across_protected=False`, and that the default's disagreement is
exactly the join across a deleted block and nothing else.

**Re-evaluate if**: a synth run is being paid for anyway (flip the default in the
same batch, then re-measure both floors in
`livingthing/tests/test_corpus_stability.py`); or the next scale curriculum is
chosen — in which case pick it against the *served* distribution (median 434,
mean 919, p90 2,658) rather than against what `_post_targets` cuts, because the
served one is what the adapter meets.

## D28: a conformance fixture, not a shared import, pins the second implementation

**Decision (2026-08-11)**: `livingthing.qmd_core`'s copy of the segmenter is
**permanent**, and is pinned rather than deleted.
`tests/fixtures/segmenter_conformance.{qmd,json}` is one document plus one
checked-in expectation, asserted from both repos, with a
`STYLEBOT_REGEN_SEGMENTS=1` regeneration hatch and a `fixture_sha256` so a stale
copy of the document in the consuming repo cannot agree by accident.

**Alternatives considered**:

- **Delete the copy; `import stylebot.markdown`.** The strong guarantee — one
  implementation cannot drift from itself.
- **Leave it forked and unpinned.** The status quo ante, and how five copies
  diverged in the first place.

**Deciding constraint**: livingthing is making stylebot **optional** (D30), so
its editing and publishing path must segment markdown with stylebot
uninstalled. The import is therefore unavailable, not merely unattractive. Given
that, the choice is between undetected drift and detected drift, and the fixture
is the same instrument that already pins the TypeScript port
(`tests/test_marker_segmentation.py`). It worked immediately: the fixture
exposed livingthing's missing `~~~`-fence support, which was closed the same
day.

**This is a weaker guarantee than one implementation, and should be read as
one.** A fixture pins the behaviours it happens to exercise; a construct absent
from the document is unpinned, and both sides can drift on it freely. The
document is the contract, so extending the contract means extending the
document.

**Re-evaluate if**: livingthing's editing path stops needing to run without
stylebot (then delete the copy and import); or the two implementations diverge
on something the fixture did not cover, which is the fixture's failure and an
argument to grow the document rather than abandon the approach.

**SUPERSEDED 2026-09-03 (W3), by its own re-evaluation clause.** D30 fell (see
below), so the import became available and the copy was deleted.
`livingthing.qmd_core.segment_for_edit` is now a thin call to
`protected_spans(text, block_only=False, raw_html=True, html_comments=True)` —
the `livingthing` named configuration, which is exactly what D29 was for — and
the blog's `tests/fixtures/segmenter_conformance.{qmd,json}`, its
`fixture_sha256` gate and `tests/test_segmenter_conformance.py` are gone with it.
Verified before deleting: the conformance test passed unmodified against the
import-based implementation, and `segment_for_edit`'s protected spans are
byte-identical on **all 1758 blog `.qmd`** (0 files differing). The weaker
guarantee this entry warned about was the right call while it held, and its cost
was paid twice — a frontmatter stamper rewrote the `.qmd` half of the matched
pair in `128308fa55` and again in `2815a291fc`, and since neither repo runs
pytest in CI the gate only ever fired on somebody's laptop.

What survives: **D29 stands, and is what made the deletion cheap.** The three
keyword arguments now sit at one call site instead of being recorded as named
configurations in a fixture. Two loose ends this entry's disappearance leaves —
stylebot's own `tests/fixtures/segmenter_conformance.{qmd,json}` and
`tests/test_segmenter_conformance.py` now assert stylebot against its own output,
a transcript rather than a spec, and the module docstring of
`stylebot.markdown` still describes livingthing as a permanent SIBLING
implementation. Both are stylebot-side, and neither is load-bearing; the fixture
is still the honest place to record what the two configurations mean, so the
question is whether it earns its regeneration hatch now that only one repo reads
it. (`tests/test_marker_segmentation.py` pinned a genuinely separate
implementation; W4 deleted that one the same day — D44 — so no fixture in either
repo now stands between two implementations of block segmentation, because there
are no longer two.)

## D29: policy differences are parameters — `block_only`, `raw_html`, `div_is_prose`

**Decision (2026-08-11)**: where stylebot and livingthing legitimately want
*different* protection, express the difference as a keyword argument on
`protected_spans` rather than as two implementations or one forced compromise.
The conformance fixture then records **named configurations** — `stylebot`
(block-anchored math, no raw HTML) and `livingthing` (inline `$$` and
`<script>`/`<style>` protected) — both produced by `stylebot.markdown` alone.

**Alternatives considered**:

- **Force agreement**: pick one behaviour and make both sides live with it.
- **Let each side keep its own regex set** and only assert on the intersection.

**Deciding constraint**: the differences are real and derive from what each side
*does with* a protected span. livingthing **reassembles** them, so protecting an
inline `$$k$$` costs it nothing and buys safety. stylebot **deletes** them, so
protecting an inline `$$k$$` joins the remnants and mangles the sentence — the
A5 defect, which reached the corpus (`findings.md`). Forcing agreement would
have made one side wrong on purpose. Same shape for `raw_html`: only a
whole-document reflower needs `<script>`/`<style>` protection, and turning it on
in stylebot would move a corpus generated without it, so it defaults off.
`div_is_prose` is the same again — which div classes bear prose is a Quarto
convention, and CLAUDE.md's blog/generic boundary keeps it out of stylebot.

Parameterising is what makes the fixture *able* to record both sides honestly
instead of asserting a false agreement, and it is why one implementation can
serve two policies without forking.

**Re-evaluate if**: a parameter accumulates a third or fourth value, or the
configurations stop being nameable — at that point the sides have genuinely
different requirements and a shared implementation is fiction.

## D30: stylebot is livingthing's *optional* `styler` extra

**Decision**: livingthing declares `styler = ["stylebot[classifier,trainer]"]`
as an optional extra (carried by a default-on `styler` dependency group, since
uv 0.11 has no `default-extras`), not a hard dependency. `uv sync --no-group
styler` gives the lean install.

**Alternatives considered**:

- **A hard dependency.** One install, one lockfile, nothing to explain.

**Deciding constraint**: the blog's *editing and publishing* path — writing,
rendering, `ai-preen`, format-sentences, Zotero, the search index — must not
depend on ML tooling. Only `dan-style` and `bin/styler-mlx` reach for stylebot,
and they say so when it is missing. A hard dependency would put torch, the
tinker SDK and sentence-transformers in the way of rendering a blog post.

**This is the constraint behind D28**: because the extra can be absent,
livingthing's blog-core segmentation cannot `import stylebot`, so its copy of
the segmenter is permanent.

**Re-evaluate if**: the styler becomes part of the ordinary publish path rather
than an editing-time tool.

**SUPERSEDED 2026-09-03 (W3).** The decision conflated two things that turned
out to be separable: **stylebot** and **the ML tooling**. stylebot's base
install is `click` + `ruamel-yaml` + `openai`; torch, the tinker SDK and
sentence-transformers were always behind its `[classifier]`/`[trainer]` extras.
So the deciding constraint above — "a hard dependency would put torch, the
tinker SDK and sentence-transformers in the way of rendering a blog post" — was
never true of *bare* stylebot, only of the extras. What it actually bought was
a permanent second segmenter (D28), a hand-copied fixture, a sha gate and a
paragraph of skill documentation defending all three.

The line moved to where the weight is. `stylebot` is now a hard dependency in
livingthing's `[project].dependencies`; `styler = ["stylebot[classifier,trainer]"]`
keeps only the ML extras, still carried by the default-on `styler` group. The
constraint is intact and is now *tested* as what it always meant: the blog's
`tests/test_styler_optional.py` blocks `torch`/`tinker`/`tinker_cookbook` rather
than `stylebot`, and asserts blog-core imports, does real work, and drags none
of them in. Proof: a lean sync (`uv sync --no-group styler`, into a scratch
`UV_PROJECT_ENVIRONMENT` so a shared checkout is not disturbed) installs 227
packages against the full install's 243, imports `stylebot.markdown`, and fails
to import `tinker` or `tinker_cookbook`.

**One correction to the W3 brief while here.** Its done-criterion said `import
torch` should fail in the lean install. It does not, and never did: `uv tree
--invert --package torch` gives three parents and two of them are the blog's own
direct dependencies — `sentence-transformers` (for `index-blog`) and `pycid` →
`pgmpy`. Only the third, `tinker-cookbook`, arrives through stylebot. The lean
install has always carried torch, so torch was never what the `styler` extra
kept out; `tinker`/`tinker-cookbook` are. This does not change the decision — it
removes one of the arguments that had been made for the old one.

**The cost, and it is real**: a blog install now needs stylebot resolvable.
`[tool.uv.sources]` carries two sources partitioned by marker — the editable
`../stylebot` on darwin, a pinned git source everywhere else — because uv 0.12.7
permits exactly one source URL per package per marker environment (`uv.toml`
refuses a `sources` table outright; two unconditional or `group`-conditional
sources fail with "conflicting URLs … in all marker environments"; and
`uv pip install -e` over a git-sourced sync is undone by the next `uv run`).
That fixes CI and every Linux clone and leaves one gap: a fresh macOS clone
without the sibling checkout still has to clone it. Written up in the blog's
`CLAUDE.md`.

**Re-evaluate if**: stylebot's base dependencies grow something heavy (an
`openai` bump is fine; a torch-adjacent one is not — that would put the weight
back on the wrong side of the line); or stylebot is published to an index, at
which point the marker-partitioned sources collapse to an ordinary version pin.

---

## Deferred experiment: symmetric masking (D4's re-evaluation, still untested)

**The experiment**: mask inline markup — code spans, inline math, link targets,
citation keys — on **both** sides of every pair, train an adapter on the masked
corpus, and mask at inference too. Symmetric, unlike the asymmetric variant D27
refutes.

**Hypothesis**: inline markup is **20.5%** of target-side characters
(`findings.md`) and carries no style signal whatsoever — a URL is a URL in
anyone's voice. Masking it should concentrate the training signal on the prose
that does carry voice, and shorten the sequences into the bargain.

**Cost**: ~**$15** — a post-hoc transform over the pairs already on disk, so no
re-synthesis — plus one retrain (~$1.50 at run-2 scale).

**How to decide it**: on the frozen eval stratum, and **adopt only on a
judge-signal win**. A detector-only improvement is exactly the reward-hacking
failure [`eval-harness.md`](eval-harness.md) warns about in its own guardrail
section: the detector is a proxy, and masking changes the surface statistics the
proxy reads.

**Why it is written here and not built**: this is D4's stated re-evaluation
trigger, and that trigger has never been tested in either direction. D4 deferred
masking on the reasoning that mask preservation is strictly easier than markdown
preservation — true, and irrelevant to whether masking helps the *style* signal,
which is the question here.

**If it is ever run, the placeholder SHAPE is a parameter, not a detail**
(2026-08-11, evidence in [`findings.md`](findings.md) "Placeholder retention
across models"). Three constraints the experiment inherits:

1. **Make the token plausible in its slot, not distinctive.** A URL-shaped
   stand-in retained 68.9% where `〈URL_0000〉` got 45.9% and plain-ASCII
   `[[URL_0000]]` got 37.3% — the ASCII "fix" for exotic characters came last.
   That implies *different shapes per protected class* (a URL for a link target,
   a plausible identifier in backticks for a code span), not one uniform sigil,
   and a reserved-but-plausible content domain such as `example.invalid`.
   Note this cuts against `〈MASKED_*〉`, the shape the frozen `STYLE_SYSTEM`
   names — which is a reason to settle the shape *before* any masked corpus is
   generated, since the prompt cannot be changed afterwards without invalidating
   it (D2).
2. **Validate on the model we ship, not a frontier API.** Retention ranged
   100% / 37–69% / 0–13% across three models on identical inputs. Scheme choice
   only moves the outcome in the middle band, which is where open weights sit;
   a scheme tuned against a model that retains everything measures nothing.
3. **Give every placeholder a position component and a stable digest.** A
   content-addressed token collapses duplicate occurrences of the same URL,
   destroying the bijection and making reordering undetectable — observed as
   spurious duplicates in the source bench. `<hash>-<n>`, and never Python's
   salted `hash()`, which would mask the same input differently per process and
   break resume.

**What the same evidence does NOT change**: stylebot's block-level exclusion
(D27) and the anchor guard stand. Neither is a masking scheme — stylebot
substitutes nothing and has no restoration map — so the findings about sigil
shape have no surface here today. Two of the source document's warnings were
checked against our code and found already satisfied: `missing_anchors` compares
multisets, never a positional `zip()` (the diff bug that produced a wrong
conclusion there is structurally impossible here), and anchor failure fails
*closed*, keeping the input verbatim, where the source pipeline failed open. The
one warning that did land was truncation-vs-damage attribution, now fixed in
`infer.rewrite_text`.

## D31: stylebot merges the adapter; it does not quantize, patch configs, or bake templates

`train --download-adapter` writes a PEFT adapter that
`tinker_cookbook.weights.build_hf_model` — the merge it exists to feed —
**cannot consume**. stylebot's export is in serving key format
(`model.language_model.*`, `lm_head`); the cookbook expects raw Tinker keys and
applies that remap itself, so passing our own output doubles the
`language_model` prefix and the merge raises `WeightsMergeError`. The workaround
lived as 40 lines in livingthing's `bin/styler-mlx` — invisible to anyone who
installed stylebot and followed its own documented next step.

`stylebot.export` (2026-08-12) takes the merge, and stops there.

**Where the line falls, and why there.** The merge is mechanism: the key
inversion is a property of *stylebot's own* two artifacts, the base-model
resolution works around transformers not knowing a fresh architecture, and
`.partial`-then-rename is the same idempotency the rest of the pipeline uses.
None of that varies by who is running it.

Everything after the merge does vary. `mlx_lm.convert` is Mac-only;
`_VLM_MARKER_KEYS` is a Qwen3.5-config × vllm-mlx intersection; the no-think
chat-template substitution is keyed to one template's exact shape. Those are
*(model family × serving runtime)* facts, and hardcoding them here would put one
server's quirks inside generic tooling — the blog/generic boundary this repo
keeps (see D29, where the same argument made `block_only` / `div_is_prose`
parameters rather than behaviour).

**Rejected: port `styler-mlx` wholesale behind an `ai-style export-local`.** It
reads better in a demo and is worse in every other respect. It would make MLX a
first-class stylebot concern on a package that runs fine on Linux, and it would
bake `image_token_id` and a Jinja fragment into a repo that has no opinion about
either. The blog keeps its script; it now calls `merge_adapter` instead of
carrying its own copy of the inversion.

**Rejected: assume `lora_alpha == r` the way the script did.** True of every
adapter `train.py` produces, and the script's comment says so. But the rename
only preserves *meaning* while the scaling factor is 1, and an adapter from
elsewhere merged at the wrong scale is subtly wrong rather than broken — the
failure that does not announce itself. `check_adapter_scaling` refuses instead.

**Re-evaluate if**: a second consumer wants the quantize/finalize stages, at
which point they arrive as parameters (`strip_config_keys`,
`chat_template_edit`) with the caller supplying the values — not as constants.

## D32: the anchor guard is symmetric, but its two halves use different rules

**Decision**: `infer.anchor_delta` returns `(lost, gained)`, and `rewrite_text`
disqualifies a candidate for either. **Loss is a count decrease; invention is
key absence.** The asymmetry is deliberate, not an oversight.

The guard checked only loss until 2026-08-12 — the half the training corpus does
*not* teach. Measured over the 4,595-pair corpus through the production path:
117 pairs (3.1% of the anchor-bearing ones) hand the styler a target that
dropped an anchor, while 855 (18.6% of the corpus) hand it one carrying anchors
its source never had, half of those external URLs. Reverse-edit synthesis
produces that directly: a generator that drops a link while "improving" a
paragraph becomes, after the swap, an example reading *given text without links,
produce text with these links*. The failure the data most encourages was the one
nothing checked (stylebot#1, `_plans/anchor-integrity.md`).

**Why the halves differ.** Each occurrence of an anchor supports a separate
claim, so dropping one of two citations of the same key leaves a sentence
standing on nothing — loss must be a count rule. What invention guards against
is different: a destination the model conjured and cannot have verified.
Repeating a URL the source supplied conjures nothing; it points somewhere the
source already vouched for.

**Rejected: count-increase for invention too**, for symmetry's own sake. It
condemns a rewrite that splits one sentence in two and re-cites the same link —
a legitimate edit — and over the real corpus it flags 17 pairs whose entire
offence is citing a source-supplied URL twice. Consistency of *form* between the
two rules would have cost accuracy on both.

**Rejected: making `rewrite_candidates` reject as well.** That seam returns a
slate for a human to pick from, and a human is exactly who can tell a real URL
from a plausible one. It gained an `anchors_gained` flag beside `anchors_lost`
and still disqualifies nothing (D20).

**The fabrication gate defaults ON, unlike its two siblings.**
`--max-transform-sim` and `--max-length-ratio` are default-off because each
takes a *threshold* — a number whose right value is a roster property, and which
`None` honestly declines to guess. `--drop-invented-anchors` is a boolean with no
defensible off position: a pair teaching the styler to conjure a URL has no use.
So the asymmetry in defaults follows the asymmetry in what the knob *is*.

**But it ships with `--gate-retries`, and a bare gate would have been a
regression.** Anchor invention is a property of the ARM, not the passage — 1.3%
(sonnet) to 74% (deepseek-v3-0324) across nine measured arms, with llama-3.3-70b
(51.5%) and qwen3.5-9b (37.9%) still on the live roster (`findings.md`). A gate
alone therefore does two things nobody asked for: it burns roughly a fifth of a
run's spend on cells it drops and re-drops on every subsequent run, and — worse
— it biases coverage *against link-dense prose*, which is the prose anchor
fidelity matters most for (rejection runs 16.1% at one anchor, 32.9% at five or
more). `--gate-retries N` redraws a rejected cell instead. A rejected draw is
paid for either way, so a redraw costs no more than the drop already did, and of
the 148 corpus targets that ever lost an anchor only 29 lost one under every arm
that saw them: ~80% of rejections are a redraw away from clearing. Applies to all
three gates, since they share the drop idiom.

**Rejected: migrating the 855 offending pairs out of `pairs.jsonl`.** The corpus
is a frozen record. `synth --drop-invented-anchors` stops new ones being written;
`infer.is_anchor_faithful` is a `selector` for `train.assemble_training_corpus`,
so a training run excludes them from the *signal* while they stay on disk as
evidence. No migration script, no re-keying — `_synth_key` is the resume
identity for 4,140 synthetic pairs and touching it orphans the corpus.

> **Amended 2026-08-17 — the blanket rule did not survive contact.** The
> synthetic stratum *was* culled: `scripts/prune_anchor_inventing_pairs.py`
> ships, 805 pairs went in the blog's `dacf57892f`, and the blog's RUNBOOK 6a
> documents it as policy. The reasoning is sound and narrower than "frozen":
> **culling a synthetic pair leaves its `synth_key` missing, so the next batch
> refills the cell under the gate**, which converges the corpus on
> fully-gated — whereas a *captured* pair records an edit Dan made and only git
> brings it back, so those are filtered at assembly and never culled. Keep the
> re-keying prohibition, which is what actually protects the resume identity;
> read "frozen record" as **cull synthetic, filter captured**. A plan drafted
> off the un-amended clause asserted the corpus had never been mutated, which
> was false.

**Superseded in part 2026-08-17: `is_anchor_faithful` is now symmetric.** Its
`lose`/`gain` keywords both default on, because the pair this decision was
written about has a mirror the gate never checked — a generator that *conjured*
a link while producing slop yields, after the swap, a target that deletes one.
Measured on the blog corpus: 306 synthetic and 52 captured pairs. Nothing about
the two directions' *rules* changes (loss is a count decrease, invention is key
absence — that is the substance of this decision); what changed is that the
corpus-side predicate now applies both, matching the guard at inference. The
matching synthesis gate is still one-sided, so a cull of the loss stratum must
wait for `synth._gate_verdict` to gain its mirror, or the refill re-mints it.

**Re-evaluate if**: `--anchor-log` data from real runs shows the styler damaging
anchors often enough to justify the masking scheme in
`_plans/placeholder-design-2026-08.md`, which is deliberately still unbuilt.

## D33: masking goes in everywhere, with a dropout fraction — and synthesis-vs-inference is not an asymmetry

**Decision**: mask anchors at synthesis (to stop generators deleting them), keep
the corpus **canonical and unmasked on disk**, and apply masking as a
deterministic transform at each consumption point — training assembly (for a
`1 − p` fraction), inference, and independently the classifier. Storage format
and masking policy are separate concerns; conflating them was an error in the
first draft of this entry.
Placement, evidence and ordering: [`masking-placement.md`](masking-placement.md).
The *encoding* is unchanged and still lives in
[`placeholder-design-2026-08.md`](placeholder-design-2026-08.md).

Step 3 of the anchor work was gated on data that did not exist (D32). The data
arrived on the first gated v2 run and is unambiguous: at 3000-char cross-heading
chunks, rejection on the residue ran 98% (gemma-4-26b), 99% (qwen3.5-9b) and
**100%** (qwen3.6-35b-a3b, 0 written of 105 draws); rejections delete a median of
5 anchors and up to 29; and redrawing recovered 2 cells from 106. The slop
prompts already instruct verbatim link preservation, so no prompt fix remains.

**Why it collapses at v2 scale specifically.** A chunk passes only if every
anchor survives, so P(pass) ≈ s^k. Cross-heading packing moves median k from 2 to
11. **This corrects an earlier reading in `findings.md`** that the rejection
curve *plateaus* and v2 would therefore be affordable: that plateau came from
v1's 600-char chunks, where high-k cases were rare, and the requirement is
exponential in k rather than asymptotic.

**Rejected: masking everywhere with no dropout.** Simpler and internally
coherent, and the reason against it is not "keep the corpus natural" — that is
not a goal. It is that two documented paths depend on the styler *not requiring*
the mask. `workflow_style` degrades to unmasked styling when masking or its
integrity pre-check fails, so a mask-only styler turns a graceful fallback into
an out-of-distribution failure exactly when things are already going wrong; and
`ai-style run` ships no masker at all, so a mask-only adapter would only work
behind the blog's wrapper. The dropout is the `CONTEXT_DROPOUT = 0.1` idiom
("a fraction kept heading-less so the styler doesn't require one"), not a hedge.

**Rejected: "masking is easier at synthesis than at inference because you own
both texts."** Wrong, and recorded because it is easy to re-derive. You own the
mask→unmask mapping equally in both places. The real difference is train/serve
distribution: `〈MASKED_*〉` appears on the slop side of 21 of 8,264 pairs and on
the target side of **zero**, so the styler has never been asked to emit a
sentinel `STYLE_SYSTEM` has always promised it would preserve. Inference masking
is also already built (`workflow_style.rewrite_masked`); the unbuilt part is the
corpus support that makes it in-distribution.

**Rejected: storing masked pairs.** The first draft of this entry proposed it.
Masking is deterministic and reversible, so an assembly-time transform gives the
same training diet while keeping the corpus greppable, needing no migration, and
leaving `p` a sweepable knob rather than something a generation run buys once.
The two consistency contracts — styler train/serve, and classifier train/serve —
are **independent**: each needs internal agreement and neither constrains the
other, which is what decouples the classifier question entirely.

**The contract has teeth, and one violation is already live.**
`workflow_style.rewrite_masked` passes `scorer` into `infer.rewrite_text` and
unmasks only afterwards, so `--mask --best-of N` scores masked candidates with a
detector trained on unmasked text. Also load-bearing: a pair must be masked
*jointly*, one numbering across both sides, or it teaches placeholder
renumbering — a defect no existing gate can see.

**Rejected: treating the voice classifier as a blocker.** It is the one consumer
consistency cannot fully fix — a logistic head on a *frozen* `StyleDistance`
encoder, which retraining cannot teach a new token — but `train-clf` is free,
keyless and CV-reporting, so it is a prerequisite *measurement* and nothing more.
The VS Code extension is not a concern at all: it already defers to
`dan-style serve`, so masking lives server-side.

**Re-evaluate if**: the classifier measurement comes back materially worse under
masking, or the fine-tuned styler turns out to handle sentinels worse than the
general models `placeholder-design-2026-08.md` measured — its own warning, and
the one number that changes after step 2 trains on them.

## D34: a post's role is keyed on an alias-resolved identity, and synth refuses the eval stratum unasked

**Decision** (2026-09-03; amended 2026-09-04, F3). `splits.json` carries an
optional `aliases` map, `{path the post has now: path its pairs are already keyed
under}` — new → old, the opposite of what the name suggests — and every path
passes `splits.canonical_source` before anything reads a role. Both sides of the
pinned test resolve: `role_of` compares a canonical query against
`{canonical(p) for p in eval_posts}`, and `extend_eval` stores the canonical
spelling, so a post added under its new path is pinned by identity. `synth`
refuses a target resolving to `eval` — reading the corpus's own `splits.json`
when `--splits` is absent, `--role eval` the only way past — while `log save`
notices without refusing, a capture being legitimate data. A pinned post on disk
under no known spelling is an error, not a warning: `check_splits(blog_root=…)`
marks it `ERROR_PREFIX`, `check-splits` exits non-zero, and
`require_pinned_posts` raises it from `train`/`train-clf --blog-root`, since
nothing forces `check-splits` first. Numbers: `findings.md` "The alias seam".

**Why the direction is inverted.** Roles come from `meta.source` as recorded at
capture time, so the corpus is keyed on the OLD path; aliasing old → new re-hashes
every existing pair into whatever the new path draws, silently for the roles
nothing pins. New → old moves nothing:
`test_aliases_need_no_migration_of_existing_pairs` asserts it over 50 posts.

**Rejected**: a frontmatter `id:` (mint it across ~1,700 posts, and `role_of`, a
per-pair string function, becomes a disk read); `make-splits --force` (un-freezes
eval to repair a symptom); bumping `schema_version` (`aliases` is additive, and a
bump makes every older checkout refuse the file); requiring `--blog-root`.

**Re-evaluate if**: the blog gains real post ids (the map becomes the migration
record and `role_of` keys on it), or `aliases` outgrows hand-editing.

## D35: the styler backend is a library (`stylebot.backends`), and the manifest is what configures it

**Decision (2026-09-03, W1)**: `BackendOpts`, `build_styler`, the token-scaling
rule and the serving-diet rule live in `stylebot/backends.py`, a plain library
raising `ValueError` / `RuntimeError` / `ImportError`. `bin/ai_style.py` keeps
only the translation to exit codes (usage → 2, unready world → 1), and
livingthing's `dan-style run/style/serve` import the same module instead of
owning a second construction. Both front-ends take a `--manifest`, and
`opts_from_manifest` fills the checkpoint, base model, renderer and chunk diet
from it, with explicit flags winning.

**The defect it repairs.** `BackendOpts` + `build_styler` sat inside
`bin/ai_style.py`, unimportable as a library and raising `click` exceptions, so
the blog wrote its own (`style_file.build_styler`). The copy passed no
`max_tokens`, so `infer.tinker_backend` fell to `DEFAULT_MAX_TOKENS = 2000`
while the promoted run (`20260817-170037`) serves a 4500-char cross-heading
diet, for which `ai-style run` computes 3375 via `max_tokens_for`. Every blog
rewrite of a full-length chunk was being cut off mid-sentence, and the blog's
summary line omitted `n_truncated`, so nothing said so. A copy is not merely
duplication: it is duplication that loses the argument the original was making.

**Why a new module rather than growing `infer`.** `infer` is the rewriting
engine — chunking, guards, the `Styler` protocol — and is imported lazily
everywhere precisely because it is the expensive one. The backend surface is
import-light (click, `config`, `cli_options`) and is needed at *flag-declaration*
time, before any command body runs; putting it in `infer` would drag the engine
into `--help`. It also keeps `infer.py` clear for the segmentation work.

**Why `inline_headings` became tri-state.** A manifest can only supply a default
that an explicit flag beats, and with `bool` there is no way to tell
`--no-inline-headings` from silence. `None` means "nobody said"; `headings()` is
the resolved boolean every call site uses. The same reasoning would apply to any
future flag the manifest can fill.

**Why the diet rule moved but its fallback did not.** `serving_diet` reads
stylebot's own manifest schema (`data.chunk_scale`), so it was mechanism in the
wrong repo; the largest-well-supported-budget rule and the cross-heading-majority
rule came across verbatim. What stayed blog-side is the one genuinely
policy-shaped input — the fallback for runs 1-3, which predate `chunk_scale` and
were trained at `training_targets.MERGE_MAX_CHARS` — passed in as
`default_chunk_chars`. `chunk_diet` is now three lines around the library call.

**Rejected: re-exporting the surface from `bin/ai_style.py` for compatibility.**
There is one other consumer and it is in this tree; an alias would have left the
old import path working and therefore in use.

**Rejected: letting the blog keep its own `build_styler` and merely passing
`max_tokens`.** That fixes the measured bug and leaves the mechanism that
produced it. The same divergence had already cost five flags on `dan-style run`
and four on `serve`; `tests/test_dan_style_cli.py` now asserts
`flags(dan-style X) ⊇ flags(ai-style X) − EXCLUDED[X]` with a written reason per
exclusion, which is the part that keeps this from happening again.

**Rejected: taking `--detector-model` from the shared bundle blog-side.** The
bundle types it `click.Path(exists=True)` and click converts defaults through
the type, so pinning `_models/voice-clf` as a default would fail every
invocation in a checkout without the artifact — including `--best-of 1`, which
needs no detector. It stays excluded and pinned in code.

**Re-evaluate if**: a third front-end appears (the exclusion lists are per-repo
policy and would want a shared vocabulary), or manifests grow a serving section
of their own — at which point `opts_from_manifest` should read that rather than
inferring the diet from training-time histograms.

---

## D36: a pair record carries its own provenance, and uniqueness is a property of the file

**Decision (2026-09-03, W8)**: every record written from now on stamps
`meta.schema_version`, `meta.system_sha` and a five-key `meta.provenance`
(`blog_sha`, `blog_dirty`, `stylebot_sha`, `policy_digest`, `source_automation`)
— both producers, `synth._build_record` and `capture.build_pair_record`.
`record_id` uniqueness moves into `validate_pairs_file`, where the whole file is
visible, and `pairs.dedupe_pairs_file` (`ai-style validate-pairs --fix`) drops
repeats keeping the **first** occurrence. `synth.PairsWriter` makes the corpus
append safe under two concurrent writers.

**The three sub-decisions worth naming**:

- **The chunk policy rides in on the `Target`, not on a `synthesize_pairs`
  parameter.** `iter_targets` is the one construction site, and it already holds
  the composed `ChunkPolicy`, so `targets._post_targets` digests whatever it was
  handed and stamps it on each `Target`. Nothing between the policy and the
  record has to be told about it — and `provenance.policy_digest` takes the
  object rather than a field list, so the stamp keeps meaning what it says when
  the blog composes that policy in one place (W2). A `div_is_prose` predicate
  hashes as `module.qualname`; `repr()` carries a memory address and could never
  have been stable.
- **First writer wins.** The 21 duplicates were two *draws* for one cell, not
  two copies of one row, so deduping picks a winner. The first is the copy every
  committed `scores.jsonl` row and every resumable rewrite already joined to —
  keeping any other silently re-points those artifacts at prose they were not
  measured on.
- **`source_automation` is read at cut time.** The corpus is frozen and the post
  is not, so nothing else can recover what the selector was looking at. 75 pairs
  now sit on `automation: 2` posts with no record of it.

**Alternatives considered**:

- **A lock around the whole run** instead of around each append. Serialises two
  synth runs completely, which is exactly the thing anyone running two budgets
  is trying to avoid.
- **Literally "re-read `seen` before each write"** (the QA list's own wording).
  That is O(corpus) per record: 38 MB × 4,000 records. `PairsWriter` re-reads
  only the tail appended since it last looked, from a byte offset, under the
  same lock — the same guarantee at O(what actually changed).
- **Keying `synth_key` on the provenance** so a tree change mints a fresh cell.
  Rejected: resume would restart on every commit, and the corpus would grow a
  copy of each pair per SHA.
- **A closed whitelist of `meta` keys in the validator.** Rejected: the schema is
  open by design (`extra_meta`, `meta.gen`), so a whitelist would turn every
  experiment's covariate into a validation failure. The type/domain table checks
  keys it knows, when present, and ignores the rest.
- **Bumping `schema_version` for the new keys.** The absent-`schema_version`
  records ARE version 0; a consumer that must branch has the tell it needs, and
  adding optional keys is not a change anyone branches on.

**Deciding constraint**: `validate_pair_record` had been per-record since Phase
1, so the one defect it structurally could not see is the one that reached the
corpus — 21 duplicate `record_id`s, written 2026-08-12 14:03–14:56 by two synth
runs that each loaded `existing_synth_keys` before the other appended. Eval and
rewrite resume score the first and skip the second; training takes both. The
whole-file check and the locked writer are the two halves of that: one finds it,
one stops it recurring. Measured on the live corpus 2026-09-03: 21 duplicates
before, 0 after, over 8,265 → 8,244 records; every other type and domain check
in the new table passed on all 8,265 without a single relaxation, which is the
evidence that the table describes the corpus rather than an aspiration.

**Re-evaluate if**: the corpus is sharded by producer (W6) — `check_duplicates`
then has to run across the shard set rather than per file, and
`dedupe_pairs_file`'s "first" needs a defined order between shards. Also if a
non-git corpus deployment appears, at which point `blog_sha`/`blog_dirty` want a
more general name than the one deployment gave them.

---

## D37: the corpus-stability guard belongs to whoever owns the corpus

**Decision (2026-09-03, W5)**: `test_corpus_stability.py` moves to livingthing's
suite. stylebot keeps no replacement, and `python-dateutil` — which sat in its
dev group for no reason but that test's `sys.path` hop into the blog — goes with
it. **stylebot depends on no checkout it does not own**: not in `src/`, not in
`tests/`, not in its dependency groups.

**What the guard needs, and who has it.** Three things, and the blog has all
three: the corpus (which by rule never enters the public tooling repo), the
chunk policy (which is blog policy, in `livingthing.training_targets`), and a
git pin into blog history to freeze the prose. stylebot had none of them, so the
test probed `../livingthing` and `~/Source/livingthing`, patched `sys.path` to
import a package it does not depend on, and carried a transitive dependency of
*that* package to make the import work. In the blog it is an ordinary test: two
imports and the file next door.

**Alternatives considered**:

- **Keep a stylebot-side guard reading policy as data** from the newest run
  manifest's `data.chunk_scale`, skipping when no corpus is configured (the QA
  list's own fallback). Rejected: on every machine but Dan's it skips, and the
  file already argues at length that a green skip reads as "measured, fine". A
  guard that cannot run where the package is developed is decoration. What
  stylebot keeps instead is what it can honestly own — `test_markdown.py`,
  `test_protection_parity.py` and the conformance fixture pin the segmenter's
  *behaviour*; the corpus floor pins what that behaviour cost, which is a
  question about one corpus.
- **Vendor a corpus sample into stylebot.** Rejected outright: the corpus never
  enters this repo.
- **Leave it and add `python-dateutil` properly.** That is the arrow pointing
  the wrong way, written down as a dependency.

**Deciding constraint**: the failure this already caused. Dropping the unused
`python-dateutil` runtime dependency (correctly, in `aa570f4`) also dropped it
from the dev environment, `livingthing.training_targets` imports it
transitively, and the harness silently stopped measuring for hours. The fix at
the time was to make the import failure `pytest.fail` rather than skip — a
better error for a defect that should not have been reachable.

**And the floors are real again.** Moving it exposed that both arms had been
mis-measured, which is the second half of this entry:

- **Each arm gets its own pin.** The section-bounded cells were cut at
  2026-07-22, the cross-heading (v2) cells three weeks later; one pin for both
  charged v2 cells for three weeks of Dan's prose drift.
- **Each arm gets its own policy.** `div_is_prose` (callouts-as-prose) postdates
  the v1 cells and predates the v2 ones, so v1 is re-derived without it and v2
  with it.
- **Each *budget* gets its own cut.** `dan-style synth` swept three chunk
  budgets over the same prose (2381 cells at 1500 chars, 1045 at 3000, 768 at
  4500) and each cell records its own in `meta.merge_max_chars` — but the
  harness asked every cell for `MERGE_MAX_CHARS = 1500`, so two thirds of the
  corpus was charged for a cut nobody made. This is the same principle that
  already selected on `chunk_mode`, applied one level down.

Measured 2026-09-03: section-bounded 3023/3028 (99.83%, unchanged, the pin and
policy were already right for it); cross-heading **417 → 4181 of 4194**
(99.69%). The old value was annotated "stale on purpose", read 11% of its
stratum, and could catch only a catastrophic break. Both floors are now
near-equalities, which is what makes them able to fail for a reason.

**Re-evaluate if**: a third chunk regime lands — add an `Arm`, pin it at its own
generation revision, measure. Or if the corpus is sharded by producer (W6), at
which point `CORPUS` becomes a glob. If stylebot ever needs its own floor, the
honest version is a small fixture corpus committed *to stylebot* and cut from
its own test fixtures — a different instrument, measuring the segmenter rather
than the blog.

---

## D38: no `scripts/` carve-out — migrations are a hidden `ai-style data` group

**Decision (2026-09-03, W5)**: `scripts/` is deleted. Its two one-off migrations
become typed functions in `stylebot.migrations` behind `ai-style data
migrate-captures` and `ai-style data prune-invented-anchors`, a click group
declared `hidden=True`. "One entry point, subcommands" now has no exceptions.

**What the carve-out was actually protecting**, and what it cost. The stated
reason was the `--help` tree: a migration is run once, by one person, per schema
change, and does not belong beside the pipeline. That is a real concern and
`hidden=True` answers it exactly — absent from `ai-style --help`, fully
documented under `ai-style data --help`. What the directory bought on top of
that was all cost:

- `scripts/` is not in the wheel. `capture.iter_archives` refuses a legacy
  archive and prints the command to fix it; for anyone who installed stylebot
  rather than cloning it, that command named a file they did not have.
- The blog's RUNBOOK invoked both scripts as `uv run python
  ../stylebot/scripts/…` — a hardcoded blog→stylebot *path*, in the same repo
  whose OVERVIEW says stylebot never hardcodes a blog location. The arrow was
  bent in both directions at once (D37 is the other).
- "Typed once per schema change, and each carries a test" was written in
  `CLAUDE.md` while only one of the two had a test. A directory outside the
  package is a directory outside the checks.

**Alternatives considered**:

- **Move them to the blog's `scripts/`**, on the grounds that only the blog
  corpus is ever migrated (the QA list's other option). Rejected: it is true
  today and it is not what the code says. `migrate-captures` converts
  *stylebot's own* archive format and is named by *stylebot's* error message; a
  second installation would hit that error and be told to go and find Dan's
  blog. Migration of a data format belongs with the code that defines it.
- **Plain visible subcommands.** Rejected for the reason the carve-out existed:
  `ai-style --help` is a phase map, and two spent migrations in it read as
  pipeline steps.
- **A separate `stylebot-migrate` console script.** That is the `ai-style-log`
  mistake (folded back in 2026-07-20), one release later.

**Deciding constraint**: an error message that names a command must name a
command that exists. Everything else here follows from putting the migrations
where the error can reach them.

**Re-evaluate if**: nothing. A migration is deleted once the only corpus that
exists has been through it — function, command and test together — and if both
of these go, `stylebot/migrations.py` goes with them and the next one starts a
fresh file.

---

## D39: a score carries the instrument that produced it, and two instruments never share a mean

**Decision (2026-09-03, W10)**: every detector result carries
`DETECTOR_IDENTITY_KEYS` — `name`, `embed_model`, `git_sha`, `trained_at`,
`n_pairs` — read from the artifact's `meta.json` and stamped onto every
`scores.jsonl` row. `eval.score_pairs_file` refuses to append to a file whose
existing rows name a different head (`--allow-mixed-detector` to override), and
`eval.compare_scores` refuses a verdict spanning two heads
(`--allow-cross-detector`, which then flags the result `cross_detector: true`).

**The defect.** One classifier is the best-of-N reranker, the do-no-harm guard
*and* the verdict metric, and it is refit every time the corpus grows. Run 4a's
own promotion note admits the detector was refit the same day as the verdict it
produced, so its headline — "closes 90.7% of the slop→Dan gap, against runs 1–2's
~55%" — is not a like-for-like comparison and says so in a parenthesis rather
than in the data. The rows recorded `name` and `embed_model` only, both of which
are constant across every refit this project will ever do; they identified the
*recipe*, not the artifact.

**Why the refusal is at the resume, not only at the compare.** `scores.jsonl` is
append-only and resumes by pair id, which says nothing about what scored it. A
refit head plus a resumed run silently produces one file whose means blend two
instruments — no error, no marker, and `summarize_scores` reports the blend as
one number. That is a worse failure than a cross-file comparison, because
nothing downstream can even see it happened.

**Rejected: a header line, or a sidecar `scores.meta.json`.** The file is
appended to across sessions by different processes; a header records only the
first writer, which is precisely the writer a refit run is not. Per-row costs
~60 bytes against a file already dominated by Vale's `raw` alert objects, and it
makes "which rows came from which head" answerable rather than assumed.

**Rejected: probing the detector for its identity by scoring a canary string.**
An embed per guard call, on a path that is meant to be free.

**Deciding constraint**: an unconfigured (`null`) detector must never trip the
guard. A keyless re-run adds no detector scores at all, so it cannot blend
anything, and making it raise would break the documented keyless path.

**Re-evaluate if**: the detector gains a calibration layer that is versioned
separately from the head (then the identity is a pair), or artifacts start
carrying a semantic version, at which case `git_sha` becomes the tiebreak rather
than the key.

---

## D40: the eval interval clusters by post, and the headline is the real facet

**Decision (2026-09-03, W10)**: `eval.paired_delta(clusters=…)` computes the
paired difference with a **cluster bootstrap over POSTs**, and
`eval.compare_scores` clusters by `meta.source` by default. `summarize_scores`
facets `by_provenance` and names `headline_facet: "real"`; both CLIs print the
real facet first and the synthetic one separately.

**The measurement that forces it** (re-measured 2026-09-03, live corpus): the
frozen eval stratum is **6 posts / 103 pairs, 41 real and 62 synthetic**, and
**77 of the 103 come from `notebook/automatic_coding.qmd`** — 75%. (Through the
detector's near-identity gate the same stratum reads 89 / 41 / 48, which is what
`check-splits` prints; the two numbers are the corpus and the dataset, and both
are right.) Run 4's verdict was "127 chunks, paired", with a normal-approximation
interval that treats those 127 as independent observations. Chunks inside a post
share topic, register and whatever the generator did to that post, so the
effective n is nearer the 6 posts. On the synthetic-vs-real question the same
stratum is 60% our own paraphrase generators, so an unfaceted headline is
substantially a measurement of this month's slop roster.

**Why a bootstrap rather than a mixed model or a cluster-robust SE.** Stdlib
only: `splits`, `provenance` and now the verdict path must import with no extras
(`_plans/eval-harness.md`, the split contract). A percentile bootstrap over
resampled posts is a dozen lines and needs nothing; a random-effects fit would
drag statsmodels into a module the trainer imports.

**What this does NOT fix.** Widening the interval is honesty, not power: 6
clusters is 6 clusters, and the run-4 arms were already indistinguishable at the
narrow interval. The fix for power is D41 — more posts in the stratum.

**Rejected: clustering by capture rather than by post.** A capture is one
editing session, so captures nest inside posts; clustering at the finer level
would leave the between-post correlation uncounted, which is the correlation
that exists.

**Re-evaluate if**: the eval stratum passes ~30 posts (the percentile bootstrap
gets conservative at small cluster counts and a BCa correction starts to be
worth its complexity), or a verdict ever needs more than one signal jointly.

---

## D41: the automation cap applies to synthetic pairs only — a capture is *supposed* to sit on an AI-touched post

**Decision (2026-09-03, W10)**: `classify_train.assemble_dataset` grows a
caller-supplied `pair_ok` hook, and the blog's `dan-style train-clf` passes one
that drops **synthetic** pairs whose source post's frontmatter `automation`
exceeds 0 (`--max-synthetic-automation`, default 0). Real captures are never
dropped by it.

**Why the QA list's version of this was wrong.** The list says "restrict
detector targets to `automation: 0`", on the reasonable ground that the detector
pool holds pairs from AI-touched posts. Measured on the live corpus
(2026-09-03), the detector pool is 2,596 pairs from `automation: 0` posts, 517
from `1` and 139 from `2` — but split by provenance those break down as **488 of
the 656 non-zero pairs synthetic, and 168 of the 179 REAL pairs (94%) sitting on
`automation: 1/2` posts**. Applying the cap to everything would leave the honest
`real` facet — the headline metric of the whole classifier — with **11 pairs**.

The asymmetry is not an accident, it is the definition. A synthetic pair's "Dan"
side is the post's existing prose taken as ground truth, so on an AI-touched post
the detector is taught that AI-assisted prose *is* the author: exactly the
discrimination it exists to make, inverted. A *capture* is Dan editing an
AI-touched draft — the post carries `automation: 1/2` **because** the capture
happened — and its Dan side is his own keystrokes. The frontmatter field means
"AI touched this post", which is evidence about the synthetic label and evidence
*for* the real one.

**Deciding constraint**: an unreadable or missing post keeps its pairs. A file
that cannot be parsed is not evidence of contamination, and `meta.source` records
paths that renames have moved (hence the alias-resolved lookup, D34).

**Rejected: reading `meta.provenance.source_automation` instead of the file.**
W8 stamps it, but only on records written after 2026-09-03; the corpus is 8,244
records and almost none carry it. Reading the frontmatter covers the whole corpus
and needs no migration. Once provenance is universal, the stamp is the better
source — it records what the frontmatter said *at generation time*, which is the
question actually being asked.

**Rejected: hardcoding the gate in `classify_train`.** `automation` is a blog
convention; stylebot's selection rule is "selection is a user-supplied policy"
(CLAUDE.md). The hook is the mechanism, the level is the policy.

**Re-evaluate if**: `meta.provenance.source_automation` reaches most of the
corpus (switch to the stamp), or the real stratum grows enough that capping it
too costs less than the contamination it removes.

---

## D42: the eval stratum is versioned and add-only, and a manifest pins the partition it held out

**Decision (2026-09-03, W10)**: `splits.json` carries `eval_version` (v1 = the
original cut) and grows only through `splits.extend_eval`, which forms
`eval_v(n+1) = eval_v(n) ∪ posts`, bumps the version and appends an
`eval_history` entry. Removing a pinned post is not expressible. Run manifests
gain `data.splits_sha256`, `data.eval_version`, `data.record_ids_sha256`,
`data.selector_constants`, `dependencies` and `base_model_revision`; the
classifier artifact gains `splits_sha256` and `split.eval_version`; verdict
outputs are written under `runs/<run_id>/` (`eval.verdict_dir`,
`dan-style eval --run-id`).

**Why add-only is the whole safety property.** Growing the stratum moves posts
*out of* the training pools and *into* the frozen one: prose leaves training, it
never enters it. Shrinking it would retroactively license training on prose an
existing verdict was computed over — the leak D34 was written about, arrived at
deliberately. So the operation exists in one direction and the other requires
cutting a new file and admitting it.

**What v2 would actually cost, measured 2026-09-03.** 52 posts carry at least
one real pair; 6 are already eval. A maximal `eval_v2 = eval_v1 ∪ (all 46)`
gives a 52-post stratum of 1,025 pairs (398 real) — and takes **all 179 of the
detector pool's real pairs and all 178 of the styler's**. The detector's honest
`real` facet would go to zero and the styler would train on synthetic pairs
alone. Real pairs are scarce and all three roles want them; a maximal extension
is not "more eval", it is "no more anything else".

**The recommended v2 is therefore sized, and takes from one pool.** Adding the
six styler-role real-capture posts with ≥9 real pairs —
`notebook/local_llm_mac.qmd` (31 real), `post/should_we_allow_ai_articles.qmd`
(25), `notebook/open_world_inference.qmd` (19), `notebook/pdf_ingestion.qmd`
(12), `notebook/ai_search.qmd` (11), `notebook/ai_reasoning.qmd` (10) — gives a
**12-post, 245-pair, 149-real** stratum: 3.6× the real pairs, the dominant post's
share down from 75% to 31%, the detector's real facet untouched at 179, and the
styler left with 70 real pairs.

**Applied 2026-09-03** with Dan's approval, through the CLI this decision was
missing: `ai-style extend-eval` / `dan-style extend-eval` (dry-run by default,
`--apply` to write, and it prints what it would re-role first). A separate
subcommand rather than a flag on `make-splits`, because the two have opposite
safety rules — `make-splits --force` re-cuts and thereby un-freezes the
partition, while this may only ever grow it, and one command carrying both would
sit "never unfreeze eval" beside the single flag that does. Re-measured against
the post-prune corpus the stratum went 6 → 12 posts, 41 → 149 real pairs, top
post 37% → 22%, `splits_sha256` `c8c319578707bb5b` → `05b495adac68cdc4`; the
detector was refit the same day (`git_sha` `da335af66d` → `c0e2ac85be`) and its
pool did not move by a pair, since all six added posts were styler-role. Numbers,
including what the clustered CI did and the 0.026 the refit alone moves the
metric, in `findings.md` "eval_v2 — applied 2026-09-03".

**Why the manifest needed four more pins.** It named the corpus file and both git
trees, so two runs of "the same" recipe could agree on every recorded field and
train on different data: `pairs_sha256` hashes the FILE, while the role filter,
the selector and the multiplicity hook all run after it (`record_ids_sha256`
closes that); `filters.selector` records `styler_pair_ok` by name while
`RETIRED_ARMS` is a set whose membership changes the diet (`selector_constants`);
and `splits_path` recorded a path to a file that has since gained an `aliases`
map (`splits_sha256`).

**Rejected: `eval_version` as a new `schema_version`.** The file's *shape* is
unchanged; what versions is the stratum. Bumping the schema would force every
reader to update for a change none of them care about, and `load_splits` pins
the schema exactly.

**Rejected: resolving `base_model_revision` at manifest time.** It would mean a
network call to a model hub inside `--dry-run`, which is the one path that
promises to be free and offline. The key is present and `None` until the runner
can fill it from what the training service reports.

**Re-evaluate if**: a v2 cut is applied (then this entry records the reasoning
and `eval_history` records the event), or the corpus gains enough real captures
that all three roles can be fed — at which point the sizing argument above
dissolves and the maximal extension is simply right.

## D43: a passage is a stretch of prose, not a string — and weight is replication

**Decision (2026-09-03, W9)**: `train.assemble_training_corpus`'s multiplicity
hook groups by **source post + character-span containment**
(`train.passage_groups`), not by exact target body, and is renamed
`per_target` → `per_passage` to say so. A second hook, `pair_weight`, stamps
`meta.weight` and is honoured as **integer replication of the train rows** in
`_write_conversations`. Manifests record the resulting shape under
`data.chunk_scale.passages` / `.pair_weights` / `.n_train_examples`, and both
`ai-style train --dry-run` and `dan-style train --dry-run` print it
(`train.format_passage_scale`, one renderer so the two shells cannot drift).
`ChunkPolicy.sort_name` / `sort_key()` and `iter_targets(sort_key=)` are deleted.

**The measurement that forced it.** Re-measured on the 8,244-record corpus:
5,325 distinct targets are strict substrings of another target in the *same
post*, carrying 6,157 pairs (74.7%); 883 targets carry more than one pair; 92.4%
of the corpus sits in some containment-or-repeat group. Reproducing run 4a's
selection against today's corpus gives 3,981 styler-role train pairs — and the
decomposition is what settles the design:

| | count | mean |
| --- | --- | --- |
| train pairs | 3,981 | |
| distinct target bodies | 3,533 | 1.13 pairs each |
| **passages** (containment groups) | **982** | **4.05 pairs each**, max 15 |

So exact-body grouping had almost nothing to group: 3.6 of every 4.05 duplicates
were invisible to it, and that is why every manifest ever written records
`per_target: null` — switching it on would not have helped. 653 of the 678
multi-pair passages span two or more of the four cut diets
(`merge_max_chars` null / 1500 / 3000 / 4500), which confirms W5's reading: the
`dan-style synth` budget sweep is the mechanism, not generator replicates.

**The consequence is a diet nobody chose.** A manifest saying `num_epochs: 1`
describes ~4 gradient steps per stretch of prose, unevenly — 304 passages get
one and one passage gets fifteen. And it silently re-weights the axis that
matters most: real captures were cut once, synthetic pairs went through the
sweep, so real pairs are **3.09% of train pairs but 12.2% of train passages**.
The scarce human side is diluted 4× by an artifact of how the corpus was
generated.

**Why the by-post val split cannot see it.** The split is by POST, which is the
right guard for leakage and useless here: val inflates by the same mechanism
(392 pairs on 95 passages, mean 4.13), so held-out loss is computed over the
same skew and reads normal. That is the whole reason this needed measuring
rather than watching.

**Rejected: delete the promise.** The honest option, and the one W9 offered.
Rejected because the measurement turned a speculative knob into a live defect:
`meta.weight` and `pair_weight` were promised in OVERVIEW and this module's
docstring since the first draft, and `_write_conversations` stripped `meta` one
line before the trainer could see it — the promise was not merely unimplemented,
it was silently unimplementable. Deleting it would leave no way to state
"real captures count triple" short of physically duplicating corpus lines, and
no way to see the 4.05 at all.

**Rejected: dedupe by budget at assembly** (keep only the widest cut per
passage). It is one line and it fixes the count, but it throws away the
short-chunk diet the adapter is also served at, and picking which budget
survives is exactly the policy call that belongs to the blog rather than to
stylebot. The hook plus the histogram gives the caller the same power and
records what they chose.

**Rejected: a per-example loss scale.** The cookbook's
`SupervisedDatasetFromHFDataset` maps rows to data through
`conversation_to_datum` and has no field for a weight; a fractional weight would
mean forking the builder. Integer replication expresses the same thing through
the interface that exists, and separate rows get scattered by the builder's
epoch shuffle rather than presenting one passage N times in a row. The cost is
that weight cannot go below 1 — down-weighting is spelled as up-weighting
everything else, or as `per_passage` dropping pairs.

**Val is never replicated.** A weighted held-out loss is an average over a
distribution nobody chose, and it would stop being comparable across two runs
whose weights differ. `_write_conversations(..., replicate=False)`.

**Why `per_target` was renamed rather than kept.** The grouping key changed
meaning, and a caller written against the old one would keep working while
grouping 3.6× too finely. A `TypeError` is the correct outcome; there is no
alias. Nothing in either repo passed it (`filters.per_target: null` in all five
manifests), so the break costs nothing today and is loud if it ever would.

**`sort_name` went with it, and cost nothing.** `--sort` retired 0.3.1 because
arms are assigned by content hash, so target order cannot change the corpus;
the library-only field survived for a caller ordering in code, and no such
caller ever appeared in either repo. Deleting it shifts every
`provenance.policy_digest` — which is free precisely today, because W8 landed
this morning and **0 records carry a `policy_digest` yet**. A week from now it
would have been a corpus-wide re-stamp.

**Not applied: an actual weighting policy.** Nothing is re-weighted or re-cut.
The mechanism, the accounting and the histogram are in; choosing the diet
(a `per_passage` cap, a real-pair `pair_weight`, or a narrower budget sweep at
synth time) costs a paid retrain and is Dan's call. The obvious candidate, from
the numbers above, is capping `per_passage` so the effective corpus is ~1,660
pairs at cap 2 or ~2,212 at cap 3, which would lift the real share from 3.1%
toward 12%.

**Re-evaluate if**: a run is trained with a non-uniform weight (then this entry
records what it bought), or the corpus stops being cut at multiple budgets over
the same prose — at which point passages and target bodies converge and the
containment grouping costs string searches for nothing.

---

## D44: the editor asks the sidecar where the prose is

**Decision** (2026-09-03, W4). `stylebot.serve` grows a `segment` op —
`{"op": "segment", "text": <document>, "min_chars": n}` →
`{"segments": [{start, end, kind, context}]}` — backed by
`stylebot.serve.prose_paragraphs`, which composes `stylebot.markdown` with the
`div_is_prose` policy `serve_loop` was started with. The VS Code extension's
TypeScript segmenter (`vscode-voice-marker/src/segment.ts`), its test, and the
fixture pair that pinned it (`tests/fixtures/segmentation.qmd` +
`expected_segments.json`, with `tests/test_marker_segmentation.py`) are deleted.
The extension also stops computing `headingAbove` and takes each segment's
`context` from the sidecar.

`prose_paragraphs` lives in `serve` rather than in `markdown` because it needs
the two things that module refuses on purpose: frontmatter knowledge, and a
paragraph granularity that is the *editor's* unit rather than the chunk
budget's. It is the editor seam's composition, and `serve` is the editor seam.

**What the port had actually drifted into.** Measured before deleting it, over
all 1,757 `.qmd` files in the blog, comparing the port's paragraph offsets with
`prose_paragraphs`': **1,728 files identical, 29 differing.** Every one of the 29
is a correction, and they fall in two groups:

- **28 files: the heading rule.** The port split paragraphs on blank lines and
  then dropped a paragraph whose *first* line was a heading. So `## H` glued to
  the prose under it (no blank line) took the prose down with it — 24 files,
  including every `digest/`, where `### [Title](link)` sits directly on its
  paragraph and the marker therefore scored **nothing at all** in a whole
  document — and prose glued to a heading *below* it was scored with the heading
  text inside the string. `line_kinds` ends a run at a heading line, which both
  keeps the prose and keeps the heading out of it.
- **1 file: the unclosed fence.** `notebook/vs_code.qmd` ends on a bare ```` ``` ````
  with no closer. The port's `CODE_FENCE_RE` needed a closer, found none,
  matched nothing, and scored the fence line as the tail of the paragraph above.
  W2 gave `markdown.CODE_FENCE_RE` a run-to-EOF alternative, which is
  CommonMark's own reading.

Two further differences are real but latent in this corpus. The port's closer
was a bare three backticks, so a ```` ```` ````-opened block never closed and
swallowed the rest of the document; W2's `` `{3,} `` closer fixes it, and the
blog happens to contain no such block today. And the extension's *third*
heading spelling — `headingAbove`'s `/^[ \t]{0,3}#{1,6}[ \t]/`, which accepted a
tab indent and rejected a bare `##` — disagrees with `ATX_HEADING_RE` on
**0 of 36,034** contexts in the live corpus. Both are deleted anyway: a
difference that is latent is a difference waiting for one post.

**And the one the port could not have got right at all:** `div_is_prose`. It had
no hook, so a `.callout-note` paragraph was never scored and `rewriteTarget`
returned null inside one — while the styler had been trained on callout prose
since 2026-08-10. Under the blog's policy the op returns **385 more paragraphs,
96,796 more characters, across 148 files**. This is not a bug the port had; it is
a bug the port's *location* had. Which div classes bear prose is blog policy,
and the extension has no way to learn it.

**Latency: one NDJSON hop per document change, on the already-debounced path.**
Measured through the real `Sidecar` client against a stub-detector `serve_loop`,
50 timed round trips each, warm:

| document | chars | segments | median | p95 |
|---|---|---|---|---|
| `notebook/music_loopers.qmd` (corpus median) | 5,428 | 9 | **0.30 ms** | 0.38 ms |
| `digest/2026-04-24.qmd` (p90) | 17,662 | 22 | 0.65 ms | 0.71 ms |
| `notebook/automatic_coding.qmd` | 30,496 | 67 | 1.07 ms | 1.10 ms |
| `notebook/local_llm_mac.qmd` (largest in the repo) | 116,774 | 212 | **4.20 ms** | 4.36 ms |

Nearly all of it is the segmentation itself (0.28 / 0.50 / 0.90 / 3.71 ms
in-process), so the hop — JSON both ways plus the pipe — costs tens of
microseconds. Against the 34 ms the *score* op already spends on the same
document and the 500 ms debounce, it is not measurable by a typist. The
document travels up the pipe once more per change; at 117 kB that is the largest
payload in the protocol and still under a millisecond to encode.

**Alternatives.**

*Keep the port, fix its three defects.* Rejected: the same argument D28/D30 lost
on 2026-09-03. Fixing it means porting `div_is_prose` (a *callable* policy) into
TypeScript, then keeping two implementations and a hand-copied fixture in step
by hand. The fixture had already failed at that once (blog `2815a291fc`), and
the port's boundaries were the ones that won, because `rewrite_candidates`
re-segments only to veto.

*Fold segmentation into the `score` op* (send the document, get spans and scores
in one hop). Cheaper by one round trip — a round trip that costs 0.02 ms — and it
welds two questions together, so a client that wants ranges without scores (the
rewrite command does) has to pay the detector.

*Return the paragraph text as well as the offsets.* Rejected as redundant: the
client sent the document, so it can slice it. The response stays small
(4 fields × ~200 segments at worst) while the request carries the document.

**Offsets are UTF-16 code units, converted server-side.** Python indexes code
points and every editor this protocol will meet indexes UTF-16 — VS Code, and
the LSP spec it follows. One astral character above a paragraph and every
decoration below it lands a character short. `serve._utf16_mapper` converts,
with a fast path for the documents (nearly all of them) that contain none. The
client then does no arithmetic at all: `positionAt(start)` just works.

**The `context` is provenance, not a prompt — and that is the diet's call, not
the client's.** W2 made `rewrite_candidates` drop `context` outright under
`inline_headings`, and the promoted run (`20260817-170037`, `modes:
{cross-heading: 2591, None: 1799}`) is cross-heading, so the heading the sidecar
reports is *not* seen by the styler today. It is still sent back on `rewrite`
and recorded in the `prefer` record, because it is the true discourse context of
the chunk and a section-bounded arm may yet be promoted. What changed is who
decides: the client no longer has an opinion, and `rewrite_candidates` is the one
place that knows whether the diet wants it.

**Known wart, inherited not introduced:** an *unclosed* `:::` opener produces no
div span (`markdown._div_spans` documents this), so a stray `::: something` line
is offered as a scoreable paragraph. The deleted port did the same. Left alone
rather than special-cased here, since the fix belongs in `markdown` and would
move training targets.

**Re-evaluate if**: the marker is ever distributed as a standalone `.vsix` with
no Python sidecar (parked alternative B in `_plans/vscode-marker.md` — then the
segmenter has to be in TypeScript again, and this entry is the list of what it
must get right); or the `segment` round trip stops being free, which on these
numbers means a document an order of magnitude past the largest one Dan has.

## D45: one corpus location, resolved once — and the synth walker keeps its blog root

**Decision.** `stylebot.config` owns every spelling of a path inside the corpus
directory (`pairs_path`, beside W0's `splits_path`), and livingthing composes
its blog-root default on top of it in ONE function,
`training_targets.data_dir(flag, blog_root)`, with the precedence `--data-dir` >
`$STYLEBOT_DATA_DIR` > `<blog-root>/_training_pairs`. Every `dan-style` default
— corpus, splits, run manifests, the preference log — routes through it, and the
click defaults that need it are **callables**, not paths frozen at import.

*Alternatives.* Leave the blog's literals alone (they were "obviously" the same
directory — except `STYLEBOT_DATA_DIR` redirected `ai-style log`'s captures,
which go through `resolve_data_dir`, while synth/train/train-clf/run/serve kept
writing the live corpus, so pointing a scratch run at a scratch directory
silently half-worked). Or teach `resolve_data_dir` about a blog root, which puts
a caller's `--blog-root` convention inside generic tooling.

*Constraint.* `resolve_data_dir`'s own fallback is **cwd**-relative and must
stay that way; only the blog has a `--blog-root`, so only the blog folds it in.

## D46: the synth walker's blog exclusions stay pinned, not routed around

**Decision.** `dan-style synth` keeps calling `iter_targets(blog_root=…)` and
walking with stylebot's predicate. The QA list (W11.f) proposed handing it a
pre-selected `files=` list from `livingthing.lib.gather_qmd_files`, so the walk
would honour the blog's `LOCAL_SKIP_DIRS` (`src`, `bin`, `tests`). Refused on
two measurements.

*The gap is empty.* The two walkers agree on all **1757** `.qmd` in the live
tree (measured 2026-09-03) — `LOCAL_SKIP_DIRS` and `QUARTO_SKIP_DIRS` are
disjoint, and no `.qmd` has ever lived under `renv`/`packrat`/`rsconnect`/
`node_modules`. Exactly one file ever fell in the gap,
`tests/fixtures/segmenter_conformance.qmd`, deleted with the second segmenter
(W3); and even at its worst historical frontmatter (`{}`, and later
`{quality-net: 5.0}`) `make_selector` rejected it, so it could never have become
a target.

*The fix costs more than the gap.* `files=` and `blog_root=` are mutually
exclusive in `iter_targets`, and without `blog_root` `_source_label` falls back
to `str(path)`. Measured on `notebook/automatic_coding.qmd`: `blog_root=` gives
`notebook/automatic_coding.qmd`, `files=` gives the absolute path, and
`splits.role_of` then answers **`styler` instead of `eval`** — reopening the
leak W0 had just closed, and orphaning every pair keyed on the relative source.
`files=` also skips the selector, so the blog would re-implement the frontmatter
gates it currently passes as a callable.

*So instead:* the divergence is asserted. `tests/test_qmd_walker.py::
test_the_synth_walker_sees_exactly_what_this_repo_calls_a_post` compares the two
walks over the live tree and names any file either side has that the other does
not.

**Re-evaluate if** a `.qmd` ever needs to live under `src/`, `bin/` or `tests/`
again, or if `iter_targets` grows a label-only `blog_root` companion to `files=`
(the one-line change that would make the pre-selected form safe: use `blog_root`
for `_source_label` while `files` supplies the walk).

## D47: two error classes for the whole package, translated once at the group

**Decision.** `stylebot.errors` declares `ConfigError` (the arguments are wrong;
exit 2), `RunError` (the arguments are fine, the world is not; exit 1) and
`MissingExtra` (a `RunError` whose unready world is an uninstalled extra).
`cli_options.translate_errors()` turns them into click exceptions, and
`ErrorTranslatingGroup` — the class of `ai-style` and of `ai-style log` —
applies it to the whole subtree. `Synth*Error` (D-era: synth's own hierarchy)
and W11.b's `SessionError` became named subclasses rather than a third and
fourth scheme, keeping the codes they chose; `capture`'s "no changed chunks" is
still exit 1, as W11.b decided.

*The defect.* `bin/ai_style.py` had six `except ValueError:` blocks, every one
mapping to `click.ClickException` (exit 1), while `synth` — the only module with
a real hierarchy — mapped its config failures to exit 2. So `train`'s "no
training pairs left after filtering" exited 1 and the same class of mistake in
`synth` exited 2, and which you got depended on which `except` clause a
command's author had written. That message now exits 2:
`tests/test_errors.py::test_no_pairs_left_after_filtering_now_exits_2`.

*Alternatives.* (a) A decorator on each command callback — the item's own
proposal. Rejected: it is exactly the thing a new subcommand forgets, and the
Done criterion ("config errors exit 2 from *every* subcommand") then depends on
a test noticing an omission rather than on the omission being impossible.
`Group.invoke` is the funnel every descendant's body passes through, so one
`try` there covers subgroups and anything added later at any depth. (b) Leave
`click.ClickException` at each raise site inside the libraries — rejected years
ago and re-rejected here: a library reachable without click must not hand its
caller a CLI exception (this is why `synth_cli` grew its hierarchy at all). (c)
One class with a `code` attribute — a number at every raise site is a worse
prompt than a name that says which *kind* of wrong it is.

*Constraint — every class subclasses the builtin it replaces.* `ConfigError` is
a `ValueError`, `RunError` a `RuntimeError`, `MissingExtra` also an
`ImportError`. That is what let ~90 raise sites across 14 modules move in one
pass without breaking a single `except`: `serve` still turns a bad editor
selection into an NDJSON `{"error"}` field with `except ValueError`, `session`
still wraps `capture`'s refusals, and livingthing's front-ends kept working
before they were switched over.

*Constraint — the vocabulary is for failures the user can act on.* A violated
contract between stylebot's own functions (`select_targets: role= requires
splits=`, an option claimed by two bundles, an unknown `disposition` no flag can
produce) stays a bare `ValueError`. Those are bugs, and a traceback is the right
report; `translate_errors` deliberately lets them through, pinned by
`test_a_bare_builtin_is_not_translated`. This is what kept the sweep from
becoming a mechanical s/ValueError/ConfigError/ over every raise in `src/`.

*Blog side.* `dan-style synth`/`run`/`style`/`serve`/`eval`/`train` had five
hand-written copies of the same translation, and the `synth` one had already
lost `param_hint` — a bad `--slop-strategy` named the flag under `ai-style` and
said a bare "Error:" under `dan-style`. All five now call `translate_errors()`.

**Re-evaluate if** a third class earns its place — the candidate is a
"partial success" that wants a distinct code (today `SynthGenerationError`
handles it as a `quiet` `RunError`: exit 1 with nothing further printed, because
`run_synth` has already rendered every error and the summary).

## D48: `Styler` is a Protocol, and it types the call, not the tokens

**Decision.** `stylebot.infer.Styler` is a `Protocol` declaring
`__call__(messages: list[dict], num_samples: int = 1) -> list[str]`, replacing
`Callable[..., list[str]]` — which typed the return value and neither argument,
so a scorer, or a backend taking `n=` instead of `num_samples=`, checked fine.
`tests/test_backends.py` checks the real backends' signatures against the
Protocol's own, rather than against a copy of it.

*What the Protocol does not close, said out loud.* `tinker_backend` renders the
prompt with the cookbook renderer training used, so the tokens are built by the
same code. `openai_backend` hands the message list to a server that applies its
**own** chat template; a template that wraps the system turn differently,
injects a reasoning block, or prepends a default system prompt puts the adapter
out of distribution, where it stops transforming and returns its input —
silently, with no error anywhere. Both conform to `Styler`. So the Protocol
closes the argument-shape hole and *documents* the parity one. Anyone reading it
as a parity guarantee is reading it wrong, which is why the docstring says so
and `test_the_protocol_cannot_see_the_chat_template` pins it.

*What was built instead, because it is cheap and it fires at the right moment.*
`backends.manifest_mismatches` now emits `renderer unverifiable` when the
backend is `openai` and the manifest pins a renderer. Serving a manifest-pinned
adapter through an OpenAI-compatible endpoint used to look identical to serving
it correctly — both silent — and "not checked" is a different claim from
"checked and fine".

*Rejected: the raw-completion mode* (render with the cookbook renderer, POST to
`/v1/completions` with a `prompt=` string). Costed, not hand-waved:

1. `Renderer.build_generation_prompt` returns a `tinker.ModelInput` — **tokens**,
   not text. Feeding an OpenAI-compatible endpoint means detokenising and
   letting the server re-tokenise, so the mode trades a chat-template question
   for a detokenise/retokenise-fidelity question. It does not close the gap; it
   moves it somewhere with no measurement behind it.
2. The renderer lives in `tinker_cookbook`, i.e. the `trainer` extra — so the
   backend whose entire appeal is "serve locally without tinker" would acquire
   tinker to be trustworthy.
3. `/v1/completions` is not universal across the endpoints this backend exists
   for (`mlx_lm.server` and vLLM have it; not every target does).
4. The half that is testable without a paid call is the rendering, which is the
   half that was never in doubt. Whether a real server agrees is exactly what
   cannot be tested for free — and *is* what
   `findings.md` "MLX local-serving parity" already measured, once, by scoring
   the same inputs through both backends.

**Re-evaluate if** a second endpoint has to be trusted without a parity sample,
or if the cookbook grows a text-level `build_generation_prompt` (which would
remove objection 1, leaving only the extra).

## D49: asyncio is deferred; the rest of the eager import block stays

**Decision.** `stylebot.synth` imports `asyncio` inside the four coroutine
functions that use it, not at module scope. Everything else in
`bin/ai_style.py`'s import block stays eager.

*Measured 2026-09-03*, `ai-style --help`, median of 12 subprocess runs on Dan's
laptop (interpreter floor 9.0 ms; `import click` alone 18.7 ms):

| | median |
| --- | --- |
| before (this branch's starting point) | **49.8 ms** |
| after | **43.7 ms** |
| hypothetical: `synth`/`synth_cli` also gone | 38.4 ms |

A/B'd on the finished tree — `-c "from stylebot.bin.ai_style import main"` with
and without a preceding `import asyncio`, 15 runs each — asyncio alone is
**4.9 ms** (48.5 → 43.7); the rest of the 6.1 ms is run-to-run drift and the
other items on this branch. Either way it is ~10-12% of the command for a
five-line change. The whole of `synth` + `synth_cli` is a further 4.2 ms.

W11.i's premise ("`synth`/`asyncio` cost ~30 ms of a 70 ms `--help`") does not
reproduce: `--help` was 49.8 ms here, not 70, and the two together were ~9 ms,
not ~30. The win is real but it is a third of what was claimed, and saying so is
the point of the item.

*Rejected: a lazy-command group for `synth`.* `synth_cli.synth_options()` is a
**decorator**, so the option bundles must exist when the module is imported;
deferring them needs a `click.Group` subclass that overrides `format_commands`
to render a hand-declared short help without calling `get_command` — because
click's own `--help` path imports every subcommand to ask it for one. That is a
reimplementation of click's help rendering, plus a second place the command list
is written down, plus a `list_commands`/`commands` divergence that
`tests/test_errors.py`'s tree walk and `test_smoke.py`'s `_params()` both reach
into. 4.2 ms of 43.7 does not buy that.

*And the eager block is not an accident.* `classify_train`, `splits`, `train`
and `eval` are imported at module scope precisely so click owns their defaults
directly (`default=tr.DEFAULT_LORA_RANK`) instead of the triple declaration that
preceded it. Deferring them would undo a deliberate earlier choice to buy back
~10 ms; `stylebot.lib` (3.9 ms, almost all `ruamel.yaml`) is unavoidable anyway
because `is_human_authored` is a click default here.

*Cost of the deferral, recorded so it is not a surprise.* `synth.asyncio` is no
longer an attribute, so `monkeypatch.setattr(synth.asyncio, "sleep", …)` had to
become `monkeypatch.setattr(asyncio, "sleep", …)` — the same module object,
patched one level out.

**Re-evaluate if** `--help` latency is ever measured as a real irritation (it is
not today: 43.7 ms is below the ~100 ms threshold where a shell feels slow), or
if click gains first-class lazy subcommands, which would remove the whole
objection above.

## P1 (PROPOSAL, 2026-08-10): move `openai` to an `[api]` extra — REJECTED, see below

**Proposal**: demote `openai>=1.60.0` from `[project].dependencies` to a new
`[project.optional-dependencies] api` extra, alongside `classifier` and `trainer`,
and give the three lazy import sites a friendly "install `stylebot[api]`" error
instead of a bare `ModuleNotFoundError`.

**The observation**: every `openai` import in the package is lazy, at exactly
three sites — `eval.openrouter_judge`, `infer.openai_backend`, and
`generators.openai_generator` (which `openrouter_generator` is built on).
Verified by blocking the module on `sys.meta_path`: all 14 stylebot modules,
including `bin.ai_style`, still import, so the whole `--help` tree survives. Only
those three factories raise. So the SDK plus its 8 transitive packages (anyio,
distro, httpx, jiter, pydantic, sniffio, tqdm, typing-extensions) is currently
mandatory for installers who never touch a network model.

**What would actually break** without the extra installed:

| breaks | survives |
| --- | --- |
| `ai-style synth` (every generator factory is OpenAI-compatible) | `log`, `make-splits`, `export-table` |
| `ai-style eval --judge` | `eval` without `--judge` (Vale / detector / eyeball) |
| `run` / `serve` with `--backend openai` | `run --backend tinker` (the default), `serve` score-only (backend default `None`) |
| | `train` (tinker), `train-clf` (classifier extra), `stylebot.classify` |

**Not a packaging break for livingthing** — the thing that looked like the
blocker, and isn't. livingthing declares `openai>=1.108.0` as its own direct
dependency and imports it in ~9 of its own modules (`ai_core`, `refs/*`,
`workflow_preen`, `diagnostics`), so it gets the SDK regardless of what stylebot
declares. Its `stylebot[classifier,trainer]` line would need no change, and
`dan-style synth` would keep working. Any *other* consumer that installs plain
`stylebot` and calls `synth` would break, and there are none known.

**Alternatives considered**: (a) status quo — one fewer failure mode, at the cost
of making the SDK mandatory for the score-only path; (b) finer-grained extras
(`[judge]` / `[synth]` / `[serve-openai]`) — three extras for one wheel is worse
ergonomics than the problem; (c) declaring `stylebot[api]` the documented default
install so only deliberate minimal installs feel it.

**Deciding constraint (unresolved, hence a proposal)**: `synth` is a
first-class subcommand, not an add-on, and an extra makes the plain
`uv add stylebot` install unable to run it. That is a real ergonomic regression
traded against a leaner `classify`/`serve` install. Worth noting the docstring
framing is narrower than it first reads: `classify.py` claims
"ML-dependency-free **at import**", scoped to numpy/sklearn/sentence-transformers,
and that claim is literally true today. The gap is at the *package* level, not the
module level.

**Ruled out — `ruamel-yaml` is not the same case.** It is imported eagerly at
`lib.py:5`, module top level, and the frontmatter path is core to `log` /
`targets` / `synth`. It stays a hard dependency.

**Re-evaluate if**: stylebot acquires an installer other than livingthing (a PyPI
release, or a second consumer that only wants the classifier runtime); or a
non-OpenAI generator backend lands, which would make `synth` survivable without
the SDK and remove the main objection.

**Note for whoever accepts this**: the friendly-error half touches `eval.py`,
`infer.py` and `generators.py`, so it cannot ship from a `pyproject.toml`-only
change.

## P1: REJECTED 2026-09-03 — `openai` stays a hard dependency

The proposal above (move `openai>=1.60.0` to an `[api]` extra) is closed. Its
analysis is left intact because the rejection is about the balance, not the
facts, and the facts are still true: re-verified 2026-09-03 by blocking `openai`
on `sys.meta_path` — all **30** modules in the package still import, so the
entire `--help` tree survives without the SDK, and only the three lazy factories
(`eval.openrouter_judge`, `infer.openai_backend`, `generators.openai_generator`)
raise.

*Neither re-evaluate-if trigger has fired.* There is still no installer other
than livingthing — no `.github/workflows` in this repo, no PyPI publish path,
version still 0.3.0 — and there is still no non-OpenAI generator backend
(`generators.py` defines exactly `openai_generator` and `openrouter_generator`).

*And W3 moved the balance further against it.* Bare `stylebot` is now a **hard**
livingthing dependency rather than a member of the `styler` extra, so the lean
`uv sync --no-group styler` install pulls it — but livingthing declares
`openai>=1.108.0` itself and imports it in ~9 of its own modules, so the extra
would save that install exactly nothing. The only beneficiary is a third
consumer that wants `stylebot.classify`'s runtime and nothing else, which is
precisely the trigger that has not fired. The ergonomic cost is certain (`uv add
stylebot` gives you a `synth` subcommand that cannot run) and the benefit is
hypothetical.

**Re-evaluate if** either original trigger fires — a PyPI release or a second
consumer, or a non-OpenAI generator backend. The friendly-error half is the
cheap part and can land alone: `eval.py`, `infer.py` and `generators.py` would
raise `MissingExtra("… needs the 'api' extra")` (D47) at their three lazy import
sites, which is a strict improvement over a bare `ModuleNotFoundError` whether
or not the packaging ever changes.

## D50: a corpus is a directory of shards, split by producer and then by month

**Decision (W6, 2026-09-03).** `pairs.jsonl` becomes `pairs/`: one
`captures.jsonl` for real edit captures, one `synth-<YYYY-MM>.jsonl` per month
of synthesis. `stylebot.corpus` owns the routing rule (`shard_of`), the
shard-aware rewrite (`rewrite`, which does not touch a shard whose bytes would
not change), the one-per-corpus write lock, and the manifest content hash.
`config.pairs_path` — D45's single spelling — resolves to the directory when it
exists and to the single file otherwise, so **sharding is opt-in by existing**:
a plain `ai-style` install never migrates and never learns shards are a thing.
`ai-style data shard-pairs` is the one-off (D38's hidden group). Every `--pairs`
option now accepts either shape, and `iter_pairs` is the seam that makes every
reader agnostic.

*Why anything changed.* Committing the corpus is right — it is the backup story
for 398 irreplaceable human captures — but every append rewrote 38 MB whole, so
124 versions of the file carry 1.51 GB of logical blob content. Sharding does
not shrink the corpus. It shrinks *the file a write touches*, which is the
quantity the commit count multiplies: `log save` (which read-modify-writes on
every save, to apply its replace rule) now rewrites 1.7 MB, and a synth run
appends to one month and leaves the rest byte-identical.

*Alternatives.*

- **git-lfs on `pairs.jsonl` and `captures/`.** No code change, but GitHub's LFS
  bandwidth quota is charged on every clone of a Pages repo whose build does not
  need the file at all, and it makes the corpus unreadable to anyone without the
  LFS client. Rejected.
- **Shard by generator** (`synth-<model>.jsonl`), which the W6 brief offered
  first. Rejected on three counts. A generator shard grows forever, so the
  active blob is unbounded and every future run rewrites a bigger one; a month
  shard is sealed when the month ends. Generator identity is a *facet*, already
  in `meta.gen.model` where it can be selected on — putting it in a filename
  makes a second, weaker copy that a retired arm strands and a model rename
  forks. And generator ids are not filenames (`openrouter/qwen/qwen3.6-35b-a3b`),
  so it needs a lossy sanitiser and a collision rule that `2026-08` does not.
- **Route on read-identity rather than content** (write each record back to the
  shard it was read from). Rejected: routing by content makes `rewrite`
  idempotent and self-healing, and makes the migration the same function every
  rewrite already calls, at the cost of needing a home for a line that does not
  parse (`unparsed.jsonl`).

*Constraint — and the one thing sharding could have broken.* D36's
no-duplicate-cell guarantee is enforced by `PairsWriter`, which locked
`pairs.jsonl` itself. Locking each shard instead would let two runs either side
of a month boundary write the same cell twice, which is the 2026-08-12 defect
wearing a hat. So the lock is one file per **corpus** (`.pairs.lock`) and the
catch-up tail-reads **every** shard before deciding. `record_id` uniqueness is
likewise checked across the corpus, not per shard, which is why
`validate-pairs` wants the directory rather than each file in turn.

*What did NOT change.* No record was altered: the live migration round-tripped
8,182 records with an identical `record_id` set and an identical content
multiset, 0 problems and 0 duplicates. Shard membership is derived from
`meta`, so it is recomputed, never stored — nothing in a record knows which file
it is in.

**Re-evaluate if** a month's synthesis grows past a few hundred MB (split by
half-month, or by producer within the month — the routing rule is one function),
or if the corpus stops being committed at all, which would make the whole
question moot.

## D51: `git filter-repo` on the corpus is not worth a force-push — measured

**Decision (W6, 2026-09-03).** Do not rewrite livingthing's history to purge old
`pairs.jsonl` blobs. Delete `.git/subtree-cache` instead.

*The measurement that decided it.* The W6 brief cited "1.56 GB of blobs under
`_training_pairs/`, 1.54 GB of it 122 versions of a 38 MB `pairs.jsonl`, in a
repo whose `.git` is 1.8 GB" — which reads as the corpus being ~85% of the
repository. That figure is **logical (uncompressed) blob size**
(`cat-file --batch-check='%(objectsize)'`). On disk, in the packfile
(`%(objectsize:disk)`), the same 124 versions cost **49.5 MiB**, because an
append-only JSONL deltifies almost perfectly. Everything under
`_training_pairs/` is 52 MiB of the 1,107 MiB of packed blob content: **4.7%**.

Meanwhile `.git` is 1.9 GB, of which `.git/objects` is 1.2 GB and
**`.git/subtree-cache` is 683 MB** — a `git subtree` scratch directory, not
history at all. `rm -rf .git/subtree-cache` reclaims 13× what a filter-repo pass
on the corpus would, needs no rewrite, no force-push, and no coordination with
anyone who has a clone.

*Alternatives.* Run the filter-repo pass anyway: ~50 MiB reclaimed for a
force-push to a GitHub Pages repo, every commit sha in the blog's own prose and
in stylebot's run manifests invalidated (manifests pin `blog_sha` — D36), and
every existing clone and worktree needing re-cloning. The script is written and
reviewable (see the W6 handover) so the option stays open; it is simply not
worth exercising for 4.7%.

**Re-evaluate if** the corpus stops deltifying — a change that reorders or
rewrites records wholesale on every commit would defeat it, which is exactly
what D50's shard-level rewrite avoids — or if a genuinely large binary
(a merged adapter, a `.joblib`) is committed by accident, where a targeted
filter-repo on that one path is the right tool and the argument above does not
apply.

---

## D52: documentation carries no mutable operational state — `status` answers instead

**Decision (W7, 2026-09-03).** stylebot's prose — `CLAUDE.md`, `README.md` and
`_plans/OVERVIEW.md` — states **no** corpus count, run number, "serving
default" or artifact sha. Those facts move under git's feet, so a new
subcommand computes them on demand: `stylebot.status.collect()` sweeps the five
artifacts that answer "where is this pipeline?" — corpus, `splits.json`, the
runs directory, the detector artifact, and the *cross-checks between them* —
and `ai-style status` / `dan-style status` print it.

`findings.md` and `decisions.md` are explicitly **exempt**, and the distinction
is the whole design. A dated measurement ("over the live 4,595-pair corpus,
2026-08-11") is a record of what was true then and must not be edited to match
today; a status sentence ("the corpus stands at 4,595 pairs") claims to be true
now and cannot stay so. The W7 brief's done-criterion — `rg "run 2|4595|455
real" stylebot/_plans` returns nothing — is therefore wrong as literally
written: satisfying it means deleting the measurements the file exists to hold.
What was applied instead is that every surviving occurrence of such a number is
attached to the date it was measured, and every *current-state* sentence is
gone. Where run 2 appears as "the serving default", the sentence now says it
served from 2026-07-22 to 2026-08-21 and points at `status` for now.

**The evidence.** Six prose surfaces describe this system across two repos. On
2026-09-03 the stylebot half said two trained runs existed, run 2 was serving,
and the corpus was 4,595 pairs (4,140 synthetic). The blog held five completed
runs with 4a (`20260817-170037`) promoted on 2026-08-21, and 8,182 records (398
real / 7,784 synthetic) sharded across three files. Every one of those sentences
was correct when written, three weeks earlier. None of them could have been kept
correct: the facts live in the *blog's* `_training_pairs/`, a different repo on
a different commit clock, and a stylebot commit is not what changes them.

**Why `ai-style status` exists and not only the blog mirror.** stylebot cannot
discover where a blog, a runs directory or a detector artifact is — that is the
one-way dependency arrow, and inventing a `<data-dir>/runs` convention to make
`status` self-configuring would have broken it for cosmetic convenience. So the
generic command takes each artifact as an explicit flag, reports "not given" for
anything unnamed, and reports absence rather than raising for anything named but
missing (status has to run on a half-built pipeline; that is when it is most
useful). `dan-style status` is the same function with all five filled in from
`livingthing.training_targets` — a 25-line policy shell, the same shape as every
other mirror.

**The cross-checks are why this is one function rather than four commands.**
Corpus size, a promoted run and a detector sha are each printable from their own
artifact. What no artifact can see is the *disagreement between them*: a
detector fit under `splits_sha256=X` while `splits.json` now says `Y` is
internally consistent and jointly wrong — every verdict it produced held out a
different set of posts (D39, D42). Same for a promoted run trained under a
partition that has since moved, and for a run that predates the pins entirely,
which is what run 4a's manifest reports today. `check_splits`' danger report is
folded into the same list.

**The `promoted` rule moved into the library** (`status.promoted_manifest`):
newest manifest with a `result` and without `"promoted": false`. The blog's
`style_file.latest_manifest` had owned it, and `status` had to answer "which run
serves" identically or it would be lying in exactly the way this entry exists to
stop. `latest_manifest` is now that call plus a click error for the empty case.

*Alternatives.* **A section of `ai-style log list`** (the W7 brief's other
option): rejected because `log` is the capture session group — its `list` shows
open snapshots, and a pipeline-wide report is not a session. **Keeping the
numbers and adding a CI check** that greps docs against the live corpus:
rejected because the check can only run where the blog is checked out, which is
the coupling the arrow forbids, and because it would fail on `findings.md`'s
legitimate history. **A generated section written back into the docs by a
script**: a committed artifact of a machine reading another repo, stale between
runs, and a merge conflict on every capture.

*The deciding constraint.* A document that cannot be kept true is worse than no
document, because it is believed. Six surfaces × two repos × a corpus that
changes daily is not a discipline problem to be solved by remembering harder.

**Re-evaluate if** the corpus and the tooling ever live in one repo (then a
number in a doc can be updated by the commit that changes it, and a test can
pin it), or if `status` grows past what one screen can carry — at which point
the split is by artifact, not by moving facts back into prose.

---

## D53: the orphan prune is built and declined; the eval prune is applied and now needs a second pass

**Decision (W12, 2026-09-03).** `ai-style data prune-stale` ships, dry-runs by
default, keeps a `.bak` and carries a test. Its **eval-contamination** half was
applied (blog `c0e2ac85be`: 62 synthetic pairs cut from frozen-eval prose, 41
real captures spared). Its **orphan** half was measured and **not run**, and
that is the decision rather than a postponement. Full numbers:
[`findings.md`](findings.md) §"What a stale-corpus prune would cost and buy".

**W12's stated premise turned out to be false, and that is what settled it.**
The item was written expecting W2's segmentation change to have orphaned cells
the corpus would keep paying for. Re-cutting every source at its pinned revision
under the composed blog policy: W2's fence rule orphaned **zero** cells — the
two files whose protected spans it moved carry no corpus cells at all. Of the
606 orphaned cells (672 pairs, 8.6% of the synthetic corpus), **492 are prose
Dan has edited since the pair was generated**. That is the corpus being a frozen
record of prose that moved, which is neither damage nor a reason to delete
anything. Only 97 come from a policy change, and 70 of those are `div_is_prose`
charged against the **section-bounded** arm — frozen policy, so no future synth
run will refill those cells and pruning them is permanent loss of paid-for data.

**And it would not have bought what it was wanted for.** The orphan prune moves
the effective per-passage weight D43 measured from **4.06 to 4.04**, because
orphans are spread across 124 posts roughly in proportion to the corpus: it
thins passages and pairs together. Dilution is a weighting problem
(`per_passage` / `pair_weight`), and a prune is not a substitute for one.

*Alternatives.* **Prune orphans anyway**, on tidiness grounds: rejected — it
deletes 672 paid-for pairs to buy a 0.02 change in the number that motivated it,
and 349 of them can never be regenerated. **Delete the command** now that it has
been declined once: rejected — unlike the migrations D38 keeps, this one is a
*measuring* instrument as much as a destructive one, and "how much of the corpus
would today's policy no longer produce?" is a question worth being able to ask
before every retrain. It stays in `ai-style data` and dry-runs by default.

**The open half: the eval stratum is 16% synthetic again.** The two W10/W12
changes landed in the order that undoes one of them. `c0e2ac85be` pruned the
frozen stratum to 0% synthetic; `5bf689fa5a` then grew it (eval_v2, six
real-capture posts) and `notebook/ai_search.qmd` brought **28 synthetic pairs**
in with it. So `check-splits` today reports eval at 177 pairs, 149 real / 28
synthetic. It is well under the `EVAL PURITY` threshold and the headline verdict
is the real facet by construction (D41), so nothing computed is wrong — but the
clean-stratum invariant W12 established is not currently held, and
`extend-eval` does not prune. **Not fixed here**, because it is a corpus write
and W7 is a documentation item. The fix is one `prune-stale --no-orphans
--apply` followed by a detector refit (the refit is not optional — the head
currently in `_models/voice-clf` was fit *after* eval_v2, so it already holds
those posts out, but its `splits_sha256` would move with the corpus). The
durable fix is for `extend-eval` to offer the prune for the posts it is about
to freeze, which is where this belongs: the two operations are one intent.

**Re-evaluate if** a synth run's `--skip-covered` accounting starts disagreeing
with the corpus (orphans are exactly the cells that make it), or if the corpus
stops being cut at multiple budgets — at which point orphan pairs stop being
cheap to leave lying about and the 4.06 stops being the dominant term.

---

## D54: a document-scale verdict is a mean over serving-scale chunks, unaligned across fields

**Decision (2026-09-03, the same day as the measurement that forced it).** The
detector is never handed a whole document. `eval.chunked_detector` wraps any
`Detector` so a passage is first segmented exactly the way serving segments it
(`infer.prose_chunks` — the promoted diet, the same protected blocks, the same
budget), each prose chunk is scored on its own, and the passage's number is the
**unweighted mean over its chunks**, carried in the ordinary
`scores[field]["detector"]["score"]` slot with `n_chunks` and `sd` beside it, so
`summarize_scores`, `format_summary` and `render_scores_report` change not at
all. **The chunks are not aligned across `slop` / `target` / `output`**: each
field is segmented and aggregated independently, and the verdict compares two
unaligned means.

The before/after is in [`findings.md`](findings.md): the whole-document reading
scored 0.4880 / 0.4951 / 0.4914 with byte-identical min and max across all three
fields and the styler's output *above* the draft in 34 of 34 documents;
chunk-wise it is 0.3844 / 0.3622 / 0.3658, Dan below the draft in 25 of 34, and
a post-clustered CI on both `target − slop` and `output − slop` that excludes
zero.

*Alternatives.* **Pair the chunks across fields** and report a paired
chunk-level delta, which is what the first write-up of the problem assumed:
rejected on the data. An edit is not a rewrite — Dan moves paragraph boundaries,
so the three variants of a capture cut to different chunk counts (563 / 564 /
563 here, near-equal by coincidence and not by correspondence) and field *i*'s
chunk *k* is frequently not field *j*'s chunk *k*. A chunk-index join would
produce a tighter interval by pairing things that are not pairs. **Segment at
the detector's own training scale** (~1.2k chars) rather than the serving diet:
rejected because the corpus the head was fit on contains 4500-budget cells too,
so the serving diet is inside its range, and because scoring at a scale the
styler never worked at reintroduces the gap `--documents` exists to close.
**A length-weighted mean**: rejected as unmotivated — the chunk is the head's
unit of observation, and every other mean in the harness is per-observation.
**A median or a max**: rejected; a max turns the verdict into "the worst
paragraph", which is a useful *editing* signal and a bad *comparison* one.

*The deciding constraint.* An instrument has a range, and a StyleDistance
embedding of 25k chars is outside it — the shared 90-odd percent of two variants
of one document dominates the vector and the difference washes out. Nothing in
the harness knew that, because the detector had been built, validated and always
used at chunk scale; its first use outside that scale returned 0.49 for
everything and said so in a table. Scale is therefore part of the instrument's
identity, not a property of the caller: `chunked_detector` appends it to the
head's `name` (`voice-clf/chunked:4500:inline-headings`), so D39's mixing guard
refuses to blend a whole-document reading and a chunk-wise one into one mean —
which is exactly what it would otherwise have done on a resume.

`sd` is recorded because a mean with no spread beside it hides the case the mean
exists to expose: a document half Dan and half slop reads identically to a
uniformly middling one. Measured, the within-document sd is 0.2642 against a
between-document 0.0812 — documents vary far more inside themselves than from
each other, which is the mechanical statement of why one embedding of one told
us nothing. It is a dispersion, not a standard error; the interval still comes
from `paired_delta` clustered by post.

**Re-evaluate if** the backbone changes to one whose context window covers a
whole post (then the aggregation is a choice rather than a necessity, and a
whole-document embedding becomes worth re-measuring against this), or if the
eval stratum ever supplies enough captured posts that the cluster count — 7 for
these 34 documents — stops being the binding constraint on the interval.

---

## D55: the scoring pass runs in a child — and the crash it was built for was a different bug

**Superseded by D59 (2026-09-04).** The child, its `ScoreJob` marshalling and
the `classify` lock it shipped beside are deleted: with the thread pool gone,
one task drives the embedder, so the crash the isolation made legible cannot
happen and the process boundary buys only a second model load.

---

## D56: `asyncio` first; threading must stringently justify itself

**Decision (2026-09-03, Dan).** A standing project-level constraint, recorded
here because it is the kind of thing later work has to be able to find and
re-open. `asyncio` is this project's concurrency mechanism for IO, and it is
what `synth` already uses at scale (`synth._run_sessions`: one event loop, an
`asyncio.Semaphore` for the worker cap, one append handle, no locks). **New
threading needs a measured substantial win in performance or complexity** — at
least one of the two, with a number, and a `decisions.md` entry carrying it.
The rule as written is in [`CLAUDE.md`](../CLAUDE.md) (`4f4d07c` here, blog
`04909d1ddc`); this entry is why it exists and what was rejected.

The precipitating case is D55. `eval.score_pairs_file` fanned the judge *and*
the detector over a `ThreadPoolExecutor(max_workers=8)`; eight threads sharing
one `SentenceTransformer` segfaulted the process deterministically once
chunk-wise scoring raised the embed rate ~6x. It was latent for months, the
first diagnosis was wrong, and the fix cost a lock plus a subprocess-isolation
layer. Every threading episode in these two repos has cost hours of lock or
segfault diagnosis; the asyncio paths have produced no crashes and no lock
trouble.

*Alternatives.* **Ban threading outright**: rejected, and the audit below is
why — `focus.py`'s daemon HTTP server and `library_archive`'s parallel
downloads are both threads that clearly earn their keep, and a blanket ban
would push them into worse shapes (a second process, an async rewrite of a
blocking-httpx download loop). **Leave it to judgement case by case**:
rejected, because judgement is exactly what produced the eval pool — `Judge`
and `Detector` are plain sync callables and `ThreadPoolExecutor` is what one
reaches for to wrap those, so the reflex needs a rule in front of it rather
than a reminder. **Make the bar "no shared mutable state"**: rejected as too
narrow; the eval pool shared no *Python* state at all. What it shared was a
torch model and a Rust tokenizer, which is invisible at the Python level, and
that is precisely why the failure was a segfault rather than a wrong answer.

*The deciding constraint.* The bar is EITHER a substantial reduction in
complexity OR a substantial increase in performance, **measured**. Both halves
matter. "Measured" rules out the eval pool, which had never been timed and
turns out to buy nothing for the work it was doing. "Either/or" keeps the
daemon-thread cases, which win on complexity and would lose on a
performance-only bar. Three corollaries carried in `CLAUDE.md`: never share a
non-re-entrant native object across threads; **a lock as the answer means the
threads were the wrong shape**, not that the problem is solved; and for
CPU-bound work prefer **batching over fan-out**, `_vale_prepass` being the
pattern and D57 the second instance of it.

**Re-evaluate if** a free-threaded (no-GIL) interpreter becomes this project's
runtime and the CPU-bound sites stop being GIL-bound — at which point fan-out
over a thread pool becomes a real alternative to batching for the first time,
and the native-object corollary becomes *more* important rather than less.

---

## D57: the detector is batched out of the pool; the pool is kept for the judge alone

**Superseded by D59 (2026-09-04).** The batched detector pre-pass stands and is
unchanged; the pool it was kept beside does not — the judge moved to an
`asyncio.Semaphore` at parity, which is the rewrite this entry rejected as
buying nothing (it buys 1.15x, and three mechanisms).

---

## D58: size budgets are a test with a shrinking allowlist, and each "why" has one layer

**Decision (2026-09-04)**: adopt P1–P8 of the blog's
`_plans/stylebot-seam-followup-2026-09.md` — one layer per "why" (decisions →
findings → code → commit), size budgets on modules, functions and docstrings,
and dispatch rules for agents. `tests/test_budget.py` enforces the sizes against
an allowlist of everything over budget today; the blog runs a copy over
`src/livingthing`. The allowlist is the record of the sizes as measured.

**Alternatives considered**:

- **An unenforced guideline in `CLAUDE.md`** — already there in spirit, and
  2026-09-03 added 12,277 lines under it.
- **Hard budgets with no allowlist** — red on day one, so the first agent to
  meet it deletes the test rather than the bloat.
- **One budget module shared by both repos** — a test-only module in stylebot's
  public API, and a blog suite that goes red on a stale git pin.

**Deciding constraint**: the bloat was a good instinct (say why) applied at four
layers at once, so the same paragraph exists three or four times and every later
reader pays for all of them. Review does not see that; a line count does. The
allowlist is what makes a number safe to adopt mid-flight — a ratchet, not a
cliff — and a deleted entry is the only evidence of progress the process keeps.

**Re-evaluate if** an allowlist entry has to *grow* for a change that is plainly
right: the budget is then wrong, and what gets written is the new budget, not
the exception. The enforced package-wide number is the prose line count rather
than P2's ratio, because deleting mechanism raises the ratio and F2/F8 exist to
delete mechanism; a ratio enforced literally would reward keeping it.

---

## D59: one mechanism in the scoring path — the judge awaits, everything else batches

**Decision (2026-09-04).** `eval.score_pairs_file` has exactly one concurrency
mechanism: an `asyncio.Semaphore` over the judge. A slice runs one pre-pass per
signal — Vale batched, the detector batched, then the judge gathered
`max_workers` calls at a time — and assembling records is bookkeeping.
Deleted: the `ThreadPoolExecutor`, the `threading.Lock` in `classify`, the
fallback-into-the-pool in `_detector_prepass`, and the `ScoreJob` child with the
`-m stylebot.eval` entry point it spawned. Added: `openrouter_judge_async`, the
judge a corpus pass takes, and `score_pairs_file_async` under the unchanged sync
signature. `detect_many` moves to `classify`, so `serve` stops importing `eval`.

**Alternatives.**

- **Keep the pool for the judge alone** (D57's position, reversed here):
  rejected on the measurement — parity, in fact 1.15x better, because the
  semaphore gates calls where the pool gated records.
- **Keep the child for loud failures**: rejected. Its diagnosis was disproved
  (D55), the segfault it made legible was the pool's, and no pipe launders an
  exit status any more.
- **Run a sync judge under `asyncio.to_thread`**: rejected — the pool again in
  an async hat; a sync judge is documented as serial (103.4 s) instead.

**Deciding constraint**: D56's bar is a measured win in performance *or*
complexity, and the pool cleared neither once the detector left it
([`findings.md`](findings.md) "Threading audit", table 6). Four mechanisms had
been stacked on one path in a day, each containing the previous one's failure.

**Re-evaluate if** a second remote signal joins the judge (the gate then belongs
per signal), or if anything drives the embedder from more than one thread again.

## D60: a rewrite is a writer — it takes the corpus lock, and appenders re-check the inode

**Decision (2026-09-04)**: `corpus.rewrite` runs under `corpus_lock`, and
`PairsWriter` compares its open handle's `fstat` (and its remembered byte
offset) against the shard's current inode before trusting either. The lock stays
non-re-entrant and stays at one file per corpus.

**Alternatives considered**:

- **Lock only the appender** (the shipped state). Rejected: `rewrite` replaces
  each shard by rename, so an append landing inside one is lost, and the
  writer's `ab+` handle is left on an unlinked inode — every later append in
  that run then vanished with `write()` still returning `True`.
- **Hold the lock for `PairsWriter`'s lifetime** instead of per write. Rejected:
  a synth run is minutes to hours long and would block `log save` for all of it,
  which is the daily write on the irreplaceable captures shard.
- **Make the lock re-entrant** so the six read-filter-write callers could wrap
  read *and* rewrite. Deferred: a new mechanism (P8) for callers none of which
  nest today.

**Deciding constraint**: the corpus holds 398 human captures that cannot be
regenerated and the failure was silent — no exception, no short count, a `True`
return. Cheap detection beats a bigger lock: inode identity is one `stat` per
shard per write, and it makes the loss impossible rather than unlikely.

**Residual**: a caller that reads, filters and then rewrites holds nothing
across that gap, so an append arriving in it is still overwritten — narrower
than before (it was the whole rewrite), not closed. **Re-evaluate if** synth and
`log save` are run concurrently in anger, or a second appender appears: the six
callers then take the lock across read and rewrite, and it becomes re-entrant.

## D61: the executables package holds no libraries — `synth_cli` left `bin/`

**Decision (2026-08-11, recorded 2026-09-04).** 1000+ lines of synth
orchestration moved from `bin/` to `stylebot.synth_cli`, raising its own
`SynthConfigError` / `SynthRunError` / `SynthGenerationError`; `bin/ai_style.py`
translates them to exit codes at the boundary (now `cli_options.translate_errors`,
D47).

**Alternatives.** Leave it and re-export from a library module — rejected:
`bin/` is not an importable contract, and a re-export advertises one that the
package layout denies. Split it across `synth` / `targets` — rejected: it is one
orchestration, and cutting it along module lines would put the run loop in two
files.

**Deciding constraint.** A library reachable only through `CliRunner` gets
copied rather than imported. D35 is the case that proved it: the blog
reimplemented `build_styler` because it lived in `bin/`, and the copy lost
`max_tokens`, truncating every blog rewrite at the 2000-token floor while the
promoted run served a 4500-char diet.

**Re-evaluate if:** never, in this shape — `bin/` is the click shell. Recorded
late because the CLAUDE.md diet (F7a) found this was the one rationale in that
file with no home in any other layer; the rule survived the cut, the why did
not.
