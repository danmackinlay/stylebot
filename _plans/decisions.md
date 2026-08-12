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

Everything numbered `D…` was **decided and acted on**. A trailing
[Open proposals](#open-proposals-not-decided) section holds `P…` entries that are
written up but still awaiting a call — do not read them as settled.

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

---

# Open proposals (not decided)

Entries below are **awaiting a call from the repo owner**. They are written in the
same shape as the decisions above so that accepting one is an edit, not a rewrite,
but nothing here has been acted on.

## P1 (PROPOSAL, 2026-08-10): move `openai` to an `[api]` extra

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
