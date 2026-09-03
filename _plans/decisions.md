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
it. (`tests/test_marker_segmentation.py` pins a genuinely separate
implementation and is untouched — until W4 deletes that one too.)

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

**Decision**: `splits.json` gains an optional `aliases` map and `role_of`
resolves every source through it before doing anything else
(`splits.canonical_source`). An entry reads **`{path the post has now: path its
pairs are already keyed under}`** — new → old, which is the opposite of the
direction the name suggests. `ai-style synth` then refuses any target whose post
resolves to `eval`, consulting the corpus's own `<data-dir>/splits.json` when no
`--splits` was passed, and `--role eval` is the only way past it. `ai-style log
save` says so but does not refuse. `check_splits(…, blog_root=…)` reports a
pinned eval post that is on disk under no known spelling as an **error**
(`ERROR_PREFIX`), and `ai-style check-splits` exits non-zero on one.

**The defect.** Roles are keyed on the post's *path*, and paths move: `post-op
move` renamed `notebook/empowerment.qmd` → `notebook/informational_empowerment.qmd`
and `notebook/learning_from_madness_of_crowds.qmd` →
`notebook/wisdom_from_madness_of_crowds.qmd` (blog `77a1b398aa`). Measured
2026-09-03: the new paths hashed to `styler` and `detector`. Nothing caught it,
and the reasons compound — the STALE warning is about pairs, and the old paths
still carried theirs; `role_of` is open-world by design, so an unknown path gets
a role rather than a question; and the canonical corpus run (blog RUNBOOK §6)
passes no `--role`, so the next synth would have cut frozen-eval prose as
training targets. Two independent single points of failure, so the fix is two:
an identity that survives a rename, and a guard that does not wait to be asked.

**Why the alias direction is inverted.** Roles are assigned from `meta.source`
as recorded at capture/synth time, so the corpus is keyed on the OLD path.
Aliasing old → new (the plan's `renames: {old: new}` reading) re-hashes every
existing pair of a renamed post into whatever role the new path draws — a
migration of the corpus, and a silent one for `styler`/`detector` posts, where
nothing pins the answer. Aliasing new → old moves nothing: `test_aliases_need_no
_migration_of_existing_pairs` asserts exactly that over 50 hash-assigned posts.
The pinned list keeps **both** spellings (add-only), and `eval_post_spellings`
groups by identity so six pinned posts under eight names still count six.

**Rejected: a frontmatter `id:` as the stable identity.** The right answer if
the blog had one — it does not (`lib.read_w_frontmatter_text` reads whatever is
there; the quality ladder keys on the path too), so it would have to be minted
across ~1700 posts. Worse, `role_of` takes a string and is called per pair and
per chunk with no filesystem in reach; an id-keyed contract makes it a disk read,
and every existing `meta.source` would have to be rewritten to the new key. That
is precisely the migration W0 says to avoid. The alias map is the same idea with
the path as the id and the map as the rename log.

**Rejected: re-cutting the splits.** `make-splits --force` would repair the
symptom and un-freeze the eval stratum, which is the one thing the contract
exists to prevent.

**Rejected: bumping `schema_version`.** `load_splits` compares the version for
exact equality, so a bump makes every stylebot checkout without this commit
refuse the file outright. `aliases` is additive and optional: a reader that has
never heard of it degrades to exactly today's behaviour rather than to an error.
The cost is that such a reader disagrees about a renamed post silently — accepted
because both repos pin stylebot as an editable path dependency, and because the
disagreement is the status quo, not a regression.

**Why `synth` refuses and `log save` only notices.** Cutting eval prose as a
synthesis target trains the styler on the metric — the eval is no longer frozen
and no downstream number means anything. A *captured* rewrite of an eval post is
Dan's own editing work, is legitimate corpus data, and is kept out of every
fitting stage by the same contract; the only thing worth saying is that the
capture will never train anything.

**Re-evaluate if**: the blog gains real post ids (then the alias map becomes the
migration record and `role_of` can key on the id), or `aliases` grows past a
handful of entries — at which point it wants generating from `post-op move`
rather than hand-editing, which is the natural next step and deliberately not
built here.

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
styler left with 70 real pairs. That is the trade this file records; **it has not
been applied**, because it re-roles 441 pairs and obsoletes the current detector
artifact and every verdict computed with it, which is Dan's call and a retrain.

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
