# Phase 2 · Synthetic pairs — 🔧 BUILT (scale run cost-gated)

Bulk up the corpus by manufacturing `(slop → Dan)` pairs: take Dan's own
real prose as the *target*, ask LLMs to paraphrase it into slop as the
*source*. Lower-signal than real edit pairs, but cheap and plentiful.

**Parallelisable now** — depends only on the `pairs.jsonl` schema (have it) and
a sample of Dan's paragraphs. Does not need a trained model.

**Blog specifics (wired in 2026-06-27, see `CLAUDE.md` → BLOG INTEGRATION):**
`--blog-root ~/Source/livingthing`; targets are `automation: 0` posts under
`post/`+`notebook/` (`is_human_authored` defaults match, no retarget); corpus at
`~/Source/livingthing/_training_pairs/pairs.jsonl`
(`STYLEBOT_DATA_DIR=~/Source/livingthing/_training_pairs`).

## Inputs

- Training targets — supplied one of two ways (see OVERVIEW "Selection is a
  user-supplied policy"):
  1. **a pre-selected file list** the caller built itself, or
  2. **`--blog-root` + an injected `selector` / `sort_key`** so stylebot does
     the discover→filter→sort→sample walk. The function signature takes
     `selector: Callable[[dict], bool] = is_human_authored` and an optional
     `sort_key` / `sampler`; the CLI exposes the bundled default, library
     callers pass their own.
  `is_human_authored` (the `automation: 0` example) is only the default — Phase
  2 must accept the selector as an argument, never hardcode the predicate.
- `--data-dir` (write): where the resulting `pairs.jsonl` is appended.
- LLM API keys: `OPENROUTER_API_KEY` (one key, many hosted models — the default
  path), optionally `OPENAI_API_KEY` and a local/utility base model
  (`LOCAL_LLM_*`). See `.env.example`.
- `STYLE_SYSTEM` from `stylebot.ai_core` (must match Phase 1 verbatim).

Interface: a `stylebot.synth` function over explicit paths; `ai-style synth`
is the thin CLI wrapper. Paths resolved via `stylebot.config` (flag > env >
default), per OVERVIEW "Interfaces".

**Mechanism vs policy split (built 2026-06-28).** stylebot owns generic
*mechanism*; the blog owns *policy*:

- *stylebot (generic):* `iter_targets` extracts **prose only** via
  `stylebot.lib.editable_prose` / `segment_for_edit` (a stdlib fork of the
  blog's `qmd_core.segment_for_edit`) — dropping fenced code, `$$math$$`,
  `:::` divs/callouts, blockquotes — then applies generic chunk hygiene:
  `min_chars`, `max_chars` (drop, never truncate), `ignore_markers`
  (drop chunks containing a literal stub marker), and link-dump dropping. CLI:
  `--prose-only/--no-prose-only`, `--max-chars`, `--ignore-marker`,
  `--drop-link-dumps/--keep-link-dumps`.
- *livingthing (blog policy):* a thin module supplies the composite
  `selector(meta)` = `automation:0 ∧ quality>6 ∧ not draft ∧ not auxiliary`
  (reusing the blog's quality/`is_auxiliary_post` helpers + stylebot's
  `is_human_authored`) and passes `ignore_markers=["🚧TODO🚧"]`. Posts are
  marked with `🚧TODO🚧` on stub paragraphs.

Measured on the live blog: naive `automation:0`-only ≈ 27.4k junky chunks;
prose-only + hygiene ≈ 19.5k; + quality>6/draft selector ≈ 10k clean targets;
merged into passages ≈ 3.6k (median ~620 chars).

**Heading context (✅ built):** 85% of passages sit under a heading that frames
them, so the section heading is attached as fixed context (verbatim on both
sides; slop generated from the body only) via `Target.context` +
`synthesize_pairs`, on the shared `stylebot.pairs.build_pair_content` contract
with Phase 1. CLI `--heading-context immediate` / `--context-dropout`; blog
policy `HEADING_CONTEXT="immediate"`. See [`heading-context.md`](heading-context.md).

**Slop strategy + multi-source via OpenRouter (✅ built 2026-06-28).** The slop
*prompt* is a first-class knob, so the right kind of slop is found by experiment,
not guessed once:

- `STRATEGIES` (`stylebot.synth`) maps a label → a system-prompt flavour:
  `polish` (neutral baseline), `engaging` (hooks/signposting), `catalogue` (the
  stereotypical hedged LLM register, on purpose). The label is recorded as
  `meta.slop_strategy` **and folded into `synth_key`** (now
  `hash(generator, strategy, context, target)`), so iterating a prompt
  regenerates rather than colliding on resume, and flavours stay distinguishable
  for ablation. A custom/author-specific prompt is injected via
  `--slop-system-file` (library `system=`) under any label — keeping a specific
  slop catalogue out of generic stylebot.
- `openrouter_generator(model=…)` reaches many upstream models off **one**
  `OPENROUTER_API_KEY` (OpenAI-compatible; tagged `openrouter/<model>` in
  `meta.generator`), making multi-source rotation a single-credential affair.
  CLI: `--openrouter-model <id>` (repeatable) on both `ai-style synth` and the
  blog's `train-targets` (whose default is now an OpenRouter rotation — see
  `livingthing.training_targets.OPENROUTER_MODELS` / `SLOP_STRATEGY`).
- **Work the loop, not a one-shot run:** generate a small batch per strategy into
  a scratch `--data-dir`, eyeball via `--report`/`--sample`, then score it with
  `ai-style eval --pairs <scratch>/pairs.jsonl --judge --by slop_strategy --report
  scores.html` (add `--detector-model _models/voice-clf` for the trained-detector
  P(slop) signal alongside the judge) to compare flavours by a number *and* eyeball
  the slop↔Dan pairs + rationales in
  the HTML report. Promote a chosen strategy into the real corpus only after it
  earns it. Because `synth_key`
  carries the strategy, "one run per strategy into the same dir" accumulates
  distinct, non-colliding pairs.

## Method (from the post)

1. Sample paragraphs of Dan's own prose → these are the **targets**.
2. For each, ask an LLM "rewrite this passage to be clearer and more polished"
   → the output is the **slop source**.
3. **Multi-source**: rotate across ≥2 generators (several OpenRouter models, the
   local base model) so the styler learns to undo AI-writing broadly, not one
   model's tics. Tag each pair with the generator in `meta`.
4. **Secondary set**: explicitly request worst-case patterns from the slop
   catalogue for long-tail mannerisms the paraphrase pipeline underrepresents.
5. **Context-fullness sweep** (optional, post §"Whose slop?"): generate some
   slop across one growing session (fresh paragraph each turn) and tag with how
   full the context was, to study/​weight late-stage slop.

## Outputs (must match Phase 1 schema)

Append to the same `$STYLEBOT_DATA_DIR/pairs.jsonl`:

- `messages[0].content == STYLE_SYSTEM` (verbatim).
- `messages[1]` = synthetic slop; `messages[2]` = Dan's real paragraph.
- Same paragraph-chunk shape as Phase 1.
- `meta` MUST additionally carry: `synthetic: true`, `generator:
  "claude-…"|"gpt-…"|"local-…"`, and (if swept) `context_fullness`.
- Reuse `meta.tags` for provenance (e.g. `["synthetic","paraphrase"]`).

## Done-criteria

- [x] **Built** (2026-06-27): `stylebot.synth.synthesize_pairs` library function
      over explicit paths + the `ai-style synth` CLI. Output gated through
      `validate_pairs_file` in `tests/test_synth.py` (empty result == pass).
- [x] Pairs from ≥2 distinct generators, distinguishable by `meta.generator`
      (round-robin rotation by default; `--per-generator` for every-target ×
      every-generator). Tested.
- [x] Idempotent + resumable: each pair carries `meta.synth_key`
      (`hash(generator, strategy, context, target)`); assignments already in the
      file are skipped, so re-running never duplicates and a crash resumes. Tested.
- [x] Slop *prompt* is a knob (`STRATEGIES` / `--slop-strategy` /
      `--slop-system-file`, recorded as `meta.slop_strategy`); multi-source via
      `openrouter_generator` / `--openrouter-model` off one key. Tested.
- [x] Documented entrypoint: `uv run ai-style synth` (thin wrapper) /
      `stylebot.synth.synthesize_pairs` (library). `--dry-run` vets selection
      with no API spend; `--report` / `--sample` eyeball targets.
- [ ] **Experimental generation loop (the active step).** Don't batch a one-shot
      paid run; iterate. Per strategy, generate a small batch into a *scratch*
      `--data-dir`, eyeball (`--report`/`--sample`), then score the slop with
      `ai-style eval` (judge + the trained `--detector-model`). Promote a strategy into
      the real corpus only once it earns it. The blog offers ~10k clean
      quality>6 targets (merged), so volume is there. Example experiment:

      ```sh
      # one strategy, into a throwaway dir, off the OpenRouter key:
      uv run ai-style synth --blog-root ~/Source/livingthing \
        --data-dir /tmp/slop-experiments --slop-strategy catalogue \
        --openrouter-model anthropic/claude-opus-4-8 \
        --openrouter-model openai/gpt-4o --limit 40 --report /tmp/exp.html
      ```

      Once a strategy (or mix) is chosen, the corpus run is
      `cd ~/Source/livingthing && uv run train-targets --limit N` (defaults to the
      OpenRouter rotation; `--dry-run`/`--report` first).

## Generation covariates & experiments (built 2026-06-29)

Slop generation is a **measured experiment**, not a fixed recipe: every synthetic
pair records the generation parameters that may shape the slop distribution, so they
can be faceted in eval and conditioned on at train time. (Prompted by Dan's review:
the first smoke's slop missed real Claude mannerisms — e.g. incoherent mixed
metaphors — and reasoning was wrongly hard-disabled.)

- **`meta.gen` covariate bundle** (synthetic pairs only; `meta` is an open dict so
  Phase-1 real pairs simply lack it — they're the falsy-`synthetic` stratum):
  `model, reasoning_effort, temperature, top_p, max_tokens, finish_reason,
  prompt_tokens, completion_tokens, prompt_id, prompt_version, prompt_label`. Carried
  onto score records (`eval._CARRIED_GEN`), so `ai-style eval --by <covariate>` and
  the report's `--facet-by` work for free.
- **Reasoning is a covariate** (`--reasoning-effort high|medium|low|off`, default
  **high** — real AI prose is often high-reasoning). `synth._reasoning_extra_body`
  maps it to OpenRouter's `reasoning` field per model family (effort enum vs
  `max_tokens` budget vs `enabled:false`); the *requested* effort is recorded
  regardless of what the provider honors, and `finish_reason`/`completion_tokens`
  expose a model that reasoned anyway. **Do not assume reasoning=off is "identical
  quality"** — that is exactly what the sweep below measures.
- **Sampling** (`--temperature`, `--top-p`) is now sent (previously the API default
  ran, uncontrolled) and recorded. Recorded-only — NOT in `synth_key`.
- **Versioned prompt library**: `STRATEGIES` values are `SlopStrategy(label, system,
  version)`; `prompt_id = sha256(system)[:12]` identifies any prompt (registry *or*
  custom `--slop-system-file`), folded into `synth_key`. The three generic flavours
  are unvalidated starting points — treat prompts as the primary experimental axis.

**Experiments to run** (each: small batch per arm into a scratch dir → `ai-style eval
--by <covariate>` → compare the slop→Dan Δ + Vale Δ; promote nothing until it earns it):

1. **Reasoning-effort sweep** — `high|medium|low|off` × the strategy set on a fixed
   target set; `--by reasoning_effort`. Question: does reasoning change the slop
   distribution, and *toward or away* from the real Claude output Dan cleans up?
2. **Prompt ablation** — the generic flavours + Dan's `_voices/slop_patterns.md`
   catalogue (injected via `--slop-system-file`) + new prompts that explicitly target
   under-represented Claude mannerisms (mixed metaphors, false-definiteness); `--by
   prompt_id`. The main axis.
3. **Distribution-match diagnostic (DEFERRED — its own phase).** Compare synthetic
   slop against the real-slop reference (the 254 Phase-1 pairs, ≥48 from
   `automation:2`; plus `automation:1/2` prose) via a feature/mannerism extractor,
   faceted by `meta.synthetic` (real = `synthetic` falsy). The eval scores passages
   individually today; it has no distributional comparison — this is the next eval
   capability (see [`eval-harness.md`](eval-harness.md)). Seam: `eval` aggregation
   over `meta.synthetic`, consuming the covariates above.

Also note (hygiene): some selected targets are low-signal; tighten selection/hygiene
or record a target-quality covariate if it proves to matter.

## Risks / notes

- Synthetic slop may not match the real Claude output distribution Dan hits in
  practice → keep mixing in real Phase 1 pairs; consider down-weighting
  synthetic at train time (`meta.weight`?). Experiment 3 above measures the gap.
- One known failure mode (post): learning the transform but *overcorrecting* —
  looks great on metrics, bad to humans. The eval harness must guard this.
