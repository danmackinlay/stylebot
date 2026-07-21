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
  `polish` (neutral baseline), `engaging` (hooks/signposting), `casual` (the
  friendly-technical-blog register), `measured` (the mild stereotypical-LLM
  register — texture described, stock phrases banned, variety instructed). A `catalogue` strategy (exaggerated
  stereotypical-LLM tics, requested outright) was removed 2026-07-07 — it
  produced cartoonish slop unlike the real drafting distribution; old pairs
  stay resolvable via their data-dir's prompts.jsonl. The label is recorded as
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
  blog's `dan-style synth` (whose default is now an OpenRouter rotation — see
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
5. **Context-fullness sweep** (✅ built 2026-07-06, post §"Whose slop?") — live
   multi-turn sessions with parallelism, `--session-turns K` on `ai-style` /
   `dan-style synth`. Design:
   - **Sessions are the unit of parallelism**: a session = one generator + an
     ordered slice of targets; each turn sees the real prior (passage → slop)
     exchanges — true self-conditioning. `synthesize_pairs` runs sessions
     concurrently on an asyncio event loop (`--max-workers`, default auto: 16
     when OpenRouter-only, 1 with a local preset in the mix). At the default
     `--session-turns 1` this degenerates to plain per-pair parallelism — the
     days→hours corpus-augmentation speedup with sessions off.
   - **Fill is measured, not engineered** (models grow context unevenly —
     different tokenizers, reply lengths, window sizes): the covariates are the
     response's own `prompt_tokens` plus `context_window` (fetched per model
     from the keyless OpenRouter models registry) and derived `window_fill`,
     all in `meta.gen` next to `session_id`/`session_turn`. Facet eval with
     `--facet-by session_turn`.
   - **Cost/overflow control**: a session ends at
     `min(--session-max-tokens (default 32k), 0.8 × window)` estimated prompt
     tokens or at `--session-turns`, whichever first. Input *tokens* grow
     ~quadratically with session depth (each stateless call re-sends the
     history; a 32k session bills ~25× the input tokens of the same pairs
     stateless). Whether *dollars* follow depends on the serving provider's
     cache-read discount — measured live 2026-07-06: qwen-via-Groq and
     sub-1024-token OpenAI prompts get `cached_tokens=0`, i.e. genuinely
     quadratic dollars; providers with prompt caching flatten it. Don't infer:
     `meta.gen` records the billing ground truth per pair (`cost` in credits +
     `cached_tokens`, via OpenRouter `usage.include`). Per-model budget
     overrides exist as a library hook (`Generator.session_budget` /
     `run_synth(session_budgets=…)`), unused by default.
   - **Keys/resume**: multi-turn keys fold `session_id:turn` into `synth_key`
     (turns coexist; a crashed session resumes by replaying recorded slop into
     history). Stateless keys carry NO session component, so corpus resume is
     stable as the blog grows. (Full keying semantics: next section.)

## Keying semantics — append-first (settled 2026-07-21)

`synth_key` is a **cell identity**, not a completion marker:
`hash(model, strategy, effort, prompt_id, context, target[, session])`. Skip
means "this exact cell already has a pair", never "this target is done". Any
change on a design axis mints new keys and *appends* alongside the old pairs —
variants coexist, covariates ride in `meta.gen`, the corpus is append-only.
Consequences, written down so we stop re-deriving them:

- **Changing a default re-keys flag-less runs** (e.g. effort high→off,
  2026-07-20). That is a *budget* decision, not data damage: old cells remain
  valid training data with their covariates recorded; a new run fills new
  cells. Decide spend, not "dedup loss".
- **Purging is a covariate filter, never key surgery.** A variate found to
  produce defective pairs (cf. the removed `catalogue` strategy) is dropped by
  filtering `meta.gen`/`meta.slop_strategy` over pairs.jsonl. Keys never
  encode good/bad; expected to be infrequent.
- **Determinism is content-anchored, not positional.** The guarantee: an
  unchanged (text, config) cell never regenerates; edited text regenerates —
  we *want* pairs for the current prose (stale-voice policy is selection-side,
  `MIN_DATE_MODIFIED`, not purge-side). The mechanism is one convention:
  **hash content, never position** — generator assignment hashes the target
  text, context dropout hashes the body, and any subsampling must do the same
  (a `chunk_index` in the hash breaks superset stability the moment a
  paragraph is inserted above it).
- **Two accepted weak spots:** (1) merge-mode packing — an edit early in a
  section can shift passage boundaries and re-key later chunks whose own text
  didn't change (costed as regeneration, not corruption); (2) session keys are
  snapshot-scoped — session chunking depends on the whole target list, so a
  multi-turn run resumes exactly only against the same target snapshot. Runs
  that must resume across blog drift use `session_turns=1`.
- **Known gap — replicates.** One cell = one sample; temperature means a
  second draw would be genuinely new data, but the key says done. When wanted,
  add an opt-in `replicate` tag folded into the key the way `session` is
  (empty default → existing keys untouched). `assign_seed` is NOT this — it
  *moves* arms between targets rather than resampling a cell.

### DONE (2026-07-21): session component dropped from keys; leftovers reflow

Landed together with the silent-drop fix from the same day's post-mortem (a
`--session-turns 128` run silently discarded 46% of planned work when the
32k token budget bound): keys are content-only (+ optional `--replicate`
label), turns a session can't run reflow into fresh sessions instead of
dying at the `break`, resume is a set difference over cells (the replay
machinery is deleted), and the exit summary reports budget-bound sessions,
reflow volume, and — as a tripwire — any turn neither written, skipped, nor
errored. `--session-max-tokens` is documented as THE depth control;
`--session-turns` is a mode switch (1 = stateless) plus per-session backstop.
Window position stays a live treatment: depth accumulates as a recorded
covariate on every session run, and a deliberate deep arm is
`--replicate <label> --session-max-tokens <depth>`. Original analysis below.

**Coverage mode (`--skip-covered`), the cross-epoch dedup:** cell-level keys
deliberately let config variants coexist — which means every re-key epoch
(the effort flip, the key narrowing) would regenerate already-covered
targets and double their training weight (~1,970 doubled targets after the
2026-07 epoch, an imbalance correlated with walk order). `--skip-covered`
is the corpus-building answer: skip any target whose text has >=1 pair
under ANY config (context-agnostic — the recorded `meta.context` prefix is
stripped), so a coverage run generates only the genuinely uncovered
remainder. Experiments wanting the cross product simply don't pass it. The
complementary knob — a per-target cap/weighting at training-set assembly —
is Phase-3 policy, useful once multiple deliberate variants per target
exist (`--per-generator`, `--replicate`).

#### The original queued rationale (2026-07-21, retained for the record)

`session_id` hashes the generator config **plus every turn's text in order**
(`_plan_sessions`), so ANY target-list drift — one new post, one edit, a
different `--limit` — re-chunks sessions and re-keys nearly every turn in the
arm. Cross-run, a multi-turn corpus build therefore never dedupes: the same
target regenerates under the same config, differing only in an unstable
session coordinate — correlated near-duplicate pairs, pseudo-replication.
This was the right trade while window position was a candidate *treatment*
(the fold is what let the same text be resampled at different fills); the
sweep retired that treatment (drift r = −0.05), leaving cost without benefit.

The change: session folding becomes opt-in (`key_by_session`, default off);
window fill stays a recorded covariate in `meta.gen`; deliberate resampling
goes through the explicit `replicate` tag above (stable because user-chosen).
Timing: land it in the same generation epoch as the effort high→off flip —
that flip already re-keyed every flag-less cell, so the narrowing costs zero
additional regeneration — but NOT while a multi-turn run is in flight (its
partial session-keyed pairs would stop matching on resume).
   - **Cache exploitation (2026-07-07)**: live sessions pin to the provider
     that served turn 1 (`--sticky-provider`, default on) — keeps that
     provider's prefix cache hot AND holds the serving stack constant so the
     fill covariate isn't confounded by provider hops; Anthropic models get a
     moving `cache_control` breakpoint on session history (`--prompt-cache`,
     default on; 0.1× cache reads after a 1.25× write). Note prompt caching is
     **unexploitable on stateless corpus runs** — the shared prefix (the slop
     system prompt) is under every provider's ~1024-token minimum. Savings are
     verified, not assumed: sum the recorded `meta.gen` `cost`/`cached_tokens`.

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
        --data-dir /tmp/slop-experiments --slop-strategy casual \
        --openrouter-model anthropic/claude-opus-4-8 \
        --openrouter-model openai/gpt-4o --limit 40 --report /tmp/exp.html
      ```

      Once a strategy (or mix) is chosen, the corpus run is
      `cd ~/Source/livingthing && uv run dan-style synth --limit N` (defaults to the
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
  **off** since 2026-07-20 — the sweep ran: 1241 pairs, six models, off vs medium,
  detector gap flat for ~3x the wall clock, so the cheap end won and effort is now
  swept *up* on suspicion, not down). `synth._reasoning_extra_body` maps it to
  OpenRouter's `reasoning` field per model family (effort enum vs `max_tokens`
  budget vs `enabled:false`); the *requested* effort is recorded regardless of what
  the provider honors, and `finish_reason`/`completion_tokens` expose a model that
  reasoned anyway (qwen3-32b emitted ~700 reasoning tokens at `off`).
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
