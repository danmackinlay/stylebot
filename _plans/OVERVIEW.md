# stylebot — development plan

The durable source of truth for this project. Code describes *how* a thing
works and is the better description of anything already built; these files hold
what the code cannot say for itself — open work, rejected alternatives, and
measurements. Edit this when decisions change; it should always reflect current
reality, not the original aspiration.

**What lives here now** (the per-phase design plans were retired 2026-08-10,
once their work shipped and the code became the better description):

| File | Holds |
|---|---|
| this file | goal, process contract, interfaces, data contract, status, open slices |
| [`decisions.md`](decisions.md) | why each non-obvious choice landed, what we got wrong, what we rejected |
| [`findings.md`](findings.md) | every measured result — not re-derivable from git |
| [`eval-harness.md`](eval-harness.md) | the four signals, the detector decision, the split contract |
| [`v2-longer-rewrites.md`](v2-longer-rewrites.md) | the scale curriculum — mechanism shipped, generation not yet run |
| [`synth-flag-audit.md`](synth-flag-audit.md) | the `synth` option surface, as-audited evidence + open retirement proposals |
| [`vscode-marker.md`](vscode-marker.md) | the editor marker design, thresholds, honesty caveats |
| [`anchor-integrity.md`](anchor-integrity.md) | the anchor guard/gate/audit — shipped (D32); the as-audited case is kept as the argument for it |
| [`masking-placement.md`](masking-placement.md) | *where* masking goes (D33) — decided, not yet built |
| [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md) | the mask *encoding* — the open half of the masking work |

Narrative rationale (the "why") lives in the blog post
[*Fine-tuning danbot*](https://danmackinlay.name/notebook/fine_tuning_danbot.html). This file is
the operational counterpart: phase map, contracts, status.

## The goal

Train a small LoRA adapter on an open-weights model that rewrites AI-flavoured
draft prose into Dan's voice. Treat it as a *prose styler*, not a chatbot.

**Goals:** less onerous cleanup of AI edits; more comprehensible prose.
**Non-goals:** full Dan impersonation; evading slop detectors.

## How we work (process contract)

This is a **data/ML project**, not just software. Provenance, reproducibility,
and the contracts *between* stages matter more than internal abstraction. So:

1. **Plans live here, in git — until the code outgrows them.** While work is
   open, a plan states explicit **inputs / outputs / done-criteria** so a
   subagent can pick it up from the file alone. Once it ships, the plan is
   retired: the code documents itself, and what survives is the decision
   (→ `decisions.md`) and the measurement (→ `findings.md`).
2. **PRs are per phase-slice, and earn their place with a check.** A slice
   becomes a PR when it ends in something verifiable — a passing test or an
   eval number. Plan edits just commit to the branch; they don't need a PR.
3. **Phases talk through file schemas, not shared code.** Every phase's I/O is
   a documented on-disk format (below). That's what lets phases be built in
   parallel against a few fixture rows, and farmed out to subagents.
4. **The corpus and secrets never enter this public repo.** The corpus is
   gitignored here; its canonical home is the livingthing repo, where it is
   committed (`_training_pairs/pairs.jsonl`, since 2026-07, for portability —
   see `_training_pairs/README.md`). Keys arrive as environment variables —
   direnv in practice — from a file that is never committed.
5. **Verify with a number, never with vibes.** "Does the styler work" is
   exactly where an ML project rots. The eval harness is therefore an *early*
   parallel track, not a final step.

## Interfaces — library-first, explicit paths

stylebot is a **tool the blog depends on**, not a repo with a fixed home for
your data. The dependency arrow points one way (blog → stylebot) and stays
acyclic: stylebot never hardcodes a blog location, and the blog can run the
whole pipeline from its own build.

- **Library first, CLI second.** Each phase is a module with a typed function
  taking explicit paths/params (`stylebot.train.run_training(...)`). The CLI is
  a thin `click` wrapper that parses flags and calls it. The blog build can
  `import` the function directly — no subprocess/parse tax — *or* shell out to
  the CLI. Build the function; the CLI is a shell over it.
- **One entry point, subcommands** — `ai-style log | synth | split | train |
  eval` — not a scatter of loose scripts. Shared option parsing, one `--help`
  tree; loose scripts drift in flag conventions. Phase 1's logger is exempt from
  nothing: it shipped as a standalone `ai-style-log` console script until
  2026-07-20 (history, not design) and is now `ai-style log` like everything
  else. There is exactly one console script.
- **Two distinct path inputs, never collapsed:**
  - `--blog-root` — a *read* source (authored `.qmd` prose, sample paragraphs).
  - `--data-dir` — *read/write* state (the `pairs.jsonl` corpus, manifests).
  Phase 2 reads blog-root, writes data-dir; train reads data-dir only.
- **Option precedence: `--flag` > `$STYLEBOT_DATA_DIR` (env) > default.**
  Resolved in one place, `stylebot.config`. The env var is a convenience for
  interactive use, not the primary mechanism.
- **Defaults where a mistake is cheap; required flags where it isn't.**
  `ai-style log` keeps a cwd-relative default `--data-dir` (run dozens of times
  a day against the focused file — zero friction). The expensive/stateful
  commands (`train`, `split`) take **no default `--data-dir`**: an explicit
  path is the reproducibility record, and you never want to silently train on
  the wrong or empty corpus.
- **Caller decides *which* files; stylebot decides *what to do* with them.**
  New phases take file lists / roots as arguments rather than rediscovering the
  blog's conventions.
- **Selection is a user-supplied policy, not a built-in gate.** *How* a user
  picks and orders their training/reference prose — filter, sort, sample — is
  theirs to define. Because stylebot is a library, supporting that needs no
  plugin system or config DSL: the user just passes a function. Two supported
  levels, both first-class:
  1. **Pre-selected list.** The caller does its own discover→filter→sort→sample
     and hands stylebot the resulting file list. Zero stylebot machinery; works
     with any user logic.
  2. **Injected policy.** For callers who want stylebot to do the walking, the
     phase entry points accept callables — `selector: Callable[[dict], bool]`
     and a `sort_key` / `sampler` — defaulting to bundled examples. You swap the
     default by passing your own; ordering is in scope, not just inclusion.
  `stylebot.lib.is_human_authored` is a **shipped example** of such a selector,
  not a mandated path. The contract is "bring your own callable"; we ship one
  that happens to encode Dan's `automation: 0` convention.

### Where the blog/generic boundary actually falls

Most of what we imported is *generic*, not Dan-specific. The boundary:

- **Generic core (keep, treat as reusable):** frontmatter + markdown loading
  (`read_w_frontmatter_text`) and content-file discovery. Quarto's
  `.qmd` is just YAML-frontmatter markdown — the same code serves Hugo/Jekyll/
  any frontmatter blog. The only "blog-ness" in `is_valid_qmd_file` /
  `gather_qmd_files` is the hardcoded `.qmd` extension and build-dir skip-list;
  those are parameters, not Dan-knowledge. Generalise by making the glob/
  extension an argument when a phase needs it.
- **The bundled selector example — `automation`.** It is *acceptable* to ship
  a little blog-specific code here as a worked example. `automation` is a
  frontmatter level recording how much AI touched a post; `automation: 0` means
  pure-human, the prose we want as clean training targets / reference.

  ```python
  def is_human_authored(meta, *, field="automation", max_level=0) -> bool: ...
  ```

  **Built** in `stylebot.lib.is_human_authored` (conservative: missing/
  unparseable → excluded), tested in `tests/test_selection.py`. Its `field` /
  `max_level` knobs are the *shallow* retarget; the *real* extension story is
  the bullet above — a user passes their own `selector` / `sort_key`. This
  function is the default that ships, not a gate stylebot imposes. Phases must
  therefore never call it directly on a hardcoded path: they accept a selector
  argument and use this only as the default value.
- **Blog-build cruft — removed from stylebot (2026-06-29).** `migrate_ai_dates`
  (custom `date-ai-*` keys) and `is_auxiliary_post` / `AUXILIARY_TYPES`
  (`digest`/`about` post types) were copied in but never used here; they now live
  only in the blog. Don't reintroduce them — auxiliary/AI-touched filtering is
  the caller's `selector` policy, not stylebot's job.

## The shared data contract (the seams)

Phases are decoupled by these formats. Change one → update the phase file and
notify downstream.

| Artifact | Producer | Consumer | Format |
|----------|----------|----------|--------|
| `pairs.jsonl` | Phase 1 (real), Phase 2 (synthetic) | Phase 3 | chat-completion JSONL, `{messages:[system,user,assistant], meta:{...}}` — schema enforced by `stylebot.pairs.validate_pair_record`; mechanics in `_training_pairs/README.md` |
| trained adapter | Phase 3 | Phase 4, eval | LoRA weights (Tinker-trained → HF export; Fireworks-hosted or local MLX) |
| candidate text | Phase 4 | eval harness | plain prose in, plain prose out |
| `scores.jsonl` | eval harness | humans, Phase 3/4 | id-keyed (`synth_key`/`capture_id:idx`) per-pair `{meta, scores:{field:{vale,judge,detector}}}`; joins back to `pairs.jsonl` |

The crucial deliberate choice: **Phase 1 and Phase 2 emit the *same*
`pairs.jsonl` schema**, chunked the same way, so training data from real edits
and synthetic paraphrase is shape-compatible and mixable with a weight column.

## Phase map & status

Everything below is **built**; the code is the reference. Where a phase still
has a durable document, it is linked.

| Phase | Where it lives in code | Status |
|-------|------------------------|--------|
| 0 · Scaffolding | pyproject / `stylebot.config` | ✅ done |
| 1 · Pair capture | `stylebot.capture`, `ai-style log`; mechanics in `_training_pairs/README.md` | ✅ shipped, daily use; heading context on both producers (immediate depth) |
| 2 · Synthetic pairs | `stylebot.synth`, `stylebot.targets`, `ai-style synth` | ✅ shipped; corpus settled 2026-07-22 (~4k pairs) behind the identity + inflation gates (`--max-transform-sim` / `--max-length-ratio`) |
| 3 · LoRA training | `stylebot.train`, `ai-style train` | ✅ shipped — two paid runs (manifests in the blog's `_training_pairs/runs/`); run 2 is the serving default |
| E · Eval harness | `stylebot.eval`, `ai-style eval` — [`eval-harness.md`](eval-harness.md) | ✅ built; all four signals live, detector = the trained voice classifier |
| 4 · Inference CLI | `stylebot.infer`, `ai-style run` / `dan-style run` | ✅ built; Tinker-sampling backend v1, MLX exit verified, Fireworks open |
| — · VS Code marker | `stylebot.serve`, `vscode-voice-marker/` — [`vscode-marker.md`](vscode-marker.md) | 🔧 in progress — sidecar + extension marking paragraphs with `P(slop)`, plus interactive best-of rewrite |

> **Codebase hygiene (2026-06-29).** A QA declutter pass removed the dead
> blog-build code bulk-copied into stylebot (`migrate_ai_dates`,
> `is_auxiliary_post`, the `SUMMARY_*`/`QUALITY_*` prompts, unused frontmatter
> helpers) and the redundant direct-Anthropic generator + `anthropic` dep (hosted
> models go through OpenRouter now). The mechanism is lean — don't redo it; ship
> functionality.

**Where things stand (2026-08-11):** the whole pipeline runs end-to-end —
capture → synth (gated) → train (Tinker LoRA, manifests committed blog-side)
→ `dan-style run` (guarded inference) → eval (detector + judge + Vale).
Segmentation is now one implementation (`stylebot.markdown`) with training and
inference held to the same protection contract; the corpus is 4595 pairs after
the A5 prune, and neither trained run has been re-run against it. Operations
live in the blog's `_training_pairs/RUNBOOK.md`; durable decisions in
[`decisions.md`](decisions.md), measurements in [`findings.md`](findings.md).

**Open slices** (each self-contained):

- **v2 scale curriculum** — **the generation ran (2026-08)**; the scale-mixed
  training run has not. The blog corpus carries 4,532 pairs stamped
  `chunk_mode: cross-heading` at 1500/3000/4500 (2,629 / 1,096 / 807) beside
  3,335 unstamped legacy pairs, so the 2026-08-11 "no labelled arm" block is
  cleared. Full state: [`v2-longer-rewrites.md`](v2-longer-rewrites.md); the
  adapter is the blog's run 4 (`_plans/styler-run4-preflight-2026-08.md` there).
  **Read that doc's label caveat before faceting**: `merge_max_chars` records
  which batch first produced a target's text, not its design cell, so a clean
  contrast needs the budget-unique targets (2,366 / 1,077 / 761) or a
  length-stratified eval of the adapter itself.
- **Masking** ([stylebot#1](https://github.com/danmackinlay/stylebot/issues/1)
  step 3): the guard, write-gate and audit log shipped 2026-08-12 (D32), and the
  gate promptly answered the question it was built to gate on — at v2 chunk
  scale, unmasked anchor survival is 0–22% by arm and redraws recover 2%
  (`findings.md`). **Masking is now justified, and the placement decision is
  made**: mask at synthesis, store mostly-masked with a `CONTEXT_DROPOUT`-style
  unmasked fraction, mask at inference —
  [`masking-placement.md`](masking-placement.md), D33; the *encoding* stays
  [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md). One free
  prerequisite before building: re-fit the voice classifier on masked text
  (`train-clf`) — a logistic head on a frozen `StyleDistance` encoder is the one
  consumer consistency alone cannot fix. Folding the blog-side masker into
  stylebot is now load-bearing rather than tidy, since `ai-style run` ships no
  masker and a mask-first corpus would otherwise yield a blog-only adapter.
- **Fireworks serving**: PEFT upload + scale-to-zero, retry on
  `503 DEPLOYMENT_SCALING_UP`; acceptance is the same chat-template parity test
  the MLX exit passed (D8 / `findings.md`).
- ~~**Blockquote integrity**~~ — **closed 2026-08-11** by the second candidate
  fix: blockquotes are protected the way fences are, so `infer._segment` never
  hands one to the styler and there is no reflow to lose the `>` prefix
  (`decisions.md` D27, protection ground 1).
- **v3 data**: deliberate moderate-compression pairs (Dan compresses 1.5–2× in
  real edits) now that generator-accident inflation is gated at 3×.
- **Synthetic↔real distribution match**: the fifth eval capability — a
  feature/mannerism extractor over the synthetic vs real-slop strata. Spec in
  [`eval-harness.md`](eval-harness.md) §"Planned".
- ~~**`synth` option-surface retirements**~~ — **closed 2026-08-13.** `--sort`,
  `--report-max-rows`, `--all` and `--tag` retired (49 options); the substrate
  cluster and `--capture-reasoning` reviewed and kept, `--assign-seed` deferred
  as a `synth_key` input ([`synth-flag-audit.md`](synth-flag-audit.md)).
- ~~**`ai-style log` never adopted the option-bundle idiom**~~ — **closed
  2026-08-13.** `save`/`pair` share one `CaptureOpts` bundle; `list` and the
  open-session reminder share `_snapshot_dirs` / `_echo_snapshots`.
- ~~**`meta.gen`'s key list is a third copy**~~ — **closed 2026-08-13.** Split
  by writer (`GEN_BASE_KEYS` / `GEN_EXTRA_KEYS` / `GEN_REPLICATE_KEYS` /
  `GEN_SESSION_KEYS`) so each group is assertable as an equality against a real
  record from the writer that produces it, in both directions.
- **`openai` as an `[api]` extra**: every import of it is lazy, so a
  `classify`/`serve`/keyless-`eval` install carries the SDK for nothing. Written
  up as [`decisions.md`](decisions.md) P1 — undecided, and it would cost
  `ai-style synth` its plain-install path.
- **Phase 7 preference/DPO** on the same Tinker platform (decisions #D5), with
  the eval-harness Goodhart guards. Preference data already accrues from
  ordinary `--best-of` use.

## Subagent-friendliness checklist

Before farming out an open slice, confirm it has:
- [ ] explicit inputs (which files/env/keys) and outputs (which files/schema);
- [ ] done-criteria stated as a number or a test, not "looks good";
- [ ] a runnable entrypoint (an `ai-style` subcommand or script);
- [ ] enough fixture data to build against without other phases existing.

## Open questions

- **Which generation covariates matter** — reasoning effort, prompt, model and
  sampling are recorded per pair (`meta.gen`) and facetable. Effort is settled
  (`findings.md`); the **prompt ablation is the main untried axis**, including
  prompts that explicitly target under-represented mannerisms (mixed metaphors,
  false definiteness).
- Synthetic↔real slop **distribution match** — does synthetic slop resemble the
  real output Dan cleans up? Eyeballing says it misses some mannerisms; no
  distributional comparison exists yet. See `eval-harness.md` §"Planned".
- Single-source vs multi-source slop ablation — worth publishing? (post §"Whose
  slop?"; measurable now via `meta.generator`/`meta.gen`.)
- Down-weighting synthetic vs real pairs — what weight? `meta.weight` is a hook
  (`pair_weight`/`per_target` callables), uniform so far.
- Per-target multiplicity — re-key epochs and deliberate variants can stack
  several pairs on one target; assembly should expose a per-target cap so
  duplicated targets don't silently up-weight their passages.
- Base-model headroom — `Qwen/Qwen3.5-9B` today; step-up candidates are
  Qwen3.6-27B (dense) or Qwen3.6-35B-A3B (MoE, the Mac-friendliest shape).
- Detector — **settled (2026-06-30):** the trained Dan-voice classifier
  (StyleDistance backbone) is the signal, not Pangram/general AI-detectors.
  Pangram survives only as an optional independent cross-check.
