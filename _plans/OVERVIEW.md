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
| [`decisions-archive.md`](decisions-archive.md) | D1–D50 and P1, moved under P3; `decisions.md` keeps their index rows |
| [`eval-harness.md`](retired/eval-harness.md) | retired 2026-09-04 (P7); the four-signal harness — what shipped (D7, D17, D39–D42), what it refuted, the unbuilt fifth signal |
| [`v2-longer-rewrites.md`](retired/v2-longer-rewrites.md) | retired 2026-09-04 (P7); the scale curriculum — generated 2026-08, trained as the blog's run 4; step 5 and the weighting policy stayed open |
| [`synth-flag-audit.md`](retired/synth-flag-audit.md) | retired 2026-09-04 (P7); the `synth` option audit — 54 → 49 options (D21, D83), nothing open |
| [`vscode-marker.md`](retired/vscode-marker.md) | retired 2026-09-04 (P7); the editor marker — `ai-style serve` + extension shipped, alternative B parked (D44) |
| [`anchor-integrity.md`](retired/anchor-integrity.md) | retired 2026-09-04 (P7); the anchor guard/gate/audit (D32) — its open step 3 is `masking-placement.md` |
| [`masking-placement.md`](masking-placement.md) | *where* masking goes (D33) — decided, not yet built |
| [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md) | the mask *encoding* — the open half of the masking work |
| [`argued-slop-arm.md`](argued-slop-arm.md) | a slop strategy for concessive/justificatory structure — done 2026-09-07: stage 0 (`stylebot.features`, D88), stage 1 auditioned and promoted as a diversity arm, stage 2 probed and retired (D92); the real regime is not synthesisable this way |
| [`transcript-mining.md`](transcript-mining.md) | real pairs replayed from Claude Code session transcripts (with the objections Claude was answering) — built 2026-09-07 (`stylebot.transcripts`, `ai-style log mine`, D89); `--apply` and the `novel_after` weight policy are Dan's |

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
   committed — `_training_pairs/draws/` since 2026-09-05 (D84), before that
   `pairs/` shards from 2026-09-03 (D50) and a single `pairs.jsonl` from
   2026-07; views are built there, never committed; see `_training_pairs/README.md`.
   Keys arrive as environment variables —
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
- **One entry point, subcommands** — `ai-style log | synth | make-splits |
  check-splits | extend-eval | train | train-clf | promote | run | serve | eval |
  features | status | validate-pairs | export-table | merge-adapter` — not a scatter of
  loose scripts. Shared option parsing, one `--help` tree; loose scripts drift in
  flag conventions. Phase 1's logger is exempt from nothing: it shipped as a
  standalone `ai-style-log` console script until 2026-07-20 (history, not
  design) and is now `ai-style log` like everything else. There is exactly one
  console script; since 2026-09-03 no `scripts/` directory (D38); and since
  2026-09-05 no migration group either — nothing rewrites the corpus (D84).
- **Two distinct path inputs, never collapsed:**
  - `--blog-root` — a *read* source (authored `.qmd` prose, sample paragraphs).
  - `--data-dir` — *read/write* state (the corpus, manifests). The corpus is
    `<data-dir>/draws/`, which every reader takes as the view built under its
    policy at `<data-dir>/views/pairs-<policy_sha>/` (`stylebot.draws`, D84);
    `stylebot.config.pairs_path` still resolves a legacy `pairs/` or
    `pairs.jsonl`, which reads as accepted draws (D45, D50).
  Phase 2 reads blog-root, writes data-dir; train reads data-dir only.
- **Option precedence: `--flag` > `$STYLEBOT_DATA_DIR` (env) > default.**
  Resolved in one place, `stylebot.config`. The env var is a convenience for
  interactive use, not the primary mechanism.
- **Defaults where a mistake is cheap; required flags where it isn't.**
  `ai-style log` keeps a cwd-relative default `--data-dir` (run dozens of times
  a day against the focused file — zero friction). The expensive/stateful
  commands (`train`, `make-splits`) take **no default `--data-dir`**: an explicit
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
     over frontmatter, defaulting to a bundled example, and at assembly
     `per_passage` (which pairs covering one stretch of prose to keep) and
     `pair_weight` (record → the row's loss scale, D90). You swap a
     default by passing your own. **Ordering is *not* in scope**: there is no
     `sort_key` and no `sampler`. `--sort` retired in 0.3.1 because arms are
     assigned by content hash, so target order cannot change what the corpus
     holds; the library-only `ChunkPolicy.sort_name` / `sort_key()` twin
     survived it for a caller who never appeared, and was deleted 2026-09-03
     ([`decisions.md`](decisions.md) D43).
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
  unparseable → excluded), tested in `tests/test_selection.py`. Since D85 its
  `field` / `max_level` and the walk's glob are one `SourcePolicy` object:
  `DEFAULT_SOURCE` renders Dan's literals and the blog states its own `SOURCE`.
  The *real* extension story is the bullet above — a user passes their own
  `selector` (and, at assembly, `per_passage` / `pair_weight`). This function
  is the default that ships, not a gate stylebot imposes. Phases must
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
| `captures/<id>/` | Phase 1 (a save) | `log recut`, `docscale`, `status` | the real edit, verbatim: `before.md`, `after.md`, `meta.json`; git-versioned and **edited in place** — the editable truth of a real pair, re-landed by `ai-style log recut` (D96) |
| draws (`draws/captures.jsonl`, `draws/synth-<YYYY-MM>.jsonl`) | Phase 1 (human saves), Phase 2 (generator calls) | `stylebot.draws.build_view` | append-only JSONL, one record per generator call or save, accepted or not: `draw_id`, `source`, `blog_sha`, `span`, `outcome` (`accepted` / `gate:<name>` / `error:<class>`), the raw output, `supersedes` for a session's interim saves and for a recut's earlier landing (`meta.archive_id`, D96); committed (D84) |
| the view (`views/pairs-<policy_sha>/`) | `build_view` under a `ViewPolicy` | Phase 3, eval, `status` | chat-completion JSONL, `{messages:[system,user,assistant], meta:{...}}`, `pairs/`-shaped shards plus a `VIEW.json` naming `policy_sha` and `draws_sha256`; schema, types and `record_id` uniqueness enforced by `stylebot.pairs.validate_pairs_file`; built on first use, never committed; a manifest pins both shas (D84); mechanics in `_training_pairs/README.md` |
| trained adapter | Phase 3 | Phase 4, eval | LoRA weights (Tinker-trained → HF export; Fireworks-hosted or local MLX) |
| candidate text | Phase 4 | eval harness | plain prose in, plain prose out |
| `scores.jsonl` | eval harness | humans, Phase 3/4 | one appended row per `(record_id, field, signal)` — `{id, meta, field, signal, text_chars, value}`, the detector identity riding on its own rows (D39); `scoring.load_scores` folds them (and any pre-D39 whole-record file) back to `{meta, scores:{field:{vale,judge,detector}}}`; beside the corpus, joins back to it |

The crucial deliberate choice: **Phase 1 and Phase 2 emit the *same* record
schema**, chunked the same way, so training data from real edits and synthetic
paraphrase is shape-compatible and mixable with a weight column. Sharding is
below that line: `shard_of` routes by record content, so which file a record
sits in is never part of its meaning — and so is the view: a prune, re-cut or
restamp is a `ViewPolicy` change plus a rebuild, never an edit to a draw (D84);
a real capture is corrected in its archive and re-landed (D96).

## Phase map & status

Everything below is **built**; the code is the reference. Where a phase still
has a durable document, it is linked.

| Phase | Where it lives in code | Status |
|-------|------------------------|--------|
| 0 · Scaffolding | pyproject / `stylebot.config` | ✅ done |
| 1 · Pair capture | `stylebot.capture`, `ai-style log`; a save archives to `captures/` and appends draws (`stylebot.draws`, D84); `log recut` re-lands an edited archive (D96); mechanics in `_training_pairs/README.md` | ✅ shipped, daily use; heading context on both producers (immediate depth) |
| 2 · Synthetic pairs | `stylebot.synth`, `stylebot.targets`, `ai-style synth` | ✅ shipped, behind the identity + inflation gates (`--max-transform-sim` / `--max-length-ratio`) and the anchor gate (D32) |
| 3 · LoRA training | `stylebot.train`, `ai-style train --draws`; `ai-style promote` | ✅ shipped — paid runs, with `per_passage` / `pair_weight` hooks and a per-passage histogram in the manifest (D43); the manifest pins `data.policy_sha` / `data.draws_sha256`, and a run serves only once promoted (D74, D84). **Which runs exist and which one serves: `dan-style status`**, never this table |
| E · Eval harness | `stylebot.eval`, `ai-style eval` — [`eval-harness.md`](retired/eval-harness.md) | ✅ built; all four signals live, detector = the trained voice classifier, verdicts carrying the detector identity that produced them and clustered by post (D39/D40) |
| 4 · Inference CLI | `stylebot.infer`, `ai-style run` / `dan-style run` | ✅ built; the backend surface is `stylebot.backends` (D35); Tinker-sampling v1, MLX exit verified, Fireworks open |
| — · VS Code marker | `stylebot.serve`, `vscode-voice-marker/` — [`vscode-marker.md`](retired/vscode-marker.md) | ✅ built — sidecar + extension marking paragraphs with `P(slop)` and offering best-of rewrites; the extension segments through the sidecar's `segment` op rather than porting the segmenter (D44) |
| — · Pipeline state | `stylebot.status`, `ai-style status` / `dan-style status` | ✅ built 2026-09-03 — corpus composition, splits health, promoted run, detector sha, and the cross-checks between them (D52) |

> **Codebase hygiene (2026-06-29).** A QA declutter pass removed the dead
> blog-build code bulk-copied into stylebot (`migrate_ai_dates`,
> `is_auxiliary_post`, the `SUMMARY_*`/`QUALITY_*` prompts, unused frontmatter
> helpers) and the redundant direct-Anthropic generator + `anthropic` dep (hosted
> models go through OpenRouter now). The mechanism is lean — don't redo it; ship
> functionality.

**Where things stand (2026-09-03):** the whole pipeline runs end-to-end —
capture → synth (gated) → train (Tinker LoRA, manifests committed blog-side)
→ `dan-style run` (guarded inference) → eval (detector + judge + Vale).
Segmentation is one implementation (`stylebot.markdown`), training and inference
held to the same protection contract, and the editor asks the sidecar for it
(D44). The 2026-09-03 seam pass moved the backend surface, the migrations, the
log session store and the corpus-stability floor to whichever side owns them
(D35–D38, D45–D50). On 2026-09-05 the corpus became append-only draws plus
built views (D84): the migrations, the corpus lock and every rewrite path went.

**Counts and run numbers are deliberately not written here** — they change under
git's feet in another repo, and every number this paragraph used to carry was
wrong within three weeks. `dan-style status` prints the live corpus composition,
eval stratum, promoted run and detector sha (D52). Operations live in the blog's
`_training_pairs/RUNBOOK.md`; durable decisions in
[`decisions.md`](decisions.md), *dated* measurements in
[`findings.md`](findings.md) — where a number like "the 4,595-pair corpus" is a
record of what was measured then, not a claim about now.

**Open slices** (each self-contained):

- ~~**v2 scale curriculum**~~ — **generated 2026-08, trained as the blog's run
  4** (`_plans/styler-run4-preflight-2026-08.md` there; the per-budget counts
  are [`findings.md`](findings.md) "v2 post-gate distributions" and the live
  composition is `dan-style status` (D52), not here). What is *still* open is
  reading it:
  **`merge_max_chars` records which batch first produced a target's text, not
  its design cell**, so a clean scale contrast needs the budget-unique targets
  or a length-stratified eval of the adapter itself. D43's finding is the same
  fact from the training side — the budget sweep is why one passage carries
  ~4 pairs — and it means a scale facet and a multiplicity cap are the same
  knob seen twice.
- **Masking** ([stylebot#1](https://github.com/danmackinlay/stylebot/issues/1)
  step 3): the guard, write-gate and audit log shipped 2026-08-12 (D32), and the
  gate promptly answered the question it was built to gate on — at v2 chunk
  scale, unmasked anchor survival is 0–22% by arm and redraws recover 2%
  (`findings.md`). **Masking is now justified, and the placement decision is
  made**: mask at synthesis, keep the corpus canonical and unmasked on disk,
  and mask as a deterministic transform at each consumer — training assembly
  for a `1 − p` fraction (the `CONTEXT_DROPOUT` idiom), inference, and the
  classifier independently —
  [`masking-placement.md`](masking-placement.md), D33; the *encoding* stays
  [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md). One free
  prerequisite before building: re-fit the voice classifier on masked text
  (`train-clf`) — a logistic head on a frozen `StyleDistance` encoder is the one
  consumer consistency alone cannot fix. Folding the blog-side masker into
  stylebot is now load-bearing rather than tidy, since `ai-style run` ships no
  masker and a mask-first corpus would otherwise yield a blog-only adapter.
- **Coherence context** ([`coherence-context.md`](coherence-context.md),
  opened 2026-09-10): two-sided, mixed-state neighbour conditioning, measured
  by the fit judge (D97) and a priced semantic-fidelity signal. Resurrects step
  5 of the retired v2 plan and the 2026-08-21 work found unmerged.
- **Capture UX** ([`capture-ux.md`](capture-ux.md), opened 2026-09-10): the
  correction path by source, not hash — per-post listing and recut, a
  jump-to-archive editor command, and a per-post replay diff.
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
  feature/mannerism extractor over the synthetic vs real-slop strata. **Its
  minimal first cut shipped 2026-09-07** as `stylebot.features` / `ai-style
  features` (syntactic proxies pooled per stratum plus a post-grouped
  separability AUC; D88, baseline in [`findings.md`](findings.md)). Still
  open: the mannerism families (`_voices/slop_patterns.md`, hedges,
  signposting, mixed metaphors) the 2026-06-28 spec named — the stage-0
  section of [`argued-slop-arm.md`](argued-slop-arm.md) holds that spec.
- **`argued` slop arm** ([`argued-slop-arm.md`](argued-slop-arm.md)): real
  captures carry more subordinate and justificatory structure than any synth
  strategy on the same posts, and reasoning effort does not move it. **Stage
  1 shipped 2026-09-07**: `STRATEGIES["argued"]` (version 2) and two 120-pair
  scratch auditions scored against the file's four criteria
  ([`findings.md`](findings.md) "The `argued` audition"): a prompt elicits
  subordination and concession but not the justificatory connective, and
  stays separable (AUC 0.83). `argued` v2 is promoted as a diversity arm.
  **Stage 2 probed the same day (D92)**: a brief from a real objection lands
  the connective rate and is no closer on the style backbone, so it retires
  library-only; the real stratum grows through transcript mining instead.
- **Transcript mining** ([`transcript-mining.md`](transcript-mining.md)):
  **built 2026-09-07** (D89) — `ai-style log mine` replays each session onto
  the file as Claude saw it and diffs against Dan's next commit through
  `capture_pairs`, untouched paragraphs reverted, dry-run by default. The
  dry run over the real logs yields far fewer pairs than the 2026-09-03
  sizing estimated, because the logs are lossy where the tree was dirty at
  session start (counts in the blog's private sizing note). `--apply` ran
  and the `NOVEL_AFTER_MAX` clause landed the same day; the blog's
  `claude-left` hook fixes the base for every future session. What is
  open is the file's "Continuation" section: chaining back-to-back sessions,
  partial reads as checkpoints (measure first), the `novel_after` view stamp
  on the old captures, and `argued` stage 2 from the objection sidecar.
- ~~**`synth` option-surface retirements**~~ — **closed 2026-08-13.** `--sort`,
  `--report-max-rows`, `--all` and `--tag` retired (49 options); the substrate
  cluster and `--capture-reasoning` reviewed and kept, `--assign-seed` deferred
  as a `synth_key` input ([`synth-flag-audit.md`](retired/synth-flag-audit.md)).
  2026-09-04 (X3): the eight design knobs — `--sample-salt`, `--replicate`,
  `--assign-seed`, `--per-generator`, `--provider-sort`, `--sticky-provider`,
  `--prompt-cache`, `--capture-reasoning` — became library-only bundle fields
  at their old values (41 options; D83).
- ~~**`ai-style log` never adopted the option-bundle idiom**~~ — **closed
  2026-08-13.** `save`/`pair` share one `CaptureOpts` bundle; `list` and the
  open-session reminder share `_snapshot_dirs` / `_echo_snapshots`.
- ~~**`meta.gen`'s key list is a third copy**~~ — **closed 2026-08-13.** Split
  by writer (`GEN_BASE_KEYS` / `GEN_EXTRA_KEYS` / `GEN_REPLICATE_KEYS` /
  `GEN_SESSION_KEYS`) so each group is assertable as an equality against a real
  record from the writer that produces it, in both directions.
- ~~**`openai` as an `[api]` extra**~~ — **rejected 2026-09-03**
  ([`decisions.md`](decisions.md) P1). Its facts held (all 30 modules import
  with the SDK blocked), but the balance moved: after W3 the only consumer
  declares `openai` itself, so the extra would save that install nothing, while
  `uv add stylebot` would start giving you a `synth` subcommand that cannot run.
  Re-evaluate on a PyPI release, a second consumer, or a non-OpenAI generator
  backend.
- ~~**Serving is not prompt-parameterised.**~~ — **closed 2026-09-05** (round 2
  R2): the manifest pins `data.style_system` and serving reads it, else the
  `SourcePolicy` author's prompt; D85 amended.
- ~~**The CLI is only half draws-reachable.**~~ — **closed 2026-09-05** (R3):
  every corpus command takes `--draws`/`--policy`; `ai-style build-view` and
  `dan-style policy --dump` exist; `ensure_view` keeps one view (D84).
- ~~**`check-splits --draws` builds under the identity policy**~~ — **closed
  2026-09-05** (R3): `dan-style check-splits` mirrors it under `view_policy()`.
- ~~**The prose/code ratio is at the wall**~~ — **closed 2026-09-05** (R0, R7):
  the ratio is retired for `PROSE_CEILING`, which only falls (D58), and the
  cull halved the prose it bounds.
- ~~**The old-layout pair-corpus fallback is dead weight.**~~ — **closed
  2026-09-05** (R3): deleted, with the `rg` counts in the commit body.
- **Three synthetic draws carry a generator-un-escaped anchor** and await a
  policy clause in the blog's `view_policy()` (D84 — it moves the corpus sha,
  so it is Dan's); the ids are in `ship-shape-2-2026-09.md`'s residue. The
  synth-side gate that would have refused them shipped 2026-09-05 (R4, C13).
- **Phase 7 preference/DPO** on the same Tinker platform (decisions #D5), with
  D7's Goodhart guards (signals with independent failure modes). Preference
  data already accrues from
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
  real output Dan cleans up? On syntactic proxies, measured since 2026-09-07
  (`ai-style features`, D88): every arm is separable from real at AUC 0.72–0.92
  on punctuation-free structure alone. Mannerisms beyond syntax are still
  eyeball-only; see [`argued-slop-arm.md`](argued-slop-arm.md) §"Stage 0".
- Single-source vs multi-source slop ablation — worth publishing? (post §"Whose
  slop?"; measurable now via `meta.generator`/`meta.gen`.)
- Weighting real against synthetic pairs — **the mechanism shipped 2026-09-03
  (D43); the diet is still nobody's choice.** `pair_weight` stamps `meta.weight`,
  a per-example loss scale on the train file (val stays at 1.0; D90 replaced the
  original integer replication). What made it urgent is that the corpus was already
  weighted, by accident: the budget sweep cut every passage at up to four
  `merge_max_chars`, so run 4a walked each stretch of prose several times under
  a manifest claiming one epoch, while real captures went through once — the
  scarce human side diluted by an artifact of generation, and the by-post val
  split cannot see it because val inflates identically. The counts:
  [`findings.md`](findings.md) "Effective passage weight". Choosing the
  correction costs a paid retrain and is Dan's call.
- Per-passage multiplicity — **the cap shipped as `per_passage` (D43)**; what to
  set it to has not. Grouping is by source post + character-span containment,
  not exact body: exact-body grouping sees almost none of the multiplicity
  containment sees, which is why every manifest ever written records
  `per_target: null`. What each cap leaves, and what it does to the real
  share, is the same `findings.md` table; `dan-style train --dry-run` prints
  the histogram before you spend.
- Base-model headroom — `Qwen/Qwen3.5-9B` today; step-up candidates are
  Qwen3.6-27B (dense) or Qwen3.6-35B-A3B (MoE, the Mac-friendliest shape).
- Detector — **settled (2026-06-30):** the trained Dan-voice classifier
  (StyleDistance backbone) is the signal, not a general AI-detector; the
  Pangram cross-check was deleted 2026-09-04 (D7).
