# CLAUDE.md — agent entry point

You are working on **stylebot**: tooling to fine-tune a small LLM that rewrites
AI-flavoured draft prose into Dan Mackinlay's voice. This file is the operating
manual; the **plan is the source of truth** — read it first:

→ **[`_plans/OVERVIEW.md`](_plans/OVERVIEW.md)** (phase map, data contract,
interfaces, open slices) and the files it links —
[`decisions.md`](_plans/decisions.md) for why a choice landed and what we
rejected, [`findings.md`](_plans/findings.md) for every measured result.

The per-phase design plans were **retired 2026-08-10** once their work shipped:
built code is its own best description, and those documents had begun to drift
from it. What git could not preserve — rejected alternatives and measurements —
was consolidated into the two files above.

Narrative/rationale: the blog post [*Fine-tuning danbot*](https://danmackinlay.name/notebook/fine_tuning_danbot.html)
(canonical source `notebook/fine_tuning_danbot.qmd` in the livingthing blog repo — not duplicated here).

## Setup & verify

```sh
uv sync                 # install (incl. dev group: pytest, ruff)
uv run pytest -q        # must stay green
uv run ai-style log --help
uv run ai-style --help  # the full subcommand tree; every phase has landed
```

**Always run Python through `uv`.** `uv run python …`, `uv run pytest`,
`uv run <console-script>` — **never** bare `python`/`python3`, not even for a
throwaway one-liner or a quick "just parse this file" check. Bare `python3` grabs
whatever arbitrary interpreter is first on `PATH` instead of this project's
pinned version and `uv`-installed env, so it's non-reproducible and can't import
the package. To read or parse source, prefer the search tools (`rg`, `grep`) or
`uv run python -c …`. This applies to subagents too.

## Budgets and layers

Adopted 2026-09-04, after one day added 12,277 lines to `src/` and `tests/`.
Why, and what was rejected: [`decisions.md`](_plans/decisions.md) D58; the
measurement and the repair list are the blog's
`_plans/stylebot-seam-followup-2026-09.md`.

- **P1. Every "why" lives in exactly one layer; the others point at it.**
  Durable to ephemeral: `decisions.md` (why, alternatives, re-evaluate-if) →
  `findings.md` (numbers, one table per measurement) → code (what, as a
  contract) → commit message (what changed, one line why, a pointer). A
  docstring may cite `D57` and may not restate it; a decision may cite a finding
  and may not reproduce its table. **Test for a violation: if the same sentence
  could be pasted into two of the four layers, it is in the wrong one or both.**
- **P2. Size budgets, enforced by `tests/test_budget.py`.** Module ≤ 800 lines,
  function ≤ 80, docstring ≤ 15, module docstring ≤ 30, package-wide
  (docstring + comment) / code ≤ 0.40. Everything over budget today is
  allowlisted at its exact size; a recorded number may only ever be lowered, and
  the entry is deleted the moment its module comes under budget. **Adding an
  allowlist entry needs a D-entry.** The blog runs the same test over
  `src/livingthing`.
- **P3. A `decisions.md` entry is ≤ 30 lines in the D26 shape** — decision,
  alternatives one line each, deciding constraint, re-evaluate-if. Numbers are a
  citation into `findings.md`, never a copy. A superseded entry collapses to
  three lines: title, "superseded by Dn on <date>", one sentence of what
  changed. Past 1,200 lines, D1–D33 move to `decisions-archive.md` and the index
  at the top keeps their one-line summaries.
- **P4. A `findings.md` entry is one table plus ≤ 12 lines of reading** — date,
  the command, the artifact it read, the table, the D-number it decided. How a
  measurement went wrong before it went right is a decision's "alternatives", or
  nothing. A superseded measurement keeps its table and gets a one-line strike.
- **P5. Commits: subject ≤ 72 chars, body ≤ 12 lines, one concern.** The body
  says what changed and cites the decision or work item. No commit touches a
  file its subject does not cover. "Could not be separated in the working tree"
  is the working tree's problem — P8.
- **P6. This file is an operating manual ≤ 160 lines.** Every bullet is a rule
  plus a pointer; how the rule was learned belongs in `decisions.md`.
- **P7. A plan retires when its work ships** — within a week of the last item
  landing, to a ≤ 40-line stub: what it was, what shipped (commit range), what
  was refuted (D-numbers), what stayed open (the successor). Git holds the rest.
- **P8. Dispatch rules for agents.**
  - One worktree per agent, or one committer. Named pathspecs only — never
    `git add -A`/`.`/`-u`.
  - Report anything out of scope rather than fixing it.
  - **No new mechanism without a budget.** A new thread, lock, process,
    fallback path, module, or file over 300 lines needs its D-entry *before* the
    code, and the entry names the number it clears (P2's size, or D56's measured
    win). "Kept out of caution" is not a justification.
  - Measure before deleting, measure before building, write the number once (P1).
  - Nothing is pushed by an agent.

## Architecture rules (don't violate without updating the plan)

- **`asyncio` first; threading must stringently justify itself.** Set
  2026-09-03, after `eval.score_pairs_file`'s eight-worker `ThreadPoolExecutor`
  segfaulted the process (exit 139, deterministic) by driving one
  `SentenceTransformer` from several threads — latent for months, surfaced only
  when chunk-wise scoring raised the embed rate ~6x, and misdiagnosed once
  before a subprocess isolation disproved the first theory. `asyncio` is this
  project's concurrency mechanism for IO and is what `synth` already uses at
  scale; the threads in `eval` were a reflex, because `Judge`/`Detector` are
  plain sync callables, not an argument. New threading needs **a measured
  substantial win in performance or complexity** and a `decisions.md` entry
  carrying the number. Three corollaries: never share a non-re-entrant native
  object (a torch model, a Rust tokenizer) across threads; a lock as the answer
  means the threads were the wrong shape, not that the problem is solved; and
  for CPU-bound work prefer **batching over fan-out** — `_vale_prepass` (one
  `vale` process per ~256 texts) is the pattern.
- **Library-first, CLI second.** Each phase is a module with a typed function
  over explicit paths/params; the `click` CLI is a thin wrapper. The blog build
  imports the function directly. Two seams worth knowing (both 2026-08-11):
  `synth_cli` moved out of `bin/` to `stylebot.synth_cli` — 1000+ lines of
  library that happened to sit in the executables package — and now raises its
  own `SynthConfigError` / `SynthRunError` / `SynthGenerationError` instead of
  `click` exceptions, which `bin/ai_style.py` translates into exit codes at the
  boundary. And `stylebot.backends` (`BackendOpts`, `build_styler`,
  `serving_diet`, `opts_from_manifest`) is the one backend surface `run` and
  `serve` share; the sidecar's rewrite op is `run` held open, so declaring
  those twelve flags twice had already drifted. It sat in `bin/ai_style.py`
  until 2026-09-03, where livingthing could not import it and reimplemented it
  instead — losing `max_tokens` in the copy, which truncated every blog rewrite
  at the 2000-token floor while the promoted run served a 4500-char diet (D35).
  A library that can only be reached through `CliRunner` will be copied.
- **One entry point, subcommands** (`ai-style
  check-splits|eval|export-table|extend-eval|log|make-splits|merge-adapter|run|serve|status|synth|train|train-clf|validate-pairs`).
  Not a scatter of loose scripts: Phase 1's logger shipped as a standalone
  `ai-style-log` script until 2026-07-20, and was folded in. One console script,
  one `--help` tree. **No carve-out, and there is no `scripts/` directory**
  (W5, 2026-09-03). One-off data migrations used to be one — loose files that
  rewrite corpus state already on disk to a new schema, kept out of the tree so
  they would not clutter the `--help` of a tool nobody should be migrating with
  twice. The tidy `--help` was the right goal and `scripts/` was the wrong
  means: it is not in the wheel, so `capture.iter_archives` printed a command a
  `pip install stylebot` user did not have, and the blog's RUNBOOK reached
  across the seam with `uv run python ../stylebot/scripts/…` — a hardcoded
  blog→stylebot path, which is the arrow W5 straightened. `hidden=True` buys the
  tidy tree without the exception.

  So migrations are `stylebot.migrations` (typed functions) behind **`ai-style
  data`**, a hidden group: absent from `ai-style --help`, documented under
  `ai-style data --help`, and nameable by any error message or runbook step.
  Current membership: `migrate-captures` (the only path off the legacy
  `captures/<id>.json` blobs), `prune-invented-anchors` (the corpus that
  predates `--drop-invented-anchors`), `prune-stale` (W12's orphan and
  eval-contamination cull) and `shard-pairs` (the one-way move from a single
  `pairs.jsonl` to the `pairs/` shard directory, D50). All dry-run by default
  and need `--apply`; the prunes keep a `.bak`; all carry a test in
  `tests/test_migrations.py` — the claim that both were tested was made here
  while only one of them was, which is the sort of drift a directory outside the
  package makes easy. A migration is deleted once the only corpus that exists
  has been through it — `restamp_style_system.py` (the 2026-08-04 `STYLE_SYSTEM`
  amendment) and `prune_stale_synth_pairs.py` (the 2026-08-10 segmentation
  change) went that way in 0.3; users migrate on version increment rather than
  carrying spent code. Delete the function, its command and its test together.
- **Explicit paths, precedence `--flag` > `$STYLEBOT_DATA_DIR` > default**,
  resolved in `stylebot.config`. Two distinct inputs, never collapsed:
  `--blog-root` (read source) vs `--data-dir` (read/write state).
- **Defaults where a mistake is cheap, required flags where it isn't.**
  `ai-style log` has a default data-dir; `train`/`make-splits` must not.
- **Selection is a user-supplied policy.** Phases accept a
  `selector: Callable[[dict], bool]` (plus, at assembly, `per_passage` and
  `pair_weight` — `train.assemble_training_corpus`), defaulting to
  `stylebot.lib.is_human_authored` (the `automation: 0` example). There is no
  `sort_key` or `sampler` hook: `--sort` retired 0.3.1 and its library twin was
  deleted 2026-09-03 with no caller (`_plans/decisions.md` D43).
  Never hardcode the predicate; the caller can pass their own or hand in a
  pre-selected file list.
- **`STYLE_SYSTEM` (`stylebot.ai_core`) is frozen.** It is baked into every
  logged pair. Changing the string invalidates the corpus. Phase 2 must emit it
  verbatim as `messages[0]`.
- **Blog/generic boundary.** Generic frontmatter+markdown tooling is reusable;
  the only blog-specific code allowed is the bundled selector example
  (`is_human_authored`). The old blog-build cruft (`migrate_ai_dates`,
  `is_auxiliary_post`/`AUXILIARY_TYPES`, the `SUMMARY_*`/`QUALITY_*` prompts) was
  removed from stylebot (2026-06-29) — it had been copied in but never used here,
  and lives only in the blog now. Don't reintroduce it: auxiliary/AI-touched
  filtering is the caller's `selector` policy.
- **One segmenter: `stylebot.markdown`.** Markdown segmentation had been written
  five times inside the package — seven counting `livingthing.qmd_core` and the
  marker's TypeScript port — and the copies had silently diverged: what Phase 2
  excluded from *training* targets was not what Phase 4 refused to send at
  *inference*. Everything that needs to know "is this line prose?" now asks
  here. **The protection principle**: a construct is protected — excluded from
  training targets *and* inference chunks — when rewriting it would change
  meaning rather than voice, or when there is no voice to apply. Three grounds:
  (1) not the author's words (blockquotes); (2) not prose (fenced code, display
  math, figure/reference divs); (3) structural markers (heading and `:::` fence
  lines carry level/class semantics) — *though what they delimit may be prose*.
  **Corollary: a container is not protected merely for being a container.** Which
  div classes bear prose is policy — `.callout-note` is a Quarto convention — so
  it enters through the caller-supplied `div_is_prose` hook rather than being
  hardcoded here. That is the bullet above, and it is precisely the rule the
  five-way duplication had been violating.

## Never commit

- The **corpus** never enters **this repo** (`$STYLEBOT_DATA_DIR`, default
  `_training_pairs/` — gitignored except its README): stylebot is public
  tooling. The real corpus lives in the livingthing repo, where it *is*
  committed (`_training_pairs/pairs/`, since 2026-07, for portability) —
  that's the backup story. The rule here is about *this* repo, not the data.
- **Secrets.** Keys reach the process as ordinary environment variables; Dan's
  come from direnv (`direnv exec ~/Source/livingthing uv run ai-style …`).
  Whatever holds them — an `.envrc`, an `.env` you source yourself — stays out
  of git. stylebot has no dotenv loader: `python-dotenv` was dropped 2026-08
  because bare `load_dotenv()` searches upward from `src/stylebot/`, so it read
  the checkout's neighbourhood rather than the directory you ran from.

Sanity-check before any commit in stylebot: `git status` must never show
`.env` / `.envrc`, a `pairs.jsonl`, or a `pairs/` directory.

## Validating corpus data

`stylebot.pairs.validate_pairs_file(path)` checks a corpus — a `pairs.jsonl`
**or a `pairs/` shard directory** (`stylebot.corpus`, D50) — against the Phase 1
schema contract: roles, frozen system prompt, required `meta`, the types and
domains of every meta key it knows, and **`record_id` uniqueness across the
whole corpus, every shard together**. Run it as the done-criteria gate for Phase 2 output and
the Phase 3 input check; `ai-style validate-pairs PAIRS [--fix]` is the same
thing on demand, `--fix` dropping duplicate cells (first occurrence wins).

Uniqueness is a file-level property, so it cannot live in `validate_pair_record`
— which is why 21 duplicate cells reached the live corpus before anything
noticed. New records also carry `meta.schema_version`, `meta.system_sha` and
`meta.provenance` (which trees, which chunk policy, what the source post's
`automation` said at cut time). Records without them are the pre-2026-09 corpus
and stay valid. See `_plans/decisions.md` D36.

## BLOG INTEGRATION — filled in 2026-06-27 (Dan's blog)

The data-heavy phases need the prose corpus. Wired in against Dan's live blog:

- [x] Blog repo location / clone path: `~/Source/livingthing`
      (remote `git@github.com:danmackinlay/danmackinlay.github.io.git`, "The
      Living Thing"). Pass as `--blog-root`.
- [x] Post file glob & layout: Quarto `.qmd`, default glob `**/*.qmd`. Posts
      live under `post/` (~73) and `notebook/` (~1600); `digest/` is auxiliary
      (`type: digest`). Build dirs to skip: `_site`, `_freeze`, `_`-prefixed
      dirs/files, `.`-prefixed (incl. `.claude/worktrees/…` clones). The
      `is_valid_qmd_file` / `gather_qmd_files` skip-list already covers these.
- [x] Frontmatter conventions: `automation:` is an int level of how much AI
      touched a post — `0` = pure human (1591 posts), `1` (58), `2` (61).
      `automation: 0` is the slop-free target prose. Auxiliary posts carry
      `type: digest`/`about` (see `livingthing.lib.is_auxiliary_post`).
- [x] Where the **already-captured** corpus lives:
      `~/Source/livingthing/_training_pairs/pairs/` (a shard directory since
      2026-09-03; a single `pairs.jsonl` before that, and still the default
      shape for anyone who has not migrated). Point downstream phases at it with
      `STYLEBOT_DATA_DIR=~/Source/livingthing/_training_pairs` (or
      `--data-dir`).
- [x] `is_human_authored` defaults (`field="automation"`, `max_level=0`) match
      the real frontmatter exactly — no retarget needed.

## Current status (as of 2026-09-03)

**The pipeline runs end-to-end**: capture → gated synth → Tinker LoRA train →
guarded inference → multi-signal eval.

**No count, run number or "serving default" is written down here any more.**
Those facts live in the *blog's* `_training_pairs/` — a different repo on a
different commit clock — and they move on every capture, synth run and
promotion, so no stylebot file can stay true about them. This section said two
trained runs and 4,595 pairs, unchanged, while the blog reached five runs and
8,182 records. Ask the pipeline instead:

```sh
uv run ai-style status --pairs … --splits … --runs … --detector-model … --blog-root …
cd ~/Source/livingthing && uv run dan-style status   # the same, every path from policy
```

It prints corpus size and composition, the eval stratum version and
`splits_sha256`, the promoted run id and the chunk diet it serves, the detector
artifact's `git_sha`, and the cross-checks between them — a detector fit under a
stratum `splits.json` has since moved past is the failure no single artifact can
see (`stylebot.status`, [`decisions.md`](_plans/decisions.md) D52). The
authoritative record of what serves is the newest promoted manifest in the
blog's `_training_pairs/runs/`; dated measurements are in
[`findings.md`](_plans/findings.md).

| Piece | State |
| --- | --- |
| Phase 0 scaffolding + Phase 1 `ai-style log` | done; daily-used; captures heading context (contract enforced in `stylebot.pairs`; mechanics in `_training_pairs/README.md`) |
| Phase 2 `ai-style synth` / `dan-style synth` | shipped, with identity + inflation hygiene gates (`--max-transform-sim` / `--max-length-ratio`). `stylebot.synth_cli` is the library seam; the CLI translates its `Synth*Error`s |
| Segmentation `stylebot.markdown` | one implementation, 2026-08-11; training and inference now agree, pinned by `tests/test_protection_parity.py` and the cross-implementation `tests/fixtures/segmenter_conformance.*`. The corpus-stability floor (does today's segmenter still cut the corpus we paid to generate?) is the blog's `tests/test_corpus_stability.py` — it needs the corpus, the policy and a git pin, all three of which live there |
| Eval harness `ai-style eval` | built: four signals (Vale, LLM judge, detector, eyeball) + the Phase-4 `output` field; JSONL-batched + resumable ([`_plans/eval-harness.md`](_plans/eval-harness.md)) |
| Voice classifier (the detector signal) | built: StyleDistance embedding + logistic head; dep-free runtime `stylebot.classify`; artifact at livingthing `_models/voice-clf/` |
| Splits contract | `stylebot.splits` → blog's committed `splits.json`: frozen eval posts, styler posts, hash-stable detector pool |
| Phase 3 `ai-style train` / `dan-style train` | shipped: Tinker cookbook LoRA SFT behind the `[trainer]` extra; per-passage grouping and integer-replication weight hooks (D43). Which runs exist and which one serves: `dan-style status`; what they measured: [`findings.md`](_plans/findings.md) |
| Phase 4 `ai-style run` / `dan-style run` | shipped: training-protocol-faithful chunking — blockquotes, display math and `:::` divs are now protected at inference too, not just headings/fences — do-no-harm + anchor-integrity guards (inline anchors opt-in via `--strict-anchors`), best-of-N; Tinker-sampling backend v1, MLX exit verified, Fireworks open |

**Naming rule:** `dan-style X` is the blog-policy mirror of `ai-style X` — same
subcommand, swap the prefix. (The `train-targets` / `train-voice-clf` legacy
aliases were removed 2026-07-07.)

**Operations run blog-side**: the ordered what-do-I-type pipeline is the blog's
`_training_pairs/RUNBOOK.md`; blog policy constants (roster, strategies, gates,
chunking) live in `livingthing.training_targets`, and promoting a config change
means editing those constants and committing. Open work: the "Open slices" list
in [`_plans/OVERVIEW.md`](_plans/OVERVIEW.md).
