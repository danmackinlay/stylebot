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

## Architecture rules (don't violate without updating the plan)

- **Library-first, CLI second.** Each phase is a module with a typed function
  over explicit paths/params; the `click` CLI is a thin wrapper. The blog build
  imports the function directly. Two seams worth knowing (both 2026-08-11):
  `synth_cli` moved out of `bin/` to `stylebot.synth_cli` — 1000+ lines of
  library that happened to sit in the executables package — and now raises its
  own `SynthConfigError` / `SynthRunError` / `SynthGenerationError` instead of
  `click` exceptions, which `bin/ai_style.py` translates into exit codes at the
  boundary. And `BackendOpts` + `build_styler` (in `bin/ai_style.py`) are the
  one backend-flag surface `run` and `serve` share; the sidecar's rewrite op is
  `run` held open, so declaring those twelve flags twice had already drifted.
- **One entry point, subcommands** (`ai-style
  eval|export-table|log|make-splits|merge-adapter|run|serve|synth|train|train-clf`). Not a
  scatter of loose scripts: Phase 1's logger shipped as a standalone
  `ai-style-log` script until 2026-07-20, and was folded in. One console script,
  one `--help` tree. **One carve-out: one-off data migrations** — currently a
  membership of one, `scripts/migrate_capture_archives.py` (converts the
  superseded `captures/<id>.json` blobs to the `captures/<id>/` directory
  layout). It rewrites data already on disk to a new schema, is typed once per
  schema change, and would only clutter the `--help` tree of a tool nobody
  should be migrating with twice. The code that needs it names it in the error:
  `capture.iter_archives` refuses a legacy archive and prints the command. A
  migration script is deleted once the only corpus that exists has been through
  it — `restamp_style_system.py` (the 2026-08-04 `STYLE_SYSTEM` amendment) and
  `prune_stale_synth_pairs.py` (the 2026-08-10 segmentation change) went that
  way in 0.3; users migrate on version increment rather than carrying spent
  code. Anything a user runs as part of the pipeline is a subcommand, no
  exceptions.
- **Explicit paths, precedence `--flag` > `$STYLEBOT_DATA_DIR` > default**,
  resolved in `stylebot.config`. Two distinct inputs, never collapsed:
  `--blog-root` (read source) vs `--data-dir` (read/write state).
- **Defaults where a mistake is cheap, required flags where it isn't.**
  `ai-style log` has a default data-dir; `train`/`make-splits` must not.
- **Selection is a user-supplied policy.** Phases accept a
  `selector: Callable[[dict], bool]` (+ optional `sort_key`/`sampler`),
  defaulting to `stylebot.lib.is_human_authored` (the `automation: 0` example).
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
  committed (`_training_pairs/pairs.jsonl`, since 2026-07, for portability) —
  that's the backup story. The rule here is about *this* repo, not the data.
- **Secrets.** Keys reach the process as ordinary environment variables; Dan's
  come from direnv (`direnv exec ~/Source/livingthing uv run ai-style …`).
  Whatever holds them — an `.envrc`, an `.env` you source yourself — stays out
  of git. stylebot has no dotenv loader: `python-dotenv` was dropped 2026-08
  because bare `load_dotenv()` searches upward from `src/stylebot/`, so it read
  the checkout's neighbourhood rather than the directory you ran from.

Sanity-check before any commit in stylebot: `git status` must never show
`.env` / `.envrc` or `pairs.jsonl`.

## Validating corpus data

`stylebot.pairs.validate_pairs_file(path)` checks a `pairs.jsonl` against the
Phase 1 schema contract (roles, frozen system prompt, required `meta`). Run it
as the done-criteria gate for Phase 2 output and the Phase 3 input check.

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
- [x] Where the **already-captured** `pairs.jsonl` lives:
      `~/Source/livingthing/_training_pairs/pairs.jsonl` (254 records as of
      2026-06-27, validates clean). Point downstream phases at it with
      `STYLEBOT_DATA_DIR=~/Source/livingthing/_training_pairs` (or
      `--data-dir`).
- [x] `is_human_authored` defaults (`field="automation"`, `max_level=0`) match
      the real frontmatter exactly — no retarget needed.

## Current status (as of 2026-08-11)

**The pipeline runs end-to-end**: capture → gated synth → Tinker LoRA train →
guarded inference → multi-signal eval. Two trained runs exist (manifests in the
blog's `_training_pairs/runs/`; run 2 is the serving default). The 2026-08-11
pass harmonised segmentation across training and inference (defects A1–A5, see
[`decisions.md`](_plans/decisions.md) D26–D28) and pruned the 29 pairs the
unanchored `$$` rule had mangled; the corpus stands at **4595 pairs (4140
synthetic)**, and neither trained run has been re-run against it.

| Piece | State |
| --- | --- |
| Phase 0 scaffolding + Phase 1 `ai-style log` | done; daily-used; captures heading context (contract enforced in `stylebot.pairs`; mechanics in `_training_pairs/README.md`) |
| Phase 2 `ai-style synth` / `dan-style synth` | shipped; corpus settled ~4.6k pairs with identity + inflation hygiene gates (`--max-transform-sim` / `--max-length-ratio`). `stylebot.synth_cli` is the library seam; the CLI translates its `Synth*Error`s |
| Segmentation `stylebot.markdown` | one implementation, 2026-08-11; training and inference now agree, pinned by `tests/test_protection_parity.py`, `tests/test_corpus_stability.py` and the cross-implementation `tests/fixtures/segmenter_conformance.*` |
| Eval harness `ai-style eval` | built: four signals (Vale, LLM judge, detector, eyeball) + the Phase-4 `output` field; JSONL-batched + resumable ([`_plans/eval-harness.md`](_plans/eval-harness.md)) |
| Voice classifier (the detector signal) | built: StyleDistance embedding + logistic head; dep-free runtime `stylebot.classify`; artifact at livingthing `_models/voice-clf/` |
| Splits contract | `stylebot.splits` → blog's committed `splits.json`: frozen eval posts, styler posts, hash-stable detector pool |
| Phase 3 `ai-style train` / `dan-style train` | shipped: Tinker cookbook LoRA SFT behind the `[trainer]` extra; runs 1–2 trained (numbers in [`_plans/findings.md`](_plans/findings.md)) |
| Phase 4 `ai-style run` / `dan-style run` | shipped: training-protocol-faithful chunking — blockquotes, display math and `:::` divs are now protected at inference too, not just headings/fences — do-no-harm + anchor-integrity guards (inline anchors opt-in via `--strict-anchors`), best-of-N; Tinker-sampling backend v1, MLX exit verified, Fireworks open |

**Naming rule:** `dan-style X` is the blog-policy mirror of `ai-style X` — same
subcommand, swap the prefix. (The `train-targets` / `train-voice-clf` legacy
aliases were removed 2026-07-07.)

**Operations run blog-side**: the ordered what-do-I-type pipeline is the blog's
`_training_pairs/RUNBOOK.md`; blog policy constants (roster, strategies, gates,
chunking) live in `livingthing.training_targets`, and promoting a config change
means editing those constants and committing. Open work: the "Open slices" list
in [`_plans/OVERVIEW.md`](_plans/OVERVIEW.md).
