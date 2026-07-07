# CLAUDE.md — agent entry point

You are working on **stylebot**: tooling to fine-tune a small LLM that rewrites
AI-flavoured draft prose into Dan Mackinlay's voice. This file is the operating
manual; the **plan is the source of truth** — read it first:

→ **[`_plans/OVERVIEW.md`](_plans/OVERVIEW.md)** (phase map, data contract,
interfaces, sequencing) and the per-phase files it links.

Narrative/rationale: the blog post [*Fine-tuning danbot*](https://danmackinlay.name/notebook/fine_tuning_danbot.html)
(canonical source `notebook/fine_tuning_danbot.qmd` in the livingthing blog repo — not duplicated here).

## Setup & verify

```sh
uv sync                 # install (incl. dev group: pytest, ruff)
uv run pytest -q        # must stay green
uv run ai-style-log --help
uv run ai-style --help  # synth (Phase 2); split/train/eval land with their phases
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
  imports the function directly.
- **One entry point, subcommands** (`ai-style synth|split|train|eval`) + the
  shipped `ai-style-log`. Not a scatter of loose scripts.
- **Explicit paths, precedence `--flag` > `$STYLEBOT_DATA_DIR` > default**,
  resolved in `stylebot.config`. Two distinct inputs, never collapsed:
  `--blog-root` (read source) vs `--data-dir` (read/write state).
- **Defaults where a mistake is cheap, required flags where it isn't.**
  `ai-style-log` has a default data-dir; `train`/`split` must not.
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

## Never commit

- The **corpus** never enters **this repo** (`$STYLEBOT_DATA_DIR`, default
  `_training_pairs/` — gitignored except its README): stylebot is public
  tooling. The real corpus lives in the livingthing repo, where it *is*
  committed (`_training_pairs/pairs.jsonl`, since 2026-07, for portability) —
  that's the backup story. The rule here is about *this* repo, not the data.
- **Secrets** (`.env`). Copy `.env.example` → `.env` and fill in per-phase keys.

Sanity-check before any commit in stylebot: `git status` must never show
`.env` or `pairs.jsonl`.

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

## Current status (as of 2026-07-07)

| Piece | State |
| --- | --- |
| Phase 0 scaffolding + Phase 1 `ai-style-log` | done; daily-used; captures heading context ([`_plans/heading-context.md`](_plans/heading-context.md)) |
| Phase 2 `ai-style synth` / `dan-style synth` | built + curated: merge chunking + hygiene guards, OpenRouter rotation (models × strategies × efforts, all folded into `synth_key`), async parallel + window-position sessions ([`_plans/phase-2-synthetic-pairs.md`](_plans/phase-2-synthetic-pairs.md)) |
| Eval harness `ai-style eval` | built: four signals (Vale, LLM judge, detector, eyeball), JSONL-batched + resumable, scores HTML browser ([`_plans/eval-harness.md`](_plans/eval-harness.md)) |
| Voice classifier (the detector signal) | built: StyleDistance embedding + logistic head; dep-free runtime `stylebot.classify`, trainer behind the `[classifier]` extra; artifact committed at livingthing `_models/voice-clf/` |
| Splits contract | `stylebot.splits` / `ai-style make-splits` → blog's committed `splits.json`: frozen eval posts, styler posts, hash-stable detector pool ([`_plans/eval-harness.md`](_plans/eval-harness.md) "The detector decision") |
| Phase 3 train / Phase 4 | not started (data-/adapter-gated); see OVERVIEW |

**Naming rule:** `dan-style X` is the blog-policy mirror of `ai-style X` — same
subcommand, swap the prefix. (The `train-targets` / `train-voice-clf` legacy
aliases were removed 2026-07-07.)

**The active step is the experimental Phase-2 generation loop**, not a one-shot
paid run: generate a small batch into a *scratch* `--data-dir`, eyeball
(`--report`/`--sample`), score with `ai-style eval`, promote a winner into the
real corpus. **Experiments run through `dan-style synth`**
(`cd ~/Source/livingthing && uv run dan-style synth --data-dir /tmp/scratch
--slop-strategy X --limit 40`), never bare `ai-style synth` — the wrapper
carries the blog chunking/selection policy; the generic defaults produce
fragments too short to judge style on. The full audition→generate→split
runbook, including where preferred generators/strategies are recorded
(`livingthing.training_targets` constants), is
[`_plans/next-steps.md`](_plans/next-steps.md) §A.
