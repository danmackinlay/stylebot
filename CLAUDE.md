# CLAUDE.md — agent entry point

You are working on **stylebot**: tooling to fine-tune a small LLM that rewrites
AI-flavoured draft prose into Dan Mackinlay's voice. This file is the operating
manual; the **plan is the source of truth** — read it first:

→ **[`_plans/OVERVIEW.md`](_plans/OVERVIEW.md)** (phase map, data contract,
interfaces, sequencing) and the per-phase files it links.

Narrative/rationale: [`docs/fine_tuning_danbot.qmd`](docs/fine_tuning_danbot.qmd).

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

- The **corpus** (`$STYLEBOT_DATA_DIR`, default `_training_pairs/` — gitignored
  except its README). It is private and manually backed up; this repo is public.
- **Secrets** (`.env`). Copy `.env.example` → `.env` and fill in per-phase keys.

Sanity-check before any commit: `git status` must never show `.env` or
`pairs.jsonl`.

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

## Current status (as of 2026-06-28)

Phase 0 (scaffolding) and Phase 1 (`ai-style-log`, daily-used) are done; Phase 1
now also captures **heading context** (default ON; see below).

Phase 2 (synthetic pairs) is **built and curated** — `ai-style synth` /
`train-targets` over `stylebot.synth`, tested green (89 stylebot + 13 blog).
The target-curation pipeline (all generic mechanism in `stylebot`, policy in the
blog's `livingthing.training_targets`):
- prose-only extraction (`segment_for_edit`), hygiene (min/max chars, tables,
  prose-residual **link-list** guard, **list-item** drop), `stop_at_headers`
  (cuts `## Incoming`), stub-marker drop (`🚧TODO🚧`);
- **merge** mode (section-aware paragraph packing into ~1.5k-char passages);
- **heading context** — the section heading prepended verbatim+identically to
  both sides of every pair, shared by Phase 1 + 2 via
  `stylebot.pairs.build_pair_content` / `meta.context` (see
  [`_plans/heading-context.md`](_plans/heading-context.md));
- a read-only **report**/`--sample` visualiser (`stylebot.report`).
Real-blog dry-run: ~3.6k passages from 492 quality>6 posts, 85% heading-framed.

**Slop strategy + OpenRouter (built 2026-06-28).** The slop *prompt* is a knob:
`STRATEGIES` / `--slop-strategy` (`polish`|`engaging`|`catalogue`) /
`--slop-system-file`, recorded as `meta.slop_strategy` and folded into `synth_key`.
`openrouter_generator` / `--openrouter-model` reach many models off one
**`OPENROUTER_API_KEY`** (the blog's `train-targets` now defaults to an OpenRouter
rotation). Add `OPENROUTER_API_KEY` to `.env` / `.envrc` (gitignored).

**Eval harness (built 2026-06-28; batched 2026-06-28).** `stylebot.eval` +
`ai-style eval` — the offline four-signal scorer (Vale, OpenRouter LLM-judge,
pluggable detector, eyeball), keyless by default. **JSONL-native + batched:**
`ai-style eval --pairs PATH.jsonl` scores the slop/Dan *fields* of every pair
(heading-context stripped) and appends an **id-keyed** `scores.jsonl`
(`score_pairs_file`, concurrent + resumable, joins back via `synth_key`/
`capture_id`); `summarize_scores(by="slop_strategy")` gives per-strategy means.
A read-only **scores visualiser** (`--report scores.html` /
`stylebot.report.render_scores_report`) joins pairs+scores into self-contained
HTML — slop↔Dan + judge score/rationale, sortable by slop→Dan delta, faceted by
strategy (reuses the targets-report infra; generic over score fields for Phase 4).
The **statistical-detector audition** is the one remaining eval signal (deferred —
GPU/$$, operator's call).

**The active step is the experimental Phase-2 generation loop**, not a one-shot
paid run: per `--slop-strategy`, generate a small batch into a *scratch*
`--data-dir`, eyeball (`--report`/`--sample`), score with `ai-style eval`, promote
a winner into the real corpus. Then the corpus run is `cd ~/Source/livingthing &&
uv run train-targets --limit N`. Phase 3/4 are data-/adapter-gated. See OVERVIEW.
