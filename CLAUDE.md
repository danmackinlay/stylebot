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
```

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
  the only blog-specific code allowed is the bundled selector example. Do NOT
  build new phases on `migrate_ai_dates` / `is_auxiliary_post` — those are
  blog-build cruft slated to go home to the blog.

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

## BLOG INTEGRATION — fill in on first run with blog access

The data-heavy phases need the prose corpus. These specifics are unknown until
the blog is wired in; record them here (and in `_plans/phase-2-synthetic-pairs.md`)
the first time an agent has blog access:

- [ ] Blog repo location / clone path:
- [ ] Post file glob & layout (e.g. `posts/**/*.qmd`):
- [ ] Frontmatter conventions: `automation` level meaning; how slop-free posts
      are marked; any fields the selector should read.
- [ ] Where the **already-captured** `pairs.jsonl` currently lives, and the
      `STYLEBOT_DATA_DIR` to point downstream phases at.
- [ ] Confirm `is_human_authored` defaults match the real frontmatter (adjust
      `field`/`max_level`, or pass a custom selector).

## Current status

Phase 0 (scaffolding) and Phase 1 (`ai-style-log`, daily-used) are done.
**Buildable now, no trained model needed:** Phase 2 (synthetic pairs) and the
Eval harness. Phase 3/4 are data-/adapter-gated. See OVERVIEW for the next move.
