# CLAUDE.md — agent entry point

You are working on **stylebot**: tooling to fine-tune a small LLM that rewrites
AI-flavoured draft prose into Dan Mackinlay's voice. This file is the operating
manual; the **plan is the source of truth — read it first**:

→ **[`_plans/OVERVIEW.md`](_plans/OVERVIEW.md)** — phase map, interfaces, data
contract, open slices; it links [`decisions.md`](_plans/decisions.md) (why a
choice landed, what was rejected) and [`findings.md`](_plans/findings.md) (every
measured result), which together absorbed the per-phase plans. Narrative:
[*Fine-tuning danbot*](https://danmackinlay.name/notebook/fine_tuning_danbot.html).

## Setup & verify

```sh
uv sync --all-extras    # dev group + both extras; without them one test skips
uv run pytest -q        # must stay green
uv run ai-style --help  # the full subcommand tree; every phase has landed
uv run ruff check && uv run ruff format --check src tests  # lint AND format; the blog is lint-only, on purpose
```

**Always run Python through `uv`** — `uv run python …`, `uv run pytest`,
`uv run <console-script>`. **Never** bare `python`/`python3`, not even for a
throwaway one-liner: it takes whatever interpreter is first on `PATH` instead of
this project's pinned version and `uv`-installed env, so it is non-reproducible
and cannot import the package. To read or parse source prefer `rg`/`grep`, else
`uv run python -c …`. Subagents too.

## Budgets and layers

Adopted 2026-09-04, after one day's seam pass. Why and what was rejected: D58;
measurement and repairs, the blog's `_plans/stylebot-seam-followup-2026-09.md`.

- **P1. Every "why" lives in exactly one layer; the others point at it.**
  Durable to ephemeral: `decisions.md` (why, alternatives, re-evaluate-if) →
  `findings.md` (numbers, one table per measurement) → code (what, as a
  contract) → commit message (what changed, one line why, a pointer). A
  docstring may cite `D57` and may not restate it. **Violation test: if a
  sentence could be pasted into two of the four layers, it is in the wrong one.**
- **P2. Size budgets, enforced by `tests/test_budget.py`.** Module ≤ 800 lines,
  function ≤ 80, docstring ≤ 15, module docstring ≤ 30, package-wide docstring +
  comment lines ≤ `PROSE_CEILING`, which only falls. Everything over budget today
  is allowlisted at its exact size; a recorded number may only be lowered, and
  the entry is deleted when its module comes under budget. **A new allowlist
  entry needs a D-entry.** The blog runs the same test over `src/livingthing`.
- **P3. A `decisions.md` entry is ≤ 30 lines in the D26 shape** — decision,
  alternatives one line each, deciding constraint, re-evaluate-if. Numbers are a
  citation into `findings.md`, never a copy. A superseded entry collapses to
  three lines: title, "superseded by Dn on <date>", one sentence of what
  changed. Past 1,300 lines, the oldest entries move to `decisions-archive.md`
  and the index at the top keeps their one-line summaries.
- **P4. A `findings.md` entry is one table plus ≤ 12 lines of reading** — date,
  the command, the artifact it read, the table, the D-number it decided. How a
  measurement went wrong before it went right is a decision's "alternatives", or
  nothing. A superseded measurement keeps its table and gets a one-line strike.
- **P5. Commits: subject ≤ 72 chars, body ≤ 12 lines, one concern.** The body
  says what changed and cites the decision or work item. No commit touches a
  file its subject does not cover — except that an allowlist edit in
  `tests/test_budget.py` rides in the commit that earns it. "Could not be
  separated in the working tree" is the working tree's problem — P8.
- **P6. This file is an operating manual ≤ 160 lines.** Every bullet is a rule
  plus a pointer; how the rule was learned belongs in `decisions.md`.
- **P7. A plan retires when its work ships** — within a week of the last item
  landing, to a ≤ 40-line stub: what it was, what shipped (commit range), what
  was refuted (D-numbers), what stayed open (the successor). Git holds the rest.
- **P8. Dispatch rules for agents.**
  - One worktree per agent, or one committer. Named pathspecs only — never
    `git add -A`/`.`/`-u`.
  - A cross-repo slice: one agent, two worktrees, two commits merged back-to-
    back; test the blog half with `uv run --with-editable <stylebot-wt> pytest`.
  - Report anything out of scope rather than fixing it.
  - **No new mechanism without a budget.** A new thread, lock, process,
    fallback path, module, or file over 300 lines needs its D-entry *before* the
    code, and the entry names the number it clears (P2's size, or D56's measured
    win). "Kept out of caution" is not a justification.
  - Measure before deleting, measure before building, write the number once (P1).
  - A main checkout's index is the owner's: a merge gate that finds it dirty
    **stops and asks**, and never stages or restores — `ship-shape-2026-09.md`.
  - Nothing is pushed by an agent.

## Architecture rules (don't violate without updating the plan)

- **`asyncio` first; threading must stringently justify itself** — a measured
  substantial win in performance *or* complexity, in a D-entry with the number
  (D56, D59). Corollaries: never share a non-re-entrant native object (a torch
  model, a Rust tokenizer) across threads; a lock means they were the wrong
  shape; CPU-bound work prefers **batching over fan-out** (`_vale_prepass`).
- **Library-first, CLI second.** Each phase is a module with a typed function
  over explicit paths/params; the `click` CLI (`bin/ai_style.py`) is a thin
  wrapper translating `stylebot.errors` into exit codes (D47). A library
  reachable only through `CliRunner` gets copied, and the copy drifts (D35).
- **One entry point, subcommands** (`ai-style
  check-splits|eval|export-table|extend-eval|log|make-splits|merge-adapter|promote|run|serve|status|synth|train|train-clf|validate-pairs`),
  not a scatter of loose scripts. **No carve-out, and there is no `scripts/`
  directory** — an error message must name a command that exists, and a
  directory outside the package is outside the checks (D38).
- **Nothing rewrites the corpus: draws are appended, views are built** (D84).
  A prune, re-cut or restamp is a policy clause plus a rebuild. A one-off
  migration is a hidden command deleted once its corpus has run it (D38) —
  none exists today.
- **Explicit paths, precedence `--flag` > `$STYLEBOT_DATA_DIR` > default**,
  resolved in `stylebot.config`. Two distinct inputs, never collapsed:
  `--blog-root` (read source) vs `--data-dir` (read/write state) — D45, D50.
- **Defaults where a mistake is cheap, required flags where it isn't.**
  `ai-style log` has a default data-dir; `train`/`make-splits` must not.
- **Selection is a user-supplied policy.** Phases accept a
  `selector: Callable[[dict], bool]` (plus, at assembly, `per_passage` and
  `pair_weight`), defaulting to `stylebot.lib.is_human_authored` — an example,
  not a gate. Never hardcode it; no `sort_key`/`sampler` hook (D43).
- **Validate a corpus before training on it.** `pairs.validate_pairs_file(path)`
  / `ai-style validate-pairs PAIRS` checks a `pairs.jsonl`, `pairs/` dir or built
  view against the Phase 1 schema: roles, frozen system prompt, required `meta`,
  **`record_id` uniqueness across all shards**. Read-only: a duplicate is a policy
  clause plus a rebuild. Pre-`schema_version` records stay valid; D36, D50.
- **`STYLE_SYSTEM` (`stylebot.ai_core`) is frozen**: it is baked into every
  logged pair, so a change re-stamps every record — a view rebuild under D84,
  once a migration (D64). Phase 2 emits it verbatim as `messages[0]`.
- **Blog/generic boundary.** Generic frontmatter+markdown tooling is reusable; the
  only blog-specific code allowed is the bundled selector example. Don't
  reintroduce the blog-build cruft removed 2026-06-29 — OVERVIEW draws the line.
- **One segmenter: `stylebot.markdown`.** Everything asking "is this line prose?"
  asks there — five divergent copies are why. Its docstring holds the
  **protection principle**; div-class policy enters via `div_is_prose` (D26–D29).

## Never commit

- The **corpus** never enters **this repo** (`$STYLEBOT_DATA_DIR`, default
  `_training_pairs/`, gitignored except its README): stylebot is public tooling.
  It lives in livingthing, where it *is* committed — the backup story (D37).
- **Secrets.** Keys arrive as ordinary environment variables, from direnv
  (`direnv exec ~/Source/livingthing uv run ai-style …`); whatever holds them
  stays out of git. stylebot has no dotenv loader (D24).
- So before any commit here, `git status` must never show `.env` / `.envrc`, a
  `pairs.jsonl`, or a `pairs/`, `draws/` or `views/` directory.

## Blog integration (Dan's blog, wired 2026-06-27)

- Blog repo `~/Source/livingthing` (`danmackinlay.github.io`), passed as `--blog-root`.
- The corpus there is `_training_pairs/` — committed `draws/`, built `views/`.

## Current status

End-to-end: capture and gated synth append draws → view built under policy →
Tinker LoRA train → guarded inference → multi-signal eval; per-phase state is
OVERVIEW's phase map. **No count, run number or "serving default" is written
here** — they move in another repo on another clock and every one went stale (D52):

```sh
uv run ai-style status --draws … --splits … --runs … --detector-model … --blog-root …
cd ~/Source/livingthing && uv run dan-style status   # the same, every path from policy
```

**Naming rule:** `dan-style X` is the blog-policy mirror of `ai-style X`.
**Operations run blog-side**: the ordered pipeline is the blog's
`_training_pairs/RUNBOOK.md`; policy constants (roster, strategies, gates,
chunking, `view_policy()`) are `livingthing.training_targets` — promoting a
config change is a commit there; what serves is the run `dan-style promote
RUN_ID` marked `promoted: true` (D74). Open work: OVERVIEW's "Open slices".
