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

## Current status (as of 2026-06-29)

> **Codebase decluttered (2026-06-29).** A QA pass removed dead blog-build code
> (`migrate_ai_dates`, `is_auxiliary_post`, the `SUMMARY_*`/`QUALITY_*` prompts,
> unused frontmatter/date helpers) and the redundant direct-Anthropic generator
> + the `anthropic` dependency — hosted models reach stylebot via OpenRouter now.
> The mechanism is lean and this housekeeping is done; next moves (shipping) live
> in [`_plans/next-steps.md`](_plans/next-steps.md).

Phase 0 (scaffolding) and Phase 1 (`ai-style-log`, daily-used) are done; Phase 1
now also captures **heading context** (default ON; see below).

Phase 2 (synthetic pairs) is **built and curated** — `ai-style synth` /
`dan-style synth` over `stylebot.synth`, tested green (101 stylebot + 13 blog).
(**Naming rule:** `dan-style X` is the blog-policy mirror of `ai-style X` —
same subcommand, swap the prefix. `train-targets` / `train-voice-clf` are
legacy aliases of `dan-style synth` / `dan-style train-clf|eval`.)
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
**`OPENROUTER_API_KEY`** (the blog's `dan-style synth` now defaults to an OpenRouter
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
HTML — slop↔Dan + judge score/rationale + detector P(slop) badge, sortable by
slop→Dan delta, faceted by strategy (reuses the targets-report infra; generic over
score fields for Phase 4).

**Trained voice classifier — the detector signal (built 2026-06-30; trainer
generalised into stylebot 2026-07-06).** The 4th eval signal is a Dan-vs-AI-slop
classifier, NOT a general AI-detector/Pangram: a frozen **style** embedding
(**StyleDistance** — bake-off winner over the content-matched pairs: 0.78
pairwise / 0.72 AUC, beating the mxbai semantic baseline) + a logistic head.
Both halves are stylebot mechanism now, split by dependency weight:
- `stylebot.classify` — the **runtime**: pure-Python dot product over the
  plain-JSON `head.json`. **Dep-free at import, always** (enforced by a test).
- `stylebot.classify_train` — the **trainer**: dataset assembly from
  `pairs.jsonl`, the leakage-safe by-POST split methodology, artifact I/O.
  Needs the **`stylebot[classifier]` extra** (scikit-learn/numpy/
  sentence-transformers); lazy imports keep the module importable without it.
  Generic CLI: `ai-style train-clf`.
livingthing keeps only *policy* (~100 lines): the backbone pin, the
free-positives selector (`blog_free_positives`), and blog path defaults
(`dan-style train-clf`, which delegates). The artifact (`head.json` + `meta.json`)
is committed at livingthing `_models/voice-clf/`. Wire it via
`ai-style eval --detector-model PATH` or `dan-style eval`;
`score = P(slop)` (composes with `mean_detector_score`), `p_dan` for reward use.
**Eval vs reward — the shared splits contract (2026-07-06):** one canonical
three-role partition (`stylebot.splits`, stdlib-only; `ai-style make-splits` →
the blog's committed `_training_pairs/splits.json`): frozen **eval** posts
(pinned, real-capture only, never embedded by the trainer), **styler** posts
(Phase 3), **detector** pool (the rest, hash-assigned so new posts flow in
stably). `train-clf --splits` / `dan-style train-clf` (automatic when the file
exists) fit on the detector pool, report a styler-posts holdout metric, and
record role counts + a **danger report** (dangerously-small-strata warnings) in
`meta.split`. **C is selected by nested group-CV** (`select_C`/`C_GRID`;
`--C` = explicit override — never sweep it against the printed metric). Judge +
eyeball stay the orthogonal anti-Goodhart guard (a split fixes leakage, not
over-optimisation). Pangram is only an optional one-shot cross-check. See
`_plans/eval-harness.md` "The detector decision".

**The active step is the experimental Phase-2 generation loop**, not a one-shot
paid run: per `--slop-strategy`, generate a small batch into a *scratch*
`--data-dir`, eyeball (`--report`/`--sample`), score with `ai-style eval`, promote
a winner into the real corpus. **Experiments run through `dan-style synth` too**
(`cd ~/Source/livingthing && uv run dan-style synth --data-dir /tmp/scratch
--slop-strategy X --limit 40`), not bare `ai-style synth` — the wrapper carries
the blog chunking/selection policy (merge into ~1.5k-char passages, quality>6,
heading context); the generic CLI's defaults are unmerged ~100-char fragments,
too short to judge style on. Then the corpus run is the same command minus
`--data-dir`, with `--limit N`. Phase 3/4 are data-/adapter-gated. See OVERVIEW.
