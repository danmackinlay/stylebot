# `ai-style synth` flag audit — the proposal, and what was decided

At the time of the audit `ai-style synth` exposed **54 click options** (47 from
the shared `synth_cli._OPTION_SPECS` table + 7 declared on `synth_cmd` itself,
plus the `FILES` argument, which is not counted here). The next-largest
subcommand has 19. This document classifies all 54 against evidence so the repo
owner can decide what to retire.

> `_OPTION_SPECS` was the spec table the shared surface was declared in at audit
> time. It has since been replaced by the `cli_options.opt()` bundles, where each
> field carries its own flag; the *count* and the classification below are
> unaffected, and the name is left as audited rather than back-edited.

The classification below is **as-audited** and left that way on purpose — it is
the evidence, and rewriting it to match today's surface would destroy the record
of what the evidence actually said. What was acted on is in Decisions.

## Decisions — 2026-08-10

One decision taken, one deferred; everything else in class C stands as an open
proposal.

- **Retired: the `gpt`/`local` direct-API preset path** — cluster 3, 4, 5, 10
  (`--generator`, `--gpt-model`, `--local-model`, `--context-window`). All 4169
  synthetic pairs were OpenRouter-served and every recorded `gen.context_window`
  came from the OpenRouter models registry, so the path had never produced a
  pair. Gone with them: the `PRESETS` tuple, `Arm.kind` and its branch in
  `build_generators`, the preset arms in `iter_arms`, `_resolve_windows`'
  manual-window fallback, the `and not generator_names` half of the
  `--max-workers` auto rule, and `generators.local_generator` (plus its
  `stylebot.synth` re-export). `openai_generator` **stays** — it is what
  `openrouter_generator` is built on. Dan: *"let us not bother with the local
  path if it adds complexity."*
  **Resume identity is untouched.** OpenRouter arms are named
  `openrouter/<model>` exactly as before, and `iter_arms` enumerated presets
  *before* OpenRouter models, so with no presets the arm list and its order are
  bit-identical. No cell re-keys; no paid corpus regenerates.
  Downstream: livingthing's `bin/train_targets.py` dropped `gpt_model` /
  `local_model` from its `synth_options(exclude=…)` tuple — `synth_options`
  validates exclude names at import, so `dan-style synth` would otherwise have
  failed to import. That was the only downstream consumer.
  `ai-style synth` is now a **50**-option command.
- **Kept: `--temperature` and `--top-p`** (items 7 and 8). Both are `None` on
  all 4169 pairs, so the audit filed them as retirement candidates — but their
  importance is *untested*, not disproven, and they were designed as
  recorded-but-unkeyed covariates precisely so sampling can be swept without
  re-keying. Dan's explicit call: not retiring on an absence of evidence.

## Decisions — 2026-08-13 (closes the audit)

The remaining class-C proposals were reviewed together and the file is now
**closed**: nothing in it is an open question. `ai-style synth` is a
**49**-option command.

- **Retired: `--sort`, `--report-max-rows`, `--all`, `--tag`** (items 1, 2, 15,
  16). All four are non-`synth_key` by construction, and each field survives
  as a library-only knob with its current value hardcoded, so **no cell
  re-keys and no paid corpus regenerates**:
  - `--sort` → `ChunkPolicy.sort_name = "none"`. Arms are assigned by content
    hash, so target *order* cannot reach the corpus at all. (The library-only
    field was deleted outright 2026-09-03, no caller having appeared — D43.)
  - `--report-max-rows` → `InspectOpts.report_max_rows = 2000`.
  - `--all` → gone entirely; `--max-level` already relaxes the selector and
    `FILES` bypasses it, which is what the docstring now says.
  - `--tag` → `run_synth(extra_tags=…)` stays. The one tag on the live corpus
    (`livingthing`) was always set through the library parameter, never the flag.
- **Kept: `--capture-reasoning`** (item 6). The audit's own risk note carries
  it: reading what a model deliberated has no substitute, and the tokens are
  paid for whether or not the trace is saved. "Never run yet" is not evidence
  against a diagnostic.
- **Kept: `--splits` / `--role` / `--sample-frac` / `--sample-salt`** (items
  11–14). Retiring this cluster is a bet that measurement runs will never need
  a substrate disjoint from the detector's training pool. That bet reads worse
  now than in August: the masking work (D33) needs the voice classifier re-fit
  on masked text and then measured, which is exactly the `--role styler` case.
- **Deferred: `--assign-seed`** (item 9). A `synth_key` input — it decides
  which arm a target gets, hence the generator name in the hash — so it falls
  outside the "retire only non-key flags" rule this pass worked under.

## Summary (as audited)

| Class | Count | What it means |
| --- | ---: | --- |
| **A — Load-bearing** | 14 | Swept across runs, folded into `synth_key`, recorded as a `meta.gen` covariate with ≥2 observed values, or typed in the blog RUNBOOK's command lines. Removing or re-defaulting these breaks resume or loses analysis covariates. |
| **B — Policy constants** | 24 | Set once by the blog wrapper (`synth_options` override or `exclude`) or left at their default and never varied. Real, but the value is pinned; the *flag* is optional, the *value* often is not. |
| **C — Retirement candidates** | 16 | No usage found in either repo beyond the option definition and its own tests. **8 have since been retired** and 8 were reviewed and kept or deferred (see Decisions). None remains an open proposal. |
| **Total** | **54** | 49 today. |

Two cross-cutting caveats before the tables:

- "Load-bearing" here covers both **experiment axes** (swept, keyed, or
  recorded) and **operator-critical** flags (`--dry-run`, `--report`,
  `--data-dir`) that are not axes but are typed on every run. They are marked.
- Several B and C entries are **`synth_key` inputs** (marked 🔑). Retiring the
  *flag* is safe; changing the *value* it currently supplies is not — it
  re-keys cells and silently regenerates paid corpus. Any retirement must
  hardcode the current value, not a "nicer" one.

## Method — where the evidence comes from

1. **`synth_key` inputs** (`stylebot/synth.py::_synth_key`): the hash is
   `(generator name, strategy, reasoning_effort, prompt_id, context, target
   text[, replicate])`. That transitively pulls in every chunking flag (they
   change the target text), `--heading-context` / `--context-dropout` (they
   change `context`), `--slop-system-file` (via `prompt_id`) and
   `--assign-seed` (via which arm — and so which generator name — a target
   gets). Deliberately **excluded**: `--temperature`, `--top-p`, and all
   session/position knobs.
2. **Recorded covariates**: `pairs.GEN_META_KEYS` / `GEN_FACET_KEYS`, built in
   `generators.py` and extended in `synth.py`.
3. **The live corpus** — `~/Source/livingthing/_training_pairs/pairs.jsonl`,
   read read-only: **4624 records, 4169 synthetic**, aggregated by distinct
   value per covariate. What has *actually* varied across two years of runs is
   the strongest evidence available.
4. **The blog RUNBOOK** — `~/Source/livingthing/_training_pairs/RUNBOOK.md`,
   the ordered what-do-I-type pipeline. It types **11 distinct synth flags**.
5. **The blog policy pin** — `livingthing/src/livingthing/training_targets.py`
   and `bin/train_targets.py`, which call `synth_options(exclude=…, **overrides)`
   with a 7-name exclude list and 13 overrides.
6. **This repo's `_plans/*.md` and `README.md`.** Note: `README.md` mentions
   **no synth flag at all**; all doc evidence is from `_plans/`.

`~/Source/livingthing` exists and was read (read-only, unmodified).

### What the corpus says varied

| Covariate | Distinct values in 4169 synthetic pairs |
| --- | --- |
| `gen.model` | 9 (incl. 3 since retired from the roster) |
| `gen.prompt_label` / `meta.slop_strategy` | 6 (polish 943, measured 896, casual 783, engaging 567, linkedin 520, verbose 460) |
| `gen.reasoning_effort` | 2 — `off` 2632, `high` 1537 |
| `gen.max_tokens` | 2 — `8192` (4130), `16384` (39) |
| `gen.replicate` | `verbose1` on 460 pairs, unset on 3709 |
| `gen.session_id` | 190 sessions, 99 distinct turn indices |
| `meta.context_mode` | `immediate` 3212, unset 957 (`--context-dropout` working) |
| `gen.temperature`, `gen.top_p` | **1 each — always `None`** |
| `gen.provider_sort` | **1 — always `throughput`** (the default) |
| `gen.context_window` | 5, **all resolved from the OpenRouter registry** — never from `--context-window` (which is why the flag went; the registry is now the only source) |
| `meta.merge_max_chars`, `meta.chunk_mode` | **`None` on all 4624** — the v2 scale curriculum has not been run |
| `meta.tags` | **1 — `["synthetic","paraphrase","livingthing"]` on every pair** |

## A — Load-bearing (14)

Do not touch without a migration plan.

| Flag | Why | 🔑 |
| --- | --- | :-: |
| `--openrouter-model` | The generator roster. 9 models observed; feeds the generator name. RUNBOOK types it. | 🔑 |
| `--slop-strategy` | 6 strategies swept; `meta.slop_strategy` is the primary eval facet. RUNBOOK types it. | 🔑 |
| `--reasoning-effort` | 2 values swept (`off`/`high`); recorded in `meta.gen`; RUNBOOK types it. | 🔑 |
| `--replicate` | 460 pairs carry `verbose1`. Its whole purpose is minting new cells without colliding. | 🔑 |
| `--max-tokens` | Actually varied (8192 → 16384 for 39 pairs) and recorded in `meta.gen`. | |
| `--session-turns` | The session experiment: 190 sessions, 99 turn indices, `session_turn`/`window_fill` covariates. RUNBOOK types `--session-turns 32`. | |
| `--merge-max-chars` | The v2 scale-curriculum axis; RUNBOOK sweeps `for n in 1500 3000 4500`. **Caveat: not yet exercised** — `meta.merge_max_chars` is `None` corpus-wide, and RUNBOOK §6b is labelled "NOT YET RUN". Load-bearing by design, not yet by evidence. | 🔑 |
| `--limit` | RUNBOOK types `--limit 40`, `--limit 3000`. Cost control on every real run. | |
| `--skip-covered` | RUNBOOK types it; it is what stops a re-key epoch double-weighting covered passages. | |
| `--dry-run` | Operator-critical. RUNBOOK types it before every paid run to vet resume math. | |
| `--report` | Operator-critical. RUNBOOK types it to vet target selection pre-generation. | |
| `--sample` | Operator-critical. RUNBOOK offers it as the `--report` alternative. | |
| `--data-dir` | Structural. Which corpus grows; RUNBOOK types it for the side-experiment dir. | |
| `--blog-root` | Structural. One of the two mutually exclusive input modes. | |

## B — Policy constants (24)

Set once and never varied. Retiring the flag is a real simplification, but the
**value must be preserved verbatim** where marked 🔑.

### Pinned by the blog wrapper's `synth_options(**overrides)` (8)

| Flag | Pinned to | 🔑 |
| --- | --- | :-: |
| `--min-chars` | `MIN_CHARS = 200` | 🔑 |
| `--merge/--no-merge` | `MERGE = True` | 🔑 |
| `--drop-list-items/--keep-list-items` | `DROP_LIST_ITEMS = True` | 🔑 |
| `--heading-context` | `HEADING_CONTEXT = "immediate"` | 🔑 |
| `--context-dropout` | `CONTEXT_DROPOUT = 0.1` | 🔑 |
| `--cross-heading/--no-cross-heading` | `CROSS_HEADING = True` | 🔑 |
| `--max-transform-sim` | `MAX_TRANSFORM_SIM = 0.95` — the identity gate | |
| `--max-length-ratio` | `MAX_LENGTH_RATIO = 3.0` — the inflation gate | |

### Excluded by the blog wrapper — never exposed there at all (5)

The blog passes `ignore_markers` / `stop_at_headers` **directly to
`iter_targets`** as library arguments, and lets the other three sit at the
stylebot default.

| Flag | Effective value | 🔑 |
| --- | --- | :-: |
| `--max-chars` | stylebot default 8000 | 🔑 |
| `--prose-only/--no-prose-only` | always `True` | 🔑 |
| `--ignore-marker` | `STUB_MARKER` (`🚧TODO🚧`), passed as a library arg | 🔑 |
| `--stop-at-header` | `STOP_HEADERS` (`## Incoming`), passed as a library arg | 🔑 |
| `--drop-link-dumps/--keep-link-dumps` | always `True` | 🔑 |

### Left at default, never varied (8)

| Flag | Evidence | Note |
| --- | --- | --- |
| `--session-max-tokens` | Never overridden by the blog; docstrings call it "THE session depth control". | Highest-leverage never-varied knob here; cost grows ~quadratically in it. |
| `--provider-sort` | `gen.provider_sort` is `throughput` on all 4169 pairs. | Recorded covariate, so it survives as analysis metadata even unvaried. |
| `--sticky-provider` | 1 `_plans` mention; never flipped. | Confound control for the session experiment — the reason it exists. |
| `--prompt-cache` | 1 `_plans` mention, 1 test. | Economics only; `cached_tokens` is recorded regardless. |
| `--timeout` | 2 tests; no doc/RUNBOOK invocation. | The operational escape hatch when an upstream stalls. |
| `--max-workers` | RUNBOOK prose (not a command line); the `0 = auto` rule is tested. | Auto rule covers both real cases. |
| `--per-generator` | 3 `_plans` mentions; no run evidence. | Documented fully-crossed design switch (n× cost). |
| `--slop-system-file` | 4 `_plans` mentions; corpus shows only the **6 registry** `prompt_id`s, `prompt_version` always `1`. | **Watch list**: never exercised, but it is the CLAUDE.md architecture seam that keeps blog-specific prompt text out of stylebot. Would be a C on evidence alone; kept as B on architecture. |

### Command-level selector knobs (3)

| Flag | Evidence |
| --- | --- |
| `--field` | Default `automation` already matches the blog exactly (CLAUDE.md: "no retarget needed"). Never varied — but it is the public-tooling knob for a non-Dan frontmatter. |
| `--max-level` | Default `0`; the blog wrapper uses its own `make_selector(min_quality)` instead. |
| `--glob` | Default `**/*.qmd`; never varied. |

## C — Retirement candidates (16)

One line of rationale and the risk of removing each. "No usage found" is the
finding; none of these is provably *unused* — absence of evidence in two repos
is the strongest claim available.

The **Verdict** column records what was decided (2026-08-10); blank = still an
open proposal, nothing done.

| # | Flag | Rationale | Risk | Verdict |
| --: | --- | --- | --- | --- |
| 1 | `--sort` | No occurrence in either repo. Arm assignment is by content hash, so target *order* cannot change the corpus. | **Low** — affects only which chunks `--limit` takes and report row order. | **RETIRED** 2026-08-13 |
| 2 | `--report-max-rows` | No occurrence; the 2000 default has never been raised or zeroed. | **Low** — a huge selection would render every row. | **RETIRED** 2026-08-13 |
| 3 | `--generator` | The `gpt`/`local` direct-API preset path. **All 4169 synthetic pairs were OpenRouter-served**; no corpus pair has ever come from a preset. | **Medium** — with 4 and 5, deletes the only non-OpenRouter generation path. | **RETIRED** 2026-08-10 |
| 4 | `--gpt-model` | Excluded by the blog wrapper; only test hits. Dead unless `--generator gpt` returns. | **Medium** — cluster with 3. | **RETIRED** 2026-08-10 |
| 5 | `--local-model` | **Zero occurrences anywhere in either repo** outside `_OPTION_SPECS`. | **Medium** — cluster with 3; local-model experiments are a live interest elsewhere in the blog. | **RETIRED** 2026-08-10 |
| 6 | `--capture-reasoning` | No `reasoning.jsonl` exists in the blog data-dir — never run against the real corpus. | **Low-Medium** — the only way to read what a model deliberated; costs nothing extra when on. | **KEPT** — no substitute for the diagnostic |
| 7 | `--temperature` | `None` on all 4169 pairs. Never varied. | **Medium** — designed as a recorded-but-unkeyed covariate precisely so sampling can be swept without re-keying; removing forecloses that. | **KEPT** — importance untested, not disproven |
| 8 | `--top-p` | `None` on all 4169 pairs. Never varied. | **Medium** — cluster with 7. | **KEPT** — cluster with 7 |
| 9 | `--assign-seed` | No occurrence in either repo. `--replicate` covers the fresh-draw story and **is** used (460 pairs). | **Low-Medium** — loses re-randomizing arm assignment without a new replicate label. | **DEFERRED** — a `synth_key` input |
| 10 | `--context-window` | Every pair's `context_window` came from the OpenRouter registry (5 distinct values); the flag only serves preset/injected generators. | **Low** — but coupled to cluster 3-5: it exists *for* the preset path. | **RETIRED** 2026-08-10 |
| 11 | `--splits` | Substrate block: no synth-side occurrence in either repo. (`--splits` appears only for `train-clf`.) | **Medium** — cluster with 12-14. | **KEPT** — cluster |
| 12 | `--role` | No occurrence. The designed way to keep measurement runs disjoint from the detector's training pool. | **Medium** — the splits contract is live for `train-clf`/`eval`; only the synth-side filter is unused. | **KEPT** — cluster |
| 13 | `--sample-frac` | No occurrence. The growth-stable alternative to `--limit`, which is what actually gets typed. | **Medium** — cluster with 11-12. | **KEPT** — cluster |
| 14 | `--sample-salt` | No occurrence; inert without `--sample-frac`. | **Low** — cluster with 13. | **KEPT** — cluster |
| 15 | `--all` | No occurrence. `--field`/`--max-level` can already relax the selector, and `FILES` bypasses it entirely. | **Low**. | **RETIRED** 2026-08-13 |
| 16 | `--tag` | Every synthetic pair carries exactly `["synthetic","paraphrase","livingthing"]`, and `livingthing` comes from `extra_tags=` in the **library** call, not this flag. | **Low** — the library parameter is what is actually used and would stay. | **RETIRED** 2026-08-13 |

### If you want the biggest single win

Items **3, 4, 5, 10** are one decision, not four: retiring the `gpt`/`local`
preset path removes 4 flags, the `PRESETS` tuple, two generator factories'
CLI reachability, and the `Arm.kind` branch in `build_generators` — and the
corpus says it has never produced a pair. **Taken, 2026-08-10 — see Decisions.**
(In the event `local_generator` was deleted outright and `openai_generator`
kept, since `openrouter_generator` is built on it.) Items **11-14** are a
second one-decision cluster (5 flags with `--sample-salt`). **Declined,
2026-08-13** — see Decisions: it is a bet that measurement runs will never
need a disjoint substrate, and the masking work is about to need one.

## Known documentation drift spotted in passing

Not part of the classification, but worth recording:

- `phase-1-pair-capture.md` and `heading-context.md` referenced
  `--no-heading-context`, which **does not exist on synth**
  (`--heading-context` is a `click.Choice`); they meant `ai-style log`. Both
  files were retired 2026-08-10, which resolved this drift by deletion.
- The blog RUNBOOK:162 audits with `--openrouter-model qwen/qwen3-32b`, a model
  commented out of the roster in `training_targets.py:109`.
- RUNBOOK:190 calls the canonical corpus run "deliberately flagless so policy
  carries it", while the line beneath carries `--limit 3000 --session-turns 32`.
- RUNBOOK:219 claims every pair is stamped with `meta.merge_max_chars` and
  `meta.chunk_mode`; both are `None` on all 4624 records.

Those live in the blog repo (except the first) and were not modified.
