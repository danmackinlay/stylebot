# Design decisions

The *why* behind each non-obvious choice, so it can be re-evaluated without
re-reading the planning conversation — and the record of what was believed and
got wrong. Git preserves what changed; it does not preserve why an alternative
was rejected.

**Shape (P3):** decision → alternatives (one line each) → deciding constraint →
re-evaluate-if, ≤ 30 lines. A number is a **citation into**
[`findings.md`](findings.md), never a copy. A superseded entry collapses to
three lines. A `P…` entry is a *proposal* in the same shape, followed by its
verdict. This file and `findings.md` are the two that keep dated numbers;
everywhere else a count is a claim about the present that git cannot keep true
(D52). **D1–D33 live in [`decisions-archive.md`](decisions-archive.md)** and
keep their numbers — the second index below is how you find them.

## Index — this file

| # | title | status |
|---|---|---|
| D34 | a post's role is keyed on an alias-resolved identity | amended 2026-09-04 |
| D35 | the styler backend is a library, and the manifest configures it | live |
| D36 | a pair record carries its own provenance; uniqueness is a file property | live |
| D37 | the corpus-stability guard belongs to whoever owns the corpus | live |
| D38 | no `scripts/` carve-out — migrations are a hidden `ai-style data` group | live |
| D39 | a score carries the instrument that produced it | live |
| D40 | the eval interval clusters by post; the headline is the real facet | live |
| D41 | the automation cap applies to synthetic pairs only | live |
| D42 | the eval stratum is versioned and add-only | live |
| D43 | a passage is a stretch of prose, not a string; weight is replication | live |
| D44 | the editor asks the sidecar where the prose is | live |
| D45 | one corpus location, resolved once | live |
| D46 | the synth walker's blog exclusions stay pinned | live |
| D47 | two error classes for the whole package, translated once at the group | live |
| D48 | `Styler` is a Protocol; it types the call, not the tokens | live |
| D49 | asyncio is deferred; the rest of the eager import block stays | live |
| P1 | move `openai` to an `[api]` extra | **rejected** 2026-09-03 |
| D50 | a corpus is a directory of shards, by producer then by month | live |
| D51 | `git filter-repo` on the corpus is not worth a force-push | live |
| D52 | documentation carries no mutable operational state | live |
| D53 | the orphan prune is built and declined; the eval prune is applied | live, one half open |
| D54 | a document-scale verdict is a mean over serving-scale chunks | live |
| D55 | the scoring pass runs in a child | **superseded** by D59 |
| D56 | `asyncio` first; threading must stringently justify itself | live |
| D57 | the detector is batched out of the pool; the pool kept for the judge | **superseded** by D59 |
| D58 | size budgets are a test with a shrinking allowlist | live |
| D59 | one mechanism in the scoring path — the judge awaits, the rest batches | live |
| D60 | a rewrite is a writer — it takes the corpus lock | live |
| D61 | the executables package holds no libraries | live |
| D62 | `stylebot.eval` keeps a lazy re-export map after the split | live |
| D63 | orphan score rows are pruned where they are read | live |
| D64 | `STYLE_SYSTEM` was amended once, 2026-08-04, and the corpus re-stamped | live |
| D65 | `--include-real` defaults off — the two strata invent different things | live |
| D66 | the `anthropic/` arms bill through OpenRouter BYOK | live |

## Index — [`decisions-archive.md`](decisions-archive.md) (D1–D33)

| # | title | status |
|---|---|---|
| D1 | train on Tinker, serve on Fireworks (fallback: local MLX → Modal) | serving half superseded by D19 |
| D2 | Qwen3 8B as the primary base | superseded 2026-07-21 → `Qwen/Qwen3.5-9B` |
| D3 | source-agnostic framing ("Dan-ifier", not "de-Claude-er") | live |
| D4 | raw markdown only; defer mask-preservation | live; re-evaluation is the deferred symmetric-masking experiment |
| D5 | Tinker is *the* trainer, SFT v1 through Phase-7 preference/RL | live |
| D6 | Unsloth — not needed; managed Tinker removes the rationale | live |
| D7 | train a Dan-vs-slop classifier directly (not Pangram, not distillation) | live |
| D8 | DiffusionGemma — speculative, not v1 | live |
| D9 | slop comes from asking for polish, never from asking for slop | live |
| D10 | real AI prose is not written at high reasoning effort | live |
| D11 | `synth_key` is a cell identity, not a completion marker | live |
| D12 | rendezvous hashing, not modulo, for generator-arm assignment | live |
| D13 | session folding in keys — right, then wrong | live (dropped 2026-07-21) |
| D14 | heading context is prepended to content, not a fourth message | live |
| D15 | the chunk scale was set an order of magnitude too high | live |
| D16 | synthetic-pair methods planned and never built (M2/M3/M5/M6) | live |
| D17 | val split by POST, not by row | live |
| D18 | keep near-copy pairs for the styler, drop them for the detector | live |
| D19 | serving v1 is the Tinker sampling client, not Fireworks | live |
| D20 | in the editor the human picks; the CLI's guard does not | live |
| D21 | the `gpt`/`local` direct-API preset path, retired | live |
| D22 | register conditioning, parked | live |
| D23 | `--seed` reached the val split but not the epoch shuffle | fixed 2026-08-09 |
| D24 | no dotenv loader; direnv is the environment mechanism | live |
| D25 | the tinker SDK's TLS trust is patched, not configured | retired 2026-08-11 (tinker ≥ 0.25.0) |
| D26 | no shared markdown package (yet) — `stylebot.markdown`, not `mdcore` | live |
| D27 | exclusion for blocks, the anchor guard for inline | live; amended 2026-09-03 (packing) |
| D28 | a conformance fixture pins the second implementation | **superseded** 2026-09-03 (W3) |
| D29 | policy differences are parameters — `block_only`, `raw_html`, `div_is_prose` | live |
| D30 | stylebot is livingthing's *optional* `styler` extra | **superseded** 2026-09-03 (W3) |
| — | deferred experiment: symmetric masking (D4's re-evaluation) | untested |
| D31 | stylebot merges the adapter; it does not quantize or bake templates | live |
| D32 | the anchor guard is symmetric, but its halves use different rules | amended 2026-08-17 |
| D33 | masking goes in everywhere, with a dropout fraction | live |

---

## D34: a post's role is keyed on an alias-resolved identity, and synth refuses the eval stratum unasked

**Decision** (2026-09-03; amended 2026-09-04, F3). `splits.json` carries an
optional `aliases` map, `{path the post has now: path its pairs are already keyed
under}` — new → old, the opposite of what the name suggests — and every path
passes `splits.canonical_source` before anything reads a role. Both sides of the
pinned test resolve: `role_of` compares a canonical query against
`{canonical(p) for p in eval_posts}`, and `extend_eval` stores the canonical
spelling, so a post added under its new path is pinned by identity. `synth`
refuses a target resolving to `eval` — reading the corpus's own `splits.json`
when `--splits` is absent, `--role eval` the only way past — while `log save`
notices without refusing, a capture being legitimate data. A pinned post on disk
under no known spelling is an error, not a warning: `check_splits(blog_root=…)`
marks it `ERROR_PREFIX`, `check-splits` exits non-zero, and
`require_pinned_posts` raises it from `train`/`train-clf --blog-root`, since
nothing forces `check-splits` first. Numbers: `findings.md` "The alias seam".

**Why the direction is inverted.** Roles come from `meta.source` as recorded at
capture time, so the corpus is keyed on the OLD path; aliasing old → new re-hashes
every existing pair into whatever the new path draws, silently for the roles
nothing pins. New → old moves nothing:
`test_aliases_need_no_migration_of_existing_pairs` asserts it over 50 posts.

**Rejected**: a frontmatter `id:` (mint it across ~1,700 posts, and `role_of`, a
per-pair string function, becomes a disk read); `make-splits --force` (un-freezes
eval to repair a symptom); bumping `schema_version` (`aliases` is additive, and a
bump makes every older checkout refuse the file); requiring `--blog-root`.

**Re-evaluate if**: the blog gains real post ids (the map becomes the migration
record and `role_of` keys on it), or `aliases` outgrows hand-editing.

## D35: the styler backend is a library (`stylebot.backends`), and the manifest is what configures it

**Decision (2026-09-03, W1)**: `BackendOpts`, `build_styler`, the token-scaling
rule and the serving-diet rule live in `stylebot/backends.py`, a plain library
raising builtins; `bin/ai_style.py` keeps only the translation to exit codes, and
`dan-style run/style/serve` import the same module. Both front-ends take a
`--manifest`, and `opts_from_manifest` fills checkpoint, base model, renderer and
chunk diet from it, explicit flags winning.

**The defect it repairs.** The surface sat inside `bin/ai_style.py`, unimportable
and raising `click` exceptions, so the blog wrote its own `build_styler`. The
copy passed no `max_tokens`, so every blog rewrite of a full-length chunk was cut
off at the 2000-token floor while the promoted run served a 4500-char diet — and
the blog's summary omitted `n_truncated`, so nothing said so. A copy is
duplication that loses the argument the original was making (D61).

**Alternatives.** *Grow `infer`* — the backend surface is needed at
flag-declaration time and `infer` is the expensive import deferred on purpose.
*Re-export from `bin/`* — an alias leaves the old path in use. *Keep the blog's
copy and merely pass `max_tokens`* — fixes the bug, keeps the mechanism; the
flag-superset assertion in `test_dan_style_cli.py` is the durable half. *Take
`--detector-model` from the shared bundle* — click converts defaults through
`click.Path(exists=True)`, failing every invocation without the artifact.

**Deciding constraint**: `inline_headings` is tri-state because a manifest can
only supply a default an explicit flag beats, and `bool` cannot tell
`--no-inline-headings` from silence. **Re-evaluate if** a third front-end
appears, or manifests grow a serving section — `opts_from_manifest` should read
it rather than infer the diet from training-time histograms.

## D36: a pair record carries its own provenance, and uniqueness is a property of the file

**Decision (2026-09-03, W8)**: every record stamps `meta.schema_version`,
`meta.system_sha` and a five-key `meta.provenance` (`blog_sha`, `blog_dirty`,
`stylebot_sha`, `policy_digest`, `source_automation`) from both producers.
`record_id` uniqueness moves into `validate_pairs_file`, where the whole file is
visible; `dedupe_pairs_file` keeps the **first** occurrence, and `PairsWriter`
makes appends safe under two concurrent writers.

**Three sub-decisions worth naming.** *The chunk policy rides in on the `Target`*
— `iter_targets` is the one construction site and already holds the composed
`ChunkPolicy`; `policy_digest` takes the object, and a `div_is_prose` predicate
hashes as `module.qualname` because `repr()` carries a memory address. *First
writer wins* — the duplicates were two draws for one cell, and the first is what
every committed `scores.jsonl` row already joined to. *`source_automation` is
read at cut time*, because the corpus is frozen and the post is not.

**Alternatives.** A lock around the whole run — serialises two synth runs.
Re-read `seen` per write — O(corpus); `PairsWriter` tail-reads from a byte offset
under the same lock. Key `synth_key` on the provenance — resume restarts every
commit. A closed whitelist of `meta` keys — the schema is open by design, so it
turns every experiment's covariate into a validation failure. Bump
`schema_version` — absent *is* version 0.

**Deciding constraint**: `validate_pair_record` had been per-record since Phase
1, so the one defect it structurally could not see is the one that reached the
corpus — duplicate `record_id`s from two runs that each loaded
`existing_synth_keys` before the other appended; eval resumes score the first and
skip the second while training takes both. **Re-evaluate if** the corpus is
sharded (D50 — "first" then needs an order between shards).

## D37: the corpus-stability guard belongs to whoever owns the corpus

**Decision (2026-09-03, W5)**: `test_corpus_stability.py` moves to livingthing's
suite, and `python-dateutil` — in stylebot's dev group for no reason but that
test's `sys.path` hop into the blog — goes with it. **stylebot depends on no
checkout it does not own**: not in `src/`, not in `tests/`, not in its groups.

**What the guard needs, and who has it.** The corpus (which by rule never enters
the public tooling repo), the chunk policy (blog policy, in `training_targets`),
and a git pin into blog history to freeze the prose. stylebot had none of the
three, so the test probed `../livingthing`, patched `sys.path` and carried a
transitive dependency to make the import work. In the blog it is two imports.

**Alternatives.** *A stylebot-side guard reading policy from the newest manifest
and skipping when no corpus is configured* — on every machine but Dan's it skips,
and a green skip reads as "measured, fine". What stylebot keeps instead is what
it can own: `test_markdown.py` and `test_protection_parity.py` pin the
segmenter's *behaviour*; the floor pins what that behaviour cost, a question
about one corpus. *Vendor a sample* — the corpus never enters this repo.

**Deciding constraint**: the failure this already caused — dropping the unused
`python-dateutil` runtime dependency also dropped it from the dev environment and
the harness silently stopped measuring for hours. Moving it also exposed that
both arms had been mis-measured: each now gets its own git pin, its own policy
and its own budget cut (`findings.md` "Corpus-stability floor").
**Re-evaluate if** a third chunk regime lands, or the corpus is sharded.

## D38: no `scripts/` carve-out — migrations are a hidden `ai-style data` group

**Decision (2026-09-03, W5)**: `scripts/` is deleted. Its one-off migrations
become typed functions in `stylebot.migrations` behind an `ai-style data` click
group declared `hidden=True`. "One entry point, subcommands" has no exceptions.

**What the carve-out was protecting, and what it cost.** The stated reason was
the `--help` tree — a migration is run once per schema change and does not belong
beside the pipeline — and `hidden=True` answers that exactly. The directory
bought nothing else: `scripts/` is not in the wheel, so `capture.iter_archives`
printed a fix-it command naming a file anyone who *installed* stylebot did not
have; the blog's RUNBOOK invoked them by a hardcoded `../stylebot/scripts/…`
path, in the repo whose OVERVIEW says stylebot never hardcodes a blog location
(D37 is the same arrow bent the other way); and "each carries a test" was written
in `CLAUDE.md` while only one of the two had one, because a directory outside the
package is a directory outside the checks.

**Alternatives.** *Move them to the blog's `scripts/`* — true today, not what the
code says: `migrate-captures` converts *stylebot's own* archive format and is
named by *stylebot's* error message. *Plain visible subcommands* — `ai-style
--help` is a phase map, and spent migrations read as pipeline steps there. *A
`stylebot-migrate` console script* — the `ai-style-log` mistake, one release on.

**Deciding constraint**: an error message that names a command must name a
command that exists. Everything else follows from putting the migrations where
the error can reach them.

**Re-evaluate if**: nothing. A migration is deleted once the only corpus that
exists has been through it — function, command and test together.

## D39: a score carries the instrument that produced it, and two instruments never share a mean

**Decision (2026-09-03, W10)**: every detector result carries
`DETECTOR_IDENTITY_KEYS` — `name`, `embed_model`, `git_sha`, `trained_at`,
`n_pairs` — read from the artifact's `meta.json` and stamped on every
`scores.jsonl` row. Scoring refuses to append to a file whose rows name a
different head, and `compare_scores` refuses a verdict spanning two heads; both
have an override flag, and the second flags the result `cross_detector`.

**The defect.** One classifier is the best-of-N reranker, the do-no-harm guard
*and* the verdict metric, and it is refit every time the corpus grows. Run 4a's
promotion note admits the detector was refit the same day as the verdict it
produced. Rows recorded `name` and `embed_model` only — constant across every
refit this project will ever do; they identified the *recipe*, not the artifact.
A refit alone moves the metric by about two thirds of what that promotion claimed
(`findings.md` "eval_v2 — applied 2026-09-03").

**Why the refusal is at the resume, not only at the compare.** `scores.jsonl` is
append-only and resumes by pair id, which says nothing about what scored it. A
refit head plus a resumed run silently yields one file whose means blend two
instruments — worse than a cross-file comparison, because nothing downstream can
see it happened. *Alternatives*: a header line records only the first writer,
precisely the writer a refit run is not; probing the head with a canary string
costs an embed per guard call, on a path meant to be free.

**Deciding constraint**: an unconfigured (`null`) detector must never trip the
guard — a keyless re-run adds no detector scores, so it cannot blend anything.
**Re-evaluate if** the detector gains a calibration layer versioned separately
from the head, or artifacts carry a semantic version.

## D40: the eval interval clusters by post, and the headline is the real facet

**Decision (2026-09-03, W10)**: `eval.paired_delta(clusters=…)` computes the
paired difference with a **cluster bootstrap over POSTs**, and `compare_scores`
clusters by `meta.source` by default. `summarize_scores` facets `by_provenance`
and names `headline_facet: "real"`; both CLIs print the real facet first.

**The measurement that forces it** (`findings.md` "Eval-stratum composition"):
the frozen stratum was six posts, three quarters of its pairs from one of them,
and 60% our own paraphrase generators. Run 4's verdict was "127 chunks, paired",
with a normal-approximation interval treating those chunks as independent.
Chunks inside a post share topic, register and whatever the generator did to that
post, so the effective n is nearer the post count; and an unfaceted headline is
substantially a measurement of this month's slop roster.

**Alternatives.** *A mixed model or a cluster-robust SE* — the verdict path must
import with no extras (`_plans/eval-harness.md`), and a random-effects fit would
drag statsmodels into a module the trainer imports; a percentile bootstrap over
resampled posts is a dozen lines. *Cluster by capture* — captures nest inside
posts, so the finer level leaves the between-post correlation uncounted, which is
the correlation that exists.

**What this does NOT fix.** Widening the interval is honesty, not power: six
clusters is six clusters, and the run-4 arms were already indistinguishable at
the narrow interval. The fix for power is D42.

**Re-evaluate if**: the stratum passes ~30 posts, where a BCa correction starts
to earn its complexity, or a verdict needs more than one signal jointly.

## D41: the automation cap applies to synthetic pairs only — a capture is *supposed* to sit on an AI-touched post

**Decision (2026-09-03, W10)**: `classify_train.assemble_dataset` grows a
caller-supplied `pair_ok` hook, and `dan-style train-clf` passes one that drops
**synthetic** pairs whose source post's frontmatter `automation` exceeds 0
(`--max-synthetic-automation`, default 0). Real captures are never dropped.

**Why the QA list's version was wrong.** It said "restrict detector targets to
`automation: 0`", on the reasonable ground that the pool holds pairs from
AI-touched posts. Measured (`findings.md` "Eval-stratum composition"), 94% of the
detector pool's *real* pairs sit on `automation: 1/2` posts, so the cap applied
to everything would leave the honest `real` facet — the headline metric of the
whole classifier — with eleven pairs.

The asymmetry is the definition, not an accident. A synthetic pair's "Dan" side
is the post's existing prose taken as ground truth, so on an AI-touched post the
detector is taught that AI-assisted prose *is* the author: the discrimination it
exists to make, inverted. A *capture* is Dan editing an AI-touched draft — the
post carries `automation: 1/2` **because** the capture happened.

**Alternatives.** *Read `meta.provenance.source_automation`* — D36 stamps it, but
almost none of the corpus carries it. *Hardcode the gate in `classify_train`* —
`automation` is a blog convention, and selection is a user-supplied policy.
**Deciding constraint**: an unreadable post keeps its pairs; a file that cannot
be parsed is not evidence of contamination. **Re-evaluate if**
`source_automation` reaches most of the corpus, or the real stratum grows.

## D42: the eval stratum is versioned and add-only, and a manifest pins the partition it held out

**Decision (2026-09-03, W10)**: `splits.json` carries `eval_version` and grows
only through `splits.extend_eval`, which forms `eval_v(n+1) = eval_v(n) ∪ posts`,
bumps the version and appends an `eval_history` entry. Removing a pinned post is
not expressible. Run manifests gain `data.splits_sha256`, `data.eval_version`,
`data.record_ids_sha256`, `data.selector_constants`, `dependencies` and
`base_model_revision`; verdicts are written under `runs/<run_id>/`.

**Why add-only is the whole safety property.** Growing the stratum moves posts
*out of* the training pools and *into* the frozen one: prose leaves training, it
never enters it. Shrinking would retroactively license training on prose an
existing verdict was computed over — D34's leak, arrived at deliberately. Hence a
separate subcommand rather than a flag on `make-splits`, whose `--force`
un-freezes the partition: one command carrying both would sit "never unfreeze
eval" beside the single flag that does.

**Applied 2026-09-03** with Dan's approval, as a *sized* v2: a maximal extension
would have taken every real pair from both other roles, so six styler-role
capture posts were added instead (`findings.md` "Eval-stratum composition",
"eval_v2 — applied 2026-09-03"). **The manifest needed four more pins** because
two runs of "the same" recipe could agree on every recorded field and train on
different data — `pairs_sha256` hashes the FILE while role filter, selector and
multiplicity hook all run after it.

**Alternatives.** *`eval_version` as a new `schema_version`* — the file's shape
is unchanged; what versions is the stratum. *Resolve `base_model_revision` at
manifest time* — a network call inside `--dry-run`, the one free offline path.
**Re-evaluate if** the corpus gains enough real captures to feed all three roles,
at which point the sizing argument dissolves and the maximal cut is right.

## D43: a passage is a stretch of prose, not a string — and weight is replication

**Decision (2026-09-03, W9)**: the multiplicity hook groups by **source post +
character-span containment** (`train.passage_groups`), not by exact target body,
and is renamed `per_target` → `per_passage` to say so. A second hook,
`pair_weight`, stamps `meta.weight` and is honoured as **integer replication of
the train rows**. Manifests record the shape under `data.chunk_scale`, printed by
one renderer both `--dry-run` shells share; the dead `sort_key` hook is deleted.

**The measurement that forced it** (`findings.md` "Effective passage weight"):
the corpus is cut at four budgets over the same prose, so exact-body grouping had
almost nothing to group and a manifest saying `num_epochs: 1` describes about
four gradient steps per stretch of prose, unevenly. It silently re-weights the
axis that matters most — real captures were cut once, synthetic pairs went
through the sweep — so the scarce human side is diluted about fourfold, and the
by-post val split cannot see it because val inflates by the same mechanism.

**Alternatives.** *Delete the promise* (W9's honest option) — the measurement
turned a speculative knob into a live defect: `meta.weight` was promised since
the first draft and `_write_conversations` stripped `meta` one line before the
trainer could see it. *Dedupe by budget at assembly* — throws away the
short-chunk diet the adapter is also served at. *A per-example loss scale* — the
cookbook's dataset builder has no field for one, so integer replication says it
through the interface that exists.

**Val is never replicated**: a weighted held-out loss averages over a
distribution nobody chose. **Not applied: an actual weighting policy** — choosing
the diet costs a paid retrain and is Dan's call. **Re-evaluate if** a run is
trained with a non-uniform weight, or the corpus stops being cut at multiple
budgets — passages and target bodies then converge.

## D44: the editor asks the sidecar where the prose is

**Decision** (2026-09-03, W4). `stylebot.serve` grows a `segment` op — text in,
`{"segments": [{start, end, kind, context}]}` out — backed by
`serve.prose_paragraphs`, which composes `stylebot.markdown` with the
`div_is_prose` policy `serve_loop` was started with. The extension's TypeScript
segmenter, its test and the fixture that pinned it are deleted, and it takes each
segment's `context` from the sidecar. `prose_paragraphs` lives in `serve` because
it needs what `markdown` refuses: frontmatter, and the *editor's* granularity.

**What the port had drifted into**, measured over every blog `.qmd` before
deleting it (`findings.md` "Segmenter implementations: 3 → 1"): every difference
was a correction — a heading glued to its paragraph took the prose down with it,
including whole `digest/` documents that scored nothing at all — and under the
blog's policy the op returns hundreds more paragraphs the port could not see,
because `div_is_prose` is a blog-side *callable*. That is not a bug the port had;
it is a bug the port's *location* had.

**Alternatives.** *Keep the port and fix its defects* — the argument D28/D30 lost
the same day: two implementations and a hand-copied fixture kept in step by hand,
which had failed once already. *Fold segmentation into `score`* — saves a round
trip costing 0.02 ms and welds two questions together. *Return paragraph text as
well as offsets* — the client sent the document and can slice it.

**Deciding constraint**: the hop is free on the already-debounced path
(`findings.md` "The sidecar `segment` op"), so location is chosen on correctness
alone. Offsets are UTF-16 code units, converted server-side, because every editor
this protocol will meet indexes UTF-16. **Re-evaluate if** the marker ever ships
as a `.vsix` with no sidecar — this entry is then the list of what a TypeScript
segmenter must get right (`_plans/vscode-marker.md`, alternative B).

## D45: one corpus location, resolved once — and the synth walker keeps its blog root

**Decision.** `stylebot.config` owns every spelling of a path inside the corpus
directory (`pairs_path`, beside W0's `splits_path`), and livingthing composes
its blog-root default on top of it in ONE function,
`training_targets.data_dir(flag, blog_root)`, with the precedence `--data-dir` >
`$STYLEBOT_DATA_DIR` > `<blog-root>/_training_pairs`. Every `dan-style` default
— corpus, splits, run manifests, the preference log — routes through it, and the
click defaults that need it are **callables**, not paths frozen at import.

*Alternatives.* Leave the blog's literals alone (they were "obviously" the same
directory — except `STYLEBOT_DATA_DIR` redirected `ai-style log`'s captures,
which go through `resolve_data_dir`, while synth/train/train-clf/run/serve kept
writing the live corpus, so pointing a scratch run at a scratch directory
silently half-worked). Or teach `resolve_data_dir` about a blog root, which puts
a caller's `--blog-root` convention inside generic tooling.

*Constraint.* `resolve_data_dir`'s own fallback is **cwd**-relative and must
stay that way; only the blog has a `--blog-root`, so only the blog folds it in.

## D46: the synth walker's blog exclusions stay pinned, not routed around

**Decision.** `dan-style synth` keeps calling `iter_targets(blog_root=…)` and
walking with stylebot's predicate. The QA list (W11.f) proposed handing it a
pre-selected `files=` list from `livingthing.lib.gather_qmd_files`, so the walk
would honour the blog's `LOCAL_SKIP_DIRS` (`src`, `bin`, `tests`). Refused on
two measurements.

*The gap is empty.* The two walkers agree on every `.qmd` in the live tree
(`findings.md` "The synth walker's gap") — the two skip-lists are disjoint, and
no `.qmd` has ever lived under `renv`/`packrat`/`rsconnect`/`node_modules`.
Exactly one file ever fell in the gap and the selector rejected it anyway.

*The fix costs more than the gap.* `files=` and `blog_root=` are mutually
exclusive in `iter_targets`, and without `blog_root` `_source_label` falls back
to `str(path)` — so `splits.role_of` answers `styler` instead of `eval`,
reopening the leak W0 had just closed and orphaning every pair keyed on the
relative source. `files=` also skips the selector, so the blog would
re-implement the frontmatter gates it currently passes as a callable.

*So instead:* the divergence is asserted.
`tests/test_qmd_walker.py::test_the_synth_walker_sees_exactly_what_this_repo_calls_a_post`
compares the two walks over the live tree and names any file either side has that
the other does not.

**Re-evaluate if** a `.qmd` ever needs to live under `src/`, `bin/` or `tests/`
again, or if `iter_targets` grows a label-only `blog_root` companion to `files=`
— the one-line change that would make the pre-selected form safe.

## D47: two error classes for the whole package, translated once at the group

**Decision.** `stylebot.errors` declares `ConfigError` (the arguments are wrong;
exit 2), `RunError` (the arguments are fine, the world is not; exit 1) and
`MissingExtra` (a `RunError` whose unready world is an uninstalled extra).
`cli_options.translate_errors()` turns them into click exceptions, and
`ErrorTranslatingGroup` applies it to the whole subtree. `Synth*Error` (D61) and
`SessionError` became named subclasses rather than a third and fourth scheme.

*The defect.* `bin/ai_style.py` had six `except ValueError:` blocks, every one
mapping to exit 1, while `synth` — the only module with a real hierarchy —
mapped its config failures to exit 2. So `train`'s "no training pairs left after
filtering" exited 1 and the same class of mistake in `synth` exited 2, and which
you got depended on which `except` a command's author had written.

*Alternatives.* (a) *A decorator per command callback* — exactly the thing a new
subcommand forgets, so the done-criterion would depend on a test noticing an
omission rather than on the omission being impossible; `Group.invoke` is the
funnel every descendant passes through. (b) *Keep `click.ClickException` at the
raise sites* — a library reachable without click must not hand its caller a CLI
exception, which is why `synth_cli` grew its hierarchy at all. (c) *One class
with a `code` attribute* — a number is a worse prompt than a name.

*Constraints.* Every class subclasses the builtin it replaces, which let ~90
raise sites across 14 modules move in one pass without breaking a single
`except`. And the vocabulary is for failures the user can act on: a violated
contract between stylebot's own functions stays a bare `ValueError`, let through
deliberately and pinned by `test_a_bare_builtin_is_not_translated`.
**Re-evaluate if** a third class earns its place — the candidate is a "partial
success" wanting a distinct code, handled today as a `quiet` `RunError`.

## D48: `Styler` is a Protocol, and it types the call, not the tokens

**Decision.** `stylebot.infer.Styler` is a `Protocol` declaring
`__call__(messages: list[dict], num_samples: int = 1) -> list[str]`, replacing
`Callable[..., list[str]]` — which typed the return value and neither argument,
so a scorer, or a backend taking `n=` instead of `num_samples=`, checked fine.
`tests/test_backends.py` checks the real backends against the Protocol's own
signature rather than against a copy of it.

*What the Protocol does not close, said out loud.* `tinker_backend` renders the
prompt with the cookbook renderer training used. `openai_backend` hands the
message list to a server applying its **own** chat template; one that wraps the
system turn differently, injects a reasoning block or prepends a default system
prompt puts the adapter out of distribution, where it stops transforming and
returns its input — silently. Both conform to `Styler`, so the Protocol closes
the argument-shape hole and *documents* the parity one.

*What was built instead*, because it is cheap and fires at the right moment:
`backends.manifest_mismatches` emits `renderer unverifiable` when the backend is
`openai` and the manifest pins a renderer. "Not checked" is a different claim
from "checked and fine".

*Rejected: the raw-completion mode* (cookbook renderer, POST to
`/v1/completions`). `build_generation_prompt` returns **tokens**, so it trades a
chat-template question for a detokenise-fidelity one with no measurement behind
it; the renderer lives in the `trainer` extra, so the backend whose appeal is
"serve locally without tinker" would acquire tinker to be trustworthy; and
`/v1/completions` is not universal. **Re-evaluate if** a second endpoint must be
trusted without a parity sample (`findings.md` "MLX local-serving parity").

## D49: asyncio is deferred; the rest of the eager import block stays

**Decision.** `stylebot.synth` imports `asyncio` inside the four coroutine
functions that use it. Everything else in `bin/ai_style.py`'s import block stays
eager. Timings: `findings.md` "`ai-style --help` import cost".

**Alternatives.** A lazy-command group for `synth` — rejected: `synth_options()`
is a decorator, so deferring it needs a `click.Group` subclass reimplementing
`format_commands`, plus a second place the command list is written down, for
~4 ms. Defer the rest of the block — rejected: `classify_train`, `splits`,
`train` and `eval` are eager *on purpose*, so click owns their defaults instead
of the triple declaration that preceded it, and `stylebot.lib` is unavoidable
because `is_human_authored` is a click default here.

**Deciding constraint.** W11.i's premise did not reproduce — `--help` was well
under the claimed 70 ms and the two imports were a third of the claimed saving.
The win is real and small; the item exists to say so rather than to ship churn.

**Cost.** `synth.asyncio` is no longer an attribute, so one test patches
`asyncio` directly.

**Re-evaluate if** `--help` latency becomes a real irritation (it is far below
the ~100 ms where a shell feels slow), or click gains lazy subcommands.

## P1: move `openai` to an `[api]` extra — PROPOSED 2026-08-10, REJECTED 2026-09-03

**Proposal**: demote `openai` from `[project].dependencies` to an `api` extra
beside `classifier` and `trainer`, and give the three lazy import sites a
friendly "install `stylebot[api]`" error. The observation holds and was
re-verified 2026-09-03 by blocking `openai` on `sys.meta_path`: every module
still imports, so the whole `--help` tree survives without the SDK, and only
`eval.openrouter_judge`, `infer.openai_backend` and `generators.openai_generator`
raise. What breaks is `synth`, `eval --judge` and `--backend openai`; `log`,
`make-splits`, `export-table`, `train`, `train-clf` and `classify` survive.

**Rejected, on the balance rather than the facts.** Neither re-evaluate-if
trigger has fired: still no installer other than livingthing, still no non-OpenAI
generator backend. And D30's supersession moved the balance further — bare
`stylebot` is now a hard livingthing dependency, but the blog declares `openai`
itself and imports it in about nine of its own modules, so the extra would save
that install exactly nothing. The only beneficiary is a third consumer wanting
`stylebot.classify`'s runtime and nothing else, which is precisely the trigger
that has not fired. The ergonomic cost is certain (`uv add stylebot` gives you a
`synth` subcommand that cannot run) and the benefit is hypothetical.

**Alternatives.** Finer-grained extras (`[judge]`/`[synth]`/`[serve-openai]`) —
three extras for one wheel is worse than the problem. Declare `stylebot[api]` the
documented default install — a way of admitting the extra is not wanted.
`ruamel-yaml` is **not** the same case: eagerly imported at module top level, and
the frontmatter path is core to `log`/`targets`/`synth`.

**Re-evaluate if** either original trigger fires. The friendly-error half is
cheap and can land alone: the three lazy sites would raise `MissingExtra` (D47),
a strict improvement over a bare `ModuleNotFoundError` either way.

## D50: a corpus is a directory of shards, split by producer and then by month

**Decision (W6, 2026-09-03).** `pairs.jsonl` becomes `pairs/`: one
`captures.jsonl` for real edit captures, one `synth-<YYYY-MM>.jsonl` per month of
synthesis. `stylebot.corpus` owns the routing rule, the shard-aware `rewrite`,
the one-per-corpus write lock and the manifest content hash. `config.pairs_path`
— D45's single spelling — resolves to the directory when it exists and the single
file otherwise, so **sharding is opt-in by existing**: a plain install never
migrates. `iter_pairs` is the seam that makes every reader agnostic.

*Why anything changed.* Committing the corpus is right — it is the backup story
for irreplaceable human captures (D37) — but every append rewrote the whole file,
so a hundred-odd commits carried gigabytes of logical blob content (`findings.md`
"Corpus blob cost"). Sharding does not shrink the corpus; it shrinks *the file a
write touches*, which is what the commit count multiplies.

*Alternatives.* **git-lfs** — GitHub charges LFS bandwidth on every clone of a
Pages repo whose build does not need the file, and it makes the corpus unreadable
without the client. **Shard by generator** (the W6 brief's first offer) — a
generator shard grows forever while a month shard is sealed when the month ends;
generator identity is a *facet* already in `meta.gen.model`, and a filename copy
is stranded by a retired arm and forked by a rename. **Route on read-identity** —
routing by content makes `rewrite` idempotent and the migration free.

*Constraint — the one thing sharding could have broken.* D36's no-duplicate-cell
guarantee was enforced by a lock on `pairs.jsonl` itself; locking each shard
would let two runs either side of a month boundary write the same cell twice, the
2026-08-12 defect wearing a hat. So the lock is one file per **corpus** and
`record_id` uniqueness is checked across it. **Re-evaluate if** a month's
synthesis grows past a few hundred MB, or the corpus stops being committed.

## D51: `git filter-repo` on the corpus is not worth a force-push — measured

**Decision (W6, 2026-09-03).** Do not rewrite livingthing's history to purge old
`pairs.jsonl` blobs. Delete `.git/subtree-cache` instead.

*The measurement that decided it* (`findings.md` "Corpus blob cost"). The W6
brief's figure — the corpus as ~85% of the repository — is **logical
(uncompressed) blob size**. On disk, in the packfile, an append-only JSONL
deltifies almost perfectly and the same versions cost a small fraction of that;
everything under `_training_pairs/` is under 5% of packed blob content.
Meanwhile `.git/subtree-cache` — a `git subtree` scratch directory, not history
at all — is an order of magnitude larger than anything filter-repo could reclaim,
and `rm -rf` on it needs no rewrite, no force-push and no coordination with
anyone holding a clone.

*Alternatives.* Run the pass anyway: tens of MiB reclaimed for a force-push to a
GitHub Pages repo, every commit sha in the blog's own prose and in stylebot's run
manifests invalidated (manifests pin `blog_sha` — D36), and every existing clone
and worktree needing re-cloning. The script is written and reviewable in the W6
handover, so the option stays open; it is simply not worth exercising.

**Re-evaluate if** the corpus stops deltifying — a change that reorders or
rewrites records wholesale on every commit would defeat it, which is exactly what
D50's shard-level rewrite avoids — or if a genuinely large binary (a merged
adapter, a `.joblib`) is committed by accident, where a targeted filter-repo on
that one path is the right tool and this argument does not apply.

## D52: documentation carries no mutable operational state — `status` answers instead

**Decision (W7, 2026-09-03).** stylebot's prose — `CLAUDE.md`, `README.md` and
`_plans/OVERVIEW.md` — states **no** corpus count, run number, "serving default"
or artifact sha. Those facts move under git's feet, so `status.collect()`
computes them on demand, sweeping the five artifacts that answer "where is this
pipeline?" — corpus, `splits.json`, runs directory, detector artifact, and *the
cross-checks between them*.

`findings.md` and `decisions.md` are explicitly **exempt**, and the distinction
is the whole design. A dated measurement records what was true then; a status
sentence claims to be true now and cannot stay so. The W7 brief's done-criterion
— grep the plans for the live numbers and find nothing — is wrong as written:
satisfying it deletes the measurements the file exists to hold. Applied instead:
every surviving number is attached to its measurement date.

**Why `ai-style status` and not only the blog mirror.** stylebot cannot discover
where a blog, a runs directory or a detector artifact is — the one-way dependency
arrow — so the generic command takes each as an explicit flag and reports absence
rather than raising. **The cross-checks are why this is one function**: each
number is printable alone, but what no artifact can see is the disagreement
between them (D39, D42), which is also why the `promoted` rule moved into the
library rather than staying the blog's.
*Alternatives.* **A section of `ai-style log list`** — `log` is the capture
session group. **Keep the numbers and add a CI check** — it can only run where
the blog is checked out, the coupling the arrow forbids. **A generated section
written back into the docs** — stale between runs, a merge conflict on every
capture. *Deciding constraint*: a document that cannot be kept true is worse than
none, because it is believed. **Re-evaluate if** corpus and tooling ever live in
one repo, or `status` grows past one screen — the split is then by artifact.

## D53: the orphan prune is built and declined; the eval prune is applied and now needs a second pass

**Decision (W12, 2026-09-03).** `ai-style data prune-stale` ships, dry-runs by
default, keeps a `.bak` and carries a test. Its **eval-contamination** half was
applied (blog `c0e2ac85be`); its **orphan** half was measured and **not run**,
and that is the decision rather than a postponement. Numbers: `findings.md`
"What a stale-corpus prune would cost and buy".

**W12's stated premise turned out false, and that is what settled it.** The item
expected W2's segmentation change to have orphaned cells the corpus would keep
paying for; it orphaned none. Four fifths of the orphans are prose Dan has edited
since the pair was generated — the corpus being a frozen record of prose that
moved, which is neither damage nor a reason to delete anything. Most of the rest
is `div_is_prose` charged against the **section-bounded** arm, which is frozen
policy: no future synth run refills those cells, so pruning is permanent loss.

**And it would not have bought what it was wanted for**: the prune barely moves
the effective per-passage weight D43 measured, because orphans are spread across
posts roughly in proportion to the corpus. *Alternatives*: prune anyway on
tidiness grounds — hundreds of paid-for pairs, half unregenerable, for a change
of 0.02; or delete the command — but unlike D38's migrations this is a
*measuring* instrument, and its question is worth asking before every retrain.

**The open half: the eval stratum is synthetic again.** The two changes landed in
the order that undoes one of them — the prune took the stratum to 0% synthetic
and eval_v2 then grew it with a post carrying synthetic pairs. Nothing computed
is wrong (D40's real facet is the headline), but the invariant is not held and
`extend-eval` does not prune; the durable fix is for it to offer the prune for
the posts it is about to freeze, since the two are one intent.
**Re-evaluate if** `--skip-covered` accounting starts disagreeing with the corpus.

## D54: a document-scale verdict is a mean over serving-scale chunks, unaligned across fields

**Decision (2026-09-03, the same day as the measurement that forced it).** The
detector is never handed a whole document. `eval.chunked_detector` wraps any
`Detector` so a passage is first segmented exactly the way serving segments it
(`infer.prose_chunks` — the promoted diet, the same protected blocks, the same
budget), each chunk is scored on its own, and the passage's number is the
**unweighted mean over its chunks**, in the ordinary detector slot with
`n_chunks` and `sd` beside it so nothing downstream changes. **The chunks are not
aligned across fields.** Before/after: `findings.md` "Document-scale eval".

*Alternatives.* **Pair the chunks across fields**, which the first write-up
assumed — rejected on the data: an edit is not a rewrite, Dan moves paragraph
boundaries, so the three variants cut to different chunk counts and field *i*'s
chunk *k* is often not field *j*'s; a chunk-index join buys a tighter interval by
pairing things that are not pairs. **Segment at the detector's training scale** —
scoring at a scale the styler never worked at reintroduces the gap `--documents`
closes. **A max** — turns the verdict into "the worst paragraph", a good editing
signal and a bad comparison one.

*The deciding constraint.* An instrument has a range, and a StyleDistance
embedding of a whole document is outside it — the shared ninety-odd percent of two
variants dominates the vector and the difference washes out. Nothing knew that,
because the detector had been built, validated and always used at chunk scale.
Scale is therefore part of the instrument's *identity*: `chunked_detector`
appends it to the head's `name`, so D39's mixing guard refuses to blend a
whole-document reading with a chunk-wise one on a resume. `sd` is recorded
because a mean with no spread hides the case it exists to expose.
**Re-evaluate if** the backbone's context window covers a whole post, or the eval
stratum supplies enough captured posts that the cluster count stops binding.

## D55: the scoring pass runs in a child — and the crash it was built for was a different bug

**Superseded by D59 on 2026-09-04.** The child, its `ScoreJob` marshalling and
the `classify` lock beside it are deleted: with the pool gone one task drives the
embedder, so the process boundary buys only a second model load.

## D56: `asyncio` first; threading must stringently justify itself

**Decision (2026-09-03, Dan).** A standing project-level constraint, recorded
here because later work has to be able to find and re-open it. `asyncio` is this
project's concurrency mechanism for IO, and it is what `synth` already uses at
scale (one event loop, an `asyncio.Semaphore` for the worker cap, one append
handle, no locks). **New threading needs a measured substantial win in
performance or complexity** — one of the two, with a number, and an entry here
carrying it. The rule as written is in [`CLAUDE.md`](../CLAUDE.md).

The precipitating case is D55: eight threads sharing one `SentenceTransformer`
segfaulted the process deterministically once chunk-wise scoring raised the embed
rate several-fold. It was latent for months, the first diagnosis was wrong
(libomp double-load — refuted by isolating the pass in a torch-only child, where
it still crashed), and the fix cost a lock plus a subprocess-isolation layer.

*Alternatives.* **Ban threading outright** — `focus.py`'s daemon HTTP server and
`library_archive`'s parallel downloads earn their keep, and a ban would push them
into worse shapes. **Judgement case by case** — judgement is what produced the
eval pool: `Judge` and `Detector` are plain sync callables and a pool is what one
reaches for to wrap those. **Bar it on "no shared mutable state"** — too narrow;
the pool shared a torch model and a Rust tokenizer, invisible at Python level.

*The deciding constraint.* EITHER a substantial reduction in complexity OR a
substantial increase in performance, **measured**. "Measured" rules out the eval
pool, never timed and buying nothing (`findings.md` "Threading audit");
"either/or" keeps the daemon threads. Corollaries in `CLAUDE.md`: never share a
non-re-entrant native object across threads; **a lock as the answer means the
threads were the wrong shape**; batching beats fan-out. **Re-evaluate if** a
free-threaded interpreter becomes this project's runtime.

## D57: the detector is batched out of the pool; the pool is kept for the judge alone

**Superseded by D59 on 2026-09-04.** The batched detector pre-pass stands
unchanged; the pool it was kept beside does not — the judge moved to an
`asyncio.Semaphore` at parity, the rewrite this entry had rejected as buying
nothing.

## D58: size budgets are a test with a shrinking allowlist, and each "why" has one layer

**Decision (2026-09-04)**: adopt P1–P8 of the blog's
`_plans/stylebot-seam-followup-2026-09.md` — one layer per "why" (decisions →
findings → code → commit), size budgets on modules, functions and docstrings, and
dispatch rules for agents. `tests/test_budget.py` enforces the sizes against an
allowlist of everything over budget today; the blog runs a copy over
`src/livingthing`. The allowlist is the record of the sizes as measured.

**Alternatives considered**:

- **An unenforced guideline in `CLAUDE.md`** — already there in spirit, and
  2026-09-03 added 12,277 lines under it.
- **Hard budgets with no allowlist** — red on day one, so the first agent to meet
  it deletes the test rather than the bloat.
- **One budget module shared by both repos** — a test-only module in stylebot's
  public API, and a blog suite that goes red on a stale git pin.

**Deciding constraint**: the bloat was a good instinct (say why) applied at four
layers at once, so the same paragraph exists three or four times and every later
reader pays for all of them. Review does not see that; a line count does. The
allowlist is what makes a number safe to adopt mid-flight — a ratchet, not a
cliff — and a deleted entry is the only evidence of progress the process keeps.

**Re-evaluate if** an allowlist entry has to *grow* for a change that is plainly
right: the budget is then wrong, and what gets written is the new budget, not the
exception. The enforced package-wide number is the prose line count rather than
P2's ratio, because deleting mechanism raises the ratio and F2/F8 exist to delete
mechanism; a ratio enforced literally would reward keeping it.

## D59: one mechanism in the scoring path — the judge awaits, everything else batches

**Decision (2026-09-04).** `eval.score_pairs_file` has exactly one concurrency
mechanism: an `asyncio.Semaphore` over the judge. A slice runs one pre-pass per
signal — Vale batched, the detector batched, then the judge gathered
`max_workers` calls at a time — and assembling records is bookkeeping.
Deleted: the `ThreadPoolExecutor`, the `threading.Lock` in `classify`, the
fallback-into-the-pool in `_detector_prepass`, and the `ScoreJob` child with the
`-m stylebot.eval` entry point it spawned. Added: `openrouter_judge_async`, the
judge a corpus pass takes, and `score_pairs_file_async` under the unchanged sync
signature. `detect_many` moves to `classify`, so `serve` stops importing `eval`.

**Alternatives.**

- **Keep the pool for the judge alone** (D57's position, reversed here):
  rejected on the measurement — parity, in fact 1.15x better, because the
  semaphore gates calls where the pool gated records.
- **Keep the child for loud failures**: rejected. Its diagnosis was disproved
  (D55), the segfault it made legible was the pool's, and no pipe launders an
  exit status any more.
- **Run a sync judge under `asyncio.to_thread`**: rejected — the pool again in
  an async hat; a sync judge is documented as serial instead.

**Deciding constraint**: D56's bar is a measured win in performance *or*
complexity, and the pool cleared neither once the detector left it
([`findings.md`](findings.md) "Threading audit", table 6). Four mechanisms had
been stacked on one path in a day, each containing the previous one's failure.

**Re-evaluate if** a second remote signal joins the judge (the gate then belongs
per signal), or if anything drives the embedder from more than one thread again.

## D60: a rewrite is a writer — it takes the corpus lock, and appenders re-check the inode

**Decision (2026-09-04)**: `corpus.rewrite` runs under `corpus_lock`, and
`PairsWriter` compares its open handle's `fstat` (and its remembered byte
offset) against the shard's current inode before trusting either. The lock stays
non-re-entrant and stays at one file per corpus.

**Alternatives considered**:

- **Lock only the appender** (the shipped state). Rejected: `rewrite` replaces
  each shard by rename, so an append landing inside one is lost, and the
  writer's `ab+` handle is left on an unlinked inode — every later append in
  that run then vanished with `write()` still returning `True`.
- **Hold the lock for `PairsWriter`'s lifetime** instead of per write. Rejected:
  a synth run is minutes to hours long and would block `log save` for all of it,
  which is the daily write on the irreplaceable captures shard.
- **Make the lock re-entrant** so the six read-filter-write callers could wrap
  read *and* rewrite. Deferred: a new mechanism (P8) for callers none of which
  nest today.

**Deciding constraint**: the corpus holds human captures that cannot be
regenerated and the failure was silent — no exception, no short count, a `True`
return. Cheap detection beats a bigger lock: inode identity is one `stat` per
shard per write, and it makes the loss impossible rather than unlikely.

**Residual**: a caller that reads, filters and then rewrites holds nothing
across that gap, so an append arriving in it is still overwritten — narrower
than before (it was the whole rewrite), not closed. **Re-evaluate if** synth and
`log save` are run concurrently in anger, or a second appender appears: the six
callers then take the lock across read and rewrite, and it becomes re-entrant.

## D61: the executables package holds no libraries — `synth_cli` left `bin/`

**Decision (2026-08-11, recorded 2026-09-04).** 1000+ lines of synth
orchestration moved from `bin/` to `stylebot.synth_cli`, raising its own
`SynthConfigError` / `SynthRunError` / `SynthGenerationError`; `bin/ai_style.py`
translates them to exit codes at the boundary (now `cli_options.translate_errors`,
D47).

**Alternatives.** Leave it and re-export from a library module — rejected:
`bin/` is not an importable contract, and a re-export advertises one that the
package layout denies. Split it across `synth` / `targets` — rejected: it is one
orchestration, and cutting it along module lines would put the run loop in two
files.

**Deciding constraint.** A library reachable only through `CliRunner` gets
copied rather than imported. D35 is the case that proved it: the blog
reimplemented `build_styler` because it lived in `bin/`, and the copy lost
`max_tokens`, truncating every blog rewrite at the 2000-token floor while the
promoted run served a 4500-char diet.

**Re-evaluate if:** never, in this shape — `bin/` is the click shell. Recorded
late because the CLAUDE.md diet (F7a) found this was the one rationale in that
file with no home in any other layer; the rule survived the cut, the why did
not.

## D62: `stylebot.eval` keeps a lazy re-export map after the split

**Decision (2026-09-04, F8).** `eval.py` split into `eval` (the four signals),
`scoring`, `verdict`, `docscale` and `covariates`. `eval` keeps a module-level
`__getattr__` that resolves the 28 names that moved, so every existing
`from stylebot.eval import X` and `ev.X` lookup still works.

**Alternatives.** Update the callers — rejected for the blog's, which stylebot
cannot edit and which import `verdict_options` at module scope plus ten `ev.*`
lookups; a public API that breaks on a refactor the caller cannot see is the
D35 defect from the other direction. Re-export eagerly at the top of `eval`
— rejected: `scoring` imports `eval`, so a top-level import back is a cycle.
Leave `eval.py` at 1,965 lines — rejected: it was the largest module in the
package and the thing F8 exists to fix.

**Deciding constraint.** The blog pins stylebot by git source, so a name that
disappears breaks a repo on a different commit clock. 25 lines of `__getattr__`
buys a split the callers never see.

**Re-evaluate if:** the blog imports from the owning modules directly, at which
point the map is dead weight and should be deleted rather than maintained.

## D63: orphan score rows are pruned where they are read

**Decision.** `scoring.prune_orphan_scores` drops score rows whose pair is no
longer in the corpus, at read time, rather than leaving them for a caller to
filter. Scale: `findings.md` "`scores.jsonl` orphan rot".

**Alternatives.** Leave them and filter per consumer — rejected: a stranded row
inflates every `summarize_scores` mean silently, and puts a phantom row in a
committed `covariate_table`, which is a number someone acts on. Refuse to score
against a corpus whose ids have moved — rejected: a prune is the ordinary
consequence of `data prune-stale` and should not be an error.

**Deciding constraint.** A score file outlives the corpus it scored; the join is
by `record_id` and nothing else re-checks it. This is distinct from D53, which
prunes the **corpus**, not its scores.

**Re-evaluate if:** score rows ever carry the corpus hash they were computed
against, which would let a mismatch be detected rather than silently repaired.

## D64: `STYLE_SYSTEM` was amended once, on 2026-08-04, and the corpus was re-stamped

**Decision (2026-08-04, recorded 2026-09-04).** The frozen system prompt dropped
`headings` from its "preserve … verbatim" list and gained *"Section headings may
be restyled, but keep each heading's level and meaning."* Every existing record's
`messages[0]` was re-stamped by a one-off script the same day. This is the only
edit ever made to the string every pair in the corpus is keyed on.

**Why it was permitted.** Cross-heading chunking (D15's consequence) packs
heading lines into the chunk as rewritable **content**, so a prompt demanding
they be preserved verbatim contradicts what the data teaches. The change is
forced by the diet, not chosen for taste: Dan's sections are short enough that
raising the byte budget alone moves the median target *down* (`findings.md`
"Section length", "Chunk yield by budget and mode"), so packing across headings
was the only route to v2 scale.

**Why re-stamping retroactively is legitimate**, the load-bearing half: the
migration touched `messages[0]` only — the user and assistant sides, the actual
(slop, Dan) transform, were never read or written. What changed is the
instruction the pair is *conditioned on*, which is exactly what should change
when the instruction's meaning changes. Pre-v2 records never contained a heading
as content, so they never exercise the new clause.

**Alternatives — none were recorded at the time, and that is itself a finding.**
Keeping `headings` and forbidding cross-heading chunks would cap the diet at the
section wall (D15); forking a v2 prompt would split the corpus on the one field
the validator pins. **Deciding constraint**: amending the string fails the whole
corpus until every record is updated, and there is deliberately **no re-stamping
script in the tree** — the one-off was deleted in 0.3 (D38). **Re-evaluate if** a
masked corpus is generated: its placeholder shape must be settled first.

## D65: `--include-real` defaults off — the two strata invent different things

**Decision (2026-08-12, recorded 2026-09-04).** The anchor-invention cull
(`ai-style data prune-invented-anchors`, D38) drops **synthetic** pairs only
unless `--include-real` is passed, and real pairs are excluded from *training* by
`infer.is_anchor_faithful` (D32) rather than culled from disk.

**Why, part one: regenerability.** A culled synthetic cell leaves its `synth_key`
missing, so the next `ai-style synth` refills it — under the gate this time, and
with `--gate-retries` most cells clear on a redraw (D32). A captured pair records
an edit Dan actually made, and nothing but git brings it back.

**Why, part two: the two strata fail differently**, which is the argument D32 was
missing. Measured 2026-08-12 (`findings.md` "Anchor provenance by stratum"): what
the *synthetic* stratum "invents" is mostly an external URL — a generator artifact
of the reverse-edit swap. What the *real* stratum shows is Dan wiring a note into
his own graph while editing: internal cross-references and citation keys
dominate. So the same gate reads as "the generator hallucinated a destination" on
one side and "the author added a cross-reference" on the other — genuine
practice, even though the styler has no way to know Dan's note graph.

**Alternatives.** *Cull both strata* — the flag exists and is never the default;
on the v1 corpus it took several hundred more pairs, all unregenerable. *Gate
neither and filter only at assembly* — wrong for the synthetic side, because a
cell left filled is never refilled under the gate (D32's amendment).
**Deciding constraint**: a synthetic pair is a draw and a captured pair is
evidence. **Re-evaluate if** real captures become numerous enough that a few
hundred are affordable to lose, or the synthesis gate gains its loss mirror.

## D66: the `anthropic/` arms bill through OpenRouter BYOK

**Decision (2026-08-16, recorded 2026-09-04).** An Anthropic key in OpenRouter's
workspace BYOK settings bills those arms to the Anthropic account rather than to
OpenRouter credits. Nothing in stylebot changes: `Arm.name` stays
`openrouter/anthropic/…`, so `synth_key`s do not fork and the corpus dedupes.

**The economics, which "effectively free" in the blog's `CLAUDE.md` stands for.**
OpenRouter's BYOK fee is **5% of list**, waived under a **$25,000/month allowance
measured in list-price inference** — orders of magnitude above this corpus, whose
entire v1 generation cost about fifteen dollars (`findings.md` "Cost: forecast vs
actual"). At this scale the fee is not payable.

**Alternatives.** *Keep paying OpenRouter credits* — simpler, and it makes
`meta.gen.cost` unambiguous; rejected because the blog is work output and the
employer's Anthropic account is the right payer. *Route Anthropic direct* — forks
`Arm.name` and therefore every `synth_key` on those arms, orphaning the pairs
already drawn from them (D11).

**Deciding constraint**: setup is console-only, and **"Always use for this
provider" must be ticked** — otherwise a rate-limited BYOK key silently falls
back to OpenRouter's shared capacity and spends credits. **Unverified,
deliberately**: `meta.gen.cost` comes from OpenRouter's `usage.include`, so under
BYOK it may report the fee or zero; the ledger records **list** prices
throughout. **Re-evaluate if** the blog stops being essentially work output —
the arrangement is justified only while it is.
## D67: the blog's Linux/CI machinery for `stylebot` is deleted

**Decision.** The blog resolves `stylebot` from one source — the editable
sibling `../stylebot`. The marker-partitioned git source W11.d added and the
Ubuntu `test` job W5 added are both removed (2026-09-04), with that job's
`setup-pandoc` and `setup-uv` steps. A fresh clone clones the sibling first;
the blog `CLAUDE.md` says so. F13 in the blog's
`_plans/stylebot-seam-followup-2026-09.md` is the work; F10 is the argument.

**Alternatives.** Keep both and make the refresh routine — push stylebot,
`uv lock --upgrade-package stylebot`, commit, push the blog — rejected: never
performed, and F10's own warning decides it, "until that is routine the job
will drift red again, which is worse than no job." Keep the git source and drop
only the job — rejected: the job was the source's sole consumer.

**Deciding constraint.** P8: no mechanism without a budget. Nothing on the only
machine that resolves this project read the pin — `direct_url.json` in the blog
venv records the editable path, and the git entry was gated on
`sys_platform != 'darwin'`. The job neither gated publishing nor passed:
`uv.lock` held stylebot at `2388335` (2026-08-17), predating every eval symbol
the blog now imports at module scope. A red X beside a live site is worse than
no signal.

**Not affected.** `bin/local-uv-run.sh`'s `$GITHUB_ACTION` exit stays: the
publish job installs `uv` only *after* the render, so the post-render chain has
no interpreter either way, and a job elsewhere never shared its filesystem.

**Re-evaluate if:** a second environment actually resolves this project — a
container, an HPC box — or CI is wanted as a merge gate. The pin then returns
with a RUNBOOK step that refreshes it, not without one.

