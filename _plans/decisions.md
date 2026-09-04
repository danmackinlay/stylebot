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
(D52). **D1–D48 live in [`decisions-archive.md`](decisions-archive.md)** and
keep their numbers — the second index below is how you find them.

## Index — this file

| # | title | status |
|---|---|---|
| D49 | asyncio is deferred; the rest of the eager import block stays | reverted 2026-09-04 |
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
| D67 | the blog's Linux/CI machinery for `stylebot` is deleted | live |
| D68 | segmenter tests run eleven hard-post excerpts; the sweep is opt-in | live; amended 2026-09-04 (F15) |
| D69 | a closing code fence back-references its opener's length | live |
| D70 | *(never written — reserved during concurrent work; the cull it was for is D73)* | — |
| D71 | the blog's serving policy is a module, not a `bin/` script | live |
| D72 | the blog's classifier commands are a shell over `voice_classifier` | live |
| D73 | the anchor-invention cull is spent, so it is deleted | live |
| D74 | `status` lists every run manifest, ordered by the manifest's own clock | live |
| D75 | a sampled message's content may be parts, not a string | live |
| D76 | an unclosed `:::` opener yields no span; the source is where it is fixed | live |
| D77 | `index_blog` is a package of stages under a click shell | live |
| D81 | captures are gap-coalesced, not windowed — untouched prose is not endorsed | live |
| D82 | the scale arms are nested — three honest analyses, and the stamp is not a facet | live |
| D83 | the synth flag audit closed 2026-08-13 — four retired, four kept, one deferred | live |

## Index — [`decisions-archive.md`](decisions-archive.md) (D1–D48)

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

---

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

**Reverted 2026-09-04 (T5):** the deferred import cost two docstrings (`_draw`,
`_draw_past_gates` had `__doc__ is None`); the 4.9 ms is below the re-evaluate line.

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
allowlist of everything over budget, plus P3's and P6's doc budgets; the blog
runs a copy over `src/livingthing`. The allowlist records the sizes as measured.

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
exception. The package-wide check is a prose count *or* a pinned ratio, either
sufficient (F16c): the count alone makes one documented helper cost a docstring
deleted elsewhere, and the ratio alone fails the code diet F2/F8 exist to do.

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
- **Run a sync judge under `asyncio.to_thread`**: rejected — the pool in an
  async hat; >1 worker now refuses one, and the async client is per pass (F19c).

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
  read *and* rewrite. Rejected: re-entrancy was never the blocker — a transform
  passed *into* the critical section wraps both without one (F19d).

**Deciding constraint**: the corpus holds human captures that cannot be
regenerated and the failure was silent — no exception, no short count, a `True`
return. Cheap detection beats a bigger lock: inode identity is one `stat` per
shard per write, and it makes the loss impossible rather than unlikely.

**Closed (2026-09-04, F19d)**: `corpus.rewrite_with(corpus, transform)` reads,
transforms and writes in one critical section, so nothing is held across a gap;
all six callers use it and `rewrite` is now only for a caller that already holds
the records. **Re-evaluate if** a transform ever needs to write to the corpus it
was handed, which the non-re-entrant lock still forbids.

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
dropped **synthetic** pairs only unless `--include-real` was passed, and real
pairs are excluded from *training* by `infer.is_anchor_faithful` (D32) rather
than culled from disk. The cull itself is deleted (D73); the training-side half
of this entry is the part still live.

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
## D68: segmenter tests run eleven hard-post excerpts; the whole-blog sweep is opt-in

**Decision** (2026-09-03; amended 2026-09-04, F15 — thirteen posts became
eleven). `tests/test_segment_hard.py` runs excerpts of real blog posts from
`tests/fixtures/hard/`, pinned span-by-span in `spans.json`, on every `pytest`
run; the same invariants over the whole blog are the pre-release check, behind
`STYLEBOT_SWEEP_BLOG_ROOT`. Both call one `violations()`, and a test asserts the
sweep over the fixtures equals the fast path, so the fast one is a proxy, not a
second opinion. `KNOWN_DEFECTS` is a shrinking allowlist in `test_budget.py`'s
idiom (D58). Costs, sizes, set cover: `findings.md` "The hard-post fixture set".

**Alternatives.** Keep sweeping every change — rejected: four agents did it in
one week, minutes a run, so it happened at the end instead of every commit.
Extend the synthetic `segmenter_conformance.qmd` instead — rejected: it was
written from the constructs already known, while the defects W2 found were
interactions a fixture author does not think to combine. Pick the dozen by eye —
rejected: the ranking (`fixtures/hard/manifest.json`) put the sole instance of
three constructs outside any plausible hand-picked set.

**Deciding constraint.** The first sweep found the selection wrong: two posts
break a fence-closer rule neither the twelve nor the synthetic fixture carried,
so `blogdown.qmd` was added (`findings.md` "The fence closer"). A construct that
is zero corpus-wide stays in `segmenter_conformance.qmd` rather than cost a
fixture — the unclosed fence at EOF, and, after blog `25cf42e5c5` and D69, the
unclosed `:::` opener, which took two fixtures with it.

**Re-evaluate if** the fast path stops failing on a break the sweep catches —
the answer is another fixture, never a loosened assertion — or if the fixtures
outgrow the budget `findings.md` records.

## D69: a closing code fence back-references its opener's character and length

**Decision (2026-09-04, F15).** `markdown.CODE_FENCE_RE` captures the opener's
whole run of backticks or tildes and requires the closer to be the same
character and at least as long — CommonMark's rule. W2 relaxed the closer to
`{3,}` so a longer line could close a shorter opener, and left the length
unmatched: a `` `````{verbatim} `` quoting a `` ```{r} `` was closed by the
inner fence, and its own closing line re-read as a fresh opener, merging two
spans. Numbers: `findings.md` "The fence closer".

**Alternatives.** Allowlist the two posts and leave it — rejected: this is the
one segmenter, so the damage lands in training targets, `segment_for_edit` and
the marker's `segment` op, not in a test. Fix the posts — rejected: variable
fence lengths are what CommonMark has them for, and the segmenter must read the
markdown the corpus is allowed to contain. Make the opener's run atomic
(`{2,}+`) so a failed search cannot re-read it shorter — measured identical on
all 1,758 posts, because the run-to-EOF alternative always matches and the
opener never backtracks; rejected as mechanism with nothing to buy (P8).

**Deciding constraint.** The sweep: 2 posts of 1,758 move, both corrections,
1,525 and 744 chars of prose returned from a fence span to the document.

**Re-evaluate if** a post wants a fence closed by a *shorter* line — Pandoc
accepts it, CommonMark does not, and the answer is a count of how many posts do
it before any loosening.


## D71: the blog's serving policy is a module, not `bin/style_file.py`

**Decision (2026-09-04, F11; written after the code, which is the defect F12b
records).** `livingthing.styler` holds the blog's *serving* policy: manifest
resolution, endpoint probing and the local staleness warning, the backend
option skin, `arm_styler`, the scorer and preference sink, the chunk diet, the
corpus rewrite and `serve_blog`. `bin/style_file.py` keeps option declarations,
the three commands, `click_emitter()` and `translate_errors()`; it speaks no
`click` outside `blog_backend_options`, every reporting function taking an
`emit: Emitter` and every `click.*` raise becoming a `stylebot.errors` class
(D47, F19b). Anything that is a thin call into
`stylebot.backends` is imported rather than wrapped.

**Alternatives.** Leave it in `bin/` — rejected: the seam tests already reached
into `livingthing.bin` to use it, and the copy that shape produces had lost
`--max-tokens` from `run` and `serve`, so every rewrite truncated at the
2000-token floor while the promoted run served a 4500-char diet. Split it
further (manifest / backend / sidecar) — rejected: all three read the same
manifest and diet, so the split buys three mutually importing modules and no
seam a caller asks for. Fold it into `training_targets` — rejected: that module
answers what a *training target* is, is imported by synth and the classifier,
and must not drag the backend surface in behind them.

**Deciding constraint.** D35: a library reachable only through a CLI module
gets copied, and the copy drifts. At 510 lines with no function over 80 it is
inside P2's budgets, so no allowlist entry is needed for it.

**Re-evaluate if** it passes 800 lines, or `serve_blog` grows a second
protocol — the editor sidecar is the natural first split.

## D72: the blog's classifier commands are a shell over `voice_classifier`

**Decision (2026-09-04, F12c; completed F19b).** `train_pass`, `eval_pass` and
`document_pass` — the orchestration behind `dan-style train-clf` and
`dan-style eval` — move from `bin/train_voice_clf.py` into
`livingthing.voice_classifier`, all three reporting through stylebot's `Emitter`
and failing with `stylebot.errors`, with `click_emitter()` and
`translate_errors()` supplied at the CLI boundary (D47). The
bin module keeps flags, the four commands and exit codes: 414 → 191 lines, no
plain top-level def, and all four of its `test_budget.py` allowlist entries
deleted.

**Alternatives.** Split `status` and `extend-eval` into a second `bin/` module —
rejected: a new module (P8) that moves ~90 lines and still leaves the shell over
250, because what is over budget is the orchestration, not the command count.
Allowlist 414 and stop — rejected: F11 recorded it as a debt and the allowlist
only shrinks. A new `livingthing.voice_eval` — rejected: `document_pass` already
lives in `voice_classifier` and the eval pass calls it, so the two modules would
import each other.

**Deciding constraint.** The blog's 250-line `bin/` budget (F11, on D58's
allowlist rule) and D61's rule that the executables package holds no libraries.
`voice_classifier` goes 175 → 440 lines, inside P2's 800 with no function over
80 and no docstring over 15; package prose does not grow (`PROSE_CEILING`
8580 → 8579).

**Re-evaluate if** `voice_classifier` passes 800 lines — the eval pass is the
natural split, and it takes `document_pass` with it.

## D73: the anchor-invention cull is spent, so it is deleted

**Decision (2026-09-04, F12j).** `migrations.find_inventing`, `anchor_kind`,
`ai-style data prune-invented-anchors` and their tests are deleted. D38 says a
migration goes once the only corpus that exists has been through it, and this one
has: the dry run drops **0 of 8,156** records (`findings.md`, "Two constructs the
corpus no longer holds"). `write_pruned` stays — `prune-stale` also calls it.

**Alternatives.** *Keep it as a report* — rejected: `infer.invented_anchors` and
`is_anchor_faithful` are the predicate, reachable without a command, and a
migration kept for its printout is a migration nobody re-reads before running.
*Keep it for `--include-real`* — rejected, and the measurement is why: that flag
would drop **49 real captures**, and reading them they are Dan adding a
cross-reference while editing, not a generator inventing a destination. Culling
them is the thing D65 exists to refuse.

**Deciding constraint.** The synthesis gate (`--drop-invented-anchors`, on by
default since 0.3) means no future run writes such a pair, so the count cannot
rise; a migration pinned at zero is dead code with an `--apply` flag on it.

**Re-evaluate if** a corpus predating 0.3 is ever restored from a backup; git
holds the function, and the predicate it wrapped is still in `infer`.

**Follow-up outside this repo:** the blog's `_training_pairs/RUNBOOK.md` §6a
still names the command. It needs deleting there.

## D74: `status` lists every run manifest, ordered by the manifest's own clock

**Decision (2026-09-04, F12f).** `stylebot.status` gains `load_manifests` and a
`RunEntry` per manifest — run id, `kind`, state (serving / completed / not
promoted / incomplete) and the artifacts it pinned — which `format_status`
prints under the runs block. `promoted_manifest` keeps its contract and now reads
the same list. This is what makes F6's "`dan-style status` lists the docscale run
with its pins" true; before it, a verdict manifest showed only as the count
going up.

**The ordering changes with it.** Manifests were sorted by filename in reverse on
the convention that a run id is a timestamp. A verdict manifest's stem
(`docscale-20260904`) sorts above every real timestamp, so the convention was
load-bearing for a rule it was never asked about. Ordering is now the manifest's
own `created_at` / `produced_at`, parsed (the two shapes carry different UTC
offsets), with the filename as the tiebreak.

**Alternatives.** *Leave the sort and rely on `promoted: false`* — rejected: it
works for a reason nothing states, and the two docscale manifests carry both
guards only because their author found the trap by hand. *Sort by file mtime* —
rejected: a checkout rewrites mtimes. *A separate `ai-style runs` command* —
rejected under P8, a new command for data `status` already reads.

**Deciding constraint.** The guards on the two committed verdict manifests stay
load-bearing: no `result` key, `promoted: false`. Nothing here may make a
manifest without a `result` promotable, and a test pins that.

**Re-evaluate if** a runs directory grows past ~30 manifests, when the listing
wants a `--limit` more than it wants completeness.

## D75: a sampled message's content may be parts, not a string

**Decision (2026-09-04, F12h).** `infer._message_text` flattens a renderer
`Message.content` — `str | list[ContentPart]` — before `tinker_backend` strips
it. A `ThinkingPart` is **re-wrapped** as `<think>…</think>` rather than dropped.

**The defect it fixes.** One eval document, capture `43f0e688`
(`notebook/local_llm_mac.qmd`), failed every document-scale run with
`AttributeError: 'list' object has no attribute 'strip'`, caught by the
per-document guard so 34 of 35 completed. The post's prose contains a literal
`` `<think>`…`</think>` `` and the styler faithfully reproduces it;
`parse_content_blocks` then splits any such sample into parts, so `content` is a
list. Two blog posts contain the construct.

**Alternatives.** *Keep only the `TextPart`s* — rejected: it silently deletes the
author's own text from the rewrite, and no guard sees it (a `<think>` span
carries no anchor, and the loss is too small to read as truncation). *Bypass
`parse_response` and decode the tokens* — rejected: it re-implements stop-token
handling to avoid one line. *Reconstruct `tool_calls` too* — not done: no blog
post contains a `<tool_call>` block, and a prose styler emitting one is a
malformed sample, not text to preserve.

**Deciding constraint.** The served renderer is `qwen3_5_disable_thinking`, so a
thinking part is not reasoning — it is prose that happened to look like a tag,
and the styler's contract is that its answer is text.

**Re-evaluate if** a thinking-enabled renderer is ever served, when a real
reasoning block would be re-wrapped into the document.

## D76: an unclosed `:::` opener yields no span; the source is where it is fixed

**Decision (2026-09-04, F12h).** `markdown._div_spans` keeps its behaviour: an
opener with no closer produces no protected span at all, so the `::: foo` line
itself is offered as prose. Documented, not fixed. Corpus-wide there are **0**
unclosed openers across 1,758 `.qmd` (`findings.md`, "Two constructs the corpus
no longer holds") since the blog fixed five asymmetric fences at `25cf42e5c5`.

**Alternatives.** *Protect to end of document* — pandoc's own reading (it closes
the div implicitly and warns), rejected here: one typo would remove the rest of
the post from training targets and from every rewrite, while a div's contents
render as the same markdown either way, so nothing is destroyed by leaving them
editable. An unclosed **fence** is the opposite case and runs to EOF (D69),
its contents being code. *Protect the opener line alone* — the narrow reading of
ground 3, rejected on P8: a live change to the one segmenter, moving training
targets and the blog's editing path, for a construct no post contains.

**Deciding constraint.** The construct is a source typo that pandoc warns about,
so the fix belongs in the `.qmd`, where it happened, and where it has already
happened five times.

**Re-evaluate if** an unclosed opener reappears in the corpus sweep, or a rewrite
is ever observed altering a `:::` line — either would make the narrow fix cheap
and obviously right.

## D77: `index_blog` is a package of stages under a click shell

**Decision (2026-09-04, F12i).** `livingthing/bin/index_blog.py` — 1,949 lines,
48 plain top-level functions, F11's explicit non-goal — becomes the package
`livingthing/index_blog/`: `config` (paths and knobs), `store` (the
`index_data/` caches and the skip hash), `embed` (the lazily-loaded model),
`indexing`, `similarity`, `topic_map`. The `bin/` module keeps options, six
commands and the `__main__` block the `map` child re-invokes; every module is
under the 800-line budget and the shell holds no plain defs. Three constraints
survive by construction: the offline/tokenizer env hardening moved to the
package `__init__`, which runs before any submodule and so before anything can
reach transformers; `sentence_transformers` is still named only inside
`embed.get_embed_model()`; and `run_map_isolated` still spawns `python -m
livingthing.bin.index_blog map` with the sentinel and the two thread pins.

**Alternatives.** *Leave it allowlisted* — the module is the worst instance of
the defect the same allowlist was built to retire. *Split the four long
functions too* — 335-line exporter and 104-line transform carried across
intact and re-allowlisted at their exact sizes; a pure move is diffable against
a byte-identical pipeline output, a rewrite is not. *Delete the six functions
found on the way with no caller in either repo* — reported, not done: outside
an item whose fence is no behaviour change.

**Deciding constraint.** This module segfaults *silently* when it is unhappy, so
a green suite is not evidence. The split had to be checkable by artifact
equality — identical `--help` for every command, byte-identical `topic_map.json`
and all 1,679 related JSONs — which a move gives and a rewrite does not.

**Re-evaluate if** the exporter or the UMAP transform is next touched for its own
reasons: that is when the 80-line function budget gets paid, not before.

## D81: captures are gap-coalesced, not windowed — untouched prose is not endorsed

**Decision (2026-08-04, v2 step 2; recorded 2026-09-04, T1).** A capture pair is
cut from the *changed* regions of a `before.md`/`after.md` archive, and two
regions merge only when the untouched run between them is within both
tolerances (`capture.diff_chunks(coalesce_gap_paras=, coalesce_gap_chars=)`,
`ai-style log recut --gap-paras --gap-chars`, `session.recut`). A pair never
grows by padding out to a byte budget. The asymmetry with synthesis is
deliberate: a synthetic target is Dan's published prose, all of it endorsed, so
cross-heading windows may be long (D15); a capture's before-side is
slop-bearing and only the edited stretch is endorsed, so capture pairs stay
changed-region-focused.

**Alternatives.** *A fixed-width window around each edit* — the plan's first
draft; untouched prose in a capture is text the author did not get to, not text
he approved, so the pair would teach the styler to leave slop alone. *No
bridging at all* — one multi-paragraph rewrite fragments into several short
pairs, the chunk-scale problem v2 exists to fix. *Also pair pure insert/delete
opcodes* — not worth a mechanism: skipping them costs one opcode in 41 archives.

**Deciding constraint.** Changed-fraction measures what a pair endorses: 0.96
under coalescing against 0.41 under a 4500-char window, with pairs a third as
many and longer (`findings.md` "Capture diffing loses almost nothing"). The
do-no-harm lesson is the one D18's near-copies had already taught.

**Re-evaluate if** captures start arriving from a whole-document review, where
every untouched paragraph *was* read and approved — a window is then honest —
or the tolerances change, which re-opens the changed-fraction measurement.

## D82: the scale arms are nested — three honest analyses, and the stamp is not a facet

**Decision (2026-08-12, v2; recorded 2026-09-04, T1).** `meta.merge_max_chars`
records which batch first produced a passage's text, not a design cell, because
`synth._synth_key` hashes the target text and not the budget (D11): a section
shorter than the smallest budget is packed identically at every budget and is
one cell. So "did scale help" is put one of three ways, and the write-up says
which: (1) restrict to targets **unique** to one budget — disjoint by
construction, immune to run order; (2) treat the corpus as the one mixed-scale
diet the curriculum intends and ask the *adapter*, by evaluating it on
length-stratified inputs — measures what the run is for and needs no clean
strata; (3) facet on the stamp anyway, stated as a short-versus-long-target
contrast that is neither randomised nor clean.

**Alternatives.** *Key the budget into `synth_key`* — mints a duplicate pair per
budget for identical text, multiplying the weight of exactly the short passages
the section wall already over-represents; `--skip-covered` exists to prevent
that. *Re-stamp every pair with its design cell* — there is none to record; a
short passage belongs to all three arms at once.

**Deciding constraint.** The overlap is measured: 5–19% of each arm's targets
sit in another arm, the stamp is contaminated by run order at the ~2–16% level,
and anchor density rises with budget (`findings.md` "Chunk budgets overlap",
"v2 post-gate distributions"). Run 4 trained the mixed diet; no
length-stratified verdict has been read yet; D43 made multiplicity the knob.

**Re-evaluate if** a run is trained on budget-unique targets only, or the corpus
stops being cut at several budgets — the arms are then disjoint and a facet is
honest.

## D83: the synth flag audit closed 2026-08-13 — four retired, four kept, one deferred

**Decision (2026-08-13; recorded 2026-09-04, T1).** The audit's remaining
retirement candidates were reviewed together and the file closed. Retired:
`--sort`, `--report-max-rows`, `--all`, `--tag` — each non-`synth_key` by
construction and each surviving as a library knob at its old value
(`synth_cli.InspectOpts.report_max_rows`, `run_synth(extra_tags=)`;
`ChunkPolicy.sort_name` was deleted outright 2026-09-03, D43), so no cell
re-keyed and no paid corpus regenerated. Kept: `--capture-reasoning` — reading
what a model deliberated has no substitute, the tokens are paid for whether or
not the trace is saved, and "never run" is not evidence against a diagnostic.
Kept: `--splits` / `--role` / `--sample-frac` / `--sample-salt` — retiring
them bets that no measurement run will ever need a substrate disjoint from the
detector's pool, and the masking work (D33) needs exactly that `--role styler`
case. Deferred: `--assign-seed` — a `synth_key` input (it picks the arm, hence
the generator name in the hash, D12), so outside the rule this pass ran under.

**Alternatives.** *Retire the substrate cluster as the "biggest single win"* —
declined for the bet above. *Retire `--assign-seed` with the rest* — re-keys the
corpus. *Keep everything* — 54 options against 19 for the next-largest
subcommand was the defect (`findings.md` "Surface size").

**Deciding constraint.** Retire only non-key flags, and hardcode the value they
supplied: D11's key is the line between a flag whose removal is free and one
whose removal regenerates a paid corpus. D21 is the 2026-08-10 half of the
same audit.

**Re-evaluate if** the masking work ships without ever using `--role styler`,
or a use for re-randomised arm assignment appears — `--assign-seed` is then
decided, not deferred.
