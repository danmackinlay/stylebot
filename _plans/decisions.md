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
(D52). **D1–D50 live in [`decisions-archive.md`](decisions-archive.md)** and
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
| D60 | a rewrite is a writer — it takes the corpus lock | **superseded** by D84 |
| D61 | the executables package holds no libraries | live |
| D62 | `stylebot.eval` keeps a lazy re-export map after the split | live |
| D63 | orphan score rows are pruned where they are read | live |
| D64 | `STYLE_SYSTEM` is frozen text with an author slot; a re-stamp is a rebuild | live |
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
| D78 | the four command bodies are stylebot passes over option bundles; `dan-style` calls them | live |
| D79 | blog corpus defaults are callables that read `--blog-root` off the click context | live |
| D80 | `iter_targets` takes `files=` with `blog_root=` — the root labels and selects, the list replaces the walk | live; supersedes D46 |
| D81 | captures are gap-coalesced, not windowed — untouched prose is not endorsed | live |
| D82 | the scale arms are nested — three honest analyses, and the stamp is not a facet | live |
| D83 | the synth flag audit closed 2026-08-13 — four retired, four kept, one deferred | live |
| D84 | the corpus is append-only draws plus derived views | live; the real stratum relaxed by D96 |
| D85 | the blog's literals are a `SourcePolicy` parameter, not a fork | live |
| D86 | `ruff format` is the formatter of record; D58's ceilings re-baseline once | live |
| D87 | a judge refusal is unscored, never a guessed digit; `SCHEMA_VERSION` 3 → 4 | live |
| D88 | `stylebot.features` is the fifth eval signal's first cut: pooled syntactic proxies and a grouped AUC | live |
| D89 | `stylebot.transcripts` replays a Claude Code session and hands the differ two file states | live |
| D90 | a pair weight is a per-example loss scale, not a replication | live |
| D91 | the trainer's cookbook adapter is `stylebot.train_data` | live |
| D92 | `argued2` is a brief→write composite over two generators, the objection real | live; probe failed, library-only |
| D93 | `<!-- sem -->` marks content in a real edit; the view cuts it, the styler never sees it | live |
| D94 | `ruff format` is adopted; a formatting commit is the one commit that may re-baseline a budget | live |
| D95 | D94's re-baseline follows the reformatting, not the formatting commit | live |
| D96 | a real capture's archive is the editable truth; `log recut` re-lands it | live; amends D84 |

## Index — [`decisions-archive.md`](decisions-archive.md) (D1–D52, plus P1)

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
| D43 | a passage is a stretch of prose, not a string; weight is replication | passage half live; weight half superseded by D90 |
| D44 | the editor asks the sidecar where the prose is | live |
| D45 | one corpus location, resolved once | live |
| D46 | the synth walker's blog exclusions stay pinned | superseded by D80 |
| D47 | two error classes for the whole package, translated once at the group | live |
| D48 | `Styler` is a Protocol; it types the call, not the tokens | live |

---

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

**Decision (2026-09-04)**: adopt P1–P8 of the blog's 2026-09 seam follow-up as
CLAUDE.md "Budgets and layers"; `tests/test_budget.py` enforces the sizes, P3's
and P6's doc budgets, and an allowlist of everything over budget as measured.

**Alternatives considered**:
- **An unenforced guideline in `CLAUDE.md`** — already there in spirit, and one
  day's seam pass (2026-09-03) showed what it held.
- **Hard budgets with no allowlist** — red on day one, so the first agent to meet
  it deletes the test rather than the bloat.
- **One budget module shared by both repos** — a test-only module in stylebot's
  public API, and a blog suite that goes red on a stale git pin.

**Deciding constraint**: the bloat was a good instinct (say why) applied at four
layers at once, so one paragraph exists three or four times and every reader
pays for all of them; review misses that, a line count does not. The allowlist
is a ratchet, not a cliff, and a deleted entry is the only progress it records.

**Re-evaluate if** an allowlist entry must *grow* for a plainly right change: the
budget is then wrong; write the new budget, not the exception.

**Amended 2026-09-04 (M1b, M1c)**: a module that comes under budget still fails
the test; the allowlist edit rides in the commit that earns it (P5); rejected
hint-not-fail, since an unread hint is how the allowlist stops shrinking.
`findings.md` is gated the same way (`FINDINGS_ALLOWLIST`, prose ≤ 12 after the table).

**Amended 2026-09-05 (R0, then the same day)**: the ratio failed on a five-line
de-duplication; a fall-only `PROSE_CEILING` then priced out a new module within a day.
Allowance: `PROSE_CEILING + 0.20 × new code` — deletion never fails, growth funds its contracts.

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

**Superseded by D84 on 2026-09-05 (W4 step 4).** Nothing rewrites the corpus —
a save is a draw, a prune a policy clause plus a rebuild — so there is no
writer lock: `corpus.rewrite`, `rewrite_with`, `corpus_lock` and `PairsWriter`'s
inode re-check are deleted.

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

**Amended 2026-09-05 (W4 step 5 item 5, D39).** One stranded record is now one
row per `(record_id, field, signal)`, so the returned count — and the line
`run_eval` prints — is **rows, not records**. The join key and the read-time
placement are unchanged.

## D64: `STYLE_SYSTEM` is frozen text with an author slot; a re-stamp is a rebuild

**Decision (2026-08-04, recorded 2026-09-04).** The frozen system prompt dropped
`headings` from its "preserve … verbatim" list and gained *"Section headings may
be restyled, but keep each heading's level and meaning."* Every existing record's
`messages[0]` was re-stamped by a one-off script the same day — the only edit
ever made to the string every pair in the corpus is keyed on.

**Why it was permitted.** Cross-heading chunking (D15's consequence) packs
heading lines into the chunk as rewritable **content**, so a prompt demanding
they be preserved verbatim contradicts what the data teaches; the change is
forced by the diet, not taste (`findings.md` "Section length", "Chunk yield by
budget and mode"). It moved the instruction the pair is *conditioned on* and
never the (slop, Dan) transform, which is what makes it legitimate.

**Alternatives.** Keeping `headings` and forbidding cross-heading chunks caps the
diet at the section wall (D15); forking a v2 prompt splits the corpus on the one
field the validator pins. **Deciding constraint**: a corpus is conditioned on one
prompt, so that prompt must have exactly one value, and it must be nameable.

**Amended 2026-09-05 (W4 step 5).** The text stays frozen; the **author** is a
slot — `ai_core.style_system(author)` renders `STYLE_SYSTEM_TEMPLATE`, and the
default renders the 2026-08-04 string byte for byte. Which prompt a corpus
carries is `ViewPolicy.system_prompt`: the view stamps `messages[0]` from the
policy, the draw's own turn becomes history nothing reads, and validation takes
the prompt as an argument. **Superseded**: "amending the string fails the whole
corpus until every record is re-stamped by a one-off" — a re-stamp is now a
policy change plus a view rebuild (D84), and there is still no migration.

**Re-evaluate if** a masked corpus is generated: settle its placeholders first.

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
`_plans/retired/stylebot-seam-followup-2026-09.md` (blog) is the work; F10 is the argument.

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
has: the dry run drops nothing (`findings.md`, "Two constructs the corpus no
longer holds"). `write_pruned` stays — `prune-stale` also calls it.

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

## D74: `status` lists every run manifest, ordered by the manifest's own clock

**Decision (2026-09-04, F12f).** `stylebot.status` gains `load_manifests` and a
`RunEntry` per manifest — run id, `kind`, state (serving / completed / not
promoted / incomplete) and the artifacts it pinned — which `format_status`
prints under the runs block; `promoted_manifest` reads the same list. Before
it, a verdict manifest showed only as the count going up (F6).

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
load-bearing: no `result` key, `promoted: false`; a test pins that nothing here
makes a manifest without a `result` promotable.

**Re-evaluate if** a runs directory passes ~30 manifests and wants a `--limit`.

**Amended 2026-09-05 (W4 step 4).** Promotion is opt-in: `promoted_manifest` serves
only `promoted: true`, which `ai-style promote RUN_ID` (`status.promote`) sets and
clears elsewhere; `train --dry-run` writes nothing. Live `runs/` serves `20260817-170037` under both rules.

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

## D78: the four command bodies are stylebot passes over option bundles; `dan-style` calls them

**Decision (2026-09-04, seam brief §1–§2; code in lanes A1–A3, B1–B4).** `eval`,
`train-clf`, `run` and `serve` become library passes of one shape — bundle(s),
policy hooks as keywords, `emit: Emitter = silent`, `stylebot.errors` only, data
returned: `scoring.run_eval`, `docscale.document_pass` and `verdict.report_eval`
over `EvalOpts`; `classify_cli.train_clf` over `TrainClfOpts`;
`backends.rewrite_corpus`/`rewrite_document` over `RunOpts`;
`serve.serve_sidecar`. Both CLIs render their flags from the bundles
(`bundle_options` with `defaults=`/`exclude=`, never a retyped flag), every
default is a named constant the library also uses, and the blog's mirror test
compares each shared flag's default and type, not its name. `classify_cli.py`
is the one new module — `classify_train.py` is allowlisted at 965 lines — and
is capped at 300 lines.

**Alternatives.** *Keep the blog's copies, pinned by the flag-name test* — the
pin missed five drifts in one day: the D34 `pair_ok` gate skipped, the dry-run
splits table lost, `[:5]` against `[:10]`, the anchor lost/invented breakdown
dropped, `== 1`/`== 0` where the label constants belong. *Grow
`classify_train.py`* — allowlisted, and P8 forbids growing it. *One `passes.py`
for all four* — a grab-bag; each pass belongs beside the library it wraps.

**Deciding constraint.** D35: a library reachable only through a CLI gets
copied, and the copy drifts. What this clears: the `functions` of the
`bin/ai_style.py` allowlist row (`eval_cmd` 139, `run_cmd` 105, `train_clf_cmd`
88); the blog deletions the seam brief §3 lists; `classify_cli.py` ≤ 300 lines.

**Re-evaluate if** a third front-end appears, or a bundle needs a per-field
callback — `bundle_options` re-pins defaults and drops flags, nothing more.

## D79: blog corpus defaults are callables that read `--blog-root` off the click context

**Decision (2026-09-04, seam brief §5; code in lane B5).**
`training_targets.context_default(resolve)` returns a zero-argument callable
that reads `data_dir`/`blog_root` from `click.get_current_context().params` and
returns `resolve(flag, blog_root)`; every blog `--blog-root`/`--data-dir` is
`is_eager=True`, so both are processed before any default is computed.
`train-clf` (`--pairs`, `--splits`), `eval` (`--pairs`; it gains `--blog-root`),
`extend-eval` and `train` take it; `status`/`synth` already resolve in the body.
The splits default keeps "used when it exists" by resolving to `None` when the
file is absent, while an explicit missing `--splits` stays loud via `exists=True`.
Amends D45: a callable default was necessary, not sufficient — `DEFAULT_PAIRS =
tt.pairs_path` is called by click with no arguments, so `train-clf --blog-root
/x --dry-run` read the cwd's corpus while `status --blog-root /x` read `/x`'s.

**Alternatives.** *Resolve in each command body* — a line per path per command,
and impossible for a default pinned inside a bundle instance
(`ExtendEvalOpts(pairs_path=…)`). *A `callback=` on `bundle_options`* — new
mechanism (P8) for the one repo that needs it. *Resolve from cwd and drop
`--blog-root`* — D45's precedence exists so a scratch run can be redirected.

**Deciding constraint.** `bundle_options` re-pins defaults and nothing else, so
the only hook a bundle-pinned default has is being a callable, and the context
is the one thing a zero-argument callable can see; click computes defaults
inside `make_context`, where that context is current.

**Re-evaluate if** click changes when defaults are computed relative to eager
parameters, or a second project needs the same resolution — then it belongs in
`stylebot.config`, not the blog.

## D80: `iter_targets` takes `files=` with `blog_root=` — the root labels and selects, the list replaces the walk

**Decision (2026-09-04, seam brief §6; code in lanes C1–C2). Supersedes D46.**
`iter_targets` drops its files/blog_root refusal. The rule: `blog_root` supplies
the label root *and* the selector; `files` replaces only the walk. With both,
each file is loaded, kept iff `selector(meta)`, and labelled by
`_source_label(path, root)`; `files` alone stays pre-selected and unlabelled;
neither stays an error, and `ai-style synth` keeps its one-of UsageError. The
blog hands `iter_targets` its own walk, `gather_qmd_files([], base=blog_root)`,
so `LOCAL_SKIP_DIRS` applies without stylebot knowing it; the blog's
`is_valid_qmd_file` re-composes over stylebot's predicate for its three other
callers rather than being deleted. D46's re-evaluate clause named exactly this:
"a label-only `blog_root` companion to `files=`".

**Alternatives.** *Keep the divergence pin* — D46 refused the merge because the
gap was empty (`findings.md` "The synth walker's gap is empty") and asserted it
instead; X5 (blog `0550ad99ef`) has already cut that double walk to
`test_no_qmd_hides_under_a_local_skip_dir`, a three-line rglob assert kept
until lane C deletes it: an empty gap asserted forever. *Teach stylebot
`LOCAL_SKIP_DIRS`* — project exclusions in generic tooling (D45); the comment
over `QUARTO_SKIP_DIRS` saying the blog reads them from `_quarto.yml` is false,
and C1 rewords it. *`files=` alone (W11.f)* — no selector, `str(path)` labels.

**Deciding constraint.** `Target.source` must stay blog-relative, or
`splits.role_of` answers `styler` for a frozen-eval post — D34's leak, reopened
by the walker instead of by a rename.

**Re-evaluate if** a second project needs the walk itself parameterised (a
predicate, not a list), or a `.qmd` must live under a blog-excluded directory.

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
construction, each surviving as a library knob at its old value
(`synth_cli.InspectOpts.report_max_rows`, `run_synth(extra_tags=)`;
`ChunkPolicy.sort_name` deleted outright 2026-09-03, D43): no cell re-keyed,
no paid corpus regenerated. Kept: `--capture-reasoning` — the tokens are paid
whether or not the trace is saved; "never run" is no evidence against a
diagnostic. Kept: `--splits`/`--role`/`--sample-frac`/`--sample-salt` — the
masking work (D33) needs exactly this `--role styler` substrate. Deferred:
`--assign-seed` — a `synth_key` input (it picks the arm, D12), outside this rule.

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

**Amended 2026-09-05 (R1)**: surface accepted at 41 flags; library-only fields
stay reachable through the bundles (`InspectOpts`, `run_synth(...)`).

## D84: the corpus is append-only draws plus derived views

**Decision (2026-09-04, W4 step 1).** Two layers. *Draws* record what happened — one per
generator call or human edit, keyed on its cell (`draws.draw_id`), raw output kept, `outcome`
of `accepted | gate:<name> | error:<class>`; an interim save stays a draw and its successor
names it (`supersedes`). *Views* are a function of the draws and a `ViewPolicy` whose sha names
the file; built, never committed. Never the reverse: a prune is a policy clause plus a rebuild,
a recut a chunk-policy change plus a rebuild, and nothing edits a draw. The five
null-provenance cells are kept; `draws.py` is the one new module (≤ 800, P2; no thread, lock or
process), and readers are unchanged.

**Alternatives.** *A rejects-only side file* — a recut still orphans, and the eight rewrite
paths stay. *SQLite* — versions worse in git, and the RUNBOOK repairs by hand. *Span-based
`synth_key`* — every prune still rewrites. *Dedupe interim saves* — no order is recorded.

**Deciding constraint.** The recut orphans every synthetic cell (`synth_key` hashes the cut
text), and the rejects — near half the spend — are unrecoverable, the pair alone having been
kept (`findings.md` "The view reproduces the corpus").

**Amended 2026-09-05.** (Steps 3–4) `resolve` moved to the blog (D45); D60's lock, the
migrations and every rewrite are gone. (Step 5) `keep_superseded` is off; a wrong draw is
retired by a **policy clause**, not the successor draw the hand-off asked for — the body
licenses both, but a clause writes nothing into the corpus and catches every later map row.
`ViewPolicy` gained `superseded_captures`, `eval_sources` and the `system_prompt` the view
stamps into `messages[0]` (D64). `pairs/` leaves git; `synth` and `capture` write draws only,
and an absent `draws/` is created. `findings.md` "The view drops the superseded captures".

**Re-evaluate if** a draw shard passes a few hundred MB (finer shards, not a database) or a
view is needed offline more often than it is rebuilt.

## D85: the blog's literals are a `SourcePolicy` parameter, not a fork

**Decision (2026-09-04 by Dan; landed 2026-09-05, W4 step 4).** The blog-shaped literals
stylebot carried become one object: `stylebot.lib.SourcePolicy(glob, meta_field, max_level)`,
with `selects()` and `level_of()`. `DEFAULT_SOURCE` renders today's literals, so no existing
caller moves; the blog states its own `SOURCE` and hands it to the walk, to the selector and to
`iter_targets`. It went in `lib.py` — the generic frontmatter layer that already owned the two
functions whose *defaults* were those literals — and not in a new module (P8: a seam this small
does not earn a file). The author slot of the same re-stamp is D64's.

**Alternatives.** *Write "danbot's engine, split out for hygiene" into OVERVIEW and stop paying
the generality tax* — the real contender, put to Dan as the either/or; he chose the parameter.
*Leave the literals as keyword defaults* — what was already there, and it hid
`field="automation"` from a caller that believed it had passed its policy: the blog's
`make_selector` passed `max_level` and took the field from stylebot's default.

**Deciding constraint.** stylebot has exactly one editable-path consumer, so generality was
unpaid-for until something forced the seam to be *named*; the step-4 re-stamp was the one
moment the corpus could absorb it — a view rebuild, not a migration (D84).

**Re-evaluate if** a second consumer appears. The author slot reached inference and serving on
2026-09-05 (R2): `train` pins `data.style_system`, `opts_from_manifest` carries it into
`BackendOpts.system_prompt`, `infer` takes `system=`, and draws stamp `SourcePolicy.system_prompt`.

## D86: `ruff format` is the formatter of record; D58's ceilings re-baseline once

**Decision (2026-09-05 by Dan; landed the same day, round-2 R1).** Run `ruff format` over the tree and adopt its output as the
canonical shape. It touched 11 of 82 files and 92 lines, all cosmetic — no behaviour, no
public name. Three D58 numbers moved with it and are re-recorded at the formatted size:
`targets.py::_post_targets` 95 → 99, `train.py` 995 → 996, `train.py::run_training` 97 → 98.
All three grew for one reason: a hand-compressed argument list the formatter puts back on one
line per parameter. This is the only re-baseline; from here P2's ratchet runs as written.

**Alternatives.** *Hand-compress the three signatures back under their old numbers* — the code
would read as it did, but the compressed form is exactly what the next `ruff format` undoes,
so the ratchet would be measuring the formatter, not the complexity. *Shave unrelated lines
from `run_training` until it fits* — the gaming P2 exists to stop. *Leave the 11 files
unformatted* — the state that had `--check` reported red across four waves and never actioned.

**Deciding constraint.** P2's ratchet is a proxy for complexity, and a formatter changes line
counts without changing any. A proxy that a mechanical, idempotent tool can move is only
honest if it is re-measured against that tool's output, once, in the open.

**Re-evaluate if** `line-length` or the `[tool.ruff]` style config changes — that is a second
re-baseline and needs its own entry. Nothing enforces the formatter yet: `ruff format --check`
is in the round-2 acceptance list but in no gate `pytest` runs, so the 11 files can drift
back; wiring it in is the rest of that plan's R1, not this entry.

**Amended 2026-09-07.** `.githooks/pre-commit` blocks a staged `.py` that `ruff format --check` or
`ruff check` rejects (`git config core.hooksPath .githooks`, once per clone). The blog stays
lint-only on purpose: its marimo notebooks are rewritten by marimo, not ruff.

## D87: a judge refusal is unscored, never a guessed digit; `SCHEMA_VERSION` 3 → 4

**Decision (landed 2026-09-04 as round-1 C4; recorded 2026-09-05, round-2 R4).** A judge reply
with no parseable score is `None`: it enters no mean, is counted in `n_unscored`, and that count
is printed beside every judge mean — `format_summary`, `eval --by`, the verdict. `SCHEMA_VERSION`
moved 3 → 4 because the judge mean itself moved once refusals stopped entering it as digits.

**Alternatives.** *Take a stray digit from the refusal's prose as its score* — what the parser
did. *Drop the row* — hides the refusal rate, a property of the judge worth seeing. *Leave the
version at 3* — the fields are additive, but a consumer comparing means across the bump would
be comparing different populations.

**Deciding constraint.** A mean is comparable across runs only when its denominator is visible.

**Re-evaluate if** the judge gains a structured output the parser can rely on, so that "no
score" becomes a protocol error rather than a verdict.

## D88: `stylebot.features` is the fifth eval signal's first cut: pooled syntactic proxies and a grouped AUC

**Decision (2026-09-07, `argued-slop-arm.md` stage 0).** One new module, `stylebot.features`,
under P2's 800 lines, its prose funded by e2b807a's density allowance (D58 amended).
Three functions: `syntax_profile(text)` is the per-text vector (per-sentence rates of commas,
subordinators, em-dashes, semicolons, justificatory connectives and concession markers, words
per sentence, sentence-length variance; regexes are module constants); `profile_pairs` pools
counts per facet over one side of any number of corpora, the real stratum being `meta.synthetic`
falsy; `separability` is the out-of-fold AUC of a logistic head on the punctuation-free vector,
real vs one facet, folds grouped by `meta.source`. `ai-style features` is the shell.

**Alternatives.** *Per-facet medians of per-record rates*, as the plan's text says — a
connective rate of 0.01 a sentence is zero on most records, so the median is zero on every
facet and the plan's own table (pooled) cannot be regenerated from it; medians stay per record,
in the separability vector. *Ungrouped folds* — paragraphs of one post land on both sides of a
fold and the AUC reports topic. *Score it through `eval`'s `scores.jsonl`* — the signal is over
a stratum, not a passage, and the eval schema keys one passage; D39's rows have nowhere to put
a stratum. *A second `--draws`* for the scratch audition — `--pairs` repeats instead: the
audition's view is a scratch pairs path beside the live view, as RUNBOOK §4 already runs them.

**Deciding constraint.** The `argued` criteria are numbers over two strata, one of them in a
scratch corpus, so the tool must pool across corpora and must not trust an AUC that can see a
post twice; the corpus was 8,142 records and 395 real when this was measured
(`findings.md` "Syntactic proxies by stratum").

**Re-evaluate if** the real stratum grows past a few thousand records (medians become
meaningful and the pooled rate stops being the only estimator), or the transcript sidecar lands
and the objection source stops being fabricated.

## D89: `stylebot.transcripts` replays a Claude Code session and hands the differ two file states

**Decision (2026-09-07, `transcript-mining.md`).** One new module, `stylebot.transcripts`, under
P2's 800 lines, funded by e2b807a's density allowance. It parses Claude Code's session JSONL
(the only format it claims), takes every `Edit`/`Write` on a `.qmd` under the session's `cwd`,
reconstructs the file as Claude left it — base is the session's last full `Read` of that path,
else `git show` at the last commit before the first edit; edits replay in order and the first
`old_string` that does not apply stops that file — and pairs it with the file at the first
commit after the session's last edit that touches the path. The two states go to
`capture.capture_pairs`, the same call `log save` makes, with a `keep` predicate that admits only
chunks whose before-side lies in a paragraph Claude's edits changed. `ai-style log mine` is the
shell, dry-run by default; the user turns and Claude's text/thinking beside each edit go to a
gitignored `transcript_turns.jsonl`, never a pair field.

**Alternatives.** *Paragraph-match Claude's `new_string`s against HEAD* — the sizing pass did;
at ratio 0.6 it pairs unrelated paragraphs and conflates Dan's edit with later drift. *Trust
the `Edit` result's `originalFile`* — empty on most 2026-09 edits, so a check, not the base.
*Diff the whole file* — Dan's edits to paragraphs Claude never touched would become Dan→Dan
pairs; the `keep` predicate is D81's rule applied here. *Backfill `meta.novel_after` by
migration* — D84 has none; new pairs carry it, old captures gain it as a view stamp.

**Deciding constraint.** The captures are the only slop with an objection behind it, and their
sessions record every edit exactly; reading it off invents no pair. Counts: the blog's sizing note.

**Amended 2026-09-07.** Most windows fell to a git base a dirty tree had made wrong, so the
blog's `claude-left.sh` hook keeps each `.qmd` as Claude left it and `--claude-left DIR` takes
that copy as the before state, replaying nothing; the transcript is the record, the copy a cache.

**Re-evaluate if** Claude Code's log format changes (the parser names its fields in one place),
or the real stratum grows large enough that hand capture alone keeps pace.

## D90: a pair weight is a per-example loss scale, not a replication

**Decision (2026-09-07 by Dan).** `meta.weight` is a float, default 1.0, that `train`'s
`map_fn` multiplies into the datum's per-token loss weights after the cookbook has normalised
them to sum to one per example; the work-dir JSONL carries it as `"weight"` beside
`"messages"`, the val file carries 1.0, and no row is replicated. `pair_weight` returns the
float; a non-positive value is a `ConfigError`, since a pair to drop is the selector's. The
manifest keeps the `pair_weights` histogram and gains `weight_mass`, the sum over the train
file, printed beside the row count. The blog's `novel_after_weight` is the first policy over
it: 1.0 to 0.2, linear to 0.25 at 0.6, dropped above (`transcript-mining.md`).

**Alternatives.** *Keep D43's integer replication and add a second float knob* — proposed, and
rejected by Dan: no manifest ever recorded a non-unit weight and no record carries one, so
there were no semantics in use to preserve, and two knobs for one idea alias. *Replication
only* — cannot express a down-weight below 1 without inflating every other row.

**Deciding constraint.** Tinker's `Datum` already takes per-token float weights; the integer
rule lived in stylebot's assembly layer alone, so it was a restriction the trainer never asked
for.

**Re-evaluate if** a run needs weights above 1: a scale of 3 on one appearance is not three
shuffled appearances, and whichever paid run first uses a non-unit weight is the first with
any weighting at all — its manifest should say so.

## D91: the trainer's cookbook adapter is `stylebot.train_data`

**Decision (2026-09-07).** The three functions that turn an assembled corpus into what
tinker-cookbook consumes — `write_conversations` (the work-dir JSONL, one row per pair with its
`weight`), `scale_datum` (D90's loss scale on a `Datum`) and `make_presplit_builder` (the
by-post pre-split `ChatDatasetBuilder`, D17) — move from `train.py` into `stylebot.train_data`.
`train.py` keeps assembly, hyperparameters, the manifest and the runner.

**Alternatives.** *Cut 25 lines of prose from `train.py`* — the module's docstrings are already
pointers; the cut would land on contracts. *Grow the allowlist* — it only shrinks (D58).

**Deciding constraint.** D90 put `train.py` at 1,011 lines against its 986 allowance; the
adapter is the one seam in it with a different dependency (the cookbook, lazily) and its own
tests, so the split follows a boundary that was already there. The number it clears: `train.py`
drops under its recorded ceiling and the entry is lowered to the new size.

**Re-evaluate if** the runner itself outgrows `train.py`; the next seam is the manifest.

## D92: `argued2` is a brief→write composite over two generators, the objection real

**Decision (2026-09-07 by Dan, `argued-slop-arm.md` stage 2).** `generators.brief_then_write`
composes two `Generator`s into one: a cheap *brief* call reduces the target to its claims and
the alternatives the passage itself rules out, bound to an objection sampled (seeded) from the
mined `transcript_turns.jsonl`; the *write* call gets that brief and the objection in its user
turn under a stage-2 system prompt and rewrites the passage. `GenOutput.meta` sums tokens and
cost across both calls and records the brief model and the objection's hash, never its text;
`prompt_id` hashes both prompts and `prompt_system` archives both; session history reaches the
write call only. Strategy label `argued2`, so the two designs facet apart. Built for a 40-pair
probe on a scratch data-dir, library-only: no CLI seam until the probe earns one.

**Alternatives.** *Iterate the single-call prompt again* — two versions bracketed the connective
rate at 5× and 0.5× real with nothing between (`findings.md` "The `argued` audition"). *A
fabricated objection* — what the plan first allowed; the sidecar exists now, and an invented
objection teaches invented justifications. *A registry `SlopStrategy`* — a strategy is one
system prompt; this is two calls, so it is a `Generator` factory.

**Deciding constraint.** On the style backbone every prompt-only arm separates from the real
captures at AUC 0.96+ and the phrase counts say a paraphrase call has nothing to argue from;
the probe decides whether any synthesis can approach the regime the real captures come from.

**Verdict (same day, `findings.md` "The `argued2` probe").** Embedding AUC 0.978 against v2's
0.962: no better. The brief landed the connective rate for the first time, so the mechanism
stays, library-only and tested; no CLI seam, no promotion. The real stratum is the route.

**Re-evaluate if** the real stratum passes ~1,000 pairs and a brief drawn from a *matched*
session (the objection that produced that very passage) can be tested instead of a sampled one.

## D93: `<!-- sem -->` marks content in a real edit; the view cuts it, the styler never sees it

**Decision (2026-09-07 by Dan, protocol in the blog's RUNBOOK §0c).** Dan wraps a span of his
own edit that changes *what the text says* in `<!-- sem -->…<!-- /sem -->`. Capture and mining
store the edit verbatim, markers included. A `ViewPolicy.cut_sem_spans` clause removes the
spans (and markers) from both sides of every record at build time, re-stamps `before_chars`,
`after_chars` and `novel_after`, records `meta.sem_spans`, and excludes a record whose two
sides are identical after the cut (`sem_only`). Synthesis strips the markers from a target and
keeps the text. `html_comments` (whole-line comments protected) travels beside `div_is_prose`
through `infer`, `backends`, `docscale` and `serve`, off by default for segment parity; the
blog opts in (`PROTECT_HTML_COMMENTS`, since its training drops comment chunks). The markers
are one stylebot constant (`markdown.SEM_OPEN/CLOSE`);
a span never crosses a paragraph break, an unclosed opener runs to the paragraph's end and is
counted, a stray closer is an ordinary comment. The editor shortcut is the voice-marker
extension's `voiceMarker.markSemantic` (Cmd+K Cmd+M), a toggle over the selection.

**Alternatives.** *Drop any chunk carrying a comment* (the synth rule, proposed for captures)
— throws away the style pair in every paragraph that also had a content change, the very pairs
the marker decouples. *Cut at capture time* — loses the label; D84 wants the raw edit kept and
the convention revisable by a rebuild. *Infer content from `novel_after` alone* — a statistic
where an explicit label is available; the two now measure different things (labelled vs
unlabelled content).

**Deciding constraint.** A real pair with an unmarked content insertion teaches the styler to
assert what Dan would have said; a rule that discards the whole paragraph discards the scarce
side of the corpus with it.

**Re-evaluate if** the labelled share of real pairs turns out negligible (then the marker is
overhead) or dominant (then the corpus is a content-editing record and the styler's premise is
wrong).

## D94: `ruff format` is adopted; a formatting commit may re-baseline a budget

**Decision (2026-09-07 by Dan, executed in the blog).** The blog adopts `ruff format` over
`src`, `tests`, `bin` and `scripts`; stylebot's tree was already format-clean, so the two
converge rather than diverge. `[tool.ruff.format]` excludes marimo notebooks — marimo
re-serialises a notebook on save, so a formatted one is a churn ping-pong — and excludes them
from the *formatter only*, leaving `ruff check` on them. `docstring-code-format` stays off.
A formatting commit carries nothing else and is listed in `.git-blame-ignore-revs`; the bar for
a SHA in that file is `ast.dump()` identical before and after, which is what makes it the one
commit type that may raise `test_budget.py` allowlist numbers wholesale. Every other commit
still only shrinks or deletes an entry (D58). Only entries for files the formatting commit
actually touched may be re-taken: a breach that predates it stays red.

**Alternatives.** *Lint only, no formatter* — the blog's standing rule until now; it kept
diffs small but left ~120 files in a hand-wrapped style that no check defended, and a
disagreement about wrapping cost a review comment each time. *Format without the blame-ignore
file* — one commit becomes the author of 288 lines in the worst file (measured on
`tests/test_dan_style_cli.py`; 19 with the file). *Exclude marimo notebooks from ruff entirely*
— loses lint coverage on the one notebook, which has a real per-file B018 suppression.

**Deciding constraint.** The allowlist rule "only ever shrinks" is what makes the budgets
credible, so the one mechanism allowed to raise them has to be independently verifiable.
AST equality is that check, and it is also exactly the bar for blame-ignoring the commit —
one property, two uses.

**Re-evaluate if** a second formatting pass is ever needed for a reason other than a ruff
version bump: that would mean the exception is being used as a routine escape from the budgets.

## D95: D94's re-baseline follows the reformatting, not the formatting commit

**Decision (2026-09-07).** D94 let a formatting commit re-take `test_budget.py` numbers and
restricted that to "files the formatting commit actually touched". Three blog files were
reformatted a commit early — by hand, in the working tree, riding along with the D93 wiring
(`780e9a4ffe`, whose own message flags the entanglement) — so the formatting commit
`af48c09a0e` found them already clean, touched nothing, and their re-baseline never happened.
Four numbers stayed red against ceilings taken from hand-wrapped source. They are re-taken
now, on the same bar: `ast.dump()` identical across the reformat.

The bar is what the clause was protecting, not the commit boundary. Measured by reformatting
each file's last pre-D93 revision in isolation: `bin/style_file.py` 209 → 328 lines,
`style_cmd` 88 → 116; `workflow_style.py` 389 → 409, `run_style` 87 → 99;
`training_targets.py` 793 → 815. Every pair is AST-identical. The D93 wiring itself adds
2, 5 and 7 lines on top — a kwarg pass-through per call site. So of the 121-line growth in
`bin/style_file.py`, 119 lines are the formatter rendering `@click.option(...)` one argument
per line.

**Alternatives.** *Leave them red* — D94's letter, and it makes the suite permanently red for
a reason no future reader could reconstruct. *Cut code to pay for the whitespace* — the
preferred remedy under D58, and wrong here: it deletes working code to fund the formatter's
line breaks, and it cannot reach the target anyway. `bin/style_file.py` is ~200 lines of
option declarations and has **zero** plain top-level defs, so F11's actual invariant (a bin
module holds only click commands and `_opt*` helpers) already holds; extracting the one
library-shaped block in `style_cmd` would leave the module at ~305 against a 250 budget and
still need the entry.

**Deciding constraint.** The budgets are a proxy for hand-written size, and the proxy is only
credible while the one exception to "ceilings only shrink" stays independently checkable. AST
equality is checkable; "was in the formatting commit" is a proxy for the proxy. Re-baseline on
the property, keep the property strict.

**Re-evaluate if** a third file turns up needing this. Twice is an accident of one entangled
commit; a pattern would mean formatting is routinely landing inside feature commits, and the
answer then is the pre-commit format gate (D86 amended), not another decision.

## D96: a real capture's archive is the editable truth; `log recut` re-lands it

**Decision (2026-09-10 by Dan).** D84's "nothing edits a draw" holds for a synthetic draw — a paid
generator event — and is relaxed for the real stratum. `captures/<id>/{before,after}.md` is the
record of a human edit, git is its ledger, and a human corrects it in place: a typo, a misplaced
`sem` marker, a paragraph reverted to retract its pair. `ai-style log recut ARCHIVE_ID` re-lands
the edited archive as draws under a fresh `capture_id`, `meta.archive_id` naming the archive and
`supersedes` naming every earlier landing, so identity (`capture_id:chunk_index`), the supersession
clause and session partitioning are untouched; the archive's `whole`, `source` and `snapped_at` are
kept. `draws/captures.jsonl` stays append-only, a draw's `record` being the cut at landing time.
`status --draws` lists archives whose live records are no longer substrings of them ("edited since
landing"), which replaces the blog's `MANIFEST.json`, its hook stage and its byte-pinning test — all
deleted. `corpus_provenance` follows `archive_id`.

**Alternatives.** *Re-cut at view build* — a fresh cut reproduces 117 of 124 archives only by
choosing between two gap policies per archive, and seven not at all (`findings.md` "A fresh cut
reproduces"), so a build-time cut changes the view silently. *Reuse the capture id with
`meta.recut`* — the clause supersedes per capture id, so a later `--redo` would leave the recut's
draws standing. *Keep the manifest, add an edit command* — a byte pin the command must regenerate
is the burden being removed; the drift line catches the same mutation, and the five archives the
pin had frozen already drifted.

**Deciding constraint.** Edit data is valuable, but keeping every scrap is not the goal when it
slows the collection of new edits: a burden of perfection on each save lowers the total (Dan). And
the pin bought nothing: a real draw's identity was never a text hash; archives were edited anyway
(`c0f8fbfa6b`, five by hand after landing); the guard was red and unrun; a correction past the last
save cost a four-place hand edit or a deletion. `findings.md` "A fresh cut reproduces 117 of 124".

**Re-evaluate if** the real stratum needs a whole-document re-cut under a new chunk policy (then a
landing should record its knobs), or a recut is wanted on the 104 pre-archive captures (a
chunk-scale archive would have to be minted from the draw).
