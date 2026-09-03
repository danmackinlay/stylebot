# Eval harness — ✅ BUILT (library + CLI; trained detector wired)

How we know the styler works. Runs **offline** — it scores candidate output and
is *not* wired into the trainer or baked into the served adapter. Built early and
in parallel: it's the ground truth every later phase reports against, and it
needs only sample prose, not a trained model.

**Status (2026-09-03):** the harness passed its own integrity audit and changed
because of it, and none of those changes are in the 2026-06-30 note below. A
score now carries the detector artifact that produced it and two shas are never
averaged together (D39); the paired interval clusters by post rather than
treating chunks as independent (D40); the headline verdict is the *real* facet
with the synthetic one reported separately (D41); and the frozen eval stratum is
versioned and add-only, with `dan-style extend-eval` the only way it grows and
`splits_sha256` pinned into every manifest (D42). The Pangram cross-check this
document has always described was finally *run* — see
[`findings.md`](findings.md), including why "built and tested, not run" was
worth less than it sounded. Live stratum size and composition:
`dan-style status`, never a number here.

**Status (2026-06-30):** `stylebot.eval` ships the four-signal scorer as typed
library functions + the thin `ai-style eval` CLI. Runs keyless by default (Vale
+ null judge/detector); `--judge` scores voice via OpenRouter. **All four signals
are now live:** the statistical detector is the **trained Dan-voice classifier**
(`stylebot.classify` seam + the livingthing-trained head), wired via
`ai-style eval --detector-model PATH` / `dan-style eval`. It replaced the
"audition a general AI-detector vs Pangram" plan — see "The detector decision"
below. Keyless + free per pair.

## Inputs

- **A `pairs.jsonl` corpus** (the pipeline lingua franca) — eval scores the
  *fields* of each pair: `slop` (`messages[1]`) and `target` (Dan, `messages[2]`),
  with the shared heading-context prefix stripped so only the transform region is
  judged. (One-prose-per-file is gone — we never have that.) A Phase-4 styler run
  scores the same way once it emits an `output`-bearing JSONL (add an `"output"`
  field extractor then).
- A **trained detector artifact** (`_models/voice-clf/`, livingthing-side) for
  the detector signal — keyless. `PANGRAM_API_KEY` only if you ever want Pangram
  as an *optional* independent cross-check (see "The detector decision").
- `OPENROUTER_API_KEY` for the LLM judge (one key, many models).
- Vale + an optional Vale config (a parameter — never a hardcoded blog ruleset).

Interface: `stylebot.eval` functions over `pairs.jsonl` (`score_pairs_file` →
id-keyed `scores.jsonl`; `summarize_scores` for the aggregate view); `ai-style
eval` is the thin CLI. Paths resolved via `stylebot.config`.

## The four signals

1. **Vale** — mechanical slop (banned words, indefinite "you", -ize spelling).
2. **LLM-as-judge** — "is this Dan-shaped" at the voice level. **Demoted
   2026-08-17 to an aggregate sanity check on a subsample**: measured against
   the detector it ranks Dan above slop on *real* captured pairs 3/20 of the
   time (14 ties) and cannot resolve two adapters at n=40. Never use it as a
   tie-breaker between runs; see `findings.md`, "The LLM judge". Its cost was
   the wrong worry (~$2.71 for 160 calls); its resolution is the right one.
3. **Trained detector** — "is this Dan vs AI-slop", scored by the trained
   voice classifier (`P(slop)`, higher = more AI-like). Keyless, free per pair.
4. **Dan's eyeball** — the veto channel (not automatable; reported alongside).

## The detector decision (settled 2026-06-30 — trained voice classifier)

**Decision: the detector is a trained Dan-vs-AI-slop classifier, not a
general-purpose AI-detector.** We already hold content-matched labels — every
pair is a `(slop, Dan)` couple over the *same* passage — so we can score the
question we actually care about ("does this read like Dan?") directly, instead of
borrowing someone else's human-vs-AI-in-general detector (Pangram) and paying per
call. It is **keyless, free per pair, reproducible**, and drops straight into the
existing `Detector` seam.

**How it's built** (both halves are stylebot mechanism, split by dependency
weight): the **runtime** is `stylebot/classify.py` — a frozen **style** embedding +
a logistic head, scored by a pure-Python dot product (dep-free at import,
enforced by a test); the **trainer** is `stylebot/classify_train.py` (dataset
assembly, the POST-split methodology, artifact I/O) behind the
`stylebot[classifier]` extra, with the generic CLI `ai-style train-clf`.
Author-specific *policy* — backbone pin, free-positives selector, path
defaults — lives in livingthing (`voice_classifier.py` / `dan-style train-clf`,
a thin delegate). The
backbone was a **measured** choice — a bake-off over the content-matched pairs
ranked candidates by content-matched pairwise accuracy + AUC (split by POST):

| backbone | pairwise_acc | AUC |
|---|---|---|
| **StyleDistance** (`StyleDistance/styledistance`, 768-d) — **winner** | **0.78** | **0.72** |
| LUAR-MUD | 0.76 | 0.71 |
| mxbai (semantic baseline) | 0.75 | **0.62** |
| Wegmann CISR | 0.73 | 0.67 |

The AUC column is the tell: mxbai's topic-dominance collapses on the absolute "is
this slop" question; a content-independent *style* embedding holds. Polarity is
`score = P(slop)` so it composes with `mean_detector_score` unchanged; the factory
also returns `p_dan` for reward callers.

### Eval vs reward — and the leakage-safe split contract

The detector plays two roles with different safety requirements:

- **Eval signal (measure only).** A clean **by-POST** train/test split
  (`GroupShuffleSplit`) gives an honest, in-distribution estimate. Since eval only
  *measures*, that is sufficient — this is the keyless cross-check for the paid
  judge. ✅ built.
- **Reward signal (optimise against — Phase-3 weighting, Phase-4 best-of-N).**
  Here a split is *necessary but not sufficient*. Splitting removes
  **data-leakage** circularity (the detector hasn't memorised the styler's posts);
  it does **not** remove **Goodhart / proxy over-optimisation** — optimising a
  policy against a frozen proxy pushes its outputs into the proxy's blind spots
  *regardless of how the proxy was trained* (Gao et al., reward-model
  over-optimisation). The guard against *that* is a signal with **independent
  failure modes** — the LLM judge and Dan's eyeball — not a fresh split of the
  same model. Best-of-N with small N is gentle (KL ≈ log N), so it's fairly safe;
  DPO/RL needs the orthogonal channel in the loop.

**The split contract (load-bearing, materialised):** one **shared by-POST
partition** governs all three stages — styler-train, detector-train, and the
frozen eval — so the detector never trains on the posts the styler trains on or
that eval scores. It lives in one canonical file, `splits.json`
(`ai-style make-splits`; the blog's is committed at
`_training_pairs/splits.json`), with three roles: **eval** (a frozen pinned
list, sampled from posts with ≥1 *real* pair), **styler**, and **detector** —
the latter two assigned by a deterministic hash rule so new posts flow in with
no file churn and eval can never drift. `ai-style train-clf --splits FILE`
(auto-used by `dan-style train-clf` when the file exists) fits on the detector pool
only, never even embeds the eval posts, reports a bonus styler-posts holdout
metric, and records role counts + the danger report in `meta.split`. The
ad-hoc `--holdout-frac/--holdout-posts` flags remain for experiments; the
splits file is the contract. (The default fit-all artifact is for *measurement*
of an independent styler, and `meta.split` says so.)

**Regularization is nested, not hand-tuned:** C defaults to selection by inner
`GroupKFold` *within each training side* (`select_C`, grid `C_GRID`), so the
reported CV metric covers the tuning step; `--C` is an explicit override, and
sweeping it against the printed number is exactly the tuning-on-the-test-set
this design removes. Residual caveat: the *backbone* bake-off already selected
StyleDistance on this corpus's CV metric, so the headline carries mild
winner's-curse inflation — the frozen eval posts are where that gets settled
honestly, once, at final-eval time. **There is deliberately no fourth
materialised split for hyperparameters** — nested CV inside the detector pool
serves that role; a dedicated validation stratum at 28-post scale would starve
every other role for nothing.

**Synth-augmented training (the provenance facet):** synthetic pairs
(`meta.synthetic`) can enlarge the training corpus, but "how well do we detect
our own paraphrase generator" must never masquerade as the honest number. On a
mixed corpus both eval modes automatically facet the metric by provenance —
`metrics.by_provenance.real` / `.synthetic` in `meta.json`, echoed by the CLIs;
`meta.n_pairs_real`/`n_pairs_synthetic` record the composition. The **real**
facet is the headline when synth pairs are in the fit.

### Pangram — optional independent cross-check only (was: the planned signal)

Pangram is no longer the path; it survives only as an **optional, one-shot,
independent** sanity check — useful precisely because it's a *different model
family* with different blind spots, so it's the one thing the voice-clf (and the
judge) can't self-certify: run it once before trusting the detector as a *reward*
to confirm the styler lowers AI-ness independently rather than gaming our
embedding. It is a paid API (~$0.05/1k words, `PANGRAM_API_KEY`). The judge +
eyeball already supply the orthogonality more cheaply, so Pangram is a
nice-to-have, not a dependency.

**Built 2026-09-03, not yet run.** `eval.pangram_detector(...)` is the opt-in
factory: `score = fraction_ai` (the house polarity, higher = more AI-like), a
stdlib `urllib` POST rather than the OpenAI SDK, and a `post=` transport seam so
tests exercise everything but the spend. It is **never** the default and
**never** goes in a hot loop — best-of-N reranking would draw it per candidate.
`_flatten_scores` treats a missing `fraction_ai` as `None`, not 0.0: no signal is
said out loud rather than read as "certainly human".

The one-shot pass, when Dan wants it (a `--detector-model`-shaped hook is not
enough here — the detector is injected, so this is a short library call):

```sh
# The REAL eval pairs, both sides = 2 calls each. Under eval_v2 that is 149
# pairs / 298 calls; the corpus median target is ~1.2k chars ≈ 200 words, so
# ~60k words ≈ $3.00 at $0.05/1k words (it was ~$0.80 under eval_v1's 41).
direnv exec . uv run python -c "
from stylebot import eval as ev
from stylebot import splits as sp
from stylebot.pairs import iter_pairs
from stylebot.jsonl import write_jsonl_atomic
S = sp.load_splits('_training_pairs/splits.json')
keep = [r for r in iter_pairs('_training_pairs/pairs.jsonl')
        if sp.role_of((r.get('meta') or {}).get('source') or '?', S) == 'eval'
        and not (r.get('meta') or {}).get('synthetic')]
write_jsonl_atomic('_tmp/pangram-eval-pairs.jsonl', keep)
ev.score_pairs_file('_tmp/pangram-eval-pairs.jsonl', '_tmp/pangram-scores.jsonl',
                    detector=ev.pangram_detector(), prune_orphans=False)
print(ev.format_summary(ev.summarize_scores('_tmp/pangram-scores.jsonl')))
"
```

What it answers: does an *independent* detector also rank Dan's side below the
slop side, on the same pairs where ours does? Record the pairwise number in
`findings.md` beside the voice-clf's `real`-facet 0.734, and stop — this is a
one-shot instrument check, not a signal to add to the loop.

## Planned — synthetic↔real distribution match (a fifth capability)

The four signals score *individual passages* and aggregate scalar means. They cannot
yet answer the question Phase-2 needs: **does synthetic slop match the distribution of
real Claude output Dan actually cleans up?** (Eyeballing showed synthetic slop missing
real mannerisms — e.g. incoherent mixed metaphors.) There is no feature/mannerism
extraction and no distributional comparison (KL / Wasserstein / frequency tables).

Planned (its own phase; deferred): a feature extractor (hedge rate, signposting
density, sentence-length variance, the `_voices/slop_patterns.md` families, optionally
an AI-detector score) run over **two strata** — synthetic pairs vs the **real-slop
reference** (the 254 Phase-1 pairs, ≥48 from `automation:2`; plus `automation:1/2`
prose) — and a side-by-side distribution view per strategy/covariate. **Seam:** the
`meta.synthetic` facet (real = `synthetic` falsy/absent) over the existing aggregation;
it consumes the `meta.gen` covariates synth records per pair.

## Outputs

- **`scores.jsonl`** — one **id-keyed** record per pair (`id` = `synth_key` or
  `capture_id:chunk_index`), carrying a `meta` subset (`source`, `synthetic`,
  `generator`, `slop_strategy`, the flattened `meta.gen` covariates, …) and
  `scores: {<field>: {vale, judge, detector,
  …}}`. The id is the join key back to the corpus — so scores feed Phase-3
  filtering/weighting and Phase-4 best-of-N without re-deriving anything. The run
  is **idempotent/resumable** (scored ids skipped), like `synth`.
  It is also **orphan-free**: append-only + join-by-id means a re-cut or a
  regenerated corpus would strand the superseded row forever, and no consumer
  checks the join — orphans inflate `summarize_scores` counts/means and emit
  phantom rows into the committed covariate CSV. So `score_pairs_file` prunes
  records whose pair is gone before it scores (atomic rewrite; count in
  `ScoreResult.pruned_orphans`, reported loudly by the CLI). `--no-prune-orphans`
  is the escape hatch for `--out` belonging to a *different* corpus.
- **A summary** (`summarize_scores`, `schema_version: 3`) aggregating per field
  across rows (fields-across-rows = the movement view: slop vs target vs output),
  optionally faceted `by` a meta key — `by="slop_strategy"` turns "is `casual`
  slop better than `polish`?" into per-strategy mean judge/vale/detector scores.
  Since v3 it also carries `by_provenance.real` / `.synthetic` with
  `headline_facet: "real"`, and `detectors` — the artifact identities present in
  the file (see "Eval integrity" below).
- **A verdict** (`compare_scores`) when two runs are scored over the same ids:
  the paired difference on one field/signal with a bootstrap CI **clustered by
  POST**, restricted to the real facet by default, and refusing to span two
  detector artifacts.

## Done-criteria

- [x] All four signals runnable from one entrypoint — Vale (live, optional),
      LLM-judge (live via OpenRouter), detector (seam + `null_detector`), eyeball
      (passthrough field). `stylebot.eval` + `tests/test_eval.py` (15 tests, no
      API/network/GPU). Runs keyless.
- [x] The detector decision made + the signal wired. The trained voice classifier
      (StyleDistance backbone, bake-off-selected) is the detector; `Detector` seam
      unchanged (`prose -> {score=P(slop), p_dan, name}`), served via
      `stylebot.classify` and `ai-style eval --detector-model PATH` /
      `dan-style eval`. Held-out by-POST pairwise 0.78 / AUC 0.72; the
      `--holdout-frac`/`--holdout-posts` split makes it reward-safe (see above).
- [x] Scores emitted in a stable, **id-keyed** schema (`scores.jsonl` +
      `schema_version: 2` summary) that Phase 3/4 can cite and join back to pairs.
- [x] A documented entrypoint: `ai-style eval --pairs PATH.jsonl [--field
      slop|target] [--judge --judge-model …] [--vale-config …] [--max-workers N]
      [--out scores.jsonl] [--summary summary.json] [--by slop_strategy] [--limit N]
      [--report scores.html] [--sample N]`. Batched, concurrent, resumable.
- [x] **Eval integrity** (2026-09-03, W10; D39-D42): every score row names the
      detector artifact that produced it and mixing two is refused; the paired
      verdict bootstraps over POSTs; the headline is the real facet; the eval
      stratum is versioned and add-only (`splits.extend_eval`, driven by
      `ai-style extend-eval` / `dan-style extend-eval` — dry-run by default);
      manifests pin the splits sha, the kept record-id list, the selector's
      constants and the installed stack. Tests: `tests/test_eval_integrity.py`
      (25, no network/key/model).
- [x] A read-only **scores visualiser** (`stylebot.report.render_scores_report` /
      `--report`): self-contained HTML joining `pairs.jsonl` + `scores.jsonl` by id —
      slop↔Dan text + judge score + rationale per pair, sortable by the slop→Dan
      delta, filterable by strategy, under a per-strategy headline. Generic over
      score *fields* (a Phase-4 `output` field renders with no change). Reuses the
      targets report's `_CSS` / `_histogram_svg` / escaping. No spend (re-renders
      from `scores.jsonl`).

## Status detail (as built 2026-06-28)

- **Public API** (`stylebot.eval`): per-text primitives `vale_score`,
  `openrouter_judge` (factory) / `Judge` / `JUDGE_SYSTEM` / `parse_judge_reply`,
  `Detector` protocol / `null_detector`, `score_candidate`; **batched layer**
  `score_pairs_file` (corpus → id-keyed `scores.jsonl`, concurrent + resumable) /
  `summarize_scores(by=…)` / `live_record_ids` / `prune_orphan_scores`. The
  pair-schema primitives `record_id` / `pair_body` / `FIELD_EXTRACTORS` live in
  `stylebot.pairs`, as does the shared reader `stylebot.pairs.iter_pairs`;
  import them from there. (`stylebot.eval` re-exported some of them until
  2026-08-10; nothing ever consumed the alias.) (The legacy in-memory
  conveniences `evaluate_groups` / `read_prose_files` were removed — the batched
  JSONL path is the one way to score.)
- **Keyless by default.** No judge ⇒ `judge: null`; Vale absent ⇒
  `{"available": false}`; detector unconfigured ⇒ `{"score": null}`. The harness
  always completes; signals it lacks are explicit nulls, never faked.
- **Judge via OpenRouter** (`OPENROUTER_API_KEY`, optional `OPENROUTER_BASE_URL`),
  default model `anthropic/claude-opus-4-8`, scores 1–5 + rationale; injectable so
  tests pass a fake. Mirrors `synth`'s OpenRouter wiring (one key, many models).
- **Detector (built 2026-06-30; trainer generalised 2026-07-06):** the trained
  voice classifier is the 4th signal, wired keyless. See "The detector decision"
  above; the generic trainer is `stylebot.classify_train` (`ai-style train-clf`,
  `stylebot[classifier]` extra); blog policy + the artifact live in livingthing
  (`voice_classifier.py` / `_models/voice-clf/`).

## Eval integrity — the instrument, the interval, the stratum (2026-09-03, W10)

Four things a verdict has to say besides the number. Decisions D39–D42; measured
composition in `findings.md`, "Eval-stratum composition" and "eval_v2 — applied
2026-09-03". **The blog's stratum is at `eval_version: 2`
(`splits_sha256=05b495adac68cdc4`, 2026-09-03): 12 posts, 177 pairs, 149 real /
28 synthetic, largest post 22%.** The counts in the four points below are the
v1 measurements the design was written against; the shape of each argument is
unchanged and v2 is what made two of them survivable.

1. **Which head produced it.** Every detector result carries
   `classify.DETECTOR_IDENTITY_KEYS` (`name`, `embed_model`, `git_sha`,
   `trained_at`, `n_pairs`) onto every `scores.jsonl` row.
   `score_pairs_file` refuses to *resume* a file written by a different head
   (`--allow-mixed-detector`) and `compare_scores` refuses a verdict spanning two
   (`--allow-cross-detector`). Resume is keyed on the pair id and says nothing
   about the instrument, so a refit head plus a resumed run used to blend two
   instruments into one mean with nothing to show for it.
2. **What the effective n is.** `paired_delta(clusters=…)` bootstraps over POSTs;
   `compare_scores` clusters on `meta.source` by default (`--cluster-by`).
   77 of eval_v1's 103 frozen-eval pairs came from one post, so a chunk-level
   interval over the stratum is too narrow by roughly √(mean cluster size).
   eval_v2 takes the bootstrap from 6 clusters to 12, which is where the
   clustered interval stops being *narrower* than the chunk-level one — the
   inversion that means "too few clusters to say" (`findings.md`).
3. **Real or synthetic.** `summarize_scores` facets `by_provenance` and names
   `headline_facet: "real"`; `compare_scores` compares the real facet by default
   (`provenance=None` for everything). 62 of eval_v1's 103 were our own
   paraphrases; the W12 prune took that to 0 and eval_v2 brought 28 back with
   `notebook/ai_search.qmd`, so the facet still earns its keep.
4. **At what scale it was scored.** `rewrite_documents_file` runs
   `infer.rewrite_text` over whole **capture archives** — `before.md` rewritten,
   scored against `before` (slop) and `after.md` (Dan) — and emits the same
   pair-shaped `output` JSONL the chunk path does, so nothing downstream changes.
   `--documents` on both CLIs. Everything else scores chunks that were cut before
   the diet under test existed, which is why a diet switch could not move its own
   verdict.

Verdict outputs belong under `runs/<run_id>/` (`eval.verdict_dir`,
`dan-style eval --run-id`), beside the manifest of the adapter they judge. Run
4a's lived in `_tmp/eval/` and were never committed.

### Document-scale eval — the command, and what it costs

```sh
# Free: what would be rewritten. Measured 2026-09-03 under eval_v2:
# 35 documents, 1,561,740 chars (86 capture archives exist; under eval_v1 only
# 3 sat on eval-role posts, for 86,820 chars).
uv run python -c "
from stylebot import eval as ev, splits as sp
S = sp.load_splits('_training_pairs/splits.json')
docs = list(ev.iter_document_pairs('_training_pairs/pairs.jsonl', splits=S, role='eval'))
print(len(docs), 'documents,', sum(len(d['before']) for d in docs), 'chars in')
"

# PAID: rewrite them through the promoted adapter, then score with the detector.
direnv exec . uv run dan-style eval --documents --run-id <run-id> --field slop --field target
```

Sizing, at the promoted 4500-char cross-heading diet: 1,561,740 chars is ~350
prose chunks and ~390k output tokens, so at Tinker's ~$0.60/1M that is **≈ $0.27
and the better part of an hour** (under eval_v1 it was 86,820 chars, ≈ $0.015 and
a few minutes). Still not expensive; it is a *paid* pass, which is the reason to
think before running it. D42's other half — "document-scale eval wants more of
its posts to be *captured* posts" — is what eval_v2 delivered: the thin
three-document verdict is gone. Score with
`prune_orphans=False` (the CLI does): the output is a different corpus from
`pairs.jsonl` and every id would otherwise read as orphaned.

## Guardrail (policy, not optional)

This is **not** a slop-detection evader (see post Non-goals). The detector is
useful only because its tripwires — uniform rhythm, hedging, signposting,
generic vocab — are the same patterns the slop catalogue targets. A styler that
lowers its detector score should do so *as a consequence* of writing more like Dan.

The reward-hacking guard is structural, not aspirational: a held-out by-POST split
stops the detector rewarding *memorised* posts, but the deeper risk when you
optimise against the detector is **Goodhart** (the styler finds inputs that score
"Dan" without being Dan) — which the split does not touch. The defence is signals
with **independent failure modes**: the LLM-judge and Dan's eyeball must confirm
the detector's movement is real. Detector movement alone is never the success
criterion. (See "Eval vs reward" above.)
