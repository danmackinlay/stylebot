# `argued` — a slop arm for concessive / justificatory structure

**Status: done 2026-09-07, both stages tried (D88, D92).** Stage 1 fails
match and separability with two prompts ([`findings.md`](findings.md) "The
`argued` audition"); stage 2's probe lands the connective rate but is no
closer on the style backbone ("The `argued2` probe"), which is D92's
retirement condition. `argued` (v2) is promoted as a diversity arm; the
`brief_then_write` composite stays library-only. **The write-up this file's
"Done" asked for is D92's verdict: the real stratum cannot be synthesised
this way, and grows instead through `transcript-mining.md`.** Retires to
`_plans/retired/` under P7 once the promoted arm has run in a corpus batch.

**Three corrections made while building** (each is in D88's alternatives): the
2026-09-03 table is *pooled* per-sentence rates, not medians — a median of a
0.01-a-sentence rate is zero on every stratum — so every "median" below reads
as the facet's pooled rate; the audition goes to a **scratch data-dir**, as
RUNBOOK §4 has always run auditions, so `features` pools a scratch view beside
the live one through a repeatable `--pairs`; and the separability folds are
grouped by post, or the AUC reports topic.

**What moved under it since it was written** (2026-09-05, after the two
ship-shape rounds): the corpus is append-only `draws/` plus a view built under
`training_targets.view_policy()` (D84), so every `pairs.jsonl` below reads as
"the view": `stylebot.features` takes `--draws`/`--policy` like every other
reader, `--pairs` only for a scratch file, and `dan-style features` reads the
blog's view. The measurement table stands as dated (its rows came from the
8,157-record corpus of 2026-09-03; the view is 8,142 today). `synth` is at 41
flags; `--slop-strategy`, `--session-turns` and `--limit` are unchanged. A
gate now refuses a draw whose target gained a backslash-escaped anchor (round-2
C13), and a judge refusal scores `None` (D87), so a `--judge` audition's mean
is over scored pairs only. The real stratum is still `meta.synthetic` falsy.
Transcript mining ([`transcript-mining.md`](transcript-mining.md)) is still
unbuilt, so stage 2's objection source remains the fabricated brief.

## The claim, and the evidence for it

Dan's suspicion (2026-09-03): the tortured subordinate clauses in the AI prose
he cleans up are not produced by *reasoning* as such but by a model **explaining
the outcome of a deliberation** — conceding an alternative, saying why it
fails, delivering the point as the consequence. None of the six registry
strategies asks for that shape: they are paraphrase prompts, and a paraphraser
has nothing to justify.

Measured on the corpus (per-sentence rates over the *slop* side; regex proxies,
the script is reproduced under "Stage 0"):

| slop stratum | N | words/sent | commas | subordinators | em-dash | semicolon | justificatory connectives |
|---|---|---|---|---|---|---|---|
| real captures (Claude in an editing session) | 398 | 24.3 | 1.45 | 0.25 | 0.36 | 0.15 | 0.016 |
| synth, **same posts as the real captures** | 628 | 20.1 | 1.12 | 0.23 | 0.17 | 0.05 | 0.007 |
| synth, `measured` | 1613 | 21.9 | 1.26 | 0.27 | 0.14 | 0.09 | 0.017 |
| synth, `verbose` | 1166 | 26.7 | 1.49 | 0.34 | 0.14 | 0.09 | 0.013 |
| synth, `casual` | 1470 | 18.5 | 0.98 | 0.17 | 0.16 | 0.06 | 0.005 |
| synth, effort=`off` / `high` | 6485 / 1382 | 20.1 / 21.0 | 1.07 / 1.10 | 0.20 / 0.22 | 0.14 / 0.20 | 0.07 / 0.09 | 0.008 / 0.008 |

*Subordinators* = which / although / though / while / whereas / because / since /
unless / whether / so that / rather than / in that / given that / such that.
*Justificatory connectives* = "which is why|exactly|precisely", "the catch is",
"that is the", "the problem is", "not because", "one might", "the obvious …".
The "same posts" row is the fair reference: it removes most of the topic
confound (the real captures skew to a handful of dense AI-topic notebook posts).

Three readings, all load-bearing for the design:

1. **Reasoning effort moves nothing** on any structural proxy (`off` vs `high`
   rows), consistent with the 2026-07-20 detector sweep (`findings.md`). The
   hypothesis is about the *prompt regime*, not hidden traces — so this is a
   strategy, not an effort setting, and `--capture-reasoning` stays optional.
2. **A system prompt alone already spans a 3× range** on these proxies
   (`casual` → `verbose`: subordinators 0.17 → 0.34, connectives 0.005 →
   0.017). Elicitation by prompt is demonstrably feasible at zero extra tokens;
   the question is only whether *this* shape can be described in texture.
3. **`measured` already matches real on subordinator and connective rate**; the
   robust residual gaps are sentence length, em-dash (2.5×) and semicolon
   (1.7×). Punctuation is a cheap lexical shortcut for the classifier (the
   `linkedin` emoji argument in `generators.py`), so the arm's success
   criterion is **structural separability**, not a dash count. A prompt that
   just sprinkles dashes fails this spec.

## What "reliably elicit" means here

An arm counts as eliciting the structure when the real↔synth gap on the
structural proxies closes *and* the pairs stay usable:

- **Match**: on ≥120 audition pairs across ≥4 roster models, the arm's pooled
  rates for subordinators, commas and connectives are ≥ 90% of the real
  stratum's, and words/sentence lands in 22–27.
- **Separability**: a cross-validated logistic regression on the *punctuation-
  free* proxy vector (subordinators, connectives, concession markers, words/
  sentence, sentence-length variance, commas), folds grouped by post,
  distinguishing real captures from the arm's slop has **AUC ≤ 0.70**, and is
  lower than the same model's AUC against `measured` (the baseline stage 0
  measured; `verbose` turned out marginally lower still — `findings.md`).
- **Not degenerate**: identity / inflation / anchor gate drop rate ≤ 1.5× the
  roster median for the same run; detector gap (existing classifier, P(slop) on
  arm − P(slop) on Dan) ≥ 0.07, i.e. not collapsed from the 0.09 norm.
- **Not cartoonish**: 20 eyeballed pairs — ≥15 read as a writer justifying a
  decision; ≤3 open with the same construction (sampler convergence, the
  failure mode the registry comment warns about).

Failing *match* or *separability* with a single-call prompt is what triggers
stage 2; failing *degenerate* or *cartoonish* is a prompt-text iteration.

## Token burn — the short answer is no

Per surviving pair today (`meta.gen` medians, effort=`off`, 32-turn sessions):

| model | prompt tok | completion tok | $/pair |
|---|---|---|---|
| claude-sonnet-4.6 | 8.1k | 370 | 0.012 |
| gemma-4-26b / qwen3.5-9b / qwen3.6-35b / deepseek-v4-flash | 6–8k | 340–540 | 0.0008–0.002 |

Prompt tokens are dominated by session history, not the system prompt, so
a longer strategy prompt (+~250 tokens) is noise. **Stage 1 costs exactly what
any other strategy costs**; the audition is stateless (`--session-turns 1`),
which makes it *cheaper* than production (prompt ≈ 1.5k). Stage 2 adds one
short call per pair (≈600 prompt + ≈200 completion for the brief, and the
write call's prompt grows by the brief): ~1.6–2× per pair, i.e. ~$0.02 on
Claude and under $0.005 on the open models. Budget for the whole slice through
promotion: **< $30**, most of it the Claude arm.

## Stage 0 — the proxy extractor (prerequisite, keyless, no spend) — BUILT 2026-09-07

Shipped as specified below with the three corrections above: `stylebot.features`
(`syntax_profile`, `profile_pairs`, `separability`, D88), `ai-style features`
and `dan-style features`, `tests/test_features.py`. The table regenerates from
`dan-style features --by slop_strategy --separability`; the baseline is in
`findings.md`. The original specification follows, unchanged.

The done-criteria above need a repeatable measurement. This is the minimal
first cut of the eval harness's planned fifth capability, scoped to what this
slice needs. That capability, as specified 2026-06-28 and never built: the
four signals score *individual passages* and aggregate scalar means, so they
cannot answer the question Phase 2 needs — **does synthetic slop match the
distribution of real Claude output Dan actually cleans up?** (Eyeballing showed
synthetic slop missing real mannerisms, e.g. incoherent mixed metaphors.) The
plan was a feature extractor (hedge rate, signposting density, sentence-length
variance, the `_voices/slop_patterns.md` families, optionally an AI-detector
score) over **two strata** — synthetic pairs against the real-slop reference
(the Phase-1 captures plus `automation:1/2` prose) — with a side-by-side
distribution view per strategy/covariate. **Seam:** the `meta.synthetic` facet
(real = `synthetic` falsy/absent) over the existing aggregation, consuming the
`meta.gen` covariates synth records per pair.

- **Library**: `stylebot.features.syntax_profile(text) -> dict[str, float]` —
  per-sentence rates for the proxies named above plus sentence-length
  variance and a *concession marker* rate (although / though / while / granted /
  admittedly / to be fair / of course / yes, / true,). Regexes are module
  constants so the table in this file is reproducible from the code, not from a
  chat transcript. Sentence split: `(?<=[.!?])\s+`, drop sentences ≤ 2 words —
  the same rule used for the table.
- **Aggregation**: `stylebot.features.profile_pairs(pairs_path, *, field="slop",
  by=None)` → per-facet medians, faceting by any `meta` key with `synthetic`
  falsy/absent as the real stratum (the seam named above). A
  `same_sources_as_real: bool` filter reproduces the fair reference row.
- **Separability**: `stylebot.features.separability(pairs_path, facet_value,
  *, by="slop_strategy")` → cross-validated AUC of a logistic head on the
  punctuation-free proxy vector, real vs that facet. `sklearn` is already in
  the `[classifier]` extra; import lazily and fail with the install hint if
  missing.
- **CLI**: `ai-style features --pairs PATH [--field slop|target] [--by KEY]
  [--separability]` printing the table above. Thin wrapper, no defaults on
  `--pairs`.
- **Tests** (`tests/test_features.py`): a fixture sentence with a known count
  of each marker; `profile_pairs` on a 6-row fixture corpus with one real and
  two strategies; separability on a synthetic separable/inseparable pair of
  facets hits AUC ≈ 1.0 / ≈ 0.5.

Done: the table in this file regenerates from `dan-style features --by
slop_strategy` and the `measured` baseline AUC is recorded in `findings.md`.

## Stage 1 — the `argued` strategy (single call, system prompt only) — REGISTERED 2026-09-07

`STRATEGIES["argued"]` version 1 was the draft below with two words changed:
"the obvious alternative" became "the alternative a reader would first reach
for" (the draft quoted a stage-0 connective and failed its own test), and
"argued rather than stated" became "argued instead of stated" ("rather than" is
the subordinator Dan strips most, §"Risks"). It auditioned cartoonish (the
staged objector converged on one phrase), so **version 2, the registry text
today**, forbids an imagined reader, argues once a paragraph at most and asks
for mixed sentence lengths; `prompts.jsonl` in each scratch dir archives both.
The audition command is the scratch form: `direnv exec . uv run dan-style synth --data-dir
_tmp/argued-audition --slop-strategy argued --session-turns 1 --sample-frac
0.05 --limit 120` (the `--sample-frac` spreads 120 chunks over ~100 posts
instead of the walk's first 16), then `ai-style build-view --draws
_tmp/argued-audition/draws` and `dan-style features --pairs <view> --by
slop_strategy --separability`. The original specification follows.

Register in `stylebot.generators.STRATEGIES` as `"argued"`, version 1. It is a
generic AI-prose flavour, not a Dan-specific catalogue, so it belongs in the
registry, not in `--slop-system-file`. Obey the registry rule: **describe the
texture, never quote a tic** — no "which is why", no "the catch is" in the
prompt, or every output opens with it.

Draft prompt (iterate on it; the criterion is the numbers, not the wording):

> You are a careful technical writer explaining the outcome of an
> investigation to a sceptical collaborator. Rewrite the user's passage so
> that every claim is argued rather than stated: before each conclusion, name
> the obvious alternative a reader might reach for and say why it does not
> hold, concede what that alternative gets right, then deliver the point as
> the consequence of the concession. Let the reasoning show in the syntax —
> link causes to conclusions inside the sentence with subordinate and relative
> clauses, close a sentence by naming what follows from it, and prefer one
> qualified sentence to two plain ones. Add no facts, examples or opinions the
> passage does not already contain or clearly imply: the alternatives you
> dismiss must be ones the passage already rules out. Keep the length between
> the original's and one-and-a-half times it. Do not open with a summary and
> do not close with one, and vary your sentence openings and structure from
> passage to passage. *(+ `_SLOP_PRESERVE`)*

Mechanics — all existing seams, no new ones:

- `prompt_id` is the content hash, so prompt iterations are faceted and deduped
  for free; `prompts.jsonl` archives each version.
- Audition: `direnv exec . uv run dan-style synth --slop-strategy argued
  --session-turns 1 --limit 120` from the blog root, roster as configured; the
  gates (`MAX_TRANSFORM_SIM` 0.95, `MAX_LENGTH_RATIO` 3.0, anchors) apply
  unchanged. The prompt's 1.5× ceiling sits well inside the inflation gate.
- Measure with stage 0: `dan-style features --by slop_strategy --separability`.
- Test: `test_argued_strategy_registered` mirroring the `measured` one, plus an
  assertion that no connective from the stage-0 regex list appears verbatim in
  the prompt text (the texture-not-tics rule, made mechanical).

Promotion is the blog's deliberate act: append `"argued"` to
`livingthing.training_targets.SLOP_STRATEGIES` with the audition numbers in the
comment, as `verbose` was promoted 2026-07-22. At a 1/7 rotation share the arm
enters new generation at no cost multiplier.

## Stage 2 — brief → write (contingent on stage 1 failing match/separability) — PROBED 2026-09-07

Built as `generators.brief_then_write` (D92), objections sampled from the mined
`transcript_turns.jsonl`; a 35-pair probe on sonnet and qwen3.6 with a
deepseek brief. Verdict in `findings.md` "The `argued2` probe": the only arm to
land the connective rate, and no closer to the real captures on the embedding.
The specification follows, unchanged.

If a paraphrase-shaped call cannot argue because it has nothing to argue
*from*, give it something: two calls per pair.

1. **Brief**: a small, cheap model (whatever the roster's cheapest is) turns the
   target into a terse note — the claims as bullet fragments, plus for each
   claim the one alternative the passage implicitly rejects. ≈200 completion
   tokens. Prompted to invent nothing; the anchor gate still runs on the final
   output. Once [`transcript-mining.md`](transcript-mining.md) has run, the
   objection should be *sampled* from its `transcript_turns.jsonl` sidecar —
   real pushback Dan actually typed — rather than fabricated, and the brief
   step only has to bind it to the passage.
2. **Write**: the roster model gets the brief in the user turn under the
   `argued` system prompt, framed as "write this up for the collaborator" —
   the regime the real captures come from.

Mechanics: a `brief_then_write(...)` factory in `generators.py` returning a
`Generator` whose `generate` composes the two calls; `GenOutput.meta` sums
`prompt_tokens` / `completion_tokens` / `cost` across both; `prompt_id` hashes
both prompt texts concatenated and `prompt_system` archives both (so
`prompts.jsonl` can still resolve an id); session `history` is passed only to
the write call. Strategy label `argued2` so the two designs facet apart.
Everything downstream (gates, `synth_key`, `lost_draws`, eval) is unchanged.

Do not build stage 2 speculatively — stage 1 is free, and reading 2 says a
prompt may be enough.

## Risks, and what to watch

- **Invented justifications teach over-deletion.** Checked against the real
  captures (2026-09-03): Dan does *not* concede less — `although` / `though` /
  `of course` are as common or commoner on his side (9 vs 3, 33 vs 27, 6 vs 3).
  What his edits strip is the *justificatory* scaffolding: "that is the" 27→7,
  "which is exactly/precisely" 10→1, "which is why" 16→8, "rather than"
  165→102 — usually an argument against an alternative the reader never
  raised, often one that leaked from the editing session. And the real edits
  are not compressions: slop/Dan length ratio is median 1.05, q3 1.22, over
  1.5× in 40 of 398 pairs. So the arm must invent *justifications*, not
  hedges, and the styler learns to drop them, not to shorten. Watch: on the
  frozen eval posts, the styler output's concession-marker rate relative to
  the target's, before and after the arm enters training. A drop past 30% is
  the signal to lower the arm's rotation share or `meta.weight`, not to pull
  it.
- **Proxy Goodhart.** A prompt can hit the rate targets with rhythm alone. That
  is why separability and the eyeball sample are criteria alongside the
  medians, and why punctuation is excluded from the separability vector.
- **The real reference is small and topical.** 398 pairs, mostly from a few
  notebook posts. Compare against the same-posts row, and re-derive it as the
  captures grow.
- **Sampler convergence** on a favourite construction (the registry warning).
  Caught by the ≤3-same-opening eyeball rule; fixed in the prompt text.

## Inputs / outputs / done

- **Inputs**: the blog corpus `pairs.jsonl` (real stratum present), OpenRouter
  key via direnv, the roster in `training_targets.SLOP_MODELS`.
- **Outputs**: `stylebot.features` + `ai-style features`; the `argued` registry
  entry; ~120 audition pairs stamped `slop_strategy: argued`; numbers in
  `findings.md` (audition medians, separability AUC vs the `measured`
  baseline, gate drop rates); on promotion, the `SLOP_STRATEGIES` edit.
- **Done** when the four criteria under "What reliably elicit means" are met
  by stage 1 or stage 2 and recorded, or when both fail and the failure is
  written up in `decisions.md` as the reason the real stratum cannot be
  synthesised this way.
