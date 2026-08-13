# Where masking goes: synthesis, training, inference — and what gets stored

Step 3 of [`anchor-integrity.md`](anchor-integrity.md), unblocked 2026-08-12 by
the measurement it was waiting for.

**Scope.** This decides *placement*: which pipeline stages mask, what the corpus
stores, and what has to be measured first. It does **not** decide the encoding —
placeholder shape, failure modes, verification labels and the "re-measure per
model" warning are [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md),
which stands unchanged. Read that one for *what a placeholder looks like*; this
one for *where it is applied and what survives into the corpus*.

## The evidence that unblocked it

The gate was deliberately built to answer "is masking needed at all?" before
anyone built masking. It answered, on the first v2 run at 3000-char
cross-heading chunks:

- **On link-dense passages the models fail almost always.** Rejection on the
  3000-budget residue: gemma-4-26b 98% of draws, qwen3.5-9b 99%,
  qwen3.6-35b-a3b **100% (0 written of 105 draws)**.
- **The failure is wholesale, not marginal.** Across 1,215 anchor rejections in
  `lost_draws.jsonl`: median **5** anchors deleted per rejected draw, mean 6.2,
  max **29**. These models are not slipping on one link, they are stripping the
  passage.
- **Redrawing does not help.** 106 cells redrawn, 212 extra draws spent, **2
  recovered**. That is the "poor recovery means something structural" signal, at
  1.9%.
- **The generators are already told not to.** Every slop prompt says *"Preserve
  all markdown structure (code fences, math, links, list markers, blank lines)
  verbatim."* Instruction-following collapses under length and density; there is
  no cheaper prompt fix left.
- **Why it collapses, and why it is worse at v2 scale.** A chunk passes only if
  *every* anchor survives, so P(pass) ≈ s^k in the anchor count k. Cross-heading
  packing takes median k from 2 (v1, 600 chars) to 8 (3000) to 11 (4500). At
  k=10 a model needs per-anchor fidelity of 0.97 merely to pass three runs in
  four. **This is why the earlier "the rejection curve plateaus, so v2 is
  affordable" reading was wrong** (`findings.md`): the plateau was measured on
  v1's short chunks where high-k cases were rare, and the requirement is
  exponential in k, not asymptotic.

Secondary but decisive for the *yield* argument: the residue self-selects for
difficulty, so re-running never converges. At the 3000 budget, cells already
written have median 8 anchors and 51.0% carry 8+; the 101 still missing have
median **10** and **72.3%** carry 8+.

## The three stages, and what each buys

Masking is not one decision. Three stages, three different purposes:

| stage | what it prevents | independent? |
|---|---|---|
| **synthesis** | the *generator* deleting links while producing slop | yes — fixes yield whatever else is decided |
| **storage** (what the pair contains) | — | this is the choice that couples the other two |
| **inference** | the *styler* deleting or inventing links | needs storage support to be in-distribution |

**Synthesis masking is unconditional and separable.** Mask the target's anchors
before sending, and the generator cannot delete what it never saw as a link. It
fixes the 92%-and-stalling yield problem on its own, and it is correct
regardless of what the corpus ends up storing, because unmasking before writing
is always available.

## The asymmetry that is NOT there — read this before proposing one

An earlier draft of this argued that masking is "easier at synthesis than at
inference, because at synthesis you own both texts". **That is wrong and the
error is worth naming, because it is an easy one to re-derive.** You own the
mask→unmask mapping in both places equally; nothing about inference prevents
substituting anchors back.

The real differences are narrower:

1. **Train/serve distribution — the only structural one.** Synthesis masking is
   invisible downstream *if you unmask before storing*: the corpus then holds
   ordinary markdown, which is what the styler trains on and meets at inference.
   Inference masking changes what the styler is asked to do at serving time.
   Measured over the 8,264-pair corpus: `〈MASKED_*〉` appears on the slop side of
   **21** pairs and on the target side of **zero** — the styler has never once
   been asked to *emit* a sentinel, though `STYLE_SYSTEM` has always promised it
   would preserve one. That is a promise the data never kept.
2. **Cost of failure.** Batch retry versus a human waiting — real, but the
   do-no-harm guard already covers it.
3. **Which model.** Large general instruct models at synthesis; a narrow LoRA at
   inference, whose sentinel handling is untested (and which
   `placeholder-design-2026-08.md` explicitly flags as unmeasured).

**And inference masking is already built**, blog-side:
`livingthing.workflow_style.rewrite_masked` (the `--mask` flag) word-masks and
pandoc-masks, renames into the `〈MASKED_*〉` namespace, runs an integrity
pre-check, and degrades to unmasked styling on failure. The unbuilt part was
never the inference wrapper — it is the corpus support that would make it
in-distribution.

## The storage decision: mask everywhere, with a dropout fraction

Store *mostly* masked, keeping a deliberate unmasked fraction. **This is not a
hedge; it is the `CONTEXT_DROPOUT` idiom**, already blog policy for exactly this
shape of reason:

```python
CONTEXT_DROPOUT = 0.1
# A fraction kept heading-less so the styler doesn't require one.
```

Substitute "masked" for "heading" and the argument transfers intact. Two
concrete paths depend on the styler not *requiring* the mask:

- **`workflow_style` degrades to unmasked styling** when masking or its
  integrity pre-check fails. That fallback exists precisely because masking can
  fail — and a mask-only styler would convert graceful degradation into an
  out-of-distribution failure at the exact moment things are already wrong.
- **`ai-style run` ships no masker.** stylebot is public tooling; `pandoc_mask` /
  `word_mask` are blog-side. A mask-only adapter would work only behind the
  blog's wrapper, which inverts the generic/policy boundary the same way the MLX
  export script did before `merge_adapter`.

A benefit under-weighted in earlier discussion, and it argues *for* masking more
than the yield case does: **masking recovers prose budget.** 20.5% of
target-side characters are inline markup (links, code, math, citations). Masked,
most of that collapses to short sentinels — which serves the v2 longer-rewrite
goal directly rather than incidentally, since the section wall means chunk
budget is the scarce resource.

## Prerequisite measurement: the voice classifier

One consumer cannot be fixed by consistency alone, and it is worth being precise
about why, because "just be consistent everywhere" is otherwise a complete
answer:

- the **styler** is fine-tuned, so consistency suffices — feed it masked data
  and it learns masked;
- **Vale** is rule-based, and the **LLM judge** is a large general model, robust
  to sentinels;
- the **voice classifier** is a logistic head on a *frozen* pretrained encoder
  (`StyleDistance/styledistance`). Retraining the head on masked text restores
  consistency but cannot give the backbone a representation for a token it never
  saw in pretraining. Consistency makes both classes equally unfamiliar; it does
  not restore separability.

**This is a measurement, not a blocker.** `dan-style train-clf` is free, keyless
and needs no GPU, and reports cross-validated metrics: fit the head on masked
text, compare against the current one, and only then commit. There is decent
prior reason to expect it passes — link *density* is plausibly one of the
classifier's stronger features (Dan runs a median of 8 anchors per 2,700-char
chunk while the slop arms strip 51–99% of them), and masking preserves anchor
count and position, destroying only the destination string.

The VS Code extension is **not** a concern here: it already talks to
`dan-style serve`, so masking lives server-side and there is nothing to port.
(The TypeScript segmentation port is a separate parity surface, pinned by
`tests/fixtures/segmenter_conformance.*`, and masking does not touch it.)

## Work, in order

0. **Measure the classifier under masking** — `train-clf` on masked vs unmasked
   text. Free. Gates everything below, the way `--anchor-log` gated this doc.
1. **Synthesis masking.** Mask target anchors before generation, validate the
   placeholder multiset on return (`placeholder-design-2026-08.md`'s load-bearing
   conclusion: no scheme guarantees recall, so validating at demask time is the
   part that matters), unmask, then apply the existing fabrication gate — which
   becomes a round-trip check rather than a quality gate.
2. **The masked fraction.** Choose it deliberately and record it as blog policy
   beside `CONTEXT_DROPOUT`. Store masked pairs on both sides for `1 − p`, plain
   for `p`. **Do not guess p by analogy**: `CONTEXT_DROPOUT = 0.1` was chosen for
   headings, and the failure it insures against here (a masking failure at
   serving time) has a different base rate.
3. **Inference.** `--mask` already exists; what changes is that it stops being
   out-of-distribution. Re-check retention against the *fine-tuned* styler
   specifically — `placeholder-design-2026-08.md` warns its numbers came from
   other models and that a fine-tune trained on this corpus shape may behave
   differently again. With step 2 done, it will have been trained on it.

## Do not

- **Do not mask everything with no dropout.** See the two degradation paths
  above; the styler must not *require* preprocessing that is documented to fail
  open.
- **Do not re-derive the synthesis/inference asymmetry.** It is not a capability
  difference. See the section above.
- **Do not change `STYLE_SYSTEM`.** It already promises `〈MASKED_*〉`
  preservation; this work makes the promise true rather than amending it.
- **Do not treat the existing corpus as needing regeneration.** Masking is a
  deterministic, reversible transform over stored text, so a masked *training
  assembly* can be produced from unmasked pairs without re-synthesis. Only the
  yield fix (step 1) needs to happen at generation time.
