# Eval harness — 📋 PLANNED (build EARLY)

How we know the styler works. Runs **offline** — it scores candidate output and
is *not* wired into the trainer or baked into the served adapter. Build this
early and in parallel: it's the ground truth every later phase reports against,
and it needs only sample prose, not a trained model.

## Inputs

- Candidate prose passed by path/list (styler input, styler output, and
  pure-Dan reference) — the caller supplies the files.
- `--blog-root` (optional, read): to pull pure-Dan reference paragraphs.
- Test corpus (small): ~10 paragraphs of styler input, ~10 of styler output,
  ~10 of pure-Dan prose.
- `PANGRAM_API_KEY` (only if the paid detector wins the audition below).
- Vale + the existing Vale ruleset / slop catalogue.

Interface: `stylebot.eval` functions over supplied files; `ai-style eval` is
the thin CLI. Paths resolved via `stylebot.config`.

## The four signals

1. **Vale** — mechanical slop (banned words, indefinite "you", -ize spelling).
2. **LLM-as-judge** — "is this Dan-shaped" at the voice level.
3. **Statistical detector** — "would an external classifier flag this".
4. **Dan's eyeball** — the veto channel (not automatable; reported alongside).

## The detector decision (the one real choice)

Run an **audition before paying**: test open-weights detectors —
[Binoculars](https://github.com/ahans30/Binoculars),
[Ghostbuster](https://github.com/vivek3141/ghostbuster),
[RADAR](https://huggingface.co/TrustSafeAI/RADAR-Vicuna-7B), OpenAI's
deprecated RoBERTa — against [Pangram](https://www.pangram.com/) on the small
test corpus, in the regime that matters: **single-paragraph, lightly-humanised
text**. Prior: none of the free ones will be good enough at *this* regime, but
the audition is one afternoon.

- If an open-weights detector tracks Pangram well enough → use it, skip the spend.
- If Pangram wins → make it cheap via **distillation**: ~$50 once to label
  ~10,000 paragraphs, then train a small local classifier (DistilBERT-class,
  <100ms) on those labels. That classifier becomes a free reward signal
  callable anywhere on our side (eval, best-of-N at inference, a future DPO
  loop) — and never touches the served adapter. One spend, no more API calls.

## Outputs

- Per-candidate scores as JSON: `{vale: …, judge: …, detector: …}`.
- A small report comparing styler-input vs styler-output vs pure-Dan, so a
  training run can be judged by movement across the three groups.

## Done-criteria

- [ ] All four signals runnable over the test corpus from one entrypoint.
- [ ] The detector audition completed and decision recorded here.
- [ ] Scores emitted in a stable JSON schema that Phase 3/4 can cite.
- [ ] A documented entrypoint, e.g. `ai-style-eval <candidates>`.

## Guardrail (policy, not optional)

This is **not** a slop-detection evader (see post Non-goals). The detector is
useful only because its tripwires — uniform rhythm, hedging, signposting,
generic vocab — are the same patterns the slop catalogue targets. A styler that
lowers its detector score does so *as a consequence* of writing more like Dan.
Watch for reward-hacking (passes-detector-but-still-bad output); the LLM-judge
and eyeball signals exist to catch exactly that.
