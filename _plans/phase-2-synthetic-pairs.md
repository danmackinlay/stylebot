# Phase 2 · Synthetic pairs — 📋 PLANNED

Bulk up the corpus by manufacturing `(slop → Dan)` pairs: take Dan's own
real prose as the *target*, ask LLMs to paraphrase it into slop as the
*source*. Lower-signal than real edit pairs, but cheap and plentiful.

**Parallelisable now** — depends only on the `pairs.jsonl` schema (have it) and
a sample of Dan's paragraphs. Does not need a trained model.

## Inputs

- `--blog-root` (read): Dan's authored prose (the ~1.6M-word corpus of
  human-written posts; the slop-free ones are marked in frontmatter). The
  *caller* selects which files; this phase operates on what it's handed.
- `--data-dir` (write): where the resulting `pairs.jsonl` is appended.
- LLM API keys (multi-source by design): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
  and a local/utility base model (`LOCAL_LLM_*`). See `.env.example`.
- `STYLE_SYSTEM` from `stylebot.ai_core` (must match Phase 1 verbatim).

Interface: a `stylebot.synth` function over explicit paths; `ai-style synth`
is the thin CLI wrapper. Paths resolved via `stylebot.config` (flag > env >
default), per OVERVIEW "Interfaces".

## Method (from the post)

1. Sample paragraphs of Dan's own prose → these are the **targets**.
2. For each, ask an LLM "rewrite this passage to be clearer and more polished"
   → the output is the **slop source**.
3. **Multi-source**: rotate across ≥2 generators (Claude, GPT, the local base
   model) so the styler learns to undo AI-writing broadly, not one model's
   tics. Tag each pair with the generator in `meta`.
4. **Secondary set**: explicitly request worst-case patterns from the slop
   catalogue for long-tail mannerisms the paraphrase pipeline underrepresents.
5. **Context-fullness sweep** (optional, post §"Whose slop?"): generate some
   slop across one growing session (fresh paragraph each turn) and tag with how
   full the context was, to study/​weight late-stage slop.

## Outputs (must match Phase 1 schema)

Append to the same `$STYLEBOT_DATA_DIR/pairs.jsonl`:

- `messages[0].content == STYLE_SYSTEM` (verbatim).
- `messages[1]` = synthetic slop; `messages[2]` = Dan's real paragraph.
- Same paragraph-chunk shape as Phase 1.
- `meta` MUST additionally carry: `synthetic: true`, `generator:
  "claude-…"|"gpt-…"|"local-…"`, and (if swept) `context_fullness`.
- Reuse `meta.tags` for provenance (e.g. `["synthetic","paraphrase"]`).

## Done-criteria

- [ ] A few thousand synthetic pairs in `pairs.jsonl`, schema-validated against
      the Phase 1 contract (a validator script that round-trips every record).
- [ ] Pairs from ≥2 distinct generators, distinguishable by `meta.generator`.
- [ ] Idempotent + resumable: re-running doesn't duplicate; tracks which source
      paragraphs are already done.
- [ ] A documented entrypoint, e.g. `ai-style-synth` or
      `uv run python -m stylebot.synth`.

## Risks / notes

- Synthetic slop may not match the real Claude output distribution Dan hits in
  practice → keep mixing in real Phase 1 pairs; consider down-weighting
  synthetic at train time (`meta.weight`?).
- One known failure mode (post): learning the transform but *overcorrecting* —
  looks great on metrics, bad to humans. The eval harness must guard this.
