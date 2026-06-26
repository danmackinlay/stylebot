# stylebot — development plan

The durable source of truth for this project's phases. Code comments describe
*how* a thing works; these files describe *what we're building, in what order,
and how we'll know it's done*. Edit this when decisions change — it should
always reflect current reality, not the original aspiration.

Narrative rationale (the "why") lives in
[`../docs/fine_tuning_danbot.qmd`](../docs/fine_tuning_danbot.qmd). This file is
the operational counterpart: phase map, contracts, status.

## The goal

Train a small LoRA adapter on an open-weights model that rewrites AI-flavoured
draft prose into Dan's voice. Treat it as a *prose styler*, not a chatbot.

**Goals:** less onerous cleanup of AI edits; more comprehensible prose.
**Non-goals:** full Dan impersonation; evading slop detectors.

## How we work (process contract)

This is a **data/ML project**, not just software. Provenance, reproducibility,
and the contracts *between* stages matter more than internal abstraction. So:

1. **Plans live here, in git.** One file per phase, each with explicit
   **inputs / outputs / done-criteria**. Decisions are diffable. A subagent
   should be able to pick up a phase from its file alone.
2. **PRs are per phase-slice, and earn their place with a check.** A slice
   becomes a PR when it ends in something verifiable — a passing test or an
   eval number. Plan edits just commit to the branch; they don't need a PR.
3. **Phases talk through file schemas, not shared code.** Every phase's I/O is
   a documented on-disk format (below). That's what lets phases be built in
   parallel against a few fixture rows, and farmed out to subagents.
4. **The corpus and secrets never enter this public repo.** Corpus lives under
   `$STYLEBOT_DATA_DIR` (gitignored, manually backed up — see
   `_training_pairs/README.md`); keys live in `.env` (gitignored — see
   `.env.example`).
5. **Verify with a number, never with vibes.** "Does the styler work" is
   exactly where an ML project rots. The eval harness is therefore an *early*
   parallel track, not a final step.

## The shared data contract (the seams)

Phases are decoupled by these formats. Change one → update the phase file and
notify downstream.

| Artifact | Producer | Consumer | Format |
|----------|----------|----------|--------|
| `pairs.jsonl` | Phase 1 (real), Phase 2 (synthetic) | Phase 3 | chat-completion JSONL, `{messages:[system,user,assistant], meta:{...}}` — schema in `phase-1-pair-capture.md` |
| trained adapter | Phase 3 | Phase 4, eval | LoRA weights (Together download / Fireworks-hosted) |
| candidate text | Phase 4 | eval harness | plain prose in, plain prose out |
| eval scores | eval harness | humans | JSON: `{vale, judge, detector, ...}` per candidate |

The crucial deliberate choice: **Phase 1 and Phase 2 emit the *same*
`pairs.jsonl` schema**, chunked the same way, so training data from real edits
and synthetic paraphrase is shape-compatible and mixable with a weight column.

## Phase map & status

| Phase | File | Status | Blocked by |
|-------|------|--------|-----------|
| 0 · Scaffolding | this repo's pyproject/.env/_plans | ✅ done | — |
| 1 · Pair capture | [`phase-1-pair-capture.md`](phase-1-pair-capture.md) | ✅ shipped (daily use) | — |
| 2 · Synthetic pairs | [`phase-2-synthetic-pairs.md`](phase-2-synthetic-pairs.md) | 📋 planned | corpus schema (have it) |
| 3 · LoRA training | [`phase-3-training.md`](phase-3-training.md) | 📋 planned | enough pairs |
| 4 · Inference CLI | [`phase-4-inference-cli.md`](phase-4-inference-cli.md) | 📋 planned | a trained adapter |
| E · Eval harness | [`eval-harness.md`](eval-harness.md) | 📋 planned | only sample prose |

## Sequencing — what's parallel vs serial

- **Serial, done-first:** Phase 0 scaffolding (✅). Unblocks everything.
- **Parallel now (each needs only the schema + sample paragraphs):**
  Phase 2 synthesis, the Eval harness, and the detector audition inside it.
  None of these need a trained model. **Build the eval harness early** — it's
  the ground truth every later phase reports against.
- **Data-gated tail:** Phase 3 (needs corpus volume) and Phase 4 (needs an
  adapter). These wait on *data*, not on engineering.

## Subagent-friendliness checklist

Before farming out a phase, confirm its file has:
- [ ] explicit inputs (which files/env/keys) and outputs (which files/schema);
- [ ] done-criteria stated as a number or a test, not "looks good";
- [ ] a runnable entrypoint (a `ai-style-*` subcommand or script);
- [ ] enough fixture data to build against without other phases existing.

## Open questions

- Single-source vs multi-source slop ablation — worth publishing? (post §"Whose slop?")
- Down-weighting synthetic vs real pairs — what weight? (needs a `meta.weight` field?)
- Which base model: Qwen3 8B first, 70B if headroom is short.
- Pangram vs open-weights detector — settled by the audition in `eval-harness.md`.
