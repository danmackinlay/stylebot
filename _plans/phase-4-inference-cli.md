# Phase 4 · Inference CLI — 📋 PLANNED

Wrap the trained adapter as a text transformer Dan runs over AI drafts.
Gated on a trained adapter existing (Phase 3).

## Inputs

- A served/loadable LoRA adapter (Fireworks-hosted, scale-to-zero).
- `FIREWORKS_API_KEY` (or whichever serving path Phase 3 settles on).
- `STYLE_SYSTEM` from `stylebot.ai_core` (the same system prompt seen in
  training — must match, or the adapter sees out-of-distribution input).

## Target UX

```sh
uv run ai-style run <file>    # rewrite a draft into Dan's voice, to stdout / --write
```

A subcommand of the one `ai-style` entry point (see OVERVIEW "Interfaces"),
a thin CLI over a `stylebot.infer` library function the blog build can import.

- Chunk the input the same way Phase 1 chunks (paragraph regions), transform
  each chunk, reassemble — so inference matches training-time granularity.
- Preserve markdown structure verbatim (code fences, math, links, headings,
  list markers, blank lines) — already mandated by `STYLE_SYSTEM`.
- Preserve any `〈MASKED_*〉` tokens verbatim (per `STYLE_SYSTEM`).

## Outputs

- Rewritten prose (in place with backup, or to stdout — decide; default to
  stdout, `--write` to edit in place).
- Optional: best-of-N candidate selection using the trained voice classifier
  (`p_dan = 1 - score`) as a cheap reward signal — a later enhancement. **Reward
  safety:** train the detector with a held-out by-POST split (`train-voice-clf
  train --holdout-frac/--holdout-posts`) so it isn't fit on this styler's posts,
  and keep the LLM-judge + eyeball as the orthogonal anti-Goodhart guard (a split
  fixes leakage, not over-optimisation). See `eval-harness.md` "Eval vs reward".

## Done-criteria

- [ ] `ai-style <file>` round-trips a real draft: structure preserved, prose
      shifted toward Dan's voice.
- [ ] Scale-to-zero serving confirmed (no idle spend).
- [ ] Eval-harness scores on its output beat the prompt-only baseline.
- [ ] Registered as a `[project.scripts]` entry.
