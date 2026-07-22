# Slice · Blog inference integration: workflow_style + --mask + preference log — 📋 briefed

Three pieces of one seam (the blog's invocation path around
`stylebot.infer`): the batch styling workflow, the masking defense-in-depth,
and the best-of preference logger (Phase-7 fodder). One agent owns all
three; only this slice may touch stylebot (one small hook).

## 1. `workflow_style.py` (livingthing)

The batch consumer of `stylebot.infer` — structurally a one-pass sibling of
`workflow_preen.py`, NOT a modification of it (see the historical
ai-style-fine-tune.md §"Phase 5" for original intent; supersede freely):

- `run_style(files, *, styler, best_of, mask, write)` — per file: read,
  rewrite via `infer.rewrite_text` (the guards come free), stamp frontmatter
  `date-ai-style: <ISO now>` on changed files (ruamel round-trip like
  ai-preen does; do NOT touch date-modified), write with `.bak`.
- CLI: `dan-style style` (register in `bin/dan_style.py`), taking files or
  `--updated-since=<date>` via `workflow_digest.files_changed_since` (same
  convention as ai-preen/format-sentences).
- Reuses `bin/style_file.py`'s manifest-driven backend construction —
  factor that into a shared helper rather than duplicating.

## 2. `--mask` defense-in-depth (livingthing, off by default per D4)

The sentinel mismatch (phase-4 "Open slices"): `STYLE_SYSTEM` promises
preservation of `〈MASKED_*〉`, but `pandoc_mask` emits
`〈CODE_/MATH_/DMATH_/URL_NNNN〉`. Do NOT touch the frozen STYLE_SYSTEM or
pandoc_mask's scheme (ai-preen depends on it). Instead a rename shim in the
style path only:

- Before rewrite: `word_mask.mask_problematic_words` →
  `pandoc_mask.mask_inline_elements` → rename `〈X_NNNN〉` → `〈MASKED_X_NNNN〉`.
- After: rename back → `validate_masking_integrity` /
  `has_leaked_placeholders` → unmask. Any violation reverts THAT CHUNK to
  input (anchor-guard style, visible decision line), never the whole file.
- Wire as `--mask` on both `dan-style run` and `dan-style style`.

## 3. Best-of preference logger (the one stylebot touch)

`--best-of` decisions are exactly Phase-7 preference data (chosen vs
rejected candidates on real inputs). Capture them:

- stylebot: `rewrite_text(..., on_decision=None)` — callback
  `(chunk, context, candidates, scores, chosen_index, kept_input)` invoked
  per prose chunk when candidates were sampled. Pure addition; default None;
  unit-tested with a fake styler (no network). Nothing else in stylebot.
- livingthing: a sink appending JSONL to `_training_pairs/preferences.jsonl`
  (committed, like pairs): `{ts, run_id (styler manifest), context, chunk,
  candidates, scores, chosen, kept_input}`. Wired by default whenever
  best_of > 1 in `dan-style run` / `dan-style style`.
- Document in RUNBOOK (a short §11: "preference data accrues as you use
  --best-of; Phase 7 will consume it").

## Constraints

- File ownership (parallel-agent discipline): this slice owns livingthing
  `workflow_style.py`, `bin/style_file.py`, `bin/dan_style.py`,
  `_training_pairs/RUNBOOK.md` §§ it adds, and stylebot `infer.py` +
  `tests/test_infer.py` (the hook only). Touch nothing else; NO
  `pairs.jsonl`/corpus writes; no policy/roster edits; no git commits —
  implement, test, report (the reviewer commits).
- Keyless: every test uses fake stylers (see `tests/test_infer.py`
  patterns). No paid sampling in this slice.
- stylebot suite must stay green (`uv run pytest -q` from stylebot).

## Done-criteria

- [ ] `dan-style style --updated-since=<date>` styles changed posts with
      guards + frontmatter stamp; `.bak`s left; dry-runnable.
- [ ] `--mask` round-trips a fence/math/link-heavy post with zero sentinel
      leaks (keyless test via identity-fake styler + a mutating fake).
- [ ] `--best-of` runs append well-formed `preferences.jsonl` records
      (schema above); stylebot hook unit-tested.
- [ ] RUNBOOK sections written; stylebot tests green.
