# Changelog

Notable changes to `stylebot`. Rationale for *why* a choice landed belongs in
[`_plans/decisions.md`](_plans/decisions.md) and measurements in
[`_plans/findings.md`](_plans/findings.md); this file answers the narrower
question an importer has: **what broke, and what do I change?**

Pre-1.0, so breaking changes land in minor versions.

## 0.2.0 — 2026-08-11

Markdown segmentation was implemented five times inside this package and had
silently diverged, breaking the contract that inference must chunk prose the
way training did. This release consolidates it, fixes the five defects that
divergence caused, and tightens several public surfaces that had accreted.

### Breaking

| change | migration |
| --- | --- |
| `stylebot.bin.synth_cli` removed | import `stylebot.synth_cli`. It is a library, not a CLI module — `bin/ai_style.py` is the only click layer. |
| `synth_cli.run_synth` no longer exits the process | it raises `SynthConfigError` / `SynthRunError` / `SynthGenerationError`. Translate at your CLI boundary; see `bin/ai_style.py::_synth_errors` for the exit codes (usage 2, other 1). A caller that ignored the old `SystemExit` now gets a traceback. |
| 9 names dropped from `stylebot.synth`'s re-export facade | import from the implementation home: `SLOP_SYSTEM*`, `SlopStrategy`, `openai_generator` from `stylebot.generators`; `ChunkPolicy` from `stylebot.targets`. The 13 names the CLI kit resolves by attribute lookup are unchanged. |
| 5 names dropped from `stylebot.eval`'s facade | import `extract_slop`, `extract_target`, `extract_output`, `pair_body`, `_recut_digest` from `stylebot.pairs`. |
| `markdown.find_div_blocks` → `_find_div_blocks` | use `protected_spans()`, which is the supported seam. |

### Fixed

- **Blockquotes, `$$display math$$` and `:::` divs were sent to the styler at
  inference** though excluded from training targets. A style rewrite of quoted
  text is misattribution; the model had never seen either construct.
- **Mid-line `$$k$$` corrupted training targets.** `MATH_BLOCK_RE` was not
  line-anchored, so `high-$$d$$ behaviour` became `high-behaviour` in the
  assistant side. 24 target texts across 15 posts; `scripts/prune_stale_synth_pairs.py`
  removes the 29 affected pairs.
- **`~~~` fences** were protected at inference but not in training.
- **Indented ATX headings** (`  ## X`, legal CommonMark) were headings to the
  training path and prose to the inference path.
- **`--inline-headings` injected heading context** the cross-heading diet never
  records, making served chunks out-of-distribution.
- **The HTML scores report's controls were dead** — sort, filter and facet were
  bound to the headline table, which precedes the data table.

### Added

- `stylebot.markdown` — the one segmenter. `protected_spans()` is the seam;
  `block_only`, `div_is_prose` and `raw_html` express policy as parameters so a
  second implementation need not fork.
- `tests/fixtures/segmenter_conformance.{qmd,json}` — a shared spec with named
  configurations, asserted by this repo, the TypeScript port, and (by copy)
  `livingthing.qmd_core`. Three implementations, two languages, two repos, none
  able to drift silently.
- `tests/test_corpus_stability.py` — guards segmentation changes against the
  real corpus. **Fails rather than skips** when `LIVINGTHING_ROOT` is set but
  unusable; a skip here reads as "measured, fine".
- `--strict-anchors` on `ai-style run`, extending the anchor guard to inline
  code and math. Off by default: it roughly triples the rate at which every
  candidate is disqualified (6.4% → 19.0%).
- `BackendOpts` — one declaration behind `run` and `serve`, which had drifted.

### Removed

- `python-dateutil` from runtime dependencies (unused). It is in the dev group
  only, so `test_corpus_stability.py` can import the sibling blog.
