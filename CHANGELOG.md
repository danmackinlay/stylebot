# Changelog

Notable changes to `stylebot`. Rationale for *why* a choice landed belongs in
[`_plans/decisions.md`](_plans/decisions.md) and measurements in
[`_plans/findings.md`](_plans/findings.md); this file answers the narrower
question an importer has: **what broke, and what do I change?**

Pre-1.0, so breaking changes land in minor versions.

## 0.3.0 — unreleased

A debt-clearing pass. The policy behind most of it: **prefer a small codebase
over back-compat — users migrate on a version increment.** Nothing here changes
what the pipeline computes; the corpus, the splits and every trained artifact
are untouched.

### Breaking

| change | migration |
| --- | --- |
| Legacy single-blob capture archives (`captures/<id>.json`) are no longer read | run `uv run python scripts/migrate_capture_archives.py <data-dir> --apply`. `capture.iter_archives` now **raises** naming that command, rather than skipping — a silent skip is indistinguishable from "that capture had no changed regions", which is the failure the old warning existed to prevent. |
| `scripts/restamp_style_system.py` and `scripts/prune_stale_synth_pairs.py` deleted | each was written for one schema event (the 2026-08-04 `STYLE_SYSTEM` amendment; the 2026-08-10 segmentation change) and each has run against the only corpus that exists. If a future change needs one, write a fresh one-off — that is what the `scripts/` carve-out is for. `migrate_capture_archives.py` stays: it is the only path off the legacy archive format. |
| `capture.write_pairs_atomic` removed | import `write_jsonl_atomic` from `stylebot.jsonl`. It was a bare re-export that hid where the function lives; the atomic rewrite is a JSONL-level operation, not a pair-level one. |
| `stylebot.tinker_tls` removed, and the `train._patch_pyqwest_tls_trust` alias with it | nothing to do — delete any call. The `trainer` extra's floor is now `tinker>=0.25.0`, which carries the fix upstream. |
| `synth.covered_target_bodies` → `covered_target_keys` | it returns normalised coverage keys now, so the old name lied. See the `--skip-covered` note below. No consumer outside this repo. |
| `synth --drop-invented-anchors` defaults **on** | a synth run now drops pairs whose target carries a link its slop lacks, where 0.2.x wrote them. Pass `--keep-invented-anchors` for the old behaviour. Pair it with `--gate-retries` — see Added, and note the drop rate is roster-dependent (1.3%–51% across arms currently in the roster). |
| `synth_cli.run_synth` no longer prints — it is **silent without a sink** | pass `emit=synth_cli.click_emitter()` to keep CLI output, or read the returned `SynthResult`. A click wrapper that does neither goes mute *and still exits 0*, so this is worth grepping for: livingthing's `dan-style synth` needed exactly this change. |

### Removed

- **`stylebot.tinker_tls`** — the monkey patch on the tinker SDK's private
  transport factory, and its call sites in `train.py`, `infer.py` and
  livingthing's `bin/styler-mlx`. tinker 0.25.0's own
  `_default_pyqwest_transport` passes `tls_include_system_certs=True`, and
  guards that call against older pyqwest rejecting the kwarg — a `TypeError`
  our workaround never handled. Verified by rebuilding the SDK's *own* default
  transport unpatched and completing a live handshake against
  `tinker.thinkingmachines.dev` (HTTP 200; the same probe fails
  `UnknownIssuer` at the old pin). `tinker-cookbook` was unaffected — its floor
  is `tinker>=0.22.3`, so it neither blocked nor was dragged by the bump, and
  all 17 cookbook APIs `train.py`/`infer.py` use resolve unchanged. See D25 in
  [`_plans/decisions.md`](_plans/decisions.md).

  **Unexercised:** verification covered the TLS handshake and import/signature
  parity, not a paid run. The first real `ai-style train` under 0.25.0 has not
  happened.

### Fixed

- **`is_valid_qmd_file` tested the filename, not the path**, so `_draft.qmd`
  was excluded while `_tmp/scratch.qmd` and `_freeze/…` were walked, and
  `_site` survived only by being hardcoded beside `renv`. The `_`/`.` rule now
  runs over every path segment (which subsumes that hardcoding) and the skip
  set becomes `QUARTO_SKIP_DIRS`, mirroring upstream rather than a two-name
  guess. `is_valid_qmd_file` also takes an optional `base`: the rules are about
  *project-relative* structure, so a checkout under a `_`- or `.`-prefixed
  directory should not disqualify its own contents. `gather_qmd_files` passes
  it on the glob branch.

  This one is worth reading twice, because the predicate feeds `iter_targets`:
  a scratch copy of a post was being offered to `synth` as prose to generate
  training pairs from, and `synth_key` hashes target text, so the pair would
  have been indistinguishable from a real one afterwards. **The corpus is
  clean** — re-derived at `PINNED_BLOG_REV`, both stability floors hold at
  3681/417 — but only because the blog's scratch copies carry `automation: 2`
  and `is_human_authored` rejects them. The defence that held is not the one
  anybody designed. The suffix stays unchecked on purpose: the walk's glob is
  caller policy (`iter_targets(glob=…)`, `ai-style synth --glob`). Found and
  designed blog-side; see livingthing `_plans/qmd-walker-2026-08.md`, whose
  differential test against `quarto inspect` is the guard this predicate cannot
  carry generically.

- **The anchor guard checked only half the failure, and the wrong half.**
  `infer.missing_anchors` flagged a candidate that *dropped* a link, URL or
  citation and said nothing about one that *invented* one — while the corpus
  teaches invention ~7× more often. Measured over the 4,595 pairs through the
  production path: 117 pairs (3.1% of the anchor-bearing ones) carry a target
  that lost an anchor, **855 (18.6% of the corpus) carry one that gained an
  anchor its source never had**, half of those external URLs whose destination
  no model can know. Reverse-edit synthesis produces this directly — a generator
  that drops a link while "improving" a paragraph becomes, after the swap, an
  example reading *given text without links, produce text with these links*
  ([stylebot#1](https://github.com/danmackinlay/stylebot/issues/1), D32).

  `infer.anchor_delta(source, candidate)` now returns `(lost, gained)` and
  `rewrite_text` disqualifies either. **The halves use different rules on
  purpose**: loss is a count *decrease* (each occurrence anchors a separate
  claim), invention is key *absence* (repeating a URL the source supplied
  conjures nothing). `missing_anchors` and the new `invented_anchors` are thin
  faces on it, so existing callers are unaffected.

  **What to change if you read the counters:** `RewriteResult.n_anchor_rejected`
  now counts disqualifications from *either* cause; `n_anchor_lost` and
  `n_anchor_invented` say which, and overlap when one sample does both.
  `rewrite_candidates` grew an `anchors_gained` key beside `anchors_lost` and
  still flags rather than rejects — that seam hands a slate to a human, who is
  the one able to tell a real URL from a plausible one.

- **`infer.rewrite_candidates` blamed the chunk budget for an empty selection** —
  `""` reported as "selection is 0 chars — over the 1500-char chunk budget".
  Library-API only: `serve`'s rewrite op already rejects empty text, so the
  sidecar and the VS Code extension never saw it.

### Changed

- **`--skip-covered` matches on a normalised coverage key, not exact text.**
  The corpus is a frozen record, so cosmetic drift since a pair was generated
  should not buy that passage a second pair — and two-thirds of measured blog
  drift is mechanical rather than authorial (one link-anchor repair script
  orphaned 71 cells by rewriting `#fragment`s; quote-curling 19 more).
  `synth.coverage_key` folds three things, each invisible to what the styler
  learns: inline markdown link destinations (`STYLE_SYSTEM` has the model
  preserve links verbatim, so a rewritten URL carries no voice — the link
  *text* is kept), curly quotes to ASCII, and whitespace runs. Measured against
  blog HEAD, **92 of the 207 drifted cells now read as covered** (~44% of the
  drift), with **zero** collisions between distinct passages across all 3,685
  corpus bodies — the single collision found is the same passage recorded twice
  either side of a sentence-per-line reflow. Case, punctuation, dashes and
  emphasis are deliberately **not** folded: a false "covered" silently costs a
  passage its only training pair, so each extra rule would trade real risk for
  a smaller saving. `_synth_key` is untouched, so no existing pair is re-keyed.
- **`synth_cli.run_synth` no longer owns its output.** It takes an `emit` sink
  (`emit(message, *, level=…)`) and defaults to silence, so a library caller
  gets a clean stdout/stderr and reads the run off the returned `SynthResult`
  instead of scraping it. The *exception* vocabulary was carved out of click in
  0.2.0; the output channel was not, and ~20 `click.echo`/`click.secho` calls
  stayed behind in a module whose own docstring called itself a library.
  `cli_options.click_emitter()` (re-exported as `synth_cli.click_emitter`) is
  the single adapter both front-ends use — `ai-style synth` and livingthing's
  `dan-style synth` — with the routing in one table, `EMIT_LEVELS`. A level
  names a **(stream, emphasis)** pair rather than a syslog severity, because
  the hygiene-drop counts are yellow on *stdout*: they are part of the run's
  report, not a diagnostic. Rendered CLI output is byte-identical to 0.2.0,
  streams and ANSI colour included (diffed through a pty).
- **`tests/test_corpus_stability.py` runs by default, and now measures
  segmentation rather than Dan's editing.** Two changes.

  It used to skip whenever `LIVINGTHING_ROOT` was unset, so a bare
  `uv run pytest` never exercised the corpus guard. Unset now means *go and
  look*: it probes the conventional locations, skips only when no checkout
  exists anywhere (naming every path it tried), and **fails** when a checkout
  is named or found but unusable. That includes the set-but-missing-corpus
  case, which 0.2.0's note claimed already failed and which in fact skipped —
  so both the unset path and the explicitly-broken path had been reading as
  "measured, fine". Opting out is now `LIVINGTHING_ROOT=` (empty).

  More importantly it re-derives targets at a **pinned blog revision**
  (livingthing `1aa290e463`, 2026-07-22) instead of HEAD. Reading HEAD
  conflated a segmentation regression — the bug this file exists to catch —
  with an ordinary prose edit, so the number sagged on a schedule set by the
  blog's own churn (a single bulk `ai-preen` pass cost 24 cells), the floor had
  to be hand-lowered to absorb it, and each lowering bought less teeth. The
  revision was chosen by sweeping 244 commits and taking the plateau that spans
  the corpus's generation window. Reproducibility against it is 3681/3686
  (99.86%), so the floors become **equalities**: with the prose frozen, nothing
  benign can move the number. A two-cell regression now fails, where the old
  HEAD-relative floor had a two-hundred-cell noise band.
- **Vale runs batched.** `vale_score_many` hands many `*.md` files to one `vale`
  invocation and splits the filename-keyed JSON back apart; `score_pairs_file`
  runs it as a per-slice pre-pass instead of one process per scored text
  (~9,200 spawns on the live corpus → ~36). **18.9× on the Vale signal**,
  measured against the real baseline — which already spread Vale across the
  thread pool, so the tempting 122× serial comparison would have been
  flattering. `vale_score` keeps its signature, return shape and never-raises
  contract as a batch of one; per-line flushing and resumability are unchanged,
  and a batch failure still degrades to `available: False` rather than failing
  records.
- **`targets.iter_targets` parses each post once, not twice.** The `blog_root`
  walk reused nothing from the frontmatter read it did for the selector.
  **~1.8 s off a 1,771-post walk (36%)**, target list byte-identical (verified
  by hash and by a 12-configuration differential run). One deliberate
  difference, log output only: a post with unparseable frontmatter now warns
  once rather than twice under a permissive selector.
- **`classify_train.select_C` scores only each inner fold's validation rows**
  instead of the whole embedding matrix — 224 full-corpus `predict_proba` calls
  per training run became slices. Selected C values and every reported metric
  are bit-identical (verified by full-precision diff over stub-embedding
  fixtures). Worth ~8% of `select_C`, which is less than the call count
  suggests: the lbfgs fits dominate it, and embedding dominates a run.
- **`splits.role_of` scans the pinned `eval_posts` list** instead of rebuilding
  a set on every call. No memo — one was tried and reverted in `870d030` for
  going stale on in-place edits, and the guard test for that stays.
- **CLI defaults are declared once.** `train`, `train-clf` and `make-splits` had
  each default written three times: `default=None`, a hand-written
  `[default: X]` in the help string, and an `x if x is not None else DEFAULT_X`
  at the call site, with nothing tying them together — a library constant could
  move and `--help` would keep lying. click owns them directly now
  (`default=tr.DEFAULT_LORA_RANK, show_default=True`). The genuinely computed
  defaults (`--max-tokens`, `--max-chunk-chars`, `--run-id`, `--lr`,
  `--renderer`) keep `None` as a meaningful sentinel. **No default value
  changed.**

### Added

- **`ai-style run --anchor-log PATH`** — appends per-chunk anchor accounting to
  a JSONL sidecar (`infer.write_anchor_audit`, populated from
  `RewriteResult.anchor_audit`). Anchor keys and lengths only, never candidate
  text, so it stays joinable without becoming a second copy of the draft. The
  counters were computed, printed to stderr and discarded on every run, which
  is why the question the masking work is gated on — *how often does an
  unmasked candidate lose or invent an anchor?* — had never been answered.
  Denominator: candidates not marked `truncated`. Numerator: those with a
  non-empty `lost` or `gained`. A truncated sample gets no anchor verdict at
  all, since its delta measures the token budget rather than the model.

- **`ai-style synth --drop-invented-anchors`, on by default** — write-gate in the
  `--max-transform-sim` idiom (count loudly as `skipped_invented_anchors`, write
  nothing, leave the cell un-keyed so a re-run retries it) for pairs whose target
  carries an anchor its slop never had. It defaults **on** where its two siblings
  default off, because they take a threshold whose right value is a roster
  property while this one is a boolean with no defensible other side. Pass
  `--keep-invented-anchors` to reproduce pre-0.3.0 behaviour. For the 855 pairs
  already written, `infer.is_anchor_faithful` is a `selector` for
  `train.assemble_training_corpus` — the corpus is a frozen record, so they are
  excluded from the training *signal* rather than migrated away, and
  `_synth_key` is untouched.

- **`scripts/prune_anchor_inventing_pairs.py`** — the one-off cull for a corpus
  written before the gate existed. Dry-run by default, `.bak` beside the
  rewrite, atomic. **Defaults to synthetic pairs only**, because the two strata
  are different propositions: a culled synthetic cell leaves its `synth_key`
  missing and the next `synth` run refills it, while a captured pair is a record
  of a real edit that nothing but git brings back. They also fail differently —
  51.4% of what the synthetic stratum invents is an external URL (a generator
  artifact), against 11.5% for the real stratum, which is mostly Dan adding
  internal cross-references and citations while editing. `--include-real` culls
  those too; `infer.is_anchor_faithful` excludes them from training without
  culling. On the live corpus: 4,595 → 3,790 (synthetic only) or 3,740 (both),
  with 657 of 3,685 targets emptied and therefore due for regeneration.

- **`ai-style synth --gate-retries N`** — redraws a cell a hygiene gate rejected,
  up to N times, before giving up on it. Applies to all three gates.
  **Set this if you run any gate.** Anchor invention is a property of the arm,
  not the passage (1.3% to 74% across nine measured arms; `findings.md`), so a
  bare gate burns roughly a fifth of a run's spend on cells it drops and
  re-drops every subsequent run — and biases coverage *against link-dense prose*,
  which is the prose anchor fidelity matters most for (16.1% rejection at one
  anchor, 32.9% at five or more). A rejected draw is paid for either way, so a
  redraw costs no more than the drop; ~80% of rejections clear on one.
  `SynthResult` gained `gate_redraws` / `redrawn_cells` / `redraws_recovered`,
  and a low recovery rate is the signal that more retries will not help.

- **`stylebot.export` / `ai-style merge-adapter`** — merges a trained LoRA
  adapter back into its base weights, emitting one HF checkpoint a local
  runtime can take from there (`mlx_lm.convert`, llama.cpp, vLLM, plain
  transformers). It exists because stylebot emitted an artifact its own
  documented next step could not consume: `train --download-adapter` writes a
  *serving*-format PEFT export, while `tinker_cookbook.weights.build_hf_model`
  expects *raw* Tinker keys and applies that remap itself, so feeding it our
  own output doubled the `language_model` prefix and failed. Needs the
  `trainer` extra. Refuses an adapter whose `lora_alpha != r`, since the key
  inversion only preserves meaning while the scaling factor is 1. Quantization,
  config-key surgery and chat-template edits are deliberately **not** included
  — they are (model family × serving runtime) policy; see D31 in
  [`_plans/decisions.md`](_plans/decisions.md).
- A test asserting each CLI default **is** its library constant, and that no
  option help in those commands contains a hand-written `[default:`. Their
  absence is what made the duplication dangerous.
- CLI-level coverage for the legacy-archive refusal: `ai-style log recut`
  against an unmigrated archive exits 1 with a legible `Error:` naming the
  migration command, not a traceback.

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
  assistant side. 24 target texts across 15 posts; a one-off prune script
  removed the 29 affected pairs (the script itself was deleted in 0.3.0, once
  it had run).
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
