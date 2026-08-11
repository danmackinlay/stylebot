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

- **`infer.rewrite_candidates` blamed the chunk budget for an empty selection** —
  `""` reported as "selection is 0 chars — over the 1500-char chunk budget".
  Library-API only: `serve`'s rewrite op already rejects empty text, so the
  sidecar and the VS Code extension never saw it.

### Changed

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
- **`tests/test_corpus_stability.py` runs by default wherever the blog is.**
  With `LIVINGTHING_ROOT` unset it probes the conventional locations instead of
  skipping, so a bare `uv run pytest` actually exercises the corpus guard. It
  skips only when no checkout exists anywhere, naming every path it looked in;
  a checkout that is named or found but unusable now **fails**. That includes
  the set-but-missing-corpus case, which 0.2.0's note claimed already failed
  and which in fact skipped — so both the unset path and the explicitly-broken
  path had been reading as "measured, fine". Opting out is now
  `LIVINGTHING_ROOT=` (empty), since unsetting means "go and look".
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
