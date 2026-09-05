# Ship-shape runbook, 2026-09-04 — a master agent and its swarm

**Status: wave 1 landed 2026-09-04 (stylebot `fc56ce7..7e04234`, blog `8fcb92f639..93d3b62c88`); wave 2 landed (stylebot `fae4053..2359469`, blog `93d3b62c88..8a59e037a6`; C6 held on the floor decision); wave 3 landed (stylebot `3261ba6..1aba2c6`, blog `8a59e037a6..22d257ef96`; C6 still held); C6 landed with re-measured floors; wave 4 steps 1–4 landed 2026-09-05 (stylebot `7b1a236..d9a25f3`, blog `22d257ef96..5975e421e2`); step 5 (retire `pairs/`, flip `keep_superseded`, the re-stamp) is held for the owner — it moves the corpus sha by design.** Written from a four-review design
QA of both repos (seam, data contract, mechanism, docs apparatus) run against
stylebot `880a579` (main) and blog `8056cb219c` (master), with both suites green
(917 / 1 skipped; 800 / 2 skipped). The QA transcript is not in git; every
finding below is restated with the symbol that carries it. Companion briefs:
[`seam-consolidation-2026-09.md`](seam-consolidation-2026-09.md) (wave 3),
[`plan-retirement-2026-09.md`](plan-retirement-2026-09.md) (wave 1 evidence,
stylebot side), the blog's `_plans/plan-retirement-2026-09.md` (blog side), and
the blog's `_plans/corpus-draws-and-views-2026-09.md` (wave 4, already
specified). This file retires under P7 when wave 4 lands or is declined.

## 0. How the swarm works

**Roles.** One *master* agent in each repo's main checkout: dispatches, merges,
runs the gates, never edits a slice's files itself. One *slice agent* per slice
below, in its own worktree, on its own branch, committing only named pathspecs
(P8). The *reviewer* is a third party (Claude, in a later session) who checks
each merged wave against §7 and the slice's done-line. Nothing is pushed by
anyone but Dan.

**Worktrees.** stylebot slices: `git worktree add ../stylebot-wt-<slice>`,
`uv sync` there is cheap. Blog slices: the blog's `pyproject.toml` pins stylebot
as `{ path = "../stylebot", editable = true }`, resolved relative to the
pyproject, so a blog worktree under `.claude/worktrees/` cannot `uv sync`.
Place blog worktrees as siblings, `~/Source/livingthing-wt-<slice>`, so
`../stylebot` is the main stylebot checkout; budget several minutes and GBs per
worktree for torch. A slice that needs a change on both sides lands the stylebot
half into stylebot `main` first; the blog half then runs against it. Every
cross-repo slice below is therefore two sequential commits, stylebot then blog.

**Every slice brief must carry**, before dispatch: inputs (files, env, keys —
"keyless" unless stated), the change, a *done-line* (one command and its
expected output), and a *budget note* (P8: what number it clears, and that it
adds no module, thread, lock, process or fallback; or the D-number that
approves one). The master refuses a brief without a done-line.

**Merging.** The master merges a slice branch only when: both suites are green
in the slice worktree; `git diff main..<branch> --stat` names only files the
slice's commit subjects cover (P5); `tests/test_budget.py`'s allowlists show no
increased number; `git status` in the blog shows no `.env`, `.envrc`,
`pairs.jsonl` or `pairs/` (CLAUDE.md "Never commit"). Merge with
`git merge --ff-only` after `git rebase main` in the worktree; a conflict goes
back to the slice agent, never resolved by the master.

**Reporting.** A slice agent that finds something out of scope reports it in
its final message under "Out of scope"; the master appends it to §8 of this
file. Nobody fixes what they were not sent to fix.

## 1. Wave 0 — baseline (master, ≤ 10 minutes, keyless)

Record these in the master's first message so the reviewer can diff them:

```sh
cd ~/Source/stylebot && uv run pytest -q | tail -1
cd ~/Source/livingthing && uv run pytest -q | tail -1
cd ~/Source/livingthing && uv run dan-style status
cd ~/Source/livingthing && uv run ai-style validate-pairs _training_pairs/pairs
cd ~/Source/livingthing && uv run ai-style check-splits --splits _training_pairs/splits.json --blog-root . --pairs _training_pairs/pairs
```

No number from that output is written here (D52): the master's wave-0 message
is the baseline the reviewer diffs against. Today validate reports 0 problems
and 0 duplicates, check-splits passes, and `status` prints one warning (the
promoted run predates the D42 pins). **The corpus `content_sha256` must not
move until wave 4 step 5**; any wave that moves it has rewritten the corpus
and is reverted.

## 2. Wave 1 — docs and test mechanics (parallel, all keyless)

Docs-only slices touch no `src/` file; they can share one committer in the main
checkout if worktrees are scarce, one commit per slice.

- **T1 · stylebot plan retirement.** Retire `eval-harness.md`,
  `v2-longer-rewrites.md`, `synth-flag-audit.md`, `anchor-integrity.md`,
  `vscode-marker.md` to ≤ 40-line P7 stubs from the draft stubs and evidence in
  [`plan-retirement-2026-09.md`](plan-retirement-2026-09.md); move the "unique
  numbers" that file lists into `findings.md` as P4 entries (one table, ≤ 12
  lines) and cite them from the stubs; write the three "why"s the evidence
  file says have no D-entry (gap-coalescing not windowing; the three honest
  analyses of the nested scale arms; the 2026-08-13 synth keep/defer
  reasoning) as D-entries, ≤ 30 lines each, after T6 has made room; repoint
  the 13 docstrings and the one `ai_style.py` help string that cite plan
  sections by name at the D-entry or the `git show` sha; move eval-harness
  §"Planned" into `argued-slop-arm.md` (no `stylebot.features` exists). Then
  delete `plan-retirement-2026-09.md` in the same commit. Done: `wc -l` each
  stub ≤ 40; the inbound references the evidence file lists all resolve;
  `uv run pytest tests/test_budget.py -q` green. Budget: removes ~2,400 doc
  lines including the evidence file.
- **T2 · blog plan retirement.** Act on the blog's
  `_plans/plan-retirement-2026-09.md` Table 1, row by row: seven *stub*
  (shipped code plans, incl. `styler-run4-preflight-2026-08.md`, whose header
  is false on three counts the table cites, and
  `placeholder-design-for-parsers-2026-08.md`, a strict subset of stylebot's
  copy, which becomes a ≤ 8-line pointer); three *abandoned-stub* (≤ 10
  lines, keep the 4-row summary the table names for
  `embeddings_benchmarking.md`); nine *leave + status line* (the proposed line
  is in the row); two *content* left alone, one with a status line. Unique
  numbers the table lists go to stylebot `findings.md` (stylebot commit) or
  are dropped where the row says so. Delete the evidence file in the same
  commit. Done: every row's action applied; every "inbound links" cell
  resolves; `uv run pytest tests/test_budget.py -q` green.
- **T3 · stylebot stale claims** (after T1, since it edits OVERVIEW's table).
  `OVERVIEW.md`: file table rows for the five stubs; add `decisions-archive.md`
  to the table; delete the two private numbers leaked from the blog's
  `transcript-mining-sizing.md` ("+~70% on the real stratum", "~290 clean
  pairs"); replace the 3,981 / 982 / 4.05 / 1,660 / 2,212 passage in "Open
  questions" with a citation of the `findings.md` heading that holds the
  table; the masking bullet in "Open slices" still says "store
  mostly-masked", which D33 reversed — restate from D33; D21's "remaining
  open proposals" (none remain) and D44's dependence on vscode-marker's
  "alternative B" get their pointers fixed. `CLAUDE.md`: drop the "12,277
  lines" anecdote (it does not reproduce
  from `git log --numstat`; D58 owns the story); reword "`STYLE_SYSTEM` is
  frozen" to "frozen: a change means re-stamping every record, done once
  (D64)". `decisions.md` D73: delete the sentence saying the blog RUNBOOK still
  names `prune-invented-anchors` (blog `987e3e38ba` removed it); reconcile its
  "0 of 8,156" with the blog stub's "0 of 8,148" by citing `findings.md`
  instead. `README.md`: move the "1,757 files, 1,728 identical" measurement to
  `findings.md` or delete it. Done: `rg -n "12,277|70%|290 clean|3,981|1,757"
  CLAUDE.md README.md _plans/OVERVIEW.md _plans/decisions.md` empty; CLAUDE.md
  ≤ 160 lines; budget test green.
- **T4 · blog stale claims.** `CLAUDE.md`: `dan-style` row lists all nine
  subcommands (`eval extend-eval run serve status style synth train
  train-clf`) and `ai-style log` adds `recut`; the `dan-style` row's "Needs the
  `styler` extra" sentence says the extras are for `train-clf`/`train`/`serve`
  only. `.claude/skills/voice-marker-install/SKILL.md`: the pointer to
  `stylebot/_plans/slice-vscode-rewrite.md` (nonexistent) becomes
  `stylebot/_plans/vscode-marker.md`. `.claude/skills/markdown-protection/SKILL.md`:
  replace each paragraph the triage table marks as a retelling of D26–D30 with
  a one-line citation; keep the operational content. `_training_pairs/README.md`:
  the `scores.jsonl` bullet says the top-level file is retired (gitignored,
  stale since 2026-08-12) and verdicts live under `runs/<id>/`. Done: `uv run
  dan-style --help` lists what CLAUDE.md lists; `rg slice-vscode-rewrite
  .claude` empty; budget test green.
- **T5 · stylebot comments the code no longer honours, and D49's revert.**
  `synth_cli.py` comment citing D30 for the direct-API preset → D21.
  `bin/ai_style.py` `serve_cmd`'s `make_styler` comment ("caught by serve's
  `except ValueError`") → `serve.handle_request` catches `Exception`; keep the
  `ConfigError` re-raise only if a test shows it changes the reply, else
  delete it. `backends.build_styler` docstring: D47 vocabulary (`ConfigError` /
  `RunError`), not `ValueError`/`RuntimeError`. `--max-workers` help: Vale is a
  batched pre-pass (D59); only the judge is concurrent. D49: move `import
  asyncio` to module scope in `synth.py` (its `_draw` and `_draw_past_gates`
  docstrings are currently expression statements, `__doc__ is None`); amend
  D49 to "reverted 2026-09-xx: the deferred import cost two docstrings; the
  4.9 ms is below the re-evaluate line". If a revived docstring exceeds 15
  lines, trim it; do not allowlist. Done: `uv run python -c "from stylebot
  import synth; assert synth._draw.__doc__"`; `rg -n "D30" src/stylebot/synth_cli.py`
  empty; both budget tests green.
- **T6 · archive D36–D48** (first, before any slice writes a D-entry).
  `decisions.md` is at 1,266 of its 1,300-line cap and D36–D48 are 378 lines;
  move them to `decisions-archive.md`, leave index rows, keep the archive index
  consistent. Waves 2–4 write at least D78–D82. Done: `decisions.md` ≤ 900
  lines; `uv run pytest tests/test_budget.py -q` green; no D-number appears in
  full in both files.
- **M1 · budget-test mechanics, both repos.** (a) Over-budget function
  ceilings keyed by qualified name (`ast` has it), not by sorted position, so a
  139-line function replaced by a new 100-line one is flagged. (b) A module
  that comes *under* budget prints a hint instead of failing, so a good
  refactor is not red until its allowlist edit lands; alternatively CLAUDE.md
  P5 gains "an allowlist edit rides in the commit that earns it" — the master
  picks one and records it in D58. (c) Add a P4 gate for `findings.md` (entry
  ≤ 12 prose lines after its table) or strike P4 from CLAUDE.md — owner
  decision §6. (d) Mirror (a)–(b) into the blog's `tests/test_budget.py`,
  which is a copy by design. Done: a test that a renamed over-budget function
  fails; both suites green; the allowlist numbers unchanged.

## 3. Wave 2 — independent code fixes (parallel worktrees, keyless)

Each is one commit, one concern, no new mechanism. Where a slice lands in code
that wave 4 later deletes, it is marked *interim* and still worth doing: the
window it closes is on the irreplaceable stratum or a paid path.

- **C1 · blog: delete the optional-stylebot machinery** (stylebot is a hard
  dependency since 2026-09-03, D30 superseded). Delete `styler_dep.py`, the
  `_StylerGroup` / `_unavailable` branch in `bin/dan_style.py`, the
  `try/except ModuleNotFoundError → StylerNotInstalled` wrappers in
  `styler.py`, `bin/style_file.py`, `bin/train_styler.py`,
  `bin/train_targets.py`, `voice_classifier.py`, and
  `tests/test_styler_optional.py`. The one reachable failure — a missing
  `[classifier]`/`[trainer]` extra after `uv sync --no-group styler` — must
  surface as stylebot's own `MissingExtra`-class error through
  `translate_errors`, named in a one-line click error. Done: `rg -n
  "styler_dep|StylerNotInstalled|is_missing_stylebot" src tests` empty; a test
  that blocks `sentence_transformers` and asserts `dan-style train-clf
  --dry-run` exits with the extra named; blog suite green; `bin/` allowlist
  numbers in `test_budget.py` shrink or vanish.
- **C2 · stylebot: `log save` appends under the corpus lock** (*interim*; D60
  says every writer takes it, `capture.capture_pairs` does not — its
  `append_jsonl` after the locked `rewrite_with(drop_prior)` is a second,
  unlocked write). Route the append through `corpus_lock` or `PairsWriter`;
  no new lock. Done: a test that runs `rewrite_with` and `capture_pairs`
  concurrently and finds both results in the shard; suite green.
- **C4 · stylebot: a judge refusal is not a 3.** `eval.parse_judge_reply`:
  no JSON and no digit → `{"score": None, "parse_error": True, "rationale":
  reply}`; delete the bare-digit-in-prose fallback. `eval._aggregate` and
  `verdict` exclude `None` and report `n_unscored`; `report` shows it.
  Done: `parse_judge_reply("I cannot evaluate this passage.")["score"] is None`;
  `parse_judge_reply("There are 3 paragraphs; I refuse.")["score"] is None`;
  a fixture with one refusal changes `n_unscored`, not the mean.
- **C5 · stylebot: the reranker's identity rides on the output row.**
  `infer.rewrite_pairs_file` and `docscale._document_record` stamp
  `meta.reranker` (the D39 instrument identity, or null when `best_of == 1`).
  `verdict.compare_scores` refuses `signal == "detector"` when the row's
  reranker identity equals the scoring identity, with the message naming
  `--compare-signal judge`. Done: a test for the refusal; a `--best-of 1`
  output row carries `reranker: null`; suite green.
- **C6 · blog: `tests/test_corpus_stability.py` points at the sharded corpus.**
  `CORPUS = tt.pairs_path()`; when the blog checkout is present the guard
  `pytest.fail`s rather than skips. Done: `uv run pytest
  tests/test_corpus_stability.py -rs` shows 0 skipped.
- **C7 · blog: `dan-style train-clf` passes `blog_root=`** so
  `classify_train.train`'s D34 missing-pin gate runs, as `train` already does
  since 2026-09-04. Two lines in `voice_classifier.train_pass`; superseded by
  wave 3 but cheap now. Done: a test that a missing pinned post raises
  `RunError` through `dan-style train-clf --dry-run`.
- **C12 · stylebot: `check_splits` flags every corpus source missing from
  disk**, not only pinned eval posts (a renamed hash-pool post changes role
  silently, D34). It already builds `corpus_posts`; add the check and the
  alias hint. Done: a fixture corpus with one renamed source fails
  `check_splits`; `ai-style check-splits` on the live blog still prints 12/12
  ok and 0 missing sources.
- **C13 · stylebot: anchor keys stop at trailing punctuation.** In
  `infer` the bare-URL pattern strips trailing `.,;:!?)` (CommonMark's autolink
  rule) and the citation pattern requires `(?<![\w.])@` so an email local part
  is not a key. Then re-measure the "Dan rewrites disqualified" share the
  `findings.md` anchor entry reports and record the new number as a P4 entry.
  Done: `see https://x.com.` → `see https://x.com,` passes `anchor_check`;
  `dan@x.name` yields no key; the findings entry cites the command that
  produced the new share.
- **C14 · stylebot: a provider failure during a gate redraw is one lost draw,
  once.** In `synth._draw_past_gates`, a redraw that raises writes a
  `kind="error"` lost-draw and returns a sentinel so `_run_session` skips its
  own gate accounting; the exception reaches `result.errors` and
  `errors_by_generator`. Done: a test with a generator that fails on its
  second call asserts one `lost_draws` row, one error, `gate_redraws == 0`,
  cost charged once.
- **C18 · stylebot: `serve` says it is one request in flight.** Add the sentence
  to `serve.py`'s protocol docstring and to `vscode-voice-marker/README.md`; the
  asyncio variant needs a D-entry and is not in this runbook. Done: docstring
  present; no code change.
- **C19 · stylebot: `infer` tells the truth about whitespace.** Normalise
  CRLF to LF before segmenting and say so in `rewrite_text`'s docstring, or
  capture lead/trail as `\s` runs; the identity styler round-trips
  `"Para one.\r\n\r\nPara two.\r\n"` and a trailing-two-space hard break.
  Done: those two round-trip tests pass.
- **C20 · stylebot: the async scoring entry point documents its blocking
  prepasses**, and `scoring._score_record` no longer hands the async judge to
  `score_candidate`'s inline fallback (delete the fallback). Done: a test that
  a missing `judge_by_field` slot raises rather than storing a coroutine.

## 3b. Wave 2 — the cull (parallel with wave 2; measure before deleting, P8)

Where the lines are, measured 2026-09-04 against stylebot `880a579` (re-measure
in the slice, record the number once in `findings.md`). stylebot `src/` is
18,100 lines: 13,309 code and 4,791 docstring/comment (ratio 0.36). The CLI
layer (`bin/*.py`, `cli_options.py`, `synth_cli.py`) is 4,005 of them, and
2,318 test lines exercise that plumbing. Back-compat is *not* the story: 30
lines in stylebot and 37 in the blog match `legacy|compat|deprecated|shim`.
Dead code is not either: every top-level function is referenced from `src`, a
click decorator, the blog, or tests. The volume is surface breadth (554
functions, 834 tests at 19 lines each) and one day's mechanism: the 2026-09-03
seam pass alone added `eval.py` +1,129, `session.py` +871, `migrations.py`
+651, `bin/ai_style.py` +548, `splits.py` +483, `status.py` +416,
`bin/ai_style_data.py` +390, `backends.py` +358, which D58 already calls an
overbuild. Waves 3 and 4 remove the largest shares (the duplicated command
bodies; the eight rewrite paths and the migrations). Two slices are deletable
now under rules that already exist:

- **X1 · spent migrations** (CLAUDE.md: "delete function, command and test
  together once its corpus has run it", D38). `migrations.migrate_captures`,
  `migrate_snapshots`, `migrate_capture_archives` and `ai-style data
  migrate-captures`: zero legacy JSON archives remain under the blog's
  `captures/`. `shard_pairs_cmd` and `ai-style data shard-pairs`: the blog's
  `pairs.jsonl` no longer exists, `pairs/` does. Keep `prune-stale` (RUNBOOK
  §2a) until wave 4 step 4 deletes it. Delete the matching tests in
  `tests/test_migrations.py` and any `log recut` demand for `migrate-captures`.
  Done: `uv run ai-style data --help` lists only `prune-stale`; the
  `bin/ai_style_data.py` and `migrations.py` allowlist entries shrink or go;
  suite green; the blog RUNBOOK §1 sentence about `migrate-captures` deleted
  (blog commit).
- **X2 · features only tests call.** `report.render_comparison_report`
  (130 lines), `eval.pangram_detector` (73; the 2026-09-03 QA found the
  endpoint dead, and D7 keeps Pangram only as an optional cross-check) and
  `infer.invents_anchors` (7) are referenced from nowhere in `src`, the blog
  or a click command. Delete each with its tests; amend D7 in three lines
  ("Pangram cross-check deleted 2026-09-xx: endpoint dead, no caller"). Done:
  `rg -n "render_comparison_report|pangram|invents_anchors" src tests` empty;
  `report.py` comes under 800 and its allowlist entry goes.
- **X4 · stylebot test cull** from [`test-audit-2026-09.md`](test-audit-2026-09.md),
  which classifies all 836 tests: 643 keep, 123 low-signal duplicates, 33
  decorative, 20 tautological, 17 performative; ≈ 2,185 of 15,600 lines
  deletable (14%). The signal problem is larger than the volume problem:
  **0 of 14 defects the QA found would have failed a test, and five tests
  pin the defect as the specification** (`test_eval:
  test_judge_reply_parser_clamps_and_falls_back` asserts the bare-digit
  fallback C4 removes; `test_synth:test_gate_redraw_failure_does_not_cost_the_session`
  asserts `errors == []` on the path C14 fixes; `test_splits:
  test_aliases_need_no_migration_of_existing_pairs` asserts the hash pool's
  silence C12 ends; `test_status:test_promoted_manifest_takes_the_newest_completed_non_nixed_run`
  asserts absent-key-means-promoted, which wave 4 §2 reverses; `test_pairs:
  test_pre_provenance_record_still_validates` pins a half-stamped
  `system_sha` as valid). Those five are *rewritten by the slice that fixes
  the defect*, never deleted here. X4 itself: delete the D/T/P rows and the
  L rows' named twins; merge the six segmenter files to three as the audit's
  table says (`test_markdown.py` canonical, `test_segment_hard.py` the
  invariant runner over `fixtures/hard/`, conformance keeps its four
  policy-row tests, parity becomes `test_infer_segment.py`, `test_segment.py`
  and `test_div_policy.py` go); move the `record_id` tests to `test_pairs.py`
  and the `diff_chunks` tests to `test_capture.py`; drop the seven
  `importorskip("torch")` tests in `test_export.py` that are green by
  skipping, or put torch in a test extra and run them. Done: test count falls
  by the audit's D+T+P+L total ± 10 with every deletion citing its audit row
  in the commit body; `uv run pytest -q` green with 0 skipped; the audit file
  deleted in the same commit.
- **X5 · blog test cull** from the blog's `_plans/test-audit-2026-09.md`
  (137 stylebot-facing tests: 84 keep, 22 duplicates, 19 decorative, 6
  unreachable, 4 performative, 2 tautological; ≈ 800 of 3,700 lines, 22%).
  `test_styler_optional.py` goes with C1, and its
  `test_is_missing_stylebot_ignores_other_modules` (which asserts a missing
  `tinker` is not a problem) is replaced by C1's missing-extra test; the
  four `test_serve_refuses_*` twins of stylebot's `test_infer` go; the D46
  double-walk pin becomes a three-line `rglob` assert until wave 3 lane C
  deletes it; the five identical 16-line `_run()` harnesses premised on "no
  pytest installed" go (one sits mid-file in `test_workflow_style.py` and
  would hide eight tests); C6 fixes `test_corpus_stability.py`. Add the one
  test the audit says would have caught three defects at once: keyless
  `dan-style train-clf --dry-run --blog-root <tmp>` asserting the splits
  summary, the D34 pin error, and the resolved data dir. Done: as X4, with
  the blog suite at 0 skipped except the two documented ones.
- **X3 · the option surface** is the owner decision in §6 (`synth` 50 → ≤ 20
  flags); it deletes `opt()` fields and their `test_synth_cli.py` cases, not
  library code, and is the one cull that needs Dan's list first.

## 4. Wave 3 — seam consolidation (sequential chain, keyless)

The brief is [`seam-consolidation-2026-09.md`](seam-consolidation-2026-09.md):
promote `eval_cmd` (with `_document_pass`), `train_clf_cmd`, `run_cmd` and
`serve_cmd` bodies from `bin/ai_style.py` into typed stylebot library functions
taking an `opt()` bundle plus policy hooks plus `emit`; delete the blog's
re-implementations in `voice_classifier.py` and `styler.py`; `EvalOpts` /
`TrainClfOpts` bundles so no `dan-style` flag carries a literal default; the
parity test compares `default`, `type` and `multiple`; one data-dir callback
that sees both `--data-dir`/`--pairs` and `--blog-root`; `iter_targets` accepts
`files=` with `blog_root=` supplying label root and selector, so the blog's
`QUARTO_SKIP_DIRS` copy and the D46 divergence pin (the blog's
`tests/test_qmd_walker.py`) go, with the blog's `is_valid_qmd_file` recomposed
over stylebot's for its three other callers. Lane A is sequential because every
step touches `bin/ai_style.py`; lane B follows it step by step; lane C (walker)
runs in parallel worktrees. Its D-entries D78–D80 are written before its code
(P8) and need T6 to have made room. It also adopts the blog's `--field` choice
list, since `ai-style eval` refuses `output` while `run --pairs` documents it.
Done-lines
are in the brief; the wave-level line is: every `dan-style` command body ≤ 20
lines, `rg -n "0\.25|== 1|== 0" src/livingthing/voice_classifier.py` empty,
`rg -n QUARTO_SKIP_DIRS src/livingthing` empty, both suites green, allowlists
shrunk.

## 5. Wave 4 — draws and views (sequential, the design change)

The blog's `corpus-draws-and-views-2026-09.md` is already specified with a
five-step migration and done-criteria; its stated prerequisites (F0, F2, F8)
have shipped. It is the designed answer to three findings this QA left as
*interim* fixes — the archive/record double truth, the unlocked capture append,
prune-at-read scores — so those are not re-fixed here; they are deleted with
`corpus.rewrite` at its step 4. **This QA adds four inputs the plan must absorb
before step 1 is dispatched:**

1. **Interim archives.** `capture_pairs` mints a random `capture_id` and a new
   archive on every save; `drop_prior` replaces pairs but never archives; so
   six (source, `snapped_at`) sessions today hold two or three archives each,
   and `log recut` has already re-emitted superseded `--keep-open` interim
   saves as pairs. Layer 1 of the new model is that archive, so the plan's
   "captures/ is byte-identical" criterion conflicts with the data being
   wrong. Owner decision (§6): dedupe to the final save per session before the
   backfill, or keep every save as a draw with a `superseded_by` field.
   Either way the archive is keyed on the session, as `snapshots-closed/` is.
2. **Promotion.** `status.promoted_manifest` serves the newest manifest whose
   `promoted` is not `False`; `train` never writes the key; three of ten
   manifests are `dry_run: true`. Step 4 (the manifest pins `policy_sha`) is
   the moment to make promotion opt-in (`promoted: true` set by an explicit
   `dan-style promote`, or a committed `runs/SERVING` pointer) and to stop
   `--dry-run` writing into `runs/`. Needs a D-entry amending D74.
3. **Score views keyed on signal.** `scores-<policy>-<detector>.jsonl` rows
   are keyed on (`draw_id`, field, signal, instrument), so adding `--judge` to
   a detector-scored view appends rather than no-ops (today's RUNBOOK §4
   footgun), and the D39 continuity guard runs *before* any prune.
4. **`meta.system_sha` is on 9 of 8,157 records.** Under the new model the
   view carries `STYLE_SYSTEM` and the draw carries nothing; the backfill drops
   `messages[0]` from draws and the half-stamped field goes with it (D36 and
   D64 both amended).

Dispatch one agent per migration step, each ending in the step's done-criterion
from that file; the master runs `dan-style status` after each and the content
sha must be unchanged through step 4. Step 5 changes it by design, and the
reviewer signs that off against the step-1 byte-identity test.

## 6. Owner decisions (block the slices they name; everything else proceeds)

- **Interim archives** (wave 4 §1): dedupe or keep-with-`superseded_by`.
- **Promotion mechanism** (wave 4 §2): explicit `promote` command, or a
  committed pointer file.
- **The three open questions** at the end of `corpus-draws-and-views-2026-09.md`
  (commit views?; raw output in the draw?; the five unreachable cells).
- **P4**: gate `findings.md` like `decisions.md`, or strike the rule (M1c).
- **P5 versus the allowlist** (M1b): hint-not-fail, or an explicit exemption.
- **`synth`'s 50 options**: demote the design/replication knobs
  (`--sample-salt`, `--replicate`, `--assign-seed`, `--per-generator`,
  `--provider-sort`, `--sticky-provider`, `--prompt-cache`,
  `--capture-reasoning`) to library-only fields set from `training_targets`,
  target ≤ 20 flags. The mechanism exists (`InspectOpts.report_max_rows` did
  it); the choice of knobs is policy.
- **Blog-agnosticism**: stylebot's `STYLE_SYSTEM` names Dan, `.qmd` and
  `automation` are defaults not parameters, and version 0.3.0 has one
  editable-path consumer. Either write "danbot's engine, split out for
  hygiene" into OVERVIEW and stop paying the generality tax, or a
  `SourcePolicy(glob, meta_field, max_level)` and an author slot at the next
  re-stamp (wave 4 step 4 is that moment). Not dispatched until decided.

**Decided 2026-09-04 (Dan):** allowlist edits ride the commit that earns them (M1b, option B, D58); `findings.md` gets a P4 gate (M1c, option A); X3's eight knobs as listed; interim archives keep every save with `superseded_by` (checked read-only first: all six sessions are interim, see §8); promotion is an explicit `dan-style promote` (D74 amended); views are not committed, the raw generator output stays in the draw, the five null-provenance cells are kept; blog-agnosticism is a `SourcePolicy` plus an author slot at the wave 4 step 4 re-stamp. Added to wave 2: **C21 · blog: archive immutability guard** — a committed per-archive hash manifest with a test that existing archives never change, and a pre-commit check refusing modified or deleted paths under `captures/`; the six archives `c0f8fbfa6b` edited are *not* restored.

## 7. Acceptance, per wave (the reviewer's checklist)

- Both suites green in both main checkouts; test counts not lower than
  wave 0's unless a commit subject says which tests it deleted and why.
- `validate-pairs` 0 problems / 0 duplicates; `check-splits` 12/12; `status`
  content sha unchanged (until wave 4 step 5).
- `git log --format='%h|%s|%b' main@{wave-start}..main` — every subject ≤ 72,
  every body ≤ 12 lines, every commit's files covered by its subject; no
  commit by an agent touches a file outside its slice.
- `tests/test_budget.py` allowlists in both repos: no number increased, entries
  deleted where a module came under budget; no new module over 300 lines
  without a D-entry cited in its docstring.
- No new D-entry over 30 lines; `decisions.md` ≤ 1,300 lines.
- Nothing pushed (`git status -sb` shows `ahead` in both repos until Dan
  pushes); `git status` never shows `.env`, `.envrc`, `pairs.jsonl`, `pairs/`.
- Each slice's own done-line, run by the reviewer, not read from the report.
- Tests: every deleted test cites its audit row in the commit body; a test
  the audit marks "pins the defect" is rewritten in the fixing slice's
  commit, with the new assertion stated; no slice that fixes a QA defect
  merges without a test that fails on the pre-fix code (the reviewer checks
  by reverting the source hunk under the new test).

## 8. Not in this runbook, and out-of-scope reports

Deliberately out: the asyncio `serve` (D-entry first); Fireworks serving;
masking (`masking-placement.md`, blocked on a classifier refit); the argued
slop arm and transcript mining (Dan's calls, per the seam stub); per-passage
weighting (a paid retrain). The master appends slice agents' out-of-scope
reports below this line, dated, one line each.

- 2026-09-04 (seam brief): the blog's `bin/train_styler.py` `train_cmd` is a
  fifth copy of `ai_style.py`'s `train_cmd` echo block; a successor slice to
  wave 3, same shape (D78), not folded in.
- 2026-09-04 (seam brief): the blog's `styler.chunk_diet` is a thin twin of
  `backends.serving_diet`, pinned by tests; fold when wave 3 lane B3 touches
  `styler.py`, or report again.
- 2026-09-04 (T4): the blog's `bin/dan_style.py:30-33` docstring and `pyproject.toml:100` comment still describe stylebot as optional; `styler_dep.INSTALL_HINT` names `--extra styler` where the docs say `--no-group styler`; C1 deletes the first and third, the comment is a blog docs fix.
- 2026-09-04 (T4): `voice_classifier.eval_pass` hardcodes `backend="tinker"` for `eval --documents`, so unlike `run`/`style`/`serve` it has no `--backend openai` path; wave 3 lane B1 touches that function.
- 2026-09-04 (M1): stylebot `capture.py: recut_archives` measures 98 lines under its recorded 99; `ruff format --check` fails on both repos' `tests/test_budget.py` and 13 stylebot files (pre-existing; `ruff check` clean).
- 2026-09-04 (T2): the blog's `model-roster-2026-08.md` optional "§Gotchas into the head" trim was not done; stylebot `v2-longer-rewrites.md:29` cites the run-4 preflight stub for evidence that now lives only in `findings.md` "v2 post-gate distributions".
- 2026-09-04 (T5): `PROSE_CEILING` sits at zero headroom (4688 of 4688); the next docstring under `src/stylebot` must be paid for in the same change.
- 2026-09-04 (T1): blog `training_targets.py:176-182` restates findings numbers (a P1 copy); `synth_cli.py:162` "(synth-flag-audit C2)" could cite D83; no test pins D40's "the verdict path imports with no extras".
- 2026-09-04 (T3): `findings.md` disagrees with itself on the corpus size across one day (8,244 on 2026-09-03 vs 8,156 on 2026-09-04; the blog's seam stub says 8,148) and none says what it counted; D73's "49 real captures" is a copied number; OVERVIEW's transcript-mining sizing now has no home in either repo.
- 2026-09-04 (wave-4 pre-check, read-only): the six multi-archive sessions are all interim saves of one session (Dan's keep-with-`superseded_by` holds); no save order is recorded anywhere (`meta.json` has no timestamp, `iter_archives` orders by random id); a recut today re-emits superseded saves and a floor-failing archive keeps stale records through `capture.py:545-547`; the corpus holds byte- and near-duplicate real pairs from this; one final save (more_measured, 2026-08-06) followed an AI edit pass by four minutes; blog commit `c0f8fbfa6b` (2026-09-01) edited `before.md`/`after.md` inside six archives — archives are mutable in practice, hence the new C21 guard.
- 2026-09-04 (C12): `bin/ai_style.py:919` says "N pinned eval post(s) missing" though the count now includes corpus sources; the no-`--pairs` branches (`ai_style.py:906`, `status.py:225`) cannot run the new check; `require_pinned_posts` (`train.py:737`, `classify_train.py:772`) still gates paid runs on eval posts only.
- 2026-09-04 (C13): `infer.py:319` `content_anchors` docstring copies "6.4% → 19.1%" (P1; findings now strikes 6.4%); three synthetic pairs carry a generator-un-escaped `\@handle` (D32's reverse-edit artefact) — a synth-side check is a separate concern.
- 2026-09-04 (C4): no D-entry records the refusal contract or `SCHEMA_VERSION` 3 → 4; `ai_style.py:511` (`eval --by`) and blog `voice_classifier.py:364` print judge means without `n_unscored`.
- 2026-09-04 (C14): `lost_draws` rows write `attempt=st.gen_retries` on the error path where gate rows write the draw index; `synthesize_pairs`/`_run_session` measure 126 under their 128 rows.
- 2026-09-04 (C19): `markdown.py:198` `_find_div_blocks` uses `splitlines(keepends=True)` while `line_kinds` uses `split("\n")` — a lone `\r` counts lines differently.
- 2026-09-04 (C20): `_score_record`'s `judge` parameter is now only a flag and is typed `Judge | None` against a `Judge | AsyncJudge | None` caller; `_score_slice` writes the line outside its per-record `try`.
- 2026-09-04 (C5): at `best_of == 1` the D20 do-no-harm guard still lets the detector choose between input and sample, and the row reads `reranker: null`; blog `eval --documents` routes no reranker; C5 landed as three commits, and the first's body still says `infer.py` stays at 863 (it keeps X2's 854).
- 2026-09-04 (X1): stylebot `README.md:76,227`, blog `_training_pairs/README.md:85` and blog `bin/train_styler.py:82` still name the deleted `ai-style data shard-pairs`; `bin/ai_style_data.py`'s "Why hidden" docstring restates D38.
- 2026-09-04 (X2): the cull raised the prose ratio (0.4084 → 0.4116) because what went was code-heavy; `PROSE_CEILING`/`RATIO_PIN` could ratchet to 4,630.
- 2026-09-04 (X3): the surface landed at 41 flags, not §6's "≤ 20"; D83's "Kept … Deferred" lines want a one-line amendment naming the demotion.
- 2026-09-04 (C1): the missing-extra test passes on the pre-fix code too (the old `except ImportError` already named the extra), so C1 does not meet §7's fails-before rule — it is a deletion, not a defect fix; the four K-class lean-install boundary tests went with `test_styler_optional.py` and are unpinned; stylebot `export.py:130,211` raise a plain `ImportError`, not `MissingExtra`; `pyproject.toml:105-107` still describes `[classifier]` as optional.
- 2026-09-04 (C7): the D34 gate raises `ConfigError` (exit 2), not the `RunError` the runbook names; on a real run it now fires twice (`train_pass` and `classify_train.train`).
- 2026-09-04 (C21): `_training_pairs/README.md`'s older "can be fixed by hand" sentence and blog `CLAUDE.md:111`'s hook description predate the append-only check.
- 2026-09-04 (C6, held): with its skip-guard fixed the corpus-stability test fails both arms (2998/3005 and 4108/4130 against floors 3000 and 4148); bisected to the second eval prune (`d15aeb993d`) and D69 (`89e06e7`), both decided; the floors are the corpus owner's to re-measure — owner decision pending.
- 2026-09-04 (master, process): three agents (C4, C13, X1) reported "waiting on the suite", stopped, and later resumed to commit while a finisher agent worked the same worktree; every duplicate was a no-op and every diff was verified identical, but "stop before commit" is not a state the master can see. X1 used one `git stash`/`pop` (clean). C5's compare_scores hit 80 lines only because C4 and C5 both grew it; the trim is its own commit.
- 2026-09-04 (X4): `src/stylebot/markdown.py:28` names `tests/test_segment.py` and `test_protection_parity.py` (now `test_markdown.py` / `test_infer_segment.py`); `bin/ai_style_log.py:40`'s `noqa` comment says tests reach `log.diff_chunks` (they no longer do); `findings.md:381` cites `tests/fixtures/segmenter_conformance.qmd` (now under `fixtures/hard/`); `test_infer_segment.py`'s packing test carries measured numbers in its docstring (P4); `test_classify_train.py` still `importorskip`s numpy/sklearn.
- 2026-09-04 (X5): audit defects 2 and 4 are live in `src/` (`train_voice_clf.DEFAULT_PAIRS` is a zero-arg callable, so `--blog-root` does not move the corpus default; `train_pass` never prints the role summary) — pinned as a strict xfail `test_train_clf_dry_run_reads_the_corpus_under_blog_root` for wave 3 B5; `test_the_promoted_manifest_arms_a_styler_at_its_own_diet` reads the live `runs/` and hardcodes 4500; X5 deleted M1d's `test_a_renamed_over_budget_function_is_flagged` on the blog audit's P verdict, which reverses a wave-1 done-line — the named-ceiling mechanism stands, the unit test does not.
- 2026-09-04 (A0): the seam brief's §6/C2 named `test_the_synth_walker_sees_exactly_what_this_repo_calls_a_post`, which X5 had already replaced; T6's done-line "`decisions.md` ≤ 900" was met at 888 and then overtaken by D78–D83 (1,074 now, gate 1,300).
- 2026-09-04 (A1): `eval_cmd`'s `ImportError → ClickException` wrap around `--detector-model` was dead (the embed import is lazy; a missing extra surfaces per record in `result.errors`) and was dropped, not moved; the `--documents-role` refusal no longer names the splits file.
- 2026-09-04 (A2): two `train-clf` failures changed exit code 1 → 2 (`ConfigError` through D47: empty corpus names `--pairs`; the holdout exclusivity check); `classify_train.py`'s row is 964 after a P1-duplicate comment went to make room for two constants.
- 2026-09-04 (A3): `backends.rewrite_document`'s "not both" refusal can name a FILE that was not passed; the runbook's A3 blog check names `tests/test_styler.py`, which does not exist.
- 2026-09-04 (B1/B2): a bundle default pinned to a callable renders as `[default: (dynamic)]` in `--help`; the counting gate's line now reads `detector_pair_ok: dropped …` with no threshold shown (stylebot prints `pair_ok.__name__` verbatim); `--max-synthetic-automation` left the blog CLI for `tt.DEFAULT_MAX_SYNTHETIC_AUTOMATION`; the old "point `--splits` at a missing file to opt out" route is unreachable behind stylebot's `exists=True`; `voice_classifier.py` re-exports twelve `classify_train` names nothing imports; one unnamed blog test failed once in a full run and never again (possibly order-dependent).
- 2026-09-04 (B3): `styler.chunk_diet` is still the thin twin of `backends.serving_diet`, pinned by `test_chunk_diet_falls_back_for_pre_v2_manifests`; `RunOpts.best_of`'s help says "needs --detector-model", which the blog pins; `dan-style run`'s body is 21–22 lines against the brief's ≤ 20 (two lazy imports); `serve.preference_recorder` stamps `ts` with `stylebot.provenance.now_iso` where the blog used `ai_core.now_iso_utc` — formats not compared.
- 2026-09-04 (B5): `train-clf`/`extend-eval`'s `--blog-root` are stylebot `opt()` fields and so not `is_eager`; correct today because click processes an invoked flag before any defaulted one, but D79's "every `--blog-root` is eager" needs `is_eager=True` on `classify_cli.py:52` and `splits.py:549`; `train_styler.py:100` still computes the manifest home from the cwd (`default_runs_dir(blog_root=blog_root)` is the one-line fix).
- 2026-09-04 (W1/W2): the blog's `lib.py` comment above `LOCAL_SKIP_DIRS` narrates the 2026-08-11 fixture-stamp incident (P1); the recomposed predicate now stats the path (`not path.is_dir()`), harmless for its four callers.
- 2026-09-04 (B4): three literal shadows the default/type comparison now pins rather than repairs — `styler.blog_backend_options` defaults serve's `--backend` to a literal `"tinker"` over stylebot's `None` (score-only serving); `--blog-root` is spelled `"."` on synth/train and `Path(".")` elsewhere; status's `--blog-root` adds `exists=True` where stylebot's does not.
- 2026-09-05 (S1a): `corpus.rewrite(dir, …)` on a directory that does not yet exist writes a single file named after it (`_write_locked`'s `is_dir()` check); every fresh-shard caller must `mkdir` first.
- 2026-09-05 (S1b): the synth shards carry no `policy_digest` key (only the nine recut captures do), so `chunk_policy_digest` names the live policy, not something the old rows can be checked against; `build-view --pairs` has `exists=True` and must change when step 5 retires `pairs/`; once stylebot grows `ai-style build-view`, `build-view` leaves `BLOG_ONLY` in the parity test.
- 2026-09-05 (S2): a native synth draw carries `span=None` until the cutter records offsets, so chunks of one source under one arm share a `draw_id` (S4a decides); transient retries inside `_draw` are not draws (only draws that arrived or finally failed are); `<data-dir>/draws` is joined only in `run_synth` — D45 wants a `config.draws_path` (S4a adds it); the paid check: 4 accepted + 2 gate draws for $0.002, view sha equal to pairs, live corpus untouched.
- 2026-09-05 (S3a/S3c): real draws first came back month-grouped (view sha `51d2cbab04780985`); fixed by routing a draw with a record through `corpus.shard_of`; `migrations.backfill_draws`'s docstring still says "month shards"; `synth.py:849` says "its month shard".
- 2026-09-05 (S3b): the backfill recovered 7,748 of 8,157 blog shas (7,741 at the two pinned revisions, 7 by a history walk; 2 synthetic cells unresolved, not the design's 5); 174 of 407 real records got a span into `before.md`, 233 have no archive; `draws/` is 72 MB, 2.0× `pairs/` (the accepted draw carries `record`, `target_text` and `raw_output`); **7 duplicate `draw_id`s, all real records** — a real draw's key has no capture id, so an interim save and its successor editing the same paragraph collide; one pair (`d89d26a4` → `f608a742`, `hyperscaling_coase.qmd`, 2026-09-04) is a ninth interim session not in `SUPERSEDED_CAPTURES`; `_lift_pair` crashed on `replicate: "verbose1"` (374 records) and was fixed on the way; the clock rolled to 2026-09-05 mid-run.
- 2026-09-05 (S4a): `PROSE_CEILING`/`RATIO_PIN` went to `None` because the package met the 0.40 ratio (the test's own ratchet); native draws key on `chunk_index` rather than source offsets (`_post_targets` cuts a transformed stream through five stages and is allowlisted at 95 lines); `ensure_view` rebuilds on a draws-hash change as well as a policy change; the live `runs/` serves the same `20260817-170037` under the explicit rule; **score views keyed on signal are still open** — smallest design: `views/pairs-<policy_sha>/scores-<instrument>.jsonl`, rows keyed on `(draw_id, field, signal)`, `eval` upserts by that key with the D39 guard before any write (~60 lines in `scoring.py`).
- 2026-09-05 (S4b): `stylebot.draws.view_options()` takes no `defaults=`/`exclude=`, so the blog reaches into `__click_params__` to re-pin `--draws` and drop `--policy`; `train`'s manifest home now follows `--blog-root`; `train-clf --dry-run` reports 265 posts / 3,550 pairs where `status` reports 286 / 4,009 (the classifier's own gating, pre-existing).
- 2026-09-05 (S4c): `Draw.supersedes` is a tuple (a `--replace` save supersedes several sessions); a real draw's `span` is the before-chunk's offset in the archived `before.md`; with no `draws/` in a data dir `capture_pairs` still appends to `pairs/` (the old-layout branch step 5 removes); the design's done-line pattern `rg "def rewrite|corpus_lock"` matches the styler's `rewrite_*` functions and needs the corpus-specific form; the eval `scores.jsonl` prune stays (D63).
- 2026-09-05 (step 5, held for the owner): retire `pairs/` from git and flip `view_policy(keep_superseded=False)` — both move the corpus sha by design; the re-stamp with `SourcePolicy` and an author slot (D36/D64 amended, `meta.system_sha` dropped); `PairsWriter` draws-only; `build-view --pairs`'s `exists=True`; the 7 duplicate real `draw_id`s want `capture_id` in a real draw's key first, and the ninth interim session (`d89d26a4` → `f608a742`) wants a `SUPERSEDED_CAPTURES` row.
- 2026-09-05 (S5a): `draws.policy_sha` raises `TypeError` if any of `retired_arms`, `excluded_draw_ids` or `superseded_captures` holds a literal `None` (the pre-existing `sorted(d[key])`); `ruff format --check` still reports 9 pre-existing stylebot files; `capture.py:376`'s "W4 step 5 removes that branch" comment is item 3's to retire.
- 2026-09-05 (S5b): the stylebot half alone turns blog `master` red — `test_view_policy.py` enumerates every `ViewPolicy` field and asserts each is filled, so a new stylebot field fails the blog suite until the blog half lands; a cross-repo slice's two commits are not independently green, and §0's "stylebot first" rule should say so.
- 2026-09-05 (master, item 1): §9's literal instruction — give `draws.draw_id` a `capture_id` — was not done: the live capture path keys a real draw on `pairs.record_id` (`capture_id:chunk_index`) and never collided; the 7 duplicates are artefacts of the retired backfill, which keyed on the span. The exclusion went in as D84's policy-clause route instead of hand-appending successor draws into the committed `captures.jsonl`, so step 5 writes nothing to the corpus.
- 2026-09-05 (S5c): `dan-style status`'s splits block still reads `pairs/` while its view block reads the view, so the two disagree by 12 until item 3 retires `pairs/`; `dan-style` has no `check-splits` mirror, so §9's `check-splits --data-dir` line cannot be run as written (`ai-style check-splits` wants `--splits` and an explicit `--pairs` view path); `status.py:519` prints `(unbuilt)` unconditionally, even for a view already on disk; the new `test_view_policy.py` constants pin literal shas and 8,145, so the next appended capture moves all three.
- 2026-09-05 (item 3, stylebot): `session.py:125-128,281-286` still carry the old-layout `corpus_path`/`read_pairs` fallback, now reachable only for a data dir nothing has saved into, and its "until W4 step 5" note is stale; `corpus.shard_path_for` has zero callers in either repo; `ai-style eval`, `make-splits`, `validate-pairs` and `train-clf` still have no `--draws`, so only `train`, `status` and `check-splits` are draws-reachable; `bin/ai_style.py`'s per-role table moved to `splits.role_lines` to pay for `--draws` (a refactor riding a feature commit, the alternative being an allowlist rise); 9 files remain not ruff-format-clean.
- 2026-09-05 (item 3, blog): `dan-style status` now builds the view as a side effect of resolving its splits corpus, while `stylebot/status.py:519` still prints a literal `(unbuilt)` beside the view path; `build-view --pairs` defaulted to `view_pairs_path`, not `pairs_path`, so the `exists=True` failure the runbook predicted never fired — the default was repointed at `pairs_path` as well as losing `exists=True`; `check-splits --draws` builds under the identity policy rather than `view_policy()`, so its styler count is 4,007 where `status` says 4,006, and it leaves a second gitignored view behind — either `--policy` needs a way to dump `view_policy()` or `dan-style` needs a `check-splits` mirror.

## 9. Hand-off — for the next master (written 2026-09-05, after wave 4 step 4)

**Read first, in order:** `CLAUDE.md` (P1–P8 bind you), this file (§0 roles, §5 wave 4, §6
decisions, §8 reports), D84/D74 in `_plans/decisions.md`, the blog's
`_plans/corpus-draws-and-views-2026-09.md`. Everything below is verifiable from the trees.

**State.** stylebot `main` `d842d69` (ahead 74; `uv run pytest -q` → 728 passed, 2 deselected).
Blog `master` `671e237579` (ahead 33; 755 passed). `cd ~/Source/livingthing && uv run dan-style
status` shows `policy_sha=42c1ceefe1fd0e4c`, `content_sha256=d49ba7ad60774ea6`, serving
`20260817-170037`. Nothing is pushed; only Dan pushes. Dan's tree has uncommitted prose
(`friends.qmd`, `notebook/hyperscaling_coase.qmd`, `topic_map.json`): never stage it.

**Your job.** Wave 4 step 5, in this order, one slice agent per item, each in its own worktree:

1. **Real-draw key.** stylebot: `draws.draw_id` takes `capture_id` for a real draw (today two
   saves of one paragraph collide: 7 duplicates, see §8). Blog: add `"d89d26a4": "f608a742"`
   to `training_targets.SUPERSEDED_CAPTURES`; re-run nothing — the backfill is deleted; write
   the two draws' ids by a one-off `uv run python` over `_training_pairs/draws/captures.jsonl`
   only if the key change needs the file re-keyed (it will: re-key in place is forbidden by D84,
   so append successor draws that `supersedes` the old ones instead). Done: `duplicate draw ids`
   is 0 when you count `draw_id`s over `iter_draws`; the view sha is unchanged.
2. **Flip the view.** Blog: `view_policy()` drops `keep_superseded=True`. This changes the corpus
   sha **by design** (11 superseded real records leave the view). Record the new sha in
   `findings.md` as a P4 row; `dan-style status` is the instrument. Done: the step 1 test's
   pairs-lifted assertion is deleted in the same commit with the body saying why; the
   `view_pairs_path()` test asserts the new sha.
3. **Retire `pairs/`.** Blog: `git rm -r _training_pairs/pairs` (git holds it), `.gitignore`
   lines 68–95 about `pairs.bak`/`*.bak`. stylebot: `PairsWriter` writes draws only;
   `capture.capture_pairs` loses its "no draws dir → append to pairs" branch; `build-view
   --pairs` loses `exists=True`; `ai-style check-splits` gains `--draws` (it takes only `--pairs`);
   RUNBOOK §2's `check-splits --pairs` line. Done: `rg -n "pairs/" src/stylebot | grep -v
   views` names only the view layout; both suites green; `dan-style status` reports the view.
4. **The re-stamp.** stylebot: `SourcePolicy(glob, meta_field, max_level)` and an author slot in
   `ai_core.STYLE_SYSTEM` (D64 amended: the second change ever; D36 amended: `meta.system_sha`
   dropped — the view carries the system sha, the draw carries none). The view rebuild is the
   restamp (D84's rule); no migration. Done: `rg -n system_sha src/stylebot` names only
   `pairs.system_sha()` and the view sidecar; the blog passes its policy and author.
5. **Score views keyed on signal** (§8, S4a's design): `views/pairs-<policy_sha>/scores-
   <instrument>.jsonl`, rows keyed `(draw_id, field, signal)`, `eval` upserts by key, the D39
   guard runs before any write. ~60 lines in `scoring.py`. Done: adding `--judge` to a
   detector-scored view appends judge rows and touches no detector row.
6. A view clause for the old eval-role prune (`prune-stale --no-orphans` had no successor;
   §8 S5-docs item 1): `ViewPolicy.role_rule` already carries the splits sha — add the clause
   that drops synthetic draws on eval-role posts, and say in `findings.md` what it excludes.

**How to run a slice (the mechanics that cost time when skipped).**
- Worktrees: stylebot `git -C ~/Source/stylebot worktree add ../stylebot-wt-<X> -b <X> main`
  then `uv sync --all-extras` there (without `--all-extras` ten tests skip). Blog:
  `git -C ~/Source/livingthing worktree add ../livingthing-wt-<X> -b <X> master` then `uv sync`
  (2 s; it resolves stylebot from `../stylebot` = the main checkout, so a stylebot change is
  live in the blog only after it merges — cross-repo slices land stylebot first, and a blog
  agent tests an unmerged stylebot branch with `uv run --with-editable ../stylebot-wt-<X> pytest`).
- Always `git -C <path>` or an explicit `cd` per command: a bare `cd ../x && …` in a chained
  shell line silently ran three merges in the wrong directory once. The Bash tool is zsh: never
  write `echo ===` (the `=cmd` expansion errors).
- Every brief carries: inputs, the change, a done-line with expected output, a budget note (P8),
  the file list, and the rules block (uv only; named pathspecs; subject ≤ 72 chars by `wc -m`;
  body ≤ 12 lines; no push/rebase/merge; `git status` free of `.env*`, `pairs.jsonl`; no
  allowlist number rises; tests first, paste them failing). Agents that end with "waiting on the
  suite" have stopped: check `git status` in their worktree and send a finisher into it.
- Merge only after: `git -C <wt> rebase main` clean; `git diff main..<X> --stat` names the
  slice's files only; suite green in the worktree; then `git -C ~/Source/stylebot merge
  --ff-only <X>`, suite in main, `git worktree remove`. Blog: the same with `master`.
- After each item: append the agent's "Out of scope" lines to §8 (one dated line each), run
  §7, and post: commit ranges, done-lines with actual output, suite tails, `dan-style status`.

**Not yours.** `.claude/worktrees/*` (Dan's), the 49 slice branches (the reviewer's), pushing,
and anything in §8 you were not sent to fix.
