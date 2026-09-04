# Ship-shape runbook, 2026-09-04 — a master agent and its swarm

**Status: specified 2026-09-04, not started.** Written from a four-review design
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
