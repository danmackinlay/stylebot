# Ship-shape, round 2 — closing what the swarm reported

**Status: specified 2026-09-05; R1 and R0 landed that day (`999e943..17755b3`,
blog `37577b378d`); R2–R7 dispatched in parallel, R7 to merge last.** The reviewer's QA of the round-1
swarm (`ship-shape-2026-09.md`, retired) verified both suites, the corpus gates,
P5 over all 142 commits, the allowlists and every done-line that is a command,
and found the work landed as reported. What remains is the live residue of its
55 out-of-scope reports (`git show 6fcee88:_plans/ship-shape-2026-09.md` §8),
re-checked against the tree on 2026-09-05, plus four owner decisions Dan made
that day. Everything here is a fix or a deletion; there is no data-model change
and the corpus `content_sha256` does not move. Retires under P7 when R1–R6 land.

**Owner decisions, 2026-09-05.** Serving takes the prompt from the manifest, not
the module (R2). Draws are the corpus interface everywhere; `--pairs` survives
for scratch files only (R3). stylebot adopts `ruff format` (R1). `synth` stays
at 41 flags; D83 records it (R1). **P2's package-wide prose/code ratio is
retired for an absolute prose ceiling that only ratchets down** (R0), after the
ratio failed the suite on a five-line de-duplication; and a prose cull runs
this round (R7).

## 0. How this round runs

Same roles and gate as round 1 §0, with three changes learned there:

- **R1 lands alone, first.** Its reformat touches eleven files; everything else
  branches from the formatted tree or spends its time in rebase conflicts.
- **A cross-repo slice is one agent, two worktrees, two commits merged
  back-to-back.** The stylebot half is not independently green (the blog's
  `test_view_policy.py` enumerates every `ViewPolicy` field); the agent tests the
  blog half against the unmerged stylebot branch with
  `uv run --with-editable ../stylebot-wt-<X> pytest` and hands the master both
  branches at once. R1 writes this into CLAUDE.md P8.
- **Every command runs with `git -C <path>` or an explicit absolute path.** The
  Bash tool's cwd persists between calls and drifts; two rounds of QA lost
  batches to it.

Critical path: R1's remainder, then R0 (each ≤ 1 h, sequential: both edit
CLAUDE.md and `tests/test_budget.py`), then R2–R7 in parallel (≤ 3 h). R7
merges last among the parallel slices, because it touches docstrings in files
the others edit; it rebases onto them. Nothing waits on Dan.

## 0b. R0 — prose is bounded by a ceiling, not a ratio (both repos, after R1)

The ratio `(docstring+comment)/code ≤ 0.40` (D58, P2) fails when code is
deleted: `_post_targets`'s `keep_kw` hoist, five duplicated lines out, tipped
the package to 0.4001 and was backed out. A rule that prices out
simplification is the opposite of its purpose. Measured 2026-09-05 at
`999e943`: prose 4,516 (docstrings 3,488, comments 1,028), non-blank code
≈ 11,300.

1. `tests/test_budget.py`: `test_prose_does_not_outgrow_the_code_it_explains`
   becomes `test_prose_stays_under_its_ceiling`: `prose <= PROSE_CEILING`, with
   `PROSE_CEILING` set to the measured count at the commit; a second assertion
   fails when prose is more than 50 lines *under* the ceiling ("lower
   `PROSE_CEILING` to N") so the ratchet is enforced, not hoped for. `RATIO_BUDGET`,
   `RATIO_PIN` and the ratio assertion go. Per-docstring (15), module-docstring
   (30) and function/module line caps are unchanged.
2. CLAUDE.md P2: replace "package-wide (docstring + comment) / code ≤ 0.40"
   with "package-wide docstring + comment lines ≤ `PROSE_CEILING`, which only
   falls" (same length).
3. D58 amended in ≤ 4 lines: what the ratio did, the measurement that showed it,
   and that the ceiling ratchets down on every commit that lowers prose.
4. The blog mirrors 1 and 2 (its `tests/test_budget.py` is a copy by design;
   its CLAUDE.md P8 bullet states the budgets).
5. **Proof:** re-land the backed-out `_post_targets` `keep_kw` hoist (99 → 94)
   in the same slice as its own commit; it must pass with no allowlist edit
   other than lowering `_post_targets`'s number.

Done: `rg -n "RATIO" tests/test_budget.py ../livingthing/tests/test_budget.py`
empty; both suites green; `_post_targets` at 94 in the allowlist; the ceiling
constant equals the measured prose count in the commit body. **Landed
`5d83f6a` (items 1–3), blog `37577b378d` (item 4), `17755b3` (the proof).** The
ceilings are the test's own `_modules()` counts, which differ from the estimate
above; the commit bodies carry them.

## 1. R1 — rules and the reformat (stylebot, then blog docs; first, alone)

1. ~~`uv run ruff format src tests` as one reformat-only commit~~ — **landed
   `999e943`**; D86 records the three allowlist numbers the formatter moved.
   ~~What remains of item 1: `ruff format --check src tests` joins the merge
   gate (§7 below), and CLAUDE.md gains a "Ruff" line~~ — **landed `a5e3e35`**.
2. ~~CLAUDE.md P8 gains the cross-repo sentence from §0 (≤ 2 lines), paid for
   by cutting "Blog integration" to two bullets~~ — **landed `a5e3e35`**, 158.
3. ~~`decisions.md` D83: three-line amendment~~ — **landed `78154dc`**, D83
   tightened by three lines to pay; still 31.
4. ~~`ship-shape-2026-09.md` stub: the step-5 range ends at `ae5dfb6`~~ —
   **landed `ebd78a6`**.

Done: `uv run ruff format --check src tests` clean; `uv run pytest -q` green;
CLAUDE.md ≤ 160 lines; `rg -n "6fcee88" _plans/ship-shape-2026-09.md` names only
the `git show` pointer.

## 2. R2 — the prompt slot reaches serving (stylebot, then blog)

D85's re-evaluate clause, discharged. `infer.py`'s two chat builds (the
`rewrite_text` and `rewrite_candidates` message lists) and `capture.py`'s draw
record emit `ai_core.STYLE_SYSTEM` by name; `backends.manifest_mismatches`
compares against it. Change: the prompt is a parameter. `train` already pins
`data.style_system_sha`; it now also pins `data.style_system` (the text), and
`backends.opts_from_manifest` carries it into `RunOpts`/serving so `infer`
receives `system=` from the manifest, falling back to `style_system(author)`
from the `SourcePolicy` when no manifest is in play (`ai-style run` without
`--manifest`, `log save`). `capture` takes the prompt from the data dir's policy
(the blog's `view_policy().system_prompt`; stylebot's default renders
`STYLE_SYSTEM` byte for byte, so no existing draw changes). `manifest_mismatches`
compares the manifest's text against the policy's, and its message names the
author. Blog: `styler.arm_styler` passes the manifest prompt; `dan-style run`
without a manifest passes `tt.SOURCE`'s author. D85 amended in ≤ 3 lines.

Done: `rg -n "STYLE_SYSTEM\b" src/stylebot --glob '!ai_core.py'` names only
`pairs.py` (the validator) and `train.py`'s docstring; a test trains a manifest
with `style_system("Someone Else")` and asserts `rewrite_text` sent that prompt
and `manifest_mismatches` is empty for it; the corpus sha is unchanged (the
default author renders the same bytes).

## 3. R3 — draws are the interface everywhere (stylebot, then blog)

Today only `train`, `status` and `check-splits` take `--draws`; `build-view` is
blog-only; `check-splits --draws` builds under the identity policy and leaves a
stray `views/` dir (four exist now); `status.py` prints `(unbuilt)` from a
literal; the blog re-pins `--draws` by reaching into `__click_params__`
(`styler.py`, `bin/train_styler.py`) because `draws.view_options()` takes no
`defaults=`/`exclude=`.

1. `draws.ViewOpts` as an `opt()` bundle (`draws_dir`, `policy_path`, `views_root`)
   with `view_options(defaults=, exclude=)`; `ai-style build-view` in stylebot,
   `dan-style build-view` a shell over it. A policy on the command line is a JSON
   file: `ViewPolicy.dump()`/`load()` (a dataclass round-trip, no new format);
   `dan-style policy --dump` writes `view_policy()` so any `ai-style` command can
   run under the blog's policy.
2. `eval`, `make-splits`, `validate-pairs`, `train-clf` gain `--draws` +
   `--policy` and resolve to a view through `ensure_view`; `--pairs` keeps its
   help text as "a scratch file, not the corpus". `check-splits --draws` takes
   `--policy` the same way and never builds under identity.
3. `ensure_view` deletes views under `views_root` whose name is not the current
   `(policy_sha, draws_sha)` pair — they are disposable by D84; `status` reads
   the disk and prints `built <path>` or `unbuilt`.
4. `dan-style check-splits` mirror (defaults: splits, blog root, draws, policy).
5. Delete the old-layout fallback: `session.py`'s `corpus_path`/`read_pairs`
   branch and its "until W4 step 5" notes, `capture.py:376`'s comment,
   `corpus.shard_path_for` (zero callers in either repo), `synth.py:849`'s
   "month shard". Measure before deleting: `rg` each symbol across both repos in
   the commit body.
6. `test_view_policy.py`: the three literal pins (`VIEW_POLICY_SHA`,
   `VIEW_CONTENT_SHA`, `VIEW_RECORDS`) are replaced by a fixture draws dir with
   known contents, so a `log save` does not break the blog suite; the live
   corpus's sha is `dan-style status`'s to print (D52).

Done: `uv run ai-style build-view --help` exists; each of the four commands
accepts `--draws`; `ai-style check-splits --draws … --policy <dump>` prints the
same styler count as `dan-style status`; after `dan-style status`,
`ls _training_pairs/views` shows one directory; `rg -n "__click_params__"
src/livingthing` empty; `rg -n "shard_path_for|old layout|until W4" src/stylebot`
empty; both suites green; the corpus sha unchanged.

## 4. R4 — correctness leftovers (stylebot, one agent, one commit each)

Each is small and in a different file; none needs a D-entry.

- **C5.** At `best_of == 1` the D20 do-no-harm guard still lets the detector
  choose between input and sample, and `reranker_identity` returns None. Return
  the identity whenever a scorer influenced the choice; `verdict.compare_scores`
  then refuses correctly. Test: `best_of=1` with a scorer stamps `meta.reranker`.
- **C12.** `require_pinned_posts` (called from `train.py` and `classify_train.py`
  before a paid run) checks eval posts only; route it through
  `missing_sources` over corpus sources as `check_splits` already does; the
  `ai_style.py` message says "missing source(s)", not "pinned eval post(s)".
- **C19.** `markdown._find_div_blocks` uses `splitlines(keepends=True)` while
  `line_kinds` uses `split("\n")`; a lone `\r` counts lines differently. One
  splitter; a test with a lone `\r`.
- **C20.** `scoring._score_record`'s `judge` parameter is typed `Judge | None`
  against a `Judge | AsyncJudge | None` caller; `_score_slice` writes the row
  outside its per-record `try`. Fix both; a test that a judge exception yields
  an error row and no partial line.
- **S5a.** `draws.policy_sha` raises `TypeError` when `retired_arms`,
  `excluded_draw_ids` or `superseded_captures` is a literal `None`
  (`sorted(d[key])`); treat `None` as empty; test.
- **C14.** `lost_draws` rows write `attempt=st.gen_retries` on the error path
  where gate rows write the draw index; one meaning, documented in the field's
  docstring line.
- **A3.** `backends.rewrite_document`'s "not both" refusal can name a FILE that
  was not passed; name what was passed.
- **C1.** `export.py` raises plain `ImportError` with `_EXTRA_HINT`; raise the
  D47 `MissingExtra` class so `translate_errors` renders it.
- **C7.** The D34 gate fires twice on a real `dan-style train-clf` run
  (`classify_cli.train_clf` and `classify_train.train`); fire once, in the
  library.
- **C13.** Three synthetic draws carry a generator-un-escaped `\@handle`
  (D32's reverse-edit artefact); a synth-side gate rejects a draw whose target
  gained a backslash-escaped anchor, with a test; the three existing draws are
  excluded by a policy clause (D84), not edited.
- **C4.** `ai-style eval --by` and the blog's summary print judge means without
  `n_unscored`; print it beside every judge mean. A ≤ 15-line D-entry records
  the refusal contract and `SCHEMA_VERSION` 3 → 4, which C4 landed without one.

Done: each bullet's test fails on the pre-fix code (paste the failing run in
the commit body); suite green; no allowlist number rises.

## 5. R5 — the blog's leftovers (blog, one agent; the fifth copy is cross-repo)

- **`bin/train_styler.py` `train_cmd`** is 100 lines and the fifth copy of
  `ai_style.py`'s `train_cmd` echo block (round-1 report 1). Same shape as D78:
  a `TrainOpts` bundle and `train.run_training_pass(opts, *, selector, per_passage,
  pair_weight, emit)` in stylebot; the blog command ≤ 20 lines. Cross-repo:
  stylebot first, back-to-back.
- **D79 is not eager.** `--blog-root` on `train-clf` and `extend-eval` are
  stylebot `opt()` fields without `is_eager=True` (`classify_cli.py`,
  `splits.py`); correct today only by click's processing order. Add it; a test
  that an invoked `--blog-root` after a defaulted `--pairs` still moves the
  corpus default.
- **B4's literal shadows**, one spelling each: serve's `--backend` default
  (`"tinker"` in the blog, `None` in stylebot — pick stylebot's and pass the blog
  policy through `defaults=`); `--blog-root` is `"."` on synth/train and
  `Path(".")` elsewhere; status's `--blog-root` adds `exists=True` where
  stylebot's does not. The parity test then has three fewer `REPINNED` rows.
- **`voice_classifier.py`** re-exports twelve `classify_train` names nothing
  imports (`rg` each across the blog); delete the block.
- **Fragile tests.** `test_the_promoted_manifest_arms_a_styler_at_its_own_diet`
  reads the live `runs/` and hardcodes 4500; give it a fixture manifest. One
  blog test failed once in a full run and never again (round-1 B1/B2): run the
  suite three times in a row and report the name or that it did not recur.
- **`test_classify_train.py`** (stylebot) still `importorskip`s numpy and
  scikit-learn, which the dev group installs; drop the skips.

Done: `uv run python -c` shows `train_styler.train_cmd`'s body ≤ 20 lines;
`rg -n "is_eager" src/stylebot/classify_cli.py src/stylebot/splits.py` hits;
`REPINNED` in `test_dan_style_cli.py` has three fewer rows; suite green three
runs in a row, 0 skipped.

## 6. R6 — doc drift (both repos, docs only, one agent)

stylebot: `infer.py:319` `content_anchors` docstring copies "6.4% → 19.1%" (P1;
cite findings); `test_infer_segment.py`'s packing docstring carries measured
numbers (P4; cite); `findings.md` "The view drops the superseded captures" says
`policy_sha` moved twice — replace the count with "see `dan-style status`";
`findings.md:381`'s fixture path is now `fixtures/hard/`; OVERVIEW still calls
`is_human_authored` "the one blog-specific seam" (D85 made it a parameter) and
its masking bullet is correct now, leave it. Blog: RUNBOOK §4 "adding `--judge`
… is a no-op" is false since round-1 item 5 (rows are signal-keyed; say
"appends judge rows and touches no detector row"); `pyproject.toml:105-107`
still describes `[classifier]` as optional; `bin/dan_style.py`'s docstring, if
it still says stylebot is optional; `_training_pairs/README.md`'s older "can be
fixed by hand" sentence and CLAUDE.md's hook description predate the
append-only check (round-1 C21). Delete the stale gitignored
`_training_pairs/scores.jsonl` (2026-08-12, superseded by `runs/<id>/`): Dan's
file, so the agent reports it and does not delete it.

Done: `rg -n "6\.4%|no-op" src/stylebot/infer.py ../livingthing/_training_pairs/RUNBOOK.md`
empty; `rg -n "optional" ../livingthing/pyproject.toml` names no stylebot line;
budget tests green in both repos.

## 6b. R7 — the prose cull (stylebot, docs-in-code only, merges last)

P1 says a docstring cites a decision and may not restate it; P2 caps a
docstring at 15 lines. Measured 2026-09-05: 172 prose lines read as history or
narrative (dates, "used to", "until W4", wave and slice labels), 136 docstrings
over 8 lines hold 1,907 of the 3,488 docstring lines, and module docstrings
average 14 lines with seven at 27 or more (`train`, `splits`, `classify_train`,
`status`, `serve`, `markdown`, `cli_options`).

Rules for the cull, in order of yield: a sentence that tells how a thing came
to be is replaced by `(Dn)` or deleted; a docstring that restates its
function's signature or its caller list is cut to the contract (what it takes,
what it returns, what it raises, one invariant); a module docstring is the
module's contract and its D-citations, not its history; a comment that names a
wave, slice, F-number or date is deleted or becomes a D-citation; a `#:`
attribute comment stays. Nothing moves *into* `decisions.md` unless the "why"
lives nowhere else, and then as a ≤ 5-line amendment to the nearest D-entry.
Target: prose 4,516 → ≤ 3,500, i.e. `PROSE_CEILING` lowered by ≥ 1,000 in the
same commit. One agent, one commit per module group (`synth*`, `eval`/`scoring`
/`verdict`, `train`/`classify*`, `infer`/`serve`/`backends`, the rest), each
green, each lowering the ceiling.

Done: `PROSE_CEILING` ≤ 3,500 with the suite green; every remaining docstring
≤ 15 lines and module docstring ≤ 30 with the allowlist's docstring rows gone
or shrunk; a `rg` for `\b(W\d|F\d{1,2}|S\d[a-c]?|step \d|20\d\d-\d\d)\b` over
`src/stylebot` docstrings and comments returns only D-entry citations and
dated `#:` constants.

## 7. Acceptance (the reviewer's checklist)

Round 1 §7 in full (the merge gate: both suites green in the worktree, the
diff names only the slice's files, no allowlist number rises, `git status`
free of `.env`/`pairs.jsonl`, rebase then `--ff-only`), plus: `uv run ruff
format --check src tests` clean in stylebot; after R0, `PROSE_CEILING` never
rises and is lowered by any slice that lowers prose; the corpus `content_sha256` equals the master's wave-0 message
throughout (no slice here may move it); `ls _training_pairs/views` is one
directory after `dan-style status`; every R4 bullet's commit body carries its
failing pre-fix test run; no `__click_params__` in the blog; both suites green
and 0 skipped.

## 8. Out-of-scope reports

Appended by the master, dated, one line each.

- 2026-09-05 (R1): the blog's `ruff check` finds 43 errors (27 UP006, 6 B023,
  5 I001, 3 UP035, 2 UP032; 34 auto-fixable) and `ruff format` would touch 207
  of 462 files; the blog is lint-only by its own CLAUDE.md and warn-only at
  commit, so this is a blog slice for another round, not R1's.
- 2026-09-05 (R6): the blog's `.githooks/pre-commit` excludes `tests/fixtures/*.qmd`
  and says `segmenter_conformance.qmd` is sha-pinned against stylebot; the blog
  has no `tests/fixtures/` (the fixture is stylebot's `tests/fixtures/hard/`) — a
  dead pathspec and a stale comment in a hook.
- 2026-09-05 (R6): stylebot CLAUDE.md "Setup & verify" says `uv sync`, but a fresh
  env fails collection at `tests/test_export.py` without `--all-extras`
  (`safetensors`, X4's module-scope import); one line, 158/160.
- 2026-09-05 (R6): `tests/test_synth.py:482` docstring copies "876 pairs, 19.1%" —
  the same P1 drift as §6's two named sites.
- 2026-09-05 (R6): `bin/dan_style.py`'s docstring says `build-view` is blog-only
  "until ai-style grows one"; stale once R3 lands.
- 2026-09-05 (R6): `_training_pairs/scores.jsonl` (270 rows, 2026-08-12,
  gitignored) is still in the main blog checkout — Dan's to delete.
