# Ship-shape, round 2 — closing what the swarm reported

**Status: specified 2026-09-05, not started.** The reviewer's QA of the round-1
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
at 41 flags; D83 records it (R1).

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

Critical path: R1 (≤ 1 h), then R2–R6 in parallel (≤ 3 h). Nothing waits on Dan.

## 1. R1 — rules and the reformat (stylebot, then blog docs; first, alone)

1. `uv run ruff format src tests` as one reformat-only commit ("style: adopt
   ruff format (owner decision 2026-09-05)"), then `ruff format --check` joins
   the merge gate in §5 and CLAUDE.md's "Ruff" line says lint *and* format,
   diverging from the blog's lint-only rule on purpose. `tests/test_budget.py`
   allowlist numbers may change from the reformat: record the new exact sizes
   in the same commit, no number may rise above budget.
2. CLAUDE.md P8 gains the cross-repo sentence from §0 (≤ 2 lines; the file is at
   160/160, so trim the "Measure before deleting" bullet's second clause to pay).
3. `decisions.md` D83: three-line amendment, "surface at 41 flags, accepted
   2026-09-05; library-only fields stay reachable through the bundles".
4. `ship-shape-2026-09.md` stub: the step-5 range ends at `ae5dfb6`, not
   `6fcee88`.

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

## 7. Acceptance (the reviewer's checklist)

Round 1 §7 in full, plus: `uv run ruff format --check src tests` clean in
stylebot; the corpus `content_sha256` equals the master's wave-0 message
throughout (no slice here may move it); `ls _training_pairs/views` is one
directory after `dan-style status`; every R4 bullet's commit body carries its
failing pre-fix test run; no `__click_params__` in the blog; both suites green
and 0 skipped.

## 8. Out-of-scope reports

Appended by the master, dated, one line each.
