<!-- Status: specified 2026-09-04, not started. Wave 3 of ship-shape-2026-09.md; the D-entries in §9 are written before any code (P8). Retires under P7 with the runbook. -->

# Seam consolidation brief — `dan-style X` = `ai-style X` + policy, for real

Survey 2026-09-04 of `~/Source/stylebot` (HEAD 880a579) and `~/Source/livingthing`. stylebot is an
**editable path dependency** of the blog (`livingthing/pyproject.toml`: `stylebot = { path =
"../stylebot", editable = true }`), so every stylebot commit is live in the blog suite at once:
stylebot steps must be *additive* until the blog step that stops importing the old thing has landed.
Rules: stylebot `CLAUDE.md` P1–P8; `stylebot/tests/test_budget.py` (module ≤ 800, function ≤ 80,
docstring ≤ 15, `DECISIONS_LINES` 1300, `DECISION_ENTRY_LINES` 31); blog `tests/test_budget.py`
(`BIN_MODULE_LINES` 250; bin modules hold only `*_cmd`/`@click.command` defs and `_opt*` helpers).
Mechanism to extend: `stylebot/src/stylebot/cli_options.py:opt|bundle_options|bundles_from_kwargs|
Emitter|silent|click_emitter|translate_errors`, as done by `backends.py:BackendOpts|backend_options`,
`splits.py:ExtendEvalOpts|apply_eval_extension`, `verdict.py:VerdictOpts|DocumentOpts|verdict_options`.
Corrections to the task text: the walker divergence pin is the **blog's** `tests/test_qmd_walker.py`
(stylebot has none); the blog's `lib.is_valid_qmd_file` has three other callers (`workflow_digest.py`,
`bin/massage_header_img.py`, `bin/zotero_export_bibs.py`), so it is re-composed, not deleted (§6).

## 1. The four command bodies become library passes

Every pass: `(bundle(s), *, policy hooks as keywords, emit: Emitter = silent)`; raises only
`stylebot.errors`; returns data. "Takes X from A to B" names statements in
`stylebot/src/stylebot/bin/ai_style.py`.

**eval** (the split the blog already made in `livingthing/voice_classifier.py:document_pass|
eval_pass|_emit_summary|_emit_verdict_and_report`):
- `stylebot/docscale.py:DocumentPass` (moved verbatim from `voice_classifier.py:DocumentPass`) and
  `docscale.document_pass(pairs_path, out, doc_opts: DocumentOpts, score_fields, *, manifest: dict,
  splits: dict | None, div_is_prose=None, emit=silent) -> DocumentPass`: `_document_pass` from
  `styler = backends.build_styler(...)` to `return` — role check → `ConfigError(param_hint=
  "--documents-role")`, `docs_out` default, progress at `status`, "rewrote N document(s)",
  `errors[:10]` at `warn`, fields ∪ `output`, `chunk_scale` incl. `div_is_prose`. ≈ 45 lines. Caller
  resolves manifest/splits: `ai-style` keeps the `--documents-manifest` UsageError and
  `sp.load_splits(config.splits_path(pairs_path.parent))`; blog passes `styler.resolve_manifest`,
  `tt.splits_path()`, `tt.div_is_prose`.
- `stylebot/scoring.py:run_eval(opts: EvalOpts, verdict: VerdictOpts, *, detector=null_detector,
  documents: DocumentPass | None = None, emit=silent) -> dict`: `eval_cmd` from `out = out_path or`
  to `wrote summary` — out/fields defaults, judge, progress, `chunked_detector(**documents.
  chunk_scale)`, `score_pairs_file`, prune notice (`status`), "scored N", errors `[:10]` at `error`,
  `summarize_scores`/`format_summary`, `--by` lines (adopt the blog's `det:/judge:` form), summary
  JSON; then `report_eval`. ≈ 60 lines. Drifts resolved: `[:5]`→`[:10]`; `--by` shows both means.
- `stylebot/verdict.py:report_eval(out, pairs_path, fields, verdict, opts: EvalOpts, *, emit=silent)`:
  `eval_cmd` from `if verdict_opts.compare_path` to the end. ≈ 30 lines. All three go into
  `eval.py:_SPLIT_OUT`.
- Blog hooks kept: `detector=build_detector(model_dir)`, `div_is_prose`, promoted-manifest default,
  `tt.splits_path()`, the `--run-id` home (`ev.verdict_dir`) applied as `replace(opts, out_path=…,
  summary_path=…, report_path=…)`. `ai-style eval` body after: ≈ 25 lines.

**train-clf** — `classify_train.py` is allowlisted at 965 lines, so this is the one new module,
`stylebot/src/stylebot/classify_cli.py` (≈ 220 lines; D78 names it):
- `TrainClfOpts` (§2), `train_clf_options(**kw)`, and `train_clf(opts, *, extra_positives=None,
  pair_ok=None, git_repo=None, emit=silent) -> tuple[Path, dict] | None`: `train_clf_cmd` from
  `ds = ct.assemble_dataset` to `wrote artifact` — exclusivity check as `ConfigError`; assembly under a
  **counting gate** (`voice_classifier.py:_counting_gate` moves here; emits `"<pair_ok.__name__>:
  dropped N pair(s) from M post(s)"`); the count line via `ct.LABEL_SLOP`/`ct.LABEL_DAN` (fixes the
  blog's `== 1`/`== 0`) plus `sum(ds.is_free)`; `n_pairs == 0` → `ConfigError(param_hint="--pairs")`;
  the **dry-run block verbatim** (`sp.require_pinned_posts`, `sp.summarize_roles` table,
  `sp.check_splits` warnings at `warn` — the lines `dan-style` lost); holdout list; "embedding with";
  `ct.train(..., git_repo=git_repo, blog_root=opts.blog_root, pair_ok=pair_ok)` — **both** passed,
  restoring the D34 gate the blog skips. ≈ 78 lines → split at the embed: `train_clf` (assembly +
  dry-run, ≈ 45; returns `None` on dry-run) and `_fit_and_report` (train, metrics, `meta.json`
  head_C/warnings, "wrote artifact", ≈ 35).
- `format_clf_metrics(metrics, indent="") -> list[str]`: `_echo_clf_metrics` as a renderer (it is
  the blog's `_metric_lines`; delete both).
- Blog hooks kept: `extra_positives=blog_free_positives(...)` under `--free-positives`,
  `pair_ok=tt.make_detector_pair_ok(...)`, `git_repo=blog_root`; the blog's "using shared splits
  contract"/"holdout split" notes move into `train_clf` (generic once `splits_path` is a field).

**run** — into `stylebot/backends.py` (333 → ≈ 470 lines; its docstring already claims the surface
`run` and `serve` share):
- `RunOpts` (§2), `run_options(**kw)`, `load_manifest(path) -> dict` (from `ai_style._load_manifest`;
  bad JSON → `ConfigError(param_hint="--manifest")`), `MANIFEST_HELP` moves here.
- `rewrite_corpus(opts: RunOpts, styler, *, scorer=None, selector=None, emit=silent) -> int`: the
  `if pairs_path is not None:` branch of `run_cmd`, plus the three flag-combination checks both CLIs
  retype (`--pairs` with FILE / without `--out`; `--out`/`--limit` without `--pairs`) as `ConfigError`.
  ≈ 25 lines. Blog keeps `styler.role_selector(role)` (≈ 10 lines: the closure inside today's
  `styler.rewrite_corpus`) and its "styling the … stratum" banner.
- `rewrite_document(text, opts: RunOpts, backend: BackendOpts, styler, *, scorer=None,
  on_decision=None, div_is_prose=None, rewrite=None, source: Path | None = None, emit=silent) ->
  RewriteResult`: `run_cmd` from `result = infer.rewrite_text(` through the in-place write —
  `rewrite or infer.rewrite_text` (the blog's `workflow_style.rewrite_masked` hook), decisions at
  `status`, `write_anchor_audit`, `format_rewrite_summary`, `.bak` + write when `opts.write_in_place
  and source`. stdout printing stays in the CLI. ≈ 40 lines.
- `format_rewrite_summary(result) -> str`: the `[N chunk(s), …]` line with the `(N lost, M invented)`
  breakdown `styler.summary_line` dropped; both CLIs and `dan-style style` use it.

**serve** — into `stylebot/serve.py` (241 → ≈ 310):
- `preference_recorder(path, *, run_id=None, default_chooser=None) -> Callable[[dict], None]`:
  `serve_cmd`'s `prefer_op` generalised to the blog's record (`ts`, `run_id`, `chooser` popped with a
  default, `**decision`); parent dir created. ≈ 12 lines.
- `serve_sidecar(backend: BackendOpts, *, detector, meta: dict, make_styler: Callable[[], Styler] |
  None, prefer=None, div_is_prose=None, stdin=sys.stdin, stdout=sys.stdout)`: `serve_cmd` from
  `chunk_chars = opts.chunk_chars()` to `serve_loop(...)` — the `@cache`d lazy styler with its
  `ConfigError` wrap (`make_styler`: `ai-style` → `partial(backends.build_styler, opts)`, blog →
  `lambda: arm_styler(...)[0]`), `detector_scorer`, `rewrite_op`, `serve_loop(div_is_prose=…)`.
  ≈ 40 lines.
- Blog hooks kept: `build_detector(DETECTOR_DIR)`, `meta={"name": "voice-clf", "styler_run": …}`,
  `arm_styler` (probe, staleness warning, banner), `default_preferences_path()`, `chooser="human"`,
  `div_is_prose` to the loop only (keep `test_the_rewrite_op_cannot_observe_div_is_prose`).
  `styler.serve_blog` shrinks to ≈ 22 lines.

## 2. Option bundles (`opt()` fields; every default a named constant)

`verdict.py:EvalOpts` (`cli_section = "Scoring"`); `EVAL_PARAMS = {"eval": EvalOpts, "verdict":
VerdictOpts, "documents": DocumentOpts}` (`VERDICT_PARAMS` kept as alias); `eval_options()` decorates
all three. Fields: `pairs_path` (`None`, `--pairs`, `required=True`, `Path(exists=True)`); `fields`
(`()`, `--field`, `multiple`, `Choice(["slop","target","output"])` — ai-style's list lacked `output`
though `run --pairs` documents `eval --field output`; the blog's is right); `judge` (`False`);
`judge_model` (new `eval.DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4-8"`, also the default of
`openrouter_judge|openrouter_judge_async`); `detector_model` (`None`, `Path(exists=True,
file_okay=False)`); `vale_config` (`None`); `max_workers` (new `scoring.DEFAULT_MAX_WORKERS = 8`,
also `score_pairs_file`'s default); `out_path` (`None`, `--out`); `summary_path`; `by`; `limit`;
`prune_orphans` (`True`); `report_path`; `report_max_rows` (new `report.DEFAULT_MAX_ROWS = 2000`,
also `render_scores_report`'s); `facet_by` (new `report.DEFAULT_FACET_BY = "slop_strategy"`);
`sample_n` (`--sample`). Help strings move verbatim from `eval_cmd`'s decorators.

`classify_cli.py:TrainClfOpts` (`"Voice classifier"`): `pairs_path` (`--pairs`, required,
`exists=True`); `out_dir` (`--out`, required); `splits_path` (`None`, `Path(exists=True,
dir_okay=False)`); `blog_root` (`None`); `test_size` (`ct.DEFAULT_TEST_SIZE`); `n_splits`
(`ct.DEFAULT_N_SPLITS`); `C` (`None`); `embed_model` (`ct.DEFAULT_EMBED_MODEL`); `holdout_frac`
(new `ct.DEFAULT_HOLDOUT_FRAC = 0.0`); `holdout_seed` (new `ct.DEFAULT_HOLDOUT_SEED = 0`) — both also
`ct.train`'s defaults (replace literals in place; the module may not grow); `holdout_posts`;
`dry_run` (flag).

`backends.py:RunOpts` (`"Rewrite"`): `best_of` (new `backends.DEFAULT_BEST_OF = 1`); `strict_anchors`
(flag); `anchor_log`; `write_in_place` (`--write`); `pairs_path` (`--pairs`, `exists=True`);
`out_path` (`--out`); `limit`; `manifest_path` (`--manifest`, `help=MANIFEST_HELP`). `RUN_PARAMS =
{"run": RunOpts, "opts": BackendOpts}`. No `ServeOpts`: `serve` keeps `--manifest`/`--preferences`
as its own two options (flag-level parity covers them). The blog pins policy with `defaults=`/
`exclude=` only, never by retyping.

## 3. What the blog deletes (line counts by `ast`, 2026-09-04)

- `src/livingthing/voice_classifier.py` (440): delete `DocumentPass` 7, `document_pass` 55,
  `_metric_lines` 18, `_counting_gate` 10, `_emit_summary` 18, `_emit_verdict_and_report` 35;
  `_assemble` 37 → `detector_inputs(...) -> (extra_positives, pair_ok)` ≈ 12; `train_pass` 65 → ≈ 20;
  `eval_pass` 70 → ≈ 25. Expected ≈ 210.
- `src/livingthing/styler.py` (511): delete `summary_line` 19, `rewrite_corpus` 39 (→ `role_selector`
  ≈ 10); `serve_blog` 67 → ≈ 22. `chunk_diet` 12 stays (tests pin it; thin over `serving_diet` —
  report, out of scope). Expected ≈ 440.
- `src/livingthing/bin/train_voice_clf.py` (191): `train_cmd` 15 option lines + 9 → `@train_clf_options
  (defaults=[TrainClfOpts(pairs_path=…, out_dir=DEFAULT_OUT, splits_path=…, blog_root=Path("."),
  embed_model=EMBED_MODEL)])` + 4 blog flags + ≈ 10-line body; `eval_cmd` 24 + 10 → `@eval_options
  (defaults=[EvalOpts(pairs_path=…, detector_model=DETECTOR_DIR)])` + `--run-id`, `--blog-root` +
  ≈ 10; `--model` is renamed `--detector-model` (RUNBOOK lines 176, 248–249 use neither). ≈ 120.
- `src/livingthing/bin/style_file.py` (248): `run_cmd` 16 + 55 → `@run_options(exclude=
  ["manifest_path"])` (blog's `--manifest` help differs) + `--role`, `--mask` + ≈ 18-line body;
  `serve_cmd` 19 → ≈ 12; `style_cmd` (88, allowlisted) only swaps `summary_line` for
  `backends.format_rewrite_summary`. ≈ 190; its `functions: (88,)` row unchanged.
- `src/livingthing/bin/train_styler.py` (182): **not touched**; `train_cmd` (80) is a fifth copy of
  `ai_style.py:train_cmd`'s echo block — the successor slice, reported not folded in.

## 4. Parity test at default/type granularity

`livingthing/tests/test_dan_style_cli.py:_flags` → `_params(command) -> dict[str, click.Option]`.
For every shared command and flag not in `EXCLUDED[name]`: `param.multiple` equal;
`param.type.to_info_dict()` equal unless in new `REDECLARED[name]` (run/serve `--backend`,
`--base-url`, `--model`: the blog adds `local` + `DAN_STYLE_*` envvars, `styler.BACKEND_EXCLUDE`);
`param.default` equal unless in new `REPINNED[name]` (flag → the policy constant), and a REPINNED
default must really differ (a stale row fails, like today's `stale` check). Expected `REPINNED`: eval
`--pairs`, `--detector-model`; train-clf `--pairs`, `--out`, `--splits`, `--blog-root`; status
`--blog-root`, `--detector-model`; extend-eval `--blog-root`, `--pairs`, `--splits`; train
`--blog-root`, `--splits`, `--renderer`; synth its 13 `chunk_policy`/`GenOpts`/`SessionOpts`/
`CorpusOpts` pins, each naming its `training_targets` constant. **Stops being excluded**:
`EXCLUDED["eval"]["--detector-model"]` (bundle flag, default `DETECTOR_DIR`); `EXCLUDED["eval"]`
becomes `{}`. Stays excluded (D35): run/serve `--detector-model`, `--sampler-path`, `--base-model`,
`--renderer`; status's three. Literal shadows this catches today: train-clf `0.25`/`8`, eval
`--judge-model`, `--max-workers 8`, `--report-max-rows 2000`, `--facet-by`, run `--best-of 1`.

## 5. Data-dir precedence: one callable that sees `--data-dir` and `--blog-root`

`train_voice_clf.py:DEFAULT_PAIRS = tt.pairs_path` is called by click with no arguments, so
`train-clf --blog-root /x` reads `./_training_pairs` while `status_cmd` calls `tt.pairs_path(data_dir,
blog_root)`. Fix in `livingthing/training_targets.py`: `context_default(resolve)` returns a zero-arg
callable that reads `click.get_current_context().params` for `data_dir`/`blog_root` and returns
`resolve(flag, blog_root)`; every blog `--blog-root`/`--data-dir` becomes `is_eager=True` so click has
processed them before any default is computed (defaults run inside `make_context`, context current).
This also works for bundle-pinned defaults (`ExtendEvalOpts(pairs_path=context_default(tt.pairs_path))`),
which a per-option `callback=` cannot — `bundle_options` takes none, and adding one is new mechanism.
train-clf's splits default: `context_default(lambda f, r: p if (p := tt.splits_path(f, r)).exists()
else None)` — today's "used when it exists"; an explicit missing `--splits` stays loud via
`exists=True`. Commands that change: `train-clf` (`--pairs`, `--splits`), `eval` (`--pairs`; gains
`--blog-root` for the `--run-id` home and `document_pass`'s splits), `extend-eval` (three pins),
`train` (`--pairs`, `--splits`, manifest home). `status`/`synth` already resolve in the body;
`run`/`serve` have no `--blog-root` (report, do not change). Test `test_blog_root_moves_every_
corpus_default`: `tmp/_training_pairs/pairs.jsonl` empty; `train-clf --blog-root tmp --dry-run` and
`status --blog-root tmp` both name `tmp/_training_pairs`, neither names the cwd's; same for
`extend-eval --blog-root tmp --post x`.

## 6. Walker merge

`stylebot/targets.py:iter_targets`: drop the files/blog_root refusal; rule: "`blog_root` supplies the
label root **and** the selector; `files` replaces only the walk" — with both, each file is
`_load_meta_body`'d, kept iff `selector(meta)`, labelled by `_source_label(path, root)`; `files` alone
stays pre-selected and unlabelled; neither stays an error. `ai-style synth` keeps its "one of
FILES/--blog-root" UsageError. Keep `tests/test_synth_cli.py`'s "not both" pin (it is `policy`+kwargs);
add a files+blog_root test (relative `Target.source`, selector applied). Blog: `bin/train_targets.py:
main` → `synth.iter_targets(files=gather_qmd_files([], base=blog_root), blog_root=blog_root,
selector=make_selector(min_quality), policy=chunk)`. `livingthing/lib.py`: delete `QUARTO_SKIP_DIRS`;
`is_valid_qmd_file(path, base=None)` = `stylebot.lib.is_valid_qmd_file(path, base) and not
LOCAL_SKIP_DIRS & set(parts[:-1]) and path.suffix == ".qmd"` (name kept for its callers; docstring
≤ 15). `tests/test_qmd_walker.py`: import `QUARTO_SKIP_DIRS` from `stylebot.lib`; delete
`test_the_synth_walker_sees_exactly_what_this_repo_calls_a_post`; add `test_synth_targets_come_from_
this_repos_walk` (tmp root with selectable `tests/x.qmd` and `notebook/y.qmd`: only the latter is a
target, labelled relatively). `stylebot/lib.py`, the comment above `QUARTO_SKIP_DIRS` ("the blog's
live in its `_quarto.yml`, and `livingthing.lib` reads them there") is false — the blog hand-writes
`LOCAL_SKIP_DIRS` and a test pins it to `_quarto.yml`; reword to "a project's own exclusions are the
caller's (the blog composes `LOCAL_SKIP_DIRS` over this predicate)". D46 collapses to three lines.

## 7. Steps, commits (≤ 72-char subjects), parallelism — each step green in BOTH suites

Lane A, sequential (stylebot; all touch `bin/ai_style.py` + `tests/test_budget.py`):
- A0 `plans: archive D36-D48; write D78-D80 before the seam code` — `_plans/decisions.md`,
  `_plans/decisions-archive.md` (decisions.md is 1266/1300; D36–D48 total 378 lines).
- A1 `eval: EvalOpts, document_pass and run_eval are library passes (D78)` — `verdict.py`,
  `scoring.py`, `docscale.py`, `eval.py`, `report.py`, `bin/ai_style.py`, budget (lower the
  `bin/ai_style.py` row), tests.
- A2 `train-clf: classify_cli holds TrainClfOpts and the training pass (D78)` — new `classify_cli.py`,
  `classify_train.py` (two constants), `bin/ai_style.py`, budget, tests.
- A3 `run/serve: RunOpts, rewrite passes and serve_sidecar are libraries (D78)` — `backends.py`,
  `serve.py`, `bin/ai_style.py`, budget, tests.
Lane B, sequential (blog; all touch `tests/test_dan_style_cli.py` + `tests/test_budget.py`); Bn after An:
- B1 `eval: dan-style eval renders EvalOpts and calls scoring.run_eval (D78)` — `voice_classifier.py`,
  `bin/train_voice_clf.py`, tests (`test_document_eval_rewrites_and_scores_callout_prose` imports
  `stylebot.docscale.document_pass` and passes `manifest=`/`splits=`/`div_is_prose=`).
- B2 `train-clf: dan-style train-clf is TrainClfOpts + blog gates (D78)` — same two modules.
- B3 `run/serve: dan-style run and serve call the stylebot passes (D78)` — `styler.py`,
  `bin/style_file.py`, tests (serve tests keep mocking `styler.arm_styler`/`voice_classifier.
  build_detector`; both survive).
- B4 `test: the mirror test compares defaults and types, not names (D78)` — `test_dan_style_cli.py`.
- B5 `training_targets: corpus defaults read --blog-root off the context (D79)` — `training_targets.py`,
  `bin/train_voice_clf.py`, `bin/train_styler.py` (default swap only), test.
Lane C, own worktrees, no file overlap with A/B:
- C1 (stylebot, after A0) `targets: files= with blog_root= walks the caller's list (D80)` —
  `targets.py`, `lib.py` (comment), `tests/test_synth_cli.py`.
- C2 (blog, after C1) `synth: the blog hands iter_targets its own walk (D80)` — `lib.py`,
  `bin/train_targets.py`, `tests/test_qmd_walker.py`; budget only if `lib.py`'s docstring row drops.
Nothing in stylebot is deleted after the fact: A1–A3 remove `_echo_*`/`_load_manifest` with their
bodies, and the blog never imported them.

## 8. Done-criteria

- `cd livingthing && uv run dan-style train-clf --dry-run` prints the `N content-matched pair(s)` line,
  the `eval:`/`styler:`/`detector:` rows, any `check_splits` `WARNING:`, then `[dry-run] no embedding,
  no artifact written`; `--blog-root /nonexistent` exits 2 naming it.
- `uv run dan-style train-clf --help | grep -E "test-size|n-splits"` shows `[default: 0.25]`/`[default:
  8]` from `ct.DEFAULT_*`; `rg -n "0\.25|== 1|== 0" src/livingthing/voice_classifier.py
  src/livingthing/bin/train_voice_clf.py` → empty.
- `rg -n "QUARTO_SKIP_DIRS" src/livingthing` → empty; `rg -n "not both" src/stylebot/targets.py` → the
  `policy`/kwargs line only.
- `rg -n "def (document_pass|_metric_lines|_counting_gate|_emit_summary|_emit_verdict_and_report|
  summary_line)\b" src/livingthing` → empty; `rg -n "def (_document_pass|_echo_clf_metrics|
  _echo_summary|_echo_verdict|_load_manifest)" src/stylebot/bin` → empty.
- `uv run pytest -q tests/test_dan_style_cli.py` green with `REPINNED`/`REDECLARED` populated and
  `EXCLUDED["eval"] == {}`; `uv run dan-style eval --help` lists `--detector-model`, no `--model`.
- Both repos: `git diff <base>..HEAD -- tests/test_budget.py` shows no number increased, no row added;
  stylebot's `bin/ai_style.py` row loses its `functions` key; the blog's `BIN_*_ALLOWLIST`s gain no
  row. `wc -l src/stylebot/classify_cli.py` ≤ 300; every other touched module stays under its row.
- Blog, `uv run python -c`: `inspect.getsource` body (decorators excluded) ≤ 20 lines for
  `train_voice_clf.train_cmd|eval_cmd`, `style_file.run_cmd|serve_cmd`.
- `git status` in both repos shows no `.env*`, `pairs.jsonl`, `pairs/`; nothing pushed.

## 9. D-entries to write first (P8), D26 shape, ≤ 30 lines, `stylebot/_plans/decisions.md`

- **D78: the four command bodies are stylebot passes over option bundles; `dan-style` calls them.**
  Decision: `scoring.run_eval`/`docscale.document_pass`/`verdict.report_eval`, `classify_cli.train_clf`,
  `backends.rewrite_corpus|rewrite_document`, `serve.serve_sidecar` take `EvalOpts`/`TrainClfOpts`/
  `RunOpts` + policy hooks + `emit`; both CLIs render flags from the bundles; the mirror test compares
  defaults and types. Alternatives: keep the blog's copies pinned by test (five drifts in one day:
  D34 gate, dry-run table, `[:5]`, anchor breakdown, `== 1`); grow `classify_train.py` (allowlisted
  965); one `passes.py` for all four (a grab-bag). Constraint: D35 — a library reachable only through
  a CLI gets copied. Clears: `bin/ai_style.py` functions (139, 105, 88); ≈ 310 blog lines; new
  `classify_cli.py` ≤ 300. Re-evaluate if a third front-end appears or a bundle needs per-field callbacks.
- **D79: blog corpus defaults are callables that read `--blog-root`/`--data-dir` off the click context.**
  Decision: `training_targets.context_default(resolve)`; the two flags are eager. Amends D45 (callables
  were necessary, not sufficient). Alternatives: resolve in each body (a line per path per command,
  impossible for bundle-pinned defaults); `callbacks=` on `bundle_options` (new mechanism). Constraint:
  `bundle_options` re-pins defaults only. Re-evaluate if click changes default-vs-eager ordering.
- **D80: `iter_targets` takes `files=` with `blog_root=` — the root labels and selects, the list
  replaces the walk.** Supersedes D46 (its re-evaluate clause named this). Alternatives: keep the
  divergence pin (an empty gap asserted forever); teach stylebot `LOCAL_SKIP_DIRS` (project policy in
  generic tooling, D45). Constraint: `Target.source` must stay blog-relative or `splits.role_of` answers
  `styler` for a frozen-eval post. Re-evaluate if a second project needs the walk parameterised.
