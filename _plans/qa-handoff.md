# QA handoff — `stylebot` ↔ `livingthing` joint review

**Purpose.** A complete map of every code file implicated in the stylebot
fine-tuning project and its blog integration, for a QA agent assessing the two
repos **jointly for clarity and quality**. This is a *map*, not a diff: it covers
the whole surface, and flags which files the recent work-arc (Phase-2 curation →
slop-strategy/OpenRouter → batched eval → scores report) changed, so the reviewer
has full context rather than only the last commit.

Two repos, checked out side by side:
- `stylebot` (this repo, `~/Source/stylebot`) — the **generic mechanism**: a
  library + `ai-style` CLI to manufacture and score `(slop → Dan)` training pairs.
  Public.
- `livingthing` (`~/Source/livingthing`) — Dan's blog. Supplies the **policy** and
  the prose corpus; depends on stylebot via a local editable path dep. Private.

## The cross-repo contract (assess adherence to this first)

The architecture is a deliberate **mechanism (stylebot) vs policy (livingthing)**
split. Authoritative rules: `CLAUDE.md` and `_plans/OVERVIEW.md` ("Where the
blog/generic boundary actually falls"). Invariants to verify:

1. **One-way dependency: `livingthing → stylebot`.** stylebot must NEVER import
   `livingthing`. *Verified clean:* the only match in stylebot for "livingthing"
   is a provenance **comment** at `src/stylebot/lib.py:151` (not an import).
2. **No leakage.** Nothing blog-specific (hardcoded blog paths, the `quality` /
   `automation` frontmatter fields, the `🚧TODO🚧` marker, `## Incoming`) should
   appear in stylebot; nothing generic should be stranded in livingthing. The one
   sanctioned blog-ism in stylebot is the bundled selector example
   `stylebot.lib.is_human_authored` (the `automation: 0` convention), documented
   as an example, not a gate.
3. **Shared on-disk contracts** (the seams — `_plans/OVERVIEW.md` "shared data
   contract"): the `pairs.jsonl` schema (`stylebot.pairs.validate_pairs_file`),
   `build_pair_content` (heading context, byte-identical on both sides), and the
   `scores.jsonl` id-keyed schema. Both pair *producers* (Phase-1 logger, Phase-2
   synth) must emit the same shape.
4. **Frozen string:** `STYLE_SYSTEM` (`stylebot.ai_core`) is baked into every pair;
   changing it invalidates the corpus. It must be emitted verbatim by every producer.

## `stylebot` — full file map

`◆` = central to the recent work-arc · `·` = pre-existing/stable · `○` = empty stub.

### Source (`src/stylebot/`)
| File | L | | Role |
|---|--:|:--:|---|
| `ai_core.py` | 34 | · | `STYLE_SYSTEM` — the **frozen** system prompt baked into every pair. |
| `config.py` | 84 | · | Path/secret resolution: `resolve_data_dir` (flag > `$STYLEBOT_DATA_DIR` > default), `get_key`/`require_key` (`.env`). The one place keys/paths resolve. |
| `lib.py` | 405 | ◆ | Generic frontmatter+markdown utilities: `is_human_authored` (bundled selector example), `gather_qmd_files`, `read_w_frontmatter`, `split_paragraphs`, and `segment_for_edit`/`editable_prose` (stdlib fork of the blog's `qmd_core` — see provenance below). |
| `pairs.py` | 153 | ◆ | The corpus schema: `validate_pairs_file`/`validate_pair_record`, `build_pair_content` (heading-context contract + validator invariant), `iter_pairs` (shared JSONL reader). |
| `synth.py` | 908 | ◆ | Phase-2 engine. Target extraction/curation (prose-only, hygiene, merge, link-list guard, heading context), `STRATEGIES` + `resolve_strategy` (slop-prompt knob), `Generator` + `anthropic_/openai_/local_/openrouter_generator`, `synthesize_pairs` (idempotent/resumable), `synth_key`. |
| `eval.py` | 648 | ◆ | Eval harness. Per-text primitives (`vale_score`, `openrouter_judge`, `Detector`/`null_detector`, `score_candidate`) + batched layer (`score_pairs_file` → id-keyed `scores.jsonl`, concurrent+resumable; `summarize_scores(by=…)`; `load_scores`, `record_id`, `pair_body`, `FIELD_EXTRACTORS`). |
| `report.py` | 541 | ◆ | Read-only HTML visualisers (no deps beyond stdlib, XSS-safe): `render_targets_report` (synth targets) and `render_scores_report` (joins pairs+scores: slop↔Dan + judge score/rationale, sortable, per-strategy headline). Shared `_CSS`/`_histogram_svg`/escaping. |
| `bin/ai_style.py` | 397 | ◆ | The `ai-style` CLI group — thin wrappers: `synth` (Phase 2) and `eval`. `split`/`train` land with their phases. |
| `bin/ai_style_log.py` | 1087 | ◆ | Phase-1 `ai-style-log` — the daily manual pair-capture logger (shipped, daily use). Recently extended with heading context. **Note the duplication with the blog's copy — see provenance.** |
| `__init__.py`, `bin/__init__.py` | 0 | ○ | Package stubs. |

### Tests (`tests/`) — all run under `uv run pytest` (103 passing)
| File | L | Covers |
|---|--:|---|
| `test_synth.py` | 373 | Synth engine: rotation, idempotent resume, heading context, slop-strategy/synth_key, openrouter generator. |
| `test_eval.py` | 245 | Eval: `evaluate_groups`, judge-reply parsing, batched `score_pairs_file` (id precedence, context-strip, resumable, keyless, faceting). |
| `test_ai_style_log.py` | 240 | Phase-1 capture / chunk-diff / heading context. |
| `test_merge.py` | 160 | Section-aware paragraph merge + link-list guard. |
| `test_report.py` | 159 | Targets report + scores report (self-contained, XSS-escaped, join, headline, sort attrs, max_rows, keyless). |
| `test_pairs.py` | 112 | Schema validator + `build_pair_content` + context invariant. |
| `test_smoke.py` | 71 | Phase-0 smoke. |
| `test_segment.py` | 69 | `segment_for_edit`/`editable_prose`. |
| `test_selection.py` | 38 | `is_human_authored`. |

### Plans & narrative (the spec — read for intent)
`_plans/OVERVIEW.md` (phase map, contracts, boundary), `_plans/eval-harness.md`,
`_plans/heading-context.md`, `_plans/phase-1-pair-capture.md`,
`_plans/phase-2-synthetic-pairs.md`, `_plans/phase-3-training.md`,
`_plans/phase-4-inference-cli.md`; `docs/fine_tuning_danbot.qmd` (rationale);
`CLAUDE.md` (operating manual + architecture rules).

## `livingthing` — files implicated

### Integration code (NEW — written by this work; the blog half of the seam)
| File | L | Role — what to assess |
|---|--:|---|
| `src/livingthing/training_targets.py` | 78 | The blog **policy** module. `make_selector(min_quality)`/`select_training_post` (composite `automation:0 ∧ quality>N ∧ not draft ∧ not auxiliary`) + policy constants (`STUB_MARKER`, `STOP_HEADERS`, `MERGE`, `MERGE_MAX_CHARS`, `MIN_CHARS`, `DROP_LIST_ITEMS`, `HEADING_CONTEXT`, `CONTEXT_DROPOUT`, `SLOP_STRATEGY`, `OPENROUTER_MODELS`). Assess clean policy/mechanism separation + constant docs. |
| `src/livingthing/bin/train_targets.py` | 151 | The `train-targets` console script — a **thin CLI wrapper** over `stylebot.synth.iter_targets`/`synthesize_pairs` + `stylebot.report`. Assess that it stays thin (no business logic), flag/default clarity, generator construction. |
| `tests/test_training_targets.py` | 232 | Standalone test runner (**blog has no pytest**: zero-arg functions + a `__main__` runner, `uv run python tests/test_training_targets.py`). Assess coverage + the runner pattern. |

### Config (modified)
| File | Role |
|---|---|
| `pyproject.toml` | Adds the `stylebot` dep, the `[tool.uv.sources]` **editable path dep** (`../stylebot`, dev/local-only), and the `train-targets` script entry. |
| `uv.lock` | **Autogenerated** from the dep change — exclude from clarity review. |

### Read-only deps (used by the seam, NOT modified)
| File | L | Used for |
|---|--:|---|
| `src/livingthing/lib.py` | 258 | `is_auxiliary_post` (+ `AUXILIARY_TYPES`) — composed into the policy selector. |
| `src/livingthing/cli_utils.py` | 115 | `setup_cli_environment` — called by `train_targets`. |

### Provenance / parallel producers (pre-existing, NOT touched — context for the joint review)
| File | Note |
|---|---|
| `src/livingthing/qmd_core.py` (~370 L) | The **source** from which `stylebot.lib.segment_for_edit`/`editable_prose` was ported (stdlib-only fork; see `stylebot/src/stylebot/lib.py:151`). A reviewer may check the port stayed faithful and whether the duplication is intentional (it is: stylebot must not depend on the blog). |
| `src/livingthing/bin/ai_style_log.py` (~37 KB) | A **blog-local Phase-1 logger**, parallel to `stylebot/src/stylebot/bin/ai_style_log.py`. Pre-dates the stylebot port and is **out of scope** for this work, but the reviewer should be aware the two may have **diverged** — worth flagging if they assess the Phase-1 producer. The canonical/maintained one going forward is the stylebot copy. |

## What the recent work changed (the delta, for orientation)

A multi-step arc, all green (`uv run pytest -q` = 103 passing in stylebot; the blog
test runner passes 13/13):
1. **Phase-2 target curation** — prose-only extraction, hygiene, section-aware
   merge, link-list guard, heading context (shared `build_pair_content`).
2. **Slop-strategy knob + OpenRouter** — `STRATEGIES`/`--slop-strategy`/
   `--slop-system-file` (recorded as `meta.slop_strategy`, folded into `synth_key`);
   `openrouter_generator`/`--openrouter-model` (one `OPENROUTER_API_KEY`, many
   models). Blog `train-targets` defaults to an OpenRouter rotation.
3. **Eval harness, batched + JSONL-native** — `ai-style eval --pairs` scores the
   slop/Dan fields of every pair → id-keyed `scores.jsonl`; `summarize_scores(by=…)`.
4. **Scores visualiser** — `ai-style eval --report scores.html` /
   `render_scores_report`.

## Assessment focus & known constraints (don't file these as bugs)
- **Tests in the blog don't use pytest** — `test_training_targets.py` is a standalone
  `__main__` runner by design (the blog repo has no pytest).
- **`.env` / `.envrc` are gitignored**; `OPENROUTER_API_KEY` lives there. Live judge
  runs need `direnv exec uv run …` (the key is only in env that way). The harness runs
  **keyless** by default (Vale + null detector + null judge) — that's intended.
- **The editable path dep** (`livingthing` → `../stylebot`) is dev/local-only by design.
- **Deferred, not bugs:** the statistical-detector audition (eval's 4th signal),
  `meta.weight` derivation (Phase 3), and the Phase-4 styler are planned, not built —
  see the `_plans/`. The eval renderer is already generic over score *fields* for the
  Phase-4 `slop/output/target` view.
- **Out of QA scope:** any uncommitted `notebook/*.qmd` content edits in livingthing
  (Dan's own writing, unrelated to this integration).
