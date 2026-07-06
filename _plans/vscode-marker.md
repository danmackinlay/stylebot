# VS Code paragraph marker — 🔧 IN PROGRESS (design settled 2026-07-06)

Mark paragraphs **dan-or-bot** live in the editor, using the trained voice
classifier. A paragraph's `P(slop)` shows as a graded background tint +
scrollbar mark while Dan drafts; suspicious paragraphs get rewritten, and the
existing `ai-style-log` tasks capture the `(before, after)` pair. The marker is
the **front-end of the capture loop**, not a standalone gadget.

Lives here (stylebot): nothing in the sidecar or extension is Dan-specific —
segmentation mirrors `stylebot.lib`, the sidecar wraps
`stylebot.classify.sklearn_detector` (which lazy-imports sentence-transformers,
so stylebot's no-ML-deps import contract holds), and the extension is
parameterized by a command + artifact path. The blog contributes only the
trained artifact (`livingthing/_models/voice-clf/`) and workspace settings —
the usual mechanism/policy split.

## Feasibility (measured, StyleDistance on Dan's Mac, MPS)

The only worry was latency; it's a non-issue:

| | latency |
|---|---|
| cold model load (once, at sidecar startup) | **5.0 s** |
| warm: 1 paragraph | 7 ms |
| warm: whole 30-paragraph notebook | **34 ms** |
| head dot-product → `P(slop)` | 28 µs |

After a one-time 5 s load, re-scoring an entire document is ~34 ms — fast
enough to run on debounced idle keystrokes, not just on save.

## Settled design decisions (2026-07-06)

- **Scores over the wire, not vectors.** The sidecar wraps the finished
  detector (`classify.sklearn_detector`) and returns `P(slop)` — exact parity
  with `ai-style eval`, simplest protocol. Retraining the head = restart the
  sidecar (5 s); fine for a personal tool.
- **No heading prefix.** The classifier trained on *body* text with heading
  context stripped (`eval.py` strips via `pair_body` before scoring; dataset
  assembly did the same). Scoring bare paragraph text is therefore
  on-distribution — the earlier "prepend the nearest heading" idea was wrong,
  not deferred.
- **NDJSON over stdin/stdout, not LSP.** LSP has no decoration concept
  (ErrorLens, the closest prior art, is not an LSP server). One JSON object
  per line, both directions; ~30 lines a side, no framing deps.
- **Sidecar = `ai-style serve`,** a subcommand of the existing entry point
  (one entry point, subcommands — per OVERVIEW), not a loose script.
- **Extension owns the sidecar lifecycle.** Spawn on activation, `info`
  handshake covers the model load (status bar: "voice: loading…" → ready),
  restart with capped backoff on crash, kill on deactivate. If
  `voiceMarker.command` is unset in a workspace, the extension stays dormant —
  it only runs where the blog's workspace settings configure it. No manual
  launching.
- **UX: graded tint + overviewRuler + hover, warped buckets.** Decoration
  types are fixed-style, so grading = ~5 `TextEditorDecorationType` buckets.
  Bucket thresholds are **calibrated quantiles** of the detector-score
  distribution from the corpus `scores.jsonl` (Dan-side vs slop-side), not
  equal-width bins — raw logistic scores cluster, and 0.2/0.4/0.6/0.8 would
  waste most buckets.
- **`.qmd` language id** is `quarto` with the Quarto extension installed,
  `markdown` without → activate on `onLanguage:markdown` + `onLanguage:quarto`.

## Components

1. **Sidecar** — `stylebot.serve.serve_loop(detector, stdin, stdout)` + CLI
   `ai-style serve --detector-model DIR`. Requests
   `{"id", "op": "score", "texts": [...]}` → `{"id", "scores": [...]}`;
   `{"op": "info"}` → the artifact's `meta.json` (doubles as the ready
   handshake). Malformed input → an `error` response, never a crash. Run it
   from the blog env: `uv run --project ~/Source/livingthing ai-style serve
   --detector-model _models/voice-clf`.
2. **Extension** (`vscode-voice-marker/`, TS + esbuild): spawn/manage the
   sidecar (`PYTHONUNBUFFERED=1`), segment, debounce (~500 ms), score, bucket,
   decorate, hover. Settings: `voiceMarker.command` (string[]),
   `.cwd`, `.thresholds`, `.minChars`, `.debounceMs`, `.enabled`.
3. **Segmentation parity** (what keeps scores honest): blank-line paragraph
   splitting (`stylebot.lib.split_paragraphs`) after dropping protected blocks
   — code fences, `$$math$$`, `:::` divs, blockquotes
   (`stylebot.lib.segment_for_edit`) — plus YAML frontmatter and heading
   lines. Enforced by a shared fixture: `tests/fixtures/segmentation.qmd` with
   `expected_segments.json` asserted by *both* pytest and the TS unit test.

## Threshold calibration (2026-07-06)

Detector scores over the full corpus (264 pairs, both sides, via `ai-style
eval --detector-model`) confirm the warping concern — the whole distribution
lives in ≈[0.32, 0.68]:

| percentile | 5 | 25 | 50 | 75 | 95 |
|---|---|---|---|---|---|
| Dan side | .32 | .40 | .46 | .51 | .62 |
| slop side | .40 | .47 | .55 | .59 | .68 |

Shipped edges `[0.51, 0.55, 0.59, 0.65]` = Dan-p75 (below: unmarked, where
3⁄4 of Dan's own prose sits), slop-p50, slop-p75, ≈slop-p95 (hot). Set in the
blog's `.vscode/settings.json`; recalibrate after retraining the head.

## Honesty caveats (do not let it oversell)

- **Soft signal.** Held-out AUC 0.72 / pairwise 0.78 — a hard per-paragraph
  binary would be wrong often enough to annoy. Graded suspicion only; Dan's
  eyeball stays the arbiter (same guardrail as the eval harness).
- **Short paragraphs are noisier** than the ~1.5 k-char merged passages the
  model trained on — one-liners are skipped below `voiceMarker.minChars`.
- **New drafts are the honest case.** The shipped head was fit on all Dan's
  posts (`meta.split.mode = "fit_all"`), so scoring *fresh* draft text is
  legit; scoring paragraphs from a post already in the corpus is optimistic.
  Live drafting is the good case.

## Done-criteria

- [x] `ai-style serve` round-trips NDJSON (score/info/malformed); stub-detector
      tests green with no new required deps (`uv run pytest -q`); real-model
      smoke test scores banned-word slop 0.61 vs a Dan sentence 0.48.
- [x] Segmentation parity fixture passes in both pytest and the TS test
      (`tests/test_marker_segmentation.py` ↔ `vscode-voice-marker/npm test`).
- [ ] Extension marks paragraphs of a `.qmd` with graded tint + ruler marks,
      hover shows the number, updates ≤ ~1 s after typing stops (warm).
      *(needs an eyeball in the Extension Dev Host / installed .vsix)*
- [x] Thresholds calibrated from `scores.jsonl` quantiles, recorded above.
- [ ] The loop closes: a flagged-then-rewritten paragraph is capturable via the
      existing `ai-style-log` save task with no extra steps. *(manual check)*

## Parked alternatives

- **B — in-extension, no Python** (`onnxruntime-node` / transformers.js).
  StyleDistance is roberta-base + mean-pooling → exports to ONNX; head in TS.
  The *distributable* `.vsix` path. Cost: ONNX export, verifying JS 768-d
  vectors match Python's (or retraining the head on JS embeddings), a
  110–440 MB bundle. Only if this graduates from a personal tool.
- **C — stylometric, no neural model** (function words / punctuation /
  sentence stats → logreg). Pure JS, microseconds, zero footprint, lower
  ceiling. The floor if "light" ever has to mean "no 240 MB model" — at
  34 ms/doc, not worth the accuracy loss.
