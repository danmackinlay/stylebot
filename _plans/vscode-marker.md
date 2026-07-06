# VS Code paragraph marker — 📋 PLANNED (feasibility-checked 2026-07-06)

Mark paragraphs **dan-or-bot** live in the editor, using the trained voice
classifier. A paragraph's `P(slop)` shows as a graded gutter/scrollbar mark while
Dan drafts; suspicious paragraphs get rewritten, and the existing `ai-style-log`
tasks capture the `(before, after)` pair. The marker is the **front-end of the
capture loop**, not a standalone gadget.

Lives here (livingthing), not stylebot: the sidecar needs the *embedder*
(sentence-transformers), which is a livingthing dep — stylebot only owns the
dep-free head (`stylebot.classify`). Same mechanism/policy split as the trainer.

## Feasibility (measured, StyleDistance on Dan's Mac, MPS)

The only worry was latency; it's a non-issue. Numbers from `scratchpad/latency.py`:

| | latency |
|---|---|
| cold model load (once, at sidecar startup) | **5.0 s** |
| warm: 1 paragraph | 7 ms |
| warm: whole 30-paragraph notebook | **34 ms** |
| head dot-product → `P(slop)` (pure Python/TS) | 28 µs |

After a one-time 5 s load, re-scoring an entire document is ~34 ms — fast enough
to run on debounced idle keystrokes, not just on save. Latency is not the
constraint; the classifier *head* is genuinely light (17 KB JSON, `_models/voice-clf/head.json`).

## Architecture

**Recommended (A): a long-lived Python sidecar the extension manages.** It loads
StyleDistance once and scores paragraphs over stdin/stdout JSON-RPC (or localhost
HTTP). It is a thin wrapper over code that already exists —
`livingthing.voice_classifier.build_detector("_models/voice-clf")` /
`stylebot.classify.sklearn_detector` return `{score: P(slop), p_dan}` per text.
Highest fidelity (identical to `ai-style eval`); lowest effort given the env + HF
cache are already here.

Clean split inside A: the **sidecar is a generic `text → 768-vec` embedder**; the
**extension owns `head.json` + the dot product + the UI**. Retraining the
classifier is then a 17 KB JSON swap in the extension; the embedder never changes
(mirrors the project's mechanism/policy split). The sidecar can return either the
raw vector (extension applies the head) or the finished `P(slop)`.

**Parked alternatives:**
- **B — in-extension, no Python** (`onnxruntime-node` / `@xenova/transformers`).
  StyleDistance is `roberta-base` + mean-pooling → exports to ONNX; head in TS.
  This is the *distributable* `.vsix` path. Cost: one-time ONNX export, **verify
  the JS 768-d vectors match the Python ones** (or retrain the head on JS
  embeddings), bundle ~110 MB (int8)–440 MB weights. Do this only if it graduates
  from a personal tool to something shipped.
- **C — stylometric, no neural model** (function words / punctuation / sentence
  stats / `_voices/slop_patterns.md` n-grams → logreg). Pure JS, microseconds,
  zero footprint, but a lower ceiling (weaker on short paragraphs). The floor if
  "light" ever has to mean "no 240 MB model" — but at 34 ms/doc, not worth the
  accuracy loss.

## Components to build

1. **Sidecar** (`livingthing/bin/voice_serve.py`, new console script
   `voice-serve`): read JSON lines `{id, paragraphs:[...], contexts:[...]}` from
   stdin → `build_detector(model_dir)` → write `{id, scores:[p_slop,...]}`. Load
   the detector once; keep serving. ~100 lines.
2. **Extension** (`.vscode` extension, TS): activate on `.qmd`/`.md`; segment;
   debounce; call the sidecar; apply decorations; hover provider. ~200–300 lines.
3. **Segmentation must match training** (the parity that keeps scores honest):
   - split on blank lines (mirror `stylebot.lib.split_paragraphs`);
   - drop protected blocks — code fences, `$$math$$`, `:::` divs, blockquotes
     (mirror `stylebot.lib.editable_prose`), so they aren't scored as prose;
   - prepend the nearest section heading to each paragraph before embedding (the
     `stylebot.pairs.build_pair_content` contract the model trained under). v1 may
     skip the heading prefix — slightly off-distribution but acceptable.

## UX decisions

- **Graded, not binary.** Show a continuous `P(slop)` (colour intensity, or a
  0–1 number), framed as *suspicion*, never a hard "🤖/✍️" verdict — see caveats.
- **Marks:** a `TextEditorDecorationType` per paragraph range — a gutter dot +
  an `overviewRuler` mark (so the scrollbar shows slop hotspots at a glance) +
  optional faint line-background tint scaled by `P(slop)`. Hover → the number.
  (`CodeLens` "P(slop) 0.72" per paragraph is an option but noisier.)
- **Trigger:** debounced `onDidChangeTextDocument` (~300 ms idle) — 34 ms/doc
  makes this feel instant — plus an explicit "rescore" command.
- **Prior art:** the Vale VS Code extension uses the same "sidecar scores ranges →
  decorations" pattern; the `ai-style-log` tasks (`tasks.json` + `Cmd+K` chords)
  already wire capture into VS Code.

## Honesty caveats (do not let it oversell)

- **Soft signal.** Held-out AUC 0.72 / pairwise 0.78 — a hard per-paragraph
  binary will be wrong often enough to annoy. Graded suspicion only; Dan's eyeball
  stays the arbiter (same guardrail as the eval harness).
- **Short paragraphs are noisier** than the ~1.5 k-char merged passages the model
  trained on — expect more jitter on one-liners.
- **New drafts are the honest case.** The shipped head was fit on all Dan's posts
  (`meta.split.mode = "fit_all"`), so scoring *fresh* draft text (unseen) is
  legit; scoring paragraphs from a post already in the corpus is optimistic. Live
  drafting is the good case. (A `--holdout`-trained head would be cleaner if the
  score ever drives anything automated, but for a display it doesn't matter.)

## Done-criteria

- [ ] `voice-serve` sidecar: stdin JSON → per-paragraph `P(slop)`, detector loaded
      once; a fixture round-trips (slop paragraph scores higher than a Dan one).
- [ ] Extension marks paragraphs of the focused `.qmd` with graded `P(slop)`,
      updating on idle in < ~100 ms for a typical notebook (after warm load).
- [ ] Segmentation drops code/math/`:::`/blockquote blocks and splits on blank
      lines (parity with `editable_prose` / `split_paragraphs`).
- [ ] Hover shows the number; an `overviewRuler` mark surfaces hotspots.
- [ ] The loop closes: a flagged-then-rewritten paragraph is capturable via the
      existing `ai-style-log` save task with no extra steps.

## Effort

Path A ≈ a weekend: an evening for the sidecar (wrapping `build_detector`), a day
or two for the TS extension (segment → debounce → call → decorate → hover). Path B
adds a day or two for the ONNX export + vector-parity check.

## Open questions

- **Vector vs score over the wire** — sidecar returns the 768-vec (extension holds
  the head, hot-swappable) vs the finished `P(slop)` (simpler sidecar). Lean vec,
  for the clean head/embedder split.
- **Heading-context parity** in v1 — skip (simpler) vs prepend the nearest heading
  (on-distribution). Start skipped; add if scores look jumpy across headings.
- **Distribution** — personal tool (path A) forever, or eventually a shipped
  `.vsix` (path B)? Decides whether the ONNX-export work is ever worth it.
