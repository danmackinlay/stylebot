# stylebot

Tooling to fine-tune a small open-weights LLM that rewrites AI-flavoured draft
prose into [Dan Mackinlay's](https://danmackinlay.name) voice — a specialist
*prose styler*, not a chatbot.

The motivation, design, and economics are written up in the blog post
[*Fine-tuning danbot*](https://danmackinlay.name/notebook/fine_tuning_danbot.html). The development
plan and phase contracts live in [`_plans/`](_plans/OVERVIEW.md).

## What's here

This repo is the **tooling** for the project. The **corpus** (training pairs)
and **secrets** (API keys) deliberately live outside the repo so it can be
public — see [Data & secrets](#data--secrets) below.

| Phase | Status | Component |
|-------|--------|-----------|
| 1 | ✅ shipped | `ai-style-log` — manual (slop → Dan) pair capture |
| 2 | 🔧 built | `ai-style synth` — synthetic-pair generation (OpenRouter multi-model) |
| 3 | planned | LoRA SFT (Tinker-trained; Fireworks/local-MLX served) |
| 4 | planned | `ai-style` inference CLI |
| E | 🔧 built | `ai-style eval` — multi-signal eval (Vale + LLM-judge + trained voice classifier) |
| — | 🔧 built | `ai-style serve` + [VS Code voice marker](#vs-code-voice-marker) — live P(slop) paragraph marks |

## Quickstart

```sh
uv sync
uv run ai-style-log --help
uv run ai-style --help
```

The pair-logger is run from inside the prose working tree (e.g. the blog repo),
where it writes captured pairs to `$STYLEBOT_DATA_DIR/pairs.jsonl`.

## VS Code voice marker

[`vscode-voice-marker/`](vscode-voice-marker/) is a small editor extension
that marks each prose paragraph of a markdown/Quarto document with a graded
**P(slop)** — a faint background tint plus scrollbar hotspots, the number on
hover — so suspect paragraphs get noticed, rewritten, and captured as training
pairs by the existing `ai-style-log` tasks. It is the front-end of the capture
loop, not a verdict machine: the classifier is a soft signal (held-out AUC
~0.72), so the marks are framed as *suspicion* and your eyeball stays the
arbiter. Design and caveats: [`_plans/vscode-marker.md`](_plans/vscode-marker.md).

### How it works

The extension spawns **`ai-style serve --detector-model DIR`** as a child
process — a long-lived sidecar (`stylebot.serve`) that loads a trained
voice-classifier artifact (`head.json` + `meta.json`, produced by the blog's
`train-voice-clf`) once (~5 s for the embedding model) and then scores batches
of texts over stdin/stdout NDJSON in tens of milliseconds per document. The
extension owns the process lifecycle: spawn on activation, status-bar spinner
until the `info` handshake, capped-backoff restart on crash, kill on window
close. You never launch the sidecar by hand.

Segmentation in the extension (`src/segment.ts`) is a **parity port** of
`stylebot.lib.segment_for_edit` + `split_paragraphs`: paragraphs split on
blank lines after dropping code fences, `$$math$$`, `:::` divs, blockquotes,
frontmatter, and heading lines — so the classifier sees the same prose shape
it was trained and evaluated on. A shared fixture
(`tests/fixtures/segmentation.qmd` + `expected_segments.json`) is asserted by
both pytest and the extension's `npm test`; if either side's segmentation
changes, the other's test breaks.

### Install & configure

```sh
cd vscode-voice-marker
npm install && npm test && npm run package
code --install-extension voice-marker-0.0.1.vsix
```

The extension is **dormant** until `voiceMarker.command` is set, so configure
it per workspace (the repo that holds your trained artifact) in
`.vscode/settings.json`:

```jsonc
{
  // run from the workspace folder (the extension's default cwd)
  "voiceMarker.command": [
    "uv", "run", "ai-style", "serve", "--detector-model", "_models/voice-clf"
  ],
  // ascending P(slop) bucket edges — calibrate to YOUR score distribution
  "voiceMarker.thresholds": [0.51, 0.55, 0.59, 0.65]
}
```

Other settings: `voiceMarker.minChars` (skip short/noisy paragraphs, default
80), `.debounceMs` (default 500), `.enabled`. Commands: *Rescore Document*,
*Toggle Marks*, *Restart Sidecar*.

**Thresholds are calibrated quantiles, not equal-width bins.** Raw detector
scores cluster in a narrow band (≈[0.32, 0.68] on Dan's corpus), so naive
0.2/0.4/0.6/0.8 edges would leave every bucket empty or full. Score your
corpus with `ai-style eval --detector-model`, take quantiles of the two
sides' detector scores, and set the edges at roughly human-p75 (floor of the
first mark), slop-p50, slop-p75, slop-p95. Worked recipe + current numbers:
[`_plans/vscode-marker.md`](_plans/vscode-marker.md) §"Threshold calibration".

### Updating after retraining the classifier

The artifact is data, not code — an improved classifier almost never touches
the extension:

1. **Retrain the head** (in the repo that owns the trainer, e.g. the blog:
   `uv run train-voice-clf train --pairs … --out _models/voice-clf`) and
   commit the refreshed `head.json` + `meta.json`.
2. **Restart the sidecar** — the *Voice Marker: Restart Sidecar* command (or
   reload the window). The sidecar rebuilds the detector from `meta.json` at
   spawn, so even a backbone swap needs nothing more: `meta.json` pins
   `embed_model`, and the dim-mismatch guard in `stylebot.classify` fails
   loudly if head and embedder disagree.
3. **Recalibrate `voiceMarker.thresholds`** — the score distribution moves
   with every retrain. Gotcha: `scores.jsonl` is resumable (already-scored
   ids are skipped), so point the recalibration run at a *fresh* `--out`
   path or you'll silently compute quantiles over the old model's scores.
   Threshold changes apply live, no restart.

Rebuilding/reinstalling the `.vsix` is needed only when the **segmentation
contract** (`stylebot.lib`) or the **wire protocol** (`stylebot.serve`)
changes — regenerate the shared fixture with
`STYLEBOT_REGEN_SEGMENTS=1 uv run pytest tests/test_marker_segmentation.py`,
confirm `npm test` passes, then `npm run package` and reinstall.

## Data & secrets

- **Corpus** — `pairs.jsonl` and open editing sessions are written under
  `STYLEBOT_DATA_DIR` (default `./_training_pairs`). That directory is
  **gitignored here**: the corpus never enters this public repo. Its
  canonical home is the (livingthing) blog repo, where it *is* committed
  (since 2026-07, for portability). See
  [`_training_pairs/README.md`](_training_pairs/README.md).
- **API keys** — copy `.env.example` to `.env` and fill in. `.env` is
  gitignored and never committed.
