# Voice Marker

Marks paragraphs of markdown/Quarto documents with a graded **P(slop)** —
background tint, scrollbar hotspots, hover for the number — scored by a
stylebot voice-classifier sidecar (`ai-style serve`). Design + honesty
caveats: [`../_plans/vscode-marker.md`](../_plans/vscode-marker.md).

The extension is dormant until `voiceMarker.command` is set, so configure it
per workspace (the repo that holds your trained artifact), e.g. in
`.vscode/settings.json`:

```jsonc
{
  "voiceMarker.command": [
    "uv", "run", "--project", "/path/to/blog",
    "ai-style", "serve", "--detector-model", "_models/voice-clf"
  ],
  "voiceMarker.cwd": "/path/to/blog",
  // Calibrate to your detector's score quantiles, not equal-width bins.
  "voiceMarker.thresholds": [0.5, 0.55, 0.6, 0.65]
}
```

The sidecar is spawned on activation (status bar: spinner while the embedding
model loads, ~5 s), restarted with backoff if it crashes, and killed on
window close. Commands: `Voice Marker: Rescore Document`, `Toggle Marks`,
`Restart Sidecar` (after retraining the head).

## Build, test, install

```sh
npm install
npm test            # segmentation parity vs ../tests/fixtures (shared with pytest)
npm run compile     # esbuild → dist/extension.js
npm run package     # vsce → voice-marker-<version>.vsix
code --install-extension voice-marker-0.0.1.vsix
```

Dev loop: open this folder in VS Code and F5 (Extension Development Host).

Segmentation (`src/segment.ts`) must stay in behavioural lockstep with
`stylebot.lib.segment_for_edit` / `split_paragraphs` — the shared fixture
test on both sides pins it. If stylebot's segmentation changes, regenerate
the expectation with `STYLEBOT_REGEN_SEGMENTS=1 uv run pytest
tests/test_marker_segmentation.py` and re-run `npm test`.
