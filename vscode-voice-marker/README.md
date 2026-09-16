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
window close. One request is in flight at a time: the sidecar writes a reply
before it reads the next line, so a long `rewrite` delays a queued `score`.
Commands: `Voice Marker: Rescore Document`, `Toggle Marks`,
`Restart Sidecar` (after retraining the head), `Switch Detector`.

## Several detectors, one marking (stylebot D106)

`ai-style serve --detector voice-clf --detector sapling` arms both the local
voice head and Sapling's hosted human-vs-AI detector (`$SAPLING_API_KEY`).
The sidecar's `info` lists what it armed; `voiceMarker.detector` (or the
*Switch Detector* command) picks which one marks, and its edges come from
`voiceMarker.thresholds` keyed by name — a detector with no entry is
*uncalibrated* and marks nothing. Sapling spends per character sent, so set
`voiceMarker.trigger` to `save` or `manual`; the sidecar answers unchanged
paragraphs from a cache, and the status-bar tooltip shows the session's spend.
The two instruments are never blended: *Rewrite in Dan's Voice* is reranked by
the voice head only, whichever one is marking.

```jsonc
{
  "voiceMarker.command": ["direnv", "exec", ".", "uv", "run", "dan-style", "serve",
                          "--detector", "voice-clf", "--detector", "sapling"],
  "voiceMarker.trigger": "save",
  "voiceMarker.thresholds": {
    "voice-clf": [0.27, 0.32, 0.77, 0.97],
    "sapling-20251027": [0.5, 0.7, 0.9, 0.97]
  }
}
```

## Marking a content edit (`Cmd+K Cmd+M`)

`Voice Marker: Mark Selection as Content Edit (sem)` toggles `<!-- sem -->…<!-- /sem -->`
around the selection, or around the paragraph under the cursor when nothing is
selected. The markers tell stylebot's view that the span changed what the text
*says*, not how, so it is cut out of the training pair instead of taught as
style (stylebot D93; the edit protocol is the blog's `_training_pairs/RUNBOOK.md`
§0c). Pure text logic in `src/sem.ts`, tested without VS Code.

## Build, test, install

```sh
npm install
npm test            # sidecar client: the segment/score wire, timeouts, restarts
npm run compile     # esbuild → dist/extension.js
npm run package     # vsce → voice-marker-<version>.vsix
code --install-extension voice-marker-0.0.1.vsix
```

Dev loop: open this folder in VS Code and F5 (Extension Development Host).

This package has no segmenter. Which spans of a document are prose is the
`segment` op's answer (`stylebot.serve.prose_paragraphs`), under the
`div_is_prose` policy the sidecar was started with — so a stylebot-side
segmentation change reaches the editor by restarting the sidecar, not by
repackaging the `.vsix`. There was a TypeScript port here until 2026-09-03;
stylebot `_plans/decisions.md` D44 records what it had drifted into and why the
fixture that pinned it could not have caught the worst of it.
