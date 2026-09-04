# VS Code paragraph marker — retired 2026-09-04 (P7)

**What it was.** Mark paragraphs dan-or-bot live in the editor with the
trained voice classifier — a graded `P(slop)` tint, ruler mark and hover,
"Rewrite in Dan's Voice" on a selection — as the front-end of the capture
loop, not a gadget. Designed and calibrated 2026-07-06. Full text
`git show ff0e994:_plans/vscode-marker.md`.

**What shipped** — stylebot `4daf001..bb50a86` (2026-07-06 → 2026-09-03):
`ai-style serve` (`serve.serve_loop`: NDJSON over stdin/stdout, ops `info` /
`score` / `segment` / `rewrite` / `prefer`, malformed input answered, never a
crash) wrapping `classify.sklearn_detector`; the extension
`vscode-voice-marker/` at 0.0.6, dormant unless `voiceMarker.command` is set,
owning the sidecar lifecycle (capped-backoff restart, per-op timeouts) and its
preference records (`offsets`, `para_before`/`para_after`). Segmentation is
one implementation, `serve.prose_paragraphs` over `stylebot.markdown` with the
sidecar's `div_is_prose`, UTF-16 offsets converted server-side, pinned by
`tests/test_serve.py` under both policies (D44). Scores over the wire, not
vectors; bare body text, no heading prefix (`pairs.pair_body`). The blog
contributes the artifact, `dan-style serve`, and the thresholds in its
`.vscode/settings.json`. Numbers, and the quantiles the thresholds are cut
from: `findings.md` "The voice classifier", "The sidecar `segment` op",
"Segmenter implementations".

**What it refuted.** Latency was the only worry and is nothing (5 s once, 34
ms a document, the `segment` hop under 5 ms on the largest post). Prepending a
heading was wrong, not deferred. The TypeScript segmenter kept in parity by a
fixture (D28) had drifted on 29 posts and could not see 385 callout
paragraphs; it is gone (D44). LSP has no decoration concept, and the
preference corpus is biased by presentation and selection, logged so (D20).
The head is no longer `fit_all` — it is fit under `splits.json` — so "fresh
drafts are the honest case" is weaker than written.

**What stayed open.** Two done-criteria never got their eyeball: the
repackaged 0.0.6 `.vsix` in the Dev Host, and the loop closing through
`ai-style log`. The thresholds predate two refits of the head and want
recalibrating from `scores.jsonl` quantiles; "rewrite section" is allowed by
`rewrite_candidates` and not yet offered. **Alternative B** — in-extension
ONNX, no sidecar, a distributable `.vsix` — is parked; D44 is what a port must
get right.
