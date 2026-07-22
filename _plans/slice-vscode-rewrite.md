# Slice · VS Code rewrite UX — ✅ BUILT 2026-07-22

Interactive best-of over the serve sidecar.

Grow the `voice-marker` extension from a passive P(slop) meter into the
interactive front-end of the styler: select prose (or sit in a paragraph),
ask for rewrites, choose among candidates, and have the choice recorded as
preference data. The marker already tells Dan *where* the slop is; this
closes the loop by offering the fix in place.

## Why here, why this shape

- **The server already exists.** `ai-style serve` is a long-lived NDJSON
  sidecar the extension spawns and keeps warm; the detector loads once.
  Adding rewrite ops to the same process means the candidates come back
  pre-scored with zero extra infrastructure — no HTTP server, no ports.
- **All the weird logic stays in Python.** Chunk protocol, heading context
  (`build_pair_content`/`pair_body`), anchor integrity, scoring — already
  in `stylebot.infer`, already tested. The extension stays dumb: send text,
  show candidates, report the choice.
- **Prior art doesn't cover it.** Generic shell-filter extensions
  (Filter Text, edit-with-shell) can pipe a selection through `dan-style
  run` but give one candidate, no scores, no diff, no preference capture.
  Generic AI-edit extensions (Copilot inline, Continue) bypass the chunk
  protocol and guards entirely.

## Protocol (stylebot.serve, mechanism)

Two new ops beside `info`/`score`, armed by injected seams (score-only
serving is unchanged — both ops error politely when unarmed):

    -> {"id": 3, "op": "rewrite", "text": "...", "context": "## Heading",
        "best_of": 4}
    <- {"id": 3, "input_score": 0.52,
        "candidates": [{"text": "...", "score": 0.31, "anchors_lost": []},
                       {"text": "...", "score": 0.35,
                        "anchors_lost": ["./cooperation.qmd"]}]}

    -> {"id": 4, "op": "prefer", "decision": {"chunk": "...", "context": ...,
        "candidates": [...], "scores": [...], "chosen": 0,
        "kept_input": false, "doc": "notebook/foo.qmd"}}
    <- {"id": 4, "ok": true}

Design choices:

- **Candidates are returned in full, none auto-chosen.** The CLI's
  do-no-harm guard picks for you; the editor's whole point is that *Dan*
  picks. Anchor violations therefore FLAG (`anchors_lost`) rather than
  disqualify — the human can see what a candidate dropped and take it
  anyway (run-3 lesson: sometimes the deletion is the edit you wanted;
  usually it isn't, and the flag says which).
- **Single-chunk contract.** `rewrite` takes prose at training chunk scale
  (`STYLE_CHARS_PER_CHUNK`); selections containing headings/fences or over
  budget get a clean error, surfaced as a friendly message. Multi-chunk
  interactive flows are per-chunk sequential decisions — out of scope for
  v1 (the paragraph-at-cursor default keeps everything under budget
  naturally).
- **The styler builds lazily** on the first `rewrite` — the editor can
  start offline and still score; rewrite errors surface per-request.

`stylebot.infer.rewrite_candidates(text, styler, *, context, best_of,
scorer)` implements the op: validates the single-chunk contract via
`_segment`, samples, cleans (`pair_body`), anchor-checks, scores.

## CLI (mechanism → policy)

- `ai-style serve` gains the styler flags of `ai-style run` (`--manifest`
  for the Tinker backend, `--base-url`/`--model` for any OpenAI-compatible
  endpoint, `--temperature`) plus `--preferences PATH` to arm the prefer
  sink. No styler flags → score-only, exactly today's behaviour.
- `dan-style serve` (new mirror, livingthing): detector `_models/voice-clf`,
  checkpoint from the newest **promoted** manifest, preferences to
  `_training_pairs/preferences.jsonl`. The workspace spawn becomes
  `direnv exec . uv run dan-style serve` (direnv carries TINKER_API_KEY;
  a local MLX endpoint slots in with the openai flags, keyless).

## Extension (voice-marker 0.0.2)

One command, `voiceMarker.rewrite` ("Rewrite in Dan's voice"), editor
context menu + palette:

1. Target = non-empty selection, else the prose paragraph at the cursor
   (existing `proseParagraphs` offsets).
2. Context = nearest markdown heading above the target.
3. `withProgress` while sampling (net + N samples ≈ seconds).
4. QuickPick, candidates ascending by P(slop): label carries score and Δ
   vs input, description the first line; anchor-losing candidates get a
   `$(warning) loses: …` marker; last item is "Keep original". Esc aborts
   — no replacement, **no preference logged** (an abort is not a
   judgement).
5. Accept → `WorkspaceEdit` replace (one undo step) → `prefer` op
   (fire-and-forget) → rescore, so the marker repaints on the new text.

Config: `voiceMarker.bestOf` (default 4).

## Preference capture — how natural is the DPO leverage?

Very. `preferences.jsonl` was *designed* as the Phase-7 shape ({chunk,
context, candidates, scores, chosen, kept_input}); the editor path writes
the same records with two additions: `chooser: "human"` (CLI best-of
records are implicitly detector-chosen) and `doc`. Each accepted rewrite
yields chosen-vs-rejected pairs on real blog prose in situ — best-of 4 ≈
3 DPO pairs per invocation; "keep original" is the input preferred over
every candidate, also usable signal. Ordinary editing accrues training
data with zero extra ceremony, and human choices are strictly stronger
labels than the detector's.

Two honesty caveats, recorded now for the Phase-7 consumer:

- **Presentation bias**: candidates are shown sorted by detector score, so
  human picks are anchored toward the detector's ranking. `scores` +
  `chooser` are logged, so the bias is reconstructable; if it matters,
  shuffle presentation later and log the shown order.
- **Selection bias**: rewrites get requested where the marker glows red —
  the preference corpus over-samples high-P(slop) prose. Probably the
  *right* bias (that's where the styler operates), but worth remembering.

## Done-criteria

- [x] `stylebot.serve` rewrite/prefer ops, injected seams, tests (stub
      styler/sink — no ML deps, matching the module's discipline).
- [x] `rewrite_candidates` in `stylebot.infer` + tests (contract
      violations, anchor flagging, score alignment).
- [x] `ai-style serve` styler flags; `dan-style serve` mirror; mirror-rule
      test still passes. (Both build the styler LAZILY on first rewrite —
      a flaky network at editor start must not take scoring down.)
- [x] voice-marker 0.0.2: rewrite command (selection or paragraph-at-
      cursor, heading context, QuickPick sorted by score with anchor-loss
      warnings, Esc aborts unlogged), packaged and installed into the
      `quartosites` profile.
- [x] End-to-end (2026-07-22): NDJSON smoke against the real sidecar —
      handshake reports the PROMOTED run (run 2, the nixed run 3 correctly
      skipped), rewrite moved a slop chunk 0.589 → 0.230 with the link
      intact, and `prefer` landed the full `chooser: "human"` record
      (removed after verification — smoke data is not preference data).
      Remaining: first real in-editor use is Dan's (reload the window so
      0.0.2 activates; the sidecar spawn is now
      `direnv exec . uv run dan-style serve`).
