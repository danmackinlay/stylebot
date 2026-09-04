# v2 · Longer rewrites — retired 2026-09-04 (P7)

**What it was.** Dan's 2026-07-22 note: paragraph-by-paragraph rewriting loses
coherence and 1500 chars is a third of the length wanted. The plan diagnosed
why the styler was pinned at 1500, laid the groundwork so the data problem
stopped compounding, and set the path to a scale-mixed adapter. Full text
`git show ff0e994:_plans/v2-longer-rewrites.md`.

**What shipped** — stylebot `f5abafc..7cfe445` (2026-07-23 → 2026-09-03):
the capture archive (`capture.capture_pairs`, directories of `.md`) and
`para_before` in preference records; `meta.merge_max_chars` / `meta.chunk_mode`
on every pair (`targets.Target`); cross-heading chunking with headings as
content (`targets.ChunkPolicy`, `infer.prose_chunks`) and the one amendment
ever made to `STYLE_SYSTEM` (D64); gap-coalesced recuts (`capture.diff_chunks`,
`ai-style log recut`, D81); serving that reads the diet off the manifest
(`backends.serving_diet`); `per_passage` / `pair_weight` (D43). Blog: the
1500/3000/4500 curriculum generated 2026-08-12/13 under the anchor gate, run 4a
and 4b trained 2026-08-17 (`8cbb626696`, `d38532f48c`); which serves is
`dan-style status`. Numbers: `findings.md` "Section length", "Chunk yield by
budget and mode", "Real captures", "Capture diffing", "v2 post-gate
distributions", "Chunk budgets overlap", "Effective passage weight", "Run 4's
verdict, re-analysed".

**What it refuted.** Raising the byte budget does nothing — Dan's sections are
short, so packing across headings is the only route (D15); a fixed-width
capture window teaches the styler to leave slop alone (D81), and `diff_chunks`
drops one opcode in 41 archives; the budget stamp names the first batch that
reached a passage, not a design cell, so "did scale help" cannot be a facet —
the arms are nested and have three honest analyses (D82), and the same sweep
is a 4× up-weight of synthetic prose (D43); run 4a's gain over run 2 straddles
zero clustered by post (D40); the anchor rejection curve does not plateau at
v2 density (D33).

**What stayed open.** Step 5, conditioning on the previous rewritten paragraph
— decide after a v2 eval, not before. The detector is paragraph-scale and the
editor thresholds `[0.51, 0.55, 0.59, 0.65]` predate two refits; the
extension does not yet offer "rewrite section". Choosing a weighting policy
is Dan's call (D43); the masking work is
[`masking-placement.md`](masking-placement.md).
