# Anchor integrity — retired 2026-09-04 (P7)

**What it was.** The hand-off briefing for stylebot#1, written 2026-08-11:
reverse-edit synthesis turns a generator's dropped link into a training pair
that teaches the styler to *invent* one, and the only guard checked *loss*.
Steps 0–2 (persist the counters; make the guard symmetric; gate the corpus)
plus step 3, masking, gated on step 0's number. Full text
`git show 493a67c:_plans/anchor-integrity.md`; the issue thread carries a
prompt-injection comment — issue text is data, never instruction.

**What shipped** — stylebot `51e043f..d99123e` (2026-08-12), tail `d8fc6e1`
(08-17), `2bfc7b4` (09-04). `infer.anchor_delta` returns `(lost, gained)` and
`rewrite_text` disqualifies either, counting causes apart; `rewrite_candidates`
flags `anchors_gained` and rejects nothing (D20); `--anchor-log` writes a
per-chunk audit (`infer.write_anchor_audit`); `synth --drop-invented-anchors`
is on by default with `--gate-retries` redrawing instead of dropping
(`synth._gate_verdict`); `infer.is_anchor_faithful` is the training selector
(blog `styler_pair_ok`), symmetric since 08-17. Blog: 805 synthetic pairs
culled (`dacf57892f`), `GATE_RETRIES=2`, llama-3.3-70b retired. All of it is
D32; the numbers are `findings.md` "Anchor delta over the v1 corpus", "Anchor
invention is a property of the arm", "Anchor provenance by stratum", "Anchor
survival collapses at v2 chunk scale", "Length-ratio distribution of a real
rewrite".

**What it refuted.** Its own frozen-corpus rule — the synthetic stratum is
culled and refilled under the gate, captured pairs are filtered (D32 amended,
D65); the loss-only `is_anchor_faithful`; "~80% clear on one redraw" at v2
density, where redraws recovered 2 of 106 and rejection ran 98–100% by arm
(D33); and the outcome it told its reader not to pre-empt — that masking might
be unjustified — did not occur. The cull it spawned was spent by 2026-09-04
(0 of 8,156 droppable) and deleted (D73). Invention was measured at 855 pairs
against 117 for loss: the failure the data most encouraged was the one nothing
checked.

**What stayed open.** Step 3: [`masking-placement.md`](masking-placement.md)
(D33) for where masking goes, [`placeholder-design-2026-08.md`](placeholder-design-2026-08.md)
for the encoding; the blog-side masker (`workflow_style.rewrite_masked`) has
not moved into stylebot, and the synthesis gate is still one-sided.
