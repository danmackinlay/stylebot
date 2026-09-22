# Capture UX — correcting a pair without hashes

**Status 2026-09-10: opened; nothing built.** Follows D96, which made a real
capture's archive the editable truth and shipped `ai-style log recut` plus the
"archives edited since landing" line in `status`. Owner: Dan.
**2026-09-22:** the typo half of item 2's acceptance ("fix a typo in a capture
without leaving VS Code") is met from the other end by D107 — *Preen Paragraph*
fixes it in the file before the pair exists, and `recut --finish` fixes an
archive after. Locating a month-old archive from the editor is still open.

## The problem

The correction path speaks in eight-hex capture ids. Dan edits posts, does not
recall captures by hash, cannot see what a capture taught the model, and cannot
attribute a model behaviour to a source. So a workflow that is mechanically
sound is one he will not exercise.

## What is settled

- **The archive is the pair's moment, and corrections happen inside it.**
  Letting the after side follow the moving post was proposed and withdrawn on
  the numbers: half the live real pairs are rewritten or unrecognisable in
  today's posts (`findings.md` "Captured after-paragraphs against today's
  posts"). A pair is only pairwise sane. Nothing here changes what a pair is.
- Attribution of a fine-tuned model's behaviour to one pair is not on offer;
  what is on offer is seeing whether a pair *took*.

## Three items

**1. Everything by source, nothing by hash.**
- `ai-style log list <file>`: that post's captures, newest first — date, chunk
  count, the first words of each chunk, and a drift mark where the archive has
  moved since landing. No id needed to read it; ids shown for the escape hatch.
- `ai-style log recut <file>`: re-land every archive of that source; a bare id
  still works. The gap knobs are per landing, so the command reports which
  archives it re-cut and how many pairs each yielded.
- The `status` drift line prints source paths and first words, not ids.
- Acceptance: the RUNBOOK's correction section never mentions a hash.

**2. Jump from the paragraph to its archive.** The voice-marker extension knows
the file and the paragraph under the cursor. A command "Open this paragraph's
capture" asks the sidecar (a new `locate` op) which archive's `after.md`
contains the paragraph, opens it at that line, and offers to run `recut` for the
file when the archive is saved. The fix happens in the editor Dan is already
in, on the pair's own moment. Acceptance: fix a typo in a month-old capture
without leaving VS Code or typing an id.

**3. Replay per post.** `dan-style replay <file>`: for each live pair of the
post, run the promoted adapter over the captured before chunk and print a
three-way diff — slop, Dan's edit, the model's rewrite — with the detector's
`P(slop)` on each. Inference only, no judge spend, seconds per post. It answers
"did this edit take" and "did the model learn my typo" at the level of the file
Dan remembers. Real pairs in the styler split were trained on, so replay shows
fit, not generalisation; the eval stratum's posts show the latter. Acceptance:
one command, one post, readable without a number.

## Steps

1. Item 1 in stylebot (`session.summarize_state` grows a per-source detail;
   `recut_capture` accepts a source) and the blog's RUNBOOK §1b rewritten
   without ids. No new module; no D-entry needed.
2. Item 3 as a library pass over `infer.rewrite_text` and `capture.iter_archives`
   with a thin `dan-style` shell (D78); a D-entry if it becomes a module.
3. Item 2: the sidecar `locate` op, then the extension command. Last, because it
   depends on 1 for the recut call and is the only item with editor plumbing.

## Open

- Whether `replay` should also run over the eval stratum's captures in one go as
  a per-post companion to `eval --documents` (D54).
- Whether the drift mark should block a `train` dry run, or only report. Today it
  only reports, which matches the D96 premise.
