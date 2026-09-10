# Coherence context — neighbour conditioning for the styler

**Status 2026-09-10: opened; nothing built.** Resurrects step 5 of
[`retired/v2-longer-rewrites.md`](retired/v2-longer-rewrites.md) and the
2026-08-21 amendment found uncommitted in a stale worktree (two-sided context,
now D97 with the fit judge, [`fit-judge.md`](fit-judge.md)). Owner: Dan. The
instrument exists and has a number to beat: run 4a vs run 2 at 62 boundaries
was 8–10 with 44 ties, against a control that separates Dan from slop 31–12.

## The problem

Every pair is an island. A record carries the section heading as context (D14)
and nothing else, so the styler is asked to rewrite a paragraph without the
referents of its pronouns, the terms the previous paragraph defined, or the
register either side of it. Inference matches: `infer.rewrite_text` sends
heading plus chunk per call, and the editor's `rewrite` op sends one paragraph.
No signal in the harness saw this until the fit judge (D97); the four older
ones score a passage alone (D54 for why a whole-document score does not help).

## Premise, in Dan's words (2026-09-10)

Editing is not linear. He starts in the middle, reorders, rewrites at several
levels, adds and deletes meaning, introduces typos and fixes them later. The
data must not be collected under "the fiction that I am a linear
style-rewriting bot". Principles: avoid catastrophically bad outcomes; stay
flexible; take incremental improvements; **dataset purity is not achieved by
dataset emptiness**; the one hard constraint is never to train toward a
high-cost failure — a subtle change of meaning, a mangled equation. Those can
be guarded by a priced semantic check rather than by discarding data.

## Design

**1. The context slot is two-sided and its state is mixed.** A record's
context becomes `heading`, `prev`, `next`. Each neighbour is drawn, at view
build, from a state mix: Dan's final text (archive `after.md`), the slop
(archive `before.md`), or absent; the two sides are drawn independently, with
a `CONTEXT_DROPOUT`-style rate per state that is *measured, not guessed*
(`masking-placement.md`'s warning). Why mixed rather than "the previous
rewritten paragraph": it is the serving distribution. In the editor the
neighbours are whatever state Dan left them in, often already his, sometimes
still slop; in whole-file `run` the previous neighbour is the styler's own
output and the next is slop. It is also the honest record of how Dan works.
The loss is on the target span only, so an indeterminate context is
regularisation, not a corrupted label. `sem` spans (D93) are cut from context
exactly as from the pair.

**2. Neighbours are re-assembled, never regenerated.** A real draw carries
`span` into `before.md`; the target is content-located in `after.md`; the
flanking prose either side, bounded by headings and protected blocks, is the
neighbour. The 104 pre-archive captures get no neighbours — that is the
"absent" arm, not a loss. A synthetic pair's neighbours are its sibling chunks
(`meta.source` + `chunk_index`, the fit judge's exact `sibling` source). All of
this is a `ViewPolicy` clause (`neighbour_context`) plus a rebuild: `record_id`
is unchanged, no draw is touched (D84, D96).

**3. Neighbours live in the user turn only.** D14 prepends the heading to both
sides so the assistant reproduces it and `pair_body` strips it. Copying two
whole paragraphs into every assistant turn would teach copying and spend the
loss on text that is not the lesson. So `prev`/`next` go into `messages[1]`
inside a fixed frame the system prompt names, and `messages[2]` stays the
rewritten span. That is one amendment to `STYLE_SYSTEM`'s policy slot, a view
rebuild under D64 as amended, and a `pair_body` that strips the frame.

**4. Inference passes what it has.** `rewrite_text` sends the previous *output*
chunk and the next *source* chunk; the editor sends `para_before`/`para_after`,
which the extension already records on every preference decision. The
do-no-harm guard, truncation check and anchor guard (D32) are unchanged.

**5. The high-cost failures get their own signal.** A semantic-fidelity judge:
a small model asked, per span, whether the rewrite asserts anything the source
does not or drops a claim it made, forced choice with a one-line reason. It is
the check that lets everything else be loose. Priced before it is wired: a
span-scale prompt is ~600 tokens, so the frozen eval stratum costs cents. It
runs offline with `eval`; wiring it into `run` as a gate is a later, separate
decision with its own latency and spend numbers.

## Measurement, before and after

- Fit judge (D97) with its positive control on the frozen boundary stratum,
  the promoted run against the conditioned run. Success: win-rate CI
  excluding 0.5, or a tie rate that falls with the control held decisive.
- Paired detector (D40) held or improved; anchor audit clean; the semantic
  judge shows no regression against the promoted run.
- Ablation for free: the "absent" arm of the state mix is today's format, so
  one adapter yields the conditioned-vs-bare contrast at inference by
  withholding neighbours.
- Per-post replay ([`capture-ux.md`](capture-ux.md) item 3): slop, Dan's edit,
  the model's rewrite, side by side for one post — the one reading that does
  not route through a number, and the plan exists partly because Dan's eye
  disagreed with the detector on run 4a.

## Steps

0. Port the fit judge onto main (D97, D98; `fit-judge.md`) — in progress.
1. Semantic-fidelity judge as a sixth eval signal, priced (its D-entry first).
2. `ViewPolicy.neighbour_context`: archive and sibling re-assembly, the state
   mix, the user-turn frame, validator and `pair_body` updates, tests.
3. The prompt amendment, as blog policy (`SOURCE.system_prompt`), and a view
   rebuild; `dan-style status` names the new `policy_sha`.
4. `rewrite_text` and the `rewrite` op take `prev`/`next`; the extension sends
   its two paragraphs; whole-file mode feeds its own previous output.
5. Train, then the measurement block above. Promote only on the fit judge.

## Open

- The state-mix rates. Measure the editor's actual neighbour states from the
  preference log before choosing them.
- Whether `next` leaks forward: the model pulling the next paragraph's terms
  into the rewrite. The semantic judge and the fit judge both see it.
- Two-sided context roughly triples prompt length; check the 8192-token cap
  against the 4500-char arm before training.
