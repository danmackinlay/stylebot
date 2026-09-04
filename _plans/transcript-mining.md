# Transcript mining — real pairs from Claude Code session logs

**Status: specified 2026-09-03, not built.** A capture-expansion slice: grows
the *real* stratum, which the [`argued`](argued-slop-arm.md) synth arm is
trying to approximate. Self-contained; a subagent can build it from this file.

## Why

The real captures (398 pairs) are the only slop with provenance: each one is
Claude answering a specific objection that was live in an editing session,
which is the structure synth cannot fabricate (the justificatory scaffolding
Dan strips — `argued-slop-arm.md` §"Risks"). The sessions that produced them
are on disk as Claude Code transcripts, and they record every edit as an
explicit diff, so the slop side is *read off*, not recovered by matching.

Measured 2026-09-03 on `~/.claude/projects/-Users-dan-Source-livingthing/`:

| | |
|---|---|
| sessions on disk | 209 (120 edit a `.qmd`) |
| `Edit` calls on `.qmd` files | 2,798 — each with exact `old_string` / `new_string` |
| edits rewriting text Claude wrote earlier in the *same* session | 1,370 (49%) — the layering Dan expects |
| unique terminal Claude paragraphs (chains chased to their last version) | 1,439 |
| … verbatim in HEAD (accepted) | 151 |
| … edited by Dan, difflib ratio 0.80–0.97 | 154 |
| … edited by Dan, ratio 0.60–0.80 | 133 |
| … unmatched (< 0.60: deleted, moved, restructured, or drifted) | 993 |
| terminal paragraphs with a substantive user turn earlier in the session | 96% |

So ~290 clean pair candidates against HEAD, i.e. **+70% on the real stratum**,
and that is a floor: matching against HEAD conflates Dan's edit with months of
later drift, which the time-bounded design below removes. The join to the
*existing* 398 captures is small (29 exact) because the logger snapshots a
file when Dan opens his own editing session, after Claude's session has ended
— the two sources sit beside each other rather than overlapping.

## Design: replay the session, then reuse `capture_pairs`

Do **not** paragraph-match (the quick estimate above did, and at ratio 0.6 it
pairs unrelated paragraphs). Reconstruct the two document states and hand them
to the existing differ, which already does paragraph alignment, coalescing,
heading context, archiving and provenance.

1. **Parse** the transcript (`stylebot.transcripts`): one JSONL per session;
   records with `type` in `{user, assistant}`, `timestamp`, `sessionId`,
   `isSidechain`, `cwd`; assistant `message.content` is a list whose
   `tool_use` items carry `name` and `input`. Take `Edit` (`file_path`,
   `old_string`, `new_string`, `replace_all`) and `Write` (`file_path`,
   `content`) on `.qmd` paths under `cwd`. Take user text turns too (string
   content, or list items of `type: text`), skipping tool results and
   `<`-prefixed system-reminder blocks. Sidechain (subagent) edits are still
   Claude prose: keep them, stamp `sidechain: true`.
2. **Base state** of each touched file: the transcript's own last `Read`
   tool result for that path before the first edit (the file exactly as
   Claude saw it) — falling back to `git show <sha>:<path>` at the last commit
   before the session's first edit timestamp. Record which.
3. **Replay** the session's edits in order onto the base (`str.replace` of
   `old_string`, first occurrence unless `replace_all`; `Write` replaces the
   whole file). An `old_string` that does not apply means the base is wrong:
   stop replaying that file, count it, move on. The result is the **before**
   text: the file as Claude left it, layering resolved by construction.
4. **After** text: the file at the first commit whose author date is after
   the session's last edit timestamp and which touches the path (`git log
   --reverse --after=<ts> -- <path>`); if none, the working tree, with
   `blog_dirty` stamped true by `repo_provenance`. **Skip the window** if
   another session edits the same file between session end and that commit —
   the after text would contain Claude's later prose. Count skips.
5. **Diff and write** with `capture.capture_pairs(before, after,
   pairs_path=…, source=<repo-relative path>, snapped_at=<session end>,
   extra_meta={…}, blog_root=…)` — the same call `ai-style log save` makes,
   so records are schema-identical to hand captures, archived under
   `captures/<id>/`, and carry `meta.provenance` (`source_automation` from the
   *after* frontmatter, as the logger does).
6. **Dedup** before writing: skip a window whose after-bodies are all already
   `covered_target_keys` of the corpus (hand captures of the same edit), and
   skip sessions already mined (`meta.session_id` present).

`extra_meta` (stamp on every record; add the keys to `pairs._META_TYPES` if
`validate_pair_record` rejects unknowns — verify, don't assume):
`producer: "transcript"`, `session_id`, `session_edits` (count applied),
`base: "read" | "git"`, `sidechain: bool`.

**The deliberation sidecar.** The transcript holds both halves of the
regime the real slop comes from: Dan's pushback and Claude's own account of
what it did and why. Measured 2026-09-03 across the 209 sessions: 8,951
`thinking` blocks (8.2M chars) and 6,134 visible `text` turns (5.2M chars)
beside the 15,374 tool calls, and the text turns are exactly the "explaining
the outcome" prose — *"the sentence leaves the noun implicit and also skates
over a logical wrinkle: if no such prices exist, no process can find them, so
… Making both explicit:"* — followed by the Edit. Keep both, per edit:

- **User turns**: substantive only (> 25 words, not tool output, not
  `<`-prefixed reminder blocks), the ones preceding the edit in the session.
  These are the real objections the `argued` arm's stage 2 samples from.
- **Claude turns**: the `text` blocks and `thinking` blocks in the same
  assistant message as the Edit and in the message before it. These are the
  deliberation whose summary the slop is — the material for testing Dan's
  original hypothesis directly (does the slop restate the trace?), which
  `--capture-reasoning` was built for and never ran.

Write them to `<data-dir>/transcript_turns.jsonl`, one line per capture
keyed by `capture_id` with `{user: [...], claude_text: [...],
claude_thinking: [...]}`, each entry truncated to a few thousand chars — the
way `reasoning.jsonl` is kept beside, never inside, `pairs.jsonl`. It is Dan's
private typing and Claude's raw traces: **gitignored in the blog's data dir**,
never a pair field, and `--no-turns` skips writing it.

## Interface

- **Library**: `stylebot.transcripts.iter_sessions(dir) -> Iterator[Session]`,
  `Session.edits_by_file`, `Session.user_turns`, `replay(base, edits) ->
  str | ReplayError`; `stylebot.capture.mine_transcripts(transcripts_dir,
  *, blog_root, pairs_path, since=None, dry_run=True) -> MineReport` (per-file
  outcome counts: mined / already-covered / base-missing / replay-failed /
  window-skipped, and the pair count).
- **CLI**: `ai-style log mine --transcripts DIR --blog-root … --data-dir …
  [--since DATE] [--apply]` — under the `log` group because it is a producer,
  not a migration; dry-run by default and prints the report table, `--apply`
  writes. `dan-style log mine` supplies the transcript dir and paths from
  policy (the Claude Code path is a *caller* fact, like `--blog-root`).
- **Explicit paths, no defaults** on `--transcripts` / `--data-dir`: this
  appends to the corpus.

## Tests (`tests/test_transcripts.py`, no network, no real transcripts)

- A 6-line fixture transcript: a `Read` result, two chained `Edit`s on one
  file (the second's `old_string` inside the first's `new_string`), one
  sidechain `Edit`, one substantive user turn, one `please continue`. Assert
  chain resolution, sidechain stamping, turn filtering.
- `replay` applies in order and reports the first non-applying `old_string`.
- End to end in `tmp_path`: `git init`, commit a base file, the fixture
  session edits it, commit "Dan's" version after the session timestamp;
  `mine_transcripts` yields exactly the changed regions via `capture_pairs`,
  records validate, `dry_run` writes nothing, a second `--apply` is a no-op
  (dedup), and a window with an intervening session is skipped and counted.

## Content drift: weight insertions down, leave deletions alone

Dan's edits are not pure register changes, and this slice grows exactly the
stratum where that matters. The two directions train different things:

- **Deleting Claude's content is the target behaviour.** The justifications
  Dan strips are content Claude invented (an argument against an alternative
  from the session). A styler that removes them is doing its job, and
  `MAX_LENGTH_RATIO` bounds over-deletion.
- **Inserting Dan's content is the hazard.** A pair whose Dan side adds a new
  claim or aside (*"What roll-outs can a gene do?"* — `empowerment.qmd`)
  teaches the styler that Dan-ifying includes asserting what Dan would have
  said. That is the anchor-deletion failure of the early runs in a worse
  form: invented opinions in Dan's voice.

So stamp every real pair — mined *and* hand-captured — with a
**`novel_after` score**: the fraction of after-side sentences with no near
match on the before side (sentence-level `difflib` ratio < 0.5 against every
before sentence; keyless, no model). Written as `meta.novel_after` by
`build_pair_record` when both bodies are present, backfilled over the
existing 398 by a one-shot `ai-style data` migration (dry-run default,
`--apply`, `.bak`, test — the `migrations` contract). It is a *covariate*,
not a gate: the blog's `pair_weight` policy in `training_targets` maps it to
a weight (e.g. 1.0 below 0.2, linear to 0.25 at 0.6, and the selector drops
pairs above that), and the number is recorded so the threshold can be
re-fit against eval rather than guessed once. `dan-style status` should
print the real stratum's `novel_after` distribution beside its count.

## Done-criteria

- [ ] `meta.novel_after` on every real pair, the backfill migration run over
      the existing captures and its distribution in `findings.md`; the
      `pair_weight` policy landed blog-side *before* the mined pairs enter a
      training run.

- [ ] `dan-style log mine --apply` on the real transcripts writes ≥ 200 new
      validating real pairs; the report's skip/failure counts are in
      `findings.md` with the yield.
- [ ] `dan-style status` shows the real stratum grown, and `validate-pairs`
      still passes corpus-wide (`record_id` uniqueness across shards).
- [ ] `transcript_turns.jsonl` exists, is gitignored in the blog, and
      `argued-slop-arm.md` stage 2 can name it as its objection source.
- [ ] The syntax-proxy table in `argued-slop-arm.md` re-derived over the
      grown real stratum (it is the reference the arm is judged against).

## Non-goals

- Paragraphs Dan deleted outright (most of the 993 unmatched) are not pairs;
  `diff_chunks` already drops pure deletes. A "what Dan deletes" study is a
  different slice.
- Transcripts from other tools (Claude Desktop chats, Cursor) have other
  formats; the parser is Claude Code's JSONL only, and says so.
