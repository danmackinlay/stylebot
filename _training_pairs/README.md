# `_training_pairs/` — the corpus

This directory holds the **training corpus**: the captured and synthesised
`(slop → Dan)` pairs the fine-tune is trained on, the project's valuable and
private asset.

**Everything in here except this README is gitignored.** The stylebot code
repo is public tooling and never carries the data. Do not `git add -f` it.
The *canonical* corpus lives in the livingthing repo, where its `draws/` **are**
committed — the never-commit rule is about this repo, not the data.

This file is the user-facing manual for `ai-style log`, the Phase 1 capture
tool; the schema contract is in
[`../_plans/OVERVIEW.md`](../_plans/OVERVIEW.md).

## Layout

```
_training_pairs/
├── README.md                     # tracked (this file)
├── draws/                        # the corpus: append-only JSONL shards
│   ├── captures.jsonl            #   one draw per changed region of a save
│   └── synth-<YYYY-MM>.jsonl     #   synthetic draws, one shard per month
├── views/pairs-<policy-sha>/     # built from the draws under a ViewPolicy
├── captures/<capture-id>/        # per-save archive: before.md, after.md, meta.json
├── snapshots/<rel-path>/         # open sessions: content.md + meta.json
└── snapshots-closed/<rel-path>/  # closed sessions, retained so `reopen` works
```

Two layers, D84. A **draw** is one human save or generator call, appended and
never rewritten; a superseded save stays on disk and the **view** drops it. A
view is a function of the draws and a `ViewPolicy` whose sha names the
directory, built on demand by whichever command reads the corpus
(`ai-style build-view` builds it now and prints the path). So a prune or a
re-cut is a policy clause plus a rebuild, not an edit — the one exception being
a real capture's archive, which is editable truth ("Fixing a capture after the
fact", D96).

Open sessions live under `snapshots/` because the file *is* a snapshot of the
source's initial state — that names its role at the storage layer. The CLI verb
is `open` because that names your *action*.

## Where the data lives (`STYLEBOT_DATA_DIR`)

The corpus location resolves through the shared ladder
`--data-dir` > `$STYLEBOT_DATA_DIR` > the cwd-relative default
`./_training_pairs`. Capture runs inside the prose working tree, so with no env
var set draws land in `<blog>/_training_pairs/draws/`; the downstream phases
(synthesis, training, eval) run from *this* repo and point at that same corpus:

```sh
export STYLEBOT_DATA_DIR=/path/to/your/corpus
```

One canonical corpus path, rather than copies scattered per repo, avoids the
classic "which corpus is the real one" failure. `log` deliberately takes no
`--data-dir` flag — it is the convenience command, where a mistake is cheap,
while `train` and `make-splits` demand one. Every `log` command that writes
echoes the resolved **absolute** draws path and the running pair count, so a
save landing somewhere unexpected is visible immediately.

## The session cycle

Every rewrite of a file is a **session**: opened explicitly, saved when
finished, optionally banked mid-stream and superseded later.

```sh
uv run ai-style log open <file>    # begin a rewrite session
# ... you edit the file by hand, across one or more sittings ...
uv run ai-style log save <file>    # write draw(s) for what changed, close session
```

The session is **sticky** — `open` refuses to overwrite an existing one, so the
original ("worst") state of the file survives across editing sittings. Open once
at the start of a rewrite project and save when you are happy; you do not have
to finish in one sitting.

Every command except `list` ends by printing a short reminder of
currently-open sessions to stderr, with their relative age, so stale sessions
become visible; `-q` / `--quiet` at the group level
(`ai-style log -q save <file>`) suppresses it for one invocation.

`drop <file>` abandons an open session without writing pairs; `list` shows open
sessions, live pairs grouped by source, and the closed sessions a `--redo`
could reopen.

## Chunk-level save (the default)

`save` does **not** emit a single whole-file pair. It diffs the session's
recorded "before" state against the current file at paragraph granularity and
emits one pair *per changed region*, skipping untouched paragraphs entirely.

Why: a real rewrite touches a few paragraphs out of many, so a whole-file pair
teaches "preserve everything except these spans" instead of the transformation
itself. Per-chunk pairs also match the chunk size Phase 2 uses for synthetic
pairs, so both strata are shape-compatible.

The diff is paragraph-level (split on blank lines) plus `difflib`
`SequenceMatcher` opcodes. **Contiguous changed paragraphs coalesce into one
pair**, regardless of N → M paragraph counts on either side — the `replace`
opcode pairs spans, not paragraphs. A pair breaks when an *untouched* paragraph
sits between two changed regions and the gap is wider than the caller's
coalescing tolerance (D81); `save`'s own tolerance is zero.

| edit shape | pairs emitted |
| --- | --- |
| rewrote whole document | 1 (entire body → entire body) |
| rewrote paras 2..4 of 5 | 1 (paras 2+3+4 → the rewrite) |
| rewrote paras 2 and 4 of 5 | 2 (the gap at para 3 breaks them) |
| rewrote 3 paras into 2 | 1 (3 joined → 2 joined) |
| rewrote 1 para into 5 | 1 (1 → 5 joined) |
| moved a paragraph | 0 (pure delete + pure insert) |

Pure inserts (paragraphs you added with no source) and pure deletes (slop
removed entirely) are skipped — there is no transformation to learn.

Use `--whole` when the diff would be actively misleading: heavy structural
rearrangement, paragraph moves, or sentence-level edits scattered across every
paragraph (where the chunk algorithm emits one giant `replace` anyway).

Each chunk carries the nearest preceding section heading as fixed context on
both sides, recorded in `meta.context`; `--no-heading-context` captures bare
bodies instead, and `--whole` has no single heading to carry.

## Worst → best: what happens to prior pairs

A real rewrite often spans multiple sittings, sometimes with `--keep-open` to
bank interim progress. The corpus wants different things from the two cases:
among **same-session interim saves** the latest should win, earlier ones being
scaffolding; **cross-session saves** (you closed a session, did unrelated work,
opened a fresh one later) are both legitimate training data and both survive.

`save` distinguishes these automatically by keying on each pair's
`meta.snapped_at` — the timestamp of the session that produced it, recorded when
`open` ran. On every save:

- prior pairs whose `meta.snapped_at` **matches** the current session's snapshot
  timestamp are **superseded** (same-session interim);
- prior pairs whose `meta.snapped_at` **differs** are **kept** (cross-session).

Superseded never means deleted: the new save's draws name the `capture_id`s
they replace in `supersedes`, every draw stays in `draws/`, and the view leaves
the superseded ones out (D84).

You never need a flag in the common cases — open, edit, save, and add
`--keep-open` for banking. Explicit overrides for the rare ones:

| flag | effect |
| --- | --- |
| `--replace` | supersede every prior pair for this source, including cross-session ones. For genuinely abandoning earlier sessions. |
| `--append` | supersede nothing, so this session's interim pairs stay in the view too. Only if you want multiple interim states preserved (very rare). |
| `--keep-open` | orthogonal to the above: leaves the snapshot in place after saving, for further `save` calls against the same starting state. |
| `--redo` | reopen the last closed session for this file, then save. The after-the-fact do-over. |

`--tag` labels saves by provenance (`from-claude-chat`, `dan-voice-draft`,
`heavy-edit`) for downstream filtering.

### Worked examples

Interim-progress loop — no flags beyond `--keep-open`:

```sh
open foo.qmd                 # session 1, T0 recorded
save --keep-open foo.qmd     # interim pairs from session 1
# ... more editing ...
save --keep-open foo.qmd     # supersedes session 1's interim pairs
# ... finished ...
save foo.qmd                 # final; supersedes interim, closes session 1
```

Multiple de-slop sessions on the same file, separated by unrelated substantive
edits — no flag needed, the smart default keeps both:

```sh
open foo.qmd                 # session 1
# ... de-slop, T0 -> T1 ...
save foo.qmd                 # session 1 pairs written, session closed
# ... unrelated edits OUTSIDE any session, T1 -> T2 ...
open foo.qmd                 # session 2 (different snapped_at)
# ... de-slop, T2 -> T3 ...
save foo.qmd                 # session 2 pairs ADDED; session 1 pairs kept
                             # "kept 1 pair(s) from 1 earlier session(s)"
```

**Close session 1 before doing unrelated edits.** Had you used
`save --keep-open` between the two, session 2's diff would run against T0,
folding the structural edits into the pair. The clean close is what keeps
non-style work out of the training data.

## Fixing a capture after the fact

Two repairs, depending on how far back the mistake is.

**The last save of a file: `save --redo`.** Closing a session *archives* its
snapshot to `snapshots-closed/`, so the base text survives and the pair can be
recaptured against it:

```sh
open foo.qmd                 # session 1, T0 recorded
save foo.qmd                 # pairs written, session closed
# ... fix the typo, T1 -> T1' ...
save --redo foo.qmd          # reopens session 1, re-diffs T0 -> T1'
                             # same snapped_at, so this SUPERSEDES the first save
```

`reopen foo.qmd` is that same restore as a separate step, for looking before
you save, and the recovery path for an accidental `drop`. The preserved
`snapped_at` is what lands the re-save in the same session, making it a
correction rather than a near-duplicate. One snapshot archive per source,
overwritten on each close; no deeper undo history.

**Any older save: edit its capture archive, then `log recut`.** Nothing
rewrites a draw, but a real capture's `captures/<id>/{before.md,after.md}` is
the editable truth of that pair (D96) — ordinary files, so `diff` works on
them. Fix a typo, move a `sem` marker, or revert a paragraph to its before text
to retract that chunk, then re-land it:

```sh
uv run ai-style log recut <archive-id>                                # a `save` landing (gap 0/0)
uv run ai-style log recut <archive-id> --gap-paras 2 --gap-chars 800  # cut at a wider gap
```

The fresh draws land under a new `capture_id` whose `meta.archive_id` names the
archive, and they supersede every earlier landing of it; the archive's `whole`,
`source` and `snapped_at` are kept, and nothing under `captures/` is written.
`ai-style status --draws …` lists the archives whose live records no longer
match them ("edited since landing"), which is the reminder that a `recut` is
due. A capture wrong beyond repair is retired by an excluding clause in the
`ViewPolicy` plus a rebuild, not by deletion.

## Recommended editing cadence

The natural unit of capture is "this file, as a rewrite project" — not "this
editing sitting". A typical cycle:

1. `open` once at the start of the project (typically when the file is
   `automation: 2`). The session is sticky, so this marks "begin work".
2. Edit at your own pace. One sitting or one week — it does not matter.
3. `save` when you have reached a satisfying state. The default emits
   per-region pairs, supersedes any interim pairs for this file, and closes the
   session. `save --keep-open` banks progress without closing.
4. Skip `open` entirely for files written from scratch (no slop input to learn
   from), for tiny mechanical fixes (no transformation signal), and per
   paragraph (`save` already isolates changed paragraphs). Three or four open
   sessions at once is plenty — `list` shows everything pending.

## One-shot entry: `pair`

For ad-hoc pairs (clipboard paste, ephemeral content, scripted pipelines) use
`pair` with explicit before/after, skipping the session dance:

```sh
uv run ai-style log pair --before before.txt --after after.txt
pbpaste | uv run ai-style log pair --before - --after rewritten.txt \
    --source clipboard --tag hand-typed          # '-' is stdin
```

`pair` chunks by default (the same paragraph diff as `save`); pass `--whole` to
keep the input as a single pair. It defaults to **append**, not supersede, since
the typical use is logging one-off pairs that add to the corpus rather than
banking-with-supersede. Pass `--replace` (which requires `--source`) to
supersede prior pairs for that source.

`log mine` is the third entry point: it replays Claude Code sessions and pairs
what the assistant left with what you changed afterwards, through the same
differ, writing the same archives and draws (D89). Dry-run until `--apply`.

## Record shape

A view record — what `views/pairs-<sha>/` holds, and what a draw carries whole
as its `record`, beside its provenance and `outcome`:

```json
{"messages": [{"role": "system", "content": "<STYLE_SYSTEM>"},
              {"role": "user", "content": "<slop chunk>"},
              {"role": "assistant", "content": "<Dan-voice chunk>"}],
 "meta": {"schema_version": 4,
          "source": "post/foo.qmd",
          "captured_at": "...", "snapped_at": "...",
          "capture_id": "<short id>",
          "chunk_index": 2, "chunk_total": 5,
          "before_chars": 0, "after_chars": 0,
          "context": "## the nearest heading", "context_mode": "immediate",
          "before_frontmatter": {}, "after_frontmatter": {},
          "provenance": {}, "tags": []}}
```

`capture_id` groups the chunks of a single landing; the frontmatter keys appear
only on chunk 0; a recut record adds `archive_id`. `before_chars` and
`after_chars` count the body, not the context prepended to both sides.

### What survives a rename

`meta.source` is the relative path at save time. A rename does not invalidate
the pair — the text is what trains — but `save`'s disposition keys on the source
string to identify "the same file", so pairs under the old path are not
superseded automatically. Retire them with `save --replace` under the old path,
or an excluding clause in the `ViewPolicy`.

## Backup policy

The canonical corpus is versioned in the (livingthing) blog repo, so git
history is the primary backstop. Remaining practical care: treat `draws/` and
`captures/` as precious — the draws are the only record of what happened, and
an archive is human editing work no money regenerates — while `views/` is
disposable, rebuilt on demand. Commit (or copy) the corpus before any bulk
operation, and keep the directory inside whatever machine backup you already
run. If it can't live in a git repo (too big, too private), an object-storage
bucket is the natural alternative — `STYLEBOT_DATA_DIR` makes that a one-line
change with no code edits.
