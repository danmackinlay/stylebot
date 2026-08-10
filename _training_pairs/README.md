# `_training_pairs/` — the corpus

This directory holds the **training corpus**: the captured and synthesised
`(slop → Dan)` pairs that the fine-tune is trained on. It is the project's
valuable, private asset.

**Everything in here except this README is gitignored.** The stylebot code
repo is public tooling and never carries the data. Do not `git add -f` it.
The *canonical* corpus lives in the livingthing repo, where it **is**
committed (`livingthing/_training_pairs/pairs.jsonl`, since 2026-07, for
portability) — the never-commit rule is about this repo, not the data.

This file is the user-facing manual for `ai-style log`, the Phase 1 capture
tool. The schema contract is in
[`../_plans/OVERVIEW.md`](../_plans/OVERVIEW.md).

## Layout

```
_training_pairs/
├── README.md                     # tracked (this file)
├── pairs.jsonl                   # the corpus: one chat-completion record per line
├── snapshots/<rel-path>/         # open sessions: content.md + meta.json
├── snapshots-closed/<rel-path>/  # closed sessions, retained so `reopen` works
└── captures/<capture-id>/        # per-save archive: before.md, after.md, meta.json
```

The directory for open sessions is called `snapshots/` because the file *is* a
snapshot of the source's initial state — that names its role at the storage
layer. The CLI verb is `open` because that names your *action*.

## Where the data lives (`STYLEBOT_DATA_DIR`)

The corpus location resolves through the shared ladder
`--data-dir` > `$STYLEBOT_DATA_DIR` > the cwd-relative default
`./_training_pairs`. So:

- **Capturing pairs** with `ai-style log` happens inside the prose working
  tree (the blog repo). With no env var set, pairs land in
  `<blog>/_training_pairs/pairs.jsonl`.
- **Downstream phases** (synthesis, training, eval) run from *this* repo and
  should point at the same corpus:

  ```sh
  export STYLEBOT_DATA_DIR=/path/to/your/corpus
  ```

Keeping one canonical corpus path (rather than copies scattered per repo)
avoids the classic "which pairs.jsonl is the real one" failure.

`log` deliberately takes no `--data-dir` flag: it is the convenience command,
where a mistake is cheap. The expensive, stateful commands (`train`, `split`)
demand an explicit `--data-dir` instead. Every `log` command that writes echoes
the resolved **absolute** pairs.jsonl path, so a save landing somewhere
unexpected is visible immediately.

## The session cycle

Every rewrite of a file is a **session**: opened explicitly, saved when
finished, optionally banked mid-stream and superseded later.

```sh
uv run ai-style log open <file>    # begin a rewrite session
# ... you edit the file by hand, across one or more sittings ...
uv run ai-style log save <file>    # write pair(s) for what changed, close session
```

The session is **sticky** — `open` refuses to overwrite an existing one, so the
original ("worst") state of the file survives across editing sittings. Open once
at the start of a rewrite project and save when you are happy; you do not have
to finish in one sitting.

Every command except `list` ends by printing a short reminder of
currently-open sessions to stderr, with their relative age, so stale sessions
become visible. Pass `-q` / `--quiet` at the group level to suppress it for one
invocation:

```sh
uv run ai-style log -q save <file>
```

`drop <file>` abandons an open session without writing pairs; `list` shows open
sessions, saved pairs grouped by source, and totals.

## Chunk-level save (the default)

`save` does **not** emit a single whole-file pair. It diffs the session's
recorded "before" state against the current file at paragraph granularity and
emits one pair *per changed region*, skipping untouched paragraphs entirely.

Why: a real rewrite typically touches a few paragraphs out of many. A whole-file
pair would force the model to learn "preserve everything except these spans,
which get rewritten" — not the lesson we want. Per-chunk pairs isolate the
actual transformation, and they match the chunk size Phase 2 uses for synthetic
pairs, so training data from both sources is shape-compatible.

The diff is paragraph-level (split on blank lines) plus `difflib`
`SequenceMatcher` opcodes. **Contiguous changed paragraphs coalesce into one
pair**, regardless of N → M paragraph counts on either side — the `replace`
opcode pairs spans, not paragraphs. A pair only breaks when an *untouched*
paragraph sits between two changed regions.

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
rearrangement, paragraphs moved around, or sentence-level edits scattered
across every paragraph (where the chunk algorithm would emit one giant
`replace` covering most of the file anyway).

`recut` re-cuts existing pairs from the `captures/` archives at a different gap
policy, without recapturing anything.

## Worst → best: what happens to prior pairs

A real rewrite often spans multiple sittings, sometimes days, sometimes with
`--keep-open` to bank interim progress. The corpus wants different things from
the two cases:

- **Same-session interim saves** (typically from `--keep-open`): the *latest*
  interim save should win; earlier ones are scaffolding and should not pollute
  training data.
- **Cross-session saves** (you finished a session, did unrelated work, later
  opened a fresh session): both sets are legitimate training data and should
  both survive.

`save` distinguishes these automatically by keying on each pair's
`meta.snapped_at` — the timestamp of the session that produced it, recorded when
`open` ran. On every save:

- prior pairs whose `meta.snapped_at` **matches** the current session's snapshot
  timestamp are **replaced** (same-session interim);
- prior pairs whose `meta.snapped_at` **differs** are **kept** (cross-session).

You never need a flag in the common cases. Open, edit, save; add `--keep-open`
for banking. The tool works out the disposition.

Explicit overrides for the rare cases:

| flag | effect |
| --- | --- |
| `--replace` | force-drop every prior pair for this source, including cross-session ones. For genuinely abandoning earlier sessions. |
| `--append` | force-keep every prior pair for this source, including interim pairs from the current session. Only if you want multiple interim states preserved (very rare). |
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
save --keep-open foo.qmd     # replaces session 1's interim pairs
# ... finished ...
save foo.qmd                 # final; replaces interim, closes session 1
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
`save --keep-open` between the two, session 2's diff would run against T0 —
folding the structural edits into the pair. The clean close is what keeps
non-style work out of the training data.

The do-over: you saved, then noticed a typo in your own rewrite. Closing a
session *archives* its snapshot to `snapshots-closed/` rather than deleting it,
so the base text survives and the pair can be recaptured against it:

```sh
open foo.qmd                 # session 1, T0 recorded
save foo.qmd                 # pairs written, session closed
# ... fix the typo, T1 -> T1' ...
save --redo foo.qmd          # reopens session 1, re-diffs T0 -> T1'
                             # same snapped_at, so this REPLACES the first save
```

`reopen foo.qmd` is that same restore as a separate step, for when you want to
look before you save. Because the archive preserves the original `snapped_at`,
the re-save lands in the same session — which is what makes it a correction
rather than a near-duplicate in the corpus. The archive is also the recovery
path for an accidental `drop`. There is one archive per source, overwritten
each close; no deeper undo history.

`tidy [source]` is the retroactive cleanup if you used `--append` recklessly and
want only the latest save per source.

## Recommended editing cadence

The natural unit of capture is "this file, as a rewrite project" — not "this
editing sitting". A typical cycle:

1. `open` once at the start of the rewrite project (typically when the file is
   `automation: 2`). The session is sticky; this marks "begin work", not "start
   a sitting".
2. Edit at your own pace. One sitting or one week — it does not matter.
3. `save` when you have reached a satisfying state. The default emits
   per-region pairs, supersedes any interim pairs for this file, and closes the
   session.
4. To bank interim progress without closing, `save --keep-open`. Later saves
   replace the interim pairs by default.
5. Skip `open` entirely for files written from scratch (no slop input to learn
   from), for tiny mechanical fixes (no transformation signal), and per
   paragraph (`save` already isolates changed paragraphs).
6. Cap simultaneous open sessions at three or four to keep the bookkeeping
   manageable — `list` shows everything pending.

## One-shot entry: `pair`

For ad-hoc pairs (clipboard paste, ephemeral content, scripted pipelines) use
`pair` with explicit before/after, skipping the session dance:

```sh
uv run ai-style log pair --before before.txt --after after.txt
uv run ai-style log pair --before - --after after.txt          # stdin
pbpaste | uv run ai-style log pair --before - --after rewritten.txt \
    --source clipboard --tag hand-typed
```

`pair` chunks by default (the same paragraph diff as `save`); pass `--whole` to
keep the input as a single pair. It defaults to **append**, not replace, since
the typical use is logging one-off pairs that add to the corpus rather than
banking-with-supersede. Pass `--replace` to remove prior pairs for a given
`--source` first.

## Record shape

Pairs are written in Together / OpenAI chat-completion JSONL:

```json
{"messages": [{"role": "system", "content": "<STYLE_SYSTEM>"},
              {"role": "user", "content": "<slop chunk>"},
              {"role": "assistant", "content": "<Dan-voice chunk>"}],
 "meta": {"source": "post/foo.qmd",
          "captured_at": "...", "snapped_at": "...",
          "capture_id": "<short id>",
          "chunk_index": 2, "chunk_total": 5,
          "before_chars": 0, "after_chars": 0,
          "before_frontmatter": {},
          "after_frontmatter": {},
          "tags": []}}
```

`capture_id` groups the chunks from a single save; the frontmatter keys appear
only on chunk 0.

Rewrites of `pairs.jsonl` (the default `save`, `pair --replace`, `tidy`) go
through an atomic `.jsonl.tmp` + rename, so an interrupted run cannot corrupt
the audit trail. Unparseable lines are preserved as opaque `_raw` records across
rewrites.

### What survives a rename

`meta.source` is the relative path at save time. Renaming the source file later
does not invalidate the pair — the text content is what matters for training —
but `save`'s disposition logic and `tidy` both use the source string to identify
"the same file". So if you rename a file and then re-save, the old pairs will
not be superseded automatically. Use `tidy <old-path>` to clean them up, or
hand-edit the JSONL.

## Backup policy

The canonical corpus is versioned in the (livingthing) blog repo as of
2026-07, so git history is the primary backstop. Remaining practical care:

- Treat `pairs.jsonl` as precious: a careless `ai-style log save --replace`
  (or `--append` then `tidy`) can drop pairs irrecoverably.
- Snapshot it before any bulk/destructive operation. A dated copy is enough:

  ```sh
  cp "$STYLEBOT_DATA_DIR/pairs.jsonl" \
     "$STYLEBOT_DATA_DIR/backups/pairs.$(date +%Y%m%d-%H%M%S).jsonl"
  ```

  (`backups/` is gitignored along with everything else here.)
- Make sure the corpus directory is inside whatever your machine backup
  already covers (Time Machine, Backblaze, rsync target, …).

If your corpus can't live in a git repo (too big, too private), the natural
alternative is an object-storage bucket — `STYLEBOT_DATA_DIR` makes that a
one-line change with no code edits.
