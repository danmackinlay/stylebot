# `_training_pairs/` — the corpus

This directory holds the **training corpus**: the captured and synthesised
`(slop → Dan)` pairs that the fine-tune is trained on. It is the project's
valuable, private asset.

**Everything in here except this README is gitignored.** The stylebot code
repo is public; the corpus is not. Do not `git add -f` the data.

## Layout

```
_training_pairs/
├── README.md            # tracked (this file)
├── pairs.jsonl          # the corpus: one chat-completion record per line  [gitignored]
└── snapshots/           # open ai-style-log sessions (transient)            [gitignored]
    └── <rel-path>.json
```

`pairs.jsonl` records are Together/OpenAI chat-completion JSONL — see the
schema contract in [`../_plans/phase-1-pair-capture.md`](../_plans/phase-1-pair-capture.md)
and the module docstring in `src/stylebot/bin/ai_style_log.py`.

## Where the data lives (`STYLEBOT_DATA_DIR`)

The corpus location is `$STYLEBOT_DATA_DIR`, defaulting to `./_training_pairs`
relative to the current working directory. So:

- **Capturing pairs** with `ai-style-log` happens inside the prose working
  tree (the blog repo). With no env var set, pairs land in
  `<blog>/_training_pairs/pairs.jsonl`.
- **Downstream phases** (synthesis, training, eval) run from *this* repo and
  should point at the same corpus:

  ```sh
  export STYLEBOT_DATA_DIR=/path/to/your/corpus
  ```

Keeping one canonical corpus path (rather than copies scattered per repo)
avoids the classic "which pairs.jsonl is the real one" failure.

## Backup policy (manual)

The corpus is **not** versioned in any git repo. Its only recovery backstop is
your own backup. Practical minimums:

- Treat `pairs.jsonl` as precious: a careless `ai-style-log save --replace`
  (or `--append` then `tidy`) can drop pairs irrecoverably.
- Snapshot it before any bulk/destructive operation. A dated copy is enough:

  ```sh
  cp "$STYLEBOT_DATA_DIR/pairs.jsonl" \
     "$STYLEBOT_DATA_DIR/backups/pairs.$(date +%Y%m%d-%H%M%S).jsonl"
  ```

  (`backups/` is gitignored along with everything else here.)
- Make sure the corpus directory is inside whatever your machine backup
  already covers (Time Machine, Backblaze, rsync target, …).

If you later outgrow manual backup, the natural upgrades are a private git
repo for the corpus or an object-storage bucket — `STYLEBOT_DATA_DIR` makes
either a one-line change with no code edits.
