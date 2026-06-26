# Phase 1 · Pair capture — ✅ SHIPPED

Status: shipped and in daily use. This file documents the **as-built reality**
and the canonical `pairs.jsonl` schema that every other phase depends on.

## What it is

`ai-style-log`: a manual logger. Every time Dan rewrites an AI draft into his
own voice, it captures the `(before, after)` diff at paragraph granularity into
`pairs.jsonl`. No automation, no git hook, no surprise side effects — invoked
by hand. Full behaviour is in the module docstring of
`src/stylebot/bin/ai_style_log.py`; that docstring is authoritative for
mechanics, this file for the contract.

## Inputs

- Prose files (`.qmd`/markdown) being hand-rewritten, in the prose working tree.
- Run as `uv run ai-style-log <cmd>` (or the installed `ai-style-log`).

## Outputs (THE shared contract)

`$STYLEBOT_DATA_DIR/pairs.jsonl` — one JSON record per line:

```json
{
  "messages": [
    {"role": "system", "content": "<STYLE_SYSTEM, verbatim>"},
    {"role": "user", "content": "<slop chunk>"},
    {"role": "assistant", "content": "<Dan-voice chunk>"}
  ],
  "meta": {
    "source": "post/foo.qmd",
    "captured_at": "ISO-8601",
    "snapped_at": "ISO-8601",
    "capture_id": "<8-hex>",
    "chunk_index": 2,
    "chunk_total": 5,
    "before_chars": 0,
    "after_chars": 0,
    "before_frontmatter": {},
    "after_frontmatter": {},
    "tags": []
  }
}
```

Contract rules that downstream phases (esp. Phase 2 and 3) MUST honour:

- **`messages[0]` is `STYLE_SYSTEM`** from `src/stylebot/ai_core.py`, verbatim.
  Changing that string after pairs are logged invalidates them. Phase 2 must
  emit the same system string.
- **Chunking** is paragraph-level (blank-line split) + `difflib`
  SequenceMatcher; one pair per changed region. Phase 2 synthetic pairs should
  use the same chunk shape so the two sources are mixable.
- `meta.capture_id` groups chunks from one save; `meta.source` identifies "the
  same file" for replace/tidy.
- `pair --before X --after Y` is the one-shot entry (no session); use for
  scripted/clipboard pairs.

## Done-criteria (met)

- [x] `open`/`save`/`drop`/`list`/`tidy`/`pair` subcommands.
- [x] Smart cross-session vs interim-save disposition.
- [x] Atomic rewrite of `pairs.jsonl`.
- [x] VS Code task + keybinding wiring (docs appendix).

## Follow-ups / debt

- `STYLEBOT_DATA_DIR` override added in Phase 0; default unchanged so daily use
  is unaffected.
- Possible future `meta.weight` field if Phase 3 wants to down-weight synthetic
  pairs relative to these real ones (currently absent — see OVERVIEW open Qs).
- No automated test of the chunk-diff coalescing rules yet → see
  `tests/` (Phase 0 added a smoke test only).
