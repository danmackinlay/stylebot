# stylebot

Tooling to fine-tune a small open-weights LLM that rewrites AI-flavoured draft
prose into [Dan Mackinlay's](https://danmackinlay.name) voice — a specialist
*prose styler*, not a chatbot.

The motivation, design, and economics are written up in
[`docs/fine_tuning_danbot.qmd`](docs/fine_tuning_danbot.qmd). The development
plan and phase contracts live in [`_plans/`](_plans/OVERVIEW.md).

## What's here

This repo is the **tooling** for the project. The **corpus** (training pairs)
and **secrets** (API keys) deliberately live outside the repo so it can be
public — see [Data & secrets](#data--secrets) below.

| Phase | Status | Component |
|-------|--------|-----------|
| 1 | ✅ shipped | `ai-style-log` — manual (slop → Dan) pair capture |
| 2 | planned | synthetic-pair generation |
| 3 | planned | LoRA SFT (Together / Fireworks) |
| 4 | planned | `ai-style` inference CLI |
| — | planned | multi-signal eval harness (Vale + LLM-judge + detector) |

## Quickstart

```sh
uv sync
uv run ai-style-log --help
```

The pair-logger is run from inside the prose working tree (e.g. the blog repo),
where it writes captured pairs to `$STYLEBOT_DATA_DIR/pairs.jsonl`.

## Data & secrets

- **Corpus** — `pairs.jsonl` and open editing sessions are written under
  `STYLEBOT_DATA_DIR` (default `./_training_pairs`). This directory is
  **gitignored**; the corpus is private and backed up out-of-band. See
  [`_training_pairs/README.md`](_training_pairs/README.md).
- **API keys** — copy `.env.example` to `.env` and fill in. `.env` is
  gitignored and never committed.
