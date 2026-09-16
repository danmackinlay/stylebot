# stylebot

Tooling to fine-tune a small open-weights LLM that rewrites AI-flavoured draft
prose into [Dan Mackinlay's](https://danmackinlay.name) voice — a specialist
*prose styler*, not a chatbot.

The motivation, design, and economics are written up in the blog post
[*Fine-tuning danbot*](https://danmackinlay.name/notebook/fine_tuning_danbot.html). The development
plan lives in [`_plans/`](_plans/OVERVIEW.md) — including
[`decisions.md`](_plans/decisions.md) (why each choice landed, and what was
rejected) and [`findings.md`](_plans/findings.md) (the measured results).

## What's here

This repo is the **tooling** for the project. The **corpus** (training pairs)
and **secrets** (API keys) deliberately live outside the repo so it can be
public — see [Data & secrets](#data--secrets) below.

| Phase | Status | Component |
|-------|--------|-----------|
| 1 | ✅ shipped | `ai-style log` — manual (slop → Dan) pair capture |
| 2 | 🔧 built | `ai-style synth` — synthetic-pair generation (OpenRouter multi-model) |
| 3 | 🔧 built | `ai-style train` — LoRA SFT (Tinker cookbook recipe; committed run manifests) |
| 4 | 🔧 built | `ai-style run` — inference CLI (chunked, structure-protected, best-of-N with do-no-harm + anchor guards) |
| E | 🔧 built | `ai-style eval` — multi-signal eval (Vale + LLM-judge + trained voice classifier) |
| — | 🔧 built | `ai-style serve` + [VS Code voice marker](#vs-code-voice-marker) — live P(slop) paragraph marks |
| — | 🔧 built | `ai-style status` — corpus, splits health, promoted run, detector sha, and the cross-checks between them |

## What it does

One worked example, from the **first** trained styler (2026-07; Qwen3.5-9B +
LoRA, one epoch, ~$2 of compute) — kept because it is a clean illustration, not
because it is the current adapter. What is serving today, and on what corpus,
is what `ai-style status` prints; the dated numbers are in
[`_plans/findings.md`](_plans/findings.md). The styler is shown
de-slopping a machine-inflated paragraph of — fittingly — Dan's AI-usage
policy. A held-out validation pair; the styler never saw this post in
training:

> **Input (machine slop):** I welcome disagreement regarding my definition of
> "value" or where I draw its boundaries. Similarly, please feel free to point
> out any errors in my application of AI or suggest more effective
> methodologies. […] I also believe it is worthwhile to discuss the instances
> where I choose *not* to use AI, and the reasoning behind those decisions.

> **Styler output:** I welcome disagreement about my definition of "value" or
> where I draw the boundaries. Please feel free to point out errors in my use
> of AI or suggest better methods. […] I also think it is worth discussing the
> instances where I choose *not* to use AI and why.

> **Dan's original, for reference:** You are welcome to disagree with me about
> what "value" means and where the line is. […] I think it might be worthwhile
> mentioning where I _don't_ use AI, and why.

Measured on the voice classifier of the day, the rewrite moves `P(slop)` from
0.56 to 0.27 against the original's 0.26; the LLM judge moves 2 → 3 (Dan's
original: 4). Across 40 held-out pairs that styler closed about half the
slop→Dan gap on every signal, where the same base model with the same prompt
but no LoRA makes prose *more* AI-flavoured than the input. Later runs, a
larger corpus and a frozen eval stratum are all in `findings.md` — including
the 2026-09 re-analysis that took a later run's headline gain back to
"indistinguishable" once the interval was clustered by post.

## Quickstart

```sh
uv sync
uv run ai-style log --help
uv run ai-style --help
uv run ai-style status --draws _training_pairs/draws  # what the corpus holds
```

The pair-logger is run from inside the prose working tree (e.g. the blog repo),
where a save archives the full before/after under `captures/` and appends one
*draw* per changed region to `draws/captures.jsonl`, both under
`$STYLEBOT_DATA_DIR`. Nothing rewrites a draw: the training corpus is a *view*
of the draws, built on first use under a policy (`ai-style train --draws`,
`status --draws`; D84). A legacy `pairs.jsonl` or `pairs/` shard dir still reads
as accepted draws.

## Specializing stylebot for your own blog

stylebot is split into **mechanism** (this repo: prose extraction, chunk
hygiene, pair synthesis, eval scoring, classifier training — everything that
operates on stylebot's own contracts) and **policy** (your repo: which posts
count as your voice, which chunks are worth keeping, which models generate the
slop). The `ai-style` commands run the mechanism with bare defaults; the
intended way to adopt stylebot is a thin policy layer in *your* blog repo —
[Dan's blog](https://github.com/danmackinlay/danmackinlay.github.io) is the
worked example, and its whole synthesis layer is two small files:

1. **A policy module** (~100 lines; example `livingthing/training_targets.py`):
   your constants (quality thresholds, stop-headers, stub markers, chunking
   defaults, slop-model rotation) and a composite `selector(meta: dict) -> bool`
   over your frontmatter conventions. Every stylebot phase accepts `selector=`;
   the predicate is never hardcoded in the mechanism.
2. **A CLI mirror** (~90 lines; example `livingthing/bin/train_targets.py`),
   composed from `stylebot.synth_cli`:
   `@synth_options(exclude=…, **your_defaults)` re-declares nothing — it applies
   the shared option surface with your defaults swapped in and your pinned
   options removed — and the command body is just
   `iter_targets(selector=…)` → `run_synth(…)`. Inspection modes, generator
   wiring, progress and resume all come from stylebot, so the wrapper can't
   drift. `run_synth` raises `SynthConfigError` / `SynthRunError` /
   `SynthGenerationError` rather than exiting — mirror `ai-style`'s
   `_synth_errors` boundary to map them onto exit 2 / exit 1.

**Naming convention:** mirror the `ai-style` subcommands under your own prefix.
On Dan's blog, `dan-style synth|train-clf|eval` are the policy versions of
`ai-style synth|train-clf|eval` — same subcommand, swap the prefix. That one
rule is the entire translation table, and it tells you where behaviour comes
from: if `dan-style synth` produces ~1.5k-char merged passages where
`ai-style synth` produces raw fragments, the difference is (visible, greppable)
policy, not mechanism.

The **voice classifier** usually needs no wrapper code at all:
`ai-style train-clf --pairs … --out …` (requires the `stylebot[classifier]`
extra) trains the linear head from your captured pairs; policy enters as an
optional injected `extra_positives=` iterable (library-level) and the choice of
embedding backbone (`--embed-model`). The default backbone, StyleDistance, won
a bake-off *on Dan's corpus* — re-run that comparison for your own author
before trusting it.

## VS Code voice marker

[`vscode-voice-marker/`](vscode-voice-marker/) is a small editor extension
that marks each prose paragraph of a markdown/Quarto document with a graded
**P(slop)** — a faint background tint plus scrollbar hotspots, the number on
hover — so suspect paragraphs get noticed, rewritten, and captured as training
pairs by the existing `ai-style log` tasks. It is the front-end of the capture
loop, not a verdict machine: the classifier is a soft signal (held-out AUC
~0.7 on real pairs, and it moves with every refit — `ai-style status` prints
the artifact's own honest number), so the marks are framed as *suspicion* and
your eyeball stays the arbiter. Design and caveats: [`_plans/retired/vscode-marker.md`](_plans/retired/vscode-marker.md).

### How it works

The extension spawns **`ai-style serve --detector-model DIR`** as a child
process — a long-lived sidecar (`stylebot.serve`) that loads a trained
voice-classifier artifact (`head.json` + `meta.json`, produced by
`ai-style train-clf` or a blog policy mirror like `dan-style train-clf`)
once (~5 s for the embedding model) and then scores batches
of texts over stdin/stdout NDJSON in tens of milliseconds per document. The
extension owns the process lifecycle: spawn on activation, status-bar spinner
until the `info` handshake, capped-backoff restart on crash, kill on window
close. You never launch the sidecar by hand.

The sidecar can arm more than one instrument — `--detector voice-clf
--detector sapling` adds [Sapling](https://sapling.ai/docs/api/detector/)'s
hosted human-vs-AI detector (`$SAPLING_API_KEY`; spends per character, so the
extension's *Switch Detector* command picks which one marks, its `trigger`
setting can wait for a save, and unchanged paragraphs are answered from an
in-process cache). The two are never blended: only the voice head reranks
*Rewrite in Dan's Voice* (D106).

The extension does **not** segment. It asks the sidecar: `{"op": "segment",
"text": <document>}` returns the scoreable prose spans, computed by
`stylebot.serve.prose_paragraphs` over `stylebot.markdown` — paragraphs are runs
of prose lines, so code fences (both ``` and `~~~`, unclosed ones running to end
of document), display `$$math$$`, `:::` divs, blockquotes, frontmatter and
heading lines all end one, and the classifier sees the same prose shape it was
trained on. **Which div classes bear prose is the sidecar's policy**
(`dan-style serve` hands it the blog's `div_is_prose`), which is precisely what
a client-side port could not know: the TypeScript port that used to live here
never scored a callout paragraph, and never offered to rewrite one. It was
deleted 2026-09-03 (`_plans/decisions.md` D44) after measuring its boundaries
against this one over every blog `.qmd` — each difference a correction
(`_plans/findings.md`, "Segmenter implementations"). The remaining
cross-implementation fixture, pinning
`livingthing.qmd_core`, is `tests/fixtures/segmenter_conformance.*`.

### Install & configure

```sh
cd vscode-voice-marker
npm install && npm test && npm run package
code --install-extension voice-marker-*.vsix   # the version npm run package just built
```

The extension is **dormant** until `voiceMarker.command` is set, so configure
it per workspace (the repo that holds your trained artifact) in
`.vscode/settings.json`:

```jsonc
{
  // run from the workspace folder (the extension's default cwd)
  "voiceMarker.command": [
    "uv", "run", "ai-style", "serve", "--detector-model", "_models/voice-clf"
  ],
  // ascending P(slop) bucket edges — calibrate to YOUR score distribution
  "voiceMarker.thresholds": [0.51, 0.55, 0.59, 0.65]
}
```

Other settings: `voiceMarker.minChars` (skip short/noisy paragraphs, default
80), `.debounceMs` (default 500), `.enabled`. Commands: *Rescore Document*,
*Toggle Marks*, *Restart Sidecar*.

**Thresholds are calibrated quantiles, not equal-width bins.** Raw detector
scores cluster in a narrow band (≈[0.32, 0.68] on Dan's corpus), so naive
0.2/0.4/0.6/0.8 edges would leave every bucket empty or full. `ai-style calibrate`
cuts them from quantiles of the two sides' detector scores — human-p75 (floor
of the first mark), slop-p50, slop-p75, slop-p95 by default — on the real
pairs the head never trained on (D105). The quantiles the current edges
were cut from: [`_plans/findings.md`](_plans/findings.md) "The voice classifier".

### Updating after retraining the classifier

The artifact is data, not code — an improved classifier almost never touches
the extension:

1. **Retrain the head** (in the repo that owns the corpus and artifact, e.g.
   the blog: `uv run dan-style train-clf`, or generically
   `uv run ai-style train-clf --pairs … --out _models/voice-clf`) and
   commit the refreshed `head.json` + `meta.json`.
2. **Restart the sidecar** — the *Voice Marker: Restart Sidecar* command (or
   reload the window). The sidecar rebuilds the detector from `meta.json` at
   spawn, so even a backbone swap needs nothing more: `meta.json` pins
   `embed_model`, and the dim-mismatch guard in `stylebot.classify` fails
   loudly if head and embedder disagree.
3. **Recalibrate `voiceMarker.thresholds`** — the score distribution moves
   with every retrain. `ai-style calibrate --detector-model DIR --pairs …
   --splits …` scores the eval role's real pairs (the ones the head never
   saw) with the fresh artifact and prints the edges under the quantile
   rule (`--rule`, default Dan-p75, slop p50/p75/p95); paste them in, keyed
   by the instrument's name when the sidecar arms several
   (`{"voice-clf": [...], "sapling-20251027": [...]}` — `--detector sapling`
   calibrates that one). Threshold changes apply live, no restart.

Rebuilding/reinstalling the `.vsix` is needed only when the **wire protocol**
(`stylebot.serve`) changes — a segmentation change is now server-side and needs
no repackaging at all. Confirm `npm test` passes, then `npm run package` and
reinstall.

## Data & secrets

- **Corpus** — draws and open editing sessions are written under
  `STYLEBOT_DATA_DIR` (default `./_training_pairs`): `captures/` (the real-edit
  archive, append-only), `draws/` (one record per generator call or human save,
  accepted or not, append-only) and `views/` (built from the draws under a
  policy; disposable, never committed). A `pairs.jsonl` / `pairs/` shard dir is
  read as accepted draws by an install that predates draws (D84). That
  directory is **gitignored here**: the corpus never enters
  this public repo. Its canonical home is the (livingthing) blog repo, where it
  *is* committed (since 2026-07, for portability). See
  [`_training_pairs/README.md`](_training_pairs/README.md).
- **API keys** — read from the process environment (`OPENROUTER_API_KEY`,
  `TINKER_API_KEY`, …). Export them, or let [direnv](https://direnv.net) do it:
  `direnv exec <dir-with-an-.envrc> uv run ai-style …`. stylebot loads no
  dotenv file of its own. Whatever file holds your keys stays gitignored.

## Acknowledgements

stylebot's development is supported by [ACS Research](https://acsresearch.org).
