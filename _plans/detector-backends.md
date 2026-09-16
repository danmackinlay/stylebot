# Detector backends for VS Code

*Research note · September 2026 — moved here from the shelved `sloggle` repo on 2026-09-17; it backs D106. Two corrections since: Sapling's current default model is `20251027` (`20240606` on request), and its batch form (`texts`, 1–10 per request, identical items billed once) exists.*

Scoring your own paragraphs for AI authorship, in the editor. Three viable backends, one
large caveat, and a build path that keeps the backend swappable.

> **Short answer.** **Sapling is the hosted backend to start with** — cheapest of the
> credible services by an order of magnitude, and the only one whose API is already shaped
> for this job: it returns a per-sentence score array, not just a document verdict. For
> local, run **Fast-DetectGPT** on MLX rather than Binoculars. Build behind an interface so
> you can swap.

---

## 1. Cost

Detection APIs price per word or per character, and the spread between them is close to
20×. At roughly five characters per English word, USD per **million words** scanned:

| Backend        | $/M words |
| -------------- | --------: |
| Local model    |        $0 |
| Sapling        |       $25 |
| Pangram 3      |       $50 |
| Originality.ai |      $100 |
| Pangram 4      |      $500 |

GPTZero does not publish developer API rates — access is bundled into the $45.99/mo
Professional plan or quoted by sales — so it is not listed.

Concretely: re-scanning an 8,000-word paper end to end costs `$0.20` on Sapling, `$0.40` on
Pangram 3, `$4.00` on Pangram 4. With per-paragraph hash caching you only pay for
paragraphs you actually changed, so real usage is a small fraction of that — but the ratios
decide whether you can afford to score on every save.

> ⚠️ **The Pangram cliff.** Pangram's own pricing widget offers two model generations at the
> same dollar figure with different denominators: **Pangram 4 is $0.05 per 100 words;
> Pangram 3 is $0.05 per 1,000 words.** A 10× difference for what looks like one product.
> If you use Pangram, pin the model explicitly and know which one you are billing against.
> Their batch API is 20% off for async submission, which suits a background scan well.

---

## 2. Hosted services, by API shape

For paragraph highlighting, what matters more than headline accuracy is whether the API
returns *spans*. A document-level verdict forces one HTTP call per paragraph, which
multiplies both latency and cost.

### Sapling — **start here**

- **Granularity:** document, **per-sentence** and per-token probabilities, plus an optional
  pre-rendered HTML heatmap.
- **Rate:** $0.005 / 1k chars.
- `POST /api/v1/aidetect`, 200k character limit; `texts` batches 1–10 items per request.
  Pinnable model versions: `20251027` (default), `20240606` (on request).
- Caches identical text for 3 days, so re-scans of untouched documents are free.
- Self-hosting available on request.

### Pangram

- **Granularity:** document verdict plus interpretability spans.
- **Rate:** $0.05 / 100 words (v4) · $0.05 / 1,000 words (v3).
- Best measured accuracy of the four. Free tier is 2,000 words/day; the $65/mo Professional
  plan bundles $200 of API credit, which is the only thing that makes the v4 rate tolerable.

### GPTZero

- **Granularity:** document, **paragraph** and sentence, with `highlight_sentence_for_ai`.
- **Rate:** not published.
- The paragraph-level field is exactly your unit of work, and `confidence_category`
  (high/medium/low) gives you a ready-made abstain signal. States it does not retain API
  inputs. Opaque pricing is the blocker.

### Originality.ai

- **Granularity:** document and sentence.
- **Rate:** $0.01 / 100 words.
- Credits: 1 per 100 words, 2 if you add plagiarism. Highest false-positive rate of the
  three independently tested — 3.8% overall, though only ~1% on scientific prose.

None of them publish an official VS Code extension. Sapling, GPTZero, Originality, Copyleaks
and Pangram all ship browser extensions and LMS integrations only, so you are writing the
client either way.

---

## 3. Running it locally

Two families with very different profiles. Trained encoders are small, fast and permissively
licensed, but inherit their training set's blind spots. Zero-shot statistical methods need a
base LLM but are inspectable and carry no training-set bias.

### ✅ `desklib/ai-text-detector-academic-v1.01` — MIT

*DeBERTa-v3-large · 0.4B params · ~1.7 GB · ~10–50 ms per paragraph*

An encoder fine-tuned specifically on academic text — the closest match to what you write.
Its general-purpose sibling claims the top of the RAID leaderboard. Two catches: the
academic variant publishes **no** accuracy numbers at all, and it uses a custom
`PreTrainedModel` subclass, so `pipeline()` won't load it — you paste the class from the
model card. Run CPU-first; DeBERTa-v3 on PyTorch MPS has a history of unsupported ops.

### ✅ Fast-DetectGPT — MIT

*1 forward pass · 1 model · repo last pushed Feb 2026*

Conditional probability curvature: the analytic replacement for DetectGPT's
hundred-perturbation loop, ~340× faster. The better zero-shot pick over Binoculars — one
model instead of two, so no tokenizer-alignment trap, permissively licensed, actively
maintained. The repo reports a Llama-3-8B base/instruct pair substantially outperforming the
original Falcon setup.

### ⚠️ Binoculars — BSD-3

*2 aligned models (base + instruct) · 2 forward passes · ~10–16 GB resident at 4-bit*

Observer perplexity over cross-perplexity. Elegant and widely cited, but it scores **84.4
AUROC on clean RAID** against 97–99 for trained encoders. Worth correcting a common belief:
the original licence is plain BSD-3 — the restrictive PolyForm Noncommercial licence belongs
to a popular *fork*. Note that the shipped threshold is calibrated for the Falcon-7B pair
specifically and is meaningless with any other models.

### ⚠️ `MayZhou/e5-small-lora-ai-generated-detector` — MIT

*e5-small + LoRA · 33M params · single-digit ms · RAID-test accuracy 0.939*

Small enough to run on every keystroke on CPU, with a standard sequence-classification head
so it wires up in three lines. The right first-run experience: instant, no 1.7 GB download,
no MPS risk. Use it as a triage tier that decides which paragraphs deserve a real scoring
pass.

### ❌ `roberta-base-openai-detector` · `chatgpt-detector-roberta` — obsolete

GPT-2 era and January-2023-ChatGPT era respectively. The Hello-SimpleAI model was already
down to ~57% off-distribution accuracy within months of release. Both still rank high in
search results; ignore them.

> 🚨 **Runtime gotcha — read before picking a stack.**
>
> **Neither llama.cpp's server nor Ollama can score text you already have.** Both expose
> logprobs for *generated* tokens only; llama.cpp's `n_probs` array is explicitly of length
> `n_predict`, and `echo=true` returns a single token rather than the prompt. Binoculars
> needs more still — the full next-token distribution, not top-k, which no server API
> returns.
>
> The routes that do work on Apple Silicon: **MLX**, where `model(tokens)` returns logits
> for all positions in one pass and the scoring loop is roughly ten lines;
> `llama-cpp-python` in-process with `logits_all=True`; or plain transformers on MPS. Budget
> around 2 GB of logit memory per pass for a 128k-vocab model at 4k tokens.

---

## 4. Reality check

Single source for everything in this section: Jaeho Lee, ["AI detectors rarely flag human
writing, but sometimes miss AI text imitating real
authors"](https://epoch.ai/data-insights/ai-detectors-false-negatives), Epoch AI Data
Insight, 15 July 2026 ([data and figure
code](https://github.com/jaeholee-brown/ai-text-detectors), CC-BY). 495 human passages of
~500 words from 99 authors — blogging from pre-2022 Wayback snapshots, published short
fiction, and **scientific writing pulled from arXiv** — against 594 AI passages from Claude
Opus 4.8, GPT-5.5 and Gemini 3.1 Pro. Detector versions pinned to June 2026.

The scientific-prose result comes with a qualifier that changes what you should do about it.
**Scientific text is not intrinsically hard to classify — it is hard only when the AI has
been shown samples to imitate.** Under a plain prompt, scientific AI text is caught
essentially always:

| Detector    | FP, all genres | FP, scientific | Missed AI (sci), basic prompt | Missed AI (sci), style-imitated |
| ----------- | -------------: | -------------: | ----------------------------: | ------------------------------: |
| Pangram     |     0% (0/495) |     0% (0/165) |                    0% (0/99)  |                    25% (25/99)  |
| GPTZero     |     0% (0/495) |     0% (0/165) |                    1% (1/99)  |                    24% (24/99)  |
| Originality |    4% (19/495) |     1% (1/165) |                    0% (0/99)  |                    29% (29/99)  |

- **False positives on writing like yours are genuinely near zero.** Pangram and GPTZero
  flagged none of 165 arXiv passages. "Detectors constantly flag human academic writing" is
  out of date for those two at document level on ~500-word passages.
- **The 25% miss rate is specific to few-shot style imitation.** Lee's method: show the
  model five real passages by an author and ask for a new piece in that style. Worst single
  cell was Gemini-on-scientific evading Pangram, 16 of 33. Under the matched basic-prompt
  condition the same genre gives 0–1%. So the failure mode is *voice-matched* AI, not
  technical subject matter as such — which is arguably the exact case you'd want flagged,
  since it's what LLM-assisted drafting of your own paper produces.
- **Read the cell sizes before leaning on this.** Genre-by-model cells hold 33 passages
  each, so a "48%" is 16 events. Only three commercial detectors were tested; only one
  evasion method; 500-word passages only, and detection depends on length. It's a
  well-documented Data Insight, not a peer-reviewed benchmark.
- **Nothing here covers open detectors.** No published evaluation tests any open model
  against Claude 4.x, GPT-5.x or Gemini 3.x; RAID's generator set is 2023-era. An April 2026
  paper found detectors lean on dataset-specific stylistic cues rather than stable
  authorship signal, and that under distribution shift they assign *high-confidence machine
  labels to human text from unseen domains*. Markdown-and-LaTeX prose is an unseen domain
  even to the arXiv-trained ones.

**Practical consequence: calibrate your threshold on a corpus of your own known-human
writing before you highlight anything.** The default 0.5 is very likely wrong for you. And
build an abstain band rather than a binary highlight — every detector degrades below about
50 words, so refuse to score short paragraphs and say so in the hover instead of emitting a
meaningless colour.

> ⚠️ **A note on the tooling's motives.** The `ngpepin/binoculars` README describes itself as
> an "AI text forensics *and humanization* toolkit" — for every flagged span it generates
> three rewrite options ranked by estimated improvement in the Binoculars score. That is
> evasion tooling wearing a detector's clothes, and it has a design consequence: features
> tuned to make flagged text pass will drift away from features that make the score
> *diagnostic*. If you borrow its UX, borrow the analysis half and leave the rewrite loop.

---

## 5. Prior art

Nobody has shipped a Marketplace extension that tints paragraphs using a real detector. The
two that decorate well use regex and stylometry heuristics; the one that uses a real
detector isn't distributable.

### `ngpepin/binoculars` — PolyForm NC, **dormant**

*Local Binoculars, Llama-3.1-8B pair via llama-cpp-python · extension id
`npepin.binoculars-vscode`*

The licence is a non-issue for personal use. The maintenance state is the real cost: **1
star, 0 forks, last commit 27 Feb 2026**, one author, hardcoded `/home/npepin/…` paths as
config defaults. Treat it as a design document rather than a dependency.

Worth stealing: stale-state yellow on edited paragraphs, faint "prior contributor"
backgrounds, ~1.5 s delayed hovers, a warm-then-unload model daemon, hash-checked sidecar
files, live estimate in the status bar. Its HTTP mode (`POST /score`) is still the fastest
way to prototype against a real scorer.

### `lmscan` — Apache-2.0

*`ZachBg.lmscan` · local CLI, 12 statistical heuristics*

Closest architectural match on the Marketplace: paragraph background decorations,
diagnostics, status-bar document score, `autoScan` off by default. Copy the settings shape;
replace the detector.

### `llm-slop-detector` — MIT

*`thias-se.llm-slop-detector` · regex and Unicode rules, no model*

Best-engineered of the lot. Its markdown scoping (skips fenced code, link URLs, frontmatter),
`<!-- slop-disable -->` directives and workspace-trust gating are directly reusable.

### `houtini-ai/ai-detect` — MIT

*desklib DeBERTa, sentence by sentence, plus a 126 MB int8 ONNX fallback*

Ready-made CLI, Python package and MCP server. The ONNX path sidesteps the MPS question
entirely. Note `--device` offers only `auto|cpu|cuda` — no MPS.

### Two non-obvious dead ends

**Vale cannot be extended for this** — its `extends: script` hook runs sandboxed Tengo with
no network, no exec and no file I/O, and Vale Server was removed in v0.15.0.

**Harper likewise:** its `Linter` trait is synchronous and offline by design with no
external-checker config, so adding a scorer means a Rust fork.

The one genuine hijack available is **LTeX+**'s `ltex.ltex-ls.languageToolHttpServerUri` —
implement LanguageTool's `POST /v2/check` shape and you inherit its markdown/LaTeX offset
mapping for free. Diagnostics only, so no background tint, but it's a two-day prototype.

---

## 6. An architecture that keeps the backend swappable

### 1. Scorer as a Python LSP server

Use `pygls`. Your detector will be Python regardless — transformers, MLX, llama-cpp-python —
so this avoids inventing a bridge protocol, and you get JSON-RPC framing, `$/cancelRequest`
and incremental document sync for free. Expose one pluggable backend interface:
`sapling | pangram | gptzero | fastdetect-local | encoder-local`. Publish diagnostics at
`Information` severity with the score in `data: {score, hash}`.

### 2. Thin VS Code client that adds the tint

LSP has no background-colour primitive — `publishDiagnostics` carries severity and tags and
nothing else. So intercept your own diagnostics in `vscode-languageclient`'s
`middleware.handleDiagnostics`, read `diagnostic.data`, and drive `setDecorations` yourself.
Neovim, Helix and Zed then get diagnostics for free on day one.

```ts
// create ONCE at activation — per-render types leak and stack
const flagged = vscode.window.createTextEditorDecorationType({
  backgroundColor: 'rgba(176,106,21,0.14)',   // rgba, or you kill find-highlight
  isWholeLine: true,
  overviewRulerColor: 'rgba(176,106,21,0.7)', // free minimap stripe
  overviewRulerLane: vscode.OverviewRulerLane.Right,
  rangeBehavior: vscode.DecorationRangeBehavior.ClosedClosed,
});
editor.setDecorations(flagged, ranges);       // replaces ALL ranges of this type
```

### 3. Hash paragraphs for identity and cache

Segment on blank lines, markdown-aware so you skip fenced code, frontmatter and tables. Key
the cache on `sha256(normalisedText) + backendId + modelVersion + threshold` — omit any of
those and switching model silently serves stale scores. On `onDidChangeTextDocument`, walk
`contentChanges` and mark only intersecting paragraphs stale rather than re-segmenting the
document. VS Code translates decoration ranges through edits for you; what it won't tell you
is when a score has gone *stale*, and a stale amber tint is worse than no tint.

### 4. Two-tier debounce, manual by default

Around 300 ms to update positions and staleness locally, which is free; several seconds of
idle — or explicit command only — before anything goes over the wire. Default to `manual`,
offer `onSave`, and never auto-scan on open with a remote backend. Show spend in the
status-bar tooltip: "142 paragraphs scored this session, ~38k characters" is the difference
between a tool you trust and one that surprises you with a bill.

### 5. Ship the tiny model first

First-run should be a 33M-parameter e5-small or the 126 MB ONNX model, running on CPU in
single-digit milliseconds. The 1.7 GB DeBERTa and the 8B-pair zero-shot backend are opt-in
upgrades, not the install. Put the "this is a probabilistic signal, not proof of authorship"
caveat in the hover, not just the README.

---

## Corrections since the first draft

- The PolyForm Noncommercial licence on `ngpepin/binoculars` was flagged as a blocker; it
  isn't one for personal use, and the repo's dormancy and single-author status are the real
  cost.
- The claim that scientific prose is the weakest genre for detection was stated without its
  necessary qualifier — it holds only under few-shot style imitation, not under plain
  prompting.

## Not verified

- GPTZero's developer API rate, which is not published. The $0.39/1k-word figure in
  circulation comes from a third-party reseller.
- Accuracy figures for `desklib/ai-text-detector-academic-v1.01`, which publishes none.
- Whether the desklib models run cleanly on PyTorch MPS.
- Whether MELD — currently the strongest open detector on RAID at 99.8 AUROC — has a
  de-anonymised public release with a usable licence. That is the first thing to check if
  you want state of the art.

## Sources

- [Sapling detector API](https://sapling.ai/docs/api/detector/) ·
  [Sapling pricing](https://sapling.ai/docs/api/pricing/)
- [Pangram pricing](https://www.pangram.com/pricing)
- [GPTZero developers](https://gptzero.me/developers)
- [Epoch AI, Jul 2026](https://epoch.ai/data-insights/ai-detectors-false-negatives) ·
  [replication repo](https://github.com/jaeholee-brown/ai-text-detectors)
- [desklib/ai-text-detector-academic-v1.01](https://huggingface.co/desklib/ai-text-detector-academic-v1.01)
- [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt) ·
  [arXiv:2310.05130](https://arxiv.org/abs/2310.05130)
- [Binoculars](https://github.com/ahans30/Binoculars) ·
  [arXiv:2401.12070](https://arxiv.org/abs/2401.12070)
- [ngpepin/binoculars](https://github.com/ngpepin/binoculars)
- [houtini-ai/ai-detect](https://github.com/houtini-ai/ai-detect)
- [ltex-ls-plus](https://github.com/ltex-plus/ltex-ls-plus)
- [VS Code API reference](https://code.visualstudio.com/api/references/vscode-api)
