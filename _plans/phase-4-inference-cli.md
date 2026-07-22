# Phase 4 · Inference CLI — 🔧 BUILT (2026-07-22); serving slices open

Wrap the trained adapter as a text transformer Dan runs over AI drafts.
Was gated on a trained adapter existing (Phase 3 — done, run `20260721-run1`).

## As built (2026-07-22)

`stylebot.infer` + `ai-style run` (mechanism) and `dan-style run` (blog
policy) — this resolves the old framing split between this file (`ai-style
run`) and `ai-style-fine-tune.md` §"Phase 5" (`workflow_style.py`): the
mechanism/policy pattern serves both; a blog-build `workflow_style.py`
integration remains a later slice on top of the same library.

- **`stylebot/infer.py`**: a `Styler` is `callable(messages, num_samples) ->
  list[str]`. Backends: `tinker_backend` (samples the manifest's checkpoint
  via the Tinker sampling client, rendered with the SAME cookbook renderer as
  training — no chat-template skew possible; ~$0.60/1M tokens, zero serving
  infra) and `openai_backend` (any OpenAI-compatible endpoint: Fireworks,
  `mlx_lm.server`, Osaurus, vLLM). `rewrite_text` mirrors the TRAINING
  protocol: headings and code fences are protected (pairs never contain them
  as content), each prose chunk goes out with the nearest heading prepended
  via `build_pair_content` and stripped back off the sample (`pair_body`),
  frontmatter never leaves the file, whitespace reassembles byte-for-byte,
  and a blank sample falls back to the input chunk. **The chunk budget must
  match the training chunk scale** (`dan-style run` passes
  `MERGE_MAX_CHARS` = 1500): a 5.4KB single-chunk rewrite came back
  byte-identical (identity fallback at OOD length, 2026-07-22), while the
  same post at training scale moved whole-body detector P(slop) 0.589→0.281.
  `rewrite_pairs_file` emits the eval-ready outputs JSONL (pair record +
  top-level `"output"`; `eval.FIELD_EXTRACTORS["output"]` added).
- **`ai-style run <file>`**: stdout default, `--write` (with `.bak`),
  `--backend tinker|openai`, `--best-of N` reranked by the voice classifier
  (`detector_reranker`, lowest P(slop) wins; small N is Goodhart-gentle —
  keep judge+eyeball as the orthogonal guard, see eval-harness.md).
- **`dan-style run`** (livingthing `bin/style_file.py`): checkpoint/base
  model/renderer read from the NEWEST completed manifest in
  `_training_pairs/runs/` — the committed record drives serving, no flag
  soup; `--best-of` uses `_models/voice-clf`.
- Tests: `tests/test_infer.py` (chunk round-trip exactness, fence safety,
  frontmatter protection, best-of rerank, resumable outputs JSONL).

**Serving decision v1 (supersedes the Fireworks-first Inputs below): the
Tinker sampling client.** It needs no new accounts or deploys, the checkpoint
URI is pinned in the committed manifest, and cost is per-token with zero idle
spend (scale-to-zero by construction). Fireworks and local MLX are *backends
behind the same seam*, each its own verification slice (D1's build note:
Tinker documents vLLM/SGLang export only — both handoffs unverified).

## Open slices

- **Local MLX — ✅ done 2026-07-22** (run 2): livingthing `bin/styler-mlx`
  (idempotent merge→4-bit from the newest manifest) + `mlx_lm.server
  --chat-template-args '{"enable_thinking":false}'` — that flag is
  load-bearing (verified string-identical to the training render; the
  template default opens an unclosed think block). Parity on 20 val pairs
  @0.3: detector output means tinker 0.4627 vs local 4-bit 0.4529
  (Δ −0.0098 < 0.03 gate; quantization drift within gate, no 8-bit build
  needed); zero template artifacts; real-post round-trip structure-clean.
  Commands: blog RUNBOOK §12. Upstream shims (documented in the script):
  transformers 4.57 lacks the `qwen3_5` arch (resolve base from HF cache);
  cookbook `build_hf_model` expects raw Tinker adapter key naming (invert
  the serving-PEFT renames).
- **Blockquote integrity**: a reflowed `> quote` can lose its `>` prefix on
  continuation lines — quotes aren't anchor-guarded. Candidate: add
  blockquote-prefix counts to `content_anchors` or protect quote blocks
  like fences.
- **Fireworks**: upload the PEFT adapter, `--min-replica-count 0`, retry on
  `503 DEPLOYMENT_SCALING_UP`; same parity acceptance test.
- **`--mask` defense-in-depth** (D4): NB sentinel-prefix mismatch —
  `STYLE_SYSTEM` promises `〈MASKED_*〉` preservation but livingthing's
  `pandoc_mask` emits `〈CODE_*/MATH_*/URL_*〉`; reconcile (retrain or rename)
  before wiring.
- **Blog-build integration** (`workflow_style.py`, `date-ai-style`
  frontmatter, ai-preen-style guards) on top of `stylebot.infer`.

## Done-criteria

- [x] `dan-style run` round-trips a real draft (2026-07-22,
      notebook/causal_hierarchy, automation:2): frontmatter/headings/fences/
      callouts preserved, whole-body detector P(slop) 0.589 → 0.281.
      **Known quality issue for run 2:** the adapter sometimes over-compresses
      — on this post it collapsed a three-paragraph explanation into a list,
      deleting a citation and a link. Candidate cause: the qwen3-32b 4–7×
      compression pairs; the manifest's generator facet makes the ablation a
      one-selector v2 run.
- [x] No idle spend: v1 serving is per-token sampling (scale-to-zero moot);
      the Fireworks scale-to-zero variant moves to its own slice above.
- [x] Eval-harness scores beat the prompt-only baseline, three signals
      agreeing (40 val pairs): detector P(slop) slop 0.507 → output 0.414 →
      target 0.352 (prompt-only baseline 0.552, WORSE than slop; adapter wins
      35/40); judge 2.40 → 3.45 → 4.33; Vale alerts 3.3 → 1.28 → 1.2.
- [x] Registered under the one `ai-style` entry point (`ai-style run`), with
      the `dan-style run` mirror.
