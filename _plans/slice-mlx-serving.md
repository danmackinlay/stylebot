# Slice · Local MLX serving — 📋 briefed for an implementing agent

The local, free, offline serving exit for the trained styler (decisions #D1
fallback chain; phase-4 "Open slices"). Everything runs on the Mac; the only
paid step is the parity sample (~$0.05 of Tinker sampling).

## Inputs (all exist)

- PEFT adapter: livingthing `_models/styler/20260722-run2/`
  (`adapter_config.json` + `adapter_model.safetensors`; base
  `Qwen/Qwen3.5-9B`, trained with the `qwen3_5_disable_thinking` renderer).
  Parameterize by manifest (`_training_pairs/runs/`, newest completed) — a
  run-3 must re-merge with zero code changes.
- Base weights: already in the HF cache (fetched during run 1's export).
- `mlx_lm` CLI: user-global at `~/.local/bin/mlx_lm.*` (NOT a project dep —
  do not add it to either pyproject; shell out).
- `stylebot.infer.openai_backend` — already speaks to any OpenAI-compatible
  server; `dan-style run --backend openai --base-url ... --model ...` works
  today.

## Work

1. **Merge + quantize script** — livingthing `bin/styler-mlx` (executable,
   uv-run python or shell): read the newest completed manifest → 
   `tinker_cookbook.weights.build_hf_model` (base + adapter → merged HF
   model, gitignored scratch) → `mlx_lm.convert -q` →
   `_models/styler-mlx/<run-id>-4bit/` (gitignore it; ~5GB). Idempotent:
   skip stages whose output exists. Disk note: merged bf16 is ~18GB
   intermediate — write under `_tmp/` and clean up after conversion.
2. **Serve**: document (and optionally wrap) `mlx_lm.server --model
   _models/styler-mlx/<run-id>-4bit --port 8765`; then
   `dan-style run draft.qmd --backend openai --base-url
   http://localhost:8765/v1 --model <served-name>`.
3. **Acceptance test 1 — chat-template parity** (the load-bearing check; see
   phase-4 "Open slices"): the served tokenizer_config template must
   reproduce the training render. Sample the SAME ~20 val-pair slop inputs
   (reuse `_tmp/eval-smoke/val2_pairs.jsonl`) through (a) the tinker sampling
   backend and (b) the local server at temperature 0.3; score both with the
   voice classifier (`livingthing.voice_classifier.build_detector`,
   keyless). Accept if mean detector delta < 0.03 and no output contains
   `<think>` or template artifacts. The tinker side needs
   `direnv exec . uv run ...` (key), run from the MAIN checkout.
4. **Acceptance test 2 — quantization drift**: same-input detector means,
   4-bit local vs tinker full-precision (covered by the same 20-sample
   comparison; report the delta, don't gate separately unless it exceeds
   0.05 — then also produce an 8-bit conversion and compare).
5. **Docs**: RUNBOOK §10 (the commands); one line in decisions #D1 (MLX exit
   verified / caveats); phase-4 "Open slices" MLX entry → done with numbers.

## Constraints

- livingthing files only; NO stylebot edits (the openai backend is done).
- No corpus/`pairs.jsonl` writes; no roster/policy edits; no git commits —
  implement, test, report (the reviewer commits).
- Free/keyless by default; the single paid parity sample is the only spend.

## Done-criteria

- [ ] `bin/styler-mlx` builds `_models/styler-mlx/<run-id>-4bit/` from the
      newest manifest, idempotently.
- [ ] `dan-style run` against the local server round-trips a real post with
      structure intact (the guards do the checking).
- [ ] Parity + quantization deltas reported; template check clean.
- [ ] RUNBOOK §10 written; a fresh session could do this from docs alone.
