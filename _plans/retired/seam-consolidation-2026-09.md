<!-- Retired 2026-09-05 under P7 with the runbook: wave 3 landed 2026-09-04. -->

# Seam consolidation brief, 2026-09-04 — retired

**What it was.** Wave 3 of the [ship-shape runbook](ship-shape-2026-09.md), and
the one-line claim it had to make true: `dan-style X` is `ai-style X` plus blog
policy, for real. Four command bodies (`eval`, `train-clf`, `run`, `serve`) were
promoted out of `bin/ai_style.py` into typed stylebot passes taking an `opt()`
bundle, policy hooks and an `emit`; the blog's re-implementations in
`voice_classifier.py` and `styler.py` were deleted; the mirror test began
comparing `default`, `type` and `multiple` rather than flag names; one data-dir
callback came to see both `--data-dir`/`--pairs` and `--blog-root`; and
`iter_targets` took the caller's file list, so the blog's copy of the walk went.

**What shipped.** Wave 3: stylebot `3261ba6..1aba2c6`, blog
`8a59e037a6..22d257ef96`. Lane A (stylebot) sequential because every step
touched `bin/ai_style.py`, lane B (blog) following it step by step, lane C
(the walker) in parallel worktrees.

**What it produced**, written before the code under P8 and unchanged by it:
D78 (the four bodies are passes over bundles), D79 (blog corpus defaults are
callables reading the click context; amends D45), D80 (`iter_targets` takes
`files=` with `blog_root=`; supersedes D46).

**What stayed open**, each a successor slice in the same shape:

- the blog's `bin/train_styler.py:train_cmd` is a fifth copy of the `ai_style.py`
  echo block — the same D78 move, not folded in;
- `styler.chunk_diet` is still the thin twin of `backends.serving_diet`, and a
  blog test pins it there;
- D79 says every `--blog-root` is eager; the two stylebot bundles that define one
  (`classify_cli.py`, `splits.py`) do not set `is_eager`, and are correct today
  only because click processes an invoked flag before a defaulted one;
- the parity test pins three blog literal shadows rather than repairing them
  (serve's `--backend` default, `--blog-root` spelled as `str` in two places and
  `Path` elsewhere, status's extra `exists=True`).

Git holds the 283-line brief, its step list and its done-criteria:
`git show 6fcee88:_plans/seam-consolidation-2026-09.md`.
