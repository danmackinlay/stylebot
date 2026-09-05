# Ship-shape, round 2 — retired stub (P7)

**Status: retired 2026-09-05; R0–R7 landed that day.** The full plan and its
dated residue: `git show 302e4b2:_plans/ship-shape-2-2026-09.md` — §8 there is
the successor's input list, one dated line per report.

**What it was.** The live residue of round 1's 55 out-of-scope reports plus
four owner decisions (the manifest prompt, draws everywhere, `ruff format`,
41 synth flags), then two more made the day it ran: the prose/code ratio
retired for a ceiling that only falls (R0), and a prose cull (R7).

**What shipped.** stylebot `999e943..9dd152c`, blog `37577b378d..6faf02bca6`;
corpus `content_sha256=e3d9809bddc9a616` throughout, one view on disk.

| slice | stylebot | blog |
|---|---|---|
| R1 reformat, rules | `999e943`, `a5e3e35..ebd78a6` | — |
| R0 ceiling, proof | `5d83f6a`, `17755b3` | `37577b378d` |
| R2 prompt slot | `c77c1ea` | `a52842a1ef` |
| R6 doc drift | `45b5432..5309836` | `5b975fb51d..09f99521e8` |
| R5 blog leftovers | `f099962..85a1198` | `078097be64..5d159351b4` |
| R4 correctness | `b7f9d06..d78121a` | `f8081ecd17..6d6d348423` |
| R3 draws everywhere | `ad2a95f..3f6e041` | `e0bc8b00bf..b88895bab7`, `6faf02bca6` |
| R7 prose cull | `62ef767..413d180` | — |

D86 and D87 written; D58, D83, D85 amended. `PROSE_CEILING` 4,407 → 2,276.

**What was refuted.** C7 does not reproduce — the gate's two call sites are
exclusive branches. R5's "three fewer `REPINNED` rows": the rows are genuine
policy re-pins, respelled not removed. The plan's prose figure was another
counter's; the ceiling is the test's own count. C12's wider gate is right and
fires on dry runs by design (D34); two blog fixtures had never put their
sources on disk.

**What stayed open** (all in §8 of the shown version): C13's policy clause —
three draw ids, Dan's, because it moves the corpus sha; a run-id flake in
`test_train.py` on a second boundary; `splits.py` at 799/800; the blog's 43
repo-wide ruff errors (lint-only by choice); the `C_GRID` why with no D-home;
CLAUDE.md's `uv sync` needing `--all-extras`; loose allowlist rows the
cap-only test cannot see; Dan's `.DS_Store` and `scores.jsonl` in `_training_pairs/`.
