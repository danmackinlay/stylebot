<!-- Retired 2026-09-05 under P7: wave 4 landed in full. Git holds the 581-line runbook. -->

# Ship-shape runbook, 2026-09-04 — retired

**What it was.** A four-review design QA of stylebot and the blog (seam, data
contract, mechanism, docs apparatus) turned into a dispatch runbook: one master
agent per repo's main checkout, one slice agent per slice in its own worktree on
a brief carrying inputs, a done-line and a budget note, merged only against a
gate. Four waves plus a cull; every cross-repo slice two commits, stylebot first.

**What shipped** (C6's corpus-stability floors landed on their own, re-measured):

| wave | stylebot | blog |
| --- | --- | --- |
| 1 — docs, plan retirement, budget mechanics | `fc56ce7..7e04234` | `8fcb92f639..93d3b62c88` |
| 2 and the cull — independent fixes, X1–X5 | `fae4053..2359469` | `93d3b62c88..8a59e037a6` |
| 3 — seam consolidation | `3261ba6..1aba2c6` | `8a59e037a6..22d257ef96` |
| 4 steps 1–4 — draws and views | `7b1a236..d9a25f3` | `22d257ef96..5975e421e2` |
| 4 step 5 — the re-stamp, `pairs/` retired | `4c67d7c..ae5dfb6` | `671e237579..90b7e9a8e9` |

**What it decided**, and where the argument lives: D58 (the budgets and layers
this runbook was written under), D84 (draws and views), D85 (the `SourcePolicy`),
D64 (the author slot), D39 (a score row keyed on its signal), D74 (explicit
promotion).

**What running a swarm taught**, since nothing else records it:

- an agent whose last words are "waiting on the suite" has stopped, not paused —
  check `git status` in its worktree and send a finisher into that same worktree;
- a cross-repo slice's two commits are not independently green, so the repo that
  lands first stays red until its partner lands;
- the master found staged prose in the blog main checkout, inferred a slice had
  done it, and un-staged it. It was Dan, mid-commit, in his own terminal. A main
  checkout's index belongs to the owner: a master reads it and stops, never
  writes it. CLAUDE.md P8 now says so.

**What stayed open.** The 55 dated out-of-scope reports are §8 of
`git show 6fcee88:_plans/ship-shape-2026-09.md`; the ones that are still live
work are bullets in [`OVERVIEW.md`](OVERVIEW.md)'s "Open slices". Wave 3's own
brief retired alongside this one ([`seam-consolidation-2026-09.md`](seam-consolidation-2026-09.md)).
