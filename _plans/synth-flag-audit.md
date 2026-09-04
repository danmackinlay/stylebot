# `ai-style synth` flag audit — retired 2026-09-04 (P7)

**What it was.** A 2026-08-09/10 classification of the 54 click options
`synth` exposed (the next-largest subcommand had 19) into load-bearing (14),
policy constants (24) and retirement candidates (16), against four kinds of
evidence: `synth_key` inputs, the covariates actually varied across 4,169
synthetic pairs, the blog RUNBOOK's typed flags, and the blog policy pin. Its
rule: retire only non-key flags, and hardcode the value they supplied. Full
text `git show 7cfe445:_plans/synth-flag-audit.md`.

**What shipped** — stylebot `cb385ea..684d0e8` (2026-08-10 → 08-13). The
`gpt`/`local` preset path went with `--generator`, `--gpt-model`,
`--local-model`, `--context-window` (D21); `--sort`, `--report-max-rows`,
`--all`, `--tag` went 2026-08-13, each surviving as a library knob at its old
value (`synth_cli.InspectOpts.report_max_rows`, `run_synth(extra_tags=)`;
`ChunkPolicy.sort_name` deleted outright 2026-09-03, D43). No cell re-keyed.
`synth` is a 49-option command; the surface is declared once in
`stylebot.cli_options` bundles. Numbers: `findings.md` "Surface size", "What
has actually varied across the synthetic corpus".

**What it refuted.** Eight of its sixteen candidates were kept or deferred
(D83): `--temperature` / `--top-p` are unset on every pair but designed as
recorded-unkeyed covariates, and absence of evidence is not evidence (D21);
`--capture-reasoning` has no substitute as a diagnostic; the substrate cluster
`--splits` / `--role` / `--sample-frac` / `--sample-salt` is the `--role styler`
case the masking work (D33) needs; `--assign-seed` is a `synth_key` input and
outside the rule. Its "RUNBOOK claims stamped `merge_max_chars`; none are"
finding was true on 2026-08-10 and false from the 2026-08-12 v2 batches on.

**What stayed open.** Nothing in it. The substrate cluster's justification
rests on the masking work, which is
[`masking-placement.md`](masking-placement.md) and still unbuilt.
