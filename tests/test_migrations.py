"""Tests for the one-off migrations in `stylebot.migrations` and `ai-style data`.

These delete corpus records, so their safety properties are pinned even
though the code is deletable by design: a prune spares captured (non-synthetic)
records unless asked, and no command writes anything without `--apply`. Every
command in the group is covered here.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner

from stylebot import migrations, pairs
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style_data import main as data_cli


def _pair(slop: str, target: str, *, synthetic: bool, source: str = "post/x.qmd") -> dict:
    meta = {"source": source}
    if synthetic:
        meta |= {"synthetic": True, "generator": "g", "synth_key": slop[:8]}
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": slop},
            {"role": "assistant", "content": target},
        ],
        "meta": meta,
    }


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# prune-stale (W12): orphans, and eval contamination
#
# Every case here runs against a fixture corpus and a fixture blog under
# `tmp_path`. The live corpus is 8,244 records of paid generation and one
# irreversible edit; a test that touches it is a test that can destroy it.
# ---------------------------------------------------------------------------

from stylebot.targets import ChunkPolicy  # noqa: E402

# Paragraphs long enough to clear `min_chars` below, and distinct enough that a
# containment-based passage grouping cannot fold them together.
PARA_A = (
    "The first paragraph is about the way a corpus goes stale when nobody is "
    "looking, and it is long enough to survive the minimum-chars floor."
)
PARA_B = (
    "The second paragraph concerns the difference between a pair that cannot be "
    "regenerated and a pair that merely will not be, which is the whole point."
)
PARA_GONE = (
    "The third paragraph was in the post when the corpus was generated and has "
    "since been deleted from it, so no current cut reproduces this text at all."
)

STALE_POLICY = ChunkPolicy(min_chars=40, merge=False, drop_link_dumps=False)


def _blog(tmp_path: Path, *, paragraphs=(PARA_A, PARA_B), name: str = "post/x.qmd") -> Path:
    """A blog root holding one post, with frontmatter `iter_targets` will read."""
    root = tmp_path / "blog"
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\ntitle: X\nautomation: 0\n---\n\n" + "\n\n".join(paragraphs) + "\n",
        encoding="utf-8",
    )
    return root


def _splits(eval_posts=(), aliases=None) -> dict:
    """A splits doc where the hash rule sends everything unpinned to `styler`."""
    doc = {
        "schema_version": 1,
        "eval_posts": list(eval_posts),
        "rest_rule": {"kind": "hash", "detector_frac": 0.0, "salt": "0"},
    }
    if aliases:
        doc["aliases"] = aliases
    return doc


def _synth(target: str, *, source: str = "post/x.qmd", generator: str = "g") -> dict:
    rec = _pair("Sloppy version of it.", target, synthetic=True, source=source)
    rec["meta"] |= {"generator": generator, "synth_key": target[:12]}
    return rec


def test_orphans_are_cells_the_current_cut_no_longer_produces(tmp_path):
    root = _blog(tmp_path)
    path = _write(tmp_path, [_synth(PARA_A), _synth(PARA_GONE)])
    cells, report = migrations.find_orphans(path, blog_root=root, policy=STALE_POLICY)

    assert cells == {("post/x.qmd", PARA_GONE)}
    assert (report.orphans, report.synthetic, report.orphan_cells) == (1, 2, 1)
    assert report.per_post == Counter({"post/x.qmd": 1})


def test_a_real_capture_is_unchecked_rather_than_clean(tmp_path):
    """The asymmetry is structural, not a kindness. A capture's target comes
    from a paragraph diff of an edit archive and was never cut by
    `iter_targets`, so asking whether the chunker reproduces it is a category
    error — and answering "clean" would be a claim the measurement cannot make."""
    root = _blog(tmp_path)
    real = _pair("Before.", PARA_GONE, synthetic=False)
    path = _write(tmp_path, [real, _synth(PARA_GONE)])
    cells, report = migrations.find_orphans(path, blog_root=root, policy=STALE_POLICY)

    assert report.real_unchecked == 1 and report.synthetic == 1
    assert report.orphans == 1  # the synthetic one only
    # And the plan must not drop it, though its target is in the orphan set.
    plan = migrations.plan_stale_prune(
        path, blog_root=root, policy=STALE_POLICY, splits=None, drop_eval=False
    )
    assert cells == {("post/x.qmd", PARA_GONE)}
    assert [r["meta"].get("synthetic") for r in plan.keep] == [None]


def test_a_missing_source_is_reported_apart_from_a_real_orphan(tmp_path):
    """A post renamed without a splits alias wants an alias, not a prune, so it
    cannot be allowed to hide inside the orphan count."""
    root = _blog(tmp_path)
    path = _write(tmp_path, [_synth(PARA_A, source="post/renamed-away.qmd")])
    _cells, report = migrations.find_orphans(path, blog_root=root, policy=STALE_POLICY)
    assert report.orphans == 1 and report.orphans_missing_source == 1


def test_an_alias_re_cuts_a_renamed_post_instead_of_orphaning_it(tmp_path):
    """W0's identity-aware `role_of` shares its alias map with the re-cut: a post
    whose pairs are keyed to its old path is re-cut at its new one."""
    root = _blog(tmp_path, name="post/new.qmd")
    splits = _splits(aliases={"post/old.qmd": "post/new.qmd"})
    path = _write(tmp_path, [_synth(PARA_A, source="post/old.qmd")])
    _cells, report = migrations.find_orphans(
        path, blog_root=root, policy=STALE_POLICY, splits=splits
    )
    assert report.orphans == 0


def test_eval_contamination_drops_synthetic_and_spares_captures(tmp_path):
    """`log save` warns rather than refuses on an eval post, because a captured
    rewrite of held-out prose is still real editing work. The prune agrees."""
    splits = _splits(eval_posts=["post/frozen.qmd"])
    path = _write(
        tmp_path,
        [
            _synth(PARA_A, source="post/frozen.qmd", generator="g1"),
            _synth(PARA_B, source="post/frozen.qmd", generator="g2"),
            _pair("Before.", PARA_A, synthetic=False, source="post/frozen.qmd"),
            _synth(PARA_A, source="post/x.qmd"),
        ],
    )
    rows, report = migrations.find_eval_contamination(path, splits)
    assert rows == {0, 1}
    assert (report.synthetic_dropped, report.real_spared) == (2, 1)
    assert report.per_generator == Counter({"g1": 1, "g2": 1})


def test_eval_contamination_follows_the_alias_map(tmp_path):
    """The leak W0 closed: a pinned eval post was renamed, so its pairs hashed
    into styler/detector under the new path. Contamination has to be counted by
    identity or the prune would miss exactly the pairs that motivated it."""
    splits = _splits(eval_posts=["post/old.qmd"], aliases={"post/new.qmd": "post/old.qmd"})
    path = _write(tmp_path, [_synth(PARA_A, source="post/new.qmd")])
    rows, report = migrations.find_eval_contamination(path, splits)
    assert rows == {0} and report.per_post == Counter({"post/old.qmd": 1})


def test_plan_accounts_for_the_overlap_between_the_two_sets(tmp_path):
    """A pair can be both an orphan and eval contamination; the dropped total is
    a union, and the report says so rather than double-counting."""
    root = _blog(tmp_path, name="post/frozen.qmd")
    splits = _splits(eval_posts=["post/frozen.qmd"])
    path = _write(
        tmp_path,
        [
            _synth(PARA_A, source="post/frozen.qmd"),  # eval only
            _synth(PARA_GONE, source="post/frozen.qmd"),  # eval AND orphan
        ],
    )
    plan = migrations.plan_stale_prune(path, blog_root=root, policy=STALE_POLICY, splits=splits)
    assert (plan.n_orphans, plan.n_eval, plan.n_both, plan.n_dropped) == (1, 2, 1, 2)
    assert plan.composition["now"]["total"] == 2
    assert plan.composition["minus orphans"]["total"] == 1
    assert plan.composition["minus both"]["total"] == 0
    assert set(plan.variants) == set(plan.composition)

    # A variant that was not computed is ABSENT, not equal to "now": a row
    # reading 2 under "minus orphans" would say "there are no orphans", which is
    # the opposite of "orphans were not looked for".
    eval_only = migrations.plan_stale_prune(
        path, blog_root=None, policy=None, splits=splits, drop_orphans=False
    )
    assert set(eval_only.variants) == {"now", "minus eval synthetic"}


def test_plan_refuses_rather_than_guessing_when_an_input_is_missing(tmp_path):
    path = _write(tmp_path, [_synth(PARA_A)])
    for kwargs in (
        {"blog_root": None, "policy": STALE_POLICY, "splits": None, "drop_eval": False},
        {"blog_root": None, "policy": None, "splits": None, "drop_orphans": False},
    ):
        try:
            migrations.plan_stale_prune(path, **kwargs)
        except migrations.MigrationError:
            continue
        raise AssertionError(f"planned with {kwargs} and no input to plan from")


def test_styler_passage_scale_counts_prose_not_pairs(tmp_path):
    """W9's number, which is what says whether a prune helps the dilution: two
    budgets over the same prose are ONE passage, so a real pair cut once weighs
    far more per passage than per pair."""
    splits = _splits()
    nested = PARA_A[:80]  # a shorter cut of the same prose
    records = [
        _synth(PARA_A),
        _synth(nested),
        _pair("Before.", PARA_B, synthetic=False),
    ]
    scale = migrations.styler_passage_scale(records, splits=splits)
    assert scale["pairs"] == 3 and scale["passages"] == 2
    assert scale["real_pairs"] == 1 and scale["real_passages"] == 1
    assert scale["real_pair_share"] == 33.33 and scale["real_passage_share"] == 50.0


def test_prune_stale_dry_run_writes_nothing(tmp_path):
    root = _blog(tmp_path)
    path = _write(tmp_path, [_synth(PARA_A), _synth(PARA_GONE)])
    before = path.read_text(encoding="utf-8")
    res = CliRunner().invoke(
        data_cli,
        [
            "prune-stale",
            str(path),
            "--blog-root",
            str(root),
            "--no-eval-synthetic",
            "--policy",
            f"{__name__}:stale_policy",
        ],
    )
    assert res.exit_code == 0, res.output
    assert path.read_text(encoding="utf-8") == before
    assert not path.with_suffix(".jsonl.bak").exists()
    assert "orphaned pairs          : 1" in res.output
    assert "UNCHECKED" in res.output and "dry run" in res.output


def stale_policy() -> ChunkPolicy:
    """A `--policy MODULE:CALLABLE` factory, in the shape the blog's is."""
    return STALE_POLICY


def test_prune_stale_apply_writes_exactly_what_the_dry_run_reported(tmp_path):
    """The contract that makes a dry run worth reading: the two paths differ by
    one `write_pruned` call, so the plan the user approved is the plan applied."""
    root = _blog(tmp_path)
    real = _pair("Before.", PARA_GONE, synthetic=False)
    path = _write(tmp_path, [_synth(PARA_A), _synth(PARA_GONE), real])
    argv = [
        "prune-stale",
        str(path),
        "--blog-root",
        str(root),
        "--no-eval-synthetic",
        "--policy",
        f"{__name__}:stale_policy",
    ]

    dry = CliRunner().invoke(data_cli, argv)
    assert dry.exit_code == 0, dry.output
    res = CliRunner().invoke(data_cli, [*argv, "--apply"])
    assert res.exit_code == 0, res.output

    kept = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert [r["messages"][2]["content"] for r in kept] == [PARA_A, PARA_GONE]
    assert [bool(r["meta"].get("synthetic")) for r in kept] == [True, False]
    backup = [json.loads(x) for x in path.with_suffix(".jsonl.bak").read_text().splitlines()]
    assert len(backup) == 3
    assert "pruned 1 pair(s)" in res.output


# ---------------------------------------------------------------------------
# backfill-draws (W4 step 3, D84): pairs and side files -> draws
# ---------------------------------------------------------------------------

from stylebot import corpus, draws  # noqa: E402

GATE_ROW = {
    "at": "2026-08-12T15:31:47+00:00",
    "synth_key": "50d17b5bf22843e2",
    "source": "post/x.qmd",
    "generator": "openrouter/qwen/qwen3.5-9b",
    "slop_strategy": "linkedin",
    "reasoning_effort": "off",
    "kind": "gate",
    "gate": "skipped_invented_anchors",
    "reason": "generator deleted 6 anchor(s) the target keeps",
    "attempt": 0,
    "cost": 7.7e-05,
}
ERROR_ROW = {
    **GATE_ROW,
    "at": "2026-07-16T16:38:00+00:00",
    "kind": "error",
    "gate": None,
    "reason": "RuntimeError: provider returned no choices",
    "attempt": 2,
    "cost": None,
}


@pytest.fixture
def sharded(tmp_path, write_pairs_corpus) -> Path:
    """A `pairs/` of 8 records — two real posts, two synthetic — plus one side file."""
    flat = write_pairs_corpus(tmp_path / "flat.jsonl", n_posts=4, chunks=2, n_synth_posts=2)
    out = tmp_path / "pairs"
    out.mkdir()
    corpus.rewrite(out, corpus.read_records(flat))
    side = tmp_path / "lost_draws.jsonl"
    side.write_text(json.dumps(GATE_ROW) + "\n" + json.dumps(ERROR_ROW) + "\n", encoding="utf-8")
    return out


def _view_sha(draws_dir: Path, out: Path, **policy) -> str:
    report = draws.build_view(
        draws.iter_draws(draws_dir), draws.ViewPolicy(keep_superseded=True, **policy), out=out
    )
    return report.content_sha256


def test_backfill_dry_run_writes_nothing_and_reports(sharded, tmp_path):
    lines: list[str] = []
    before = corpus.content_sha256(sharded)
    report = migrations.backfill_draws(
        sharded,
        tmp_path / "draws",
        side_files=[tmp_path / "lost_draws.jsonl"],
        apply=False,
        log=lines.append,
    )
    assert not (tmp_path / "draws").exists()
    assert corpus.content_sha256(sharded) == before
    assert (report.records, report.resolved, report.unresolved) == (8, 0, 8)
    assert report.tombstones == {"gate:skipped_invented_anchors": 1, "error:RuntimeError": 1}
    assert report.shards == [] and not report.applied
    assert any("dry run" in x for x in lines) and "pair records            : 8" in lines


def test_backfill_apply_writes_pairs_shaped_shards_the_view_reproduces(sharded, tmp_path):
    report = migrations.backfill_draws(
        sharded,
        tmp_path / "draws",
        side_files=[tmp_path / "lost_draws.jsonl"],
        apply=True,
        log=lambda _: None,
    )
    assert [p.name for p in report.shards] == [
        "captures.jsonl",
        "synth-2026-07.jsonl",
        "synth-2026-08.jsonl",
    ]
    assert corpus.content_sha256(sharded) == corpus.content_sha256(sharded)  # untouched
    assert _view_sha(tmp_path / "draws", tmp_path / "view") == corpus.content_sha256(sharded)
    stones = [d for d in draws.iter_draws(tmp_path / "draws") if d.outcome != draws.ACCEPTED]
    assert [d.outcome for d in stones] == ["error:RuntimeError", "gate:skipped_invented_anchors"]
    assert all(d.record is None and d.raw_output is None and d.target_text is None for d in stones)
    assert stones[1].gen["reason"] == GATE_ROW["reason"] and stones[1].source == "post/x.qmd"
    assert len({d.draw_id for d in stones}) == 2 and stones[0].drawn_at == ERROR_ROW["at"]


def test_backfill_resolve_hook_fills_provenance_and_rekeys(sharded, tmp_path):
    def resolve(record: dict) -> tuple[str | None, bool, tuple[int, int] | None]:
        meta = record["meta"]
        if meta["source"] == "post/p0.qmd":
            return None, False, None  # an unreachable cell keeps null provenance
        return "abc123", False, (meta["chunk_index"] * 100, meta["chunk_index"] * 100 + 50)

    report = migrations.backfill_draws(
        sharded, tmp_path / "draws", resolve=resolve, apply=True, log=lambda _: None
    )
    assert (report.resolved, report.unresolved, report.duplicate_ids) == (6, 2, 0)
    got = list(draws.iter_draws(tmp_path / "draws"))
    records = corpus.read_records(sharded)
    assert [d.record for d in got] == records
    assert [d.draw_id for d in got[:2]] == [pairs.record_id(r) for r in records[:2]]
    assert got[2].blog_sha == "abc123" and got[2].span == (0, 50) and got[3].span == (100, 150)
    assert got[2].draw_id == draws.draw_id("post/p1.qmd", "abc123", (0, 50), None, None, None, None)
    assert got[2].draw_id != pairs.record_id(records[2])
    assert _view_sha(tmp_path / "draws", tmp_path / "view") == corpus.content_sha256(sharded)


def test_backfill_superseded_marks_the_session_and_the_default_policy_drops_it(sharded, tmp_path):
    report = migrations.backfill_draws(
        sharded,
        tmp_path / "draws",
        superseded={"cap0000": "cap0001"},
        apply=True,
        log=lambda _: None,
    )
    assert report.superseded == 2  # both chunks of the interim save
    got = list(draws.iter_draws(tmp_path / "draws"))
    assert [d.superseded_by for d in got] == ["cap0001", "cap0001"] + [None] * 6
    assert _view_sha(tmp_path / "draws", tmp_path / "kept") == corpus.content_sha256(sharded)
    dropped = draws.build_view(
        draws.iter_draws(tmp_path / "draws"), draws.ViewPolicy(), out=tmp_path / "v"
    )
    assert dropped.records == 6 and dropped.excluded == {"superseded": 2}


def test_backfill_apply_refuses_a_directory_with_anything_in_it(sharded, tmp_path):
    (tmp_path / "draws").mkdir()
    (tmp_path / "draws" / "notes.txt").write_text("x")
    with pytest.raises(migrations.MigrationError, match="not empty"):
        migrations.backfill_draws(sharded, tmp_path / "draws", apply=True, log=lambda _: None)
    assert (tmp_path / "draws" / "notes.txt").exists() and not list(
        (tmp_path / "draws").glob("*.jsonl")
    )


def test_backfill_draws_command_is_in_the_hidden_group_and_dry_runs(sharded, tmp_path):
    from stylebot.bin.ai_style import main as ai_style

    assert data_cli.hidden and "backfill-draws" in data_cli.commands
    assert "data" not in CliRunner().invoke(ai_style, ["--help"]).output
    res = CliRunner().invoke(data_cli, ["backfill-draws", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "tombstones              : 2" in res.output and "dry run" in res.output
    assert not (tmp_path / "draws").exists()

    res = CliRunner().invoke(data_cli, ["backfill-draws", str(tmp_path), "--apply"])
    assert res.exit_code == 0, res.output
    assert "wrote 3 shard(s)" in res.output
    assert _view_sha(tmp_path / "draws", tmp_path / "view") == corpus.content_sha256(sharded)
    res = CliRunner().invoke(data_cli, ["backfill-draws", str(tmp_path), "--apply"])
    assert res.exit_code != 0 and "not empty" in res.output
