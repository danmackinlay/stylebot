"""Tests for the one-off migrations in `stylebot.migrations` and `ai-style data`.

These delete corpus records and unlink archive files, so their safety
properties are pinned even though the code is deletable by design: the prune
drops a pair ONLY when the target carries an anchor the slop lacks, it spares
captured (non-synthetic) records unless asked, and neither command writes
anything without `--apply`.

Both commands are covered here. Until 2026-09-03 only the prune half had a
test, though CLAUDE.md claimed both did — which is the sort of claim a
`scripts/` carve-out makes easy to leave standing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from click.testing import CliRunner

from stylebot import migrations
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style_data import main as data_cli

# ---------------------------------------------------------------------------
# prune-invented-anchors
# ---------------------------------------------------------------------------


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


CLEAN = _pair("Slop with [docs](./d.qmd).", "Dan, tersely, with [docs](./d.qmd).", synthetic=True)
INVENTS = _pair("Slop with no link.", "Dan with [docs](./d.qmd).", synthetic=True)
LOSES = _pair("Slop with [docs](./d.qmd).", "Dan without it.", synthetic=True)
REAL_INVENTS = _pair("Before, no link.", "After, [see](./d.qmd).", synthetic=False)


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_drops_invention_only(tmp_path):
    """Loss is the OTHER failure and is not this migration's business — the
    inference guard rejects it live, and a pair whose target dropped a link the
    slop had teaches deletion, not fabrication."""
    path = _write(tmp_path, [CLEAN, INVENTS, LOSES])
    keep, report = migrations.find_inventing(path, include_real=False)
    assert len(keep) == 2
    assert report["dropped"] == {"synthetic": 1, "real": 0}
    assert [r["messages"][2]["content"] for r in keep] == [
        CLEAN["messages"][2]["content"],
        LOSES["messages"][2]["content"],
    ]


def test_spares_captured_pairs_by_default(tmp_path):
    """A synthetic cell is regenerable; a captured edit is not. The default
    reflects that asymmetry rather than treating the corpus as uniform."""
    path = _write(tmp_path, [INVENTS, REAL_INVENTS])
    keep, report = migrations.find_inventing(path, include_real=False)
    assert len(keep) == 1 and keep[0]["meta"].get("synthetic") is None
    assert report["spared_real"] == 1

    keep, report = migrations.find_inventing(path, include_real=True)
    assert keep == []
    assert report["dropped"] == {"synthetic": 1, "real": 1}
    assert report["spared_real"] == 0


def test_reports_targets_left_uncovered(tmp_path):
    """The number that says how much regeneration the cull implies: a target
    losing every pair goes back to uncovered, so the next synth run refills it."""
    other = _pair("More slop.", INVENTS["messages"][2]["content"], synthetic=True)
    path = _write(tmp_path, [CLEAN, INVENTS, other])
    _keep, report = migrations.find_inventing(path, include_real=False)
    # INVENTS and `other` share a target and both go; CLEAN's target survives.
    assert report["targets"] == 2 and report["targets_wiped"] == 1


def test_prune_dry_run_writes_nothing(tmp_path):
    path = _write(tmp_path, [CLEAN, INVENTS])
    before = path.read_text(encoding="utf-8")
    res = CliRunner().invoke(data_cli, ["prune-invented-anchors", str(path)])
    assert res.exit_code == 0, res.output
    assert path.read_text(encoding="utf-8") == before
    assert not path.with_suffix(".jsonl.bak").exists()
    assert "dry run" in res.output


def test_prune_apply_prunes_and_backs_up(tmp_path):
    path = _write(tmp_path, [CLEAN, INVENTS])
    res = CliRunner().invoke(data_cli, ["prune-invented-anchors", str(path), "--apply"])
    assert res.exit_code == 0, res.output
    kept = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1
    backup = [json.loads(x) for x in path.with_suffix(".jsonl.bak").read_text().splitlines()]
    assert len(backup) == 2  # the original survives beside the pruned file


# ---------------------------------------------------------------------------
# migrate-captures
# ---------------------------------------------------------------------------


def _legacy_capture(data_dir: Path, cap_id: str = "cap1") -> Path:
    cap_dir = data_dir / "captures"
    cap_dir.mkdir(parents=True, exist_ok=True)
    path = cap_dir / f"{cap_id}.json"
    path.write_text(
        json.dumps(
            {
                "capture_id": cap_id,
                "source": "post/x.qmd",
                "snapped_at": "2026-08-01T00:00:00+00:00",
                "whole": False,
                "before_text": "Slop before.\n",
                "after_text": "Dan after.\n",
            }
        ),
        encoding="utf-8",
    )
    return path


def _legacy_snapshot(data_dir: Path) -> Path:
    snap = data_dir / "snapshots" / "post"
    snap.mkdir(parents=True, exist_ok=True)
    path = snap / "x.qmd.json"
    path.write_text(
        json.dumps(
            {
                "source": "post/x.qmd",
                "snapped_at": "2026-08-01T00:00:00+00:00",
                "content": "Original text.\n",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_migrate_dry_run_writes_nothing(tmp_path):
    legacy = _legacy_capture(tmp_path)
    snap = _legacy_snapshot(tmp_path)
    res = CliRunner().invoke(data_cli, ["migrate-captures", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert legacy.is_file() and snap.is_file()
    assert not (tmp_path / "captures" / "cap1").exists()
    assert "dry run: 2 item(s) would convert" in res.output


def test_migrate_apply_converts_and_unlinks(tmp_path):
    """The documents land on disk as ordinary files, and the legacy blob goes
    only after the new form reads back identical."""
    legacy = _legacy_capture(tmp_path)
    _legacy_snapshot(tmp_path)
    res = CliRunner().invoke(data_cli, ["migrate-captures", str(tmp_path), "--apply"])
    assert res.exit_code == 0, res.output

    arch = tmp_path / "captures" / "cap1"
    assert not legacy.exists()
    assert (arch / "before.md").read_text(encoding="utf-8") == "Slop before.\n"
    assert (arch / "after.md").read_text(encoding="utf-8") == "Dan after.\n"
    assert json.loads((arch / "meta.json").read_text())["source"] == "post/x.qmd"

    snap = tmp_path / "snapshots" / "post" / "x.qmd"
    assert snap.is_dir() and not snap.with_suffix(".qmd.json").exists()
    assert (snap / "content.md").read_text(encoding="utf-8") == "Original text.\n"


def test_a_legacy_archive_is_refused_and_the_named_command_exists(tmp_path):
    """`capture.iter_archives` refuses a legacy blob and names its migration.

    That coupling is the whole reason these moved into the package: as a loose
    file under `scripts/` the named command was absent from the wheel, so the
    message told a `pip install stylebot` user to run a file they did not have.
    Asserting the command *resolves* is the half of the coupling that can be
    checked here; the message itself is `capture.py`'s to keep in step.
    """
    from stylebot import capture

    _legacy_capture(tmp_path)
    (tmp_path / "pairs.jsonl").write_text("", encoding="utf-8")
    try:
        list(capture.iter_archives(tmp_path / "pairs.jsonl"))
    except (ValueError, RuntimeError) as exc:
        # Widened 2026-09-03 with `stylebot.errors`: the refusal is a `RunError`
        # (a `RuntimeError`) now, not a bare `ValueError`. Both are caught
        # because what this asserts is the MESSAGE, not the class.
        message = str(exc)
    else:  # pragma: no cover - a legacy blob must be refused
        raise AssertionError("iter_archives accepted a legacy JSON archive")

    assert "ai-style data migrate-captures" in message and "--apply" in message
    help_text = CliRunner().invoke(data_cli, ["migrate-captures", "--help"]).output
    assert "--apply" in help_text


def test_data_group_is_hidden_from_the_help_tree():
    """Hidden, not absent: `--help` stays the pipeline, the command still ships."""
    from stylebot.bin.ai_style_data import main as group

    assert group.hidden is True
    assert set(group.commands) == {
        "migrate-captures",
        "prune-invented-anchors",
        "prune-stale",
        "shard-pairs",
    }


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
