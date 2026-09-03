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
        json.dumps({"source": "post/x.qmd", "snapped_at": "2026-08-01T00:00:00+00:00",
                    "content": "Original text.\n"}),
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
    except ValueError as exc:
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
    assert set(group.commands) == {"migrate-captures", "prune-invented-anchors"}
