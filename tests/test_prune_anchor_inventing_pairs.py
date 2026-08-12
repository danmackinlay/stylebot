"""Tests for the one-off `scripts/prune_anchor_inventing_pairs.py` migration.

The script deletes corpus records, so its safety properties get a test even
though `scripts/` is outside the package: it drops a pair ONLY when the target
carries an anchor the slop lacks, it spares captured (non-synthetic) records
unless asked, and it never writes without `--apply`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from stylebot.ai_core import STYLE_SYSTEM

SCRIPT = Path(__file__).parent.parent / "scripts" / "prune_anchor_inventing_pairs.py"


@pytest.fixture(scope="module")
def prune():
    spec = importlib.util.spec_from_file_location("prune_anchor_inventing_pairs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_drops_invention_only(prune, tmp_path):
    """Loss is the OTHER failure and is not this migration's business — the
    inference guard rejects it live, and a pair whose target dropped a link the
    slop had teaches deletion, not fabrication."""
    path = _write(tmp_path, [CLEAN, INVENTS, LOSES])
    keep, report = prune.find_inventing(path, include_real=False)
    assert len(keep) == 2
    assert report["dropped"] == {"synthetic": 1, "real": 0}
    assert [r["messages"][2]["content"] for r in keep] == [
        CLEAN["messages"][2]["content"],
        LOSES["messages"][2]["content"],
    ]


def test_spares_captured_pairs_by_default(prune, tmp_path):
    """A synthetic cell is regenerable; a captured edit is not. The default
    reflects that asymmetry rather than treating the corpus as uniform."""
    path = _write(tmp_path, [INVENTS, REAL_INVENTS])
    keep, report = prune.find_inventing(path, include_real=False)
    assert len(keep) == 1 and keep[0]["meta"].get("synthetic") is None
    assert report["spared_real"] == 1

    keep, report = prune.find_inventing(path, include_real=True)
    assert keep == []
    assert report["dropped"] == {"synthetic": 1, "real": 1}
    assert report["spared_real"] == 0


def test_reports_targets_left_uncovered(prune, tmp_path):
    """The number that says how much regeneration the cull implies: a target
    losing every pair goes back to uncovered, so the next synth run refills it."""
    other = _pair("More slop.", INVENTS["messages"][2]["content"], synthetic=True)
    path = _write(tmp_path, [CLEAN, INVENTS, other])
    _keep, report = prune.find_inventing(path, include_real=False)
    # INVENTS and `other` share a target and both go; CLEAN's target survives.
    assert report["targets"] == 2 and report["targets_wiped"] == 1


def test_dry_run_writes_nothing(prune, tmp_path, capsys):
    path = _write(tmp_path, [CLEAN, INVENTS])
    before = path.read_text(encoding="utf-8")
    assert prune.main([str(path)]) == 0
    assert path.read_text(encoding="utf-8") == before
    assert not path.with_suffix(".jsonl.bak").exists()
    assert "dry run" in capsys.readouterr().out


def test_apply_prunes_and_backs_up(prune, tmp_path):
    path = _write(tmp_path, [CLEAN, INVENTS])
    assert prune.main([str(path), "--apply"]) == 0
    kept = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(kept) == 1
    backup = [json.loads(x) for x in path.with_suffix(".jsonl.bak").read_text().splitlines()]
    assert len(backup) == 2  # the original survives beside the pruned file
