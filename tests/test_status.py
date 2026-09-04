"""`status` reports what is on disk, and says "absent" for what is not.

The point of the command is that documentation stopped carrying pair counts and
run ids. That only works if `status` is safe to run on a pipeline that is half
built — an absent corpus, no runs directory, no detector — because a command
that raises on a missing artifact is a command nobody puts in a README.

The other half is the cross-check: three artifacts each internally consistent
and jointly wrong (a detector fit under a different eval stratum than the one
`splits.json` now names) is the failure mode no single command could see, and
it is the reason `collect` reads all of them at once.
"""

from __future__ import annotations

import json

from click.testing import CliRunner

from stylebot import status as st
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style import main as ai_style
from stylebot.jsonl import write_jsonl_atomic


def _rec(source="post/a.qmd", *, synthetic=False, at="2026-07-21T00:00:00+00:00", body="x"):
    meta = {
        "source": source,
        "captured_at": at,
        "capture_id": f"cap{abs(hash((source, body, synthetic))) % 10**6:06d}",
        "chunk_index": 0,
        "chunk_total": 1,
    }
    if synthetic:
        meta["synthetic"] = True
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": f"It is worth noting that {body} is quite important."},
            {"role": "assistant", "content": f"{body} matters, mostly."},
        ],
        "meta": meta,
    }


def _manifest(run_id, *, result=True, promoted=None, splits_sha=None):
    doc = {
        "run_id": run_id,
        "result": {"train_tokens": 1} if result else None,
        "hyperparameters": {"base_model": "Qwen/Qwen3.5-9B", "lora_rank": 32},
        "data": {"chunk_scale": {"budgets": {"1500": 3}}},
    }
    if promoted is not None:
        doc["promoted"] = promoted
    if splits_sha is not None:
        doc["data"]["splits_sha256"] = splits_sha
    return doc


def _write_runs(runs_dir, manifests):
    runs_dir.mkdir(parents=True, exist_ok=True)
    for doc in manifests:
        (runs_dir / f"{doc['run_id']}.json").write_text(json.dumps(doc), encoding="utf-8")


# --- the promoted rule ------------------------------------------------------


def test_promoted_manifest_takes_the_newest_completed_non_nixed_run(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(
        runs,
        [
            _manifest("20260101-old"),
            _manifest("20260102-nixed", promoted=False),
            _manifest("20260103-aborted", result=False),
        ],
    )
    assert st.promoted_manifest(runs)["run_id"] == "20260101-old"
    _write_runs(runs, [_manifest("20260104-new", promoted=True)])
    assert st.promoted_manifest(runs)["run_id"] == "20260104-new"


def test_promoted_manifest_is_none_rather_than_an_error_when_nothing_qualifies(tmp_path):
    assert st.promoted_manifest(tmp_path / "nope") is None
    runs = tmp_path / "runs"
    _write_runs(runs, [_manifest("20260101-nixed", promoted=False)])
    assert st.promoted_manifest(runs) is None


# --- collect ----------------------------------------------------------------


def test_collect_on_an_empty_world_reports_absence_and_does_not_raise():
    report = st.collect()
    assert not report.corpus.exists
    assert not report.splits.exists
    assert not report.runs.exists
    assert not report.detector.exists
    text = st.format_status(report)
    assert "not given" in text


def test_collect_counts_a_sharded_corpus_across_every_shard(tmp_path):
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    write_jsonl_atomic(pairs / "captures.jsonl", [_rec(body="a"), _rec(body="b")])
    write_jsonl_atomic(
        pairs / "synth-2026-07.jsonl",
        [_rec(synthetic=True, body="c"), _rec(source="post/b.qmd", synthetic=True, body="d")],
    )
    report = st.collect(pairs=pairs)
    assert report.corpus.n_records == 4
    assert (report.corpus.n_real, report.corpus.n_synthetic) == (2, 2)
    assert report.corpus.n_posts == 2
    assert report.corpus.sharded
    assert [name for name, _ in report.corpus.shards] == [
        "captures.jsonl",
        "synth-2026-07.jsonl",
    ]
    # A single-file corpus is the same reader, differently shaped.
    flat = tmp_path / "pairs.jsonl"
    write_jsonl_atomic(flat, [_rec(body="a"), _rec(body="b")])
    unsharded = st.collect(pairs=flat).corpus
    assert (unsharded.n_records, unsharded.sharded) == (2, False)


def test_collect_flags_a_detector_fit_under_a_stale_eval_stratum(tmp_path):
    """The cross-check no single artifact can make. See the module docstring."""
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 0,
                "eval_version": 2,
                "eval_posts": ["post/a.qmd"],
                "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
            }
        ),
        encoding="utf-8",
    )
    model = tmp_path / "voice-clf"
    model.mkdir()
    head = json.dumps({"coef": [0.0], "intercept": 0.0})
    (model / "head.json").write_text(head, encoding="utf-8")
    (model / "meta.json").write_text(
        json.dumps(
            {
                "name": "voice-clf",
                "embed_model": "x",
                "splits_sha256": "deadbeefdeadbeef",
                "split": {"eval_version": 1},
            }
        ),
        encoding="utf-8",
    )
    report = st.collect(splits=splits, detector=model)
    assert any("detector was fit under" in w for w in report.warnings)


def test_collect_says_when_a_promoted_run_predates_the_splits_pin(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(runs, [_manifest("20260101-old")])
    report = st.collect(runs_dir=runs)
    assert any("predates the splits" in w for w in report.warnings)
    # ... and stops saying it once the pin is there and agrees.
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "seed": 0,
                "eval_posts": ["post/a.qmd"],
                "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
            }
        ),
        encoding="utf-8",
    )
    from stylebot.splits import splits_sha256

    _write_runs(runs, [_manifest("20260102-new", splits_sha=splits_sha256(splits))])
    report = st.collect(runs_dir=runs, splits=splits)
    assert not [w for w in report.warnings if "splits_sha256" in w]


# --- the CLI ----------------------------------------------------------------


def test_ai_style_status_prints_the_promoted_run_id_and_lists_the_rest(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(runs, [_manifest("20260101-old"), _manifest("20260102-nixed", promoted=False)])
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    result = CliRunner().invoke(
        ai_style, ["status", "--data-dir", str(data_dir), "--runs", str(runs)]
    )
    assert result.exit_code == 0, result.output
    assert "serving:  20260101-old" in result.output
    # Nixed, not hidden: it is listed, and it is not the one that serves.
    assert "20260102-nixed" in result.output
    assert "serving:  20260102-nixed" not in result.output


def test_ai_style_status_json_round_trips(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    write_jsonl_atomic(pairs, [_rec(body="a")])
    result = CliRunner().invoke(ai_style, ["status", "--pairs", str(pairs), "--json"])
    assert result.exit_code == 0, result.output
    doc = json.loads(result.output)
    assert doc["corpus"]["n_records"] == 1


# --- the per-run listing (F6 / D74) -----------------------------------------


def _verdict(run_id, *, produced_at, styler="20260817-170037"):
    """A verdict manifest: no `result`, `promoted: false`, pins under `pins`."""
    return {
        "run_id": run_id,
        "kind": "docscale-eval-verdict",
        "promoted": False,
        "produced_at": {"rewrite": produced_at, "note": "not a timestamp"},
        "pins": {
            "styler": {"run_id": styler},
            "corpus": {"content_sha256": "25524b9737caa448"},
            "splits": {"splits_sha256": "05b495adac68cdc4"},
            "detector": {"git_sha": "c0e2ac85be"},
        },
    }


def test_a_verdict_manifest_is_listed_with_its_pins(tmp_path):
    runs = tmp_path / "runs"
    _write_runs(
        runs,
        [
            _manifest("20260101-old"),
            _verdict("docscale-20260904", produced_at="2026-09-04T10:14+10:00"),
        ],
    )
    report = st.collect(runs_dir=runs)
    listed = {e.run_id: e for e in report.runs.entries}
    assert set(listed) == {"docscale-20260904", "20260101-old"}
    assert listed["docscale-20260904"].state == "not promoted"
    assert listed["docscale-20260904"].pins == {
        "styler": "20260817-170037",
        "corpus": "25524b9737caa448",
        "splits": "05b495adac68cdc4",
        "detector": "c0e2ac85be",
    }
    assert listed["20260101-old"].state == "serving"
    text = st.format_status(report)
    assert "docscale-20260904" in text and "detector=c0e2ac85be" in text


def test_a_verdict_manifest_does_not_become_the_promoted_one(tmp_path):
    """The trap F6 documented: `docscale-...` sorts above every timestamp stem.

    Its `promoted: false` and its missing `result` are both load-bearing, and
    the ordering must not depend on either.
    """
    runs = tmp_path / "runs"
    _write_runs(
        runs,
        [
            _manifest("20260101-old"),
            _verdict("docscale-20260904", produced_at="2026-09-04T10:14+10:00"),
        ],
    )
    assert st.promoted_manifest(runs)["run_id"] == "20260101-old"


def test_manifests_are_ordered_by_their_own_clock_not_their_filename(tmp_path):
    """A run id is a timestamp by convention only, so the clock decides (D74)."""
    runs = tmp_path / "runs"
    old = _manifest("20260101-old")
    old["created_at"] = "2026-01-01T00:00:00+00:00"
    new = _manifest("aardvark")
    new["created_at"] = "2026-09-04T00:00:00+00:00"
    _write_runs(runs, [old, new])
    assert [e.run_id for e in st.collect(runs_dir=runs).runs.entries] == [
        "aardvark",
        "20260101-old",
    ]
    assert st.promoted_manifest(runs)["run_id"] == "aardvark"


def test_an_offset_stamp_is_parsed_rather_than_compared_as_a_string(tmp_path):
    """`...T10:14+10:00` is EARLIER than `...T05:00+00:00`; a string sort says later."""
    runs = tmp_path / "runs"
    _write_runs(
        runs,
        [
            _verdict("verdict-a", produced_at="2026-09-04T10:14+10:00"),
            _verdict("verdict-b", produced_at="2026-09-04T05:00+00:00"),
        ],
    )
    assert [e.run_id for e in st.collect(runs_dir=runs).runs.entries] == ["verdict-b", "verdict-a"]
