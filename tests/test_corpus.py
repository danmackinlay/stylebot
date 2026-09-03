"""A corpus is one file or a directory of shards, and every consumer takes both.

The failure this file exists to prevent is not a crash. It is a reader that
silently sees *part* of a sharded corpus — which reads exactly like "the corpus
is smaller than you thought", trains on a fraction of it, and reports a clean
run. So the tests below check totals and id sets, not just that a call returns.
"""

from __future__ import annotations

import json
import re

import pytest
from click.testing import CliRunner

from stylebot import config, corpus
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style import main as ai_style
from stylebot.bin.ai_style_data import main as data_cli
from stylebot.jsonl import read_jsonl
from stylebot.pairs import dedupe_pairs_file, iter_pairs, record_id, validate_pairs_file


def _rec(source="post/a.qmd", *, synthetic=False, at="2026-07-21T00:00:00+00:00", body="x"):
    meta = {
        "source": source,
        "captured_at": at,
        "capture_id": "cap0001",
        "chunk_index": 0,
        "chunk_total": 1,
    }
    if synthetic:
        meta["synthetic"] = True
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": f"It is worth noting that {body} is important."},
            {"role": "assistant", "content": f"{body} matters."},
        ],
        "meta": meta,
    }


# --- the routing rule -------------------------------------------------------


def test_shard_of_splits_by_producer_then_month():
    assert corpus.shard_of(_rec()) == "captures.jsonl"
    assert corpus.shard_of(_rec(synthetic=True)) == "synth-2026-07.jsonl"
    assert (
        corpus.shard_of(_rec(synthetic=True, at="2026-08-30T23:59:00+00:00"))
        == "synth-2026-08.jsonl"
    )


def test_a_synthetic_record_with_no_timestamp_is_not_guessed_into_a_month():
    """`undated` rather than "this month": a wrong month is unrecoverable later."""
    rec = _rec(synthetic=True)
    rec["meta"].pop("captured_at")
    assert corpus.shard_of(rec) == "synth-undated.jsonl"


def test_a_shard_directory_and_a_file_have_the_same_data_dir():
    """`pairs_path.parent` is the data dir under BOTH shapes.

    Eight call sites spell `pairs_path.parent / "captures"` or
    `splits_path(pairs_path.parent)`. That they need no change is a property of
    naming the directory `pairs`, and it is worth a test because a rename to,
    say, `shards/pairs/` would break all eight silently.
    """
    assert (config.pairs_path("/d/dd")).parent.name == "dd"


# --- migration and round trip ----------------------------------------------


@pytest.fixture
def sharded(tmp_path, write_pairs_corpus):
    """A 4-post/2-chunk corpus migrated into `<data-dir>/pairs/`."""
    data_dir = tmp_path / "dd"
    data_dir.mkdir()
    src = write_pairs_corpus(data_dir / "pairs.jsonl", n_posts=4, chunks=2, n_synth_posts=2)
    before = read_jsonl(src)
    corpus.shard_corpus(src, data_dir / "pairs")
    src.unlink()
    return data_dir, before


def test_sharding_round_trips_every_record(sharded):
    data_dir, before = sharded
    after = list(iter_pairs(data_dir / "pairs"))
    assert len(after) == len(before)
    assert {record_id(r) for r in after} == {record_id(r) for r in before}
    assert sorted(json.dumps(r, sort_keys=True) for r in after) == sorted(
        json.dumps(r, sort_keys=True) for r in before
    )


def test_the_shards_are_the_split_the_rule_names(sharded):
    data_dir, _ = sharded
    names = {p.name for p in corpus.shard_paths(data_dir / "pairs")}
    assert names == {"captures.jsonl", "synth-2026-07.jsonl"}


def test_config_prefers_the_shard_directory_once_it_exists(sharded, monkeypatch):
    data_dir, _ = sharded
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(data_dir))
    assert config.pairs_path() == data_dir / "pairs"


def test_config_keeps_the_single_file_when_nothing_migrated(tmp_path, monkeypatch):
    """Sharding is opt-in by existing: an install that never migrates never sees it."""
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(tmp_path))
    assert config.pairs_path() == tmp_path / "pairs.jsonl"


# --- the property the whole exercise is for --------------------------------


def test_a_rewrite_touches_only_the_shards_whose_content_changed(sharded):
    """The done-criterion, at library level: one shard changes, the rest are untouched.

    Checked on mtime AND bytes, because "rewrote it with identical content" is
    still a new blob to git — which is the cost sharding exists to avoid.
    """
    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    other = pairs / "captures.jsonl"
    before_bytes = other.read_bytes()
    before_mtime = other.stat().st_mtime_ns

    records = corpus.read_records(pairs, keep_undecodable=True)
    keep = [
        r
        for r in records
        if not (r["meta"].get("synthetic") and r["meta"]["chunk_index"] == 0)
    ]
    touched = corpus.rewrite(pairs, keep)

    assert [p.name for p in touched] == ["synth-2026-07.jsonl"]
    assert other.read_bytes() == before_bytes
    assert other.stat().st_mtime_ns == before_mtime
    assert len(list(iter_pairs(pairs))) == len(keep)


def test_rewriting_a_single_file_corpus_is_unchanged(tmp_path, write_pairs_corpus):
    src = write_pairs_corpus(tmp_path / "pairs.jsonl", n_posts=2, chunks=1)
    records = read_jsonl(src)
    assert corpus.rewrite(src, records[:1]) == [src]
    assert len(read_jsonl(src)) == 1


def test_an_unparseable_line_survives_a_round_trip(tmp_path):
    """`keep_undecodable` is why: a corrupt line is preserved, not silently dropped."""
    src = tmp_path / "pairs.jsonl"
    src.write_text(json.dumps(_rec()) + "\n{not json\n", encoding="utf-8")
    counts = corpus.shard_corpus(src, tmp_path / "pairs")
    assert counts == {"captures.jsonl": 1, "unparsed.jsonl": 1}
    assert (tmp_path / "pairs" / "unparsed.jsonl").read_text(encoding="utf-8") == "{not json\n"


# --- uniqueness is a property of the CORPUS, not of a shard ----------------


def test_a_duplicate_that_straddles_two_shards_is_caught(sharded):
    """The check a per-file loop would miss, and the reason to point it at the dir."""
    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    dupe = list(iter_pairs(pairs))[0]
    with (pairs / "synth-2026-07.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(dupe, ensure_ascii=False) + "\n")

    problems = validate_pairs_file(pairs)
    assert len(problems) == 1
    where, errs = problems[0]
    assert where.startswith("synth-2026-07.jsonl:")
    assert "duplicate record_id" in errs[0]
    assert "captures.jsonl:" in errs[0]  # names the shard it first appeared in


def test_dedupe_across_shards_keeps_the_first_and_backs_up(sharded):
    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    dupe = list(iter_pairs(pairs))[0]
    with (pairs / "synth-2026-07.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(dupe, ensure_ascii=False) + "\n")
    n_before = len(list(iter_pairs(pairs)))

    report = dedupe_pairs_file(pairs, apply=True)
    assert report.dropped == 1 and report.duplicate_ids == 1
    assert report.backup == data_dir / "pairs.bak"
    assert (report.backup / "captures.jsonl").exists()
    assert len(list(iter_pairs(pairs))) == n_before - 1
    assert validate_pairs_file(pairs) == []


def test_a_single_file_corpus_still_gets_a_dot_bak(tmp_path, write_pairs_corpus):
    src = write_pairs_corpus(tmp_path / "pairs.jsonl", n_posts=1, chunks=1)
    line = src.read_text(encoding="utf-8")
    src.write_text(line + line, encoding="utf-8")
    report = dedupe_pairs_file(src, apply=True)
    assert report.backup == tmp_path / "pairs.jsonl.bak"
    assert report.dropped == 1


# --- writers ---------------------------------------------------------------


def test_synth_appends_only_to_its_own_month_shard(sharded):
    from stylebot.synth import PairsWriter

    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    captures = pairs / "captures.jsonl"
    july = pairs / "synth-2026-07.jsonl"
    before = {p: p.read_bytes() for p in (captures, july)}

    rec = _rec(synthetic=True, at="2026-09-04T10:00:00+00:00", body="a september pair")
    rec["meta"]["synth_key"] = "k1"
    with PairsWriter(pairs, set()) as w:
        assert w.write("k1", rec) is True

    assert {p: p.read_bytes() for p in (captures, july)} == before
    assert (pairs / "synth-2026-09.jsonl").exists()
    assert len(list(iter_pairs(pairs))) == 9


def test_two_writers_on_different_shards_still_refuse_the_same_cell(sharded):
    """W8's guarantee, held across the window sharding introduces.

    One lock per corpus (not per shard) plus a catch-up over every shard, so a
    run appending to September sees a key another run just put in August.
    """
    from stylebot.synth import PairsWriter

    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    aug = _rec(synthetic=True, at="2026-08-01T00:00:00+00:00", body="contested cell")
    aug["meta"]["synth_key"] = "shared"
    sep = _rec(synthetic=True, at="2026-09-01T00:00:00+00:00", body="contested cell")
    sep["meta"]["synth_key"] = "shared"

    with PairsWriter(pairs, set()) as a, PairsWriter(pairs, set()) as b:
        assert a.write("shared", aug) is True
        assert b.write("shared", sep) is False


def test_capture_save_appends_to_the_captures_shard(sharded):
    from stylebot import capture

    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    july = (pairs / "synth-2026-07.jsonl").read_bytes()
    written = capture.capture_pairs(
        "The system leverages a robust framework.\n",
        "It works.\n",
        pairs_path=pairs,
        source="post/new.qmd",
        snapped_at="2026-09-03T00:00:00+00:00",
    )
    assert written
    assert (pairs / "synth-2026-07.jsonl").read_bytes() == july
    assert all(
        r["meta"]["source"] != "post/new.qmd"
        for r in read_jsonl(pairs / "synth-2026-07.jsonl")
    )
    assert any(
        r["meta"]["source"] == "post/new.qmd" for r in read_jsonl(pairs / "captures.jsonl")
    )
    # And the archive still lands beside the corpus dir, not inside it.
    assert (data_dir / "captures").is_dir()
    assert not (pairs / "captures").exists()


def test_log_save_replace_rule_rewrites_only_the_captures_shard(sharded):
    from stylebot import capture

    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    capture.capture_pairs(
        "The system leverages a robust framework.\n",
        "It works.\n",
        pairs_path=pairs,
        source="post/new.qmd",
        snapped_at="2026-09-03T00:00:00+00:00",
    )
    july = (pairs / "synth-2026-07.jsonl").read_bytes()
    assert capture.remove_pairs_for_source(pairs, "post/new.qmd") == 1
    assert (pairs / "synth-2026-07.jsonl").read_bytes() == july


# --- every --pairs consumer accepts a directory ----------------------------
#
# One test per CLI surface that takes a corpus. They assert a NUMBER wherever
# one is printed, because "exited 0 on a directory" is satisfied by a reader
# that found nothing.


def test_validate_pairs_accepts_a_directory(sharded):
    data_dir, before = sharded
    res = CliRunner().invoke(ai_style, ["validate-pairs", str(data_dir / "pairs")])
    assert res.exit_code == 0, res.output
    assert "0 problem line(s), 0 duplicate record_id(s)" in res.output


def _role_totals(output):
    """(posts, pairs) summed over the three role lines make-splits/check-splits print."""
    rows = re.findall(r"(\d+) post\(s\), (\d+) pair\(s\)", output)
    return sum(int(a) for a, _ in rows), sum(int(b) for _, b in rows)


def test_make_splits_accepts_a_directory(sharded, tmp_path):
    data_dir, before = sharded
    out = tmp_path / "splits.json"
    res = CliRunner().invoke(
        ai_style,
        ["make-splits", "--pairs", str(data_dir / "pairs"), "--out", str(out), "--seed", "1"],
    )
    assert res.exit_code == 0, res.output
    # Every post and every pair in the corpus, not just the first shard's.
    assert _role_totals(res.output) == (4, len(before))


def test_check_splits_accepts_a_directory(sharded, tmp_path):
    data_dir, _ = sharded
    out = tmp_path / "splits.json"
    CliRunner().invoke(
        ai_style,
        ["make-splits", "--pairs", str(data_dir / "pairs"), "--out", str(out), "--seed", "1"],
    )
    blog = tmp_path / "blog"
    (blog / "post").mkdir(parents=True)
    for i in range(4):
        (blog / "post" / f"p{i}.qmd").write_text("---\ntitle: p\n---\n\nx\n", encoding="utf-8")
    res = CliRunner().invoke(
        ai_style,
        [
            "check-splits",
            "--splits", str(out),
            "--blog-root", str(blog),
            "--pairs", str(data_dir / "pairs"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert _role_totals(res.output) == (4, 8)  # the whole corpus, not one shard


def test_train_dry_run_assembles_the_whole_sharded_corpus(sharded, tmp_path):
    """`train` reaches the corpus through `--data-dir`, so this is the config seam too."""
    from stylebot import train as tr

    data_dir, before = sharded
    result = tr.assemble_training_corpus(config.pairs_path(data_dir), val_frac=0.0)
    assert result.n_source_records == len(before)
    assert len(result.train) == len(before)


def test_the_manifest_corpus_hash_covers_every_shard(sharded):
    data_dir, _ = sharded
    pairs = data_dir / "pairs"
    first = corpus.content_sha256(pairs)
    with (pairs / "synth-2026-07.jsonl").open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(_rec(synthetic=True, body="new"), ensure_ascii=False) + "\n")
    assert corpus.content_sha256(pairs) != first


def test_content_sha_of_a_single_file_is_still_the_file_sha(tmp_path, write_pairs_corpus):
    from stylebot.provenance import file_sha256

    src = write_pairs_corpus(tmp_path / "pairs.jsonl", n_posts=1, chunks=1)
    assert corpus.content_sha256(src) == file_sha256(src)


def test_shard_pairs_migration_is_dry_run_by_default(tmp_path, write_pairs_corpus):
    data_dir = tmp_path / "dd"
    data_dir.mkdir()
    write_pairs_corpus(data_dir / "pairs.jsonl", n_posts=2, chunks=2, n_synth_posts=1)
    res = CliRunner().invoke(data_cli, ["shard-pairs", str(data_dir)])
    assert res.exit_code == 0, res.output
    assert "dry run" in res.output
    assert not (data_dir / "pairs").exists()
    assert (data_dir / "pairs.jsonl").exists()


def test_shard_pairs_migration_removes_the_source_file(tmp_path, write_pairs_corpus):
    data_dir = tmp_path / "dd"
    data_dir.mkdir()
    src = write_pairs_corpus(data_dir / "pairs.jsonl", n_posts=2, chunks=2, n_synth_posts=1)
    before = read_jsonl(src)
    res = CliRunner().invoke(data_cli, ["shard-pairs", str(data_dir), "--apply"])
    assert res.exit_code == 0, res.output
    assert not src.exists()
    assert {record_id(r) for r in iter_pairs(data_dir / "pairs")} == {
        record_id(r) for r in before
    }


def test_shard_pairs_refuses_a_corpus_that_is_already_sharded(sharded):
    data_dir, _ = sharded
    (data_dir / "pairs.jsonl").write_text("", encoding="utf-8")
    res = CliRunner().invoke(data_cli, ["shard-pairs", str(data_dir), "--apply"])
    assert res.exit_code != 0
    assert "already sharded" in res.output


def test_eval_sees_every_shards_record_ids(sharded):
    """`eval`'s resume and orphan-prune both key on this id set."""
    from stylebot.eval import live_record_ids

    data_dir, before = sharded
    assert live_record_ids(data_dir / "pairs") == {record_id(r) for r in before}


def test_train_clf_assembles_from_every_shard(sharded):
    """`train-clf --pairs` reaches the corpus through `assemble_dataset`."""
    from stylebot.classify_train import assemble_dataset

    data_dir, before = sharded
    ds = assemble_dataset(data_dir / "pairs")
    assert len(ds.pair_rows) == len(before)


def test_run_pairs_mode_rewrites_every_shards_records(sharded, tmp_path):
    """`run --pairs` (corpus mode) — the surface that turns an adapter into a number."""
    from stylebot.infer import rewrite_pairs_file

    data_dir, before = sharded
    out = tmp_path / "out.jsonl"
    n = rewrite_pairs_file(
        data_dir / "pairs", out, lambda messages, num_samples=1: ["rewritten"]
    )
    assert n == len(before)
    assert {r["id"] if "id" in r else record_id(r) for r in read_jsonl(out)} == {
        record_id(r) for r in before
    }


def test_prune_stale_accepts_a_directory(sharded, tmp_path):
    """`ai-style data prune-stale`, W12's migration, over the sharded shape."""
    data_dir, before = sharded
    out = tmp_path / "splits.json"
    CliRunner().invoke(
        ai_style,
        ["make-splits", "--pairs", str(data_dir / "pairs"), "--out", str(out), "--seed", "1"],
    )
    res = CliRunner().invoke(
        data_cli,
        [
            "prune-stale",
            str(data_dir / "pairs"),
            "--no-orphans",
            "--splits", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    assert f"{len(before)} total" in res.output


def test_prune_invented_anchors_accepts_a_directory(sharded):
    data_dir, before = sharded
    res = CliRunner().invoke(
        data_cli, ["prune-invented-anchors", str(data_dir / "pairs")]
    )
    assert res.exit_code == 0, res.output
    assert f"records          : {len(before)}" in res.output
