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

from stylebot import config, corpus, draws
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style import main as ai_style
from stylebot.jsonl import read_jsonl
from stylebot.pairs import iter_pairs, record_id, validate_pairs_file
from stylebot.synth import DrawWriter


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


# --- the view and round trip -----------------------------------------------


@pytest.fixture
def sharded(tmp_path, write_pairs_corpus):
    """A 4-post/2-chunk corpus as the identity view at `<data-dir>/pairs/`."""
    data_dir = tmp_path / "dd"
    data_dir.mkdir()
    src = write_pairs_corpus(data_dir / "pairs.jsonl", n_posts=4, chunks=2, n_synth_posts=2)
    before = read_jsonl(src)
    draws.build_view(draws.draws_from_pairs(src), draws.ViewPolicy(), out=data_dir / "pairs")
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


def test_config_prefers_the_shard_directory_once_it_exists(sharded, monkeypatch):
    data_dir, _ = sharded
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(data_dir))
    assert config.pairs_path() == data_dir / "pairs"


def test_config_keeps_the_single_file_when_nothing_migrated(tmp_path, monkeypatch):
    """Sharding is opt-in by existing: an install that never migrates never sees it."""
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(tmp_path))
    assert config.pairs_path() == tmp_path / "pairs.jsonl"


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


# --- writers ---------------------------------------------------------------


def _lifted(data_dir):
    """`sharded`'s corpus lifted into the `draws/` a run appends to."""
    out = data_dir / "draws"
    for draw in draws.draws_from_pairs(data_dir / "pairs"):
        draws.append(out, draw)
    return out


def test_synth_appends_only_to_its_own_month_shard(sharded):
    data_dir, _ = sharded
    drawn = _lifted(data_dir)
    captures = drawn / "captures.jsonl"
    july = drawn / "synth-2026-07.jsonl"
    before = {p: p.read_bytes() for p in (captures, july)}

    rec = _rec(synthetic=True, at="2026-09-04T10:00:00+00:00", body="a september pair")
    rec["meta"]["synth_key"] = "k1"
    with DrawWriter(drawn, set()) as w:
        assert w.write(draws.draw_from_pair(rec)) is True

    assert {p: p.read_bytes() for p in (captures, july)} == before
    assert (drawn / "synth-2026-09.jsonl").exists()
    assert len(list(draws.iter_draws(drawn))) == 9


def test_two_writers_on_different_shards_still_refuse_the_same_cell(sharded):
    """W8's guarantee, held across the window sharding introduces: a catch-up
    over every shard, so a run appending to September sees a key another run
    just put in August. The loser keeps its draw, without the record (D84)."""
    data_dir, _ = sharded
    drawn = _lifted(data_dir)
    aug = _rec(synthetic=True, at="2026-08-01T00:00:00+00:00", body="contested cell")
    aug["meta"]["synth_key"] = "shared"
    sep = _rec(synthetic=True, at="2026-09-01T00:00:00+00:00", body="contested cell")
    sep["meta"]["synth_key"] = "shared"

    with DrawWriter(drawn, set()) as a, DrawWriter(drawn, set()) as b:
        assert a.write(draws.draw_from_pair(aug)) is True
        assert b.write(draws.draw_from_pair(sep)) is False
    assert [d.record for d in draws.iter_draws(drawn)].count(None) == 1


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
    # Nothing rewrites the corpus and nothing appends to it: the save is a draw,
    # routed to the captures shard by `draw_shard_of` (D84, W4 step 5).
    assert (pairs / "synth-2026-07.jsonl").read_bytes() == july
    assert not (data_dir / "draws" / "synth-2026-09.jsonl").exists()
    assert any(
        r["record"]["meta"]["source"] == "post/new.qmd"
        for r in read_jsonl(data_dir / "draws" / "captures.jsonl")
    )
    # And the archive still lands beside the corpus dir, not inside it.
    assert (data_dir / "captures").is_dir()
    assert not (pairs / "captures").exists()


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


def _blog(tmp_path):
    """A tree holding the four posts `sharded` names, for `--blog-root`."""
    blog = tmp_path / "blog"
    (blog / "post").mkdir(parents=True)
    for i in range(4):
        (blog / "post" / f"p{i}.qmd").write_text("---\ntitle: p\n---\n\nx\n", encoding="utf-8")
    return blog


def _splits(data_dir, out):
    CliRunner().invoke(
        ai_style,
        ["make-splits", "--pairs", str(data_dir / "pairs"), "--out", str(out), "--seed", "1"],
    )
    return out


def test_check_splits_accepts_a_directory(sharded, tmp_path):
    data_dir, _ = sharded
    out = _splits(data_dir, tmp_path / "splits.json")
    blog = _blog(tmp_path)
    res = CliRunner().invoke(
        ai_style,
        [
            "check-splits",
            "--splits",
            str(out),
            "--blog-root",
            str(blog),
            "--pairs",
            str(data_dir / "pairs"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert _role_totals(res.output) == (4, 8)  # the whole corpus, not one shard


def test_check_splits_reads_a_view_of_draws(sharded, tmp_path):
    """`--draws` health-checks the append-only layer with no `pairs/` in the
    picture (D84, W4 step 5); naming both flags is refused."""
    data_dir, before = sharded
    out = _splits(data_dir, tmp_path / "splits.json")
    draws_dir = data_dir / "draws"
    for draw in draws.draws_from_pairs(data_dir / "pairs"):
        draws.append(draws_dir, draw)
    args = ["check-splits", "--splits", str(out), "--blog-root", str(_blog(tmp_path))]

    res = CliRunner().invoke(ai_style, [*args, "--draws", str(draws_dir)])
    assert res.exit_code == 0, res.output
    assert _role_totals(res.output) == (4, len(before))

    both = CliRunner().invoke(
        ai_style, [*args, "--draws", str(draws_dir), "--pairs", str(data_dir / "pairs")]
    )
    assert both.exit_code != 0
    assert "--pairs" in both.output and "--draws" in both.output


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


def test_eval_sees_every_shards_record_ids(sharded):
    """`eval`'s resume and orphan-prune both key on this id set."""
    from stylebot.scoring import live_record_ids

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
    n = rewrite_pairs_file(data_dir / "pairs", out, lambda messages, num_samples=1: ["rewritten"])
    assert n == len(before)
    assert {r["id"] if "id" in r else record_id(r) for r in read_jsonl(out)} == {
        record_id(r) for r in before
    }

