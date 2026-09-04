"""`synth` writes draws alongside pairs, and the view equals the pairs (D84, W4 step 2).

Every generator call ends in exactly one draw — accepted, gated or errored —
and the identity view over those draws is byte-identical to the `pairs/` the
same run appended to. Keyless: the fakes from `synth_helpers`.
"""

from __future__ import annotations

import json
from pathlib import Path

from synth_helpers import _fake, _targets

from stylebot import config, corpus, draws, pairs, synth, synth_cli

LINK = " See the [docs](./d.qmd) for the rest."


def _metered(name: str, rewrite=lambda text: text) -> synth.Generator:
    """A fake that reports a `gen` block, so the draw's copy can be compared."""
    return synth.Generator(
        name=name,
        generate=lambda text: synth.GenOutput(
            f"[{name}-slop] " + rewrite(text),
            {"model": name, "cost": 0.01, "completion_tokens": 5, "reasoning_text": "hmm"},
        ),
    )


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A sharded corpus dir plus its draws dir; `pairs/` must exist to be chosen."""
    data_dir = tmp_path / "corpus"
    (data_dir / "pairs").mkdir(parents=True)
    return data_dir, data_dir / "draws"


def _view_equals_pairs(data_dir: Path, draws_dir: Path, out: Path) -> draws.ViewReport:
    pairs_dir = config.pairs_path(data_dir)
    report = draws.build_view(draws.iter_draws(draws_dir), draws.ViewPolicy(), out=out)
    assert report.content_sha256 == corpus.content_sha256(pairs_dir)
    assert corpus.content_sha256(out) == corpus.content_sha256(pairs_dir)
    view_ids = {pairs.record_id(r) for r in corpus.iter_records(out)}
    assert view_ids == {pairs.record_id(r) for r in corpus.iter_records(pairs_dir)}
    return report


def _linked(n: int) -> list[synth.Target]:
    import dataclasses

    return [dataclasses.replace(t, text=t.text + LINK) for t in _targets(n)]


# --- (a) accepted draws render the pairs the run wrote --------------------------


def test_view_from_the_runs_draws_equals_the_pairs_it_wrote(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    result = synth.synthesize_pairs(
        _targets(6), data_dir, [_fake("a"), _metered("b")], draws_dir=draws_dir
    )
    assert result.written == 6

    report = _view_equals_pairs(data_dir, draws_dir, tmp_path / "view")
    assert report.records == 6 and report.by_outcome == {draws.ACCEPTED: 6}

    written = list(draws.iter_draws(draws_dir))
    records = corpus.read_records(config.pairs_path(data_dir))
    assert [d.record for d in written] == records
    assert any(d.gen for d in written), "the metered arm's gen block reached its draws"
    for d, r in zip(written, records, strict=True):
        assert d.raw_output == pairs.extract_slop(r) and d.target_text == pairs.extract_target(r)
        assert d.gen == r["meta"].get("gen") and d.source == r["meta"]["source"]
        assert d.span is None and d.drawn_at and d.superseded_by is None
        assert not (d.gen or {}).get("reasoning_text")  # the trace stays in its sidecar
        assert d.blog_sha == r["meta"]["provenance"]["blog_sha"]
        assert len(d.draw_id) == 16 and d.record is not None
    assert {p.name for p in draws_dir.glob("*.jsonl")} == {draws.draw_shard_of(written[0])}


# --- (b) a gated draw keeps its raw output and renders nothing ------------------


def test_a_gated_draw_is_kept_with_its_raw_output_and_no_record(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    calls = {"n": 0}

    def sometimes_delinks(text: str) -> str:
        calls["n"] += 1
        drop = calls["n"] == 2  # the second draw deletes the anchor (D32)
        return text.replace("[docs](./d.qmd)", "docs") if drop else text

    result = synth.synthesize_pairs(
        _linked(3), data_dir, [_metered("flaky", sometimes_delinks)], draws_dir=draws_dir
    )
    assert result.written == 2 and result.skipped_invented_anchors == 1

    report = _view_equals_pairs(data_dir, draws_dir, tmp_path / "view")
    assert report.by_outcome == {draws.ACCEPTED: 2, "gate:skipped_invented_anchors": 1}
    assert report.excluded == {"outcome": 1}
    gated = [d for d in draws.iter_draws(draws_dir) if d.outcome.startswith("gate:")]
    assert len(gated) == 1
    raw = gated[0].raw_output
    assert raw and "docs" in raw and "./d.qmd" not in raw
    assert gated[0].record is None and gated[0].gen["model"] == "flaky"
    # The sidecar keeps being written until step 4 deletes it.
    rows = [json.loads(x) for x in (data_dir / "lost_draws.jsonl").read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["gate"]


# --- (c) an errored draw is one `error:` draw (C14) -----------------------------


def test_an_errored_draw_is_one_error_draw(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)

    def broken(text: str) -> str:
        raise RuntimeError("provider on fire")

    arm = synth.Generator(name="broken", generate=broken)
    result = synth.synthesize_pairs(_targets(1), data_dir, [arm], draws_dir=draws_dir)
    assert result.written == 0 and len(result.errors) == 1

    written = list(draws.iter_draws(draws_dir))
    assert [d.outcome for d in written] == ["error:RuntimeError"]
    assert written[0].raw_output is None and written[0].record is None
    assert written[0].target_text == _targets(1)[0].text
    report = draws.build_view(written, draws.ViewPolicy(), out=tmp_path / "view")
    assert report.records == 0 and report.excluded == {"outcome": 1}


# --- (d) resume appends no duplicate draws --------------------------------------


def test_a_rerun_appends_no_draws_for_cells_it_skips(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    targets = _targets(4)
    synth.synthesize_pairs(targets, data_dir, [_fake("a")], draws_dir=draws_dir)
    before = list(draws.iter_draws(draws_dir))
    assert len(before) == 4

    again = synth.synthesize_pairs(targets, data_dir, [_fake("a")], draws_dir=draws_dir)
    assert again.written == 0 and again.skipped_existing == 4
    covered = synth.synthesize_pairs(
        targets, data_dir, [_fake("b")], draws_dir=draws_dir, skip_covered=True
    )
    assert covered.written == 0 and covered.skipped_covered == 4
    assert list(draws.iter_draws(draws_dir)) == before
    _view_equals_pairs(data_dir, draws_dir, tmp_path / "view")


# --- (e) without draws_dir nothing is written -----------------------------------


def test_without_a_draws_dir_no_draw_is_written(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    result = synth.synthesize_pairs(_targets(2), data_dir, [_fake("a")])
    assert result.written == 2
    assert not draws_dir.exists()
    visible = sorted(p.name for p in data_dir.iterdir() if not p.name.startswith("."))
    assert visible == ["pairs", "prompts.jsonl"]


# --- the CLI: `--draws`, defaulting to `<data-dir>/draws` ------------------------


def test_run_synth_defaults_draws_to_the_data_dirs_draws_dir(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    synth_cli.run_synth(_targets(2), data_dir=data_dir, generators=[_fake("a")])
    assert len(list(draws.iter_draws(draws_dir))) == 2
    _view_equals_pairs(data_dir, draws_dir, tmp_path / "view")

    elsewhere = tmp_path / "elsewhere"
    synth_cli.run_synth(
        _targets(4),
        data_dir=data_dir,
        generators=[_fake("a")],
        corpus=synth_cli.CorpusOpts(draws_dir=elsewhere),
    )
    assert len(list(draws.iter_draws(elsewhere))) == 2  # only the two new cells
    assert len(list(draws.iter_draws(draws_dir))) == 2


def test_dry_run_writes_no_draws(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    synth_cli.run_synth(
        _targets(2), data_dir=data_dir, generators=[_fake("a")], gen=synth_cli.GenOpts(dry_run=True)
    )
    assert not draws_dir.exists()
