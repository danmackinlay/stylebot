"""`synth` writes draws, and their view is the corpus (D84, W4 steps 2 and 5).

Every generator call ends in exactly one draw — accepted, gated or errored —
and the identity view over those draws is what a reader sees; the run writes no
`pairs.jsonl` and no `pairs/`. Keyless: the fakes from `synth_helpers`.
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
    """A data dir and the draws dir a run appends to; neither exists yet."""
    data_dir = tmp_path / "corpus"
    return data_dir, data_dir / "draws"


def _viewed(draws_dir: Path, out: Path) -> draws.ViewReport:
    """Build the identity view and check it renders exactly the records the
    accepted draws carry — the corpus a reader now sees (W4 step 5)."""
    report = draws.build_view(draws.iter_draws(draws_dir), draws.ViewPolicy(), out=out)
    kept = [d.record for d in draws.iter_draws(draws_dir) if d.record is not None]
    assert corpus.read_records(out) == kept
    assert corpus.content_sha256(out) == report.content_sha256
    return report


def _linked(n: int) -> list[synth.Target]:
    import dataclasses

    return [dataclasses.replace(t, text=t.text + LINK) for t in _targets(n)]


# --- (a) accepted draws render the pairs the run wrote --------------------------


def test_the_view_of_a_runs_draws_is_the_corpus_it_wrote(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    result = synth.synthesize_pairs(
        _targets(6), data_dir, [_fake("a"), _metered("b")], draws_dir=draws_dir
    )
    assert result.written == 6
    assert not config.pairs_path(data_dir).exists()

    view = tmp_path / "view"
    report = _viewed(draws_dir, view)
    assert report.records == 6 and report.by_outcome == {draws.ACCEPTED: 6}

    written = list(draws.iter_draws(draws_dir))
    records = corpus.read_records(view)
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

    report = _viewed(draws_dir, tmp_path / "view")
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
    _viewed(draws_dir, tmp_path / "view")


# --- (e) without draws_dir the draws land under the data dir --------------------


def test_without_a_draws_dir_the_draws_land_under_the_data_dir(tmp_path):
    """No `draws_dir` is no longer "write nothing" (W4 step 5): it is
    `<data-dir>/draws`, and the run leaves nothing beside it but its sidecar."""
    data_dir, draws_dir = _corpus(tmp_path)
    result = synth.synthesize_pairs(_targets(2), data_dir, [_fake("a")])
    assert result.written == 2
    assert len(list(draws.iter_draws(draws_dir))) == 2
    visible = sorted(p.name for p in data_dir.iterdir() if not p.name.startswith("."))
    assert visible == ["draws", "prompts.jsonl"]


# --- the CLI: `--draws`, defaulting to `<data-dir>/draws` ------------------------


def test_run_synth_defaults_draws_to_the_data_dirs_draws_dir(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    synth_cli.run_synth(_targets(2), data_dir=data_dir, generators=[_fake("a")])
    assert len(list(draws.iter_draws(draws_dir))) == 2
    _viewed(draws_dir, tmp_path / "view")

    elsewhere = tmp_path / "elsewhere"
    synth_cli.run_synth(
        _targets(4),
        data_dir=data_dir,
        generators=[_fake("a")],
        corpus=synth_cli.CorpusOpts(draws_dir=elsewhere),
    )
    # `--draws` names the CORPUS now, so resume reads that one: all four cells
    # are missing from an empty dir, and the default dir is left alone.
    assert len(list(draws.iter_draws(elsewhere))) == 4
    assert len(list(draws.iter_draws(draws_dir))) == 2


def test_dry_run_writes_no_draws(tmp_path):
    data_dir, draws_dir = _corpus(tmp_path)
    synth_cli.run_synth(
        _targets(2), data_dir=data_dir, generators=[_fake("a")], gen=synth_cli.GenOpts(dry_run=True)
    )
    assert not draws_dir.exists()


# --- (f) the run writes draws and nothing else (W4 step 5) ----------------------


def test_a_run_writes_draws_and_no_pairs(tmp_path):
    """`synth` appends draws only: the corpus a reader sees is their view, and
    no `pairs.jsonl` or `pairs/` shard is written at all (D84)."""
    data_dir = tmp_path / "corpus"
    result = synth.synthesize_pairs(_targets(3), data_dir, [_fake("a")])
    assert result.written == 3
    assert not (data_dir / "pairs.jsonl").exists() and not (data_dir / "pairs").exists()
    got = list(draws.iter_draws(data_dir / "draws"))
    assert len(got) == 3 and all(d.record is not None for d in got)
    assert draws.build_view(got, draws.ViewPolicy()).records == 3


def test_a_rerun_resumes_off_the_draws_with_no_pairs_file(tmp_path):
    """Resume and coverage are reads of the draws, not of a `pairs.jsonl`."""
    data_dir = tmp_path / "corpus"
    targets = _targets(4)
    synth.synthesize_pairs(targets, data_dir, [_fake("a")])
    again = synth.synthesize_pairs(targets, data_dir, [_fake("a")])
    assert again.written == 0 and again.skipped_existing == 4
    covered = synth.synthesize_pairs(targets, data_dir, [_fake("b")], skip_covered=True)
    assert covered.written == 0 and covered.skipped_covered == 4
    assert len(list(draws.iter_draws(data_dir / "draws"))) == 4
