"""Tests for the eval harness — no network, no API key, no model downloads.

The judge is injected as a plain fake; the detector defaults to `null_detector`;
Vale is exercised only for graceful degradation when the binary is absent. These
gate the stable JSON schema `evaluate_groups` emits and the judge-reply parser.
"""

from __future__ import annotations

import json
import shutil

from stylebot import eval as ev
from stylebot import synth
from stylebot.pairs import validate_pairs_file


def fake_judge(prose: str) -> dict:
    # Deterministic verdict; no API. Carries `prose` length so we know it ran.
    return {"score": 4, "rationale": f"looks dan-shaped ({len(prose)} chars)"}


# Tiny inline 3-group fixture mirroring the plan's styler-input / styler-output
# / pure-Dan triad.
GROUPS = {
    "styler-input": [
        "It is worth noting that this leverages a robust, scalable solution.",
        "In today's fast-paced world, you really need to think outside the box.",
    ],
    "styler-output": [
        "This uses a solution that scales.",
        "You need a fresh idea.",
    ],
    "pure-Dan": [
        "The abstraction is fine until it isn't, and then it is a liability.",
        "I distrust any pipeline that lies to the model downstream of it.",
    ],
}


def test_evaluate_groups_schema_and_aggregates():
    report = ev.evaluate_groups(GROUPS, judge=fake_judge)

    assert report["schema_version"] == 2
    groups = report["groups"]

    # Both contrast groups present (the movement comparison the plan wants).
    assert "pure-Dan" in groups
    assert "styler-input" in groups
    assert "styler-output" in groups

    for name, expected_n in (("styler-input", 2), ("styler-output", 2), ("pure-Dan", 2)):
        agg = groups[name]["aggregate"]
        assert agg["n"] == expected_n
        # fake_judge always scores 4.
        assert agg["mean_judge_score"] == 4.0
        # null detector contributes no score.
        assert agg["mean_detector_score"] is None
        # Raw per-candidate records travel with the aggregate.
        assert len(groups[name]["records"]) == expected_n


def test_score_candidate_without_judge_runs_keyless():
    rec = ev.score_candidate("Some candidate prose.", judge=None)
    assert rec["judge"] is None
    # Vale + detector keys are always present even with no key / no judge.
    assert "vale" in rec
    assert "detector" in rec
    assert rec["detector"]["score"] is None
    assert rec["text_chars"] == len("Some candidate prose.")
    assert rec["eyeball"] is None


def test_score_candidate_with_fake_judge_and_eyeball():
    rec = ev.score_candidate("Dan-shaped prose.", judge=fake_judge, eyeball="ship it")
    assert rec["judge"]["score"] == 4
    assert rec["eyeball"] == "ship it"


def test_null_detector_is_unconfigured():
    out = ev.null_detector("anything at all")
    assert out["score"] is None
    assert out["configured"] is False
    assert out["name"] == "null"


def test_vale_score_degrades_gracefully_when_absent():
    result = ev.vale_score("This is a test sentence.")
    if shutil.which("vale") is None:
        assert result["available"] is False
        assert result["alerts"] == 0
    else:
        # Vale is installed but we pass no config — it must still not raise and
        # must return the documented keys regardless of availability.
        assert "available" in result
        assert "by_severity" in result
        assert isinstance(result["alerts"], int)


def test_judge_reply_parser_extracts_score_from_surrounding_prose():
    reply = (
        "Sure! Here is my assessment of the passage.\n"
        '{"score": 2, "rationale": "hedgy and generic"}\n'
        "Let me know if you need more detail."
    )
    verdict = ev.parse_judge_reply(reply)
    assert verdict["score"] == 2
    assert verdict["rationale"] == "hedgy and generic"


def test_judge_reply_parser_clamps_and_falls_back():
    # Out-of-range score is clamped into 1-5.
    assert ev.parse_judge_reply('{"score": 9, "rationale": "x"}')["score"] == 5
    # No JSON object, but a bare digit is still recoverable.
    assert ev.parse_judge_reply("I would rate this a 3 overall.")["score"] == 3


def test_read_prose_files(tmp_path):
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    p1.write_text("alpha", encoding="utf-8")
    p2.write_text("beta", encoding="utf-8")
    assert ev.read_prose_files([p1, p2]) == ["alpha", "beta"]


# --- batched, JSONL-native scoring over a pairs.jsonl corpus ----------------


def _make_pairs(data_dir, specs):
    """Build a valid pairs.jsonl via the real synth path with a fake generator.

    `specs`: list of (text, context, strategy). One pair per spec (distinct
    target text -> distinct synth_key), appended to data_dir/pairs.jsonl.
    """
    for i, (text, context, strategy) in enumerate(specs):
        target = synth.Target(text=text, source=f"post/p{i}.qmd", chunk_index=0, chunk_total=1, context=context or "")
        gen = synth.Generator(name="m", generate=lambda s: "[slop] " + s, strategy=strategy)
        synth.synthesize_pairs([target], data_dir, [gen])
    return data_dir / "pairs.jsonl"


def test_score_pairs_file_emits_id_keyed_scores(tmp_path):
    pairs = _make_pairs(tmp_path / "c", [
        ("A first paragraph of prose, long enough to be scored as a candidate.", None, "catalogue"),
        ("A second paragraph, also comfortably long enough to be a candidate.", None, "polish"),
    ])
    out = tmp_path / "scores.jsonl"
    res = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert res.written == 2

    recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(recs) == 2
    for r in recs:
        assert r["id"]  # id-keyed, joinable back to the corpus
        assert set(r["scores"]) == {"slop", "target"}
        assert r["scores"]["slop"]["judge"]["score"] == 4
        assert r["scores"]["target"]["judge"]["score"] == 4
        assert r["meta"]["generator"] == "m"  # carried meta subset


def test_score_pairs_file_strips_heading_context(tmp_path):
    pairs = _make_pairs(tmp_path / "c", [
        ("Body prose that sits beneath the heading and should be scored alone here.", "## A Heading", "catalogue"),
    ])
    assert validate_pairs_file(pairs) == []  # the heading IS a verbatim prefix in the corpus

    captured: list[str] = []

    def judge(text: str) -> dict:
        captured.append(text)
        return {"score": 3, "rationale": "x"}

    ev.score_pairs_file(pairs, tmp_path / "scores.jsonl", judge=judge)
    # The judge scored the body only — the shared heading was stripped first.
    assert captured and all("## A Heading" not in t for t in captured)


def test_record_id_precedence():
    assert ev.record_id({"meta": {"synth_key": "abc", "capture_id": "cap", "chunk_index": 2}}) == "abc"
    assert ev.record_id({"meta": {"capture_id": "cap", "chunk_index": 2}}) == "cap:2"
    rid = ev.record_id({"meta": {"source": "post/x.qmd", "chunk_index": 1}})
    assert rid == ev.record_id({"meta": {"source": "post/x.qmd", "chunk_index": 1}})  # stable
    assert len(rid) == 16


def test_score_pairs_file_resumable(tmp_path):
    pairs = _make_pairs(tmp_path / "c", [
        ("A paragraph long enough to be scored as a candidate here today.", None, "polish"),
        ("Another paragraph, also long enough to count as a candidate here.", None, "polish"),
    ])
    out = tmp_path / "scores.jsonl"
    first = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert first.written == 2

    second = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert second.written == 0
    assert second.skipped_existing == 2
    assert len([line for line in out.read_text().splitlines() if line.strip()]) == 2  # did not grow


def test_score_pairs_file_keyless(tmp_path):
    pairs = _make_pairs(tmp_path / "c", [
        ("A paragraph long enough to be a candidate for scoring now.", None, "polish"),
    ])
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, judge=None)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["scores"]["slop"]["judge"] is None
    assert rec["scores"]["slop"]["detector"]["score"] is None
    assert "vale" in rec["scores"]["slop"]


def test_score_pairs_file_concurrency_same_ids(tmp_path):
    specs = [(f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish") for i in range(6)]
    pairs = _make_pairs(tmp_path / "c", specs)
    out1, out8 = tmp_path / "s1.jsonl", tmp_path / "s8.jsonl"
    ev.score_pairs_file(pairs, out1, judge=fake_judge, max_workers=1)
    ev.score_pairs_file(pairs, out8, judge=fake_judge, max_workers=8)
    ids1 = {json.loads(line)["id"] for line in out1.read_text().splitlines() if line.strip()}
    ids8 = {json.loads(line)["id"] for line in out8.read_text().splitlines() if line.strip()}
    assert ids1 == ids8 and len(ids1) == 6  # order-free: same ids regardless of worker count


def _score_rec(strategy, slop_score, target_score):
    none_vale = {"available": False}
    none_det = {"score": None}
    return {
        "id": f"{strategy}-{slop_score}",
        "meta": {"slop_strategy": strategy},
        "scores": {
            "slop": {"judge": {"score": slop_score}, "vale": none_vale, "detector": none_det},
            "target": {"judge": {"score": target_score}, "vale": none_vale, "detector": none_det},
        },
    }


def test_summarize_scores_by_facet():
    records = [_score_rec("catalogue", 2, 5), _score_rec("catalogue", 4, 5), _score_rec("polish", 3, 4)]
    summary = ev.summarize_scores(records, by="slop_strategy")
    assert summary["pairs"] == 3
    assert summary["fields"]["slop"]["n"] == 3
    assert summary["fields"]["slop"]["mean_judge_score"] == 3.0  # (2+4+3)/3
    assert summary["by"]["catalogue"]["slop"]["mean_judge_score"] == 3.0  # (2+4)/2
    assert summary["by"]["catalogue"]["target"]["mean_judge_score"] == 5.0
    assert summary["by"]["polish"]["slop"]["mean_judge_score"] == 3.0
