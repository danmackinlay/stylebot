"""Tests for the eval harness — no network, no API key, no model downloads.

The judge is injected as a plain fake; the detector defaults to `null_detector`;
Vale is exercised only for graceful degradation when the binary is absent. These
gate the batched score schema (`score_pairs_file`) and the judge-reply parser.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from stylebot import eval as ev
from stylebot import synth
from stylebot.pairs import validate_pairs_file


def fake_judge(prose: str) -> dict:
    # Deterministic verdict; no API. Carries `prose` length so we know it ran.
    return {"score": 4, "rationale": f"looks dan-shaped ({len(prose)} chars)"}


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


def _write_glob_scoped_vale_config(root):
    """A minimal Vale setup whose styles apply ONLY under a `[*.{md,qmd}]` glob.

    Mirrors the blog's `.vale.ini` shape: `BasedOnStyles` lives under a path
    glob, not globally. Vale matches that glob against the *filename*, so the
    rule fires only when the prose reaches Vale as a real `*.md` file — exactly
    the regression this guards (stdin's synthetic filename matched no glob, so
    zero rules ever ran). Returns the `.vale.ini` path.
    """
    style = root / "styles" / "teststyle"
    style.mkdir(parents=True)
    # An `existence` rule that fires on a couple of unmistakable slop tokens.
    (style / "Slop.yml").write_text(
        "extends: existence\n"
        "message: \"Avoid AI-slop token '%s'.\"\n"
        "level: warning\n"
        "ignorecase: true\n"
        "tokens:\n"
        "  - delve\n"
        "  - leverage\n",
        encoding="utf-8",
    )
    ini = root / ".vale.ini"
    ini.write_text(
        "StylesPath = styles\n"
        "MinAlertLevel = suggestion\n"
        "\n"
        "[formats]\n"
        "qmd = md\n"
        "\n"
        "[*.{md,qmd}]\n"
        "BasedOnStyles = teststyle\n",
        encoding="utf-8",
    )
    return ini


def test_vale_score_applies_glob_scoped_styles(tmp_path):
    if shutil.which("vale") is None:
        pytest.skip("vale binary not on PATH")
    ini = _write_glob_scoped_vale_config(tmp_path)
    slop = "In today's world you can leverage this to delve into new opportunities."
    result = ev.vale_score(slop, config_path=ini)
    # The glob-scoped style fired: prose reached Vale as a real `*.md` file, not
    # stdin (whose synthetic filename matches no glob → silently 0 alerts).
    assert result["available"] is True, result
    assert result["alerts"] > 0, result
    # Clean prose with none of the banned tokens stays at zero under the same cfg.
    clean = ev.vale_score("A short, plain sentence about nothing in particular.", config_path=ini)
    assert clean["available"] is True
    assert clean["alerts"] == 0


# --- batched Vale ----------------------------------------------------------
#
# `vale_score` costs a whole process, and a corpus run scores ~2 texts per pair
# (~9k processes on the live corpus), where startup dominates. `vale_score_many`
# hands many `*.md` files to one invocation and splits the one
# `{filename: [alerts]}` map back apart. These pin the contract that makes that
# safe: input order, result-to-input mapping, and every degradation path
# producing a FULL-LENGTH list of `available: False` rather than a short one.

_SLOP = "In today's world you can leverage this to delve into new opportunities."
_CLEAN = "A short, plain sentence about nothing in particular."


def _fake_completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["vale"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_vale_score_many_empty_input_is_empty_output():
    assert ev.vale_score_many([]) == []


def test_vale_score_many_short_circuits_when_vale_is_absent(monkeypatch):
    """No binary => no processes, and still one result per input text."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: None)
    spawned = []
    monkeypatch.setattr(ev.subprocess, "run", lambda *a, **k: spawned.append(a))

    results = ev.vale_score_many([_SLOP, _CLEAN, _SLOP])
    assert spawned == []  # short-circuit: not even one process
    assert len(results) == 3
    assert all(r["available"] is False and r["alerts"] == 0 for r in results)
    assert all("PATH" in r["reason"] for r in results)
    # Distinct objects, not one dict aliased three times — callers mutate these.
    results[0]["by_severity"]["error"] = 99
    assert results[1]["by_severity"]["error"] == 0


@pytest.mark.parametrize(
    ("proc_kwargs", "expect_reason"),
    [
        ({"stdout": "", "stderr": "E100 boom"}, "E100 boom"),  # empty stdout
        ({"stdout": "not json at all"}, "not JSON"),  # non-JSON stdout
        ({"stdout": '{"Code": "E100", "Text": "no config file found"}'}, "no config"),
    ],
)
def test_vale_score_many_degrades_whole_batch_with_the_right_length(
    monkeypatch, proc_kwargs, expect_reason
):
    """A whole-run failure marks EVERY text unavailable — never a short list."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/vale")
    monkeypatch.setattr(ev.subprocess, "run", lambda *a, **k: _fake_completed(**proc_kwargs))

    results = ev.vale_score_many([_SLOP] * 5)
    assert len(results) == 5
    assert all(r["available"] is False for r in results)
    assert all(expect_reason in r["reason"] for r in results)
    assert all(r["by_severity"] == {"error": 0, "warning": 0, "suggestion": 0} for r in results)


def test_vale_score_many_degrades_when_vale_is_unexecutable(monkeypatch):
    """Present on PATH but `subprocess.run` raises OSError — still no exception."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/vale")

    def boom(*a, **k):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(ev.subprocess, "run", boom)
    results = ev.vale_score_many([_SLOP, _CLEAN])
    assert len(results) == 2
    assert all(r["available"] is False and "failed to run" in r["reason"] for r in results)


def test_vale_score_many_preserves_order_across_a_mixed_batch(tmp_path):
    """The join back from vale's filename-keyed map must land on the right text."""
    if shutil.which("vale") is None:
        pytest.skip("vale binary not on PATH")
    ini = _write_glob_scoped_vale_config(tmp_path)
    # Alternating clean / one-token / two-token, plus a duplicate of the slop
    # text to prove two identical inputs cannot collide onto one key.
    texts = [
        _CLEAN,
        "We should delve into it.",
        _SLOP,  # both `leverage` and `delve`
        _CLEAN,
        _SLOP,  # byte-identical to index 2
    ]
    results = ev.vale_score_many(texts, config_path=ini)
    assert len(results) == len(texts)
    assert all(r["available"] is True for r in results), results
    assert [r["alerts"] for r in results] == [0, 1, 2, 0, 2]
    # The alert text landed on the right element, not merely the right count.
    assert results[1]["raw"][0]["Match"].lower() == "delve"
    assert {a["Match"].lower() for a in results[2]["raw"]} == {"leverage", "delve"}


def test_vale_score_many_chunks_across_invocations_without_losing_order(tmp_path, monkeypatch):
    """More texts than fit one argv: several runs, one flat in-order result list."""
    if shutil.which("vale") is None:
        pytest.skip("vale binary not on PATH")
    ini = _write_glob_scoped_vale_config(tmp_path)
    monkeypatch.setattr(ev, "_VALE_FILES_PER_RUN", 3)

    real_run = ev.subprocess.run
    calls = []

    def counting_run(cmd, **kwargs):
        calls.append(cmd)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(ev.subprocess, "run", counting_run)

    # 8 texts, alternating alert counts, over a chunk size of 3.
    texts = [_SLOP if i % 2 else _CLEAN for i in range(8)]
    results = ev.vale_score_many(texts, config_path=ini)
    assert len(calls) == 3  # ceil(8 / 3)
    assert [r["alerts"] for r in results] == [0, 2, 0, 2, 0, 2, 0, 2]


def test_vale_score_agrees_with_a_batch_of_one(tmp_path):
    """`vale_score` is a thin wrapper; the two must not drift."""
    if shutil.which("vale") is None:
        pytest.skip("vale binary not on PATH")
    ini = _write_glob_scoped_vale_config(tmp_path)
    for text in (_SLOP, _CLEAN):
        single = ev.vale_score(text, config_path=ini)
        batched = ev.vale_score_many([text], config_path=ini)[0]
        assert single == batched, text


def test_vale_score_many_writes_md_files_so_glob_scoped_styles_still_fire(monkeypatch):
    """The batch keeps the reason `vale_score` never used stdin: styles scoped
    `[*.{md,qmd}]` match on FILENAME, so every scratch file needs the suffix."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/vale")
    seen: list[list[str]] = []

    def capture(cmd, **kwargs):
        seen.append(list(cmd))
        return _fake_completed(stdout="{}")

    monkeypatch.setattr(ev.subprocess, "run", capture)
    ev.vale_score_many([_SLOP, _CLEAN])
    paths = [arg for arg in seen[0] if arg.endswith(".md")]
    assert len(paths) == 2, seen
    assert len(set(paths)) == 2  # distinct names — no two texts share a key


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


def test_openrouter_judge_records_what_the_call_cost(monkeypatch):
    """The judge is the only signal that costs money per call — Vale and the
    detector are local and free — so the verdict carries its own price tag.
    Without it, 'is the judge worth its spend' is unanswerable after the fact,
    which is how the corpus ended up with 270 scored rows and no judge data."""
    import sys
    import types

    seen = {}

    class _Usage:
        prompt_tokens, completion_tokens, cost = 1200, 40, 0.0731

    class _Resp:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content='{"score": 4}'))]
        usage = _Usage()

    def create(**kw):
        seen.update(kw)
        return _Resp()

    fake_openai = types.SimpleNamespace(
        OpenAI=lambda **kw: types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    verdict = ev.openrouter_judge(api_key="sk-test")("Some prose.")
    assert verdict["score"] == 4
    assert verdict["usage"] == {"prompt_tokens": 1200, "completion_tokens": 40, "cost": 0.0731}
    # OpenRouter only reports cost when the request asks for it.
    assert seen["extra_body"] == {"usage": {"include": True}}


# --- batched, JSONL-native scoring over a pairs.jsonl corpus ----------------


def _make_pairs(data_dir, specs):
    """Build a valid pairs.jsonl via the real synth path with a fake generator.

    `specs`: list of (text, context, strategy). One pair per spec (distinct
    target text -> distinct synth_key), appended to data_dir/pairs.jsonl.
    """
    for i, (text, context, strategy) in enumerate(specs):
        target = synth.Target(
            text=text,
            source=f"post/p{i}.qmd",
            chunk_index=0,
            chunk_total=1,
            context=context or "",
        )
        gen = synth.Generator(name="m", generate=lambda s: "[slop] " + s, strategy=strategy)
        synth.synthesize_pairs([target], data_dir, [gen])
    return data_dir / "pairs.jsonl"


def test_score_pairs_file_emits_id_keyed_scores(tmp_path):
    pairs = _make_pairs(
        tmp_path / "c",
        [
            (
                "A first paragraph of prose, long enough to be scored as a candidate.",
                None,
                "catalogue",
            ),
            ("A second paragraph, also comfortably long enough to be a candidate.", None, "polish"),
        ],
    )
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
    pairs = _make_pairs(
        tmp_path / "c",
        [
            (
                "Body prose that sits beneath the heading and should be scored alone here.",
                "## A Heading",
                "catalogue",
            ),
        ],
    )
    assert validate_pairs_file(pairs) == []  # the heading IS a verbatim prefix in the corpus

    captured: list[str] = []

    def judge(text: str) -> dict:
        captured.append(text)
        return {"score": 3, "rationale": "x"}

    ev.score_pairs_file(pairs, tmp_path / "scores.jsonl", judge=judge)
    # The judge scored the body only — the shared heading was stripped first.
    assert captured and all("## A Heading" not in t for t in captured)


def test_record_id_precedence():
    assert (
        ev.record_id({"meta": {"synth_key": "abc", "capture_id": "cap", "chunk_index": 2}}) == "abc"
    )
    assert ev.record_id({"meta": {"capture_id": "cap", "chunk_index": 2}}) == "cap:2"
    rid = ev.record_id({"meta": {"source": "post/x.qmd", "chunk_index": 1}})
    assert rid == ev.record_id({"meta": {"source": "post/x.qmd", "chunk_index": 1}})  # stable
    assert len(rid) == 16


# --- re-cut ids (the `meta.recut` discriminator) ---------------------------
#
# `capture.recut_archives` reuses the archive's `capture_id` and renumbers
# `chunk_index` from 0, so `capture_id:chunk_index` alone can name materially
# different prose before and after a re-cut. `record_id` folds `meta.recut`
# (the gap policy) into the key for re-cut records ONLY. The end-to-end
# capture->recut regression lives in test_capture.py; these pin the id algebra.

_RECUT_A = {"gap_paras": 2, "gap_chars": 800}
_RECUT_B = {"gap_paras": 4, "gap_chars": 1600}


def test_record_id_golden_shapes_for_non_recut_records():
    """GOLDEN. Pairs with no `meta.recut` keep their id byte-for-byte.

    Load-bearing, not decorative: a committed `scores.jsonl` lives in the blog
    repo and joins on these strings. Changing either shape orphans every score
    for the never-re-cut captures that are most of the corpus, so the literals
    are pinned rather than merely round-tripped.
    """
    assert ev.record_id({"meta": {"capture_id": "13860ebc", "chunk_index": 0}}) == "13860ebc:0"
    # The no-capture_id fallback: sha256("post/x.qmd\x001").hexdigest()[:16].
    assert ev.record_id({"meta": {"source": "post/x.qmd", "chunk_index": 1}}) == "abfe53300435fc98"


def test_record_id_recut_differs_from_the_original_cut():
    """A re-cut record must not reuse the original cut's id at the same
    capture_id:chunk_index — that collision is the whole bug."""
    base = {"meta": {"capture_id": "13860ebc", "chunk_index": 0}}
    recut = {"meta": {"capture_id": "13860ebc", "chunk_index": 0, "recut": _RECUT_A}}
    assert ev.record_id(recut) != ev.record_id(base)
    # …but it is still recognisably the same capture and chunk, so the id stays
    # greppable and `capture_id` remains the leading grouping key.
    assert ev.record_id(recut).startswith("13860ebc:r")
    assert ev.record_id(recut).endswith(":0")


def test_record_id_is_stable_for_the_same_recut_policy():
    """Re-cutting twice at the SAME policy yields the SAME id — a re-cut has to
    stay idempotent end to end, not just inside pairs.jsonl. Key order in the
    stamp must not matter either (the digest is over canonical JSON)."""
    first = ev.record_id({"meta": {"capture_id": "cap", "chunk_index": 1, "recut": _RECUT_A}})
    again = ev.record_id({"meta": {"capture_id": "cap", "chunk_index": 1, "recut": dict(_RECUT_A)}})
    reordered = ev.record_id(
        {
            "meta": {
                "capture_id": "cap",
                "chunk_index": 1,
                "recut": {"gap_chars": 800, "gap_paras": 2},
            },
        }
    )
    assert first == again == reordered


def test_record_id_differs_across_recut_policies():
    """A different gap policy cuts different prose, so it must key differently
    — otherwise re-cutting at a new policy silently inherits the old scores."""
    a = ev.record_id({"meta": {"capture_id": "cap", "chunk_index": 0, "recut": _RECUT_A}})
    b = ev.record_id({"meta": {"capture_id": "cap", "chunk_index": 0, "recut": _RECUT_B}})
    assert a != b


def test_record_id_synthetic_pairs_ignore_recut():
    """Synthetic pairs key off `synth_key` and return before the re-cut branch:
    they are generated, never re-cut, and their committed scores must not move."""
    meta = {"synth_key": "abc", "capture_id": "cap", "chunk_index": 2, "recut": _RECUT_A}
    assert ev.record_id({"meta": meta}) == "abc"


def test_score_pairs_file_resumable(tmp_path):
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A paragraph long enough to be scored as a candidate here today.", None, "polish"),
            ("Another paragraph, also long enough to count as a candidate here.", None, "polish"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    first = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert first.written == 2

    second = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert second.written == 0
    assert second.skipped_existing == 2
    assert len([line for line in out.read_text().splitlines() if line.strip()]) == 2  # did not grow


def test_score_pairs_file_keyless(tmp_path):
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A paragraph long enough to be a candidate for scoring now.", None, "polish"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, judge=None)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["scores"]["slop"]["judge"] is None
    assert rec["scores"]["slop"]["detector"]["score"] is None
    assert "vale" in rec["scores"]["slop"]


def test_score_pairs_file_concurrency_same_ids(tmp_path):
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(6)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    out1, out8 = tmp_path / "s1.jsonl", tmp_path / "s8.jsonl"
    ev.score_pairs_file(pairs, out1, judge=fake_judge, max_workers=1)
    ev.score_pairs_file(pairs, out8, judge=fake_judge, max_workers=8)
    ids1 = {json.loads(line)["id"] for line in out1.read_text().splitlines() if line.strip()}
    ids8 = {json.loads(line)["id"] for line in out8.read_text().splitlines() if line.strip()}
    assert ids1 == ids8 and len(ids1) == 6  # order-free: same ids regardless of worker count


# --- the batched Vale pre-pass inside score_pairs_file ----------------------


def _count_vale_spawns(monkeypatch):
    """Pretend vale exists, count invocations, and report every file clean."""
    monkeypatch.setattr(ev.shutil, "which", lambda name: "/usr/bin/vale")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _fake_completed(stdout="{}")

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    return calls


def test_score_candidate_accepts_a_precomputed_vale_result(monkeypatch):
    """The injection point: `vale=` given => no process, value passed through."""
    calls = _count_vale_spawns(monkeypatch)
    precomputed = {"available": True, "alerts": 7, "by_severity": {}, "raw": []}
    rec = ev.score_candidate("prose", judge=None, vale=precomputed)
    assert rec["vale"] is precomputed
    assert calls == []
    # Default is unchanged for every existing caller: it still scores inline.
    ev.score_candidate("prose", judge=None)
    assert len(calls) == 1


def test_score_pairs_file_batches_vale_into_one_process_per_slice(tmp_path, monkeypatch):
    """6 pairs x 2 fields used to be 12 `vale` processes; it is now one."""
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(6)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    calls = _count_vale_spawns(monkeypatch)

    res = ev.score_pairs_file(pairs, tmp_path / "scores.jsonl", judge=fake_judge, max_workers=4)
    assert res.written == 6
    assert len(calls) == 1, calls
    # One scratch `.md` per (record, field) — 6 pairs x {slop, target}.
    assert len([a for a in calls[0] if a.endswith(".md")]) == 12


def test_score_pairs_file_still_streams_across_batches(tmp_path, monkeypatch):
    """Crash-safety: results reach the file per line, not once at the end.

    The pre-pass slices the corpus rather than swallowing it whole, so a run
    that dies partway leaves everything before the current slice on disk and
    the resume path picks up from there.
    """
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(5)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    monkeypatch.setattr(ev, "_SCORE_BATCH_RECORDS", 2)
    calls = _count_vale_spawns(monkeypatch)

    out = tmp_path / "scores.jsonl"
    # A judge that reads the file back mid-run sees earlier records already there.
    lines_seen: list[int] = []

    def peeking_judge(prose: str) -> dict:
        lines_seen.append(len([ln for ln in out.read_text().splitlines() if ln.strip()]))
        return {"score": 4, "rationale": "x"}

    res = ev.score_pairs_file(pairs, out, judge=peeking_judge, max_workers=1)
    assert res.written == 5
    assert len(calls) == 3  # ceil(5 / 2) slices, one vale run each
    assert max(lines_seen) >= 2  # the file grew during the run, not after it

    # And the run is still resumable from that partial state.
    assert ev.score_pairs_file(pairs, out, judge=fake_judge).skipped_existing == 5


def test_score_pairs_file_survives_a_vale_batch_failure(tmp_path, monkeypatch):
    """Vale is a signal, not a run-ender: a batch blow-up degrades to
    `available: False` on every text and still writes every record."""
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(3)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)

    def exploding(*a, **k):
        raise RuntimeError("vale batch exploded")

    monkeypatch.setattr(ev, "vale_score_many", exploding)

    out = tmp_path / "scores.jsonl"
    res = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert res.written == 3
    assert res.errors == []  # not a per-record failure
    for rec in _read_scores(out):
        for name in ("slop", "target"):
            assert rec["scores"][name]["vale"]["available"] is False
            assert "vale batch failed" in rec["scores"][name]["vale"]["reason"]
            assert rec["scores"][name]["judge"]["score"] == 4  # other signals unharmed


# --- orphan score records --------------------------------------------------
#
# `scores.jsonl` is append-only and joined on `record_id`; a re-cut or a
# regenerated corpus strands the superseded row there. No consumer checks the
# join, so orphans inflate every count and mean and leak phantom rows into the
# committed covariate CSV. `score_pairs_file` prunes at the source.


def _read_scores(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_score_pairs_file_prunes_orphans_and_reports_the_count(tmp_path):
    pairs = _make_pairs(
        tmp_path / "c",
        [
            (
                "A first paragraph of prose, long enough to be scored as a candidate.",
                None,
                "polish",
            ),
            ("A second paragraph, also comfortably long enough to be a candidate.", None, "polish"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, judge=fake_judge)
    survivor, doomed = _read_scores(out)

    # The corpus loses a pair (stand-in for a re-cut / regeneration).
    lines = [ln for ln in pairs.read_text().splitlines() if ln.strip()]
    keep = next(ln for ln in lines if json.loads(ln)["meta"]["synth_key"] == survivor["id"])
    pairs.write_text(keep + "\n", encoding="utf-8")

    res = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert res.pruned_orphans == 1
    remaining = _read_scores(out)
    # The orphan is gone and the live record survived byte-for-byte — pruning
    # must not rescore or reshape anything it keeps.
    assert remaining == [survivor]
    assert doomed["id"] not in {r["id"] for r in remaining}
    # …and every consumer now counts live pairs only, for free.
    assert ev.summarize_scores(out)["fields"]["slop"]["n"] == 1
    assert len(ev.covariate_table(out)[0]) == 1


def test_score_pairs_file_leaves_an_orphan_free_file_alone(tmp_path, monkeypatch):
    """A clean file is not rewritten at all — the prune must not churn a
    committed artifact (or its mtime) on every re-run."""
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A paragraph long enough to be scored as a candidate here today.", None, "polish"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, judge=fake_judge)

    rewrites: list[object] = []
    monkeypatch.setattr(ev, "write_jsonl_atomic", lambda p, r: rewrites.append(p))
    res = ev.score_pairs_file(pairs, out, judge=fake_judge)
    assert res.pruned_orphans == 0
    assert rewrites == []


def test_score_pairs_file_prune_orphans_off_is_the_wrong_corpus_escape_hatch(tmp_path):
    """`--out` pointed at ANOTHER corpus's scores would read as 100% orphaned;
    the flag is the valve, and there is deliberately no threshold heuristic."""
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A paragraph long enough to be scored as a candidate here today.", None, "polish"),
        ],
    )
    other = _make_pairs(
        tmp_path / "d",
        [
            ("A wholly different paragraph, long enough to be a candidate too.", None, "polish"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(other, out, judge=fake_judge)
    before = _read_scores(out)

    res = ev.score_pairs_file(pairs, out, judge=fake_judge, prune_orphans=False)
    assert res.pruned_orphans == 0
    assert _read_scores(out)[: len(before)] == before  # the foreign records stayed


def test_prune_orphan_scores_round_trips_undecodable_lines(tmp_path):
    """A corrupt line has no join key but must never be silently dropped by a
    rewrite (same rule as `jsonl.write_jsonl_atomic`)."""
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A paragraph long enough to be scored as a candidate here today.", None, "polish"),
        ],
    )
    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(pairs, out, judge=fake_judge)
    out.write_text(
        out.read_text() + '{"id": "ghost", "scores": {}}\n' + "not json at all\n",
        encoding="utf-8",
    )

    assert ev.prune_orphan_scores(pairs, out) == 1
    assert "not json at all" in out.read_text()
    assert "ghost" not in out.read_text()


def test_prune_orphan_scores_on_a_missing_scores_file_is_a_noop(tmp_path):
    pairs = _make_pairs(
        tmp_path / "c",
        [
            ("A paragraph long enough to be scored as a candidate here today.", None, "polish"),
        ],
    )
    missing = tmp_path / "nope.jsonl"
    assert ev.prune_orphan_scores(pairs, missing) == 0
    assert not missing.exists()


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
    records = [
        _score_rec("catalogue", 2, 5),
        _score_rec("catalogue", 4, 5),
        _score_rec("polish", 3, 4),
    ]
    summary = ev.summarize_scores(records, by="slop_strategy")
    assert summary["pairs"] == 3
    assert summary["fields"]["slop"]["n"] == 3
    assert summary["fields"]["slop"]["mean_judge_score"] == 3.0  # (2+4+3)/3
    assert summary["by"]["catalogue"]["slop"]["mean_judge_score"] == 3.0  # (2+4)/2
    assert summary["by"]["catalogue"]["target"]["mean_judge_score"] == 5.0
    assert summary["by"]["polish"]["slop"]["mean_judge_score"] == 3.0


def test_carried_gen_meta_flattened_and_faceted(tmp_path):
    # A synthetic pair's meta.gen covariates flatten onto the score record, so
    # summarize_scores(by=…) facets by them for free.
    data_dir = tmp_path / "c"
    t1 = synth.Target(
        text="A paragraph long enough to be scored as a candidate here.",
        source="post/a.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    t2 = synth.Target(
        text="Another paragraph, also comfortably long enough to score.",
        source="post/b.qmd",
        chunk_index=0,
        chunk_total=1,
    )
    g_hi = synth.Generator(
        name="m",
        reasoning_effort="high",
        generate=lambda s: synth.GenOutput("[slop] " + s, {"reasoning_effort": "high"}),
    )
    g_lo = synth.Generator(
        name="m",
        reasoning_effort="low",
        generate=lambda s: synth.GenOutput("[slop] " + s, {"reasoning_effort": "low"}),
    )
    synth.synthesize_pairs([t1], data_dir, [g_hi])
    synth.synthesize_pairs([t2], data_dir, [g_lo])

    out = tmp_path / "scores.jsonl"
    ev.score_pairs_file(data_dir / "pairs.jsonl", out, judge=fake_judge)
    recs = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    # flattened from meta.gen
    assert {r["meta"].get("reasoning_effort") for r in recs} == {"high", "low"}
    by = ev.summarize_scores(out, by="reasoning_effort")["by"]
    assert set(by) == {"high", "low"}  # facets by the gen covariate for free


def test_real_pair_without_gen_buckets_null():
    # Real (Phase-1) pairs have no meta.gen; faceting by a gen covariate buckets
    # them under "null" rather than erroring.
    rec = {
        "id": "r1",
        "meta": {"slop_strategy": None},
        "scores": {
            "slop": {
                "judge": {"score": 3},
                "vale": {"available": False},
                "detector": {"score": None},
            },
            "target": {
                "judge": {"score": 5},
                "vale": {"available": False},
                "detector": {"score": None},
            },
        },
    }
    by = ev.summarize_scores([rec], by="reasoning_effort")["by"]
    assert set(by) == {"null"}


# --- the three pre-passes inside score_pairs_file ---------------------------
#
# One pre-pass per signal per slice: Vale and the detector batch on the calling
# task, the judge is gathered under a semaphore (D57, D59). There is no pool and
# no lock, so every scoring signal below is exercised on one thread.


def _batching_detector(record: list):
    """A detector whose batch path is observable: `record` gets one entry per
    CALL, carrying how many texts that call carried. One entry of 12 is the
    batched pre-pass; twelve entries of 1 is the old batch-of-one shape."""

    def _score(prose: str) -> dict:
        return {"score": min(0.99, len(prose) / 100), "configured": True, "name": "stub"}

    def detect(prose: str) -> dict:
        record.append(1)
        return _score(prose)

    def batch_detect(texts):
        texts = list(texts)
        record.append(len(texts))
        return [_score(t) | {"batched": True} for t in texts]

    detect.identity = {key: None for key in ev.DETECTOR_IDENTITY_KEYS} | {"name": "stub"}
    detect.batch_detect = batch_detect
    return detect


def test_score_candidate_accepts_a_precomputed_detector_result():
    """The injection point: `detector_result=` given => the detector is not called."""
    calls: list[int] = []
    det = _batching_detector(calls)
    precomputed = {"score": 0.25, "configured": True, "name": "stub"}
    rec = ev.score_candidate("prose", judge=None, detector=det, detector_result=precomputed)
    assert rec["detector"] is precomputed
    assert calls == []
    # Default is unchanged for every existing caller: it still scores inline.
    ev.score_candidate("prose", judge=None, detector=det)
    assert calls == [1]


def test_score_pairs_file_batches_the_detector_into_one_call_per_slice(tmp_path, monkeypatch):
    """6 pairs x 2 fields used to be 12 batch-of-one embeds; it is now one call of 12."""
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(6)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    _count_vale_spawns(monkeypatch)
    calls: list[int] = []

    res = ev.score_pairs_file(
        pairs, tmp_path / "scores.jsonl", judge=None, detector=_batching_detector(calls),
        max_workers=4,
    )
    assert res.written == 6
    assert calls == [12], calls  # ONE call carrying every (record, field) text
    lines = (tmp_path / "scores.jsonl").read_text().splitlines()
    rows = [json.loads(ln) for ln in lines if ln.strip()]
    assert all(r["scores"][f]["detector"]["batched"] for r in rows for f in ("slop", "target"))


def test_a_failing_detector_batch_falls_back_to_per_text_scoring(tmp_path, monkeypatch):
    """A signal is never a run-ender: a broken batch path degrades to per text.

    The retry happens on the calling task, not in a pool — the pool is gone
    (D59), and putting the detector back into one was the shape that segfaulted.
    """
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(3)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    _count_vale_spawns(monkeypatch)
    per_text: list[int] = []

    def detect(prose: str) -> dict:
        per_text.append(1)
        return {"score": 0.4, "configured": True, "name": "stub"}

    def exploding_batch(texts):
        raise RuntimeError("the backbone fell over")

    detect.identity = {key: None for key in ev.DETECTOR_IDENTITY_KEYS} | {"name": "stub"}
    detect.batch_detect = exploding_batch

    res = ev.score_pairs_file(pairs, tmp_path / "s.jsonl", judge=None, detector=detect)
    assert res.written == 3 and res.errors == []
    assert len(per_text) == 6  # 3 pairs x 2 fields, one text at a time


def test_a_detector_that_fails_per_text_too_records_the_error_in_the_field(tmp_path, monkeypatch):
    """The last resort: the field says why it has no score, and the record survives."""
    specs = [("A paragraph long enough to score here.", None, "polish")]
    pairs = _make_pairs(tmp_path / "c", specs)
    _count_vale_spawns(monkeypatch)

    def detect(prose: str) -> dict:
        raise RuntimeError("the backbone fell over")

    detect.identity = {key: None for key in ev.DETECTOR_IDENTITY_KEYS} | {"name": "stub"}
    detect.batch_detect = lambda texts: (_ for _ in ()).throw(RuntimeError("and so did the batch"))

    res = ev.score_pairs_file(pairs, tmp_path / "s.jsonl", judge=None, detector=detect)
    assert res.written == 1 and res.errors == []
    row = json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])
    det_field = row["scores"]["slop"]["detector"]
    assert det_field["score"] is None and det_field["name"] == "stub"
    assert "the backbone fell over" in det_field["error"]


def test_an_async_judge_is_awaited_with_max_workers_calls_in_flight(tmp_path, monkeypatch):
    """The judge is the harness's only concurrency, and it is a semaphore (D59)."""
    import asyncio

    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(6)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    _count_vale_spawns(monkeypatch)
    live = 0
    peak = 0

    async def judge(prose: str) -> dict:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)  # a window every other pending call can enter
        live -= 1
        return {"score": 4, "rationale": "fine"}

    res = ev.score_pairs_file(pairs, tmp_path / "s.jsonl", judge=judge, max_workers=3)
    assert res.written == 6 and res.errors == []
    assert peak == 3, f"{peak} judge calls in flight against max_workers=3"
    row = json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])
    assert row["scores"]["slop"]["judge"]["score"] == 4


def test_a_judge_error_fails_its_own_record_and_no_other(tmp_path, monkeypatch):
    """The gathered pass carries an exception back per call, not per slice."""
    specs = [
        (f"Paragraph number {i}, long enough to be scored as a candidate here.", None, "polish")
        for i in range(3)
    ]
    pairs = _make_pairs(tmp_path / "c", specs)
    _count_vale_spawns(monkeypatch)

    async def judge(prose: str) -> dict:
        if "number 1" in prose:
            raise RuntimeError("the judge fell over")
        return {"score": 4, "rationale": "fine"}

    res = ev.score_pairs_file(pairs, tmp_path / "s.jsonl", judge=judge)
    assert res.written == 2
    assert len(res.errors) == 1 and "the judge fell over" in res.errors[0][1]


def test_chunked_detector_batch_path_agrees_with_the_per_passage_path():
    """Batching may not move a number: the two paths share one aggregation."""
    calls: list[int] = []
    inner = _batching_detector(calls)
    det = ev.chunked_detector(inner, max_chunk_chars=120)
    passages = [
        "First paragraph, of some length.\n\nSecond paragraph here.\n\n"
        "Third one, longer than the others by a bit.",
        "# A heading\n\nOnly one prose paragraph under it.",
        "```\ncode only, no prose at all\n```",
    ]
    one_at_a_time = [det(p) for p in passages]
    calls.clear()
    batched = det.batch_detect(passages)
    assert batched == one_at_a_time
    # Every chunk of every passage crossed the backbone in ONE call — and the
    # per-passage path above had spent one call per chunk.
    assert len(calls) == 1
    assert calls[0] == sum(r["n_chunks"] for r in one_at_a_time)
    assert one_at_a_time[2]["score"] is None and one_at_a_time[2]["n_chunks"] == 0
