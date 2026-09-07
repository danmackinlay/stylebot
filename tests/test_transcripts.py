"""`stylebot.transcripts`: parse, replay, and mine a fixture session end to end
against a `git init` blog (D89). No real transcripts, no network."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner
from conftest import corpus_records

from stylebot import transcripts as tr
from stylebot.bin.ai_style_log import main as log_cli
from stylebot.pairs import validate_pair_record

BASE = (
    "---\ntitle: T\nautomation: 0\n---\n\n## Section\n\n"
    "The first paragraph states the point plainly and says what follows from it.\n\n"
    "The second paragraph is one Claude never touches, and Dan later edits it anyway.\n\n"
    "The third paragraph will be rewritten twice in one session by Claude.\n"
)
CLAUDE_P1 = (
    "One might expect the first paragraph to state the point plainly, yet it argues it, "
    "which is why what follows from it is deferred."
)
CLAUDE_P3_A = "The third paragraph, rewritten once, now carries a subordinate clause."
CLAUDE_P3_B = "The third paragraph, rewritten twice, carries a subordinate clause, which is why."
OBJECTION = " ".join(["Why does the point need arguing at all when the passage states it?"] * 3)


def _rec(kind: str, ts: str, content, *, cwd: str, sidechain=False, result=None, uuid="x"):
    rec = {
        "type": kind,
        "timestamp": ts,
        "sessionId": "sess",
        "isSidechain": sidechain,
        "cwd": cwd,
        "uuid": uuid,
        "message": {"role": kind, "content": content},
    }
    if result is not None:
        rec["toolUseResult"] = result
    return rec


def _session_lines(cwd: str, *, sidechain_edit: bool = True) -> list[dict]:
    """A Read, two chained Edits on one file, a sidechain Edit, one substantive
    user turn and one 'please continue'."""
    fp = f"{cwd}/post/t.qmd"
    n = len(BASE.splitlines())
    return [
        _rec("user", "2026-09-01T10:00:00Z", OBJECTION, cwd=cwd),
        _rec("user", "2026-09-01T10:00:01Z", "please continue", cwd=cwd),
        _rec(
            "assistant",
            "2026-09-01T10:00:02Z",
            [
                {"type": "thinking", "thinking": "I should read the file first."},
                {"type": "tool_use", "id": "r1", "name": "Read", "input": {"file_path": fp}},
            ],
            cwd=cwd,
        ),
        _rec(
            "user",
            "2026-09-01T10:00:03Z",
            [{"type": "tool_result", "tool_use_id": "r1", "content": "1\t---"}],
            cwd=cwd,
            result={
                "type": "text",
                "file": {
                    "filePath": fp,
                    "content": BASE,
                    "numLines": n,
                    "startLine": 1,
                    "totalLines": n,
                },
            },
        ),
        _rec(
            "assistant",
            "2026-09-01T10:00:04Z",
            [
                {"type": "text", "text": "I will argue the first paragraph."},
                {
                    "type": "tool_use",
                    "id": "e1",
                    "name": "Edit",
                    "input": {
                        "file_path": fp,
                        "old_string": "The first paragraph states the point plainly and says what "
                        "follows from it.",
                        "new_string": CLAUDE_P1,
                        "replace_all": False,
                    },
                },
            ],
            cwd=cwd,
        ),
        _rec(
            "user",
            "2026-09-01T10:00:05Z",
            [{"type": "tool_result", "tool_use_id": "e1", "content": "ok"}],
            cwd=cwd,
            result={"filePath": fp, "originalFile": BASE, "userModified": False},
        ),
        _rec(
            "assistant",
            "2026-09-01T10:00:06Z",
            [
                {
                    "type": "tool_use",
                    "id": "e2",
                    "name": "Edit",
                    "input": {
                        "file_path": fp,
                        "old_string": "The third paragraph will be rewritten twice in one session "
                        "by Claude.",
                        "new_string": CLAUDE_P3_A,
                    },
                }
            ],
            cwd=cwd,
        ),
        _rec(
            "user",
            "2026-09-01T10:00:07Z",
            [{"type": "tool_result", "tool_use_id": "e2", "content": "ok"}],
            cwd=cwd,
            result={"filePath": fp, "originalFile": None},
        ),
        _rec(
            "assistant",
            "2026-09-01T10:00:08Z",
            [
                {
                    "type": "tool_use",
                    "id": "e3",
                    "name": "Edit",
                    "input": {
                        "file_path": fp,
                        "old_string": CLAUDE_P3_A,
                        "new_string": CLAUDE_P3_B,
                    },
                }
            ],
            cwd=cwd,
            sidechain=sidechain_edit,
        ),
        _rec(
            "user",
            "2026-09-01T10:00:09Z",
            [{"type": "tool_result", "tool_use_id": "e3", "content": "ok"}],
            cwd=cwd,
            sidechain=sidechain_edit,
        ),
        {"type": "attachment", "sessionId": "sess"},
    ]


def _write_session(tmp_path: Path, cwd: str, name: str = "sess.jsonl", **kw) -> Path:
    tdir = tmp_path / "transcripts"
    tdir.mkdir(exist_ok=True)
    path = tdir / name
    path.write_text("".join(json.dumps(r) + "\n" for r in _session_lines(cwd, **kw)))
    return path


def test_parse_session_resolves_the_chain_and_filters_turns(tmp_path):
    session = tr.parse_session(_write_session(tmp_path, "/blog"))
    assert session.cwd == "/blog" and session.session_id == "sess"
    assert [e.path for e in session.edits] == ["post/t.qmd"] * 3
    assert [e.sidechain for e in session.edits] == [False, False, True]
    assert session.reads["post/t.qmd"][0][1] == BASE
    assert session.base_read("post/t.qmd", "2026-09-01T10:00:04Z") == BASE
    assert session.base_read("post/t.qmd", "2026-09-01T09:00:00Z") is None
    # The objection survives, "please continue" does not; Claude's text and thinking ride along.
    assert session.user_turns == [OBJECTION]
    assert session.edits[0].claude_text == ("I will argue the first paragraph.",)
    assert session.edits[0].original_file == BASE and session.edits[1].original_file is None
    assert session.edits[2].claude_text == ()


def test_replay_applies_in_order_and_names_the_first_failure(tmp_path):
    session = tr.parse_session(_write_session(tmp_path, "/blog"))
    text, failed = tr.replay(BASE, session.edits)
    assert failed is None and CLAUDE_P1 in text and CLAUDE_P3_B in text and CLAUDE_P3_A not in text
    broken = [session.edits[0], tr.Edit("post/t.qmd", "t", "Edit", old="never there", new="x")]
    text, failed = tr.replay(BASE, broken)
    assert failed == 1 and CLAUDE_P1 in text
    text, failed = tr.replay(BASE, [tr.Edit("post/t.qmd", "t", "Write", new="whole")])
    assert (text, failed) == ("whole", None)


def _git(root: Path, *args: str, env_date: str | None = None) -> str:
    env = {"GIT_AUTHOR_DATE": env_date, "GIT_COMMITTER_DATE": env_date} if env_date else {}
    import os

    out = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            **env,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    return out.stdout


@pytest.fixture
def blog(tmp_path):
    """A git blog whose base commit predates the session and whose second
    commit is Dan's edit after it: P1 lightly restyled, P2 (untouched by
    Claude) edited too, P3 accepted verbatim."""
    root = tmp_path / "blog"
    (root / "post").mkdir(parents=True)
    _git(root, "init", "-q")
    (root / "post" / "t.qmd").write_text(BASE, encoding="utf-8")
    _git(root, "add", "post/t.qmd")
    _git(root, "commit", "-q", "-m", "base", env_date="2026-08-30T12:00:00Z")
    dan = (
        BASE.replace(
            "The first paragraph states the point plainly and says what follows from it.",
            "The first paragraph states the point plainly, and what follows from it follows.",
        )
        .replace(
            "The second paragraph is one Claude never touches, and Dan later edits it anyway.",
            "The second paragraph, which Claude never touched, is edited by Dan anyway.",
        )
        .replace(
            "The third paragraph will be rewritten twice in one session by Claude.", CLAUDE_P3_B
        )
    )
    (root / "post" / "t.qmd").write_text(dan, encoding="utf-8")
    _git(root, "add", "post/t.qmd")
    _git(root, "commit", "-q", "-m", "dan", env_date="2026-09-02T12:00:00Z")
    return root


def test_mine_pairs_only_what_claude_touched_and_dan_changed(tmp_path, blog):
    tdir = _write_session(tmp_path, str(blog)).parent
    data_dir = tmp_path / "corpus"
    report = tr.mine_transcripts(tdir, blog_root=blog, data_dir=data_dir)
    assert (report.sessions, report.files, report.base_read, report.windows, report.pairs) == (
        1,
        1,
        1,
        1,
        1,
    )
    assert report.novel_after and not data_dir.exists()  # dry run writes nothing

    report = tr.mine_transcripts(tdir, blog_root=blog, data_dir=data_dir, dry_run=False)
    assert len(report.written) == 1
    records = corpus_records(data_dir)
    assert len(records) == 1
    rec = records[0]
    assert validate_pair_record(rec) == []
    # P1: Claude's version against Dan's; P2 (Dan-only) and P3 (accepted) are not pairs.
    assert CLAUDE_P1 in rec["messages"][1]["content"]
    assert "follows from it follows" in rec["messages"][2]["content"]
    assert "Claude never touched" not in rec["messages"][2]["content"]
    m = rec["meta"]
    assert m["source"] == "post/t.qmd" and m["snapped_at"] == "2026-09-01T10:00:08Z"
    assert m["mined"]["session_id"] == "sess" and m["mined"]["base"] == "read"
    assert m["mined"]["sidechain"] is True and m["mined"]["session_edits"] == 3
    assert m["mined"]["after_sha"] and 0.0 <= m["novel_after"] <= 1.0
    turns = [json.loads(line) for line in (data_dir / tr.TURNS_FILE).read_text().splitlines()]
    assert turns[0]["capture_id"] == m["capture_id"] and turns[0]["user"] == [OBJECTION]
    assert turns[0]["claude_thinking"] == ["I should read the file first."]

    # A second apply is a no-op: the session is recorded as mined.
    again = tr.mine_transcripts(tdir, blog_root=blog, data_dir=data_dir, dry_run=False)
    assert again.sessions_mined_before == 1 and again.written == []
    assert len(corpus_records(data_dir)) == 1


def test_mine_falls_back_to_git_and_skips_a_contested_window(tmp_path, blog):
    lines = _session_lines(str(blog))
    no_read = [r for r in lines if not (r.get("toolUseResult") or {}).get("file")]
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    (tdir / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in no_read))
    report = tr.mine_transcripts(tdir, blog_root=blog, data_dir=tmp_path / "c")
    assert (report.base_read, report.base_git, report.pairs) == (0, 1, 1)
    # Another session edits the same file between this one and Dan's commit: skipped.
    later = [
        dict(r, timestamp=r["timestamp"].replace("T10:", "T11:"), sessionId="b")
        for r in lines
        if "timestamp" in r
    ]
    (tdir / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in later))
    report = tr.mine_transcripts(tdir, blog_root=blog, data_dir=tmp_path / "c")
    assert report.window_skipped >= 1
    assert (
        tr.mine_transcripts(tdir, blog_root=blog, data_dir=tmp_path / "c", since="2026-09-05").files
        == 0
    )


def test_log_mine_cli_is_dry_run_by_default(tmp_path, blog):
    tdir = _write_session(tmp_path, str(blog)).parent
    data_dir = tmp_path / "corpus"
    args = [
        "mine",
        "--transcripts",
        str(tdir),
        "--blog-root",
        str(blog),
        "--data-dir",
        str(data_dir),
    ]
    result = CliRunner().invoke(log_cli, args)
    assert result.exit_code == 0, result.output
    assert "dry run" in result.output and "pairs" in result.output and not data_dir.exists()
    result = CliRunner().invoke(log_cli, [*args, "--apply", "--no-turns"])
    assert result.exit_code == 0, result.output
    assert "wrote 1 pair(s)" in result.output and not (data_dir / tr.TURNS_FILE).exists()
    result = CliRunner().invoke(
        log_cli,
        [
            "mine",
            "--transcripts",
            str(tmp_path),
            "--blog-root",
            str(tmp_path),
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 2 and "git checkout" in result.output


def test_a_hook_copy_is_the_before_state_and_nothing_replays(tmp_path, blog):
    """`--claude-left <dir>/<session>/<path>` wins over Read and git: the copy
    is Claude's final state even when the transcript's edits would not apply."""
    tdir = _write_session(tmp_path, str(blog)).parent
    lines = [json.loads(line) for line in (tdir / "sess.jsonl").read_text().splitlines()]
    for rec in lines:  # break every old_string so replay alone would fail
        for block in (rec.get("message") or {}).get("content") or []:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block["name"] == "Edit"
            ):
                block["input"]["old_string"] = "never in any base"
    (tdir / "sess.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    left = tmp_path / "claude-left" / "sess" / "post"
    left.mkdir(parents=True)
    claude_final = BASE.replace(
        "The first paragraph states the point plainly and says what follows from it.", CLAUDE_P1
    )
    (left / "t.qmd").write_text(claude_final, encoding="utf-8")
    report = tr.mine_transcripts(
        tdir, blog_root=blog, data_dir=tmp_path / "c", claude_left=tmp_path / "claude-left"
    )
    assert (report.base_hook, report.replay_failed, report.pairs) == (1, 0, 1)
    assert "hook's copy" in "\n".join(report.lines())
    without = tr.mine_transcripts(tdir, blog_root=blog, data_dir=tmp_path / "c")
    assert (without.base_hook, without.replay_failed, without.pairs) == (0, 1, 0)
