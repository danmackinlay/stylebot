"""Claude Code session transcripts as a source of real pairs (D89;
`_plans/transcript-mining.md`).

`parse_session` reads one session JSONL into a `Session`: the `Edit`/`Write`
calls on `.qmd` files under its `cwd`, each with the user turns and Claude's
text and thinking beside it, and the last full `Read` of every path.
`replay` rebuilds a file as Claude left it; `mine_transcripts` pairs that
with the blog's next commit through `capture.capture_pairs`, dry-run by
default. Claude Code's JSONL only; the fields are named in `_FIELDS`.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from stylebot import config
from stylebot.capture import capture_pairs, novel_after, plan_pairs
from stylebot.cli_options import Emitter, silent
from stylebot.draws import capture_id_of, iter_draws
from stylebot.errors import ConfigError, RunError
from stylebot.jsonl import append_jsonl, iter_jsonl
from stylebot.lib import DEFAULT_SOURCE, SourcePolicy
from stylebot.pairs import extract_target

#: The log's own names, in one place.
_FIELDS = {
    "edit_tools": ("Edit", "Write"),
    "read_tool": "Read",
    "record_types": ("user", "assistant"),
}
#: A user turn shorter than this is "please continue", not an objection.
MIN_TURN_WORDS = 25
#: Sidecar entries are cut here; a trace is a diagnostic, not corpus.
TURN_CHARS = 4000
#: How many substantive user turns before an edit the sidecar keeps.
TURNS_KEPT = 3
TURNS_FILE = "transcript_turns.jsonl"


@dataclass
class Edit:
    """One `Edit`/`Write` on a `.qmd`, with the deliberation beside it."""

    path: str
    ts: str
    tool: str
    old: str = ""
    new: str = ""
    replace_all: bool = False
    sidechain: bool = False
    original_file: str | None = None
    user_turns: tuple[str, ...] = ()
    claude_text: tuple[str, ...] = ()
    claude_thinking: tuple[str, ...] = ()


@dataclass
class Session:
    """One session's `.qmd` edits in order, and its full reads by path."""

    session_id: str
    path: Path
    cwd: str | None = None
    edits: list[Edit] = field(default_factory=list)
    reads: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def edits_by_file(self) -> dict[str, list[Edit]]:
        out: dict[str, list[Edit]] = {}
        for edit in self.edits:
            out.setdefault(edit.path, []).append(edit)
        return out

    def base_read(self, path: str, before_ts: str) -> str | None:
        """The last full `Read` of `path` before `before_ts`, as Claude saw it."""
        reads = [content for ts, content in self.reads.get(path, ()) if ts < before_ts]
        return reads[-1] if reads else None

    @property
    def user_turns(self) -> list[str]:
        seen: list[str] = []
        for edit in self.edits:
            seen += [t for t in edit.user_turns if t not in seen]
        return seen


def _rel_qmd(path: object, cwd: str | None) -> str | None:
    """`path` as repo-relative when it is a `.qmd` under `cwd`, else None."""
    if not isinstance(path, str) or not path.endswith(".qmd"):
        return None
    if cwd and path.startswith(cwd.rstrip("/") + "/"):
        return path[len(cwd.rstrip("/")) + 1 :]
    return None if path.startswith("/") else path


def _blocks(content: object, kind: str) -> list[str]:
    if isinstance(content, str):
        return [content] if kind == "text" else []
    if not isinstance(content, list):
        return []
    return [
        c.get(kind if kind != "thinking" else "thinking") or c.get("text") or ""
        for c in content
        if isinstance(c, dict) and c.get("type") == kind
    ]


def _substantive(texts: list[str]) -> list[str]:
    return [
        t.strip()
        for t in texts
        if t and not t.lstrip().startswith("<") and len(t.split()) >= MIN_TURN_WORDS
    ]


def _full_read(result: object) -> tuple[str, str] | None:
    """`(filePath, content)` of a whole-file `Read` result, else None."""
    if not isinstance(result, dict) or not isinstance(result.get("file"), dict):
        return None
    f = result["file"]
    if f.get("startLine", 1) != 1 or f.get("numLines") != f.get("totalLines"):
        return None
    return f.get("filePath"), f.get("content")


def parse_session(path: str | Path) -> Session:
    """One session JSONL -> `Session`; records of other types are skipped."""
    path = Path(path)
    session = Session(session_id=path.stem, path=path)
    pending: dict[str, dict] = {}
    prev_text: list[str] = []
    prev_thinking: list[str] = []
    user_turns: list[str] = []
    for rec in iter_jsonl(path):
        kind = rec.get("type")
        if kind not in _FIELDS["record_types"]:
            continue
        session.cwd = session.cwd or rec.get("cwd")
        content = (rec.get("message") or {}).get("content")
        if kind == "assistant":
            text = _blocks(content, "text")
            thinking = _blocks(content, "thinking")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    pending[block.get("id")] = {
                        "name": block.get("name"),
                        "input": block.get("input") or {},
                        "ts": rec.get("timestamp") or "",
                        "sidechain": bool(rec.get("isSidechain")),
                        "text": tuple(prev_text + text),
                        "thinking": tuple(prev_thinking + thinking),
                    }
            prev_text, prev_thinking = text, thinking
            continue
        user_turns += _substantive(_blocks(content, "text"))
        for block in content if isinstance(content, list) else []:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            use = pending.pop(block.get("tool_use_id"), None)
            if use is None:
                continue
            _take_result(session, use, rec, user_turns)
    return session


def _take_result(session: Session, use: dict, rec: dict, user_turns: list[str]) -> None:
    """Fold one tool result into the session: a full `Read`, or an edit."""
    result = rec.get("toolUseResult")
    if use["name"] == _FIELDS["read_tool"]:
        full = _full_read(result)
        if full is not None and (rel := _rel_qmd(full[0], session.cwd)) is not None:
            session.reads.setdefault(rel, []).append((use["ts"], full[1] or ""))
        return
    if use["name"] not in _FIELDS["edit_tools"]:
        return
    inp = use["input"]
    rel = _rel_qmd(inp.get("file_path"), session.cwd)
    if rel is None:
        return
    original = result.get("originalFile") if isinstance(result, dict) else None
    session.edits.append(
        Edit(
            path=rel,
            ts=use["ts"],
            tool=use["name"],
            old=inp.get("old_string") or "",
            new=inp.get("new_string") if use["name"] == "Edit" else inp.get("content") or "",
            replace_all=bool(inp.get("replace_all")),
            sidechain=use["sidechain"],
            original_file=original if isinstance(original, str) else None,
            user_turns=tuple(user_turns[-TURNS_KEPT:]),
            claude_text=use["text"],
            claude_thinking=use["thinking"],
        )
    )


def iter_sessions(transcripts_dir: str | Path) -> Iterator[Session]:
    """Every `*.jsonl` under the directory that edits a `.qmd`, by file name."""
    root = Path(transcripts_dir)
    if not root.is_dir():
        raise ConfigError(f"--transcripts {root} is not a directory", param_hint="--transcripts")
    for path in sorted(root.glob("*.jsonl")):
        session = parse_session(path)
        if session.edits:
            yield session


def replay(base: str, edits: list[Edit]) -> tuple[str, int | None]:
    """Apply `edits` in order to `base` -> `(text, index of the first edit whose
    old_string does not apply, else None)`; the text stops there."""
    text = base
    for i, edit in enumerate(edits):
        if edit.tool == "Write":
            text = edit.new
            continue
        if edit.old not in text:
            return text, i
        text = text.replace(edit.old, edit.new, -1 if edit.replace_all else 1)
    return text, None


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def file_before(root: Path, path: str, ts: str) -> str | None:
    """`path` at the last commit before `ts` that touches it, else None."""
    sha = (_git(root, "log", "-1", "--format=%H", f"--before={ts}", "--", path) or "").strip()
    return _git(root, "show", f"{sha}:{path}") if sha else None


def commit_after(root: Path, path: str, ts: str) -> tuple[str, str] | None:
    """`(sha, commit date)` of the first commit after `ts` touching `path`."""
    out = _git(root, "log", "--reverse", "--format=%H\t%cI", f"--after={ts}", "--", path)
    first = (out or "").strip().splitlines()
    if not first:
        return None
    sha, date = first[0].split("\t")
    return sha, date


@dataclass
class MineReport:
    """Per-file outcomes of one mining pass, and what it would (or did) write."""

    sessions: int = 0
    sessions_mined_before: int = 0
    files: int = 0
    base_hook: int = 0
    base_read: int = 0
    base_git: int = 0
    base_missing: int = 0
    replay_failed: int = 0
    window_skipped: int = 0
    unchanged: int = 0
    nothing_kept: int = 0
    windows: int = 0
    pairs: int = 0
    novel_after: list[float] = field(default_factory=list)
    written: list[dict] = field(default_factory=list)

    def lines(self) -> list[str]:
        counts = {
            "sessions with .qmd edits": self.sessions,
            "  already mined": self.sessions_mined_before,
            "files (session x path)": self.files,
            "  base from the hook's copy (no replay)": self.base_hook,
            "  base from the session's Read": self.base_read,
            "  base from git": self.base_git,
            "  base missing": self.base_missing,
            "  replay failed (an old_string did not apply)": self.replay_failed,
            "  window skipped (another session edited it first)": self.window_skipped,
            "  unchanged by Dan": self.unchanged,
            "  nothing kept (untouched or already captured)": self.nothing_kept,
            "windows yielding pairs": self.windows,
            "pairs": self.pairs,
        }
        out = [f"{k:52s} {v:6d}" for k, v in counts.items()]
        if self.novel_after:
            s = sorted(self.novel_after)
            q = [s[int(len(s) * f)] for f in (0.5, 0.75, 0.9)]
            out.append(f"novel_after median {q[0]:.2f}, q3 {q[1]:.2f}, p90 {q[2]:.2f}")
        return out


def _covered_targets(draws_dir: Path) -> tuple[set[str], set[str]]:
    """`(target bodies of real records, session ids already mined)`."""
    targets: set[str] = set()
    sessions: set[str] = set()
    if not draws_dir.is_dir():
        return targets, sessions
    for draw in iter_draws(draws_dir):
        if draw.record is None or capture_id_of(draw) is None:
            continue
        targets.add(extract_target(draw.record).strip())
        mined = (draw.record.get("meta") or {}).get("mined") or {}
        if mined.get("session_id"):
            sessions.add(mined["session_id"])
    return targets, sessions


def _paragraphs(text: str) -> set[str]:
    return {p.strip() for p in text.split("\n\n") if p.strip()}


def touched_paragraphs(before: str, edits: list[Edit]) -> set[str]:
    """The paragraphs of `before` Claude wrote into: those holding an edit's
    `new_string` (every paragraph after a `Write`). From the edits, not a diff
    against the base, so a dirty tree's own changes never count as Claude's."""
    paras = _paragraphs(before)
    if any(e.tool == "Write" for e in edits):
        return paras
    news = [n for e in edits if (n := e.new.strip())]
    return {p for p in paras if any(n in p or p in n for n in news)}


def revert_untouched(touched: set[str], before: str, after: str) -> str:
    """`after` with every paragraph run outside `touched` put back as it stood
    in `before`, so the differ sees Dan's edits to Claude's paragraphs only (D81)."""
    from difflib import SequenceMatcher

    b, a = before.split("\n\n"), after.split("\n\n")
    out: list[str] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, b, a, autojunk=False).get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:  # aligned: decide paragraph by paragraph
            out += [
                y if x.strip() in touched else x for x, y in zip(b[i1:i2], a[j1:j2], strict=True)
            ]
        elif tag == "equal" or any(p.strip() in touched for p in b[i1:i2]):
            out += a[j1:j2]
        else:
            out += b[i1:i2]
    return "\n\n".join(out)


def _keep_for(touched: set[str], covered: set[str]) -> Callable[[str, str], bool]:
    """Admit a chunk only where Claude's edits changed a paragraph in it and Dan's
    side is not already a captured target (D81, D89)."""

    def keep(before_chunk: str, after_chunk: str) -> bool:
        return any(p in touched for p in _paragraphs(before_chunk)) and (
            after_chunk.strip() not in covered
        )

    return keep


def _after_state(
    root: Path, path: str, last_ts: str, others: list[Session], session_id: str
) -> tuple[str | None, str | None, bool]:
    """`(after text, sha, window skipped)`: Dan's next commit of `path`, else the
    working tree; skipped when another session edits the path in the window."""
    found = commit_after(root, path, last_ts)
    end = found[1] if found else "9999"
    for other in others:
        if other.session_id == session_id:
            continue
        if any(e.path == path and last_ts < e.ts < end for e in other.edits):
            return None, None, True
    if found:
        return _git(root, "show", f"{found[0]}:{path}"), found[0], False
    tree = root / path
    return (tree.read_text(encoding="utf-8") if tree.is_file() else None), None, False


def mine_transcripts(
    transcripts_dir: str | Path,
    *,
    blog_root: str | Path,
    data_dir: str | Path,
    since: str | None = None,
    dry_run: bool = True,
    write_turns: bool = True,
    claude_left: str | Path | None = None,
    heading_context: bool = True,
    coalesce_gap_paras: int = 0,
    coalesce_gap_chars: int = 0,
    source_policy: SourcePolicy = DEFAULT_SOURCE,
    system_prompt: str | None = None,
    emit: Emitter = silent,
) -> MineReport:
    """Replay every session onto its base and pair the result with Dan's next
    commit through `capture_pairs`; `dry_run` counts and writes nothing.
    Sessions already mined (a record's `meta.mined.session_id`) are skipped;
    `since` drops sessions whose last edit precedes it (ISO date). With
    `claude_left`, a `<dir>/<session>/<path>` copy (the hook's) is the file
    as Claude left it and nothing is replayed."""
    root, data_dir = Path(blog_root), Path(data_dir)
    if not (root / ".git").exists() and _git(root, "rev-parse") is None:
        raise ConfigError(f"--blog-root {root} is not a git checkout", param_hint="--blog-root")
    sessions = list(iter_sessions(transcripts_dir))
    covered, mined_before = _covered_targets(config.draws_path(data_dir))
    report = MineReport(sessions=len(sessions))
    knobs = {
        "heading_context": heading_context,
        "coalesce_gap_paras": coalesce_gap_paras,
        "coalesce_gap_chars": coalesce_gap_chars,
    }
    for session in sessions:
        if session.session_id in mined_before:
            report.sessions_mined_before += 1
            continue
        for path, edits in session.edits_by_file().items():
            if since and edits[-1].ts < since:
                continue
            report.files += 1
            _mine_file(
                session,
                path,
                edits,
                sessions,
                root,
                data_dir,
                covered,
                report,
                knobs=knobs,
                dry_run=dry_run,
                claude_left=Path(claude_left) if claude_left else None,
                source_policy=source_policy,
                system_prompt=system_prompt,
                write_turns=write_turns,
                emit=emit,
            )
    return report


def _mine_file(session, path, edits, sessions, root, data_dir, covered, report, **kw) -> None:
    """One (session, path) window: base -> replay -> after -> keep -> land."""
    first_ts, last_ts = edits[0].ts, edits[-1].ts
    left = kw.pop("claude_left", None)
    left = left / session.session_id / path if left is not None else None
    if left is not None and left.is_file():
        before, failed, kind = left.read_text(encoding="utf-8"), None, "hook"
    else:
        base = "" if edits[0].tool == "Write" else session.base_read(path, first_ts)
        kind = "read"
        if base is None:
            base, kind = file_before(root, path, first_ts), "git"
        if base is None:
            report.base_missing += 1
            return
        before, failed = replay(base, edits)
    setattr(report, f"base_{kind}", getattr(report, f"base_{kind}") + 1)
    if failed is not None:
        report.replay_failed += 1
        kw["emit"](f"{session.session_id[:8]} {path}: edit {failed} does not apply", level="status")
        return
    after, after_sha, skipped = _after_state(root, path, last_ts, sessions, session.session_id)
    if skipped:
        report.window_skipped += 1
        return
    touched = touched_paragraphs(before, edits)
    after = revert_untouched(touched, before, after) if after is not None else None
    if after is None or after == before:
        report.unchanged += 1
        return
    keep = _keep_for(touched, covered)
    try:
        planned = [t for t in plan_pairs(before, after, **kw["knobs"]) if keep(t[0], t[1])]
    except RunError:  # only inserts, deletes or whitespace changed
        planned = []
    if not planned:
        report.nothing_kept += 1
        return
    report.windows += 1
    report.pairs += len(planned)
    report.novel_after += [n for b, a, _ in planned if (n := novel_after(b, a)) is not None]
    if kw["dry_run"]:
        return
    mined = {
        "session_id": session.session_id,
        "session_edits": len(edits),
        "base": kind,
        "sidechain": any(e.sidechain for e in edits),
        "after_sha": after_sha,
    }
    _land_window(
        session, path, edits, before, after, keep, mined, root, data_dir, covered, report, **kw
    )


def _land_window(
    session,
    path,
    edits,
    before,
    after,
    keep,
    mined,
    root,
    data_dir,
    covered,
    report,
    *,
    knobs,
    source_policy,
    system_prompt,
    write_turns,
    emit,
    dry_run,
) -> None:
    """Write one window's pairs through `capture_pairs`, and its sidecar line."""
    try:
        records = capture_pairs(
            before,
            after,
            pairs_path=config.pairs_path(data_dir),
            source=path,
            snapped_at=edits[-1].ts,
            extra_meta={"mined": mined},
            blog_root=root,
            keep=keep,
            source_policy=source_policy,
            system_prompt=system_prompt,
            **knobs,
        )
    except RunError as exc:
        emit(f"{session.session_id[:8]} {path}: {exc}", level="warn")
        return
    report.written += records
    covered.update(extract_target(r).strip() for r in records)
    if records and write_turns:
        append_jsonl(data_dir / TURNS_FILE, _turns_entry(session, edits, records[0], path))


def _cut(texts) -> list[str]:
    seen: list[str] = []
    for t in texts:
        if t and t not in seen:
            seen.append(t[:TURN_CHARS])
    return seen


def _turns_entry(session: Session, edits: list[Edit], record: dict, path: str) -> dict:
    """The sidecar line for one capture: Dan's objections and Claude's account."""
    return {
        "capture_id": record["meta"]["capture_id"],
        "session_id": session.session_id,
        "source": path,
        "user": _cut(t for e in edits for t in e.user_turns),
        "claude_text": _cut(t for e in edits for t in e.claude_text),
        "claude_thinking": _cut(t for e in edits for t in e.claude_thinking),
    }
