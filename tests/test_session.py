"""Library-level twins for the `ai-style log` behaviours (W11.b).

`tests/test_ai_style_log.py` drives the same rules through click's `CliRunner`;
those stay, as the check that the shell is thin and the exit codes are right.
These call `stylebot.session` directly, which is where the rules now live:
snapshot persistence, symlink canonicalisation, `save`'s disposition rule and
`tidy`'s latest-wins rule. A behaviour that can only be asserted by parsing CLI
output is a behaviour with no library seam.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylebot import session
from stylebot.jsonl import write_jsonl_atomic
from stylebot.session import SessionError, SessionStore, Snapshot

SLOP = "## S\n\nIt is worth noting the rich tapestry.\n"


@pytest.fixture
def store(tmp_path) -> SessionStore:
    return SessionStore(tmp_path / "corpus")


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A working tree with a `post/` dir, cwd'd into it."""
    tree = tmp_path / "repo"
    (tree / "post").mkdir(parents=True)
    monkeypatch.chdir(tree)
    return tree


@pytest.fixture
def post(repo) -> Path:
    """`post/foo.qmd`, written with slop, addressed by its relative path."""
    (repo / "post" / "foo.qmd").write_text(SLOP, encoding="utf-8")
    return Path("post/foo.qmd")


def _bodies(store: SessionStore) -> list[str]:
    """The Dan-side text of every pair, with any heading-context prefix removed."""
    bodies = []
    for rec in session.read_pairs(store):
        content = rec["messages"][2]["content"]
        context = rec["meta"].get("context") or ""
        if context and content.startswith(context):
            content = content[len(context) :].lstrip("\n")
        bodies.append(content)
    return bodies


def _stamps(store: SessionStore) -> list[str | None]:
    return [rec["meta"].get("snapped_at") for rec in session.read_pairs(store)]


# ---- the store: one data dir, four paths ---------------------------------


def test_store_resolves_through_the_config_ladder(tmp_path, monkeypatch):
    """The corpus location comes from `stylebot.config`, read per call.

    Twin of the CLI's `test_env_supplied_data_dir_is_honoured`: an env var set
    *after* import must still be seen, which is why nothing here is a
    module-level constant.
    """
    monkeypatch.setenv("STYLEBOT_DATA_DIR", str(tmp_path / "env-corpus"))
    resolved = SessionStore.resolve()
    assert resolved.data_dir == tmp_path / "env-corpus"
    assert resolved.pairs_path == tmp_path / "env-corpus" / "pairs.jsonl"
    assert SessionStore.resolve(tmp_path / "flag").data_dir == tmp_path / "flag"


def test_closed_snapshots_are_a_sibling_tree_not_a_suffix(store):
    """Open listings rglob the snapshots dir; a closed session must not be in it."""
    assert store.closed_snapshots_dir != store.snapshots_dir
    assert store.closed_snapshots_dir not in store.snapshots_dir.parents
    assert store.snapshot_path("post/foo.qmd").parent.name == "post"


# ---- snapshot persistence -------------------------------------------------


def test_snapshot_round_trips_as_ordinary_files(store, tmp_path):
    snap = tmp_path / "snap"
    shot = Snapshot(source="post/foo.qmd", snapped_at="2026-06-28T00:00:00+00:00", content=SLOP)
    session.write_snapshot(snap, shot)

    # The base text is a readable file, not a JSON-escaped string...
    assert (snap / session.SNAPSHOT_CONTENT).read_text(encoding="utf-8") == SLOP
    # ...and `.md`, so the blog's *.qmd tooling cannot see it.
    assert session.SNAPSHOT_CONTENT.endswith(".md")
    assert json.loads((snap / session.SNAPSHOT_META).read_text())["source"] == "post/foo.qmd"
    assert session.read_snapshot(snap) == shot
    assert session.read_snapshot(snap).chars == len(SLOP)


def test_load_snapshots_carries_an_unreadable_one_as_data(store, tmp_path):
    """A listing must not abort because one snapshot directory is broken."""
    good = store.snapshots_dir / "good"
    session.write_snapshot(good, Snapshot(source="a", snapped_at="t", content="x"))
    bad = store.snapshots_dir / "bad"
    bad.mkdir(parents=True)
    (bad / session.SNAPSHOT_META).write_text("{not json", encoding="utf-8")

    entries = {e.path.name: e for e in session.load_snapshots(store.snapshots_dir)}
    assert entries["good"].snapshot is not None and entries["good"].source == "a"
    assert entries["bad"].snapshot is None and entries["bad"].error
    # An unloadable entry still names itself, by its directory.
    assert entries["bad"].source.endswith("bad")


def test_snapshot_dirs_is_empty_for_a_missing_tree(store):
    assert session.snapshot_dirs(store.snapshots_dir) == []
    assert session.load_snapshots(store.closed_snapshots_dir) == []


# ---- symlink canonicalisation --------------------------------------------


def test_canonical_source_keys_a_symlink_by_its_target(repo, post):
    harness = repo / "_focus" / "post"
    harness.mkdir(parents=True)
    (harness / "foo.qmd").symlink_to(repo / "post" / "foo.qmd")

    assert session.canonical_source(Path("_focus/post/foo.qmd")) == Path("post/foo.qmd")


def test_canonical_source_leaves_an_ordinary_path_alone(repo, post):
    """Only symlinks are normalized, so no existing session key can move."""
    assert session.canonical_source(post) == post
    assert session.canonical_source(None) is None


def test_canonical_source_falls_back_to_absolute_outside_the_tree(tmp_path, repo):
    """Invoked from outside the tree the symlink points into: absolute is still
    better than a key under a directory that gets wiped."""
    outside = tmp_path / "elsewhere.qmd"
    outside.write_text(SLOP, encoding="utf-8")
    link = repo / "linked.qmd"
    link.symlink_to(outside)

    got = session.canonical_source(Path("linked.qmd"), cwd=repo / "post")
    assert got.is_absolute()
    assert got == outside.resolve()


# ---- open / close / reopen / drop ----------------------------------------


def test_open_is_sticky(store, post):
    snapshot = session.open_session(store, post)
    assert snapshot.content == SLOP
    assert store.snapshot_path(post).exists()

    Path(post).write_text("## S\n\nsomething else.\n", encoding="utf-8")
    with pytest.raises(SessionError, match="already open"):
        session.open_session(store, post)
    # The original (worst) state survived the refusal.
    assert session.read_snapshot(store.snapshot_path(post)).content == SLOP


def test_save_archives_the_snapshot_instead_of_deleting_it(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    session.save_session(store, post)

    assert not store.snapshot_path(post).exists()
    archived = store.closed_snapshot_path(post)
    assert archived.exists()
    # The archive holds the ORIGINAL base, not the saved state.
    assert "rich tapestry" in session.read_snapshot(archived).content
    assert session.read_snapshot(archived).source == "post/foo.qmd"


def test_reopen_restores_and_consumes_the_archive(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    session.save_session(store, post)
    stamp = _stamps(store)[0]

    restored = session.reopen_session(store, post)
    assert restored.snapped_at == stamp  # preserved verbatim: the redo hinges on it
    assert store.snapshot_path(post).exists()
    assert not store.closed_snapshot_path(post).exists()  # cannot be double-restored


def test_reopen_refuses_when_a_session_is_open(store, post):
    session.open_session(store, post)
    with pytest.raises(SessionError, match="nothing to reopen"):
        session.reopen_session(store, post)


def test_reopen_without_an_archive_errors(store, post):
    with pytest.raises(SessionError, match="no closed session"):
        session.reopen_session(store, post)


def test_drop_is_recoverable_via_reopen(store, post):
    session.open_session(store, post)
    session.drop_session(store, post)
    assert not store.snapshot_path(post).exists()

    session.reopen_session(store, post)
    assert "rich tapestry" in session.read_snapshot(store.snapshot_path(post)).content


def test_drop_without_a_session_errors(store, post):
    with pytest.raises(SessionError, match="no open session"):
        session.drop_session(store, post)


# ---- save: the unready world ---------------------------------------------


def test_save_without_a_session_names_the_redo_route_once_one_exists(store, post):
    with pytest.raises(SessionError) as first:
        session.save_session(store, post)
    assert "run `open` first" in str(first.value)
    assert "--redo" not in str(first.value)

    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    session.save_session(store, post)

    with pytest.raises(SessionError, match="save --redo") as second:
        session.save_session(store, post)
    assert "no open session" in str(second.value)


def test_save_with_nothing_changed_errors(store, post):
    session.open_session(store, post)
    with pytest.raises(SessionError, match="nothing to save"):
        session.save_session(store, post)


def test_redo_without_an_archive_errors(store, post):
    with pytest.raises(SessionError, match="no closed session"):
        session.save_session(store, post, redo=True)


def test_redo_is_a_noop_when_a_session_is_already_open(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    notices: list[str] = []
    result = session.save_session(store, post, redo=True, notify=notices.append)
    assert result.reopened is None
    assert not any("reopened" in line for line in notices)
    assert len(session.read_pairs(store)) == 1


# ---- save: the disposition rule ------------------------------------------


def _first_session(store, post) -> str | None:
    session.open_session(store, post)
    Path(post).write_text("## S\n\nFirst rewrite here.\n", encoding="utf-8")
    session.save_session(store, post)
    return _stamps(store)[0]


def test_reopen_then_save_matches_redo(store, post):
    """`redo` is exactly `reopen` + `save`, so the two-step must land identically."""
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    session.save_session(store, post)
    stamp = _stamps(store)[0]

    session.reopen_session(store, post)
    Path(post).write_text("## S\n\nWe dig in, without a typo.\n", encoding="utf-8")
    session.save_session(store, post)

    assert _stamps(store) == [stamp]
    assert _bodies(store) == ["We dig in, without a typo."]


def test_redo_only_replaces_its_own_sessions_pairs(store, post):
    """A redo must not take an earlier session's legitimate rewrite with it."""
    _first_session(store, post)

    Path(post).write_text("## S\n\nSecond slop draft here.\n", encoding="utf-8")
    session.open_session(store, post)
    Path(post).write_text("## S\n\nSecond rewrite, typoo.\n", encoding="utf-8")
    session.save_session(store, post)
    assert len(session.read_pairs(store)) == 2

    Path(post).write_text("## S\n\nSecond rewrite, fixed.\n", encoding="utf-8")
    session.save_session(store, post, redo=True)

    assert sorted(_bodies(store)) == ["First rewrite here.", "Second rewrite, fixed."]


def test_redo_replaces_this_sessions_pairs_against_the_same_base(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in, with a typoo.\n", encoding="utf-8")
    session.save_session(store, post)
    stamp = _stamps(store)[0]

    Path(post).write_text("## S\n\nWe dig in, without a typo.\n", encoding="utf-8")
    notices: list[str] = []
    result = session.save_session(store, post, redo=True, notify=notices.append)

    assert result.reopened is not None
    assert any("reopened session" in line for line in notices)
    assert any("replaced 1 interim pair(s)" in line for line in notices)
    # Replaced, not stacked, because `snapped_at` survived the reopen.
    assert len(session.read_pairs(store)) == 1
    assert _stamps(store) == [stamp]
    assert "without a typo" in _bodies(store)[0]
    # And the redo's own close re-archived the same base for a further redo.
    assert "rich tapestry" in session.read_snapshot(store.closed_snapshot_path(post)).content


def test_smart_disposition_keeps_earlier_sessions(store, post):
    _first_session(store, post)

    Path(post).write_text("## S\n\nSecond slop draft here.\n", encoding="utf-8")
    session.open_session(store, post)
    Path(post).write_text("## S\n\nSecond rewrite here.\n", encoding="utf-8")
    result = session.save_session(store, post)

    assert result.disposition == "smart"
    assert len(result.prior.other_session) == 1
    assert result.prior.same_session == []
    assert len(session.read_pairs(store)) == 2
    assert sorted(_bodies(store)) == ["First rewrite here.", "Second rewrite here."]


def test_replace_all_drops_earlier_sessions_too(store, post):
    _first_session(store, post)

    Path(post).write_text("## S\n\nSecond slop draft here.\n", encoding="utf-8")
    session.open_session(store, post)
    Path(post).write_text("## S\n\nSecond rewrite here.\n", encoding="utf-8")
    notices: list[str] = []
    result = session.save_session(store, post, disposition="replace-all", notify=notices.append)

    assert result.disposition == "replace-all"
    assert any("from earlier session(s)" in line for line in notices)
    assert _bodies(store) == ["Second rewrite here."]


def test_append_keeps_even_this_sessions_interim_pairs(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nInterim rewrite here.\n", encoding="utf-8")
    session.save_session(store, post, keep_open=True)
    assert store.snapshot_path(post).exists()  # keep_open leaves the base sticky

    Path(post).write_text("## S\n\nFinal rewrite here.\n", encoding="utf-8")
    result = session.save_session(store, post, disposition="append")

    assert len(result.prior.same_session) == 1
    assert sorted(_bodies(store)) == ["Final rewrite here.", "Interim rewrite here."]


def test_keep_open_interim_pairs_are_replaced_by_default(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nInterim rewrite here.\n", encoding="utf-8")
    session.save_session(store, post, keep_open=True)

    Path(post).write_text("## S\n\nFinal rewrite here.\n", encoding="utf-8")
    session.save_session(store, post)
    assert _bodies(store) == ["Final rewrite here."]


def test_prior_pairs_separates_sessionless_from_earlier_sessions(store, post):
    """Pairs logged outside a session carry `snapped_at=None`.

    Sorting the mixed set of stamps used to raise `TypeError: '<' not supported
    between 'str' and 'NoneType'`, so the two kinds are counted apart.
    """
    session.capture_pair(
        store,
        before_text="It is worth noting the tapestry.",
        after_text="We dig in.",
        source=str(post),
        whole=True,
    )
    _first_session(store, post)

    Path(post).write_text("## S\n\nA later slop draft entirely.\n", encoding="utf-8")
    session.open_session(store, post)
    Path(post).write_text("## S\n\nA later rewrite entirely.\n", encoding="utf-8")
    prior = session.save_session(store, post).prior

    assert len(prior.other_session) == 2
    assert len(prior.earlier_session_stamps) == 1
    assert prior.sessionless == 1
    assert prior.total == 2
    assert len(session.read_pairs(store)) == 3


def test_save_tags_and_whole_mode_reach_the_record(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in.\n\n## T\n\nMore.\n", encoding="utf-8")
    result = session.save_session(store, post, tags=["experiment"], whole=True)

    assert result.whole is True
    assert len(result.written) == 1
    assert result.written[0]["meta"]["tags"] == ["experiment"]
    assert result.before_chars and result.after_chars


def test_unknown_disposition_is_an_argument_error(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown disposition"):
        session.save_session(store, post, disposition="nonsense")  # type: ignore[arg-type]


# ---- pair: one-shot capture ----------------------------------------------


def test_capture_pair_appends_by_default(store):
    for after in ("We dig in.", "We dig in again."):
        session.capture_pair(
            store,
            before_text="It is worth noting the rich tapestry.",
            after_text=after,
            source="clipboard",
            whole=True,
        )
    assert len(session.read_pairs(store)) == 2


def test_capture_pair_replace_drops_prior_pairs_for_the_source(store):
    session.capture_pair(
        store,
        before_text="It is worth noting the rich tapestry.",
        after_text="We dig in.",
        source="clipboard",
        whole=True,
    )
    notices: list[str] = []
    result = session.capture_pair(
        store,
        before_text="It is worth noting the rich tapestry.",
        after_text="We dig in again.",
        source="clipboard",
        replace=True,
        whole=True,
        notify=notices.append,
    )
    assert result.replaced == 1
    assert notices == ["replaced 1 prior pair(s) for clipboard"]
    assert _bodies(store) == ["We dig in again."]


def test_capture_pair_identical_texts_are_an_unready_world(store):
    with pytest.raises(SessionError, match="refusing to log"):
        session.capture_pair(store, before_text="same", after_text="same")


def test_capture_pair_replace_without_a_source_is_an_argument_error(store):
    """A ValueError, not a SessionError: the CLI turns this one into exit 2."""
    with pytest.raises(ValueError, match="--replace requires --source") as exc:
        session.capture_pair(
            store, before_text="slop here", after_text="terse", replace=True, whole=True
        )
    assert not isinstance(exc.value, SessionError)


def test_capture_pair_with_no_changed_chunk_is_an_unready_world(store):
    """Pure insert: nothing to pair. `capture`'s ValueError becomes a SessionError."""
    with pytest.raises(SessionError, match="no changed text chunks"):
        session.capture_pair(
            store,
            before_text="## H\n\nkept body",
            after_text="## H\n\nkept body\n\nbrand new added paragraph",
        )


# ---- list -----------------------------------------------------------------


def test_summarize_state_groups_by_source_and_counts_saves(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    session.save_session(store, post)
    session.capture_pair(
        store,
        before_text="It is worth noting the tapestry.",
        after_text="Terse.",
        whole=True,
    )

    state = session.summarize_state(store)
    assert state.pair_count == 2
    assert [row.source for row in state.by_source] == ["(no source)", "post/foo.qmd"]
    assert state.by_source[1].pairs == 1 and state.by_source[1].saves == 1
    # A closed session is not reported as an open one.
    assert state.open_sessions == []
    assert len(state.closed_sessions) == 1


# ---- tidy: latest save per source wins ------------------------------------


def _rec(source, captured_at, body):
    return {
        "messages": [{"role": "assistant", "content": body}],
        "meta": {"source": source, "captured_at": captured_at},
    }


def test_tidy_keeps_only_the_latest_save_per_source(store):
    write_jsonl_atomic(
        store.pairs_path,
        [
            _rec("a.qmd", "2026-01-01T00:00:00+00:00", "old a"),
            _rec("a.qmd", "2026-02-01T00:00:00+00:00", "new a"),
            _rec("b.qmd", "2026-01-01T00:00:00+00:00", "old b"),
            _rec("b.qmd", "2026-03-01T00:00:00+00:00", "new b"),
        ],
    )
    result = session.tidy_pairs(store)
    assert (result.removed, len(result.kept), result.applied) == (2, 2, True)
    assert sorted(r["messages"][0]["content"] for r in session.read_pairs(store)) == [
        "new a",
        "new b",
    ]


def test_tidy_can_be_scoped_to_one_source(store):
    write_jsonl_atomic(
        store.pairs_path,
        [
            _rec("a.qmd", "2026-01-01T00:00:00+00:00", "old a"),
            _rec("a.qmd", "2026-02-01T00:00:00+00:00", "new a"),
            _rec("b.qmd", "2026-01-01T00:00:00+00:00", "old b"),
            _rec("b.qmd", "2026-03-01T00:00:00+00:00", "new b"),
        ],
    )
    result = session.tidy_pairs(store, "a.qmd")
    assert result.removed == 1
    assert sorted(r["messages"][0]["content"] for r in result.kept) == ["new a", "new b", "old b"]


def test_tidy_dry_run_reports_without_writing(store):
    records = [
        _rec("a.qmd", "2026-01-01T00:00:00+00:00", "old a"),
        _rec("a.qmd", "2026-02-01T00:00:00+00:00", "new a"),
    ]
    write_jsonl_atomic(store.pairs_path, records)
    result = session.tidy_pairs(store, dry_run=True)
    assert (result.removed, result.applied) == (1, False)
    assert len(session.read_pairs(store)) == 2


def test_tidy_keeps_sourceless_and_undecodable_records(store):
    store.pairs_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(_rec("a.qmd", "2026-01-01T00:00:00+00:00", "old a")),
        json.dumps(_rec("a.qmd", "2026-02-01T00:00:00+00:00", "new a")),
        json.dumps({"messages": [], "meta": {}}),  # no source: nothing to compare
        "{ this is not json",
    ]
    store.pairs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = session.tidy_pairs(store)
    assert result.removed == 1
    assert any("_raw" in r for r in result.kept)  # undecodable lines round-trip
    assert len(session.read_pairs(store)) == 3


def test_tidy_on_an_empty_corpus_says_so(store):
    result = session.tidy_pairs(store)
    assert result.empty and result.removed == 0 and not result.applied


def test_tidy_with_nothing_to_remove_does_not_rewrite(store):
    write_jsonl_atomic(store.pairs_path, [_rec("a.qmd", "2026-02-01T00:00:00+00:00", "only a")])
    result = session.tidy_pairs(store)
    assert (result.empty, result.removed, result.applied) == (False, 0, False)


# ---- recut ----------------------------------------------------------------


def test_recut_refuses_legacy_archives_as_a_session_error(store):
    """`capture.iter_archives` refuses unmigrated `captures/<id>.json` blobs.

    The refusal is a library `ValueError`; `session.recut` re-raises it as the
    unready-world class so the CLI prints an instruction, not a traceback.
    """
    cap_dir = store.data_dir / "captures"
    cap_dir.mkdir(parents=True)
    (cap_dir / "deadbeef.json").write_text(
        json.dumps(
            {
                "capture_id": "deadbeef",
                "source": "post/legacy.qmd",
                "snapped_at": "2026-07-01T00:00:00+00:00",
                "whole": False,
                "before_text": "It is worth noting the rich tapestry.",
                "after_text": "We dig in.",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SessionError) as exc:
        session.recut(store, gap_paras=2, gap_chars=800, max_chars=8000)
    assert "1 legacy JSON capture archive" in str(exc.value)
    assert "ai-style data migrate-captures" in str(exc.value)


def test_recut_with_no_archives_reports_nothing(store):
    result = session.recut(store, gap_paras=2, gap_chars=800, max_chars=8000)
    assert result.records == [] and result.captures == 0
    assert result.pairs_before == result.pairs_after == 0


def test_recut_dry_run_leaves_the_corpus_alone(store, post):
    long_before = "## S\n\n" + ("It is worth noting the rich tapestry. " * 20) + "\n"
    long_after = "## S\n\n" + ("We dig in, again and again and again. " * 20) + "\n"
    Path(post).write_text(long_before, encoding="utf-8")
    session.open_session(store, post)
    Path(post).write_text(long_after, encoding="utf-8")
    session.save_session(store, post)
    before = len(session.read_pairs(store))

    result = session.recut(store, gap_paras=2, gap_chars=800, max_chars=8000, dry_run=True)
    assert result.records and result.captures == 1
    assert result.dry_run and result.pairs_after == result.pairs_before == before
    assert result.median_chars > 0 and result.max_chars >= result.median_chars
    assert len(session.read_pairs(store)) == before


# ---- the corpus receipt ---------------------------------------------------


def test_corpus_summary_counts_heading_context(store, post):
    session.open_session(store, post)
    Path(post).write_text("## S\n\nWe dig in.\n", encoding="utf-8")
    session.save_session(store, post)
    session.capture_pair(
        store,
        before_text="It is worth noting the tapestry.",
        after_text="Terse.",
        whole=True,  # whole mode never carries context
    )

    summary = session.corpus_summary(store)
    assert summary.pairs == 2
    assert summary.with_context == 1
    assert summary.path == store.pairs_path


def test_read_pairs_of_a_missing_corpus_is_empty(store):
    assert session.read_pairs(store) == []


# ---- age rendering --------------------------------------------------------


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s ago"), (90, "1m ago"), (7200, "2h ago"), (86400 * 3, "3d ago")],
)
def test_humanize_age_buckets(seconds, expected):
    from datetime import UTC, datetime, timedelta

    stamp = (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat()
    assert session.humanize_age(stamp) == expected


def test_humanize_age_passes_through_an_unparseable_stamp():
    assert session.humanize_age("not a timestamp") == "not a timestamp"
