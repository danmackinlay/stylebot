"""The session rules, at the library seam (W11.b).

These call `stylebot.session` directly, which is where the rules live:
snapshot persistence, symlink canonicalisation and `save`'s disposition
rule. `tests/test_ai_style_log.py` keeps only what the
shell adds — the corpus it resolves and its two exit-code classes — since a
thin shell is proven by those, not by a CliRunner copy of every rule (X4). A
behaviour that can only be asserted by parsing CLI output is a behaviour with
no library seam.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stylebot import session
from stylebot.errors import ConfigError
from stylebot.session import SessionError, SessionStore, Snapshot

SLOP = "## S\n\nIt is worth noting the rich tapestry.\n"


@pytest.fixture
def store(tmp_path) -> SessionStore:
    """A data dir in the draws layout (D84): a save lands under `draws/`."""
    store = SessionStore(tmp_path / "corpus")
    store.draws_dir.mkdir(parents=True)
    return store


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
    assert notices == ["superseded 1 prior pair(s) for clipboard"]
    assert _bodies(store) == ["We dig in again."]


def test_capture_pair_identical_texts_are_an_unready_world(store):
    with pytest.raises(SessionError, match="refusing to log"):
        session.capture_pair(store, before_text="same", after_text="same")


def test_capture_pair_replace_without_a_source_is_an_argument_error(store):
    """A ConfigError, not a SessionError: the CLI turns this one into exit 2."""
    with pytest.raises(ConfigError, match="--replace requires --source") as exc:
        session.capture_pair(
            store, before_text="slop here", after_text="terse", replace=True, whole=True
        )
    assert not isinstance(exc.value, SessionError)


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
    assert summary.path == store.draws_dir


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


# ---- a second save in one session supersedes the first (W4 step 4) -------


def test_a_second_save_in_one_session_supersedes_the_first(store, post):
    """Nothing is rewritten: both saves stay as draws, the successor names the
    capture it supersedes, and the identity view renders only the final one."""
    from stylebot import draws

    session.open_session(store, post)
    Path(post).write_text("## S\n\nInterim rewrite here.\n", encoding="utf-8")
    first = session.save_session(store, post, keep_open=True)
    Path(post).write_text("## S\n\nFinal rewrite here.\n", encoding="utf-8")
    notices: list[str] = []
    second = session.save_session(store, post, notify=notices.append)

    got = list(draws.iter_draws(store.draws_dir))
    assert [d.record for d in got] == first.written + second.written
    cap1 = first.written[0]["meta"]["capture_id"]
    assert (got[0].supersedes, got[1].supersedes) == (None, (cap1,))
    assert notices == ["superseded 1 interim pair(s) from this session"]
    assert len(second.prior.same_session) == 1
    assert _bodies(store) == ["Final rewrite here."]
    report = draws.build_view(got, draws.ViewPolicy())
    assert (report.records, report.excluded) == (1, {"superseded": 1})
    assert draws.build_view(got, draws.ViewPolicy(keep_superseded=True)).records == 2


# ---- recut: an edited archive is re-landed ---------------------------------


def test_recut_supersedes_the_archives_earlier_landing(store, post):
    """Edit `after.md`, `recut`: the fix is the live pair and the first landing
    is superseded; a second recut supersedes the first recut, not the original
    twice (D96)."""
    session.open_session(store, post)
    post.write_text("## S\n\nWe dig in, teh end.\n", encoding="utf-8")
    session.save_session(store, post)
    cid = session.read_pairs(store)[0]["meta"]["capture_id"]
    after_md = store.captures_dir / cid / "after.md"
    after_md.write_text(
        after_md.read_text(encoding="utf-8").replace("teh", "the"), encoding="utf-8"
    )

    result = session.recut_capture(store, cid)
    assert result.superseded == 1 and _bodies(store) == ["We dig in, the end."]
    live = session.read_pairs(store)[0]["meta"]
    assert live["archive_id"] == cid and live["capture_id"] != cid

    again = session.recut_capture(store, cid)
    assert again.superseded == 1 and len(_bodies(store)) == 1
    assert session.read_pairs(store)[0]["meta"]["capture_id"] != live["capture_id"]


def test_recut_without_an_archive_errors(store):
    with pytest.raises(SessionError, match="no capture archive"):
        session.recut_capture(store, "deadbeef")


STAYS = (
    "The second paragraph stays exactly as it was, word for word, sentence for sentence, "
    "clause for clause, with every comma where the author left it, unchanged."
)


def test_plan_recut_reports_status_and_flags_without_writing(store, post):
    """The dry run names a kept pair, a new pair and an identity-scale edit,
    plans nothing for the edit inside the code fence (D100), and appends no
    draw (D96)."""
    session.open_session(store, post)
    post.write_text(
        "## S\n\nWe dig in.\n\nGap one.\n\n" + STAYS + "\n\nGap two.\n\n```python\nx = 1\n```\n",
        encoding="utf-8",
    )
    # The before side needs the same fence and a second paragraph to diff against.
    snap = store.snapshot_path(post) / "content.md"
    snap.write_text(
        "## S\n\nIt is worth noting the rich tapestry.\n\nGap one.\n\n"
        + STAYS
        + "\n\nGap two.\n\n```python\nx = 0\n```\n",
        encoding="utf-8",
    )
    session.save_session(store, post)
    cid = session.read_pairs(store)[0]["meta"]["capture_id"]
    n_draws = len(list(session.read_pairs(store)))
    after_md = store.captures_dir / cid / "after.md"
    after_md.write_text(
        after_md.read_text(encoding="utf-8").replace(STAYS, STAYS[:-1] + "!"),
        encoding="utf-8",
    )
    planned = session.plan_recut(store, cid)
    assert [p.status for p in planned] == ["kept", "new"]
    assert planned[1].flags == ("identity",)
    assert "x = 0" not in "".join(p.before for p in planned)
    assert len(list(session.read_pairs(store))) == n_draws


def test_plan_recut_flags_a_whole_capture_that_is_not_prose(store, post):
    """`whole` bypasses the differ, so the `non-prose` flag is the dry run's
    only guard against a pair that lies inside a fence (D96, D100)."""
    session.open_session(store, post)
    post.write_text("```python\nx = 1\ny = 2\nz = 3\n```\n", encoding="utf-8")
    snap = store.snapshot_path(post) / "content.md"
    snap.write_text("```python\na = 0\nb = 0\nc = 0\n```\n", encoding="utf-8")
    session.save_session(store, post, whole=True)
    cid = session.read_pairs(store)[0]["meta"]["capture_id"]
    planned = session.plan_recut(store, cid)
    assert [p.status for p in planned] == ["kept"]
    assert "in:fence" in planned[0].flags


def test_save_pairs_a_callout_edit_only_under_a_div_policy(store, post):
    """Every div is protected by default (D100); a policy that frees callouts
    makes the edit inside one a pair."""
    callout = "::: {.callout-note}\n\nIt is worth noting the rich tapestry.\n\n:::\n"
    post.write_text(callout, encoding="utf-8")
    session.open_session(store, post)
    post.write_text(callout.replace("It is worth noting the rich tapestry.", "We dig in."))
    with pytest.raises(SessionError, match="no changed text chunks"):
        session.save_session(store, post)
    session.save_session(store, post, div_is_prose=lambda attrs: "callout" in attrs)
    assert _bodies(store) == ["We dig in."]
