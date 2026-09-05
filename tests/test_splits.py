"""Tests for the shared by-POST splits contract (`stylebot.splits`).

Pure stdlib — the splits module (and `classify_train.Dataset`, its duck-typed
input) import without any ML dependency, so these run on a minimal checkout.
"""

from __future__ import annotations

import pytest

from stylebot import splits as sp
from stylebot.classify_train import Dataset
from stylebot.errors import RunError


def _dataset(posts_pairs: dict[str, int], synth_posts: set | None = None) -> Dataset:
    """A Dataset with `n` pairs per post; posts in `synth_posts` are synthetic."""
    synth_posts = synth_posts or set()
    ds = Dataset()
    for post, n in posts_pairs.items():
        for c in range(n):
            ds._add_pair(
                f"slop {post}.{c}", f"author {post}.{c}", post, synthetic=post in synth_posts
            )
    return ds


def _counts(entry: dict) -> dict:
    """Just the tallies — `summarize_roles` also carries the concentration
    diagnostic (`top_post*`), which these assertions are not about."""
    return {k: entry[k] for k in ("posts", "pairs", "pairs_real", "pairs_synthetic")}


def test_make_splits_deterministic_and_pinned():
    posts = [f"post/p{i}.qmd" for i in range(20)]
    s1 = sp.make_splits(posts, eval_frac=0.2, seed=7)
    s2 = sp.make_splits(posts, eval_frac=0.2, seed=7)
    assert s1["eval_posts"] == s2["eval_posts"]
    assert len(s1["eval_posts"]) == 4  # round(20 * 0.2)
    # A different seed cuts a different eval stratum.
    assert sp.make_splits(posts, eval_frac=0.2, seed=8)["eval_posts"] != s1["eval_posts"]
    # eval_candidates restricts eligibility (the "real posts only" policy).
    reals = posts[:5]
    s3 = sp.make_splits(posts, eval_frac=0.2, seed=0, eval_candidates=reals)
    assert set(s3["eval_posts"]) <= set(reals)


def test_role_of_covers_and_is_growth_stable():
    posts = [f"post/p{i}.qmd" for i in range(30)]
    splits = sp.make_splits(posts, eval_frac=0.2, detector_frac=0.4, seed=0)
    roles = {p: sp.role_of(p, splits) for p in posts}
    assert set(roles.values()) <= set(sp.ROLES)
    for p in splits["eval_posts"]:
        assert roles[p] == "eval"
    # Non-eval posts land on both sides of the hash rule at these sizes.
    non_eval_roles = {r for p, r in roles.items() if p not in set(splits["eval_posts"])}
    assert non_eval_roles == {"detector", "styler"}
    # Growth stability: a post that did not exist at make time gets a stable,
    # non-eval role, and existing assignments are untouched by its arrival
    # (role_of depends only on the splits doc, never on the corpus).
    newcomer = "notebook/brand_new_post.qmd"
    assert sp.role_of(newcomer, splits) in ("detector", "styler")
    assert sp.role_of(newcomer, splits) == sp.role_of(newcomer, splits)
    assert {p: sp.role_of(p, splits) for p in posts} == roles


def test_role_of_is_a_frozen_contract():
    """Golden values: `role_of` may never change what it says about a post.

    There is a committed `splits.json` in the blog repo and every post's role
    (which stratum scores it, which model trains on it) is pinned by it. Any
    edit to the hash rule — the salting format, the digest slice, the [0,1)
    conversion — would silently reassign the whole corpus, so it fails here
    first. Values computed 2026-08-09 against the shipped rule.
    """
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/pinned_a.qmd", "post/pinned_b.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    assert {
        s: sp.role_of(s, splits)
        for s in (
            "post/pinned_a.qmd",
            "post/pinned_b.qmd",
            "notebook/alpha.qmd",
            "notebook/beta.qmd",
            "notebook/gamma.qmd",
            "post/delta.qmd",
            "post/epsilon.qmd",
            "notebook/zeta.qmd",
        )
    } == {
        "post/pinned_a.qmd": "eval",
        "post/pinned_b.qmd": "eval",
        "notebook/alpha.qmd": "styler",
        "notebook/beta.qmd": "styler",
        "notebook/gamma.qmd": "styler",
        "post/delta.qmd": "detector",
        "post/epsilon.qmd": "detector",
        "notebook/zeta.qmd": "detector",
    }
    # The salt is part of the hashed input, not a seed passed elsewhere: the same
    # sources under salt "7" cut a different (also pinned) partition.
    salt7 = {**splits, "eval_posts": [], "rest_rule": {**splits["rest_rule"], "salt": "7"}}
    assert [sp.role_of(s, salt7) for s in ("notebook/gamma.qmd", "post/delta.qmd")] == [
        "detector",
        "styler",
    ]

    # Pin the digest -> uniform [0,1) conversion itself to ~1e-5 by bracketing
    # one post's coordinate (u("0:notebook/gamma.qmd") = 0.46915…) with the
    # threshold it is compared against.
    def _with_frac(frac):
        return {**splits, "rest_rule": {**splits["rest_rule"], "detector_frac": frac}}

    assert sp.role_of("notebook/gamma.qmd", _with_frac(0.46916)) == "detector"
    assert sp.role_of("notebook/gamma.qmd", _with_frac(0.46914)) == "styler"


def test_save_load_roundtrip_and_validation(tmp_path):
    splits = sp.make_splits(["post/a.qmd", "post/b.qmd", "post/c.qmd"], eval_frac=0.34, seed=1)
    path = sp.save_splits(splits, tmp_path / "splits.json")
    loaded = sp.load_splits(path)
    assert loaded["eval_posts"] == splits["eval_posts"]
    assert loaded["rest_rule"] == splits["rest_rule"]

    (tmp_path / "bad_version.json").write_text('{"schema_version": 99}')
    with pytest.raises(RunError, match="schema_version"):
        sp.load_splits(tmp_path / "bad_version.json")
    (tmp_path / "bad_rule.json").write_text(
        '{"schema_version": 1, "eval_posts": [], "rest_rule": {"kind": "coin-flip"}}'
    )
    with pytest.raises(RunError, match="rest_rule"):
        sp.load_splits(tmp_path / "bad_rule.json")


def test_summarize_roles_counts_posts_pairs_and_provenance():
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/e1.qmd", "post/e2.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.5, "salt": "0"},
    }
    ds = _dataset(
        {"post/e1.qmd": 2, "post/e2.qmd": 1, "post/x1.qmd": 3, "post/x2.qmd": 2},
        synth_posts={"post/e2.qmd", "post/x2.qmd"},
    )
    roles = sp.summarize_roles(splits, ds)
    assert _counts(roles["eval"]) == {"posts": 2, "pairs": 3, "pairs_real": 2, "pairs_synthetic": 1}
    assert roles["detector"]["pairs"] + roles["styler"]["pairs"] == 5


def test_check_splits_danger_report():
    # Everything about this partition is dangerous: tiny strata, an all-synthetic
    # eval, a stale pinned post, a detector pool far below CV viability.
    splits = {
        "schema_version": 1,
        "eval_posts": ["post/synth_only.qmd", "post/gone.qmd"],
        "rest_rule": {"kind": "hash", "detector_frac": 0.5, "salt": "0"},
    }
    ds = _dataset(
        {"post/synth_only.qmd": 2, "post/a.qmd": 2, "post/b.qmd": 2},
        synth_posts={"post/synth_only.qmd"},
    )
    warnings = "\n".join(sp.check_splits(splits, ds))
    assert "DANGEROUSLY SMALL" in warnings
    assert "EVAL PURITY" in warnings  # eval pairs exist but none are real
    assert "STALE" in warnings and "post/gone.qmd" in warnings
    assert "CV UNSTABLE" in warnings


def test_check_splits_healthy_partition_is_quiet():
    posts = {f"post/p{i}.qmd": 3 for i in range(45)}  # 45 posts x 3 pairs
    ds = _dataset(posts)
    splits = sp.make_splits(sorted(posts), eval_frac=0.25, detector_frac=0.5, seed=0)
    assert sp.check_splits(splits, ds) == []


# --- renames: a role is keyed on identity, not on the path of the day -------
#
# The 2026-09-03 leak (W0): two pinned eval posts were renamed by `post-op
# move`, the pinned paths kept their pairs so nothing looked stale, and the new
# paths hashed straight into the styler and detector pools.


def _renamed_splits() -> dict:
    return {
        "schema_version": 1,
        "eval_posts": ["notebook/old.qmd", "notebook/new.qmd"],
        "aliases": {"notebook/new.qmd": "notebook/old.qmd"},
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }


def test_role_of_survives_a_rename_of_a_pinned_eval_post():
    splits = _renamed_splits()
    assert sp.role_of("notebook/old.qmd", splits) == "eval"
    assert sp.role_of("notebook/new.qmd", splits) == "eval"
    # ... and without the alias it would not have: the bug, reproduced.
    unaliased = {**splits, "aliases": {}, "eval_posts": ["notebook/old.qmd"]}
    assert sp.role_of("notebook/new.qmd", unaliased) != "eval"


def test_aliases_need_no_migration_of_existing_pairs(tmp_path):
    """An alias moves no pair that already has a role — and a rename without one
    is not silent.

    The constraint that picked the alias direction (new -> old): every
    hash-assigned post keeps exactly the role it had, because the path the
    corpus recorded IS the canonical one. The other half of the contract (C12,
    D34): a hash-pool post renamed on disk with no alias is reported by
    `check_splits(blog_root=…)` like a pinned one, never left to re-hash.
    """
    base = {
        "schema_version": 1,
        "eval_posts": [],
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    posts = [f"notebook/p{i}.qmd" for i in range(50)]
    before = {p: sp.role_of(p, base) for p in posts}
    aliased = {**base, "aliases": {"notebook/renamed.qmd": "notebook/p7.qmd"}}
    assert {p: sp.role_of(p, aliased) for p in before} == before
    assert sp.role_of("notebook/renamed.qmd", aliased) == before["notebook/p7.qmd"]

    # On disk, p7 has become renamed.qmd; the corpus still keys its pairs on p7.
    (tmp_path / "notebook").mkdir()
    for p in posts:
        if p != "notebook/p7.qmd":
            (tmp_path / p).write_text("x")
    (tmp_path / "notebook" / "renamed.qmd").write_text("x")
    ds = _dataset({p: 1 for p in posts})
    report = sp.check_splits(base, ds, blog_root=tmp_path)
    named = [line for line in report if "notebook/p7.qmd" in line]
    assert len(named) == 1 and sp.is_error(named[0]) and "aliases" in named[0]
    assert "1 corpus source(s) missing from disk" in "\n".join(report)
    # The alias is the fix: the same corpus and disk are then clean.
    assert not any(
        "p7" in line or "missing from disk" in line
        for line in sp.check_splits(aliased, ds, blog_root=tmp_path)
    )


def test_check_splits_names_every_corpus_source_missing_from_disk(tmp_path):
    """Every corpus source absent under `blog_root` is reported, once, with the
    spellings tried; a pinned eval post is the eval check's line, not a second
    one here (C12, D34)."""
    splits = _renamed_splits()  # pins notebook/old.qmd, aliased from new.qmd
    splits["aliases"]["post/moved.qmd"] = "post/b.qmd"
    ds = _dataset({"notebook/old.qmd": 2, "post/a.qmd": 2, "post/b.qmd": 2, "post/c.qmd": 2})
    (tmp_path / "notebook").mkdir()
    (tmp_path / "post").mkdir()
    (tmp_path / "post" / "a.qmd").write_text("x")  # present: never reported

    report = sp.check_splits(splits, ds, blog_root=tmp_path)
    corpus_lines = [line for line in report if "corpus source" in line]
    assert all(sp.is_error(line) for line in corpus_lines[:-1])
    assert not any("post/a.qmd" in line for line in report)
    on_disk = [line for line in report if "is not on disk" in line]
    assert sum("notebook/old.qmd" in line for line in on_disk) == 1  # the eval line only
    (b_line,) = [line for line in corpus_lines if "'post/b.qmd'" in line]
    assert "post/moved.qmd" in b_line and "aliases" in b_line  # the spellings tried, the hint
    assert any("'post/c.qmd'" in line for line in corpus_lines)
    assert corpus_lines[-1] == f"2 corpus source(s) missing from disk under {tmp_path}"

    # Put them on disk under their known spellings and the report is clean.
    (tmp_path / "post" / "moved.qmd").write_text("x")
    (tmp_path / "post" / "c.qmd").write_text("x")
    (tmp_path / "notebook" / "new.qmd").write_text("x")
    assert not any(
        "corpus source" in line for line in sp.check_splits(splits, ds, blog_root=tmp_path)
    )


def test_alias_chains_resolve_and_cycles_raise():
    chained = {
        "schema_version": 1,
        "eval_posts": ["a.qmd"],
        "aliases": {"c.qmd": "b.qmd", "b.qmd": "a.qmd"},
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    assert sp.role_of("c.qmd", chained) == "eval"
    cyclic = {**chained, "aliases": {"a.qmd": "b.qmd", "b.qmd": "a.qmd"}}
    with pytest.raises(RunError, match="cycle"):
        sp.role_of("a.qmd", cyclic)


def test_load_splits_rejects_a_malformed_alias_map(tmp_path):
    import json

    path = tmp_path / "splits.json"
    doc = _renamed_splits()
    path.write_text(json.dumps({**doc, "aliases": {"a.qmd": ["b.qmd"]}}))
    with pytest.raises(RunError, match="aliases"):
        sp.load_splits(path)
    # A cycle is caught at load, not deep inside a training run.
    path.write_text(json.dumps({**doc, "aliases": {"a.qmd": "b.qmd", "b.qmd": "a.qmd"}}))
    with pytest.raises(RunError, match="cycle"):
        sp.load_splits(path)


def test_eval_stratum_is_counted_by_identity_not_by_spelling():
    splits = _renamed_splits()
    assert sp.eval_post_spellings(splits) == {
        "notebook/old.qmd": ["notebook/new.qmd", "notebook/old.qmd"]
    }
    # Pairs keyed under either spelling are one eval post, and neither is STALE.
    ds = _dataset({"notebook/old.qmd": 2, "notebook/new.qmd": 2, "post/x.qmd": 2})
    assert _counts(sp.summarize_roles(splits, ds)["eval"]) == {
        "posts": 1,
        "pairs": 4,
        "pairs_real": 4,
        "pairs_synthetic": 0,
    }
    assert not [w for w in sp.check_splits(splits, ds) if w.startswith("STALE")]


def test_missing_eval_posts_is_an_error_not_a_warning(tmp_path):
    (tmp_path / "notebook").mkdir()
    splits = _renamed_splits()
    ds = _dataset({"notebook/old.qmd": 2})

    # Neither spelling on disk -> one MISSING error.
    report = sp.check_splits(splits, ds, blog_root=tmp_path)
    errors = [line for line in report if sp.is_error(line)]
    assert sp.missing_eval_posts(splits, tmp_path) == ["notebook/old.qmd"]
    assert len(errors) == 1 and "notebook/old.qmd" in errors[0]

    # The RENAMED spelling alone is enough: the identity is still on disk.
    (tmp_path / "notebook" / "new.qmd").write_text("x")
    assert sp.missing_eval_posts(splits, tmp_path) == []
    report = sp.check_splits(splits, ds, blog_root=tmp_path)
    assert [line for line in report if sp.is_error(line)] == []


def test_role_of_reads_the_pinned_list_through_the_alias_map():
    """A pin spelled with the post's NEW path still names the same identity.

    `role_of` canonicalised the query and then compared it against the RAW list,
    so a pin under the new spelling matched neither name (F3, D34).
    """
    splits = {
        "schema_version": 1,
        "eval_posts": ["notebook/new.qmd"],
        "aliases": {"notebook/new.qmd": "notebook/old.qmd"},
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    assert sp.role_of("notebook/new.qmd", splits) == "eval"
    assert sp.role_of("notebook/old.qmd", splits) == "eval"


def test_extending_with_a_renamed_post_pins_both_spellings():
    """`extend_eval` stores the identity, not the caller's spelling (F3, D34)."""
    splits = {
        "schema_version": 1,
        "eval_posts": [],
        "aliases": {"notebook/new.qmd": "notebook/old.qmd"},
        "rest_rule": {"kind": "hash", "detector_frac": 0.4, "salt": "0"},
    }
    extended = sp.extend_eval(splits, ["notebook/new.qmd"], note="F3")
    assert extended["eval_posts"] == ["notebook/old.qmd"]
    assert extended["eval_history"][-1]["added"] == ["notebook/old.qmd"]
    assert sp.role_of("notebook/new.qmd", extended) == "eval"
    assert sp.role_of("notebook/old.qmd", extended) == "eval"
    # Re-adding under either spelling is a no-op, not a version churn.
    assert sp.extend_eval(extended, ["notebook/new.qmd"]) is extended
    assert sp.extend_eval(extended, ["notebook/old.qmd"]) is extended


def test_check_splits_pass_reports_then_raises_on_a_missing_pinned_post(tmp_path):
    """The library behind `ai-style`/`dan-style check-splits`: the on-disk check
    needs no corpus, every line goes through `emit`, and a pinned eval post gone
    from `blog_root` is a `RunError` (C12's "missing source") after the report (D47)."""
    from stylebot.draws import ViewOpts

    posts = [f"post/p{i}.qmd" for i in range(10)]
    splits = sp.make_splits(posts, eval_frac=0.2, seed=1)
    path = sp.save_splits(splits, tmp_path / "splits.json")
    opts = sp.CheckSplitsOpts(splits_path=path, blog_root=tmp_path)
    lines: list[tuple[str, str]] = []

    def emit(message: str, /, *, level: str = "info") -> None:
        lines.append((level, message))

    with pytest.raises(RunError, match="2 missing source"):
        sp.check_splits_pass(opts, ViewOpts(), emit=emit)
    assert sum(1 for level, m in lines if level == "error" and m.startswith(sp.ERROR_PREFIX)) == 2
    assert any("eval posts on disk: 0/2" in m for _, m in lines)

    for post in splits["eval_posts"]:
        (tmp_path / post).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / post).write_text("prose\n", encoding="utf-8")
    lines.clear()
    sp.check_splits_pass(opts, ViewOpts(), emit=emit)
    assert any("eval posts on disk: 2/2" in m for _, m in lines)
    assert not [m for level, m in lines if level == "error"]
