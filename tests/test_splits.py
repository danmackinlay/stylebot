"""Tests for the shared by-POST splits contract (`stylebot.splits`).

Pure stdlib — the splits module (and `classify_train.Dataset`, its duck-typed
input) import without any ML dependency, so these run on a minimal checkout.
"""

from __future__ import annotations

import pytest

from stylebot import splits as sp
from stylebot.classify_train import Dataset


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


def test_role_of_memo_does_not_leak_between_splits_docs():
    """No cached pinned-list may answer for the wrong splits doc.

    `role_of` scans `eval_posts` directly and holds no state, so this passes
    trivially today. It stays as the guard on that: a memo was tried and
    reverted (870d030) for making `role_of` non-reentrant and going stale on an
    in-place edit — the last assertion here is the case that broke.
    """
    rule = {"kind": "hash", "detector_frac": 0.4, "salt": "0"}
    a = {"schema_version": 1, "eval_posts": ["post/only_in_a.qmd"], "rest_rule": rule}
    b = {"schema_version": 1, "eval_posts": ["post/only_in_b.qmd"], "rest_rule": rule}
    for _ in range(3):  # alternate so a cached set has to be invalidated each time
        assert sp.role_of("post/only_in_a.qmd", a) == "eval"
        assert sp.role_of("post/only_in_a.qmd", b) != "eval"
        assert sp.role_of("post/only_in_b.qmd", b) == "eval"
        assert sp.role_of("post/only_in_b.qmd", a) != "eval"
    # In-place growth of the pinned list is picked up too.
    a["eval_posts"].append("post/late_pin.qmd")
    assert sp.role_of("post/late_pin.qmd", a) == "eval"


def test_save_load_roundtrip_and_validation(tmp_path):
    splits = sp.make_splits(["post/a.qmd", "post/b.qmd", "post/c.qmd"], eval_frac=0.34, seed=1)
    path = sp.save_splits(splits, tmp_path / "splits.json")
    loaded = sp.load_splits(path)
    assert loaded["eval_posts"] == splits["eval_posts"]
    assert loaded["rest_rule"] == splits["rest_rule"]

    (tmp_path / "bad_version.json").write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema_version"):
        sp.load_splits(tmp_path / "bad_version.json")
    (tmp_path / "bad_rule.json").write_text(
        '{"schema_version": 1, "eval_posts": [], "rest_rule": {"kind": "coin-flip"}}'
    )
    with pytest.raises(ValueError, match="rest_rule"):
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
    assert roles["eval"] == {"posts": 2, "pairs": 3, "pairs_real": 2, "pairs_synthetic": 1}
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
