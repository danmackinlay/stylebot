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
            ds._add_pair(f"slop {post}.{c}", f"author {post}.{c}", post, synthetic=post in synth_posts)
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
