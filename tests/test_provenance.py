"""Tests for the provenance stamps carried by every committed artifact.

The interesting one is `policy_digest`: it has to hash whatever chunking policy
the producer happens to hold, including a `div_is_prose` PREDICATE, and stay
byte-stable across processes so two runs a month apart are comparable.
"""

from __future__ import annotations

import os
import subprocess
import sys

from stylebot.markdown import is_heading_line
from stylebot.provenance import (
    PROVENANCE_KEYS,
    git_repo_root,
    pair_provenance,
    policy_digest,
    repo_provenance,
    stylebot_sha,
)
from stylebot.targets import ChunkPolicy


def _prose(attrs: str) -> bool:
    return "callout" in attrs


def _other_prose(attrs: str) -> bool:
    return "note" in attrs


def test_policy_digest_is_stable_and_value_keyed():
    a = ChunkPolicy(merge=True, merge_max_chars=4500, cross_heading=True)
    b = ChunkPolicy(merge=True, merge_max_chars=4500, cross_heading=True)
    assert policy_digest(a) == policy_digest(b)
    assert policy_digest(a) != policy_digest(ChunkPolicy(merge=True, merge_max_chars=1500))
    assert policy_digest(None) is None


def test_policy_digest_sees_the_div_is_prose_predicate():
    """The A1-class asymmetry the digest exists to make visible: two runs that
    differ ONLY in whether callout prose was freed into targets must not share a
    digest. A predicate has no JSON form and `repr()` carries a memory address,
    so it is hashed by module.qualname."""
    base = ChunkPolicy(merge=True)
    with_prose = ChunkPolicy(merge=True, div_is_prose=_prose)
    other = ChunkPolicy(merge=True, div_is_prose=_other_prose)
    assert policy_digest(base) != policy_digest(with_prose)
    assert policy_digest(with_prose) != policy_digest(other)
    # Same function, a second policy object: same digest.
    assert policy_digest(with_prose) == policy_digest(ChunkPolicy(merge=True, div_is_prose=_prose))


def test_policy_digest_survives_a_process_boundary():
    """A digest recorded in last month's corpus must still match today's run, so
    it cannot depend on anything per-process (`id()`, hash randomisation).

    The predicate is an ordinary package function here, which is also the one
    caveat worth knowing: the digest is keyed on `module.qualname`, so moving
    `div_is_prose` to another module changes it — correctly, that is a policy
    change — and a predicate defined in a `__main__` script or a locally-imported
    test module is not comparable with the same code imported as a package.
    """
    code = (
        "from stylebot.markdown import is_heading_line\n"
        "from stylebot.provenance import policy_digest\n"
        "from stylebot.targets import ChunkPolicy\n"
        "print(policy_digest(ChunkPolicy(merge=True, div_is_prose=is_heading_line)))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": "1"},
    )
    assert out.returncode == 0, out.stderr
    expected = policy_digest(ChunkPolicy(merge=True, div_is_prose=is_heading_line))
    assert out.stdout.strip() == expected


def test_policy_digest_takes_a_mapping_too():
    """Captures hold cut knobs, not a `ChunkPolicy`; both are hashable here, and
    key ORDER is not part of the answer."""
    assert policy_digest({"gap_paras": 1, "gap_chars": 200}) == policy_digest(
        {"gap_chars": 200, "gap_paras": 1}
    )
    assert policy_digest({"gap_paras": 1}) != policy_digest({"gap_paras": 2})


def test_pair_provenance_always_carries_all_five_keys():
    prov = pair_provenance()
    assert sorted(prov) == sorted(PROVENANCE_KEYS)
    assert all(v is None for v in prov.values())


def test_repo_provenance_on_a_real_checkout():
    prov = repo_provenance(__file__)
    assert prov["blog_sha"], "stylebot's own tree should resolve a SHA"
    assert isinstance(prov["blog_dirty"], bool)
    assert prov["stylebot_sha"] == stylebot_sha()


def test_repo_provenance_outside_a_checkout_is_all_none(tmp_path, monkeypatch):
    prov = repo_provenance(tmp_path)
    assert prov["blog_sha"] is None and prov["blog_dirty"] is None


def test_git_repo_root_walks_up_from_a_file(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    leaf = tmp_path / "a" / "b" / "post.qmd"
    leaf.write_text("x", encoding="utf-8")
    assert git_repo_root(leaf) == tmp_path.resolve()
    assert git_repo_root(tmp_path / "a") == tmp_path.resolve()
