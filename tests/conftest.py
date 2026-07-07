"""Shared fixtures for the stylebot test suite."""

from __future__ import annotations

import pytest

HUMAN_POST = """---
title: A Human Post
automation: 0
---

This is a sufficiently long human-authored paragraph about graph theory and the
quiet dignity of well-chosen abstractions, long enough to clear the min-chars bar.

Another paragraph, also comfortably past the minimum length, musing on the way
synthetic data both helps and quietly lies to the model that consumes it.
"""

AI_POST = """---
title: An AI-Touched Post
automation: 2
---

This paragraph was heavily AI-assisted and must never become a training target,
no matter how long and superficially reasonable it manages to appear here.
"""


@pytest.fixture
def make_blog(tmp_path):
    """Factory for a minimal blog tree under ``tmp_path``.

    Default: one human post (two prose paragraphs) plus one AI-touched post the
    selector must exclude. ``ai=False`` drops the AI post; ``many=N`` writes a
    single post of N distinct paragraphs instead (hash-assigned rotations are
    multinomial, so rotation tests need enough targets for every arm to appear).
    """

    def _make(*, ai: bool = True, many: int = 0):
        root = tmp_path / "blog"
        (root / "post").mkdir(parents=True)
        if many:
            paras = "\n\n".join(
                f"Distinct paragraph number {i} about abstraction, long enough to clear "
                f"every character floor we impose on prose chunks in these tests, easily."
                for i in range(many)
            )
            (root / "post" / "many.qmd").write_text(
                f"---\ntitle: Many\nautomation: 0\n---\n\n{paras}\n", encoding="utf-8"
            )
            return root
        (root / "post" / "human.qmd").write_text(HUMAN_POST, encoding="utf-8")
        if ai:
            (root / "post" / "ai.qmd").write_text(AI_POST, encoding="utf-8")
        return root

    return _make
