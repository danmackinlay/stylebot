"""Shared fixtures for the stylebot test suite."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from stylebot.ai_core import STYLE_SYSTEM

SWEEP_ENV = "STYLEBOT_SWEEP_BLOG_ROOT"


def pytest_configure(config):
    config.addinivalue_line("markers", f"sweep: a whole-blog pass; needs {SWEEP_ENV}")


def pytest_collection_modifyitems(config, items):
    """A `sweep` test is deselected, not skipped: the suite's `0 skipped` is a
    real number, and the opt-in path is a selection the run reports as such."""
    if os.environ.get(SWEEP_ENV):
        return
    sweeps = [it for it in items if it.get_closest_marker("sweep")]
    if sweeps:
        config.hook.pytest_deselected(items=sweeps)
        items[:] = [it for it in items if it not in sweeps]


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


@pytest.fixture
def write_pairs_corpus(tmp_path):
    """Factory for a synthetic ``pairs.jsonl``: one record per (post, chunk).

    Every record is the frozen `STYLE_SYSTEM` triple — a slop user-turn, a tight
    assistant-turn — over sources ``post/p{p}.qmd``. The last `n_synth_posts`
    posts carry ``meta.synthetic`` (the `ai-style synth` stratum); the rest look
    like real edit captures.

    Two meta shapes, because two callers need different things and flattening
    them would destroy what one of them tests:

    - ``contract_complete=True`` (default) writes the full `REQUIRED_META_KEYS`
      set, so the file passes `stylebot.pairs.validate_pairs_file` — which
      `assemble_training_corpus` runs by default and raises on.
    - ``contract_complete=False`` writes only ``source``/``chunk_index``/
      ``capture_id``, deliberately short of the contract, to exercise tolerant
      assembly on records that never went through the logger.

    ``transform_sim=True`` stamps the identity covariate on the synthetic
    records (0.95 on chunk 0, a near-copy; 0.4 elsewhere). It is off by default
    because consumers gate on it: `classify_train.assemble_dataset` drops
    near-identity pairs, so stamping it silently changes pair counts.
    """

    def _write(
        path: Path | None = None,
        *,
        n_posts: int = 4,
        chunks: int = 2,
        n_synth_posts: int = 0,
        contract_complete: bool = True,
        transform_sim: bool = False,
    ) -> Path:
        path = Path(path) if path is not None else tmp_path / "pairs.jsonl"
        with path.open("w", encoding="utf-8") as fp:
            for p in range(n_posts):
                for c in range(chunks):
                    meta = {
                        "source": f"post/p{p}.qmd",
                        "capture_id": f"cap{p:04x}",
                        "chunk_index": c,
                    }
                    if contract_complete:
                        meta["captured_at"] = "2026-07-21T00:00:00+00:00"
                        meta["chunk_total"] = chunks
                    if p >= n_posts - n_synth_posts:
                        meta["synthetic"] = True
                        if transform_sim:
                            meta["transform_sim"] = 0.95 if c == 0 else 0.4
                    rec = {
                        "messages": [
                            {"role": "system", "content": STYLE_SYSTEM},
                            {
                                "role": "user",
                                "content": f"It is worth noting that post {p} chunk {c} "
                                "delves into the tapestry.",
                            },
                            {
                                "role": "assistant",
                                "content": f"Post {p} chunk {c}: the tight version.",
                            },
                        ],
                        "meta": meta,
                    }
                    fp.write(json.dumps(rec) + "\n")
        return path

    return _write


def corpus_records(data_dir) -> list[dict]:
    """The records a save or a synth run landed, read back off its draws.

    The identity view of `<data-dir>/draws` — what those commands appended to
    `pairs.jsonl` before W4 step 5 stopped them writing one (D84).
    """
    from stylebot import config, draws

    every = draws.iter_draws(config.draws_path(data_dir))
    return list(draws.view_records(every, draws.ViewPolicy()))


def corpus_file(data_dir, out) -> Path:
    """`corpus_records` written as one `pairs.jsonl` at `out`, in draw order —
    for a reader that wants the corpus as a path."""
    from stylebot.jsonl import dumps_line

    body = "".join(dumps_line(rec) + "\n" for rec in corpus_records(data_dir))
    Path(out).write_text(body, encoding="utf-8")
    return Path(out)
