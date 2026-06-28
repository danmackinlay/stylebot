"""Tests for Phase 2 synthetic-pair generation.

Generators are injected, so these exercise the real `synthesize_pairs` /
`iter_targets` path with deterministic fakes — no API keys, no network — and
gate the output through the same `validate_pairs_file` contract Phase 3 uses.
"""

from __future__ import annotations

from stylebot import synth
from stylebot.pairs import validate_pairs_file


def _fake(name: str) -> synth.Generator:
    # Deterministic "slop": tag the text so source != target but stays non-empty.
    return synth.Generator(name=name, generate=lambda text: f"[{name}-slop] {text}")


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


def _make_blog(tmp_path):
    root = tmp_path / "blog"
    (root / "post").mkdir(parents=True)
    (root / "post" / "human.qmd").write_text(HUMAN_POST, encoding="utf-8")
    (root / "post" / "ai.qmd").write_text(AI_POST, encoding="utf-8")
    return root


def test_selector_keeps_only_human_authored(tmp_path):
    root = _make_blog(tmp_path)
    targets = synth.iter_targets(blog_root=root)
    assert targets, "expected human-authored chunks"
    assert {t.source for t in targets} == {"post/human.qmd"}
    assert len(targets) == 2  # two prose paragraphs in the human post


def test_pairs_validate_and_carry_synthetic_meta(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [_fake("claude-x"), _fake("gpt-y")]
    )
    assert result.written == len(targets)

    pairs_path = data_dir / "pairs.jsonl"
    assert validate_pairs_file(pairs_path) == []  # empty == valid

    import json

    recs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    assert all(r["meta"]["synthetic"] is True for r in recs)
    assert all("synthetic" in r["meta"]["tags"] for r in recs)
    # user = slop, assistant = the real target
    for r in recs:
        assert r["messages"][1]["content"] != r["messages"][2]["content"]


def test_rotation_uses_at_least_two_generators(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [_fake("claude-x"), _fake("gpt-y")]
    )
    assert set(result.per_generator) == {"claude-x", "gpt-y"}


def test_idempotent_resume(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)
    gens = [_fake("claude-x"), _fake("gpt-y")]

    first = synth.synthesize_pairs(targets, data_dir, gens)
    assert first.written == len(targets)

    second = synth.synthesize_pairs(targets, data_dir, gens)
    assert second.written == 0
    assert second.skipped_existing == len(targets)

    # File did not grow.
    pairs_path = data_dir / "pairs.jsonl"
    n_lines = len([ln for ln in pairs_path.read_text().splitlines() if ln.strip()])
    assert n_lines == len(targets)


def test_per_generator_mode_doubles_pairs(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [_fake("claude-x"), _fake("gpt-y")], per_generator=True
    )
    assert result.written == 2 * len(targets)


def test_heading_context_prepended_both_sides(tmp_path):
    # A target carrying context -> heading is the verbatim prefix of BOTH the
    # slop (user) and the Dan body (assistant); slop is generated from the body
    # only (the fake echoes its input, so the heading must NOT appear in the
    # generated portion).
    import json

    from stylebot.pairs import validate_pairs_file

    data_dir = tmp_path / "corpus"
    target = synth.Target(
        text="A real paragraph of prose long enough to be a worthwhile target here.",
        source="post/x.qmd",
        chunk_index=0,
        chunk_total=1,
        context="## A Heading",
    )
    # Fake generator echoes the body it is GIVEN — assert it was given the body,
    # not the heading.
    given = {}

    def gen(body):
        given["arg"] = body
        return "[slop] " + body

    result = synth.synthesize_pairs(
        [target], data_dir, [synth.Generator("g", generate=gen)]
    )
    assert result.written == 1
    assert "## A Heading" not in given["arg"]  # generator saw the body only

    pairs_path = data_dir / "pairs.jsonl"
    assert validate_pairs_file(pairs_path) == []  # incl. the context-prefix invariant
    rec = json.loads(pairs_path.read_text().splitlines()[0])
    assert rec["messages"][1]["content"].startswith("## A Heading\n\n")
    assert rec["messages"][2]["content"].startswith("## A Heading\n\n")
    assert rec["meta"]["context"] == "## A Heading"
    assert rec["meta"]["context_mode"] == "immediate"


def test_context_changes_synth_key(tmp_path):
    # Same body, different context -> different synth_key (regenerates).
    from stylebot.synth import _synth_key

    assert _synth_key("g", "body") != _synth_key("g", "body", "## H")


def test_dry_run_writes_nothing(tmp_path):
    root = _make_blog(tmp_path)
    data_dir = tmp_path / "corpus"
    targets = synth.iter_targets(blog_root=root)

    result = synth.synthesize_pairs(
        targets, data_dir, [synth.Generator("claude-x"), synth.Generator("gpt-y")], dry_run=True
    )
    assert result.written == 0
    assert result.planned == len(targets)
    assert not (data_dir / "pairs.jsonl").exists()


def test_pre_selected_files_skip_selector(tmp_path):
    # A pre-selected list is taken as-is — even an AI-touched post — because the
    # caller owns selection in that mode.
    root = _make_blog(tmp_path)
    targets = synth.iter_targets(files=[root / "post" / "ai.qmd"])
    assert targets
    assert {t.source for t in targets} == {str(root / "post" / "ai.qmd")}


# --- chunk hygiene -------------------------------------------------------

POST_WITH_DIV = """---
title: Has a figure div
automation: 0
---

A genuine paragraph of prose that is comfortably longer than the minimum and
carries actual voice worth learning to reproduce in the styler.

:::{#fig-x .figure .illustration}
![](/images/x.png)
:::

A second genuine paragraph, also well past the minimum length, with real
sentences that a human plausibly wrote and edited by hand.
"""

POST_WITH_STUB = """---
title: Has a stub
automation: 0
---

A real paragraph long enough to be kept as a synthesis target, written in a
recognisably human register with some actual content.

🚧TODO🚧 flesh this section out later — it is a stub placeholder and must never
become a synthesis target, even though it is long enough to clear the min-chars bar.
"""


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_prose_only_drops_divs(tmp_path):
    root = tmp_path / "blog"
    _write(root, "post/div.qmd", POST_WITH_DIV)
    targets = synth.iter_targets(blog_root=root)
    assert len(targets) == 2  # two prose paragraphs, the :::div excluded
    assert all("fig-x" not in t.text and "images/x.png" not in t.text for t in targets)


def test_ignore_markers_drops_stub_paragraphs(tmp_path):
    root = tmp_path / "blog"
    _write(root, "post/stub.qmd", POST_WITH_STUB)
    kept = synth.iter_targets(blog_root=root, ignore_markers=["🚧TODO🚧"])
    assert len(kept) == 1
    assert all("🚧TODO🚧" not in t.text for t in kept)
    # Without the marker filter, the stub paragraph survives.
    unfiltered = synth.iter_targets(blog_root=root)
    assert len(unfiltered) == 2


def test_link_dump_dropped(tmp_path):
    root = tmp_path / "blog"
    links = "\n".join(f"* [resource {i}](https://example.com/{i})" for i in range(40))
    body = f"""---
title: Blogroll
automation: 0
---

A normal paragraph with a single [link](https://example.com) that should be kept
because it is mostly prose and only incidentally references one URL out there.

{links}
"""
    _write(root, "post/roll.qmd", body)
    targets = synth.iter_targets(blog_root=root)
    assert len(targets) == 1  # prose kept, the link-dump dropped
    assert "resource 0" not in targets[0].text


def test_max_chars_caps_giant_chunk(tmp_path):
    root = tmp_path / "blog"
    giant = "word " * 3000  # ~15k chars, single paragraph
    body = f"---\ntitle: Giant\nautomation: 0\n---\n\n{giant}\n"
    _write(root, "post/giant.qmd", body)
    assert synth.iter_targets(blog_root=root) == []
    # Raising the cap keeps it.
    assert len(synth.iter_targets(blog_root=root, max_chars=None)) == 1


def test_backward_compat_defaults_unchanged(tmp_path):
    # The original two-paragraph human post still yields exactly two targets.
    root = _make_blog(tmp_path)
    assert len(synth.iter_targets(blog_root=root)) == 2


POST_WITH_INCOMING = """---
title: Has an Incoming dump
automation: 0
---

A real authored paragraph that should survive, long enough to clear the minimum
and carrying genuine voice worth learning from in the styler.

## Incoming

A pile of undigested quotes and links with zero authored signal, long enough to
otherwise look like a real paragraph but sitting below the cut header.

Another undigested blob that must also be excluded along with everything after
the Incoming header, no matter how prose-like it appears.
"""


def test_stop_at_headers_truncates_trailing_section(tmp_path):
    root = tmp_path / "blog"
    _write(root, "post/incoming.qmd", POST_WITH_INCOMING)
    kept = synth.iter_targets(blog_root=root, stop_at_headers=["## Incoming"])
    assert len(kept) == 1  # only the paragraph before "## Incoming"
    assert "authored paragraph that should survive" in kept[0].text
    assert all("undigested" not in t.text for t in kept)
    # Level-agnostic / case-insensitive: "### incoming" also cuts.
    assert len(synth.iter_targets(blog_root=root, stop_at_headers=["incoming"])) == 1
    # Without the option, the post-Incoming paragraphs survive.
    assert len(synth.iter_targets(blog_root=root)) == 3
