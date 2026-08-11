"""Phase 2 target selection: what ``iter_targets`` keeps and what it drops.

Carved verbatim out of ``test_synth.py``. The selector policy plus the chunk
hygiene rules (divs, stub markers, link dumps, size caps, cut headers).
"""

from __future__ import annotations

from stylebot import synth


def test_selector_keeps_only_human_authored(make_blog):
    root = make_blog()
    targets = synth.iter_targets(blog_root=root)
    assert targets, "expected human-authored chunks"
    assert {t.source for t in targets} == {"post/human.qmd"}
    assert len(targets) == 2  # two prose paragraphs in the human post


def test_pre_selected_files_skip_selector(tmp_path, make_blog):
    # A pre-selected list is taken as-is — even an AI-touched post — because the
    # caller owns selection in that mode.
    root = make_blog()
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


def test_backward_compat_defaults_unchanged(tmp_path, make_blog):
    # The original two-paragraph human post still yields exactly two targets.
    root = make_blog()
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
