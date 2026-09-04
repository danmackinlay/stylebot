"""Phase 2 target selection: what ``iter_targets`` keeps and what it drops.

Carved verbatim out of ``test_synth.py``. The selector policy plus the chunk
hygiene rules (divs, stub markers, link dumps, size caps, cut headers).
"""

from __future__ import annotations

from stylebot import synth
from stylebot.targets import iter_targets


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


# --- the prose-bearing-div hook, end to end ------------------------------
#
# Mechanism here, policy in the caller: stylebot must not know that
# `.callout-note` is a Quarto convention (blog/generic boundary), so
# `iter_targets` takes a predicate and the blog supplies it. The default --
# every `:::` protected wholesale -- is what shipped before, so turning the hook
# on is a deliberate, measurable corpus change rather than a silent one.

POST_WITH_DIVS = """---
title: Divs
automation: 0
---

An opening paragraph of ordinary prose, long enough to survive the floor and
be selected as a target in its own right.

::: {.callout-note}
A callout paragraph that is the author's own voice, merely sitting inside a
container whose fence lines carry class semantics.

::: {#fig-inner}
![](inner.png)
:::

More callout prose after the nested figure, still the author's voice and still
long enough to clear the minimum.
:::

::: {#fig-outer}
![](outer.png)

An ordinary figure caption that is markup, not a passage of prose to restyle.
:::
"""

DIV_POLICY = dict(min_chars=60, drop_link_dumps=False)


def _is_callout(attrs: str) -> bool:
    return "callout" in attrs


def test_default_protects_every_div(tmp_path):
    post = _write(tmp_path, "divs.qmd", POST_WITH_DIVS)
    texts = [t.text for t in iter_targets(files=[post], **DIV_POLICY)]
    assert len(texts) == 1
    assert "opening paragraph" in texts[0]


def test_policy_admits_callout_prose_only(tmp_path):
    post = _write(tmp_path, "divs.qmd", POST_WITH_DIVS)
    texts = [t.text for t in iter_targets(files=[post], div_is_prose=_is_callout, **DIV_POLICY)]
    joined = "\n---\n".join(texts)
    assert "opening paragraph" in joined
    assert "the author's own voice" in joined
    assert "after the nested figure" in joined
    # The container's own lines, the nested figure, and an unrelated figure div
    # are all still excluded.
    assert "callout-note" not in joined
    assert "fig-inner" not in joined
    assert "inner.png" not in joined
    assert "fig-outer" not in joined
    assert "figure caption that is markup" not in joined


# --- chunk hygiene -------------------------------------------------------


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
