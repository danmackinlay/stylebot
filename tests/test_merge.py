"""Tests for section-aware paragraph merging and the prose-residual link guard."""

from __future__ import annotations

from stylebot import synth
from stylebot.targets import _is_link_list, _is_list_item, _pack_paragraphs, _split_sections


def _write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


P1 = "First real paragraph, long enough on its own to be a fine standalone target with voice."
P2 = "Second paragraph continuing the same thought, also comfortably past the minimum length here."
P3 = "Third paragraph rounding out the section with another substantial sentence of genuine prose."


# --- unit-level helpers ---------------------------------------------------


def test_split_sections_excludes_headers_and_carries_heading():
    body = "pre\n\n## A\n\npara a\n\n### B\n\npara b\n"
    secs = _split_sections(body)  # list of (heading, body)
    bodies = "\n".join(b for _, b in secs)
    assert "## A" not in bodies and "### B" not in bodies  # headers excluded from bodies
    assert ("", "pre\n") in [(h, b) for h, b in secs] or any(
        h == "" and "pre" in b for h, b in secs
    )
    # Each section carries its immediate heading.
    by_heading = {h: b for h, b in secs}
    assert "para a" in by_heading["## A"]
    assert "para b" in by_heading["### B"]


def test_pack_respects_budget():
    paras = ["x" * 100] * 10
    blocks = _pack_paragraphs(paras, merge_max_chars=250)
    assert all(len(b) <= 250 for b in blocks)
    assert len(blocks) > 1


def test_pack_keeps_oversized_paragraph_whole():
    big = "y" * 400
    blocks = _pack_paragraphs([big], merge_max_chars=250)
    assert blocks == [big]  # never split mid-paragraph


def test_is_list_item():
    assert _is_list_item("* a\n* b\n1. c")
    assert not _is_list_item("a normal sentence")


def test_is_link_list_prose_residual():
    # Pure link list -> dropped.
    assert _is_link_list("[A](http://a) [B](http://b) [C](http://c) [D](http://d)")
    # Link-dense PROSE -> kept (URLs would fool a density metric).
    prose = (
        "I recommend [The Gradient](https://thegradient.pub) which is genuinely "
        "excellent and worth your time and attention."
    )
    assert not _is_link_list(prose)


# --- iter_targets merge mode ---------------------------------------------


def test_merge_packs_short_paragraphs(tmp_path):
    root = tmp_path / "blog"
    short = (
        "A short take, under two hundred characters but real prose with voice and intent indeed."
    )
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{short}\n\n{short}\n\n{short}\n"
    _write(root, "post/s.qmd", body)
    # Non-merge with floor 200 drops all three short paras.
    assert synth.iter_targets(blog_root=root, min_chars=200) == []
    # Merge packs them into one passage above the floor.
    merged = synth.iter_targets(blog_root=root, min_chars=200, merge=True, merge_max_chars=1500)
    assert len(merged) == 1
    assert merged[0].text.count("A short take") == 3
    # The chunk-scale facet rides on the target (and thence into pair meta),
    # so mixed-scale corpora stay ablatable by scale.
    assert merged[0].merge_max_chars == 1500
    unmerged = synth.iter_targets(blog_root=root, min_chars=50)
    assert unmerged and all(t.merge_max_chars is None for t in unmerged)


def test_merge_does_not_cross_section_headers(tmp_path):
    root = tmp_path / "blog"
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{P1}\n\n## Section Two\n\n{P2}\n"
    _write(root, "post/h.qmd", body)
    blocks = synth.iter_targets(blog_root=root, min_chars=50, merge=True, merge_max_chars=5000)
    assert len(blocks) == 2  # not packed across the header
    assert all("Section Two" not in b.text and "##" not in b.text for b in blocks)


def test_merge_respects_budget_within_section(tmp_path):
    root = tmp_path / "blog"
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{P1}\n\n{P2}\n\n{P3}\n"
    _write(root, "post/b.qmd", body)
    blocks = synth.iter_targets(blog_root=root, min_chars=50, merge=True, merge_max_chars=200)
    assert len(blocks) >= 2  # budget forces a break
    assert all(len(b.text) <= 200 for b in blocks)


def test_merge_oversized_single_paragraph_kept_whole(tmp_path):
    root = tmp_path / "blog"
    big = "word " * 80  # ~400 chars, one paragraph
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{big.strip()}\n"
    _write(root, "post/o.qmd", body)
    blocks = synth.iter_targets(blog_root=root, min_chars=50, merge=True, merge_max_chars=200)
    assert len(blocks) == 1
    assert len(blocks[0].text) > 200  # emitted whole, not split


def test_merge_max_chars_hard_drop(tmp_path):
    root = tmp_path / "blog"
    giant = "word " * 3000  # ~15k chars
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{giant.strip()}\n"
    _write(root, "post/g.qmd", body)
    assert synth.iter_targets(blog_root=root, merge=True, max_chars=8000) == []


def test_merge_drops_packed_link_list(tmp_path):
    root = tmp_path / "blog"
    # Bare link lines, each short (under any density floor), no bullets.
    links = "\n\n".join(f"[Resource {i}](https://example.com/{i})" for i in range(30))
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{P1}\n\n{links}\n"
    _write(root, "post/l.qmd", body)
    blocks = synth.iter_targets(blog_root=root, min_chars=50, merge=True, merge_max_chars=1500)
    assert all("Resource 0" not in b.text for b in blocks)
    assert any("First real paragraph" in b.text for b in blocks)


def test_merge_chunk_index_sequential(tmp_path):
    root = tmp_path / "blog"
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{P1}\n\n## S\n\n{P2}\n"
    _write(root, "post/c.qmd", body)
    blocks = synth.iter_targets(blog_root=root, min_chars=50, merge=True, merge_max_chars=300)
    assert [b.chunk_index for b in blocks] == list(range(len(blocks)))
    assert all(b.chunk_total == len(blocks) for b in blocks)


def test_heading_context_attached_immediate(tmp_path):
    root = tmp_path / "blog"
    body = f"---\ntitle: t\nautomation: 0\n---\n\n{P1}\n\n## A Section\n\n{P2}\n"
    _write(root, "post/hc.qmd", body)
    # Default: no context.
    plain = synth.iter_targets(blog_root=root, min_chars=50, merge=True, merge_max_chars=5000)
    assert all(t.context == "" for t in plain)
    # immediate: the section's paragraph carries its heading; preamble stays "".
    ctx = synth.iter_targets(
        blog_root=root, min_chars=50, merge=True, merge_max_chars=5000, heading_context="immediate"
    )
    by_ctx = {t.context for t in ctx}
    assert "## A Section" in by_ctx  # P2's section heading
    assert "" in by_ctx  # P1 preamble has no heading


def test_drop_list_items_both_modes(tmp_path):
    root = tmp_path / "blog"
    body = (
        "---\ntitle: t\nautomation: 0\n---\n\n"
        f"{P1}\n\n"
        "* a bulleted item that is long enough on its own to clear the eighty character "
        "minimum bar\n"
    )
    _write(root, "post/li.qmd", body)
    # Non-merge: the bullet survives by default, dropped with drop_list_items.
    assert len(synth.iter_targets(blog_root=root)) == 2
    assert len(synth.iter_targets(blog_root=root, drop_list_items=True)) == 1


# --- cross-heading mode (v2 longer rewrites) ------------------------------


def test_cross_heading_packs_across_sections_with_headings_inline(tmp_path):
    root = tmp_path / "blog"
    body = (
        f"---\ntitle: t\nautomation: 0\n---\n\n{P1}\n\n"
        f"## Section Two\n\n{P2}\n\n### Section Three\n\n{P3}\n"
    )
    _write(root, "post/x.qmd", body)
    blocks = synth.iter_targets(
        blog_root=root, min_chars=50, merge=True, merge_max_chars=5000, cross_heading=True
    )
    assert len(blocks) == 1  # the whole document packed into one passage
    text = blocks[0].text
    # Headings are CONTENT now, in document order, interleaved with their prose.
    assert "## Section Two" in text and "### Section Three" in text
    assert text.index(P1) < text.index("## Section Two") < text.index(P2)
    assert text.index(P2) < text.index("### Section Three") < text.index(P3)
    assert blocks[0].chunk_mode == "cross-heading"


def test_cross_heading_forces_context_off(tmp_path):
    """An inline heading is text the styler rewrites, so it cannot also be
    fixed context — otherwise the heading would appear twice in the pair and
    the verbatim-prefix invariant in stylebot.pairs would be a lie."""
    root = tmp_path / "blog"
    body = f"---\ntitle: t\nautomation: 0\n---\n\n## A Section\n\n{P1}\n"
    _write(root, "post/y.qmd", body)
    blocks = synth.iter_targets(
        blog_root=root, min_chars=50, merge=True, merge_max_chars=5000,
        cross_heading=True, heading_context="immediate",
    )
    assert all(t.context == "" for t in blocks)
    assert "## A Section" in blocks[0].text


def test_cross_heading_requires_merge(tmp_path):
    """Without --merge there is no packing to do, so the flag is inert and the
    section-bounded contract (headings excluded, context available) holds."""
    root = tmp_path / "blog"
    body = f"---\ntitle: t\nautomation: 0\n---\n\n## A Section\n\n{P1}\n"
    _write(root, "post/z.qmd", body)
    blocks = synth.iter_targets(
        blog_root=root, min_chars=50, cross_heading=True, heading_context="immediate"
    )
    assert all(t.chunk_mode is None for t in blocks)
    assert all("##" not in t.text for t in blocks)
    assert blocks[0].context == "## A Section"


def test_cross_heading_is_what_buys_length_not_the_budget(tmp_path):
    """The load-bearing finding behind v2: section-bounded packing hits the
    SECTION wall, not the budget, so raising merge_max_chars alone barely moves
    the distribution. Modelled here as many short sections."""
    root = tmp_path / "blog"
    sections = "\n\n".join(f"## S{i}\n\n{P1}" for i in range(8))
    _write(root, "post/w.qmd", f"---\ntitle: t\nautomation: 0\n---\n\n{sections}\n")
    kw = dict(blog_root=root, min_chars=50, merge=True)
    # Section-bounded: tripling the budget changes nothing — every section is
    # already under it, and no block may cross a header.
    small = synth.iter_targets(**kw, merge_max_chars=1500)
    big = synth.iter_targets(**kw, merge_max_chars=4500)
    assert len(small) == len(big) == 8
    # Cross-heading: the same budget now actually binds.
    crossed = synth.iter_targets(**kw, merge_max_chars=4500, cross_heading=True)
    assert len(crossed) < len(small)
    assert max(len(t.text) for t in crossed) > max(len(t.text) for t in small)
