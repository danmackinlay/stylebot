"""The prose-bearing-div hook, end to end through `iter_targets`.

Mechanism here, policy in the caller: stylebot must not know that
`.callout-note` is a Quarto convention (blog/generic boundary), so `iter_targets`
takes a predicate and the blog supplies it. The default — every `:::` protected
wholesale — is what shipped before, so turning the hook on is a deliberate,
measurable corpus change rather than a silent one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from stylebot import synth_cli
from stylebot.targets import iter_targets

POST = """---
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

POLICY = dict(min_chars=60, drop_link_dumps=False)


def _is_callout(attrs: str) -> bool:
    return "callout" in attrs


@pytest.fixture
def post(tmp_path: Path) -> Path:
    path = tmp_path / "divs.qmd"
    path.write_text(POST, encoding="utf-8")
    return path


def test_default_protects_every_div(post: Path):
    texts = [t.text for t in iter_targets(files=[post], **POLICY)]
    assert len(texts) == 1
    assert "opening paragraph" in texts[0]


def test_policy_admits_callout_prose_only(post: Path):
    texts = [t.text for t in iter_targets(files=[post], div_is_prose=_is_callout, **POLICY)]
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


def test_chunk_opts_carries_the_hook_without_a_flag(post: Path):
    # Library-only field: no `opt()`, so it never appears in --help, but it does
    # reach `iter_targets` through the documented kwargs path.
    opts = synth_cli.ChunkOpts(min_chars=60, drop_link_dumps=False, div_is_prose=_is_callout)
    kwargs = opts.iter_targets_kwargs()
    assert kwargs["div_is_prose"] is _is_callout
    texts = [t.text for t in iter_targets(files=[post], **kwargs)]
    assert any("the author's own voice" in t for t in texts)


def test_div_is_prose_is_not_a_cli_flag():
    import click
    from click.testing import CliRunner

    @click.command()
    @synth_cli.synth_options()
    def cmd(**kw):  # pragma: no cover - never invoked
        pass

    help_text = CliRunner().invoke(cmd, ["--help"]).output
    assert "--div-is-prose" not in help_text
    assert "--cross-heading" in help_text  # the decorator really did wire ChunkOpts
