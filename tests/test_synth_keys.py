"""Golden pins for the corpus-identity hashes.

The hex values here are copied from the current implementation ON PURPOSE.
`synth_key` / `capture_id` / `prompt_id` dedupe new generations against every
pair already in the corpus; the rest of the suite only pins them *relationally*
(same inputs -> same key), so a change to the hash recipe itself would pass
everything while silently orphaning the corpus dedup. If one of these fails,
either the recipe moved by accident (fix the code) or you are deliberately
re-keying the corpus (bump the golden value and say so in the commit message).
"""

from __future__ import annotations

from stylebot import synth


def test_synth_key_golden_stateless_defaults():
    # Defaults (strategy/effort/prompt_id/context) feed the key when omitted.
    # Re-keyed 2026-07-20 when DEFAULT_REASONING_EFFORT went high -> off: a
    # flag-less run no longer dedupes against pairs generated before that date.
    assert synth._synth_key("gpt-4o", "Some target paragraph.") == "0cdf691440f1288b"


def test_synth_key_golden_all_axes():
    key = synth._synth_key(
        "openrouter/qwen/qwen3-8b",
        "Some target paragraph.",
        context="## Heading",
        strategy="casual",
        reasoning_effort="high",
        prompt_id="abc123def456",
    )
    assert key == "87f36078686c31cd"


def test_synth_key_golden_replicate():
    # The deliberate-resample axis: a label mints a distinct cell; empty label
    # keys identically to the base corpus (covered by the goldens above).
    # (The session fold this replaced was retired 2026-07-21: session
    # composition is a runtime outcome, so keying on it broke cross-run dedup.)
    key = synth._synth_key(
        "openrouter/qwen/qwen3-8b",
        "Some target paragraph.",
        context="## Heading",
        strategy="casual",
        reasoning_effort="high",
        prompt_id="abc123def456",
        replicate="deep128k",
    )
    assert key == "05145d420de40486"


def test_capture_id_golden():
    assert synth._capture_id("post/human.qmd", "gpt-4o", "casual") == "0d6265ef"


def test_prompt_id_golden():
    assert synth.prompt_id_of("You are a helpful assistant.") == "75357d685f23"


def test_default_key_axes_pinned():
    # These defaults are folded into every key generated without explicit
    # flags; changing either deliberately re-keys default runs.
    assert synth.DEFAULT_STRATEGY == "polish"
    assert synth.DEFAULT_REASONING_EFFORT == "off"
