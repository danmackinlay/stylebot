# STYLE_SYSTEM: the prompt every pair is conditioned on, shared verbatim by
# pair-logging, synthesis, training and inference. The text is frozen and the
# author is a slot: another author is a rendered prompt, not an edit here, and
# a corpus moves to a different prompt as a `ViewPolicy.system_prompt` plus a
# view rebuild — never a migration (D84, D64).
#
# "headings" is deliberately absent from the preserve list: cross-heading
# chunking packs heading lines into the chunk as rewritable CONTENT, so a prompt
# demanding they be preserved verbatim contradicts what the data teaches. The
# level/meaning clause keeps the structural guarantee that actually matters —
# a restyled heading must still be the same heading at the same depth.
DEFAULT_AUTHOR = "Dan Mackinlay"

STYLE_SYSTEM_TEMPLATE = (
    "You rewrite prose into {author}'s voice. "
    "Preserve all markdown structure (code fences, math, links, list markers, blank lines) verbatim. "  # noqa: E501 — frozen prompt text, never reflow
    "Section headings may be restyled, but keep each heading's level and meaning. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten prose, nothing else."
)


def style_system(author: str = DEFAULT_AUTHOR) -> str:
    """The prompt for `author`; the default renders `STYLE_SYSTEM` byte for byte."""
    return STYLE_SYSTEM_TEMPLATE.format(author=author)


STYLE_SYSTEM = style_system()
