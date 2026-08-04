# STYLE_SYSTEM: system prompt for the ai-style fine-tuned prose styler.
# Shared verbatim between pair-logging (ai-style log), synthetic pair generation,
# fine-tune training data, and the ai-style inference CLI. Touch with care:
# changing this string after training pairs are logged invalidates them, and
# `stylebot.pairs.validate_pair_record` will fail the whole corpus until every
# record's system message is re-stamped (`scripts/restamp_style_system.py`).
#
# v2 (2026-08-04) dropped "headings" from the preserve list. Cross-heading
# chunking packs heading lines into the chunk as rewritable CONTENT, so a prompt
# demanding they be preserved verbatim contradicts what the data teaches. The
# level/meaning clause keeps the structural guarantee that actually matters —
# a restyled heading must still be the same heading at the same depth.
STYLE_SYSTEM = (
    "You rewrite prose into Dan Mackinlay's voice. "
    "Preserve all markdown structure (code fences, math, links, list markers, blank lines) verbatim. "
    "Section headings may be restyled, but keep each heading's level and meaning. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten prose, nothing else."
)
