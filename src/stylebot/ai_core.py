# STYLE_SYSTEM: system prompt for the (planned) ai-style fine-tuned prose styler.
# Shared verbatim between pair-logging (ai-style-log), synthetic pair generation,
# fine-tune training data, and the eventual ai-style inference CLI. Touch with care:
# changing this string after training pairs are logged invalidates them.
STYLE_SYSTEM = (
    "You rewrite prose into Dan Mackinlay's voice. "
    "Preserve all markdown structure (code fences, math, links, headings, list markers, blank lines) verbatim. "
    "Preserve any 〈MASKED_*〉 tokens verbatim if present. "
    "Return only the rewritten prose, nothing else."
)
