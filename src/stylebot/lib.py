from ruamel.yaml import YAML
from pathlib import Path
from datetime import date, datetime
import re

yaml = YAML(typ="rt")
yaml.preserve_quotes = True
yaml.width = 10**9
yaml.indent(mapping=2, sequence=2, offset=0)


def read_w_frontmatter_text(text: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from an in-memory string.
    Mirrors read_w_frontmatter(Path) but avoids disk I/O.

    Returns (metadata, content_after_frontmatter).
    """
    if not text:
        return {}, ""

    lines = text.splitlines(keepends=True)
    if not lines:
        return {}, ""

    first = lines[0].rstrip("\r\n")
    if first != "---":
        return {}, text

    # Consume opening ---
    lines = lines[1:]

    to_parse = []
    while lines:
        line = lines.pop(0)
        if line.rstrip("\r\n") in ("---", "..."):
            break
        to_parse.append(line)

    meta = yaml.load("".join(to_parse))
    if meta is None:
        meta = {}
    return meta, "".join(lines)


def split_paragraphs(text: str) -> list[str]:
    """Split prose into paragraphs on blank-line boundaries.

    A "paragraph" is the natural chunk for prose diffing and synthesis: a run
    of non-blank lines with no internal blank line. List items, headings, and
    code-fence blocks each become their own paragraph block. Trailing newlines
    inside a block are stripped.

    This is the **shared** chunk shape for the corpus: Phase 1 (`ai-style-log`)
    diffs against it and Phase 2 (`stylebot.synth`) samples it, so real edit
    pairs and synthetic paraphrase pairs are the same granularity and mixable.
    Keep both producers calling this one splitter.
    """
    paras: list[str] = []
    current: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.strip() == "":
            if current:
                paras.append("".join(current).rstrip("\n"))
                current = []
        else:
            current.append(line)
    if current:
        paras.append("".join(current).rstrip("\n"))
    return paras


# --- Markdown prose/structure segmentation -------------------------------
#
# Ported (generic, stdlib-only) from livingthing's `qmd_core.py`
# `segment_for_edit` — the blog's canonical editable/protected splitter. This is
# generic markdown tooling (no frontmatter / blog knowledge), so per the
# OVERVIEW boundary it lives here alongside `split_paragraphs` / `gather_qmd_files`.
# SIBLING: `livingthing/src/livingthing/qmd_core.py::segment_for_edit` — keep
# behaviour in sync; the segmenter tests in `tests/test_segment.py` pin it.

# Protected spans the styler must not paraphrase (preserved verbatim per
# STYLE_SYSTEM): fenced code, display math, blockquotes.
CODE_FENCE_RE = re.compile(
    r"^[ \t]*```.*?$.*?^[ \t]*```[ \t]*$(?:\n|\Z)", re.MULTILINE | re.DOTALL
)
MATH_BLOCK_RE = re.compile(r"\$\$.*?\$\$[ \t]*(?:\{[^}\n]*\})?", re.DOTALL)
BLOCKQUOTE_RE = re.compile(r"^(?:[ \t]{0,3}>[^\n]*\n?)+", re.MULTILINE)


def _find_div_blocks(content: str) -> list[tuple[int, int]]:
    """Find all ``:::`` div/callout blocks, nesting-aware.

    Returns ``(start, end)`` char spans (end includes the trailing newline).
    Pandoc rule: a closing fence is a line that is exactly ``:::`` (whitespace
    allowed); a ``:::`` line with arguments opens a nested div.
    """
    spans: list[tuple[int, int]] = []
    lines = content.splitlines(keepends=True)
    i = 0
    current_pos = 0
    while i < len(lines):
        line_content = lines[i]
        if line_content.strip().startswith(":::"):
            start_pos = current_pos
            depth = 1
            j = i + 1
            while j < len(lines) and depth > 0:
                inner = lines[j].strip()
                if inner.startswith(":::"):
                    if inner == ":::":
                        depth -= 1
                    else:
                        depth += 1
                j += 1
            if depth == 0:
                end_pos = start_pos + sum(len(lines[k]) for k in range(i, j))
                spans.append((start_pos, end_pos))
                current_pos = end_pos
                i = j
                continue
        current_pos += len(line_content)
        i += 1
    return spans


def segment_for_edit(content: str) -> list[tuple[str, bool]]:
    """Split markdown into ``(text, editable)`` spans, losslessly.

    ``editable`` is False for protected blocks — fenced code, ``$$math$$``,
    ``:::`` divs/callouts, and blockquotes — which must pass through verbatim.
    Concatenating every ``text`` reproduces ``content`` exactly.
    """
    spans: list[tuple[int, int]] = []
    for pattern in (CODE_FENCE_RE, MATH_BLOCK_RE, BLOCKQUOTE_RE):
        for m in pattern.finditer(content):
            spans.append((m.start(), m.end()))
    spans.extend(_find_div_blocks(content))
    spans.sort()

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    out: list[tuple[str, bool]] = []
    i = 0
    for s, e in merged:
        if i < s:
            out.append((content[i:s], True))
        out.append((content[s:e], False))
        i = e
    if i < len(content):
        out.append((content[i:], True))
    return out


def editable_prose(content: str) -> str:
    """Return only the editable prose of ``content`` (protected blocks dropped)."""
    return "".join(seg for seg, editable in segment_for_edit(content) if editable)


def is_valid_qmd_file(path: Path) -> bool:
    """Return True if this path qualifies as a valid .qmd file."""
    if path.is_dir():
        return False
    if path.name.startswith("_"):
        return False
    if any(part.startswith(".") for part in path.parts):
        return False
    if "renv" in path.parts or "_site" in path.parts:
        return False

    return True


def gather_qmd_files(
    files: list[str],
    base: Path = Path("."),
    default_glob="**/*.qmd",
) -> list[Path]:
    """Return a list of valid QMD file paths from either provided files or default glob.

    With `files`, keep those that exist and pass `is_valid_qmd_file`; otherwise
    walk `base` for `default_glob`. Selection beyond file validity (e.g. dropping
    auxiliary or AI-touched posts) is the caller's policy — hand in a pre-filtered
    list or apply a `selector` downstream (see `is_human_authored`).
    """
    if files:
        return [Path(f) for f in files if Path(f).is_file() and is_valid_qmd_file(Path(f))]
    return [p for p in base.glob(default_glob) if is_valid_qmd_file(p)]


def is_human_authored(
    meta: dict, *, field: str = "automation", max_level: int = 0
) -> bool:
    """True if frontmatter marks this content as human-written (no/low AI).

    This is the **one** blog-specific seam stylebot is allowed to carry (see
    `_plans/OVERVIEW.md`). Everything else about loading prose — frontmatter +
    markdown — is generic across Quarto/Hugo/Jekyll/etc. Dan's blog records an
    ``automation`` level per post; ``automation: 0`` means no AI was involved,
    which is exactly the pure-human prose we want as training *targets* and
    reference. Another blog retargets this by passing its own ``field`` /
    ``max_level`` — the defaults encode Dan's convention, nothing more.

    Conservative by design: a missing or unparseable field is treated as NOT
    human-authored. The clean posts are *marked* (low ``automation``); absence
    means "unknown", and we would rather under-select clean prose than poison
    the training corpus with AI-touched text. Returns True only when the field
    is present and parses to ``<= max_level``.
    """
    raw = meta.get(field)
    if raw is None:
        return False
    try:
        return int(raw) <= max_level
    except (TypeError, ValueError):
        return False


def is_modified_after(
    meta: dict, *, field: str = "date-modified", after: str = "2021-01-01"
) -> bool:
    """True if ``meta[field]`` is a date on or after ``after`` (ISO ``YYYY-MM-DD``).

    A generic frontmatter-date gate, parallel to `is_human_authored`: the caller
    supplies the field name and threshold as policy. (Dan's blog uses
    ``date-modified`` to restrict targets to recent, current-voice prose, since his
    style has evolved.) Robust to the value being a YAML ``date``/``datetime`` object
    or an ISO-8601 string — ISO dates sort lexicographically, so the ``YYYY-MM-DD``
    prefix is compared. Conservative: a missing or malformed date is treated as NOT
    recent (excluded), matching `is_human_authored`'s "unknown means exclude".
    """
    raw = meta.get(field)
    if raw is None:
        return False
    iso = raw.isoformat() if isinstance(raw, (date, datetime)) else str(raw).strip()
    prefix = iso[:10]
    # Require a well-formed YYYY-MM-DD prefix before the lexicographic compare, so a
    # garbage string can't sort its way past the threshold.
    if len(prefix) != 10 or prefix[4] != "-" or prefix[7] != "-":
        return False
    return prefix >= after
