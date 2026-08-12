from datetime import date, datetime
from pathlib import Path

from ruamel.yaml import YAML

from stylebot import markdown

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

    A "paragraph" is a run of non-blank lines with no internal blank line, so
    list items, headings, and code-fence blocks each become their own block;
    trailing newlines inside a block are stripped.

    This is the **shared** chunk shape for the corpus: Phase 1 (`ai-style log`)
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
# The primitives now live in `stylebot.markdown` (one segmentation, shared by
# every phase). These names stay here because they are PUBLIC API: livingthing
# imports them, and the VS Code marker's parity fixture is asserted against
# them. Re-export, don't relocate.


def segment_for_edit(
    content: str, *, div_is_prose: markdown.DivIsProse | None = None
) -> list[tuple[str, bool]]:
    """Split markdown into ``(text, editable)`` spans, losslessly.

    ``editable`` is False for protected blocks — fenced code, display math,
    ``:::`` divs/callouts, and blockquotes — which must pass through verbatim.
    Concatenating every ``text`` reproduces ``content`` exactly.

    ``div_is_prose`` is the caller's policy for which div classes wrap prose
    (see `stylebot.markdown`); the default protects every div wholesale.
    """
    return markdown.segment_spans(content, div_is_prose=div_is_prose)


def editable_prose(content: str, *, div_is_prose: markdown.DivIsProse | None = None) -> str:
    """Return only the editable prose of ``content`` (protected blocks dropped)."""
    return "".join(
        seg for seg, editable in segment_for_edit(content, div_is_prose=div_is_prose) if editable
    )


# Directories Quarto itself never treats as project inputs. This set mirrors
# upstream rather than inventing policy — which is the whole of what belongs
# here: a *project's* own exclusions are the caller's business, not ours (the
# blog's live in its `_quarto.yml`, and `livingthing.lib` reads them there).
QUARTO_SKIP_DIRS = frozenset({"renv", "packrat", "rsconnect", "node_modules"})


def is_valid_qmd_file(path: Path, base: Path | None = None) -> bool:
    """Return True if this path qualifies as a valid .qmd file.

    The `_`/`.` rule runs over **every** path segment. Until 2026-08-12 it
    checked `path.name` alone — the *filename* — so `_partial.qmd` was excluded
    while `_tmp/scratch.qmd` and `_freeze/…` were walked, and `_site` survived
    only by being hardcoded. That matters more here than it looks: this
    predicate feeds `iter_targets`, so a scratch copy of a post was offered to
    `synth` as prose to generate training pairs from, and `synth_key` hashes
    target text — the resulting pair would be indistinguishable from a real one
    afterwards. Nothing caught it; the corpus is clean only because the blog's
    scratch copies happen to carry `automation: 2` and the selector rejects
    them. See livingthing `_plans/qmd-walker-2026-08.md`.

    Every rule here is about *project-relative* structure, so pass `base` when
    you have one and the judgement is made below it. Without it an absolute
    path is judged whole, which is the historical behaviour.

    Suffix is deliberately not checked: the walk's glob is caller policy
    (`iter_targets(glob=…)`, `ai-style synth --glob`), so this must not
    hardcode `.qmd` the way a single-project walker can.
    """
    parts = path.parts
    if base is not None:
        try:
            parts = path.relative_to(base).parts
        except ValueError:
            pass  # not under `base` — judge the path as given
    if any(part.startswith((".", "_")) for part in parts):
        return False
    if QUARTO_SKIP_DIRS.intersection(parts):
        return False
    return not path.is_dir()


def gather_qmd_files(
    files: list[str],
    base: Path = Path("."),
    default_glob="**/*.qmd",
) -> list[Path]:
    """Return a list of valid QMD file paths from either provided files or default glob.

    With `files`, keep those that exist and pass `is_valid_qmd_file`; otherwise
    walk `base` for `default_glob`. Selection beyond file validity is the
    caller's policy — hand in a pre-filtered list or apply a `selector`
    downstream (see `is_human_authored`).
    """
    if files:
        return [Path(f) for f in files if Path(f).is_file() and is_valid_qmd_file(Path(f))]
    return [p for p in base.glob(default_glob) if is_valid_qmd_file(p, base)]


def is_human_authored(meta: dict, *, field: str = "automation", max_level: int = 0) -> bool:
    """True if frontmatter marks this content as human-written (no/low AI).

    This is the **one** blog-specific seam stylebot is allowed to carry (see
    `_plans/OVERVIEW.md`). Dan's blog records an ``automation`` level per post;
    ``automation: 0`` means no AI was involved, which is the pure-human prose we
    want as training *targets*. Another blog retargets this by passing its own
    ``field`` / ``max_level`` — the defaults encode Dan's convention, nothing more.

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
    supplies the field name and threshold as policy. Robust to the value being a
    YAML ``date``/``datetime`` object or an ISO-8601 string — ISO dates sort
    lexicographically, so the ``YYYY-MM-DD`` prefix is compared. Conservative: a
    missing or malformed date is treated as NOT recent (excluded), matching
    `is_human_authored`'s "unknown means exclude".
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
