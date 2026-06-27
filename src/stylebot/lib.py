import logging
from ruamel.yaml import YAML
from pathlib import Path
import os
import re
from datetime import datetime, date, timezone
from io import StringIO

from dateutil import parser as _dateutil_parser

yaml = YAML(typ="rt")
yaml.preserve_quotes = True
yaml.width = 10**9
yaml.indent(mapping=2, sequence=2, offset=0)


def configure_logging(log_level: str):
    """Configure logging with specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Invalid log level: {log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    # Remove existing handlers if any
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")


def write_w_frontmatter(fname, metadata, content, strip_trailing_spaces=False):
    """
    Write a file with YAML frontmatter and content.

    Args:
        fname: Path to the file to write
        metadata: YAML metadata (typically a CommentedMap for preservation)
        content: Content to write after frontmatter
        strip_trailing_spaces: Whether to strip trailing spaces/tabs from YAML lines
                               (default False to preserve original formatting)
    """
    # 1) Render YAML to a string
    buf = StringIO()
    yaml.dump(metadata, buf)
    yml = buf.getvalue()

    # 2) Optionally strip trailing spaces/tabs at end-of-line (only inside YAML)
    #    Preserve all other spaces (esp. inside quoted scalars)
    if strip_trailing_spaces:
        yml = re.sub(r"[ \t]+(?=\r?\n)", "", yml)

    # 3) Write with exactly one blank line after the closing ---
    target = Path(fname)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf8") as fp:
        fp.write("---\n")
        fp.write(yml)
        fp.write("---\n\n")
        # Avoid double-blank if content already starts with NL
        fp.write(content.lstrip("\n"))
    # Atomic replace (Windows-safe since Python 3.3)
    os.replace(tmp, target)


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


def read_w_frontmatter(fname: Path) -> tuple[dict, str]:
    metadata = {}
    with open(fname, "r", encoding="utf8") as fp:
        lines = fp.readlines()

    if len(lines) == 0:
        return {}, ""

    if lines[0] == ("---\n"):  # YAML header
        lines = lines[1:]
        # Load the data we need to parse
        to_parse = []
        while lines:
            line = lines.pop(0)
            # When we find a terminator (`---` or `...`), stop.
            if line in ("---\n", "...\n"):
                break
            # Otherwise, keep adding the lines to the parseable.
            to_parse.append(line)

        metadata = yaml.load("".join(to_parse))

    return metadata, "".join(lines)


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
    skip_auxiliary: bool = False,
) -> list[Path]:
    """Return a list of valid QMD file paths from either provided files or default glob.

    When `skip_auxiliary=True`, drop files whose frontmatter marks them as
    auxiliary content (`type: digest`, `type: about`). This is the shared
    filter for text-/metadata-munging commands that have no business rewriting
    AI-generated digests; commands producing physical artifacts (thumbnails,
    redirects) should leave it off.
    """
    file_list = []
    if files:
        candidates = [Path(f) for f in files if Path(f).is_file() and is_valid_qmd_file(Path(f))]
    else:
        candidates = [p for p in base.glob(default_glob) if is_valid_qmd_file(p)]

    if not skip_auxiliary:
        return candidates

    for p in candidates:
        try:
            meta, _ = read_w_frontmatter_text(p.read_text(encoding="utf-8"))
        except Exception:
            file_list.append(p)
            continue
        if is_auxiliary_post(meta):
            continue
        file_list.append(p)
    return file_list


def get_file_mod_time(path: Path) -> float:
    """
    Gets the filesystem modification time of a file.

    NEVER call this on a `.qmd` source file. Filesystem mtime on a `.qmd`
    reflects whatever tool last rewrote it (a formatter, an AI-preen pass, this
    header-img massager), not the author's intent, so any freshness decision
    keyed on it is non-deterministic and self-invalidating. The canonical
    "modified time" of a `.qmd` is its frontmatter, resolved as
    `metadata.get("date-modified") or metadata.get("date")` (see frecency.py).
    Filesystem mtime is only legitimate for derived/non-authored artefacts
    (rendered HTML, generated thumbnails, lock files, IPC sentinels).
    """
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def parse_frontmatter_datetime(value) -> datetime | None:
    """Parse a frontmatter date value into a timezone-aware ``datetime``.

    Returns ``None`` if the value is absent or unparseable. The result is
    ALWAYS timezone-aware, so callers can compare and subtract without the
    ``TypeError`` Python raises when mixing naive and aware datetimes — that
    foot-gun is the whole reason this lives in one place.

    Timezone policy: frontmatter dates carry an explicit offset (e.g.
    ``2021-03-02T12:33:03+11:00``); a value with no offset is assumed UTC.
    That matches every comparison site that predated this helper
    (``qmd_core.newer``, ``frecency.calculate_age_days``, the digest mailout
    freshness check), so consolidating here changes no behaviour. Naive
    frontmatter dates are rare precisely because the writers emit offsets.

    Accepts the shapes a frontmatter value actually takes: an ISO-8601 string,
    or a ``datetime`` / ``date`` already deserialized by the YAML loader.

    NB this is for *content* dates (``date-modified`` / ``date`` and the
    ``date-ai-*`` gates) — never derive a ``.qmd``'s modified time from the
    filesystem; see ``get_file_mod_time``.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):  # bare YAML date (datetime is a date subclass)
        dt = datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            dt = _dateutil_parser.parse(s)
        except (ValueError, OverflowError):
            return None
    else:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def migrate_ai_dates(meta: dict) -> bool:
    """Copy legacy date-ai-modified -> date-ai-edit/summary once. Returns True if mutated."""
    old = meta.get("date-ai-modified")
    if not old:
        return False
    changed = False
    if not meta.get("date-ai-edit"):
        meta["date-ai-edit"] = old
        changed = True
    if not meta.get("date-ai-summary"):
        meta["date-ai-summary"] = old
        changed = True
    return changed


# Post types that are auxiliary/meta-content, not main blog posts
AUXILIARY_TYPES = frozenset({"digest", "about"})


def is_auxiliary_post(meta: dict) -> bool:
    """
    Return True if this post is auxiliary content (digest, about page, etc.)
    that should be excluded from similarity indexing and AI preening.

    Auxiliary posts have different processing rules than regular blog content.
    """
    post_type = str(meta.get("type", "")).strip().lower()
    return post_type in AUXILIARY_TYPES


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
