"""Signal 5's stratum — the boundary cases a fit judge compares (D97).

The other four signals score a passage in isolation, so a rewrite that strands
a pronoun or restarts a metaphor scores clean on every one of them. Here the
unit of judgment is a **boundary**: a span plus the prose flanking it in the
real document. The comparison, the summary and the `judge-fit` pass are
`stylebot.fit_judge`; design, strata and rejected alternatives are D97 and
`_plans/fit-judge.md`.

"Fit", not "seam": this package already spends *seam* on a substitutable
interface, so the newcomer yielded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stylebot import markdown
from stylebot.capture import archive_id_of
from stylebot.infer import _segment, split_frontmatter
from stylebot.jsonl import iter_jsonl
from stylebot.pairs import FIELD_EXTRACTORS, extract_target, iter_pairs, record_id

# ---------------------------------------------------------------------------
# The stratum
# ---------------------------------------------------------------------------

#: An item of a segmented body: `(kind, text, start, end)`, offsets into it.
Item = tuple[str, str, int, int]


@dataclass(frozen=True)
class Cut:
    """The diet a stratum is cut at; pass the one being served, or the stratum
    measures a chunking policy nobody runs."""

    max_chars: int
    inline_headings: bool = False
    div_is_prose: markdown.DivIsProse | None = None
    html_comments: bool = False

    def segment(self, body: str) -> list[Item]:
        """`_segment`'s items re-anchored to byte offsets in `body`, which is
        exact because items interleaved with separators reproduce it."""
        items, seps = _segment(
            body,
            max_chars=self.max_chars,
            inline_headings=self.inline_headings,
            div_is_prose=self.div_is_prose,
            html_comments=self.html_comments,
        )
        out: list[Item] = []
        pos = 0
        for i, (kind, text) in enumerate(items):
            if i:
                pos += len(seps[i - 1])
            out.append((kind, text, pos, pos + len(text)))
            pos += len(text)
        return out


@dataclass(frozen=True)
class FitCase:
    """One boundary: a span with the prose that flanks it in the document.

    `before`/`after` are the nearest **prose** blocks: a protected block is
    skipped rather than used as a neighbour, and a heading breaks adjacency
    outright, since under a section-bounded diet it is the span's `context`.
    `neighbour_source` names which of `NEIGHBOUR_SOURCES` placed them, and is
    empty for a document cut, whose neighbours are the document's own.
    """

    id: str
    source: str
    index: int
    before: str
    span: str
    after: str
    context: str = ""
    neighbour_source: str = ""

    @property
    def is_interior(self) -> bool:
        """Prose on BOTH sides. A one-sided case is still judgeable, so the
        stratum keeps it and counts it separately."""
        return bool(self.before.strip()) and bool(self.after.strip())


def fit_asdict(case: FitCase) -> dict:
    return asdict(case)


def _neighbours(items: Sequence[Item], idx: int) -> tuple[str, str]:
    """The nearest prose flanking `items[idx]`, stopping at a heading.

    Stripped: a neighbour is read, never reassembled, and a trailing newline
    would make two identical neighbours compare unequal.
    """

    def walk(rng: Iterable[int]) -> str:
        for j in rng:
            kind, text, _, _ = items[j]
            if kind == "heading":
                return ""
            if kind in markdown.PROTECTED_KINDS:
                continue  # not prose: not a neighbour, not a barrier either
            if text.strip():
                return text.strip()
        return ""

    return walk(range(idx - 1, -1, -1)), walk(range(idx + 1, len(items)))


def fits_from_document(text: str, source: str, *, cut: Cut) -> list[FitCase]:
    """Cut a document into fit cases: deterministic, re-derivable and
    adapter-independent, so one stratum contrasts any two arms."""
    _, body = split_frontmatter(text)
    items = cut.segment(body)
    cases: list[FitCase] = []
    context = ""
    n = 0
    for idx, (kind, chunk, _, _) in enumerate(items):
        if kind == "heading":
            if not cut.inline_headings:
                context = chunk.split("\n")[-1]
            continue
        if kind in markdown.PROTECTED_KINDS or not chunk.strip():
            continue
        before, after = _neighbours(items, idx)
        cases.append(
            FitCase(
                id=f"{source}#{n}",
                source=source,
                index=n,
                before=before,
                span=chunk,
                after=after,
                context=context,
            )
        )
        n += 1
    return cases


def locate_neighbours(post_text: str, span_text: str, *, cut: Cut) -> tuple[str, str] | None:
    """Find `span_text` in `post_text` by content and return the prose flanking
    it, or None when it is absent — a miss the caller reports, never guesses.

    Located by content, not by re-deriving the cut: the target side of a pair
    is Dan's published prose, so it is findable verbatim, and matching on it is
    stable under a chunk policy that has since moved.
    """
    _, body = split_frontmatter(post_text)
    needle = span_text.strip()
    if not needle:
        return None
    at = body.find(needle)
    if at < 0:
        return None
    items = cut.segment(body)
    end = at + len(needle)
    # A merged chunk may cover several items; take the outermost overlapping
    # ones so neighbours are measured from the span's real edges.
    overlapping = [i for i, (_, _, s, e) in enumerate(items) if s < end and e > at]
    if not overlapping:
        return None
    before, _ = _neighbours(items, overlapping[0])
    _, after = _neighbours(items, overlapping[-1])
    return before, after


# ---------------------------------------------------------------------------
# The corpus stratum — two rewrite JSONLs, joined and placed
# ---------------------------------------------------------------------------

#: Where a case's neighbours came from, best first; recorded per case, since a
#: stratum that mixes provenance must say so.
NEIGHBOUR_SOURCES = ("sibling", "archive", "post")


@dataclass
class StratumReport:
    """What the join placed, from where, and what it dropped. A stratum that
    quietly shrank reads exactly like one that was always small, so every count
    here exists to be printed."""

    n_a: int = 0
    n_b: int = 0
    n_joined: int = 0
    n_built: int = 0
    by_neighbour_source: dict[str, int] = field(default_factory=dict)
    unplaceable: list[str] = field(default_factory=list)
    empty_output: list[str] = field(default_factory=list)
    only_in_a: int = 0
    only_in_b: int = 0


def _cut_key(meta: dict) -> tuple:
    """Identity of the CUT a chunk belongs to — everything but its position.

    `chunk_total` is in the key because index 4 of a 29-chunk cut is not index
    4 of a 12-chunk one, and without it a mixed-scale corpus pairs chunks from
    different cuts of the same post.
    """
    return (
        meta.get("source"),
        meta.get("chunk_total"),
        meta.get("merge_max_chars"),
        meta.get("chunk_mode"),
    )


def build_sibling_index(corpus_path: str | Path) -> dict[tuple, tuple[str, str]]:
    """`(cut, chunk_index) -> (Dan's target, its heading context)`, over a view
    or a scratch pairs file.

    Synthetic records only: `capture.diff_chunks` emits the *changed* regions of
    an edit and stores no offsets, so consecutive real `chunk_index` values can
    sit pages apart and sibling adjacency there would be fabricated.
    """
    index: dict[tuple, tuple[str, str]] = {}
    for rec in iter_pairs(corpus_path):
        meta = rec.get("meta") or {}
        if not meta.get("synthetic"):
            continue
        key = (_cut_key(meta), meta.get("chunk_index"))
        if key not in index:
            index[key] = (extract_target(rec).strip(), str(meta.get("context") or ""))
    return index


def _sibling_neighbours(meta: dict, index: dict[tuple, tuple[str, str]]) -> tuple[str, str] | None:
    """Flanking targets from the corpus, a section break treated as a wall.

    A neighbour under a different `meta.context` is in another section — the
    adjacency `fits_from_document` refuses to cross. Cross-heading pairs carry
    no context, so this degrades to plain index adjacency, which is correct
    there because the chunks really are contiguous.
    """
    if not meta.get("synthetic"):
        return None
    cut, i = _cut_key(meta), meta.get("chunk_index")
    if i is None:
        return None
    ctx = meta.get("context") or ""

    def side(j: int) -> str:
        found = index.get((cut, j))
        if found is None:
            return ""
        text, their_ctx = found
        return "" if their_ctx != ctx else text

    before, after = side(i - 1), side(i + 1)
    return (before, after) if (before or after) else None


class _Placer:
    """The three-source neighbour recovery, each document read once."""

    def __init__(
        self,
        siblings: dict[tuple, tuple[str, str]],
        *,
        captures: Path,
        blog_root: Path | None,
        cut: Cut,
    ) -> None:
        self.siblings = siblings
        self.captures = captures
        self.blog_root = blog_root
        self.cut = cut
        self._docs: dict[Path, str | None] = {}

    def _by_content(self, path: Path, target: str) -> tuple[str, str] | None:
        if path not in self._docs:
            self._docs[path] = path.read_text(encoding="utf-8") if path.is_file() else None
        doc = self._docs[path]
        return None if doc is None else locate_neighbours(doc, target, cut=self.cut)

    def place(self, meta: dict, target: str) -> tuple[str, str, str] | None:
        """`(before, after, origin)` for one record, best source first."""
        found = _sibling_neighbours(meta, self.siblings)
        if found is not None:
            return (*found, "sibling")
        archive_id = archive_id_of(meta)
        if archive_id:
            found = self._by_content(self.captures / str(archive_id) / "after.md", target)
            if found is not None:
                return (*found, "archive")
        if self.blog_root is not None and meta.get("source"):
            found = self._by_content(self.blog_root / str(meta["source"]), target)
            if found is not None:
                return (*found, "post")
        return None


def fits_from_rewrites(
    a_path: str | Path,
    b_path: str | Path,
    *,
    corpus_path: str | Path,
    cut: Cut,
    captures_dir: str | Path | None = None,
    blog_root: str | Path | None = None,
    field_a: str = "output",
    field_b: str = "output",
) -> tuple[list[tuple[FitCase, str, str]], StratumReport]:
    """Two `ai-style run` corpus-mode outputs -> `(case, span_a, span_b)` triples.

    `field_a`/`field_b` pick which text of each record is the span: `output`
    for an adapter A/B, and `target` against `slop` on one file for the
    **positive control**, which runs with every verdict (D97). Neighbours are
    always Dan's prose, never the other arm's, or the comparison is not
    controlled; they are recovered by `_Placer` and the source recorded.
    """
    unknown = [f for f in (field_a, field_b) if f not in FIELD_EXTRACTORS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; known: {sorted(FIELD_EXTRACTORS)}")
    take_a, take_b = FIELD_EXTRACTORS[field_a], FIELD_EXTRACTORS[field_b]

    corpus_path = Path(corpus_path)
    a_recs = {record_id(r): r for r in iter_jsonl(a_path)}
    b_recs = {record_id(r): r for r in iter_jsonl(b_path)}
    shared = a_recs.keys() & b_recs.keys()
    rep = StratumReport(
        n_a=len(a_recs),
        n_b=len(b_recs),
        n_joined=len(shared),
        only_in_a=len(a_recs) - len(shared),
        only_in_b=len(b_recs) - len(shared),
    )
    if not shared:
        return [], rep

    placer = _Placer(
        build_sibling_index(corpus_path),
        captures=Path(captures_dir) if captures_dir else corpus_path.parent / "captures",
        blog_root=Path(blog_root) if blog_root else None,
        cut=cut,
    )
    triples: list[tuple[FitCase, str, str]] = []
    # Sorted, so the stratum — and every resumed run and `--limit` prefix of
    # it — is the same set in the same order.
    for rid in sorted(shared):
        a_rec = a_recs[rid]
        span_a, span_b = take_a(a_rec).strip(), take_b(b_recs[rid]).strip()
        if not span_a or not span_b:
            rep.empty_output.append(rid)
            continue
        meta = a_rec.get("meta") or {}
        placed = placer.place(meta, extract_target(a_rec))
        if placed is None or not (placed[0].strip() or placed[1].strip()):
            rep.unplaceable.append(rid)
            continue
        before, after, origin = placed
        rep.by_neighbour_source[origin] = rep.by_neighbour_source.get(origin, 0) + 1
        case = FitCase(
            id=rid,
            source=str(meta.get("source") or ""),
            index=int(meta.get("chunk_index") or 0),
            before=before,
            span=span_a,
            after=after,
            context=str(meta.get("context") or ""),
            neighbour_source=origin,
        )
        triples.append((case, span_a, span_b))
    rep.n_built = len(triples)
    return triples, rep
