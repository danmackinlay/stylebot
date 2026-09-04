"""The corpus as append-only draws plus derived views (D84).

Layer 1, **draws**: one record per generator call or human edit, keyed on what
it *was* — source, blog SHA, span, generator, strategy, prompt, effort,
replicate — with the raw generator output kept and an `outcome` of
`accepted | gate:<name> | error:<class>`. Draws live in
`draws/synth-<YYYY-MM>.jsonl` shards and are only ever appended to.

Layer 2, **views**: a function of the draws and a `ViewPolicy`, materialised
in exactly `pairs/`'s shard layout so every `corpus.iter_records` reader keeps
working. Data flows draws -> views, never back; a prune is a policy clause
plus a rebuild.

Step 1 (W4) is read-side: `draws_from_pairs` lifts today's pair records into
accepted draws, and `build_view` under an identity policy reproduces `pairs/`
byte-for-byte (`findings.md` "The view reproduces the corpus"). Step 2 adds
`new_draw`, the native draw `synth` appends per generator call. Nothing here
writes to `pairs/`; there is no lock, thread or process (P8).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from stylebot import corpus, pairs
from stylebot.errors import ConfigError
from stylebot.generators import Generator
from stylebot.jsonl import append_jsonl, iter_jsonl
from stylebot.provenance import now_iso
from stylebot.targets import Target

#: The one outcome a view may render. Anything else is a kept, paid-for reject.
ACCEPTED = "accepted"

_MONTH_RE = re.compile(r"^(\d{4}-\d{2})")
_DRAW_SHARD_GLOB = "synth-*.jsonl"


@dataclass
class Draw:
    """One generator call or human edit, as it happened (D84).

    `record` is the pair record a view emits for an accepted draw, kept whole
    so a lifted draw round-trips byte-for-byte; step 2's native draws render
    it from the other fields. `superseded_by` names the later save of the
    same session, when there is one.
    """

    draw_id: str
    source: str | None = None
    blog_sha: str | None = None
    blog_dirty: bool | None = None
    span: tuple[int, int] | None = None
    target_text: str | None = None
    context: str | None = None
    gen: dict | None = None
    raw_output: str | None = None
    outcome: str = ACCEPTED
    superseded_by: str | None = None
    drawn_at: str | None = None
    record: dict | None = None

    def to_json(self) -> dict:
        """The JSONL shape; `span` becomes a two-element list."""
        d = asdict(self)
        if self.span is not None:
            d["span"] = list(self.span)
        return d

    @classmethod
    def from_json(cls, d: dict) -> Draw:
        """Inverse of `to_json`; unknown keys are ignored so a shard may grow."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("span") is not None:
            kw["span"] = tuple(kw["span"])
        return cls(**kw)


def draw_id(
    source: str | None,
    blog_sha: str | None,
    span: tuple[int, int] | None,
    generator: str | None,
    strategy: str | None,
    prompt_id: str | None,
    reasoning_effort: str | None,
    replicate: int = 0,
) -> str:
    """16 hex of the design cell: what was drawn, from where, by whom (D84)."""
    cell = [source, blog_sha, list(span) if span else None, generator, strategy]
    cell += [prompt_id, reasoning_effort, replicate]
    return hashlib.sha256(json.dumps(cell).encode("utf-8")).hexdigest()[:16]


# --- draws ------------------------------------------------------------------


def draw_shard_of(draw: Draw) -> str:
    """`synth-<YYYY-MM>.jsonl` by `drawn_at`, else `synth-undated.jsonl`."""
    match = _MONTH_RE.match(draw.drawn_at) if isinstance(draw.drawn_at, str) else None
    return f"synth-{match.group(1) if match else 'undated'}.jsonl"


def append(draws_dir: str | Path, draw: Draw) -> Path:
    """Append one draw to its month shard; returns the shard written.

    A plain append: step 1 has no concurrent writer, so there is no lock (P8).
    """
    shard = Path(draws_dir) / draw_shard_of(draw)
    append_jsonl(shard, draw.to_json())
    return shard


def iter_draws(draws_dir: str | Path) -> Iterator[Draw]:
    """Every draw in `draws_dir`, shard by shard in name order."""
    for shard in sorted(Path(draws_dir).glob(_DRAW_SHARD_GLOB)):
        for rec in iter_jsonl(shard):
            yield Draw.from_json(rec)


def draw_from_pair(record: dict) -> Draw:
    """A pair record lifted into an accepted draw (W4 step 1).

    No span and no SHA yet — step 3 backfills them — so the `draw_id` is
    `pairs.record_id`, the key scores already join on. A synthetic pair's
    user turn is the generator's post-gate output, the only copy kept.
    """
    meta = record.get("meta") or {}
    synthetic = bool(meta.get("synthetic"))
    return Draw(
        draw_id=pairs.record_id(record),
        source=meta.get("source"),
        target_text=pairs.extract_target(record),
        context=meta.get("context"),
        gen=meta.get("gen") if synthetic else None,
        raw_output=pairs.extract_slop(record) if synthetic else None,
        outcome=ACCEPTED,
        drawn_at=meta.get("captured_at"),
        record=record,
    )


def new_draw(
    arm: Generator,
    repo_prov: Mapping | None,
    target: Target,
    context: str | None,
    *,
    gen: Mapping | None,
    raw_output: str | None,
    outcome: str,
    record: dict | None = None,
    replicate: int | str = 0,
) -> Draw:
    """A native draw for one generator call, as `synth` makes it (W4 step 2).

    `draw_id` is the cell — `arm` supplies generator, strategy, prompt and
    effort; `repo_prov` the blog SHA (`provenance.repo_provenance`) — with
    `span=None` until step 3 records offsets. A `reasoning_text` in `gen` is
    dropped: traces are diagnostics, not corpus (`synth`'s sidecar keeps them).
    """
    prov = repo_prov or {}
    sha = prov.get("blog_sha")
    cell = (target.source, sha, None, arm.name, arm.strategy, arm.prompt_id, arm.reasoning_effort)
    return Draw(
        draw_id=draw_id(*cell, replicate),
        source=target.source,
        blog_sha=sha,
        blog_dirty=prov.get("blog_dirty"),
        target_text=target.text,
        context=context or None,
        gen={k: v for k, v in gen.items() if k != "reasoning_text"} if gen else None,
        raw_output=raw_output,
        outcome=outcome,
        drawn_at=now_iso(),
        record=record,
    )


def draws_from_pairs(pairs_path: str | Path) -> Iterator[Draw]:
    """Today's corpus read as draws, in `corpus.iter_records` order."""
    for record in corpus.iter_records(pairs_path):
        yield draw_from_pair(record)


# --- views -------------------------------------------------------------------


@dataclass(frozen=True)
class ViewPolicy:
    """Every input whose change should name a different view (D84).

    Each field is optional so the blog's `view_policy()` fills what it has;
    all of them enter `policy_sha`. Only `keep_superseded` and
    `excluded_draw_ids` are applied as clauses in step 1: the gates already
    ran on today's draws and their raw output is not kept, so re-applying a
    threshold waits for draws that carry one.
    """

    chunk_policy_digest: str | None = None
    gate_thresholds: dict | None = None
    automation_cap: int | None = None
    retired_arms: tuple[str, ...] | None = None
    anchor_rule: str | None = None
    role_rule: str | None = None
    system_sha: str | None = None
    keep_superseded: bool = False
    excluded_draw_ids: tuple[str, ...] = ()


def policy_sha(policy: ViewPolicy) -> str:
    """16 hex over the policy's fields, order-free in every collection."""
    d = asdict(policy)
    for key in ("retired_arms", "excluded_draw_ids"):
        if d[key] is not None:
            d[key] = sorted(d[key])
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class ViewReport:
    """What `build_view` read and wrote: counts by outcome and by clause."""

    policy_sha: str
    records: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    shards: list[Path] = field(default_factory=list)
    content_sha256: str | None = None


Clause = tuple[str, Callable[[Draw], bool]]


def _clauses(policy: ViewPolicy) -> list[Clause]:
    """The exclusion clauses a policy applies, first match wins."""
    excluded = frozenset(policy.excluded_draw_ids)
    out: list[Clause] = [("outcome", lambda d: d.outcome != ACCEPTED)]
    if not policy.keep_superseded:
        out.append(("superseded", lambda d: d.superseded_by is not None))
    if excluded:
        out.append(("excluded_draw_ids", lambda d: d.draw_id in excluded))
    out.append(("no_record", lambda d: d.record is None))
    return out


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def build_view(draws: Iterable[Draw], policy: ViewPolicy, *, out: str | Path) -> ViewReport:
    """Materialise the view of `draws` under `policy` as a `pairs/`-shaped dir.

    Records route by `corpus.shard_of` and keep their read order within a
    shard, so an identity policy over `draws_from_pairs(p)` reproduces `p`'s
    `content_sha256`. `out` must not already hold `*.jsonl`: a view is
    rebuilt into a fresh directory, never edited (D84).
    """
    out = Path(out)
    if out.is_dir() and any(out.glob("*.jsonl")):
        raise ConfigError(f"{out} already holds a view; build into a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    report = ViewReport(policy_sha=policy_sha(policy))
    clauses = _clauses(policy)
    buckets: dict[str, list[str]] = {}
    for draw in draws:
        _bump(report.by_outcome, draw.outcome)
        clause = next((name for name, hit in clauses if hit(draw)), None)
        if clause is not None:
            _bump(report.excluded, clause)
            continue
        assert draw.record is not None
        buckets.setdefault(corpus.shard_of(draw.record), []).append(corpus.line_of(draw.record))
        report.records += 1
    for name, lines in sorted(buckets.items()):
        shard = out / name
        shard.write_text("".join(lines), encoding="utf-8")
        report.shards.append(shard)
    report.content_sha256 = corpus.content_sha256(out)
    return report
