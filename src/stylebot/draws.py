"""The corpus as append-only draws plus derived views (D84).

Layer 1, **draws**: one record per generator call or human edit, keyed on what
it *was* — source, blog SHA, span, generator, strategy, prompt, effort,
replicate — with the raw generator output kept and an `outcome` of
`accepted | gate:<name> | error:<class>`. Draws live under `draws/` in
`pairs/`'s shard layout and are only ever appended to.

Layer 2, **views**: a function of the draws and a `ViewPolicy`, materialised
in exactly `pairs/`'s shard layout so every `corpus.iter_records` reader keeps
working. Data flows draws -> views, never back; a prune is a policy clause
plus a rebuild.

`draws_from_pairs` lifts a pair record into an accepted draw, and `build_view`
under an identity policy reproduces `pairs/` byte-for-byte (`findings.md` "The
view reproduces the corpus"); `new_draw` is the native draw `synth` appends per
generator call; a capture's draw is `capture.capture_pairs`'s. Nothing here
writes to `pairs/`; there is no lock, thread or process (P8).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import click

from stylebot import config, corpus, pairs
from stylebot.errors import ConfigError
from stylebot.generators import Generator
from stylebot.jsonl import append_jsonl, iter_jsonl
from stylebot.provenance import now_iso
from stylebot.targets import Target

#: The one outcome a view may render. Anything else is a kept, paid-for reject.
ACCEPTED = "accepted"
#: A built view's sidecar: which policy and which draws it was built from.
VIEW_SIDECAR = "VIEW.json"

_MONTH_RE = re.compile(r"^(\d{4}-\d{2})")


@dataclass
class Draw:
    """One generator call or human edit, as it happened (D84).

    `record` is the pair record a view emits for an accepted draw, kept whole
    so a lifted draw round-trips byte-for-byte. A draw is never edited, so a
    later save of the same session names what it replaced: `supersedes` holds
    the prior `capture_id`s (W4 step 4); `superseded_by` is the backfill's
    forward pointer on a draw that predates it. `build_view` honours both.
    """

    draw_id: str
    source: str | None = None
    blog_sha: str | None = None
    blog_dirty: bool | None = None
    span: tuple[int, int] | None = None
    chunk_index: int | None = None
    target_text: str | None = None
    context: str | None = None
    gen: dict | None = None
    raw_output: str | None = None
    outcome: str = ACCEPTED
    superseded_by: str | None = None
    supersedes: tuple[str, ...] | None = None
    drawn_at: str | None = None
    record: dict | None = None

    def to_json(self) -> dict:
        """The JSONL shape; `span` and `supersedes` become lists."""
        d = asdict(self)
        if self.span is not None:
            d["span"] = list(self.span)
        if self.supersedes is not None:
            d["supersedes"] = list(self.supersedes)
        return d

    @classmethod
    def from_json(cls, d: dict) -> Draw:
        """Inverse of `to_json`; unknown keys are ignored so a shard may grow."""
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        if kw.get("span") is not None:
            kw["span"] = tuple(kw["span"])
        if kw.get("supersedes") is not None:
            kw["supersedes"] = tuple(kw["supersedes"])
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
    *,
    chunk_index: int | None = None,
) -> str:
    """16 hex of the design cell: what was drawn, from where, by whom (D84).

    The place is the `span` when one is known; until the cutter records
    offsets, `chunk_index` stands in for it (W4 step 4), so two chunks of one
    source under one arm are two cells. A backfilled draw has its span.
    """
    where = list(span) if span else chunk_index
    cell = [source, blog_sha, where, generator, strategy, prompt_id, reasoning_effort, replicate]
    return hashlib.sha256(json.dumps(cell).encode("utf-8")).hexdigest()[:16]


# --- draws ------------------------------------------------------------------


def draw_shard_of(draw: Draw) -> str:
    """`corpus.shard_of` its record, so a backfill keeps corpus order (W4 step 3);
    a tombstone goes to `synth-<YYYY-MM>.jsonl` by `drawn_at`, else `-undated`."""
    if draw.record is not None:
        return corpus.shard_of(draw.record)
    match = _MONTH_RE.match(draw.drawn_at) if isinstance(draw.drawn_at, str) else None
    return f"synth-{match.group(1) if match else 'undated'}.jsonl"


def append(draws_dir: str | Path, draw: Draw) -> Path:
    """Append one draw to its shard, no lock: nothing writes concurrently (P8)."""
    shard = Path(draws_dir) / draw_shard_of(draw)
    append_jsonl(shard, draw.to_json())
    return shard


def iter_draws(draws_dir: str | Path) -> Iterator[Draw]:
    """Every draw in `draws_dir`, shard by shard in `corpus.shard_paths` order."""
    for shard in corpus.shard_paths(draws_dir):
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
    effort; `repo_prov` the blog SHA (`provenance.repo_provenance`); the
    target's `chunk_index` stands in for a span the cutter does not record.
    A `reasoning_text` in `gen` is dropped: traces are diagnostics, not corpus
    (`synth`'s sidecar keeps them).
    """
    prov = repo_prov or {}
    sha = prov.get("blog_sha")
    cell = (target.source, sha, None, arm.name, arm.strategy, arm.prompt_id, arm.reasoning_effort)
    return Draw(
        draw_id=draw_id(*cell, replicate, chunk_index=target.chunk_index),
        source=target.source,
        blog_sha=sha,
        blog_dirty=prov.get("blog_dirty"),
        chunk_index=target.chunk_index,
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


def capture_id_of(draw: Draw) -> str | None:
    """The `capture_id` of a real capture's draw; None for synthetic or no record."""
    meta = (draw.record or {}).get("meta") or {}
    return None if meta.get("synthetic") else meta.get("capture_id")


def content_sha256(draws_dir: str | Path) -> str | None:
    """16 hex over the draw shards, in `corpus.content_sha256`'s shape, so a
    manifest's `draws_sha256` reads like its `pairs_sha256` (W4 step 4)."""
    return corpus.content_sha256(draws_dir)


# --- views -------------------------------------------------------------------


@dataclass(frozen=True)
class ViewPolicy:
    """Every input whose change should name a different view (D84).

    Each field is optional so the blog's `view_policy()` fills what it has;
    all of them enter `policy_sha`. `system_prompt` is applied as a *stamp*
    (`_stamped`) and `keep_superseded`, `excluded_draw_ids` and
    `superseded_captures` (the capture ids this policy declares superseded)
    as clauses; the rest are named-only so far, because the gates already ran
    on today's draws and re-applying a threshold waits for draws that carry
    their raw output.
    """

    chunk_policy_digest: str | None = None
    gate_thresholds: dict | None = None
    automation_cap: int | None = None
    retired_arms: tuple[str, ...] | None = None
    anchor_rule: str | None = None
    role_rule: str | None = None
    system_prompt: str | None = None
    keep_superseded: bool = False
    excluded_draw_ids: tuple[str, ...] = ()
    superseded_captures: tuple[str, ...] | None = None

    @property
    def system_sha(self) -> str | None:
        """`pairs.system_sha` of `system_prompt`; None when the draws' own stands.

        Derived, never supplied: a stored sha could name a prompt other than the
        one the view emits, and only one of the two could be true (D64).
        """
        return pairs.system_sha(self.system_prompt) if self.system_prompt else None


def policy_sha(policy: ViewPolicy) -> str:
    """16 hex over the policy's fields, order-free in every collection."""
    d = asdict(policy)
    for key in ("retired_arms", "excluded_draw_ids", "superseded_captures"):
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


def _clauses(policy: ViewPolicy, draws: Sequence[Draw]) -> list[Clause]:
    """The exclusion clauses a policy applies, first match wins.

    A draw is superseded when it says so (`superseded_by`, the backfill's),
    when any draw names its `capture_id` in `supersedes` (a later save's), or
    when the policy names that capture in `superseded_captures` (D84).
    """
    excluded = frozenset(policy.excluded_draw_ids)
    named = (cid for cid in (policy.superseded_captures or ()) if cid)
    replaced = frozenset(cid for d in draws for cid in (d.supersedes or ())).union(named)
    out: list[Clause] = [("outcome", lambda d: d.outcome != ACCEPTED)]
    if not policy.keep_superseded:
        out.append(
            ("superseded", lambda d: d.superseded_by is not None or capture_id_of(d) in replaced)
        )
    if excluded:
        out.append(("excluded_draw_ids", lambda d: d.draw_id in excluded))
    out.append(("no_record", lambda d: d.record is None))
    return out


def _stamped(record: dict, policy: ViewPolicy) -> dict:
    """The record a view emits: `messages[0]` is the policy's system turn (D64).

    The draw keeps whatever turn it was drawn under — history nothing reads —
    so moving a corpus to another prompt is this policy field plus a rebuild,
    never an edit to a draw (D84). No `system_prompt`, or one the draw already
    carries, returns the record itself, so a view moves no bytes it need not.
    """
    msgs = record.get("messages")
    if not policy.system_prompt or not isinstance(msgs, list) or not msgs:
        return record
    if msgs[0].get("content") == policy.system_prompt:
        return record
    return {**record, "messages": [{**msgs[0], "content": policy.system_prompt}, *msgs[1:]]}


def view_records(draws: Iterable[Draw], policy: ViewPolicy) -> Iterator[dict]:
    """The records the view of `draws` under `policy` holds, in draw order;
    what `ai-style log` reads back as the live corpus (W4 step 4)."""
    draws = list(draws)
    clauses = _clauses(policy, draws)
    for draw in draws:
        if not any(hit(draw) for _name, hit in clauses):
            assert draw.record is not None
            yield _stamped(draw.record, policy)


def _bump(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def build_view(
    draws: Iterable[Draw], policy: ViewPolicy, *, out: str | Path | None = None
) -> ViewReport:
    """Materialise the view of `draws` under `policy` as a `pairs/`-shaped dir.

    Records route by `corpus.shard_of` and keep their read order within a
    shard, so an identity policy over `draws_from_pairs(p)` reproduces `p`'s
    `content_sha256`. `out` must not already hold `*.jsonl`: a view is
    rebuilt into a fresh directory, never edited (D84). Without `out` nothing
    is written and the report says what would be: the counts, and the
    `content_sha256` the view would have (`status` prints it, W4 step 4).
    """
    if out is not None:
        out = Path(out)
        if out.is_dir() and any(out.glob("*.jsonl")):
            raise ConfigError(f"{out} already holds a view; build into a fresh directory")
        out.mkdir(parents=True, exist_ok=True)
    report = ViewReport(policy_sha=policy_sha(policy))
    draws = list(draws)  # `supersedes` points backwards, so the clauses need every draw
    clauses = _clauses(policy, draws)
    buckets: dict[str, list[str]] = {}
    for draw in draws:
        _bump(report.by_outcome, draw.outcome)
        clause = next((name for name, hit in clauses if hit(draw)), None)
        if clause is not None:
            _bump(report.excluded, clause)
            continue
        assert draw.record is not None
        record = _stamped(draw.record, policy)
        buckets.setdefault(corpus.shard_of(record), []).append(corpus.line_of(record))
        report.records += 1
    digests: list[tuple[str, str]] = []
    for name, lines in sorted(buckets.items()):
        blob = "".join(lines).encode("utf-8")
        digests.append((name, hashlib.sha256(blob).hexdigest()[:16]))
        if out is not None:
            (out / name).write_bytes(blob)
            report.shards.append(out / name)
    report.content_sha256 = corpus.shards_sha256(digests)
    return report


def read_sidecar(view_dir: str | Path) -> dict | None:
    """The `VIEW.json` a built view carries, or None for no view or a broken one."""
    try:
        doc = json.loads((Path(view_dir) / VIEW_SIDECAR).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def ensure_view(draws_dir: str | Path, policy: ViewPolicy, *, views_root: str | Path) -> Path:
    """The view of `draws_dir` under `policy`, at `views_root/pairs-<policy_sha>`.

    Built when absent; rebuilt when its sidecar names another policy or other
    draws (`content_sha256`), since a view is derived and never edited (D84).
    A rebuild removes the directory first, so `build_view` gets the fresh one
    it requires. Returns the path a `pairs/` reader can be pointed at.
    """
    draws_dir = Path(draws_dir)
    if not draws_dir.is_dir():
        raise ConfigError(f"no draws under {draws_dir}")
    sha = policy_sha(policy)
    draws_sha = content_sha256(draws_dir)
    out = Path(views_root) / config.view_name(sha)
    side = read_sidecar(out)
    if side and side.get("policy_sha") == sha and side.get("draws_sha256") == draws_sha:
        return out
    if out.exists():
        shutil.rmtree(out)
    report = build_view(iter_draws(draws_dir), policy, out=out)
    side = {
        "policy_sha": sha,
        "draws_sha256": draws_sha,
        "records": report.records,
        "content_sha256": report.content_sha256,
        "built_at": now_iso(),
    }
    (out / VIEW_SIDECAR).write_text(json.dumps(side, indent=2) + "\n", encoding="utf-8")
    return out


def training_view(draws_dir: str | Path, policy: ViewPolicy | None) -> tuple[Path, dict]:
    """The view a training run reads, built beside its draws, and the manifest
    keys that name it: `policy_sha`, `draws_sha256` and the `style_system_sha`
    the view was stamped with sit next to `pairs_sha256`, which is then the
    view's own content hash (W4 steps 4 and 5)."""
    draws_dir = Path(draws_dir)
    policy = policy or ViewPolicy()
    view = ensure_view(draws_dir, policy, views_root=config.views_root(draws_dir.parent))
    return view, {
        "policy_sha": policy_sha(policy),
        "draws_sha256": content_sha256(draws_dir),
        "style_system_sha": policy.system_sha or pairs.system_sha(),
    }


def corpus_or_view(
    pairs_path: Path | None, draws_dir: Path | None, policy_path: Path | None
) -> Path | None:
    """The corpus a `--pairs` / `--draws` pair names, for a command taking both.

    `--draws` builds (or reuses) its view and returns that, so a reader needs
    no `pairs/`; naming both is a refusal, since only one of them can be the
    corpus a report is about (D84).
    """
    if pairs_path is not None and draws_dir is not None:
        raise ConfigError("pass --pairs or --draws, not both")
    if draws_dir is None:
        return pairs_path
    return training_view(draws_dir, load_policy(policy_path))[0]


def load_policy(path: str | Path | None) -> ViewPolicy:
    """A `ViewPolicy` from a JSON object of its fields (lists become tuples);
    `None` is the identity policy. The blog builds its policy from constants
    instead (`training_targets.view_policy`); this is `ai-style`'s way in."""
    if path is None:
        return ViewPolicy()
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = (
        sorted(set(doc) - {f.name for f in fields(ViewPolicy)}) if isinstance(doc, dict) else []
    )
    if not isinstance(doc, dict) or unknown:
        raise ConfigError(f"{path}: not a ViewPolicy object; unknown field(s) {unknown}")
    for key in ("retired_arms", "excluded_draw_ids", "superseded_captures"):
        if doc.get(key) is not None:
            doc[key] = tuple(doc[key])
    return ViewPolicy(**doc)


def view_options():
    """`--draws DIR` and `--policy FILE` for a command that reads a view (W4 step 4)."""

    def deco(f):
        f = click.option(
            "--policy",
            "policy_path",
            type=click.Path(exists=True, dir_okay=False, path_type=Path),
            default=None,
            help="A ViewPolicy as a JSON object of its fields (`draws.load_policy`). "
            "[default: the identity policy]",
        )(f)
        return click.option(
            "--draws",
            "draws_dir",
            type=click.Path(file_okay=False, path_type=Path),
            default=None,
            help="A draws/ directory: build (or reuse) its view under the sibling views/ "
            "and read that instead of pairs/ (D84).",
        )(f)

    return deco
