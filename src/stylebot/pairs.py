"""Validation for the shared `pairs.jsonl` contract.

The corpus schema is the seam between Phase 1 (real edit pairs), Phase 2
(synthetic pairs), and Phase 3 (training). Phase 2 should validate its output
here as a done-criteria gate, and Phase 3 should validate its input before
spending money on a training run.

The schema is enforced by `validate_pair_record` below and produced by
`stylebot.bin.ai_style_log`. Keep this in sync with that producer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from stylebot import corpus
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.provenance import PROVENANCE_KEYS

#: The `meta.schema_version` stamped on every record written from now on.
#:
#: 1 is the shape as it stood on 2026-09-03: the message triple, the required
#: meta keys below, and `meta.provenance`. A record with NO `schema_version` is
#: exactly the pre-2026-09 corpus (D36). Bump only for a change a consumer must
#: branch on: adding an optional meta key is not one, changing what an existing
#: key MEANS is.
PAIR_SCHEMA_VERSION = 1

# meta keys every well-formed pair record carries; tags/synthetic/generator are
# optional provenance. Optional chunk-scale facets on synthetic pairs:
# `merge_max_chars` is the pack budget and `chunk_mode` is how the chunk was cut
# — None/"section" (heading carried as `context`) or "cross-heading" (headings
# packed in as CONTENT, so no `context`, so the verbatim-prefix invariant below
# does not apply).
REQUIRED_META_KEYS = (
    "source",
    "captured_at",
    "capture_id",
    "chunk_index",
    "chunk_total",
)

# --- meta.gen schema (synthetic pairs only) ----------------------------------
# The generation covariates recorded under `meta.gen`, consumed by
# `stylebot.scoring` (flattened onto score records) and `stylebot.report` (row
# columns). Split by WHICH WRITER produces each group, which is what makes the
# declaration checkable: the synth tests assert each group against a real record
# from its writer in BOTH directions, so a key the writer stopped emitting
# cannot sit here as a wish. Callers extend via a factory's `extra_meta`; those
# keys belong in `GEN_EXTRA_KEYS`.

#: Written unconditionally by `generators.openai_generator` for every draw.
GEN_BASE_KEYS = (
    "model",
    "reasoning_effort",
    "temperature",
    "top_p",
    "max_tokens",
    "finish_reason",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cost",
    "gen_seconds",
    "provider",
    "prompt_id",
    "prompt_version",
    "prompt_label",
)
#: Added through a factory's `extra_meta` — `openrouter_generator` sets
#: `provider_sort` (the routing *request*, recorded next to the `provider` it
#: actually produced).
GEN_EXTRA_KEYS = ("provider_sort",)
#: Added by `synth.synthesize_pairs`: the deliberate-resample label, folded into
#: `synth_key`. Absent means the base corpus.
GEN_REPLICATE_KEYS = ("replicate",)
#: Added by the session loop only (`session_turns > 1`).
GEN_SESSION_KEYS = ("session_id", "session_turn", "context_window", "window_fill")

#: Everything that can appear under `meta.gen`, whatever wrote it.
GEN_META_KEYS = GEN_BASE_KEYS + GEN_EXTRA_KEYS + GEN_REPLICATE_KEYS
# The experiment-relevant subset flattened onto each score record. A
# scores.jsonl outlives the pairs.jsonl it was scored from, so this keeps the
# covariates that *explain* an outcome, not just the knobs that were set:
# dropping one costs an answer no re-aggregation can recover.
GEN_FACET_KEYS = (
    "model",
    "reasoning_effort",
    "temperature",
    "top_p",
    "prompt_id",
    "prompt_version",
    "prompt_label",
    "replicate",
    "session_id",
    "session_turn",
    "context_window",
    "window_fill",
    "prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "cost",
    "gen_seconds",
    "provider",
    "finish_reason",
)


@lru_cache(maxsize=1)
def system_sha() -> str:
    """16-hex digest of the frozen `STYLE_SYSTEM`, stamped as `meta.system_sha`.

    `validate_pair_record` already refuses a record whose system message is not
    the frozen prompt, so this is not a second check of the same thing: it is
    what a *derived* artifact — a scores row, a run manifest, a training shard
    that dropped `messages[0]` — can carry to say which prompt the pairs behind
    it were built against. Stamping it on the record is what makes that
    transitive claim checkable.
    """
    return hashlib.sha256(STYLE_SYSTEM.encode("utf-8")).hexdigest()[:16]


def build_pair_content(context: str, body: str) -> str:
    """Assemble a pair message body: heading context (verbatim) + the prose body.

    The **shared contract** between Phase 1 (`ai-style log`) and Phase 2
    (`stylebot.synth`): `context` — a section heading the body sits under — is
    prepended *identically* to both the user (slop) and assistant (target) sides
    so the styler learns to preserve the heading and restyle only the body
    beneath it. Empty context returns the body unchanged.

    Both producers MUST build message content through this one function, and
    record the same `context` under `meta.context`, so real and synthetic pairs
    stay shape-compatible. See `_plans/decisions.md` D14.
    """
    context = (context or "").strip()
    body = body or ""
    return f"{context}\n\n{body}" if context else body


def pair_body(content: str, context: str | None) -> str:
    """The prose body of a pair message, with the heading-context prefix removed.

    The inverse of `build_pair_content`, so a consumer sees the transform region
    (the body), not the heading shared by both sides. No context (or content
    lacking the prefix) returns unchanged.
    """
    if not context:
        return content
    prefix = f"{context}\n\n"
    return content[len(prefix) :] if content.startswith(prefix) else content


def extract_slop(record: dict) -> str:
    """The slop-side body of a pair (messages[1], heading-context stripped).

    Public API: the voice-classifier trainer (livingthing) assembles its training
    texts through these extractors so eval and training read a pair identically.
    """
    return pair_body(record["messages"][1]["content"], (record.get("meta") or {}).get("context"))


def extract_target(record: dict) -> str:
    """The Dan-side body of a pair (messages[2], heading-context stripped)."""
    return pair_body(record["messages"][2]["content"], (record.get("meta") or {}).get("context"))


def extract_output(record: dict) -> str:
    """The styler-output body (top-level `"output"` of a Phase-4 rewrite
    JSONL — `stylebot.infer.rewrite_pairs_file` — heading-context stripped)."""
    return pair_body(record.get("output") or "", (record.get("meta") or {}).get("context"))


# Field name -> (record -> body text). Defaults cover the pairs schema
# (messages[1]=slop/user, messages[2]=Dan/assistant); "output" is the Phase-4
# styler rewrite, present only on infer.rewrite_pairs_file output JSONLs.
FIELD_EXTRACTORS: dict[str, Callable[[dict], str]] = {
    "slop": extract_slop,
    "target": extract_target,
    "output": extract_output,
}


def _recut_digest(recut: object) -> str:
    """A short, stable digest of one re-cut's gap policy (`meta.recut`).

    Canonical JSON (sorted keys, no whitespace) so the digest keys off the
    policy's VALUES, not off dict insertion order — and so any key a future
    `recut_archives` adds to the stamp participates automatically. Eight hex
    chars discriminates a handful of hand-chosen cut policies with room to spare.
    """
    try:
        canon = json.dumps(recut, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:  # a stamp JSON can't encode — degrade to repr, never raise
        canon = repr(recut)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def record_id(record: dict) -> str:
    """A stable join key linking a pair's score back to the corpus.

    Prefers `meta.synth_key` (synthetic pairs); else `capture_id:chunk_index`
    (real captures group by `capture_id`); else a hash of `source` + chunk index.

    **A re-cut record folds its cut policy (`meta.recut`) into the key**, since
    `capture.recut_archives` reuses the `capture_id` and renumbers from 0 — so
    `<cap>:0` can name a merged pair where it named one paragraph, and a
    resuming consumer would skip changed prose as done. The same policy re-run
    gives a byte-identical id. Records with no `meta.recut` keep their key
    byte-for-byte, so a committed `scores.jsonl` keeps joining (D36).
    """
    meta = record.get("meta") or {}
    if meta.get("synth_key"):
        return str(meta["synth_key"])
    recut = meta.get("recut")
    # Empty for a non-re-cut record, which preserves the historical id shape.
    policy = f":r{_recut_digest(recut)}" if recut else ""
    capture_id = meta.get("capture_id")
    chunk_index = meta.get("chunk_index")
    if capture_id is not None:
        return f"{capture_id}{policy}:{chunk_index}"
    # No capture_id (hand-built records): the policy joins the hashed material
    # rather than the visible key, so an unstamped record hashes unchanged.
    h = hashlib.sha256(f"{meta.get('source')}\x00{chunk_index}{policy}".encode())
    return h.hexdigest()[:16]


# --- meta types and domains --------------------------------------------------
# Key -> (accepted types, optional predicate, predicate description). Checked
# only when the key is PRESENT; `REQUIRED_META_KEYS` is the separate question of
# which must be there at all. Anything absent from this table is unvalidated by
# design — the schema is open, and a closed whitelist would make every
# experiment's covariate a validation failure. NoneType is accepted where a
# producer legitimately writes null.
_NoneType = type(None)
_META_TYPES: dict[str, tuple] = {
    "source": (str, _NoneType),
    "captured_at": (str,),
    "capture_id": (str,),
    "chunk_index": (int,),
    "chunk_total": (int,),
    "before_chars": (int,),
    "after_chars": (int,),
    "transform_sim": (int, float),
    "context": (str,),
    "context_mode": (str,),
    "snapped_at": (str, _NoneType),
    "merge_max_chars": (int, _NoneType),
    "chunk_mode": (str, _NoneType),
    "synthetic": (bool,),
    "generator": (str,),
    "slop_strategy": (str,),
    "synth_key": (str,),
    "tags": (list,),
    "gen": (dict,),
    "recut": (dict,),
    "before_frontmatter": (dict,),
    "after_frontmatter": (dict,),
    "schema_version": (int,),
    "system_sha": (str,),
    "provenance": (dict,),
}
_META_DOMAINS: dict[str, tuple[Callable[[object], bool], str]] = {
    "chunk_index": (lambda v: v >= 0, "must be >= 0"),
    "chunk_total": (lambda v: v >= 1, "must be >= 1"),
    "before_chars": (lambda v: v >= 0, "must be >= 0"),
    "after_chars": (lambda v: v >= 0, "must be >= 0"),
    "transform_sim": (lambda v: 0.0 <= v <= 1.0, "must be in [0, 1]"),
    "context_mode": (lambda v: v == "immediate", "must be 'immediate'"),
    "chunk_mode": (lambda v: v in (None, "cross-heading"), "must be null or 'cross-heading'"),
    "schema_version": (lambda v: v >= 1, "must be >= 1"),
    "merge_max_chars": (lambda v: v is None or v > 0, "must be null or > 0"),
}
_PROVENANCE_TYPES: dict[str, tuple] = {
    "blog_sha": (str, _NoneType),
    "blog_dirty": (bool, _NoneType),
    "stylebot_sha": (str, _NoneType),
    "policy_digest": (str, _NoneType),
    "source_automation": (int, _NoneType),
}


def _type_name(types: tuple) -> str:
    return " | ".join("null" if t is _NoneType else t.__name__ for t in types)


def _check_meta_types(meta: dict) -> list[str]:
    errors: list[str] = []
    for key, types in _META_TYPES.items():
        if key not in meta:
            continue
        value = meta[key]
        # bool is an int subclass; an int-typed key must not silently take True.
        if bool in types:
            ok = isinstance(value, types)
        else:
            ok = isinstance(value, types) and not (int in types and isinstance(value, bool))
        if not ok:
            errors.append(f"meta.{key} must be {_type_name(types)}, got {type(value).__name__}")
            continue
        domain = _META_DOMAINS.get(key)
        if domain is not None and not domain[0](value):
            errors.append(f"meta.{key} {domain[1]}, got {value!r}")
    if isinstance(meta.get("tags"), list) and not all(isinstance(t, str) for t in meta["tags"]):
        errors.append("meta.tags must be a list of strings")
    idx, total = meta.get("chunk_index"), meta.get("chunk_total")
    if (
        isinstance(idx, int)
        and not isinstance(idx, bool)
        and isinstance(total, int)
        and idx >= total
    ):
        errors.append(f"meta.chunk_index {idx} must be < meta.chunk_total {total}")
    if isinstance(meta.get("system_sha"), str) and meta["system_sha"] != system_sha():
        errors.append(
            "meta.system_sha does not match the frozen STYLE_SYSTEM "
            f"(record {meta['system_sha']}, current {system_sha()})"
        )
    prov = meta.get("provenance")
    if isinstance(prov, dict):
        missing = [k for k in PROVENANCE_KEYS if k not in prov]
        if missing:
            errors.append(f"meta.provenance missing key(s) {missing}")
        for key, types in _PROVENANCE_TYPES.items():
            if key not in prov:
                continue
            value = prov[key]
            ok = isinstance(value, types) and not (
                bool not in types and isinstance(value, bool) and int in types
            )
            if not ok:
                errors.append(
                    f"meta.provenance.{key} must be {_type_name(types)}, "
                    f"got {type(value).__name__}"
                )
    return errors


def validate_pair_record(rec: object) -> list[str]:
    """Return a list of human-readable problems with one pair record.

    Empty list == valid. Checks the message triple (system/user/assistant),
    that the system prompt is the frozen `STYLE_SYSTEM`, that user/assistant
    content is non-empty, that the required `meta` keys are present, and that
    every meta key it knows about carries the right type and a value in range.

    **Per-record by construction, which is the limit worth naming**: nothing
    here can see that two records share a `record_id`, and for two years nothing
    did — 21 duplicate cells reached the live corpus. `validate_pairs_file` is
    where that check lives.
    """
    errors: list[str] = []

    if not isinstance(rec, dict):
        return ["record is not a JSON object"]

    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        errors.append("messages must be a list of exactly 3 items")
    else:
        roles = [m.get("role") if isinstance(m, dict) else None for m in msgs]
        if roles != ["system", "user", "assistant"]:
            errors.append(f"message roles must be [system, user, assistant], got {roles}")
        else:
            if msgs[0].get("content") != STYLE_SYSTEM:
                errors.append(
                    "system message content is not the frozen STYLE_SYSTEM "
                    "(changing it invalidates the corpus)"
                )
            for role_idx, label in ((1, "user"), (2, "assistant")):
                content = msgs[role_idx].get("content")
                if not isinstance(content, str) or not content.strip():
                    errors.append(f"{label} message content is empty or non-string")

    meta = rec.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be a JSON object")
    else:
        for key in REQUIRED_META_KEYS:
            if key not in meta:
                errors.append(f"meta missing required key {key!r}")
        errors.extend(_check_meta_types(meta))

        # Heading-context invariant: a declared meta.context must be the
        # verbatim prefix of BOTH the user and assistant content — added as
        # fixed context via build_pair_content, not paraphrased into the slop.
        # Only enforced when context is present and the triple parsed cleanly.
        context = meta.get("context")
        if context and isinstance(msgs, list) and len(msgs) == 3:
            for role_idx, label in ((1, "user"), (2, "assistant")):
                msg = msgs[role_idx]
                content = msg.get("content") if isinstance(msg, dict) else None
                if not (isinstance(content, str) and content.startswith(context)):
                    errors.append(
                        f"{label} content does not start with meta.context "
                        "(heading must be a verbatim prefix)"
                    )

    return errors


def validate_pairs_file(
    path: str | Path, *, check_duplicates: bool = True
) -> list[tuple[int | str, list[str]]]:
    """Validate every line of a corpus, and the corpus-level uniqueness rule.

    Returns `(location, errors)` per invalid line; empty means valid.
    Unparseable JSON is one error for that line. `location` is a 1-based line
    number for a single-file corpus and `"<shard>.jsonl:<line>"` for a sharded
    one, which is a line somebody can open.

    **`record_id` uniqueness is a property of the FILE, not of a record** (D36),
    so it is checked here and nowhere else, across the whole corpus rather than
    per shard (D50). A repeat is reported against the *later* line, the earlier
    being what resuming consumers joined to. `check_duplicates=False` for a
    partial file whose id space is shared with lines you are not looking at.
    """
    shards = corpus.shard_paths(path)
    sharded = len(shards) > 1 or corpus.is_sharded(path)
    problems: list[tuple[int | str, list[str]]] = []
    first_seen: dict[str, int | str] = {}
    for shard in shards:
        if not shard.exists():
            continue
        with shard.open(encoding="utf-8") as fp:
            for lineno, line in enumerate(fp, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                where: int | str = f"{shard.name}:{lineno}" if sharded else lineno
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    problems.append((where, [f"invalid JSON: {exc}"]))
                    continue
                errs = validate_pair_record(rec)
                if check_duplicates and isinstance(rec, dict) and isinstance(rec.get("meta"), dict):
                    rid = record_id(rec)
                    if rid in first_seen:
                        errs.append(
                            f"duplicate record_id {rid!r} (first seen line {first_seen[rid]})"
                        )
                    else:
                        first_seen[rid] = where
                if errs:
                    problems.append((where, errs))
    return problems


@dataclass
class DedupeReport:
    """What `dedupe_pairs_file` found (and, with `apply`, did)."""

    total: int = 0
    duplicate_ids: int = 0  # distinct record_ids appearing more than once
    dropped: int = 0  # lines that would be / were removed
    kept: int = 0
    applied: bool = False
    backup: Path | None = None
    #: (record_id, position dropped, position kept), oldest first. Positions are
    #: 1-based RECORD indices with blank lines skipped, which is the file's line
    #: number for any corpus a producer wrote (nothing here emits blank lines).
    drops: list[tuple[str, int, int]] = field(default_factory=list)


def dedupe_pairs_file(
    path: str | Path, *, apply: bool = False, backup: bool = True
) -> DedupeReport:
    """Drop every repeat of a `record_id`, keeping the FIRST occurrence.

    **First, not best, and not newest**: the first is what every committed
    `scores.jsonl` row already joined to, so keeping it leaves those artifacts
    valid. The duplicates this exists for are not byte-identical rows — they are
    two *draws* for the same cell, written by two synth processes that each read
    `existing_synth_keys` before the other appended (D36).

    Dry-run by default. `apply` rewrites atomically, keeping a `.bak` beside the
    corpus unless `backup=False` — `<corpus>.bak` for a single file, a
    `<corpus>.bak/` directory of shard copies for a sharded one. Unparseable
    lines are round-tripped verbatim rather than dropped
    (`iter_jsonl(keep_undecodable=True)`).
    """
    path = Path(path)
    records = corpus.read_records(path, keep_undecodable=True)
    report = DedupeReport(total=len(records))
    first_line: dict[str, int] = {}
    kept: list[dict] = []
    for lineno, rec in enumerate(records, start=1):
        if "_raw" in rec or not isinstance(rec.get("meta"), dict):
            kept.append(rec)
            continue
        rid = record_id(rec)
        if rid in first_line:
            report.drops.append((rid, lineno, first_line[rid]))
            continue
        first_line[rid] = lineno
        kept.append(rec)
    report.dropped = len(report.drops)
    report.kept = len(kept)
    report.duplicate_ids = len({rid for rid, _, _ in report.drops})
    if apply and report.dropped:
        if backup:
            report.backup = backup_corpus(path)
        corpus.rewrite(path, kept)
        report.applied = True
    return report


def backup_corpus(path: str | Path) -> Path:
    """Copy a corpus aside before a destructive rewrite. Returns the backup path.

    `<corpus>.bak` for a single file — the shape every migration script and
    `--fix` has always left. For a sharded corpus it is a `<corpus>.bak/`
    directory holding a copy of each shard, rather than one concatenated file,
    so restoring is `cp` and not a re-shard.
    """
    path = Path(path)
    if not path.is_dir():
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_bytes(path.read_bytes())
        return bak
    bak_dir = path.with_name(path.name + ".bak")
    bak_dir.mkdir(parents=True, exist_ok=True)
    for shard in corpus.shard_paths(path):
        (bak_dir / shard.name).write_bytes(shard.read_bytes())
    return bak_dir


def iter_pairs(path: str | Path) -> Iterator[dict]:
    """Yield every pair record in a corpus, one per non-blank line.

    `path` is a `pairs.jsonl` **or a shard directory** (`stylebot.corpus`); this
    is the seam that makes every reader in the package sharding-agnostic, since
    all of them come through here. Shards are read in `shard_paths` order.

    Tolerant (`stylebot.jsonl` under the hood: UTF-8, blank / undecodable /
    non-object lines skipped, missing file yields nothing). Yields the raw
    parsed dict; callers validate with `validate_pair_record` if they need the
    contract enforced.
    """
    yield from corpus.iter_records(path)
