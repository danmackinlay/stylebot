"""Validation for the shared pair-record contract (D36).

The corpus schema is the seam between capture, synthesis and training;
`validate_pair_record` enforces it and `validate_pairs_file` adds the
corpus-level uniqueness rule.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from pathlib import Path

from stylebot import corpus
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.provenance import PROVENANCE_KEYS

#: The `meta.schema_version` stamped on every record written from now on. 1 is
#: the message triple, the required meta keys below and `meta.provenance`; a
#: record with none is the older corpus (D36). Bump only for a change a
#: consumer must branch on.
PAIR_SCHEMA_VERSION = 1

# meta keys every well-formed pair record carries; the rest are optional
# provenance and the chunk-scale facets (`merge_max_chars`, `chunk_mode`).
REQUIRED_META_KEYS = (
    "source",
    "captured_at",
    "capture_id",
    "chunk_index",
    "chunk_total",
)

# --- meta.gen schema (synthetic pairs only), split by which writer produces
# each group so the synth tests can assert each against a real record.

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
#: Added through a factory's `extra_meta` (`openrouter_generator`'s `provider_sort`).
GEN_EXTRA_KEYS = ("provider_sort",)
#: Added by `synth.synthesize_pairs`: the deliberate-resample label, folded into
#: `synth_key`. Absent means the base corpus.
GEN_REPLICATE_KEYS = ("replicate",)
#: Added by the session loop only (`session_turns > 1`).
GEN_SESSION_KEYS = ("session_id", "session_turn", "context_window", "window_fill")

#: Everything that can appear under `meta.gen`, whatever wrote it.
GEN_META_KEYS = GEN_BASE_KEYS + GEN_EXTRA_KEYS + GEN_REPLICATE_KEYS
# The subset flattened onto each score record: a scores file outlives the
# corpus it was scored from.
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


def system_sha(prompt: str | None = None) -> str:
    """16-hex digest of a system prompt — `STYLE_SYSTEM` unless one is given —
    which a derived artifact carries to name the prompt behind it (D36, D64)."""
    return hashlib.sha256((prompt or STYLE_SYSTEM).encode("utf-8")).hexdigest()[:16]


def build_pair_content(context: str, body: str) -> str:
    """Assemble a pair message body: heading context (verbatim) + the prose body,
    identically on both sides (D14); every producer builds content here."""
    context = (context or "").strip()
    body = body or ""
    return f"{context}\n\n{body}" if context else body


def pair_body(content: str, context: str | None) -> str:
    """The prose body of a pair message, heading-context prefix removed — the
    inverse of `build_pair_content`; content lacking the prefix returns unchanged."""
    if not context:
        return content
    prefix = f"{context}\n\n"
    return content[len(prefix) :] if content.startswith(prefix) else content


def extract_slop(record: dict) -> str:
    """The slop-side body of a pair (messages[1], heading-context stripped)."""
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
    """A short, stable digest of one re-cut's gap policy (`meta.recut`), over
    canonical JSON so it keys off values, not insertion order."""
    try:
        canon = json.dumps(recut, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except TypeError:  # a stamp JSON can't encode — degrade to repr, never raise
        canon = repr(recut)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:8]


def record_id(record: dict) -> str:
    """A stable join key linking a pair's score back to the corpus (D36).

    `meta.synth_key`; else `capture_id:chunk_index`; else a hash of `source` +
    chunk index. A re-cut record folds `meta.recut` into the key, since a
    re-cut reused the `capture_id` and renumbered; records without it keep
    their key byte-for-byte.
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


# --- meta types and domains: key -> (accepted types, optional predicate,
# description), checked only when the key is present. The schema is open.
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
    "archive_id": (str,),
    "before_frontmatter": (dict,),
    "after_frontmatter": (dict,),
    "schema_version": (int,),
    "system_sha": (str,),
    "provenance": (dict,),
    "novel_after": (int, float),
    "mined": (dict,),
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
    "novel_after": (lambda v: 0.0 <= v <= 1.0, "must be in [0, 1]"),
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


def _check_meta_types(meta: dict, prompt: str) -> list[str]:
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
    if isinstance(meta.get("system_sha"), str) and meta["system_sha"] != system_sha(prompt):
        errors.append(
            "meta.system_sha does not match the system prompt "
            f"(record {meta['system_sha']}, prompt {system_sha(prompt)})"
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
                    f"meta.provenance.{key} must be {_type_name(types)}, got {type(value).__name__}"
                )
    return errors


def validate_pair_record(rec: object, *, system_prompt: str | None = None) -> list[str]:
    """Return a list of human-readable problems with one pair record; empty is valid.

    Checks the message triple, that `messages[0]` is `system_prompt` (default
    `STYLE_SYSTEM`), non-empty content, the required `meta` keys and every
    known key's type and domain. Per-record: `record_id` uniqueness is
    `validate_pairs_file`'s (D36).
    """
    errors: list[str] = []
    prompt = system_prompt or STYLE_SYSTEM

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
            if msgs[0].get("content") != prompt:
                errors.append(
                    "system message content is not the view's system prompt "
                    "(a re-stamp is a policy change plus a rebuild, D84)"
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
        errors.extend(_check_meta_types(meta, prompt))

        # A declared meta.context must be the verbatim prefix of both sides (D14).
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
    path: str | Path, *, check_duplicates: bool = True, system_prompt: str | None = None
) -> list[tuple[int | str, list[str]]]:
    """Validate every line of a corpus, and the corpus-level uniqueness rule.

    Returns `(location, errors)` per invalid line; `location` is a 1-based
    line number, or `"<shard>.jsonl:<line>"` for a sharded corpus. `record_id`
    uniqueness is checked across every shard (D36, D50), a repeat reported
    against the later line; `check_duplicates=False` for a partial file.
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
                errs = validate_pair_record(rec, system_prompt=system_prompt)
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


def iter_pairs(path: str | Path) -> Iterator[dict]:
    """Yield every pair record in a corpus — a `pairs.jsonl` or a shard
    directory (D50), in `shard_paths` order. Tolerant: undecodable lines are
    skipped and a missing file yields nothing; records are not validated."""
    yield from corpus.iter_records(path)
