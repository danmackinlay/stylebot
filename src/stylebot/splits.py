"""The shared by-POST splits contract — one partition, three roles, all stages.

The reward-safety rule (see `_plans/eval-harness.md` "the split contract") is
that the detector must never train on the posts the styler trains on or that
the frozen eval scores. This module is the single source of truth for that
partition: a small committed `splits.json` mapping every POST — present and
future — to one of three roles:

- **eval**: the frozen final-eval posts. Pinned as an explicit list; nothing
  else may ever add to or remove from it silently. No fitting or selection
  decision may touch these posts.
- **styler**: Phase-3 training posts for the slop-remover.
- **detector**: the classifier pool. Hyperparameter selection (C, backbone)
  happens by *nested group-CV inside this pool* — there is deliberately no
  fourth materialised split; at blog-corpus scale it would be uselessly small.

**Growth stability.** The corpus grows (synth generation over new posts), so
only the eval list is materialised. Every non-eval post is assigned by a
deterministic hash rule (`sha256(salt:source)` → uniform [0,1) → detector if
below `detector_frac`, else styler). A post's role therefore never changes as
the corpus grows, new posts flow to detector/styler in fixed proportion with
no file churn, and the eval stratum cannot drift. Synthetic pairs inherit
their source post's role via `meta.source`.

Stdlib-only by design: Phase 3 and the eval harness import this without the
`stylebot[classifier]` extra.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ROLES = ("eval", "styler", "detector")

DEFAULT_EVAL_FRAC = 0.2
DEFAULT_DETECTOR_FRAC = 0.4

# Danger-report thresholds: below these a role's metric/training is running on
# fumes and every consumer should say so out loud.
MIN_POSTS_PER_ROLE = 5
MIN_PAIRS_PER_ROLE = 20
MIN_DETECTOR_POSTS_FOR_CV = 10


def make_splits(
    posts,
    *,
    eval_frac: float = DEFAULT_EVAL_FRAC,
    detector_frac: float = DEFAULT_DETECTOR_FRAC,
    seed: int = 0,
    eval_candidates=None,
) -> dict:
    """Create the splits contract: a pinned eval list + the hash rule for the rest.

    `posts` is the universe of POST sources known now (used only to size the
    eval sample); `eval_candidates` restricts which posts are eligible for eval
    — pass the real-capture posts so the frozen eval isn't made of our own
    synthetic paraphrases. The eval sample is deterministic in `seed`.
    """
    if not 0 < eval_frac < 1:
        raise ValueError(f"eval_frac must be in (0,1), got {eval_frac}")
    if not 0 < detector_frac < 1:
        raise ValueError(f"detector_frac must be in (0,1), got {detector_frac}")
    posts = sorted(set(posts))
    if not posts:
        raise ValueError("no posts to split")
    candidates = sorted(set(eval_candidates)) if eval_candidates is not None else posts
    if not candidates:
        raise ValueError("no eval candidates (pass posts with real pairs, or None for all)")

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    n_eval = max(1, round(len(posts) * eval_frac))
    eval_posts = sorted(shuffled[:n_eval])

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "eval_posts": eval_posts,
        "rest_rule": {"kind": "hash", "detector_frac": detector_frac, "salt": str(seed)},
    }


def role_of(source: str, splits: dict) -> str:
    """The single role-assignment function: eval if pinned, else the hash rule.

    Deterministic and open-world: any post, including ones that did not exist
    when the file was created, gets a stable role.
    """
    if source in set(splits["eval_posts"]):
        return "eval"
    rule = splits["rest_rule"]
    if rule["kind"] != "hash":  # future-proof the schema
        raise ValueError(f"unknown rest_rule kind: {rule['kind']!r}")
    digest = hashlib.sha256(f"{rule['salt']}:{source}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform [0,1)
    return "detector" if u < rule["detector_frac"] else "styler"


def save_splits(splits: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    return path


def load_splits(path: str | Path) -> dict:
    splits = json.loads(Path(path).read_text(encoding="utf-8"))
    if splits.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported splits schema_version: {splits.get('schema_version')!r}")
    for key in ("eval_posts", "rest_rule"):
        if key not in splits:
            raise ValueError(f"splits file missing {key!r}")
    if splits["rest_rule"].get("kind") != "hash":
        raise ValueError(f"unknown rest_rule kind: {splits['rest_rule'].get('kind')!r}")
    return splits


def summarize_roles(splits: dict, ds) -> dict:
    """Per-role post/pair counts (with real/synth breakdown) for a Dataset.

    `ds` is a `classify_train.Dataset` (duck-typed: needs `groups`, `pair_rows`,
    `pair_is_synthetic`). Free positives are ignored — roles govern pairs.
    """
    summary = {
        role: {"posts": set(), "pairs": 0, "pairs_real": 0, "pairs_synthetic": 0}
        for role in ROLES
    }
    for k, (s_row, _) in enumerate(ds.pair_rows):
        role = role_of(ds.groups[s_row], splits)
        entry = summary[role]
        entry["posts"].add(ds.groups[s_row])
        entry["pairs"] += 1
        if ds.pair_is_synthetic(k):
            entry["pairs_synthetic"] += 1
        else:
            entry["pairs_real"] += 1
    return {
        role: {**entry, "posts": len(entry["posts"])} for role, entry in summary.items()
    }


def check_splits(splits: dict, ds) -> list[str]:
    """The danger report: warnings for strata too small (or too synthetic) to trust.

    Returns human-readable warning strings; empty list = healthy. Consumers must
    surface these (CLI echo) and record them (artifact `meta.split.warnings`) —
    a metric computed on a dangerously small stratum should never travel without
    its caveat.
    """
    warnings: list[str] = []
    roles = summarize_roles(splits, ds)

    for role in ROLES:
        entry = roles[role]
        if entry["posts"] < MIN_POSTS_PER_ROLE:
            warnings.append(
                f"DANGEROUSLY SMALL: role '{role}' has only {entry['posts']} post(s) "
                f"(< {MIN_POSTS_PER_ROLE}) — by-POST metrics on it are noise-dominated"
            )
        if entry["pairs"] < MIN_PAIRS_PER_ROLE:
            warnings.append(
                f"DANGEROUSLY SMALL: role '{role}' has only {entry['pairs']} pair(s) "
                f"(< {MIN_PAIRS_PER_ROLE})"
            )

    if roles["eval"]["pairs"] and roles["eval"]["pairs_real"] == 0:
        warnings.append(
            "EVAL PURITY: the eval stratum contains only synthetic pairs — the frozen "
            "eval would measure detection of our own paraphrase generator, not slop"
        )

    corpus_posts = set(ds.groups)
    stale = [p for p in splits["eval_posts"] if p not in corpus_posts]
    if stale:
        warnings.append(
            f"STALE: {len(stale)} pinned eval post(s) have no pairs in this corpus "
            f"(e.g. {stale[0]}) — eval is effectively smaller than pinned"
        )

    if roles["detector"]["posts"] < MIN_DETECTOR_POSTS_FOR_CV:
        warnings.append(
            f"CV UNSTABLE: detector pool has {roles['detector']['posts']} post(s) "
            f"(< {MIN_DETECTOR_POSTS_FOR_CV}) — cross-validated metrics and nested "
            f"C-selection will be high-variance"
        )
    return warnings
