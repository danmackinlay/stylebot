"""The shared by-POST splits contract — one partition, three roles, all stages.

The reward-safety rule (see `_plans/eval-harness.md` "the split contract") is
that the detector must never train on the posts the styler trains on or that
the frozen eval scores. This module is the single source of truth for that
partition: a small committed `splits.json` mapping every POST — present and
future — to one of three roles:

- **eval**: the frozen final-eval posts. Pinned as an explicit list; no fitting
  or selection decision may ever touch them.
- **styler**: Phase-3 training posts for the slop-remover.
- **detector**: the classifier pool. Hyperparameter selection (C, backbone)
  happens by *nested group-CV inside this pool* — there is deliberately no
  fourth materialised split; at blog-corpus scale it would be uselessly small.

**Growth stability.** Only the eval list is materialised. Every non-eval post is
assigned by a deterministic hash rule (`sha256(salt:source)` → uniform [0,1) →
detector if below `detector_frac`, else styler), so a post's role never changes
as the corpus grows, new posts flow to detector/styler in fixed proportion with
no file churn, and the eval stratum cannot drift. Synthetic pairs inherit their
source post's role via `meta.source`.

**Renames.** A role is keyed on the post's *source path*, which is not stable:
`post-op move` renames posts, and a renamed post silently changes role (a
pinned eval post stops being pinned; a hash-assigned one re-hashes into the
other pool). The optional `aliases` map fixes that — see `canonical_source`.

Stdlib-only by design: Phase 3 and the eval harness import this without the
`stylebot[classifier]` extra.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
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

# A stratum where one post supplies more than this share of the pairs is not the
# n it claims to be: chunks inside a post share topic, register and whatever the
# generator did to it, so a chunk-level confidence interval over such a stratum
# is too narrow. Measured 2026-09-03: `notebook/automatic_coding.qmd` is 77 of
# the 103 eval pairs (75%), which is what run 4a's "127 chunks, paired" CI
# treated as 127 independent observations. Every comparison over the stratum
# should cluster by post (`eval.paired_delta(clusters=…)`), and the stratum
# itself should be extended (`extend_eval`) rather than merely bootstrapped.
MAX_POST_SHARE_PER_ROLE = 0.5

# The frozen eval stratum is versioned so a verdict can name the stratum it was
# computed on. v1 is the original `make_splits` cut; extensions bump it.
DEFAULT_EVAL_VERSION = 1

# `check_splits` returns one flat list of report lines. Lines starting with this
# prefix are *errors*, not warnings: a pinned eval post that is not on disk is a
# broken contract (the post was renamed and its role has silently moved), and a
# caller must fail on it rather than print it in yellow and carry on.
ERROR_PREFIX = "MISSING:"


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
    synthetic paraphrases. Deterministic in `seed`.
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
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "eval_version": DEFAULT_EVAL_VERSION,
        "eval_posts": eval_posts,
        "rest_rule": {"kind": "hash", "detector_frac": detector_frac, "salt": str(seed)},
    }


def eval_version(splits: dict) -> int:
    """Which cut of the frozen stratum this is. Unversioned files read as v1."""
    return int(splits.get("eval_version") or DEFAULT_EVAL_VERSION)


def extend_eval(splits: dict, posts, *, note: str | None = None) -> dict:
    """`eval_v(n+1) = eval_v(n) ∪ posts` — add-only, and refuses anything else.

    The frozen stratum may **grow** and may never shrink. Growing it re-roles
    the added posts out of `styler`/`detector` and into `eval`, which is the
    safe direction: prose leaves the training pools, it never enters them. The
    opposite move (dropping a post out of eval) would retroactively license
    training on prose an existing verdict was computed over, so it is not
    expressible here at all — you would cut a new file with `make_splits`.

    Returns a NEW splits dict with `eval_version` bumped and an `eval_history`
    entry recording what was added, by whom (the note) and when. Adding nothing
    new is a no-op that returns the input unchanged rather than churning the
    version.

    Two things this deliberately does NOT do. It does not touch `aliases`: a
    renamed post is the same identity and is already handled (D34). And it does
    not renumber or re-key any pair — role is derived from the recorded
    `meta.source` at read time, so extending the stratum needs no migration.
    """
    known = {canonical_source(p, splits) for p in splits["eval_posts"]}
    additions = sorted({p for p in posts if canonical_source(p, splits) not in known})
    if not additions:
        return splits
    out = dict(splits)
    out["eval_posts"] = sorted({*splits["eval_posts"], *additions})
    out["eval_version"] = eval_version(splits) + 1
    history = list(splits.get("eval_history") or [])
    history.append(
        {
            "eval_version": out["eval_version"],
            "at": datetime.now(UTC).isoformat(),
            "added": additions,
            "note": note,
        }
    )
    out["eval_history"] = history
    return out


def splits_sha256(path: str | Path) -> str:
    """16-hex content hash of a `splits.json` — the manifest pin.

    A run manifest already pins the corpus hash and both repos' SHAs but not the
    partition, so "which posts were held out of this run" was answerable only by
    trusting that the file had not moved since. It moves: D34 added `aliases`,
    and `extend_eval` moves posts into the frozen stratum. Same 16-hex
    convention as `provenance.file_sha256`, which this delegates to.
    """
    from stylebot.provenance import file_sha256

    return file_sha256(path)


def unit_from_digest(digest: bytes) -> float:
    """A sha256 digest → a uniform coordinate in [0, 1). The shared half of the rule.

    `targets.hash_unit` draws its subsamples through this same conversion. Only
    the *conversion* is shared — what goes into the hash stays each caller's
    business, because the byte-level input here is a frozen contract (the
    committed `splits.json` pins every post's role through `f"{salt}:{source}"`).
    """
    return int.from_bytes(digest[:8], "big") / 2**64


def canonical_source(source: str, splits: dict) -> str:
    """Resolve a post path to the identity its role is keyed on.

    `splits["aliases"]` is an optional `{alias: canonical}` map, consulted
    transitively. **The direction is load-bearing and is the opposite of the
    obvious one**: an entry reads "the path this post has *now* → the path its
    pairs are *already keyed under*", i.e. new → old, not old → new. Roles are
    assigned from `meta.source` as recorded at capture/synth time, so pointing
    the old path at the new one would re-hash every existing pair into a
    different role; pointing the new path at the old one leaves the whole corpus
    exactly where it is and needs no migration. Absent the key this is the
    identity function, so an old `splits.json` behaves as before.

    A cycle raises rather than looping.
    """
    aliases = splits.get("aliases") or {}
    seen = {source}
    while source in aliases:
        source = aliases[source]
        if source in seen:
            raise ValueError(f"splits aliases: rename cycle through {source!r}")
        seen.add(source)
    return source


def role_of(source: str, splits: dict) -> str:
    """The single role-assignment function: eval if pinned, else the hash rule.

    Deterministic and open-world: any post, including ones that did not exist
    when the file was created, gets a stable role. `source` is resolved through
    `canonical_source` first, so a renamed post keeps the role — and therefore
    the pairs — it had under its old path.
    """
    source = canonical_source(source, splits)
    # Scan the pinned list directly. It holds a handful of posts and this runs
    # per-pair/per-chunk, so `set(...)` was building a set per call to answer one
    # question — strictly more work than the scan it replaced. A *memoised* set
    # was tried and reverted (870d030): a module-global cache made `role_of`
    # non-reentrant and went stale on in-place edits to `eval_posts`, which
    # `test_role_of_memo_does_not_leak_between_splits_docs` still guards against.
    if source in splits["eval_posts"]:
        return "eval"
    rule = splits["rest_rule"]
    if rule["kind"] != "hash":  # future-proof the schema
        raise ValueError(f"unknown rest_rule kind: {rule['kind']!r}")
    u = unit_from_digest(hashlib.sha256(f"{rule['salt']}:{source}".encode()).digest())
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
    aliases = splits.get("aliases")
    if aliases is not None:
        if not isinstance(aliases, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
        ):
            raise ValueError("splits 'aliases' must be a {path: path} map")
        for alias in aliases:  # fail at load, not deep inside a training run
            canonical_source(alias, splits)
    return splits


def load_corpus_splits(data_dir: str | Path | None = None) -> dict | None:
    """The splits contract sitting in a corpus dir, or None when there is none.

    The guards that must not *require* a splits file — synth's frozen-eval
    refusal, `log save`'s eval notice — go through here, so "this corpus has a
    contract" is asked once and answered the same way everywhere.
    """
    from stylebot import config

    path = config.splits_path(data_dir)
    return load_splits(path) if path.exists() else None


def eval_post_spellings(splits: dict) -> dict[str, list[str]]:
    """The pinned eval stratum as `{identity: every path that names it}`.

    Pinning is by *identity*, not by string: a renamed eval post may legitimately
    appear in `eval_posts` under both spellings (add-only, so pairs keyed to the
    old one keep their role) and must still count once. Aliases of a pinned post
    join its group even when only one spelling is listed.
    """
    aliases = splits.get("aliases") or {}
    groups: dict[str, set[str]] = {}
    for post in splits["eval_posts"]:
        groups.setdefault(canonical_source(post, splits), set()).add(post)
    for alias, target in aliases.items():
        canonical = canonical_source(target, splits)
        if canonical in groups:
            groups[canonical].add(alias)
    return {canonical: sorted(paths) for canonical, paths in sorted(groups.items())}


def missing_eval_posts(splits: dict, blog_root: str | Path) -> list[str]:
    """Pinned eval identities with no spelling present under `blog_root`.

    This is the check that would have caught the 2026-09 leak: two eval posts
    were renamed, the pinned paths kept their pairs (so the STALE warning never
    fired), and the new paths hashed into `styler`/`detector`. A pinned post
    that is not on disk under *any* of its known names is a broken contract.
    """
    root = Path(blog_root)
    return [
        canonical
        for canonical, paths in eval_post_spellings(splits).items()
        if not any((root / path).exists() for path in paths)
    ]


def is_error(message: str) -> bool:
    """True for a `check_splits` line that must fail a run, not just warn."""
    return message.startswith(ERROR_PREFIX)


def summarize_roles(splits: dict, ds) -> dict:
    """Per-role post/pair counts (with real/synth breakdown) for a Dataset.

    `ds` is a `classify_train.Dataset` (duck-typed: needs `groups`, `pair_rows`,
    `pair_is_synthetic`). Free positives are ignored — roles govern pairs.
    """
    summary = {
        role: {"posts": set(), "pairs": 0, "pairs_real": 0, "pairs_synthetic": 0} for role in ROLES
    }
    per_post: dict[str, dict[str, int]] = {role: {} for role in ROLES}
    for k, (s_row, _) in enumerate(ds.pair_rows):
        # Count by identity, not by spelling: a corpus holding pairs under both
        # a post's old and new path is one post, in one role.
        post = canonical_source(ds.groups[s_row], splits)
        role = role_of(post, splits)
        entry = summary[role]
        entry["posts"].add(post)
        entry["pairs"] += 1
        per_post[role][post] = per_post[role].get(post, 0) + 1
        if ds.pair_is_synthetic(k):
            entry["pairs_synthetic"] += 1
        else:
            entry["pairs_real"] += 1
    out = {}
    for role, entry in summary.items():
        counts = per_post[role]
        top_post, top_pairs = max(counts.items(), key=lambda kv: kv[1]) if counts else (None, 0)
        out[role] = {
            **entry,
            "posts": len(entry["posts"]),
            # The concentration diagnostic: what fraction of this stratum comes
            # from its single largest post. See MAX_POST_SHARE_PER_ROLE.
            "top_post": top_post,
            "top_post_pairs": top_pairs,
            "top_post_share": (top_pairs / entry["pairs"]) if entry["pairs"] else 0.0,
        }
    return out


def check_splits(splits: dict, ds, *, blog_root: str | Path | None = None) -> list[str]:
    """The danger report: warnings for strata too small (or too synthetic) to trust.

    Returns human-readable report strings; empty list = healthy. Consumers must
    surface these (CLI echo) and record them (artifact `meta.split.warnings`) —
    a metric computed on a dangerously small stratum should never travel without
    its caveat.

    With `blog_root`, also verifies that every pinned eval post still exists on
    disk. Those lines carry `ERROR_PREFIX` and are **errors**: a caller should
    fail on `is_error(line)` rather than print it and continue.
    """
    warnings: list[str] = []
    if blog_root is not None:
        for canonical in missing_eval_posts(splits, blog_root):
            paths = ", ".join(eval_post_spellings(splits)[canonical])
            warnings.append(
                f"{ERROR_PREFIX} pinned eval post {canonical!r} is not on disk under "
                f"{blog_root} (tried: {paths}) — it was renamed or deleted, and its role "
                f"has silently moved; add the new path to 'aliases' in splits.json"
            )
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

    for role in ROLES:
        entry = roles[role]
        if entry["pairs"] and entry["top_post_share"] > MAX_POST_SHARE_PER_ROLE:
            warnings.append(
                f"CONCENTRATED: role '{role}' draws {entry['top_post_share']:.0%} of its "
                f"{entry['pairs']} pair(s) from one post ({entry['top_post']}) — a chunk-level "
                f"confidence interval over it is too narrow; cluster by post and extend the "
                f"stratum"
            )

    if roles["eval"]["pairs"] and roles["eval"]["pairs_real"] == 0:
        warnings.append(
            "EVAL PURITY: the eval stratum contains only synthetic pairs — the frozen "
            "eval would measure detection of our own paraphrase generator, not slop"
        )
    elif roles["eval"]["pairs_synthetic"] > roles["eval"]["pairs_real"]:
        warnings.append(
            f"EVAL PURITY: {roles['eval']['pairs_synthetic']} of {roles['eval']['pairs']} eval "
            f"pairs are synthetic — the headline verdict must be the REAL facet "
            f"({roles['eval']['pairs_real']} pairs), with the synthetic one reported separately"
        )

    corpus_posts = {canonical_source(g, splits) for g in ds.groups}
    stale = sorted({canonical_source(p, splits) for p in splits["eval_posts"]} - corpus_posts)
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
