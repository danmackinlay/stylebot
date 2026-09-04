"""The shared by-POST splits contract — one partition, three roles, all stages.

The reward-safety rule (D17, D42) is
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

Extra-free by design: Phase 3 and the eval harness import this without the
`stylebot[classifier]` extra. The only non-stdlib import is `click`, an
unconditional dependency, for the CLI option bundle at the bottom; the
role-assignment core above it touches nothing but the standard library.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# click is an unconditional dependency (never an extra), and `ExtendEvalOpts`
# below is a `cli_options` bundle for the same reason `eval.VerdictOpts` is:
# extending the frozen stratum has two CLIs — `ai-style extend-eval` and the
# blog's `dan-style extend-eval` — and must have one declaration.
import click

from stylebot.cli_options import Emitter, opt, silent
from stylebot.errors import ConfigError, RunError

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
        raise ConfigError(f"eval_frac must be in (0,1), got {eval_frac}")
    if not 0 < detector_frac < 1:
        raise ConfigError(f"detector_frac must be in (0,1), got {detector_frac}")
    posts = sorted(set(posts))
    if not posts:
        raise ConfigError("no posts to split")
    candidates = sorted(set(eval_candidates)) if eval_candidates is not None else posts
    if not candidates:
        raise ConfigError("no eval candidates (pass posts with real pairs, or None for all)")

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

    A post is pinned by *identity*: the path is canonicalised through `aliases`
    first, so adding a renamed post under its new spelling pins the path its
    pairs are already keyed under (D34). Nothing is renumbered or re-keyed —
    role is derived from the recorded `meta.source` at read time.
    """
    known = {canonical_source(p, splits) for p in splits["eval_posts"]}
    additions = sorted({canonical_source(p, splits) for p in posts} - known)
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
    transitively. **The direction is the opposite of the obvious one**: an entry
    reads new → old — "the path this post has *now* → the path its pairs are
    *already keyed under*" — because pointing old at new would re-hash every
    existing pair into a different role (D34). Absent the key this is the
    identity function, so an old `splits.json` behaves as before; a cycle raises.
    """
    aliases = splits.get("aliases") or {}
    seen = {source}
    while source in aliases:
        source = aliases[source]
        if source in seen:
            raise RunError(f"splits aliases: rename cycle through {source!r}")
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
    # BOTH sides go through `canonical_source`: a pin may be spelled with the
    # post's new path while the query arrives under the old one, or the reverse,
    # and only one of those matched a raw scan (D34). Rebuilt per call because a
    # memoised set was tried and reverted (870d030) — a module-global cache made
    # `role_of` non-reentrant and went stale on in-place edits to `eval_posts`,
    # which `test_role_of_memo_does_not_leak_between_splits_docs` guards.
    if source in {canonical_source(p, splits) for p in splits["eval_posts"]}:
        return "eval"
    rule = splits["rest_rule"]
    if rule["kind"] != "hash":  # future-proof the schema
        raise RunError(f"unknown rest_rule kind: {rule['kind']!r}")
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
        raise RunError(f"unsupported splits schema_version: {splits.get('schema_version')!r}")
    for key in ("eval_posts", "rest_rule"):
        if key not in splits:
            raise RunError(f"splits file missing {key!r}")
    if splits["rest_rule"].get("kind") != "hash":
        raise RunError(f"unknown rest_rule kind: {splits['rest_rule'].get('kind')!r}")
    aliases = splits.get("aliases")
    if aliases is not None:
        if not isinstance(aliases, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()
        ):
            raise RunError("splits 'aliases' must be a {path: path} map")
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


def require_pinned_posts(splits: dict, blog_root: str | Path) -> None:
    """`missing_eval_posts`, raised — the gate a run passes before it spends.

    `check_splits` reports the same fact, and nothing forces `check-splits`
    before a paid run, so the training entry points call this instead (D34).
    """
    missing = missing_eval_posts(splits, blog_root)
    if missing:
        raise ConfigError(
            f"{len(missing)} pinned eval post(s) are not on disk under {blog_root}: "
            f"{', '.join(missing)} — renamed or deleted, so the role has silently moved. "
            f"Add the new path to 'aliases' in splits.json (`ai-style check-splits`)."
        )


def is_error(message: str) -> bool:
    """True for a `check_splits` line that must fail a run, not just warn."""
    return message.startswith(ERROR_PREFIX)


def post_pair_counts(splits: dict, ds) -> dict[str, dict[str, int]]:
    """`{canonical post: {pairs, pairs_real, pairs_synthetic}}` for a Dataset.

    Counted by *identity*, not by spelling: a corpus holding pairs under both a
    post's old and new path is one post. `ds` is a `classify_train.Dataset`
    (duck-typed: needs `groups`, `pair_rows`, `pair_is_synthetic`); free
    positives are ignored — roles govern pairs.

    The per-post view `summarize_roles` aggregates and `apply_eval_extension`
    prices a candidate post with, so "how many pairs does this post carry" is
    answered once.
    """
    counts: dict[str, dict[str, int]] = {}
    for k, (s_row, _) in enumerate(ds.pair_rows):
        post = canonical_source(ds.groups[s_row], splits)
        entry = counts.setdefault(post, {"pairs": 0, "pairs_real": 0, "pairs_synthetic": 0})
        entry["pairs"] += 1
        if ds.pair_is_synthetic(k):
            entry["pairs_synthetic"] += 1
        else:
            entry["pairs_real"] += 1
    return counts


def summarize_roles(splits: dict, ds) -> dict:
    """Per-role post/pair counts (with real/synth breakdown) for a Dataset."""
    summary = {
        role: {"posts": set(), "pairs": 0, "pairs_real": 0, "pairs_synthetic": 0} for role in ROLES
    }
    per_post: dict[str, dict[str, int]] = {role: {} for role in ROLES}
    for post, counts in post_pair_counts(splits, ds).items():
        role = role_of(post, splits)
        entry = summary[role]
        entry["posts"].add(post)
        entry["pairs"] += counts["pairs"]
        entry["pairs_real"] += counts["pairs_real"]
        entry["pairs_synthetic"] += counts["pairs_synthetic"]
        per_post[role][post] = counts["pairs"]
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


# ---------------------------------------------------------------------------
# Extending the frozen stratum — the CLI surface for `extend_eval`.
#
# `extend_eval` above is the pure add-only transform; this is the operation
# around it: read the file, price what would move out of the training pools,
# refuse a post that is not on disk, print the before/after, and only then
# write. It is a *subcommand* rather than a flag on `make-splits` because the
# two have opposite safety rules — `make-splits --force` re-cuts (and thereby
# un-freezes) the partition, while this may only ever grow it — and putting
# both behind one command would sit "never unfreeze eval" next to the one flag
# that does. See stylebot CLAUDE.md: anything a user runs as part of the
# pipeline is a subcommand.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtendEvalOpts:
    """The eval-extension surface, declared once for both CLIs.

    `ai-style extend-eval` and the blog's `dan-style extend-eval` (which pins
    the corpus/blog paths) are the two, and the blog's mirror test asserts the
    second carries every flag of the first — so this is a `cli_options` bundle
    rather than eight `click.option` lines retyped in two repos.
    """

    cli_section = "Eval stratum"

    splits_path: Path | None = opt(
        None,
        "--splits",
        "splits_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="The splits.json to extend, in place.",
    )
    posts: tuple[str, ...] = opt(
        (),
        "--post",
        "posts",
        multiple=True,
        help="A POST source path (blog-root-relative, as `meta.source` spells it) to move into "
        "the frozen eval stratum. Repeatable.",
    )
    posts_file: Path | None = opt(
        None,
        "--posts-file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="File of POST source paths, one per line (# comments and blanks ignored), added to "
        "any --post.",
    )
    pairs_path: Path | None = opt(
        None,
        "--pairs",
        "pairs_path",
        # A file OR a shard directory: `stylebot.corpus` makes both first-class
        # and every reader comes through `pairs.iter_pairs`, so refusing a
        # directory here would refuse the shape the blog's corpus now has.
        type=click.Path(exists=True, path_type=Path),
        help="Corpus to price the move against (a pairs.jsonl or a shard directory): per-post "
        "pair counts and the before/after role table + danger report. Strongly recommended — "
        "without it you are extending blind.",
    )
    blog_root: Path | None = opt(
        None,
        "--blog-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Source tree the added posts must exist in (a typo'd path would pin a post that is "
        "not there). Skipped when unset.",
    )
    note: str | None = opt(
        None,
        "--note",
        help="Why this cut happened; recorded in the eval_history entry.",
    )
    apply: bool = opt(
        False,
        "--apply",
        is_flag=True,
        help="Write the extended splits.json. Without it this is a dry run: the stratum is "
        "frozen, so the default is to show the move and change nothing.",
    )


def read_posts_file(path: str | Path) -> list[str]:
    """POST source paths from a file, one per line; `#` comments and blanks out."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [s for s in (line.split("#", 1)[0].strip() for line in lines) if s]


def extend_eval_options(**kw):
    """The shared `extend-eval` flags. `defaults=`/`exclude=` as elsewhere."""
    from stylebot.cli_options import bundle_options

    return bundle_options([ExtendEvalOpts], **kw)


def apply_eval_extension(opts: ExtendEvalOpts, *, emit: Emitter = silent) -> dict:
    """Preview (and with `opts.apply`, write) `eval_v(n+1) = eval_v(n) ∪ posts`.

    The whole operation, so the two CLIs cannot drift in what they refuse or in
    what they print before writing. Returns the report as data:
    `additions` (each with the role it is leaving and its pair counts),
    `already_eval`, `roles_before`/`roles_after` when a corpus was given, and
    the eval version and file hash on each side.

    Refuses: an empty post list; a post that is not on disk under `blog_root`.
    Adding only posts that are already pinned is a no-op — it returns the report
    with `applied: False` and does not churn `eval_version`.
    """
    splits_path = Path(opts.splits_path)
    splits = load_splits(splits_path)
    posts = [*opts.posts, *(read_posts_file(opts.posts_file) if opts.posts_file else [])]
    if not posts:
        raise ConfigError("no posts to add — pass --post (repeatable) and/or --posts-file")

    if opts.blog_root is not None:
        root = Path(opts.blog_root)
        absent = [p for p in posts if not (root / p).exists()]
        if absent:
            raise ConfigError(
                f"{len(absent)} post(s) to add are not on disk under {root}: "
                f"{', '.join(absent)} — pinning a path that does not exist is the leak "
                f"check-splits exists to catch"
            )

    pinned = {canonical_source(p, splits) for p in splits["eval_posts"]}
    additions, already = [], []
    for post in dict.fromkeys(posts):  # de-dup, order preserved
        (already if canonical_source(post, splits) in pinned else additions).append(post)

    counts: dict[str, dict[str, int]] = {}
    ds = None
    if opts.pairs_path is not None:
        from stylebot import classify_train

        ds = classify_train.assemble_dataset(opts.pairs_path)
        counts = post_pair_counts(splits, ds)

    extended = extend_eval(splits, additions, note=opts.note) if additions else splits
    report = {
        "splits_path": str(splits_path),
        "splits_sha256_before": splits_sha256(splits_path),
        "eval_version_before": eval_version(splits),
        "eval_version_after": eval_version(extended),
        "already_eval": already,
        "additions": [
            {
                "post": post,
                "from_role": role_of(post, splits),
                **counts.get(canonical_source(post, splits), {"pairs": 0}),
            }
            for post in additions
        ],
        "roles_before": summarize_roles(splits, ds) if ds is not None else None,
        "roles_after": summarize_roles(extended, ds) if ds is not None else None,
        "applied": False,
    }

    emit(
        f"eval stratum v{report['eval_version_before']} "
        f"splits_sha256={report['splits_sha256_before']}  ({splits_path})"
    )
    for post in already:
        emit(f"  already eval: {post}")
    if not additions:
        emit("nothing to add — every post given is already pinned; eval_version unchanged")
        return report
    emit(f"would re-role {len(additions)} post(s) into the frozen eval stratum:")
    for entry in report["additions"]:
        pairs = (
            f"{entry['pairs']} pair(s) ({entry.get('pairs_real', 0)} real / "
            f"{entry.get('pairs_synthetic', 0)} synthetic)"
            if ds is not None
            else "pair counts unknown (pass --pairs)"
        )
        emit(f"  {entry['from_role']:>8} -> eval  {entry['post']}  [{pairs}]")

    if ds is not None:
        for role in ROLES:
            before, after = report["roles_before"][role], report["roles_after"][role]
            emit(
                f"{role:>8}: {before['posts']} -> {after['posts']} post(s), "
                f"{before['pairs']} -> {after['pairs']} pair(s) "
                f"({before['pairs_real']} -> {after['pairs_real']} real / "
                f"{before['pairs_synthetic']} -> {after['pairs_synthetic']} synthetic); "
                f"top-post share {before['top_post_share']:.0%} -> {after['top_post_share']:.0%}"
            )
        for warning in check_splits(extended, ds):
            emit(f"WARNING (after): {warning}", level="warn")

    if not opts.apply:
        emit("dry run — nothing written; pass --apply to write", level="note")
        return report

    save_splits(extended, splits_path)
    report["applied"] = True
    report["splits_sha256_after"] = splits_sha256(splits_path)
    emit(
        f"wrote {splits_path} -> eval stratum v{report['eval_version_after']} "
        f"splits_sha256={report['splits_sha256_after']} "
        f"({len(extended['eval_posts'])} pinned path(s))"
    )
    emit(
        "the detector artifact and any verdict computed under the previous stratum are now "
        "stale: refit (`train-clf --splits`) before comparing anything",
        level="note",
    )
    return report
