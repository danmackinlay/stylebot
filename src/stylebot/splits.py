"""The shared by-post splits contract — one partition, three roles, all stages.

The detector never trains on the posts the styler trains on or the frozen
eval scores (D17, D42). A committed `splits.json` maps every post, present
and future, to `eval` (a pinned list), `styler` or `detector` (the hash rule
`sha256(salt:source)` -> uniform [0,1) -> detector below `detector_frac`),
so a role never changes as the corpus grows; synthetic pairs inherit their
source post's role. A role is keyed on the post's identity through the
optional `aliases` map (`canonical_source`, D34). Stdlib plus `click` only.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# `ExtendEvalOpts` below is a `cli_options` bundle: two CLIs, one declaration (D78).
import click

from stylebot.cli_options import Emitter, opt, silent
from stylebot.errors import ConfigError, RunError

SCHEMA_VERSION = 1
ROLES = ("eval", "styler", "detector")

DEFAULT_EVAL_FRAC = 0.2
DEFAULT_DETECTOR_FRAC = 0.4

# Danger-report thresholds: below these every consumer says so out loud.
MIN_POSTS_PER_ROLE = 5
MIN_PAIRS_PER_ROLE = 20
MIN_DETECTOR_POSTS_FOR_CV = 10

# A stratum where one post supplies more than this share of the pairs is not
# the n it claims to be (D40): cluster by post, and extend the stratum.
MAX_POST_SHARE_PER_ROLE = 0.5

# The frozen eval stratum is versioned (D42); v1 is the `make_splits` cut.
DEFAULT_EVAL_VERSION = 1

# A `check_splits` line with this prefix is an error a caller must fail on (D34).
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
    `posts` sizes the eval sample; `eval_candidates` restricts who is eligible
    (pass the real-capture posts). Deterministic in `seed`."""
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
    """`eval_v(n+1) = eval_v(n) ∪ posts` — add-only; shrinking is not
    expressible (D42).

    Returns a new splits dict with `eval_version` bumped and an `eval_history`
    entry; adding nothing new returns the input unchanged. A post is pinned by
    identity through `aliases` (D34); nothing is re-keyed.
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
    """16-hex content hash of a `splits.json` — the manifest pin (D42), by
    `provenance.file_sha256`."""
    from stylebot.provenance import file_sha256

    return file_sha256(path)


def unit_from_digest(digest: bytes) -> float:
    """A sha256 digest → a uniform coordinate in [0, 1); the conversion
    `targets.hash_unit` shares, while the hash input stays each caller's."""
    return int.from_bytes(digest[:8], "big") / 2**64


def canonical_source(source: str, splits: dict) -> str:
    """Resolve a post path to the identity its role is keyed on (D34).

    `splits["aliases"]` is an optional `{alias: canonical}` map, consulted
    transitively, reading new path → the path its pairs are already keyed
    under. Absent the key this is the identity; a cycle raises.
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
    """The single role-assignment function: eval if pinned, else the hash rule;
    open-world, and `source` is resolved through `canonical_source` first."""
    source = canonical_source(source, splits)
    # Both sides go through `canonical_source` (D34). Rebuilt per call: a
    # memoised set went stale on in-place edits to `eval_posts`.
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
    """The splits contract sitting in a corpus dir, or None when there is none;
    the one reader for guards that must not require one."""
    from stylebot import config

    path = config.splits_path(data_dir)
    return load_splits(path) if path.exists() else None


def source_spellings(splits: dict, sources) -> dict[str, list[str]]:
    """`{identity: every path that names it}` for a set of post paths; a post
    listed under both spellings counts once."""
    groups: dict[str, set[str]] = {}
    for post in sources:
        groups.setdefault(canonical_source(post, splits), set()).add(post)
    for alias, target in (splits.get("aliases") or {}).items():
        canonical = canonical_source(target, splits)
        if canonical in groups:
            groups[canonical].add(alias)
    return {canonical: sorted(paths) for canonical, paths in sorted(groups.items())}


def eval_post_spellings(splits: dict) -> dict[str, list[str]]:
    """The pinned eval stratum as `{identity: every path that names it}`."""
    return source_spellings(splits, splits["eval_posts"])


def missing_sources(splits: dict, sources, blog_root: str | Path) -> dict[str, list[str]]:
    """Identities among `sources` on disk under none of their spellings (D34)."""
    root = Path(blog_root)
    return {
        canonical: paths
        for canonical, paths in source_spellings(splits, sources).items()
        if not any((root / path).exists() for path in paths)
    }


def missing_eval_posts(splits: dict, blog_root: str | Path) -> list[str]:
    """Pinned eval identities with no spelling present under `blog_root`."""
    return list(missing_sources(splits, splits["eval_posts"], blog_root))


def corpus_sources(pairs_path: str | Path) -> set[str]:
    """Every `meta.source` a pairs corpus (file, shards or view) names."""
    from stylebot.pairs import iter_pairs

    return {s for rec in iter_pairs(pairs_path) if (s := (rec.get("meta") or {}).get("source"))}


def require_sources_on_disk(splits: dict, sources, blog_root: str | Path) -> None:
    """`missing_sources` over the pinned eval posts and `sources`, raised — the
    gate a run passes before it spends, since nothing forces `check-splits`
    first (D34). Names each identity by its role: pinned eval post or corpus source."""
    missing = missing_sources(splits, set(splits["eval_posts"]) | set(sources), blog_root)
    if missing:
        pinned = eval_post_spellings(splits)
        named = ", ".join(
            f"{'pinned eval post' if c in pinned else 'corpus source'} {c!r}" for c in missing
        )
        raise ConfigError(
            f"{len(missing)} missing source(s) under {blog_root}: {named} — renamed or "
            f"deleted, so its role has silently moved. Add the new path to 'aliases' in "
            f"splits.json (`ai-style check-splits`)."
        )


def is_error(message: str) -> bool:
    """True for a `check_splits` line that must fail a run, not just warn."""
    return message.startswith(ERROR_PREFIX)


def post_pair_counts(splits: dict, ds) -> dict[str, dict[str, int]]:
    """`{canonical post: {pairs, pairs_real, pairs_synthetic}}` for a Dataset
    (duck-typed: `groups`, `pair_rows`, `pair_is_synthetic`), counted by
    identity; free positives are ignored."""
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
            # The share of this stratum from its largest post (MAX_POST_SHARE_PER_ROLE).
            "top_post": top_post,
            "top_post_pairs": top_pairs,
            "top_post_share": (top_pairs / entry["pairs"]) if entry["pairs"] else 0.0,
        }
    return out


def role_lines(splits: dict, ds, *, top: bool = False) -> list[str]:
    """The per-role post/pair table `make-splits` and `check-splits` print;
    `top` appends each stratum's dominant post and its share."""
    roles = summarize_roles(splits, ds)
    lines = []
    for role in ROLES:
        e = roles[role]
        share = ""
        if top and e["top_post"]:
            share = f"; top post {e['top_post']} {e['top_post_share']:.0%}"
        lines.append(
            f"{role:>8}: {e['posts']} post(s), {e['pairs']} pair(s) "
            f"({e['pairs_real']} real / {e['pairs_synthetic']} synthetic){share}"
        )
    return lines


def check_splits(splits: dict, ds, *, blog_root: str | Path | None = None) -> list[str]:
    """The danger report: warnings for strata too small (or too synthetic) to
    trust, as report lines (empty = healthy) a consumer surfaces and records.
    With `blog_root`, missing pinned or corpus posts are `ERROR_PREFIX` lines
    a caller must fail on (`is_error`)."""
    warnings: list[str] = []
    if blog_root is not None:
        warnings.extend(_missing_on_disk_lines(splits, ds, blog_root))
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


def _missing_on_disk_lines(splits: dict, ds, blog_root: str | Path) -> list[str]:
    """`check_splits`'s on-disk section: pinned eval posts, then corpus sources."""
    lines: list[str] = []
    for canonical, paths in missing_sources(splits, splits["eval_posts"], blog_root).items():
        lines.append(
            f"{ERROR_PREFIX} pinned eval post {canonical!r} is not on disk under "
            f"{blog_root} (tried: {', '.join(paths)}) — it was renamed or deleted, and its "
            f"role has silently moved; add the new path to 'aliases' in splits.json"
        )
    pinned = set(eval_post_spellings(splits))
    gone = {
        canonical: paths
        for canonical, paths in missing_sources(splits, set(ds.groups), blog_root).items()
        if canonical not in pinned
    }
    for canonical, paths in gone.items():
        lines.append(
            f"{ERROR_PREFIX} corpus source {canonical!r} is not on disk under {blog_root} "
            f"(tried: {', '.join(paths)}) — it was renamed or deleted, and pairs captured "
            f"under its new path hash into a role of their own; add the new path to "
            f"'aliases' in splits.json"
        )
    if gone:
        lines.append(f"{len(gone)} corpus source(s) missing from disk under {blog_root}")
    return lines


# --- extending the frozen stratum: the operation around `extend_eval`, a
# subcommand of its own because `make-splits --force` has the opposite safety rule.


@dataclass(frozen=True)
class ExtendEvalOpts:
    """The eval-extension surface, declared once for both CLIs (D78)."""

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
        # A file or a shard directory: both are first-class (D50).
        type=click.Path(exists=True, path_type=Path),
        help="Corpus to price the move against (a pairs.jsonl or a shard directory): per-post "
        "pair counts and the before/after role table + danger report. Strongly recommended — "
        "without it you are extending blind.",
    )
    blog_root: Path | None = opt(
        None,
        "--blog-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        is_eager=True,  # processed before any corpus default that reads it (D79)
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

    Returns the report as data: `additions` (with the role each leaves and its
    pair counts), `already_eval`, `roles_before`/`roles_after` when a corpus
    was given, and the version and hash on each side. Refuses an empty list or
    a post not on disk under `blog_root`; nothing new is `applied: False`.
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


@dataclass(frozen=True)
class CheckSplitsOpts:
    """The `check-splits` surface — what `check_splits_pass` takes (D78)."""

    cli_section = "Splits"

    splits_path: Path | None = opt(
        None,
        "--splits",
        "splits_path",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="The splits.json to check.",
    )
    blog_root: Path = opt(
        Path("."),
        "--blog-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        show_default=True,
        help="Source tree the pinned eval posts must still exist in.",
    )
    pairs_path: Path | None = opt(
        None,
        "--pairs",
        "pairs_path",
        type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
        help="A scratch corpus for the per-role table and danger report; the corpus is --draws.",
    )


def check_splits_options(**kw):
    """`CheckSplitsOpts` + `draws.ViewOpts`, one decorator; `kw` is `bundle_options`'."""
    from stylebot.cli_options import bundle_options
    from stylebot.draws import ViewOpts  # `targets` imports this module, so not at top

    return bundle_options([CheckSplitsOpts, ViewOpts], **kw)


def check_splits_pass(opts: CheckSplitsOpts, view, *, emit: Emitter = silent) -> None:
    """The `check-splits` pass, read-only: pinned eval posts on disk under `blog_root`,
    then, given a corpus (`view.corpus_or_view(opts.pairs_path)`), the per-role table
    and the danger report. A missing source is a `RunError` raised after the report."""
    from stylebot import classify_train as ct

    pairs_path = view.corpus_or_view(opts.pairs_path)
    splits_doc = load_splits(opts.splits_path)
    spellings = eval_post_spellings(splits_doc)
    missing = set(missing_eval_posts(splits_doc, opts.blog_root))
    sha = splits_sha256(opts.splits_path)
    emit(f"eval stratum v{eval_version(splits_doc)}  splits_sha256={sha}")
    on_disk = f"{len(spellings) - len(missing)}/{len(spellings)}"
    emit(f"eval posts on disk: {on_disk} (under {opts.blog_root})")
    for canonical, paths in spellings.items():
        mark = "MISSING" if canonical in missing else "ok"
        others = [p for p in paths if p != canonical]
        alias_note = f"  [also: {', '.join(others)}]" if others else ""
        emit(f"  {mark:>7}  {canonical}{alias_note}")
    if pairs_path is not None:
        ds = ct.assemble_dataset(pairs_path)
        emit("\n".join(role_lines(splits_doc, ds, top=True)))
        report = check_splits(splits_doc, ds, blog_root=opts.blog_root)
    else:
        report = [
            f"{ERROR_PREFIX} pinned eval post {canonical!r} is not on disk under {opts.blog_root}"
            for canonical in sorted(missing)
        ]
    errors = [line for line in report if is_error(line)]
    for line in report:
        err = is_error(line)
        emit(line if err else f"WARNING: {line}", level="error" if err else "warn")
    if errors:
        raise RunError(f"{len(errors)} missing source(s) under {opts.blog_root}")
