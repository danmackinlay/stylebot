"""What the pipeline actually holds right now — the answer docs used to carry.

Every prose description of this system had a shelf life. stylebot's `CLAUDE.md`
said two trained runs existed and run 2 was serving; the blog's manifests said
five and run 4a. `OVERVIEW.md` said 4,595 pairs; the corpus said 8,182. None of
those sentences were wrong when they were written, and git cannot keep any of
them true — the facts they state live in the blog's `_training_pairs/`, which is
a different repo on a different commit clock, and the numbers change on every
capture, synth run and promotion.

So the documentation stops carrying them and points here instead. `collect` is a
read-only sweep over the five artifacts that answer "where is this pipeline?":

1. the **corpus** — how many records, how they split real/synthetic, how they
   are sharded (`stylebot.corpus`);
2. the **splits contract** — eval stratum version and `splits_sha256`, whether
   every pinned eval post is still on disk, the per-role table and the danger
   report (`stylebot.splits`);
3. the **runs** — which manifests exist, which are completed, and which one
   serves by default (the `promoted` rule, below);
4. the **detector** — the artifact's `git_sha`, when it was fit, and on how
   many pairs (`stylebot.classify.load_artifact_meta`);
5. the **cross-checks between them**, which is the part no single artifact can
   answer and the reason this is one function rather than four commands. A
   detector fit under eval_v1 and a `splits.json` now at v2 are each internally
   consistent and jointly wrong: every verdict computed with that head held out
   the wrong posts. Same for a promoted run whose `splits_sha256` no longer
   matches. Those comparisons are the `warnings` list.

**Everything is optional and nothing is discovered.** stylebot does not know
where the blog is (`_plans/OVERVIEW.md`, "the dependency arrow points one way"),
so `collect` reports on exactly the paths it is handed and says "not given" for
the rest — `ai-style status` takes them as flags, `dan-style status` fills them
from `livingthing.training_targets`. A missing artifact is a reported absence,
never an exception: status must run on a half-built pipeline, which is when it
is most useful.

Read-only and keyless. It costs a corpus read (a few seconds on 8k records)
and nothing else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from stylebot import corpus as corpus_mod
from stylebot import splits as splits_mod


def promoted_manifest(runs_dir: str | Path) -> dict | None:
    """The newest completed, non-nixed run manifest under `runs_dir`, or None.

    A manifest is *completed* when it carries a `result` block and *promoted*
    unless it says `"promoted": false`; the default-serving run is the newest
    that is both. The rule lives here rather than in either CLI so `status` and
    serving cannot disagree about which adapter is live — the blog's
    `style_file.latest_manifest` is this function plus a click error.

    Manifests are named by run id, which is a timestamp, so a reverse filename
    sort is newest-first. `"promoted": false` skips a run without hiding it: run
    3 was nixed after an eval loss and run 4b is an ablation arm that was never
    meant to serve, and both stay on record with their `promotion_note`.

    Returns the manifest with `_path` added. None when no manifest qualifies —
    an unbuilt pipeline, not an error.
    """
    runs = Path(runs_dir)
    if not runs.is_dir():
        return None
    for path in sorted(runs.glob("*.json"), reverse=True):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        if manifest.get("result") and manifest.get("promoted") is not False:
            manifest["_path"] = str(path)
            return manifest
    return None


@dataclass(frozen=True)
class CorpusStatus:
    path: Path
    exists: bool
    sharded: bool
    shards: list[tuple[str, int]] = field(default_factory=list)
    n_records: int = 0
    n_real: int = 0
    n_synthetic: int = 0
    n_posts: int = 0
    content_sha256: str | None = None


@dataclass(frozen=True)
class SplitsStatus:
    path: Path
    exists: bool
    eval_version: int | None = None
    splits_sha256: str | None = None
    eval_posts: int = 0
    eval_posts_missing: list[str] = field(default_factory=list)
    roles: dict[str, dict] = field(default_factory=dict)
    report: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RunsStatus:
    path: Path | None
    exists: bool
    n_manifests: int = 0
    n_completed: int = 0
    promoted: dict | None = None


@dataclass(frozen=True)
class DetectorStatus:
    path: Path | None
    exists: bool
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Status:
    generated_at: str
    stylebot_sha: str | None
    corpus: CorpusStatus
    splits: SplitsStatus
    runs: RunsStatus
    detector: DetectorStatus
    warnings: list[str] = field(default_factory=list)


def _corpus_status(path: Path | None) -> CorpusStatus:
    if path is None:
        return CorpusStatus(path=Path(), exists=False, sharded=False)
    if not path.exists():
        return CorpusStatus(path=path, exists=False, sharded=False)
    shards: list[tuple[str, int]] = []
    n_records = n_real = 0
    posts: set[str] = set()
    for shard in corpus_mod.shard_paths(path):
        count = 0
        for rec in corpus_mod.iter_records(shard):
            count += 1
            meta = rec.get("meta") or {}
            if not meta.get("synthetic"):
                n_real += 1
            source = meta.get("source")
            if isinstance(source, str):
                posts.add(source)
        shards.append((shard.name, count))
        n_records += count
    return CorpusStatus(
        path=path,
        exists=True,
        sharded=path.is_dir(),
        shards=shards,
        n_records=n_records,
        n_real=n_real,
        n_synthetic=n_records - n_real,
        n_posts=len(posts),
        content_sha256=corpus_mod.content_sha256(path),
    )


def _splits_status(
    path: Path | None,
    *,
    corpus_path: Path | None,
    blog_root: Path | None,
) -> SplitsStatus:
    if path is None or not path.exists():
        return SplitsStatus(path=path or Path(), exists=False)
    doc = splits_mod.load_splits(path)
    spellings = splits_mod.eval_post_spellings(doc)
    missing = splits_mod.missing_eval_posts(doc, blog_root) if blog_root is not None else []
    roles: dict[str, dict] = {}
    report: list[str] = []
    if corpus_path is not None and corpus_path.exists():
        # `assemble_dataset` is the same reader `check-splits` and `train-clf`
        # use, so the per-role counts here are the ones those commands act on
        # — including its near-identity gate, which is why the eval stratum
        # reads slightly smaller than a raw grep of the corpus would.
        from stylebot import classify_train as ct

        ds = ct.assemble_dataset(corpus_path)
        roles = splits_mod.summarize_roles(doc, ds)
        report = splits_mod.check_splits(doc, ds, blog_root=blog_root)
    elif blog_root is not None:
        report = [
            f"{splits_mod.ERROR_PREFIX} pinned eval post {canonical!r} is not on disk "
            f"under {blog_root}"
            for canonical in missing
        ]
    return SplitsStatus(
        path=path,
        exists=True,
        eval_version=splits_mod.eval_version(doc),
        splits_sha256=splits_mod.splits_sha256(path),
        eval_posts=len(spellings),
        eval_posts_missing=list(missing),
        roles=roles,
        report=report,
    )


def _runs_status(path: Path | None) -> RunsStatus:
    if path is None or not path.is_dir():
        return RunsStatus(path=path, exists=False)
    manifests = sorted(path.glob("*.json"))
    completed = 0
    for candidate in manifests:
        try:
            doc = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict) and doc.get("result"):
            completed += 1
    return RunsStatus(
        path=path,
        exists=True,
        n_manifests=len(manifests),
        n_completed=completed,
        promoted=promoted_manifest(path),
    )


def _detector_status(path: Path | None) -> DetectorStatus:
    if path is None or not Path(path).is_dir():
        return DetectorStatus(path=path, exists=False)
    from stylebot.classify import load_artifact_meta

    try:
        meta = load_artifact_meta(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return DetectorStatus(path=path, exists=False)
    return DetectorStatus(path=Path(path), exists=True, meta=meta)


def _cross_checks(splits: SplitsStatus, runs: RunsStatus, detector: DetectorStatus) -> list[str]:
    """The comparisons no single artifact can make. See the module docstring."""
    out: list[str] = []
    live_sha = splits.splits_sha256
    live_version = splits.eval_version

    det_sha = detector.meta.get("splits_sha256")
    det_version = (detector.meta.get("split") or {}).get("eval_version")
    if detector.exists and live_sha and det_sha and det_sha != live_sha:
        out.append(
            f"detector was fit under splits_sha256={det_sha} "
            f"(eval stratum v{det_version}) but splits.json is now {live_sha} "
            f"(v{live_version}) — refit it (`train-clf`); every verdict computed "
            f"with this head held out a different set of posts"
        )

    promoted = runs.promoted or {}
    data = promoted.get("data") or {}
    run_sha = data.get("splits_sha256")
    if promoted and live_sha and run_sha and run_sha != live_sha:
        out.append(
            f"promoted run {promoted.get('run_id')} trained under "
            f"splits_sha256={run_sha} but splits.json is now {live_sha} — prose it "
            f"trained on may now be held out"
        )
    if promoted and run_sha is None:
        out.append(
            f"promoted run {promoted.get('run_id')} predates the splits/dependency "
            f"manifest pins (D42) — which partition it held out is not recorded"
        )
    if runs.exists and not promoted:
        out.append(f"no completed, promoted run manifest under {runs.path}")
    for line in splits.report:
        out.append(line)
    return out


def collect(
    *,
    pairs: str | Path | None = None,
    splits: str | Path | None = None,
    runs_dir: str | Path | None = None,
    detector: str | Path | None = None,
    blog_root: str | Path | None = None,
) -> Status:
    """Sweep whichever of the five artifacts the caller names. Read-only, keyless.

    Every argument is optional and none is discovered: what is not passed is
    reported as "not given", never guessed at. A path that does not exist is
    reported as absent rather than raised on — status is most useful on a
    pipeline that is half built.
    """
    from stylebot.provenance import now_iso, stylebot_sha

    pairs_path = Path(pairs) if pairs is not None else None
    splits_path = Path(splits) if splits is not None else None
    root = Path(blog_root) if blog_root is not None else None

    corpus_status = _corpus_status(pairs_path)
    splits_status = _splits_status(splits_path, corpus_path=pairs_path, blog_root=root)
    runs_status = _runs_status(Path(runs_dir) if runs_dir is not None else None)
    detector_status = _detector_status(Path(detector) if detector is not None else None)
    return Status(
        generated_at=now_iso(),
        stylebot_sha=stylebot_sha(),
        corpus=corpus_status,
        splits=splits_status,
        runs=runs_status,
        detector=detector_status,
        warnings=_cross_checks(splits_status, runs_status, detector_status),
    )


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _ellipsis(text: str, width: int) -> str:
    """Trim to `width` on a word boundary — a promotion note is a paragraph."""
    if len(text) <= width:
        return text
    head = text[:width].rsplit(" ", 1)[0]
    return f"{head} ..."


def format_status(status: Status) -> str:
    """Render a `Status` as the plain-text block both CLIs print."""
    lines: list[str] = []
    sha = status.stylebot_sha or "unknown"
    lines.append(f"stylebot {sha} · {status.generated_at}")

    c = status.corpus
    lines.append("")
    if not c.exists:
        lines.append(f"corpus     {c.path or '(not given)'} — absent")
    else:
        shape = "shard directory" if c.sharded else "single file"
        lines.append(f"corpus     {c.path} ({shape})")
        lines.append(
            f"           {_fmt_int(c.n_records)} records — {_fmt_int(c.n_real)} real / "
            f"{_fmt_int(c.n_synthetic)} synthetic, over {_fmt_int(c.n_posts)} source posts"
        )
        for name, count in c.shards:
            lines.append(f"             {name:<24} {_fmt_int(count):>8}")
        lines.append(f"           content_sha256={c.content_sha256}")

    s = status.splits
    lines.append("")
    if not s.exists:
        lines.append(f"splits     {s.path or '(not given)'} — absent")
    else:
        present = s.eval_posts - len(s.eval_posts_missing)
        lines.append(
            f"splits     {s.path} — eval stratum v{s.eval_version}, splits_sha256={s.splits_sha256}"
        )
        lines.append(f"           pinned eval posts on disk: {present}/{s.eval_posts}")
        for role in splits_mod.ROLES:
            entry = s.roles.get(role)
            if not entry:
                continue
            lines.append(
                f"           {role:>8}: {entry['posts']} post(s), {entry['pairs']} pair(s) "
                f"({entry['pairs_real']} real / {entry['pairs_synthetic']} synthetic)"
            )

    r = status.runs
    lines.append("")
    if not r.exists:
        lines.append(f"runs       {r.path or '(not given)'} — absent")
    else:
        lines.append(
            f"runs       {r.path} — {r.n_manifests} manifest(s), {r.n_completed} completed"
        )
        if r.promoted:
            hp = r.promoted.get("hyperparameters") or {}
            data = r.promoted.get("data") or {}
            lines.append(
                f"           serving:  {r.promoted.get('run_id')}  "
                f"({hp.get('base_model')}, lora r{hp.get('lora_rank')})"
            )
            note = (r.promoted.get("promotion_note") or "").strip().splitlines()
            if note:
                lines.append(f"           {_ellipsis(note[0], 92)}")
            scale = data.get("chunk_scale") or {}
            if scale.get("budgets"):
                budgets = ", ".join(f"{k}:{v}" for k, v in sorted(scale["budgets"].items()))
                lines.append(f"           diet:     {budgets}")
        else:
            lines.append("           serving:  (none promoted)")

    d = status.detector
    lines.append("")
    if not d.exists:
        lines.append(f"detector   {d.path or '(not given)'} — absent")
    else:
        m = d.meta
        lines.append(f"detector   {d.path} — {m.get('name')} on {m.get('embed_model')}")
        lines.append(
            f"           git_sha={m.get('git_sha')}  trained_at={m.get('trained_at')}  "
            f"n_pairs={m.get('n_pairs')}"
        )
        real = ((m.get("metrics") or {}).get("by_provenance") or {}).get("real") or {}
        acc = (real.get("pairwise_accuracy") or {}).get("mean")
        if acc is not None:
            lines.append(f"           real-facet pairwise accuracy: {acc}")

    lines.append("")
    if status.warnings:
        for line in status.warnings:
            lines.append(f"! {line}")
    else:
        lines.append("no warnings")
    return "\n".join(lines)
