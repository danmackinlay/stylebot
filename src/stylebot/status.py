"""What the pipeline holds right now — the answer docs may not carry (D52).

`collect` is a read-only, keyless sweep over the corpus (or the view a draws
dir would build, built or not), the splits contract, the runs (which is
completed, which serves — `promote` writes `promoted: true`, the one write
here, D74), the detector artifact, and the cross-checks between them, which is
why it is one function: a detector fit under one `splits_sha256` and a
`splits.json` now at another are each consistent and jointly wrong. Nothing is
discovered; a path not passed is "not given", a missing artifact a reported
absence (D52).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from stylebot import corpus as corpus_mod
from stylebot import splits as splits_mod
from stylebot.errors import ConfigError


def _run_time(doc: dict) -> float:
    """When a manifest says it was made (`created_at`, or a verdict's earliest
    `produced_at`), as a POSIX timestamp; 0.0 if it doesn't."""
    stamps: list[str] = []
    for value in (doc.get("created_at"), doc.get("produced_at")):
        if isinstance(value, str):
            stamps.append(value)
        elif isinstance(value, dict):
            stamps += [v for v in value.values() if isinstance(v, str)]
    for stamp in sorted(stamps):
        try:
            return datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            continue
    return 0.0


def load_manifests(runs_dir: str | Path) -> list[tuple[Path, dict]]:
    """Every readable `runs/*.json`, newest first by the manifest's own clock
    (`_run_time`, filename as tiebreak; D74). Unreadable ones are skipped."""
    runs = Path(runs_dir)
    if not runs.is_dir():
        return []
    found: list[tuple[Path, dict]] = []
    for path in sorted(runs.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict):
            found.append((path, doc))
    return sorted(found, key=lambda pd: (_run_time(pd[1]), pd[0].name), reverse=True)


def promoted_manifest(runs_dir: str | Path) -> dict | None:
    """The newest completed run manifest marked `promoted: true`, or None (D74).

    Completed means it carries a `result` block; serving means `promote` said
    so. The one rule for `status` and serving alike. Returns the manifest with
    `_path` added; None is an unbuilt or never-promoted pipeline, not an error.
    """
    for path, manifest in load_manifests(runs_dir):
        if manifest.get("result") and manifest.get("promoted") is True:
            manifest["_path"] = str(path)
            return manifest
    return None


def _write_manifest(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def promote(runs_dir: str | Path, run_id: str) -> Path:
    """Make `run_id` the one serving run: `promoted: true` on its manifest,
    `false` on every other completed one. An incomplete manifest (no `result`)
    is neither promotable nor touched. Returns the promoted manifest's path."""
    manifests = load_manifests(runs_dir)
    chosen = [(p, d) for p, d in manifests if str(d.get("run_id") or p.stem) == run_id]
    if not chosen:
        raise ConfigError(f"no manifest for run {run_id!r} under {runs_dir}")
    path, doc = chosen[0]
    if not doc.get("result"):
        raise ConfigError(f"run {run_id!r} has no result; only a completed run can serve")
    for other, odoc in manifests:
        if other != path and odoc.get("result") and odoc.get("promoted") is not False:
            _write_manifest(other, odoc | {"promoted": False})
    _write_manifest(path, doc | {"promoted": True})
    return path


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
class ViewStatus:
    """The view of the draws at `path` under a policy, and whether it is built."""

    path: Path
    exists: bool
    policy_sha: str | None = None
    draws_sha256: str | None = None
    records: int = 0
    by_outcome: dict[str, int] = field(default_factory=dict)
    excluded: dict[str, int] = field(default_factory=dict)
    content_sha256: str | None = None
    view_path: Path | None = None
    built: bool = False
    drifted: list[tuple[str, int]] = field(default_factory=list)


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
class RunEntry:
    """One manifest under `runs/`: what it is, and the artifacts it pinned."""

    run_id: str
    kind: str
    state: str
    pins: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunsStatus:
    path: Path | None
    exists: bool
    n_manifests: int = 0
    n_completed: int = 0
    promoted: dict | None = None
    entries: list[RunEntry] = field(default_factory=list)


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
    view: ViewStatus | None = None


def _view_status(path: Path | None, policy, views_root: Path | None) -> ViewStatus | None:
    """The view `path`'s draws render under `policy` (a `draws.ViewPolicy`, identity
    when None), and whether `views_root` (default the draws' sibling `views/`)
    holds it built; None when no draws dir was named, absent when missing."""
    if path is None:
        return None
    if not path.is_dir():
        return ViewStatus(path=path, exists=False)
    from stylebot import capture, config
    from stylebot import draws as draws_mod

    policy = policy or draws_mod.ViewPolicy()
    report = draws_mod.build_view(draws_mod.iter_draws(path), policy)
    draws_sha = draws_mod.content_sha256(path)
    live = draws_mod.live_draws(draws_mod.iter_draws(path), policy)
    drifted = capture.drifted_archives(
        path.parent / "captures", (d.record for d in live if d.record is not None)
    )
    view_path = (views_root or config.views_root(path.parent)) / config.view_name(report.policy_sha)
    side = draws_mod.read_sidecar(view_path) or {}
    return ViewStatus(
        path=path,
        exists=True,
        policy_sha=report.policy_sha,
        draws_sha256=draws_sha,
        records=report.records,
        by_outcome=dict(report.by_outcome),
        excluded=dict(report.excluded),
        content_sha256=report.content_sha256,
        view_path=view_path,
        built=side.get("policy_sha") == report.policy_sha and side.get("draws_sha256") == draws_sha,
        drifted=drifted,
    )


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
        # The same reader `check-splits` and `train-clf` use, near-identity gate
        # included, so the per-role counts match theirs.
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


def _run_pins(doc: dict) -> dict[str, str]:
    """The artifacts a manifest pinned, in either shape: a training run's
    `data`/`hyperparameters`, or a verdict's `pins`. Absent pins are absent keys."""
    pins = doc.get("pins") if isinstance(doc.get("pins"), dict) else {}
    data = doc.get("data") if isinstance(doc.get("data"), dict) else {}
    hyper = doc.get("hyperparameters") if isinstance(doc.get("hyperparameters"), dict) else {}

    def pin(block: str, key: str):
        sub = pins.get(block)
        return sub.get(key) if isinstance(sub, dict) else None

    lora = f"/r{hyper['lora_rank']}" if hyper.get("lora_rank") else ""
    found = {
        "styler": pin("styler", "run_id"),
        "model": f"{hyper['base_model']}{lora}" if hyper.get("base_model") else None,
        "corpus": pin("corpus", "content_sha256") or data.get("pairs_sha256"),
        "splits": pin("splits", "splits_sha256") or data.get("splits_sha256"),
        "detector": pin("detector", "git_sha"),
    }
    return {key: str(value) for key, value in found.items() if value}


def _run_entries(manifests: list[tuple[Path, dict]], promoted: dict | None) -> list[RunEntry]:
    """One row per manifest, newest first, with its pins (D74)."""
    serving = (promoted or {}).get("_path")
    entries = []
    for path, doc in manifests:
        if doc.get("promoted") is False:
            state = "not promoted"
        elif not doc.get("result"):
            state = "incomplete"
        else:
            state = "serving" if str(path) == serving else "completed"
        entries.append(
            RunEntry(
                run_id=str(doc.get("run_id") or path.stem),
                kind=str(doc.get("kind") or "?"),
                state=state,
                pins=_run_pins(doc),
            )
        )
    return entries


def _runs_status(path: Path | None) -> RunsStatus:
    if path is None or not path.is_dir():
        return RunsStatus(path=path, exists=False)
    manifests = load_manifests(path)
    promoted = promoted_manifest(path)
    return RunsStatus(
        path=path,
        exists=True,
        n_manifests=len(manifests),
        n_completed=sum(1 for _p, doc in manifests if doc.get("result")),
        promoted=promoted,
        entries=_run_entries(manifests, promoted),
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
        out.append(f"no promoted run manifest under {runs.path} — `promote RUN_ID` names one")
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
    draws_dir: str | Path | None = None,
    policy=None,
    views_root: str | Path | None = None,
) -> Status:
    """Sweep whichever of the five artifacts the caller names; read-only,
    keyless, nothing discovered. `draws_dir` (+ `policy`, `views_root`) adds
    the view block, which stands in for the corpus block when the draws are there."""
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
        view=_view_status(
            Path(draws_dir) if draws_dir is not None else None,
            policy,
            Path(views_root) if views_root is not None else None,
        ),
    )


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _ellipsis(text: str, width: int) -> str:
    """Trim to `width` on a word boundary — a promotion note is a paragraph."""
    if len(text) <= width:
        return text
    head = text[:width].rsplit(" ", 1)[0]
    return f"{head} ..."


def _format_runs(r: RunsStatus) -> list[str]:
    """The runs block: the counts, the serving run, then every manifest listed."""
    if not r.exists:
        return [f"runs       {r.path or '(not given)'} — absent"]
    lines = [f"runs       {r.path} — {r.n_manifests} manifest(s), {r.n_completed} completed"]
    if r.promoted:
        hp = r.promoted.get("hyperparameters") or {}
        lines.append(
            f"           serving:  {r.promoted.get('run_id')}  "
            f"({hp.get('base_model')}, lora r{hp.get('lora_rank')})"
        )
        note = (r.promoted.get("promotion_note") or "").strip().splitlines()
        if note:
            lines.append(f"           {_ellipsis(note[0], 92)}")
        scale = (r.promoted.get("data") or {}).get("chunk_scale") or {}
        if scale.get("budgets"):
            budgets = ", ".join(f"{k}:{v}" for k, v in sorted(scale["budgets"].items()))
            lines.append(f"           diet:     {budgets}")
    else:
        lines.append("           serving:  none (run `promote`)")
    for entry in r.entries:
        lines.append(f"           {entry.run_id:<26} {entry.kind:<22} {entry.state}")
        if entry.pins:
            pins = "  ".join(f"{k}={v}" for k, v in entry.pins.items())
            lines.append(f"             {pins}")
    return lines


def _format_corpus(c: CorpusStatus) -> list[str]:
    if not c.exists:
        return [f"corpus     {c.path or '(not given)'} — absent"]
    shape = "shard directory" if c.sharded else "single file"
    lines = [
        f"corpus     {c.path} ({shape})",
        f"           {_fmt_int(c.n_records)} records — {_fmt_int(c.n_real)} real / "
        f"{_fmt_int(c.n_synthetic)} synthetic, over {_fmt_int(c.n_posts)} source posts",
    ]
    for name, count in c.shards:
        lines.append(f"             {name:<24} {_fmt_int(count):>8}")
    lines.append(f"           content_sha256={c.content_sha256}")
    return lines


def _format_view(v: ViewStatus) -> list[str]:
    """The view block: what the draws render under the policy, and whether it is built."""
    if not v.exists:
        return [f"view       {v.path} — absent (no draws directory)"]
    outcomes = ", ".join(f"{k} {_fmt_int(n)}" for k, n in sorted(v.by_outcome.items()))
    excluded = ", ".join(f"{k} {_fmt_int(n)}" for k, n in sorted(v.excluded.items()))
    on_disk = f"built {v.view_path}" if v.built else "unbuilt"
    lines = [
        f"view       {v.path} -> pairs-{v.policy_sha} {on_disk}",
        f"           {_fmt_int(v.records)} record(s) would render; draws by outcome: {outcomes}",
        f"           excluded: {excluded or 'none'}",
        f"           policy_sha={v.policy_sha}  draws_sha256={v.draws_sha256}",
        f"           content_sha256={v.content_sha256}",
    ]
    if v.drifted:
        names = ", ".join(f"{aid} ({n})" for aid, n in v.drifted[:8])
        more = " …" if len(v.drifted) > 8 else ""
        lines.append(
            f"           archives edited since landing: {len(v.drifted)} — {names}{more}; "
            "`ai-style log recut ID` re-lands"
        )
    return lines


def format_status(status: Status) -> str:
    """Render a `Status` as the plain-text block both CLIs print."""
    lines: list[str] = []
    sha = status.stylebot_sha or "unknown"
    lines.append(f"stylebot {sha} · {status.generated_at}")

    lines.append("")
    if status.view is not None:
        lines.extend(_format_view(status.view))
    if status.view is None or not status.view.exists:
        lines.extend(_format_corpus(status.corpus))

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

    lines.append("")
    lines.extend(_format_runs(status.runs))

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
