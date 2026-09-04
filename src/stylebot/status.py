"""What the pipeline actually holds right now — the answer docs used to carry.

Prose cannot keep operational state true, so the documentation stops carrying it
and points here (D52). `collect` is a read-only sweep over the five artifacts
that answer "where is this pipeline?":

1. the **corpus** — how many records, how they split real/synthetic, how they
   are sharded (`stylebot.corpus`);
2. the **splits contract** — eval stratum version and `splits_sha256`, whether
   every pinned eval post is still on disk, the per-role table and the danger
   report (`stylebot.splits`);
3. the **runs** — every manifest with the artifacts it pinned, which are
   completed, and which one serves (`promoted: true`, written by `promote` —
   the one thing this module writes; the rule is `promoted_manifest`);
4. the **detector** — the artifact's `git_sha`, when it was fit, and on how
   many pairs (`stylebot.classify.load_artifact_meta`);
5. the **cross-checks between them**, which is why this is one function rather
   than four commands. A detector fit under eval_v1 and a `splits.json` now at
   v2 are each internally consistent and jointly wrong; same for a promoted run
   whose `splits_sha256` no longer matches. Those are the `warnings` list.

**Everything is optional and nothing is discovered.** stylebot does not know
where the blog is, so `collect` reports on exactly the paths it is handed and
says "not given" for the rest — `ai-style status` takes them as flags,
`dan-style status` fills them from `livingthing.training_targets`. A missing
artifact is a reported absence, never an exception: status must run on a
half-built pipeline, which is when it is most useful. Read-only and keyless.
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
    """When a manifest says it was made, as a POSIX timestamp; 0.0 if it doesn't.

    `created_at` (a training run) or `produced_at` (a verdict, which records a
    stamp per phase and takes the earliest). Parsed rather than compared as a
    string, because the two shapes carry different UTC offsets.
    """
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
    """Every readable `runs/*.json`, newest first. Unreadable ones are skipped.

    Ordered by the manifest's own clock (`_run_time`) with the filename as the
    tiebreak, so a run id that is not a timestamp — a verdict directory's
    manifest — sorts by when it says it was made rather than by where its stem
    falls in a filename sort (D74).
    """
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
    """The newest completed run manifest marked `promoted: true`, or None.

    A manifest is *completed* when it carries a `result` block; it *serves*
    only when `promote` has said so (D74 as amended, W4 step 4) — a completed
    run with no `promoted` key is on record and not serving, and `"promoted":
    false` is a nixed run, an ablation arm or a verdict manifest, all listed by
    `run_entries`. The rule lives here rather than in either CLI so `status`
    and serving cannot disagree: the blog's `styler.latest_manifest` is this
    function plus an error.

    Returns the manifest with `_path` added. None when no manifest qualifies —
    an unbuilt or never-promoted pipeline, not an error.
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


def _run_pins(doc: dict) -> dict[str, str]:
    """The artifacts a manifest pinned, in either shape it is written in.

    A training run records them under `data` and `hyperparameters`; a verdict
    records them under `pins`, and names the styler run it sampled from. Absent
    pins are absent keys, so an old manifest reads as short rather than wrong.
    """
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
    """One row per manifest, newest first — F6's "list the run with its pins"."""
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
