"""Phase 3: LoRA SFT of the styler — assembly, manifest, and the Tinker run.

`assemble_training_corpus` is keyless (the record contract, the STYLER-role
filter under `stylebot.splits`, the caller's policy hooks, a by-POST val
split); `run_training` is the paid Tinker path; `run_training_pass` is the
`train` surface both CLIs render over `TrainOpts` (D78). Every `tinker`
import is lazy, so the module imports without the ``trainer`` extra.

Data policy, pinned in D17/D18/D43 — do not "fix": near-copy pairs are KEPT
(label noise for the detector, good styler data); records train exactly as
stored, `STYLE_SYSTEM` verbatim as `messages[0]`; per-passage multiplicity is
the `per_passage` hook over a TRUE grouping (`passage_groups`); weight is
integer replication of the TRAIN file only. The manifest is the committed
reproducibility record; weights and corpus never enter git.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

import click

from stylebot import config, draws
from stylebot.ai_core import style_system
from stylebot.cli_options import Emitter, bundle_options, opt, silent
from stylebot.corpus import content_sha256
from stylebot.errors import ConfigError, MissingExtra, RunError
from stylebot.jsonl import dumps_line
from stylebot.pairs import iter_pairs, system_sha, validate_pairs_file
from stylebot.provenance import file_sha256, git_sha  # file_sha256 re-exported: public here
from stylebot.status import promote as promote  # re-exported: the run's other verb

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Defaults follow the Tinker cookbook's supervised recipe, over a current-gen
# small dense base model.
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_LORA_RANK = 32
DEFAULT_NUM_EPOCHS = 1
DEFAULT_BATCH_SIZE = 64
DEFAULT_VAL_FRAC = 0.1
DEFAULT_MAX_LENGTH = 8192
LR_SCHEDULES = ("linear", "cosine", "constant")

#: The dry-run token estimate's basis (`estimate_tokens`).
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Hyperparameters:
    """The LoRA-SFT knobs — one declaration, every consumer.

    `as_manifest()` is the manifest's `hyperparameters` block, so no knob can
    be set but unrecorded; each field is also its click flag (`cli_options.opt`),
    so both `train` CLIs render one surface and re-pin it with `defaults=`.
    """

    cli_section = "Hyperparameters"

    base_model: str = opt(
        DEFAULT_BASE_MODEL, "--base-model", show_default=True, help="Tinker base model id."
    )
    lora_rank: int = opt(
        DEFAULT_LORA_RANK, "--lora-rank", show_default=True, type=int, help="LoRA rank."
    )
    learning_rate: float | None = opt(
        None,
        "--lr",
        "learning_rate",
        type=float,
        help="Learning rate. Default: the cookbook's per-model recommendation "
        "(hyperparam_utils.get_lr).",
    )
    lr_schedule: str = opt(
        "linear",
        "--lr-schedule",
        show_default=True,
        type=click.Choice(LR_SCHEDULES),
        help="Learning-rate decay shape.",
    )
    num_epochs: int = opt(
        DEFAULT_NUM_EPOCHS,
        "--epochs",
        "num_epochs",
        show_default=True,
        type=int,
        help="Training epochs.",
    )
    batch_size: int = opt(
        DEFAULT_BATCH_SIZE,
        "--batch-size",
        show_default=True,
        type=int,
        help="Examples per training batch.",
    )
    max_length: int = opt(
        DEFAULT_MAX_LENGTH,
        "--max-length",
        show_default=True,
        type=int,
        help="Max sequence length after tokenisation.",
    )
    renderer_name: str | None = opt(
        None,
        "--renderer",
        "renderer_name",
        help="Cookbook renderer name (chat template). Default: the model's recommended renderer — "
        "pass e.g. qwen3_5_disable_thinking to pin no-think.",
    )
    seed: int = opt(
        0, "--seed", show_default=True, type=int, help="Seed for the val split and epoch shuffling."
    )

    def as_manifest(self) -> dict:
        """This bundle as the manifest's `hyperparameters` block."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


_EXTRA_HINT = (
    "stylebot.train needs the 'trainer' extra: run `uv add 'stylebot[trainer]'` (or `pip "
    "install 'stylebot[trainer]'`) for tinker + tinker-cookbook; --dry-run works without it."
)


@dataclass
class TrainCorpus:
    """The assembled styler corpus: full pair records, split by POST
    (`meta.source`) so no passage of a val post appears in training."""

    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    train_posts: list[str] = field(default_factory=list)
    val_posts: list[str] = field(default_factory=list)
    pairs_sha256: str | None = None  # 16-hex content hash of the corpus (file, or shards)
    n_source_records: int = 0  # records in pairs.jsonl before any filter
    dropped: dict[str, int] = field(default_factory=dict)  # per-filter counts

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_val(self) -> int:
        return len(self.val)

    def _count(self, synthetic: bool) -> int:
        return sum(
            1
            for rec in [*self.train, *self.val]
            if bool((rec.get("meta") or {}).get("synthetic")) is synthetic
        )

    @property
    def n_real(self) -> int:
        return self._count(False)

    @property
    def n_synthetic(self) -> int:
        return self._count(True)

    @property
    def chunk_scale(self) -> dict:
        """The chunk diet this run trained on, for serving to follow
        (`backends.serving_diet`): `budgets` (pairs per `meta.merge_max_chars`,
        "None" = unmerged), `modes` (per `meta.chunk_mode`), `max_target_chars`,
        and the multiplicity question — `passages`, `pair_weights` (D43)."""
        budgets: dict[str, int] = {}
        modes: dict[str, int] = {}
        longest = 0
        for rec in [*self.train, *self.val]:
            meta = rec.get("meta") or {}
            budgets[str(meta.get("merge_max_chars"))] = (
                budgets.get(str(meta.get("merge_max_chars")), 0) + 1
            )
            modes[str(meta.get("chunk_mode"))] = modes.get(str(meta.get("chunk_mode")), 0) + 1
            msgs = rec.get("messages") or []
            if len(msgs) == 3:
                longest = max(longest, len(msgs[2].get("content") or ""))
        return {
            "budgets": budgets,
            "modes": modes,
            "max_target_chars": longest,
            "passages": self.passage_scale,
            "pair_weights": _histogram(pair_weight_of(rec) for rec in self.train),
            "n_train_examples": self.n_train_examples,
        }

    @property
    def passage_scale(self) -> dict:
        """Per-passage multiplicity per split: `pairs_per_passage` is how many
        gradient steps one passage buys under a one-epoch manifest (D43). Both
        splits inflate alike, so val loss cannot see it."""
        out: dict = {}
        for name, records in (("train", self.train), ("val", self.val)):
            groups = passage_groups(records)
            hist = _histogram(len(g) for g in groups)
            out[name] = {
                "n_passages": len(groups),
                "pairs_per_passage": hist,
                "max_pairs_per_passage": max((len(g) for g in groups), default=0),
                "mean_pairs_per_passage": (round(len(records) / len(groups), 2) if groups else 0.0),
            }
        return out

    @property
    def n_train_examples(self) -> int:
        """Rows the trainer actually sees — `n_train` after weight replication."""
        return sum(pair_weight_of(rec) for rec in self.train)


def _hook_name(hook: Callable | None) -> str | None:
    """A policy hook's name for the manifest — `None` when unset."""
    return getattr(hook, "__name__", repr(hook)) if hook else None


def _histogram(values) -> dict[str, int]:
    """Counter as a JSON-safe, numerically sorted `{str(value): count}`."""
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {str(k): counts[k] for k in sorted(counts)}


def _target_body(rec: dict) -> str:
    """The assistant-side body, heading-context stripped (same recovery as
    `synth`'s coverage check)."""
    from stylebot.pairs import extract_target

    return extract_target(rec).strip()


def pair_weight_of(rec: dict) -> int:
    """A record's training weight: `meta.weight`, else 1 — a junk value reads
    as 1 so a paid run's assembly never fails on an optimisation knob."""
    raw = (rec.get("meta") or {}).get("weight")
    try:
        weight = int(raw)
    except (TypeError, ValueError):
        return 1
    return weight if weight > 0 else 1


def passage_groups(records: Sequence[dict]) -> list[list[dict]]:
    """Group pair records by SOURCE POST and target-body containment.

    A *passage* is a maximal stretch of prose: a record whose target body is a
    substring of a longer target from the same post joins that target's group
    (a 1500-char cut of some prose IS a substring of the 3000-char cut; exact
    equality saw two unrelated targets — the measurement is D43). Groups and
    their records keep corpus order, so the multiplicity hook sees a
    deterministic list. Never merges across posts, so the O(spans²) search
    stays per-post.
    """
    by_source: dict[str, dict[str, list[dict]]] = {}
    order: list[str] = []
    for rec in records:
        source = (rec.get("meta") or {}).get("source") or "?"
        if source not in by_source:
            by_source[source] = {}
            order.append(source)
        by_source[source].setdefault(_target_body(rec), []).append(rec)

    groups: list[list[dict]] = []
    for source in order:
        bodies = by_source[source]
        # Longest first, so a root is always seen before anything nested in it.
        roots: list[str] = []
        root_of: dict[str, str] = {}
        for body in sorted(bodies, key=len, reverse=True):
            parent = next((r for r in roots if body in r), None)
            if parent is None:
                roots.append(body)
                parent = body
            root_of[body] = parent
        merged: dict[str, list[dict]] = {}
        appearance: list[str] = []
        for body, recs in bodies.items():  # insertion order == corpus order
            root = root_of[body]
            if root not in merged:
                merged[root] = []
                appearance.append(root)
            merged[root].extend(recs)
        groups.extend(merged[root] for root in appearance)
    return groups


def assemble_training_corpus(
    pairs_path: str | Path,
    *,
    splits: dict | None = None,
    selector: Callable[[dict], bool] | None = None,
    per_passage: Callable[[list[dict]], list[dict]] | None = None,
    pair_weight: Callable[[dict], int] | None = None,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = 0,
    validate: bool = True,
    system_prompt: str | None = None,
) -> TrainCorpus:
    """Validate, filter, weight, and split the corpus for styler training.

    `splits` keeps STYLER-role pairs only (None: every pair, for fixtures);
    `selector` is the pair-level hook (default keep-all, near-copies included);
    `per_passage` sees each passage's records in corpus order and returns the
    subset to keep (D43); `pair_weight` stamps `meta.weight`, honoured as
    integer replication in the TRAIN file only; `system_prompt` is what the
    records are validated against (D64). The val split holds out `max(1,
    round(n * val_frac))` of the kept POSTs under `random.Random(seed)` —
    deterministic, disjoint by post. Raises `RunError` on a malformed corpus.
    """
    pairs_path = Path(pairs_path)
    if validate:
        errors = validate_pairs_file(pairs_path, system_prompt=system_prompt)
        if errors:
            lineno, msgs = errors[0]
            raise RunError(
                f"{pairs_path} fails the pairs contract on {len(errors)} record(s); "
                f"first at {lineno}: {'; '.join(msgs)} — refusing to train on a "
                f"malformed corpus"
            )

    dropped = {"role": 0, "selector": 0, "per_passage": 0}
    kept: list[dict] = []
    n_source = 0
    if splits is not None:
        from stylebot import splits as splits_mod

    for rec in iter_pairs(pairs_path):
        n_source += 1
        source = (rec.get("meta") or {}).get("source") or "?"
        if splits is not None and splits_mod.role_of(source, splits) != "styler":
            dropped["role"] += 1
            continue
        if selector is not None and not selector(rec):
            dropped["selector"] += 1
            continue
        kept.append(rec)

    if per_passage is not None:
        capped: list[dict] = []
        for group in passage_groups(kept):
            capped.extend(per_passage(group))
        dropped["per_passage"] = len(kept) - len(capped)
        kept = capped

    if pair_weight is not None:
        for rec in kept:
            rec.setdefault("meta", {})["weight"] = max(1, int(pair_weight(rec)))

    posts = sorted({(rec.get("meta") or {}).get("source") or "?" for rec in kept})
    if val_frac > 0 and posts:
        shuffled = list(posts)
        random.Random(seed).shuffle(shuffled)
        n_val = max(1, round(len(posts) * val_frac))
        val_posts = set(shuffled[:n_val])
    else:
        val_posts = set()

    corpus = TrainCorpus(
        train_posts=sorted(set(posts) - val_posts),
        val_posts=sorted(val_posts),
        pairs_sha256=content_sha256(pairs_path),
        n_source_records=n_source,
        dropped=dropped,
    )
    for rec in kept:
        source = (rec.get("meta") or {}).get("source") or "?"
        (corpus.val if source in val_posts else corpus.train).append(rec)
    return corpus


def estimate_tokens(records: Sequence[dict]) -> int:
    """Keyless ~4 chars/token estimate over every message, each record
    `meta.weight` times (replication is what the train file does): the dry-run
    cost preview only; the paid path counts real rendered tokens."""
    chars = sum(
        pair_weight_of(rec) * len(m.get("content") or "")
        for rec in records
        for m in rec.get("messages", ())
    )
    return chars // _CHARS_PER_TOKEN


#: Packages whose version changes what a run does; an absent one reads `None`,
#: so "not installed" differs from "this manifest predates the pin" (D38).
PINNED_DEPENDENCIES = ("stylebot", "tinker", "tinker-cookbook", "torch", "transformers", "datasets")


def dependency_versions(names: Sequence[str] = PINNED_DEPENDENCIES) -> dict[str, str | None]:
    """Installed versions of the packages a run's behaviour depends on."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _splits_sha256(splits_path: Path | None) -> str | None:
    if splits_path is None or not Path(splits_path).exists():
        return None
    return file_sha256(splits_path)


def _record_ids_sha256(corpus: TrainCorpus) -> str:
    """16-hex hash of the SORTED kept `record_id`s: which pairs were in this
    run, invariant to corpus order and to the train/val partition (`val_posts`)."""
    from stylebot.pairs import record_id

    ids = sorted(record_id(rec) for rec in [*corpus.train, *corpus.val])
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return digest[:16]


def build_manifest(
    *,
    run_id: str,
    pairs_path: Path,
    corpus: TrainCorpus,
    hp: Hyperparameters,
    splits_path: Path | None,
    dry_run: bool,
    train_price_per_mtok: float | None = None,
    filters: dict | None = None,
    splits: dict | None = None,
    selector_constants: dict | None = None,
    base_model_revision: str | None = None,
) -> dict:
    """The pre-run manifest — everything reproducibility needs before any paid
    call; `run_training` appends `result`. Which pins, and why each: D38."""
    est = estimate_tokens(corpus.train) * hp.num_epochs
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "styler-lora-sft",
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dry_run": dry_run,
        "git_sha": {
            "cwd": git_sha("."),
            "stylebot": git_sha(Path(__file__).resolve().parent),
        },
        "data": {
            "pairs_path": str(pairs_path),
            "pairs_sha256": corpus.pairs_sha256,
            "n_source_records": corpus.n_source_records,
            "n_train": corpus.n_train,
            "n_val": corpus.n_val,
            "n_real": corpus.n_real,
            "n_synthetic": corpus.n_synthetic,
            "dropped": corpus.dropped,
            "splits_path": str(splits_path) if splits_path else None,
            "style_system": style_system(),  # a view's own overrides it
            "style_system_sha": system_sha(),
            "splits_sha256": _splits_sha256(splits_path),
            "eval_version": (splits or {}).get("eval_version"),
            # which RECORDS were kept: every filter runs after `pairs_sha256` (D38)
            "record_ids_sha256": _record_ids_sha256(corpus),
            "n_train_posts": len(corpus.train_posts),
            "val_posts": corpus.val_posts,
            "filters": filters or {"selector": None, "per_passage": None, "pair_weight": None},
            "selector_constants": selector_constants,
            "chunk_scale": corpus.chunk_scale,  # the diet serving follows
        },
        # a None knob is the cookbook's recommendation; `result` records what it resolved to
        "hyperparameters": hp.as_manifest(),
        # WHICH upload of the base weights; None offline, filled by the runner
        "base_model_revision": base_model_revision,
        "dependencies": dependency_versions(),
        "estimate": {
            "train_tokens": est,
            "basis": f"~{_CHARS_PER_TOKEN} chars/token heuristic x {hp.num_epochs} epoch(s)",
            "train_price_per_mtok": train_price_per_mtok,
            "cost_usd": (
                round(est / 1_000_000 * train_price_per_mtok, 4) if train_price_per_mtok else None
            ),
        },
        "result": None,  # filled by the paid run
    }
    return manifest


def format_passage_scale(chunk_scale: dict) -> str:
    """The per-passage weight line of the `train` report; reads the manifest
    block, so an OLD manifest with no `passages` key renders too."""
    passages = (chunk_scale or {}).get("passages") or {}
    train = passages.get("train")
    if not train:
        return "passages: not recorded (manifest predates the per-passage histogram)"
    hist = train.get("pairs_per_passage") or {}
    spread = ", ".join(f"x{k}: {v}" for k, v in hist.items())
    examples = (chunk_scale or {}).get("n_train_examples")
    weights = (chunk_scale or {}).get("pair_weights") or {}
    weighted = ""
    if examples is not None and set(weights) - {"1"}:
        weighted = f"; weights {dict(weights)} -> {examples:,} training rows"
    return (
        f"{train['n_passages']:,} distinct passage(s), "
        f"{train['mean_pairs_per_passage']}x mean / "
        f"{train['max_pairs_per_passage']}x max pairs per passage "
        f"[{spread}]{weighted}"
    )


def write_manifest(manifest: dict, manifest_out: str | Path) -> Path:
    manifest_out = Path(manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_out


@dataclass
class TrainResult:
    manifest: dict
    manifest_path: Path
    corpus: TrainCorpus
    adapter_dir: Path | None = None  # PEFT adapter (never committed)


def default_run_id() -> str:
    """The run id minted at launch: a UTC timestamp, which both CLIs name files by."""
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def run_training(
    pairs_path: str | Path | None,
    work_dir: str | Path,
    manifest_out: str | Path,
    *,
    draws_dir: str | Path | None = None,
    policy: draws.ViewPolicy | None = None,
    hp: Hyperparameters | None = None,
    splits_path: str | Path | None = None,
    blog_root: str | Path | None = None,
    selector: Callable[[dict], bool] | None = None,
    per_passage: Callable[[list[dict]], list[dict]] | None = None,
    pair_weight: Callable[[dict], int] | None = None,
    val_frac: float = DEFAULT_VAL_FRAC,
    run_id: str | None = None,
    train_price_per_mtok: float | None = None,
    selector_constants: dict | None = None,
    base_model_revision: str | None = None,
    dry_run: bool = False,
    download_adapter: bool = True,
    runner: Callable[..., dict] | None = None,
) -> TrainResult:
    """Assemble → manifest → (unless dry-run) train on Tinker → export adapter.

    Data: `pairs_path`, or the view of `draws_dir` under `policy`, whose pins
    (`policy_sha`, `draws_sha256`, the `style_system` text) the manifest
    carries; `blog_root` + `splits_path` refuse a run, dry run included, whose
    pins or corpus sources left the tree (D34). A dry run writes nothing and
    returns the manifest it would write; the paid run writes it pre-run and with
    `result`. `runner` is the seam `(corpus, manifest, work_dir, hp, *, download_adapter)`.
    """
    view_pins: dict = {}
    if draws_dir is not None:
        pairs_path, view_pins = draws.training_view(draws_dir, policy)
    elif pairs_path is None:
        raise ConfigError("run_training needs pairs_path or draws_dir")
    pairs_path = Path(pairs_path)
    work_dir = Path(work_dir)
    run_id = run_id or default_run_id()
    hp = hp if hp is not None else Hyperparameters()

    splits = None
    if splits_path is not None:
        from stylebot import splits as sp

        splits = sp.load_splits(splits_path)
        if blog_root is not None:
            sp.require_sources_on_disk(splits, sp.corpus_sources(pairs_path), blog_root)

    corpus = assemble_training_corpus(
        pairs_path,
        splits=splits,
        selector=selector,
        per_passage=per_passage,
        pair_weight=pair_weight,
        val_frac=val_frac,
        seed=hp.seed,
        system_prompt=policy.system_prompt if policy else None,
    )
    if corpus.n_train == 0:
        raise ConfigError(f"no training pairs left after filtering {pairs_path}")

    filters = {
        "selector": _hook_name(selector),
        "per_passage": _hook_name(per_passage),
        "pair_weight": _hook_name(pair_weight),
    }
    manifest = build_manifest(
        run_id=run_id,
        pairs_path=pairs_path,
        corpus=corpus,
        hp=hp,
        splits_path=Path(splits_path) if splits_path else None,
        splits=splits,
        dry_run=dry_run,
        train_price_per_mtok=train_price_per_mtok,
        filters=filters,
        selector_constants=selector_constants,
        base_model_revision=base_model_revision,
    )
    manifest["data"] |= {"policy_sha": None, "draws_sha256": None, **view_pins}
    if dry_run:
        return TrainResult(manifest=manifest, manifest_path=Path(manifest_out), corpus=corpus)

    manifest_path = write_manifest(manifest, manifest_out)
    work_dir.mkdir(parents=True, exist_ok=True)
    run = runner or _tinker_runner
    result = run(corpus, manifest, work_dir, hp, download_adapter=download_adapter)
    manifest["result"] = {k: v for k, v in result.items() if k != "adapter_dir"}
    manifest_path = write_manifest(manifest, manifest_out)
    adapter = result.get("adapter_dir")
    return TrainResult(
        manifest=manifest,
        manifest_path=manifest_path,
        corpus=corpus,
        adapter_dir=Path(adapter) if adapter else None,
    )


@dataclass(frozen=True)
class TrainOpts:
    """The `train` surface — what `run_training_pass` takes (D78).

    `pairs_path`, `draws_dir`, `policy` and `hp` carry no flag: the CLI fills
    them from `--data-dir`, `draws.view_options()` and the `Hyperparameters`
    bundle; a wrapper sets them on the bundle it passes.
    """

    cli_section = "Training run"

    data_dir: Path | None = opt(
        None,
        "--data-dir",
        "data_dir",
        required=True,
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Data dir holding the corpus (pairs/ or pairs.jsonl). Required, no default: "
        "the explicit path is the reproducibility record.",
    )
    manifest_out: Path | None = opt(
        None,
        "--manifest-out",
        required=True,
        type=click.Path(dir_okay=False, path_type=Path),
        help="Where to write the run manifest JSON (the ONLY artifact that gets committed).",
    )
    work_dir: Path | None = opt(
        None,
        "--work-dir",
        required=True,
        type=click.Path(file_okay=False, path_type=Path),
        help="Scratch dir: assembled data, tinker logs, the exported adapter (never committed).",
    )
    splits_path: Path | None = opt(
        None,
        "--splits",
        "splits_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="The shared splits.json: train on STYLER-role posts only (see make-splits). "
        "Omit only for fixture/experiment corpora.",
    )
    blog_root: Path | None = opt(
        None,
        "--blog-root",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        is_eager=True,  # processed before any corpus default that reads it (D79)
        help="Tree the pinned eval posts must still exist in; a renamed one fails the run (D34).",
    )
    val_frac: float = opt(
        DEFAULT_VAL_FRAC,
        "--val-frac",
        show_default=True,
        type=float,
        help="Fraction of POSTs held out for val loss (by-post, deterministic).",
    )
    run_id: str | None = opt(
        None,
        "--run-id",
        help="Run id for the manifest [default: a UTC timestamp minted at launch].",
    )
    train_price_per_mtok: float | None = opt(
        None,
        "--price-per-mtok",
        "train_price_per_mtok",
        type=float,
        help="Train price $/1M tokens for the cost estimate (the Tinker models page).",
    )
    download_adapter: bool = opt(
        True,
        "--download-adapter/--no-download-adapter",
        show_default=True,
        help="Export the trained LoRA as a PEFT adapter into work-dir after the run.",
    )
    dry_run: bool = opt(
        False,
        "--dry-run",
        is_flag=True,
        help="Assemble, print the manifest and cost estimate, stop. Writes nothing; no key.",
    )
    assume_yes: bool = opt(
        False, "--yes", "assume_yes", is_flag=True, help="Skip the pre-spend confirmation prompt."
    )
    pairs_path: Path | None = None
    draws_dir: Path | None = None
    policy: draws.ViewPolicy | None = None
    hp: Hyperparameters | None = None


TRAIN_PARAMS: dict[str, type] = {"opts": TrainOpts, "hp": Hyperparameters}


def train_options(**kw):
    """The whole `train` surface as one decorator; `kw` is `bundle_options`'."""
    return bundle_options([TrainOpts, Hyperparameters], **kw)


def format_preview(manifest: dict) -> list[str]:
    """The pre-spend report both `train` CLIs print: counts, val posts, scale, cost."""
    data, est, hp = manifest["data"], manifest["estimate"], manifest["hyperparameters"]
    view = f"; view pairs-{data['policy_sha']}" if data.get("policy_sha") else ""
    cost = f" ≈ ${est['cost_usd']}" if est["cost_usd"] is not None else ""
    renderer = f", {hp['renderer_name']}" if hp.get("renderer_name") else ""
    return [
        f"{data['n_train']} train / {data['n_val']} val pair(s) "
        f"({data['n_real']} real / {data['n_synthetic']} synthetic; "
        f"dropped {data['dropped']}) from {data['n_source_records']} record(s), "
        f"pairs sha256 {data['pairs_sha256']}{view}",
        f"val posts: {', '.join(data['val_posts'])}",
        format_passage_scale(data["chunk_scale"]),
        f"~{est['train_tokens']:,} train tokens ({est['basis']}){cost} on "
        f"{hp['base_model']} (rank {hp['lora_rank']}, {hp['num_epochs']} epoch(s){renderer})",
    ]


def format_result(result: TrainResult) -> list[str]:
    """The paid run's report: loss-ish metrics, the checkpoint, adapter, manifest."""
    run = result.manifest["result"] or {}
    keys = ("nll", "loss", "bpb", "tokens")
    lines = [f"{k}: {v}" for k, v in sorted(run.items()) if any(t in k for t in keys)]
    if sampler := (run.get("checkpoints") or {}).get("sampler_path"):
        lines.append(f"sampler checkpoint: {sampler}")
    if result.adapter_dir:
        lines.append(f"PEFT adapter -> {result.adapter_dir} (never commit this)")
    lines.append(f"manifest -> {result.manifest_path} (commit this)")
    return lines


def run_training_pass(
    opts: TrainOpts,
    *,
    selector: Callable[[dict], bool] | None = None,
    per_passage: Callable[[list[dict]], list[dict]] | None = None,
    pair_weight: Callable[[dict], int] | None = None,
    selector_constants: dict | None = None,
    confirm: Callable[[dict], bool] | None = None,
    emit: Emitter = silent,
) -> TrainResult:
    """The `train` pass: preview and report, then the paid run unless `opts.dry_run`.

    `confirm` is asked with the preview manifest before any spend (`None` asks
    nothing). Returns the preview or the run. Raises `ConfigError` with no
    corpus, `RunError` with no key or a declined prompt.
    """
    pairs_path = opts.pairs_path
    if pairs_path is None and opts.data_dir is not None:
        pairs_path = config.pairs_path(opts.data_dir)
    if opts.draws_dir is None and (pairs_path is None or not pairs_path.exists()):
        raise ConfigError(f"no corpus (pairs/ or pairs.jsonl) in {opts.data_dir}")
    carried = ("draws_dir", "policy", "hp", "splits_path", "blog_root", "val_frac", "run_id")
    kwargs = {name: getattr(opts, name) for name in carried} | dict(
        train_price_per_mtok=opts.train_price_per_mtok,
        download_adapter=opts.download_adapter,
        selector=selector,
        per_passage=per_passage,
        pair_weight=pair_weight,
        selector_constants=selector_constants,
    )
    # Always preview first: the dry-run manifest is also the pre-spend estimate.
    preview = run_training(pairs_path, opts.work_dir, opts.manifest_out, dry_run=True, **kwargs)
    for line in format_preview(preview.manifest):
        emit(line)
    if opts.dry_run:
        emit(f"[dry-run] the manifest a paid run would write to {preview.manifest_path}:")
        emit(json.dumps(preview.manifest, indent=2))
        return preview
    if not config.get_key("TINKER_API_KEY"):
        raise RunError(
            "TINKER_API_KEY is not set — the paid run needs it. Export it, or run under "
            "direnv (`direnv exec <dir-with-an-.envrc> uv run ai-style train ...`)."
        )
    if confirm is not None and not confirm(preview.manifest):
        raise RunError("the paid run was declined", quiet=True)
    result = run_training(pairs_path, opts.work_dir, opts.manifest_out, dry_run=False, **kwargs)
    for line in format_result(result):
        emit(line)
    return result


def _write_conversations(records: Sequence[dict], path: Path, *, replicate: bool = True) -> int:
    """Write records as bare `{"messages": ...}` JSONL for the run's work dir
    (never committed); returns the row count. Weight is consumed HERE, as
    `meta.weight` copies of the row — the only up-weighting the cookbook's
    dataset can express (D18); `replicate=False` keeps the val file one row
    per pair, so val loss stays comparable across runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            line = dumps_line({"messages": rec["messages"]}) + "\n"
            for _ in range(pair_weight_of(rec) if replicate else 1):
                fp.write(line)
                rows += 1
    return rows


def _make_presplit_builder(
    train_file: Path, val_file: Path | None, *, common_config, shuffle_seed: int
):
    """A ChatDatasetBuilder over OUR by-POST pre-split files: the cookbook's
    `FromConversationFileBuilder` row-shuffles one file into train/test and
    would leak val posts' sibling chunks. Lazy: `chz` needs the trainer extra.
    `shuffle_seed` is required so the epoch shuffle takes the seed as well."""
    import chz
    import datasets as hf_datasets
    from tinker_cookbook.renderers import TrainOnWhat
    from tinker_cookbook.supervised.data import (
        SupervisedDatasetFromHFDataset,
        conversation_to_datum,
    )
    from tinker_cookbook.supervised.types import ChatDatasetBuilder

    @chz.chz
    class PreSplitConversationFilesBuilder(ChatDatasetBuilder):
        train_file: str
        val_file: str | None = None
        shuffle_seed: int = 0

        def __call__(self):
            def load(path: str) -> hf_datasets.Dataset:
                text = Path(path).read_text(encoding="utf-8")
                rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                return hf_datasets.Dataset.from_list(rows)

            train_on_what = (
                TrainOnWhat(self.common_config.train_on_what)
                if self.common_config.train_on_what
                else TrainOnWhat.ALL_ASSISTANT_MESSAGES
            )

            def map_fn(row: dict):
                return conversation_to_datum(
                    row["messages"], self.renderer, self.common_config.max_length, train_on_what
                )

            train_ds = load(self.train_file).shuffle(seed=self.shuffle_seed)
            train = SupervisedDatasetFromHFDataset(
                train_ds, batch_size=self.common_config.batch_size, map_fn=map_fn
            )
            val = None
            if self.val_file:
                val_ds = load(self.val_file)
                val = SupervisedDatasetFromHFDataset(val_ds, batch_size=len(val_ds), map_fn=map_fn)
            return train, val

    return PreSplitConversationFilesBuilder(
        train_file=str(train_file),
        val_file=str(val_file) if val_file else None,
        shuffle_seed=shuffle_seed,
        common_config=common_config,
    )


def _read_final_metrics(log_path: Path) -> dict:
    """Distil the cookbook's metrics.jsonl: final value per loss-ish key +
    total trained tokens."""
    metrics_file = log_path / "metrics.jsonl"
    final: dict = {}
    total_tokens = 0
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_tokens += int(row.get("num_tokens") or 0)
            for key, value in row.items():
                if isinstance(value, (int, float)) and (
                    "nll" in key or "loss" in key or "bpb" in key
                ):
                    final[key] = value
    final["train_tokens"] = total_tokens
    return final


def _tinker_runner(
    corpus: TrainCorpus,
    manifest: dict,
    work_dir: Path,
    hp: Hyperparameters,
    *,
    download_adapter: bool = True,
) -> dict:
    """The real Tinker execution: the cookbook supervised recipe (LoRA SFT)
    over the assembled corpus, then a PEFT-adapter export. All heavy imports
    live here; the interface is pinned against the installed tinker-cookbook."""
    try:
        import asyncio

        from tinker_cookbook import checkpoint_utils, hyperparam_utils, model_info
        from tinker_cookbook.supervised import train as sft
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    except ImportError as exc:
        raise MissingExtra(_EXTRA_HINT) from exc

    base_model = hp.base_model
    renderer_name = hp.renderer_name or model_info.get_recommended_renderer_name(base_model)
    learning_rate = hp.learning_rate or hyperparam_utils.get_lr(base_model)

    data_dir = work_dir / "data"
    train_file = data_dir / "train.jsonl"
    n_rows = _write_conversations(corpus.train, train_file)
    val_file = data_dir / "val.jsonl" if corpus.val else None
    if val_file is not None:
        _write_conversations(corpus.val, val_file, replicate=False)
    if n_rows != corpus.n_train:
        ratio = n_rows / corpus.n_train
        logger.info(
            "weighted train file: %d rows from %d pairs (%.2fx)", n_rows, corpus.n_train, ratio
        )

    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=base_model,
        renderer_name=renderer_name,
        max_length=hp.max_length,
        batch_size=hp.batch_size,
    )
    builder = _make_presplit_builder(
        train_file, val_file, common_config=common, shuffle_seed=hp.seed
    )

    log_path = work_dir / "tinker"
    sft_config = sft.Config(
        log_path=str(log_path),
        model_name=base_model,
        recipe_name="stylebot_styler_sft",
        renderer_name=renderer_name,
        dataset_builder=builder,
        learning_rate=learning_rate,
        lr_schedule=hp.lr_schedule,
        num_epochs=hp.num_epochs,
        lora_rank=hp.lora_rank,
    )
    logger.info(
        "tinker SFT: %s rank=%s lr=%.3g renderer=%s (%d train / %d val pairs)",
        *(base_model, hp.lora_rank, learning_rate, renderer_name, corpus.n_train, corpus.n_val),
    )
    asyncio.run(sft.main(sft_config))

    record = checkpoint_utils.get_last_checkpoint(str(log_path))
    result: dict = {
        "renderer_name": renderer_name,
        "learning_rate": learning_rate,
        "log_path": str(log_path),
        "checkpoints": {
            "state_path": record.state_path if record else None,
            "sampler_path": record.sampler_path if record else None,
        },
        **_read_final_metrics(log_path),
    }

    sampler_path = result["checkpoints"]["sampler_path"]
    if download_adapter and sampler_path:
        from tinker_cookbook import weights

        raw_dir = work_dir / "weights"
        adapter_dir = work_dir / "peft_adapter"
        downloaded = weights.download(tinker_path=sampler_path, output_dir=str(raw_dir))
        weights.build_lora_adapter(
            base_model=base_model, adapter_path=downloaded, output_path=str(adapter_dir)
        )
        result["adapter_dir"] = str(adapter_dir)
    return result
