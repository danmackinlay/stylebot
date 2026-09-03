"""Phase 3: LoRA SFT of the styler — assembly, manifest, and the Tinker run.

Two layers, mirroring `classify_train`'s split of pure assembly vs paid fit:
`assemble_training_corpus` is keyless and fully testable without the extra
(validate the contract, filter to **styler**-role posts under `stylebot.splits`,
apply caller policy hooks, cut a deterministic by-POST validation split);
`run_training` is the paid Tinker path. All `tinker`/`tinker_cookbook` imports
are lazy; the module imports free.

**Data policy this module deliberately encodes** (pinned in
`_plans/decisions.md` D17/D18 — do not "fix" these):

- Near-copy pairs (`meta.transform_sim > 0.85`) are KEPT. They are label noise
  for the *detector* (`classify_train.assemble_dataset` drops them) but good
  styler data: they teach the model to leave alone prose that is already fine.
- Records train exactly as stored: `STYLE_SYSTEM` verbatim as `messages[0]`,
  heading context left on both sides. Never strip or rewrite a side.
- Per-target multiplicity is a *policy hook* (`per_target`), not a built-in
  gate: re-key epochs and deliberate replicates can stack several pairs on one
  target passage, and whether that up-weights the passage is the caller's call.

**The manifest is the reproducibility record.** Weights and corpus never enter
git; the small JSON written by `write_manifest` (content hash of the corpus,
split, hyperparameters, tinker paths, cost) is what gets committed.

Needs the ``trainer`` extra for the paid path —
``uv add 'stylebot[trainer]'`` (tinker, tinker-cookbook).
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

from stylebot.cli_options import opt
from stylebot.jsonl import dumps_line
from stylebot.pairs import iter_pairs, validate_pairs_file
from stylebot.provenance import file_sha256, git_sha  # file_sha256 re-exported: public here

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

# Crude keyless token estimate for dry-run cost previews: prose runs ~4 chars
# per token. The paid path counts real rendered tokens before spending.
_CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class Hyperparameters:
    """The LoRA-SFT knobs — one declaration, every consumer.

    **This dataclass IS the manifest's `hyperparameters` block**: `as_manifest()`
    is a plain field dump, so adding a knob here puts it in the committed
    reproducibility record with no second edit, and no knob can be set but
    unrecorded. Each field also carries its own click flag via
    `cli_options.opt()`, so `ai-style train` and livingthing's `dan-style train`
    share one surface and re-pin what differs with
    `bundle_options(defaults=[Hyperparameters(...)])` instead of restating nine
    decorators each.

    It replaced a nine-name relay written out by hand at five layers — click
    decorators, the command signature, a `kwargs` dict, `run_training`,
    `build_manifest`, the manifest literal, and `**cfg` string lookups in
    `_tinker_runner`. The last of those was the dangerous one: an untyped
    `cfg["max_lenght"]` was a paid run away from a `KeyError`.

    A library module carrying click declarations is deliberate, and cheap:
    `opt()` only stashes the declaration in the field's `metadata`, which is
    inert unless `bundle_options` reads it, and click is an unconditional
    dependency. The alternative — a `train_cli.py` twin — is exactly the
    two-dataclasses-one-surface shape that `ChunkPolicy`/`ChunkOpts` was.
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
        0,
        "--seed",
        show_default=True,
        type=int,
        help="Seed for the val split and epoch shuffling.",
    )

    def as_manifest(self) -> dict:
        """This bundle as the manifest's `hyperparameters` block."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

_EXTRA_HINT = (
    "stylebot.train needs the 'trainer' extra: "
    "run `uv add 'stylebot[trainer]'` (or `pip install 'stylebot[trainer]'`) "
    "to get tinker + tinker-cookbook. Assembly and --dry-run work without it."
)


# ---------------------------------------------------------------------------
# Assembly — pure, keyless, deterministic
# ---------------------------------------------------------------------------


@dataclass
class TrainCorpus:
    """The assembled styler training corpus, split and accounted for.

    `train`/`val` hold full pair records (messages + meta), untouched. The val
    split is by POST (`meta.source`) so val loss is honest — no passage of a
    val post appears in training.
    """

    train: list[dict] = field(default_factory=list)
    val: list[dict] = field(default_factory=list)
    train_posts: list[str] = field(default_factory=list)
    val_posts: list[str] = field(default_factory=list)
    pairs_sha256: str | None = None  # 16-hex content hash of the source file
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
        """What chunk diet this run was actually trained on.

        Serving must chunk at the scale the adapter saw: at out-of-distribution
        lengths the styler falls back to copying the input verbatim, so a
        hard-coded inference budget silently rots the moment the corpus moves.
        Recording it here lets `default_chunk_chars` read the promoted manifest
        instead of guessing.

        `budgets` maps each `meta.merge_max_chars` present to its pair count
        ("null" = unmerged records); `modes` does the same for
        `meta.chunk_mode`. `max_target_chars` is the longest assistant side —
        the ceiling on what the adapter has ever been asked to emit.
        """
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
        return {"budgets": budgets, "modes": modes, "max_target_chars": longest}


def _target_body(rec: dict) -> str:
    """The assistant-side body, heading-context stripped — the per-target
    grouping key for the multiplicity hook (same recovery as `synth`'s
    coverage check)."""
    from stylebot.pairs import extract_target

    return extract_target(rec).strip()


def assemble_training_corpus(
    pairs_path: str | Path,
    *,
    splits: dict | None = None,
    selector: Callable[[dict], bool] | None = None,
    per_target: Callable[[list[dict]], list[dict]] | None = None,
    val_frac: float = DEFAULT_VAL_FRAC,
    seed: int = 0,
    validate: bool = True,
) -> TrainCorpus:
    """Validate, filter, and split the corpus for styler training.

    - `splits`: a loaded `splits.json`. When given, only **styler**-role pairs
      are kept — the detector pool and the frozen eval posts never enter
      training. When None, every pair is eligible (fixture/experiment use).
    - `selector`: pair-level policy hook (record -> keep?). Default keep-all —
      note this deliberately KEEPS near-copy pairs (see module docstring).
    - `per_target`: multiplicity hook. Pairs sharing one target body are passed
      as a list (corpus order); the hook returns the subset to keep.
    - The val split shuffles the kept POSTs with `random.Random(seed)` and
      holds out `max(1, round(n * val_frac))` of them (0 with `val_frac=0`),
      so it is deterministic and disjoint by post.
    """
    pairs_path = Path(pairs_path)
    if validate:
        errors = validate_pairs_file(pairs_path)
        if errors:
            lineno, msgs = errors[0]
            raise ValueError(
                f"{pairs_path} fails the pairs contract on {len(errors)} record(s); "
                f"first: line {lineno}: {'; '.join(msgs)} — refusing to train on a "
                f"malformed corpus"
            )

    dropped = {"role": 0, "selector": 0, "per_target": 0}
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

    if per_target is not None:
        by_target: dict[str, list[dict]] = {}
        order: list[str] = []
        for rec in kept:
            key = _target_body(rec)
            if key not in by_target:
                by_target[key] = []
                order.append(key)
            by_target[key].append(rec)
        capped: list[dict] = []
        for key in order:
            capped.extend(per_target(by_target[key]))
        dropped["per_target"] = len(kept) - len(capped)
        kept = capped

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
        pairs_sha256=file_sha256(pairs_path),
        n_source_records=n_source,
        dropped=dropped,
    )
    for rec in kept:
        source = (rec.get("meta") or {}).get("source") or "?"
        (corpus.val if source in val_posts else corpus.train).append(rec)
    return corpus


def estimate_tokens(records: Sequence[dict]) -> int:
    """Keyless order-of-magnitude token estimate (~4 chars/token) over every
    message of every record — for the dry-run cost preview only; the paid path
    counts real rendered tokens."""
    chars = sum(len(m.get("content") or "") for rec in records for m in rec.get("messages", ()))
    return chars // _CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
# Manifest — the committed reproducibility record
# ---------------------------------------------------------------------------


#: Packages whose version silently changes what a training run does. Recorded by
#: name so a manifest says which stack produced it; absent ones read `None`
#: rather than being omitted, so "not installed" is distinguishable from "this
#: manifest predates the pin".
PINNED_DEPENDENCIES = (
    "stylebot",
    "tinker",
    "tinker-cookbook",
    "torch",
    "transformers",
    "datasets",
)


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
    """16-hex hash of the SORTED kept `record_id`s (train + val).

    Sorted, so it is invariant to corpus order and to the train/val partition
    within one selection — it answers "which pairs were in this run", and the
    split itself is recorded separately as `val_posts`.
    """
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
    """The pre-run manifest: everything reproducibility needs *before* any
    paid call. The run appends `result` (tinker paths, real tokens, cost,
    losses) via `run_training`.

    Four pins were added 2026-09 (`_plans/decisions.md` D38), because the
    manifest named the corpus and both git trees but not the things that
    actually differed between two runs of "the same" recipe: `splits_sha256` +
    `eval_version` (which partition held out which posts), `record_ids_sha256`
    (which records survived the filters — the corpus hash covers the file, not
    the selection), `dependencies` (tinker/cookbook/transformers versions, which
    move under you), and `selector_constants` (the caller's policy literals:
    `RETIRED_ARMS` is a set whose membership changes the diet while
    `filters.selector` keeps saying `styler_pair_ok`).
    """
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
            "splits_sha256": _splits_sha256(splits_path),
            "eval_version": (splits or {}).get("eval_version"),
            # Which RECORDS were kept, not merely which file they came from: the
            # selector, the role filter and the multiplicity hook all run after
            # `pairs_sha256` is taken, so two runs over one corpus can (and did)
            # train on different data under identical hashes.
            "record_ids_sha256": _record_ids_sha256(corpus),
            "n_train_posts": len(corpus.train_posts),
            "val_posts": corpus.val_posts,
            "filters": filters or {"selector": None, "per_target": None},
            "selector_constants": selector_constants,
            # The chunk diet, so serving can follow it instead of hard-coding a
            # budget that goes stale the next time the corpus is re-cut.
            "chunk_scale": corpus.chunk_scale,
        },
        # `learning_rate: None` -> cookbook recommendation, `renderer_name: None`
        # -> model_info recommendation; the run records what it resolved them to
        # under `result`.
        "hyperparameters": hp.as_manifest(),
        # The base weights are named by `hyperparameters.base_model`; a revision
        # says WHICH upload of them. None when nothing can answer offline — the
        # tinker runner fills it in when the service reports one.
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


def write_manifest(manifest: dict, manifest_out: str | Path) -> Path:
    manifest_out = Path(manifest_out)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_out


# ---------------------------------------------------------------------------
# The paid run
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    manifest: dict
    manifest_path: Path
    corpus: TrainCorpus
    adapter_dir: Path | None = None  # PEFT adapter (never committed)


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def run_training(
    pairs_path: str | Path,
    work_dir: str | Path,
    manifest_out: str | Path,
    *,
    hp: Hyperparameters | None = None,
    splits_path: str | Path | None = None,
    selector: Callable[[dict], bool] | None = None,
    per_target: Callable[[list[dict]], list[dict]] | None = None,
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

    The training knobs arrive as a `Hyperparameters` bundle
    (`run_training(pairs, work, out, hp=Hyperparameters(lora_rank=64))`),
    defaulting to the cookbook-recipe defaults; everything left here is a knob
    that is *not* a hyperparameter — where the data comes from, which policy
    hooks filter it, and what to do with the artifacts.

    `val_frac` stays out of the bundle deliberately: it shapes the data split,
    not the fit, and the manifest records its *outcome* (`data.val_posts`)
    rather than the fraction. `hp.seed` drives both the split and epoch
    shuffling, and is recorded, so it belongs in the bundle.

    The manifest at `manifest_out` is written twice: once pre-run (so even an
    interrupted run leaves its record) and once with `result` filled in. On
    `dry_run` only the pre-run manifest is written — no tinker import, no key,
    no spend. `runner` is the paid-execution seam:
    `(corpus, manifest, work_dir, hp, *, download_adapter) -> result dict`,
    defaulting to `_tinker_runner`; tests inject a fake.
    """
    pairs_path = Path(pairs_path)
    work_dir = Path(work_dir)
    run_id = run_id or _default_run_id()
    hp = hp if hp is not None else Hyperparameters()

    splits = None
    if splits_path is not None:
        from stylebot import splits as splits_mod

        splits = splits_mod.load_splits(splits_path)

    corpus = assemble_training_corpus(
        pairs_path,
        splits=splits,
        selector=selector,
        per_target=per_target,
        val_frac=val_frac,
        seed=hp.seed,
    )
    if corpus.n_train == 0:
        raise ValueError(f"no training pairs left after filtering {pairs_path}")

    filters = {
        "selector": getattr(selector, "__name__", repr(selector)) if selector else None,
        "per_target": getattr(per_target, "__name__", repr(per_target)) if per_target else None,
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
    manifest_path = write_manifest(manifest, manifest_out)
    if dry_run:
        return TrainResult(manifest=manifest, manifest_path=manifest_path, corpus=corpus)

    work_dir.mkdir(parents=True, exist_ok=True)
    run = runner or _tinker_runner
    result = run(corpus, manifest, work_dir, hp, download_adapter=download_adapter)
    manifest["dry_run"] = False
    manifest["result"] = {k: v for k, v in result.items() if k != "adapter_dir"}
    manifest_path = write_manifest(manifest, manifest_out)
    adapter = result.get("adapter_dir")
    return TrainResult(
        manifest=manifest,
        manifest_path=manifest_path,
        corpus=corpus,
        adapter_dir=Path(adapter) if adapter else None,
    )


def _write_conversations(records: Sequence[dict], path: Path) -> Path:
    """Write records as bare `{"messages": ...}` JSONL — derived training data
    for the run's work dir (never committed). Meta is stripped because the HF
    dataset layer wants a uniform schema and training only reads messages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(dumps_line({"messages": rec["messages"]}) + "\n")
    return path


def _make_presplit_builder(
    train_file: Path, val_file: Path | None, *, common_config, shuffle_seed: int
):
    """A ChatDatasetBuilder over OUR by-POST pre-split files.

    The cookbook's `FromConversationFileBuilder` row-shuffles one file into
    train/test, which would leak val posts' sibling chunks into training; this
    builder takes the two files assembly already split by POST. Defined lazily:
    `chz` classes need the trainer extra.

    `shuffle_seed` is required, not defaulted: it must reach the epoch shuffle
    as well as the val split, or two "different" seeds train on the same order.
    A missing argument must fail loudly, not silently re-pin 0.
    """
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
        raise ImportError(_EXTRA_HINT) from exc

    base_model = hp.base_model
    renderer_name = hp.renderer_name or model_info.get_recommended_renderer_name(base_model)
    learning_rate = hp.learning_rate or hyperparam_utils.get_lr(base_model)

    data_dir = work_dir / "data"
    train_file = _write_conversations(corpus.train, data_dir / "train.jsonl")
    val_file = _write_conversations(corpus.val, data_dir / "val.jsonl") if corpus.val else None

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
    config = sft.Config(
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
        base_model,
        hp.lora_rank,
        learning_rate,
        renderer_name,
        corpus.n_train,
        corpus.n_val,
    )
    asyncio.run(sft.main(config))

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
