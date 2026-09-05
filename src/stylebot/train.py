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
- Per-passage multiplicity is a *policy hook* (`per_passage`), not a built-in
  gate: chunk budgets, re-key epochs and replicates all stack pairs on one
  passage, and whether that up-weights it is the caller's call. What assembly
  owes the caller is that the grouping be **true** — `passage_groups`, and
  D43 for the measurement that showed exact-body grouping was not.
- Weight is integer replication, applied to the TRAIN file only
  (`pair_weight` -> `meta.weight` -> N copies in `_write_conversations`). Val
  stays one row per pair, or the held-out loss is a weighted average of a
  distribution nobody chose.

**The manifest is the reproducibility record.** Weights and corpus never enter
git; the small JSON `write_manifest` writes (corpus and view hashes, split,
hyperparameters, tinker paths, cost) is committed — never by a dry run.

Needs the ``trainer`` extra for the paid path: ``uv add 'stylebot[trainer]'``.
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

from stylebot import draws
from stylebot.ai_core import style_system
from stylebot.cli_options import opt
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

        `passages` and `pair_weights` are the *other* scale question, and the
        one a budget sweep makes urgent: how many DISTINCT stretches of prose
        those pairs cover, and how many gradient steps each one buys. See
        `passage_groups`.
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
        """Effective per-passage multiplicity, train and val separately.

        `pairs_per_passage` is the histogram this item exists for: a corpus cut
        at four chunk budgets over the same prose reaches the trainer as N pairs
        on one passage, which is N gradient steps on that passage under a
        manifest claiming one epoch. Reported per split because the val split is
        by POST, which makes it honest about leakage and silent about this —
        both sides inflate alike, so val loss cannot see it.
        """
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
    """A record's training weight: `meta.weight`, defaulting to 1.

    Non-positive or unparseable values read as 1 rather than raising — a weight
    is an optimisation knob, and a corpus with a junk one should train, loudly
    uniform, rather than fail at the assembly step of a paid run.
    """
    raw = (rec.get("meta") or {}).get("weight")
    try:
        weight = int(raw)
    except (TypeError, ValueError):
        return 1
    return weight if weight > 0 else 1


def passage_groups(records: Sequence[dict]) -> list[list[dict]]:
    """Group pair records by SOURCE POST and character-span containment.

    A *passage* is a maximal stretch of prose: every record whose target body is
    a substring of some longer target from the same post joins that target's
    group. Groups come back in first-appearance corpus order, and each group's
    records keep corpus order, so the multiplicity hook sees a deterministic
    list whatever the input order.

    **Why containment and not equality.** Grouping by exact target body — what
    this did until 2026-09-03 — sees a passage cut at 1500 chars and the same
    prose cut at 3000 as two unrelated targets, because they are different
    strings. Measured on the live corpus that day: 3,981 styler-role train pairs
    sat on 3,533 distinct target bodies (1.13 pairs each, so exact-body grouping
    had almost nothing to group) but on only **982 passages** under this rule —
    a mean effective weight of 4.05 and a maximum of 15. The hook that was
    supposed to control multiplicity could not see 3.6 of every 4 duplicates.
    `_plans/decisions.md` D43.

    Substring containment is the right test rather than an offset interval
    because a pair record stores its target as text and never recorded where in
    the post it came from; the packer emits paragraph joins verbatim, so a
    shorter cut of the same prose IS a substring of the longer one. It costs
    O(spans²) string searches per post — a few thousand on the live corpus,
    milliseconds — and it never merges across posts, which is what would make it
    quadratic in the corpus.
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

    - `splits`: a loaded `splits.json`. When given, only **styler**-role pairs
      are kept — the detector pool and the frozen eval posts never enter
      training. When None, every pair is eligible (fixture/experiment use).
    - `selector`: pair-level policy hook (record -> keep?). Default keep-all —
      note this deliberately KEEPS near-copy pairs (see module docstring).
    - `per_passage`: multiplicity hook. Every pair covering one stretch of prose
      is passed as a list in corpus order (`passage_groups` — containment, not
      string equality); the hook returns the subset to keep (renamed from
      `per_target` on 2026-09-03, D43).
    - `system_prompt`: the prompt records are validated against (D64).
    - `pair_weight`: record -> positive int, stamped onto `meta.weight` and
      honoured as integer replication in the TRAIN file only. Default None
      leaves every record at 1, which is what every run to date did implicitly.
    - The val split shuffles the kept POSTs with `random.Random(seed)` and
      holds out `max(1, round(n * val_frac))` of them (0 with `val_frac=0`),
      so it is deterministic and disjoint by post.
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
    """Keyless order-of-magnitude token estimate (~4 chars/token) over every
    message of every record — for the dry-run cost preview only; the paid path
    counts real rendered tokens.

    Counts each record `meta.weight` times (default 1), because replication is
    what the train file does with the weight and the estimate is a *cost*
    estimate: a 3× weighted real-pair slice is 3× the tokens to pay for.
    """
    chars = sum(
        pair_weight_of(rec) * len(m.get("content") or "")
        for rec in records
        for m in rec.get("messages", ())
    )
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
            # The prompt the pairs were built against; a view's own overrides it.
            "style_system": style_system(),
            "style_system_sha": system_sha(),
            "splits_sha256": _splits_sha256(splits_path),
            "eval_version": (splits or {}).get("eval_version"),
            # Which RECORDS were kept, not merely which file: the selector, the
            # role filter and the multiplicity hook all run after `pairs_sha256`
            # is taken, so two runs over one corpus can train on different data.
            "record_ids_sha256": _record_ids_sha256(corpus),
            "n_train_posts": len(corpus.train_posts),
            "val_posts": corpus.val_posts,
            "filters": filters or {"selector": None, "per_passage": None, "pair_weight": None},
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


def format_passage_scale(chunk_scale: dict) -> str:
    """The per-passage weight line both `train` CLIs print on a dry run.

    Lives here rather than in either shell because `ai-style train` and
    livingthing's `dan-style train` must not drift on the one number that says
    how many times the run will walk over the same prose. Reads the manifest
    block, so it also renders an OLD manifest (pre-2026-09 ones have no
    `passages` key and say so) rather than only a live `TrainCorpus`.
    """
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

    Data: `pairs_path`, or the view of `draws_dir` under `policy`, which the
    manifest pins as `policy_sha`, `draws_sha256` and the `style_system` text
    (`draws.training_view`). `hp` is the fit; `val_frac` shapes the split as
    `data.val_posts`; `blog_root` + `splits_path` refuse a run whose pinned eval
    posts left the tree (D34). A dry run writes nothing and returns the manifest
    it would write; the paid run writes it pre-run and with `result`. `runner`
    is the seam `(corpus, manifest, work_dir, hp, *, download_adapter)`.
    """
    view_pins: dict = {}
    if draws_dir is not None:
        pairs_path, view_pins = draws.training_view(draws_dir, policy)
    elif pairs_path is None:
        raise ConfigError("run_training needs pairs_path or draws_dir")
    pairs_path = Path(pairs_path)
    work_dir = Path(work_dir)
    run_id = run_id or _default_run_id()
    hp = hp if hp is not None else Hyperparameters()

    splits = None
    if splits_path is not None:
        from stylebot import splits as splits_mod

        splits = splits_mod.load_splits(splits_path)
        if blog_root is not None:
            splits_mod.require_pinned_posts(splits, blog_root)

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


def _write_conversations(records: Sequence[dict], path: Path, *, replicate: bool = True) -> int:
    """Write records as bare `{"messages": ...}` JSONL — derived training data
    for the run's work dir (never committed). Returns the row count.

    Meta is stripped because the HF dataset layer wants a uniform schema and
    training only reads messages. **That strip is why `meta.weight` never did
    anything**: it was documented as a hook from the first draft and dropped one
    line before the trainer could see it. So weight is consumed HERE, as integer
    replication — `meta.weight` copies of the row — which is the only form of
    up-weighting the cookbook's `SupervisedDatasetFromHFDataset` can express
    without a per-example loss scale it has no field for. Replicas are separate
    rows, so the builder's epoch shuffle scatters them rather than presenting a
    passage N times in a row.

    `replicate=False` for the val file: a weighted held-out loss is an average
    over a distribution nobody chose, and val exists to be comparable across
    runs whose weights differ.
    """
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
        logger.info(
            "weighted train file: %d rows from %d pairs (%.2fx)",
            n_rows,
            corpus.n_train,
            n_rows / corpus.n_train,
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
