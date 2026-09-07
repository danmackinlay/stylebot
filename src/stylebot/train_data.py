"""The trainer's cookbook adapter (D91): an assembled corpus -> what tinker-cookbook consumes.

`write_conversations` writes the work-dir JSONL, one row per pair with its
`weight`; `scale_datum` applies that weight to a `Datum`'s per-token loss
weights (D90); `make_presplit_builder` is the by-post pre-split
`ChatDatasetBuilder` (D17). Cookbook imports stay inside the functions:
`--dry-run` works without the trainer extra.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from stylebot.jsonl import dumps_line


def pair_weight_of(rec: dict) -> float:
    """A record's loss scale: `meta.weight`, else 1.0 — a junk or non-positive
    value reads as 1.0 so a paid run's assembly never fails on a knob (D90)."""
    raw = (rec.get("meta") or {}).get("weight")
    try:
        weight = float(raw)
    except (TypeError, ValueError):
        return 1.0
    return weight if weight > 0 else 1.0


def write_conversations(records: Sequence[dict], path: Path, *, weighted: bool = True) -> int:
    """Write records as `{"messages": ..., "weight": w}` JSONL for the run's work
    dir; returns the row count. Meta is stripped, so `meta.weight` rides along
    as the row's loss scale (D90); `weighted=False` writes 1.0 for the val file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8") as fp:
        for rec in records:
            weight = pair_weight_of(rec) if weighted else 1.0
            fp.write(dumps_line({"messages": rec["messages"], "weight": weight}) + "\n")
            rows += 1
    return rows


def scale_datum(datum, weight: float):
    """Multiply a datum's per-token loss weights (normalised to sum 1 per
    example by the cookbook) by `weight`: the example's gradient scale (D90)."""
    if weight == 1.0:
        return datum
    import tinker

    scaled = datum.loss_fn_inputs["weights"].to_numpy() * weight
    datum.loss_fn_inputs["weights"] = tinker.TensorData.from_numpy(scaled)
    return datum


def make_presplit_builder(
    train_file: Path, val_file: Path | None, *, common_config, shuffle_seed: int
):
    """A ChatDatasetBuilder over our by-post pre-split files (D17), defined
    lazily because `chz` needs the trainer extra. `shuffle_seed` is required so
    the seed reaches the epoch shuffle (D23)."""
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
                datum = conversation_to_datum(
                    row["messages"], self.renderer, self.common_config.max_length, train_on_what
                )
                return scale_datum(datum, float(row.get("weight") or 1.0))

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
