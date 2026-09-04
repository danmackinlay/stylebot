"""The publishable artifact — a `scores.jsonl` projected to a tidy CSV.

A scores file nests and is mostly Vale's per-alert `raw` objects, so it is not
a shape you can commit next to a blog post. `covariate_table` flattens it to
one row per pair with no prose text: small enough to version, sufficient to
redraw every figure.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
from collections.abc import Iterable
from pathlib import Path

from stylebot.errors import RunError
from stylebot.eval import SCHEMA_VERSION
from stylebot.pairs import GEN_FACET_KEYS
from stylebot.provenance import file_sha256
from stylebot.scoring import load_scores


def _flatten_scores(scores: dict) -> dict:
    """`{"slop": {"detector": {"score": .4}}}` -> `{"slop.detector.score": .4}`.

    Scalars only. Nested payloads (notably Vale's per-alert `raw` list) are
    dropped: they are the bulk of the file and none of its analysable signal.
    """
    flat: dict = {}
    for field_name, signals in (scores or {}).items():
        for signal, value in (signals or {}).items():
            if value is None or isinstance(value, (str, int, float, bool)):
                flat[f"{field_name}.{signal}"] = value
            elif isinstance(value, dict):
                for key, inner in value.items():
                    if inner is None or isinstance(inner, (str, int, float, bool)):
                        flat[f"{field_name}.{signal}.{key}"] = inner
    return flat


def covariate_table(
    scores: str | Path | Iterable[dict], *, drop_constant: bool = True
) -> tuple[list[dict], dict]:
    """Project a `scores.jsonl` to tidy per-pair rows -> `(rows, constants)`.

    Each row is `id` + the carried `meta` covariates + one column per scalar
    signal. No prose text: the passages are ~95% of a corpus's bytes, contribute
    to no figure, and for a synthetic sweep are regenerable besides.

    `drop_constant` hoists columns that never vary into the returned
    `constants` dict rather than repeating them on every row — an embedder id
    or a detector name describes the *run*, not the pair, and that repetition
    is ~40% of the file on a real corpus.
    """
    records = load_scores(scores)
    rows: list[dict] = []
    for rec in records:
        row: dict = {"id": rec.get("id")}
        row.update(rec.get("meta") or {})
        row.update(_flatten_scores(rec.get("scores") or {}))
        rows.append(row)

    # Column order: id, generation covariates in schema order, then the rest
    # alphabetically — design columns before signal columns.
    seen = {k for row in rows for k in row}
    ordered = ["id", *(k for k in GEN_FACET_KEYS if k in seen)]
    ordered += sorted(seen - set(ordered))

    constants: dict = {}
    if drop_constant:
        for col in list(ordered):
            if col == "id":
                continue
            values = {row.get(col) for row in rows}
            if len(values) == 1 and len(rows) > 1:
                constants[col] = values.pop()
                ordered.remove(col)

    # Project every row onto the same column list, so a row missing a key
    # carries it as an explicit None: a rectangular CSV, not a ragged one.
    rows = [{k: row.get(k) for k in ordered} for row in rows]
    return rows, constants


def write_covariate_table(
    scores: str | Path,
    out_path: str | Path,
    *,
    extra: dict | None = None,
) -> dict:
    """Write the tidy table as CSV plus a `<stem>.meta.json` provenance sidecar.

    The sidecar records the hoisted constants, the row/column counts, and a
    digest of the scores file the table came from, so a future reader can tell
    whether a re-run is comparable or merely similar. `extra` carries the
    caller's experiment config (model list, splits role, sample fraction).
    """
    out_path = Path(out_path)
    rows, constants = covariate_table(scores)
    if not rows:
        raise RunError(f"no score records in {scores} — nothing to export")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    digest = file_sha256(scores) if Path(scores).exists() else None
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "exported_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "source": str(scores),
        "source_sha256": digest,
        "rows": len(rows),
        "columns": columns,
        "constant_columns": constants,
        **({"experiment": extra} if extra else {}),
    }
    sidecar_path = out_path.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "csv": out_path,
        "sidecar": sidecar_path,
        "rows": len(rows),
        "columns": columns,
        "constants": constants,
    }
