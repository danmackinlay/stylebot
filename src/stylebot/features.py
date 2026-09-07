"""Syntactic proxies over one side of a corpus, pooled per stratum, and how
separable two strata are on them (D88; `_plans/argued-slop-arm.md` stage 0).

`syntax_profile` is the per-text vector; `profile_pairs` pools counts per
facet, the real stratum being `meta.synthetic` falsy; `separability` is an
out-of-fold AUC of a logistic head on the punctuation-free vector, real vs
one facet, folds grouped by `meta.source`. `ai-style features` is the shell.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import click

from stylebot.cli_options import opt
from stylebot.draws import ViewOpts
from stylebot.errors import ConfigError, MissingExtra
from stylebot.pairs import FIELD_EXTRACTORS, iter_pairs

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
#: A sentence of fewer words than this is a fragment (a list stub, a heading) and is dropped.
MIN_SENTENCE_WORDS = 3

SUBORDINATORS = (
    "which", "although", "though", "while", "whereas", "because", "since", "unless",
    "whether", "so that", "rather than", "in that", "given that", "such that",
)  # fmt: skip
CONNECTIVES = (
    "which is why", "which is exactly", "which is precisely", "the catch is",
    "that is the", "the problem is", "not because", "one might", "the obvious",
)  # fmt: skip
CONCESSIONS = (
    "although", "though", "while", "granted", "admittedly", "to be fair", "of course",
    "yes,", "true,",
)  # fmt: skip


def _phrase_re(phrases: Sequence[str]) -> re.Pattern[str]:
    alts = "|".join(re.escape(p) for p in phrases)
    return re.compile(rf"(?<!\w)(?:{alts})(?!\w)", re.IGNORECASE)


MARKERS: dict[str, re.Pattern[str]] = {
    "commas": re.compile(","),
    "subordinators": _phrase_re(SUBORDINATORS),
    "em_dash": re.compile("—|--"),
    "semicolon": re.compile(";"),
    "connectives": _phrase_re(CONNECTIVES),
    "concessions": _phrase_re(CONCESSIONS),
}
#: The profile's columns, in table order.
FEATURES = ("words_per_sentence", "sentence_length_var", *MARKERS)
#: The separability vector: no em-dash or semicolon, the classifier's cheap lexical shortcut.
SEPARABILITY_FEATURES = (
    "subordinators", "connectives", "concessions", "words_per_sentence",
    "sentence_length_var", "commas",
)  # fmt: skip
REAL = "real"


def sentences(text: str) -> list[str]:
    """`text` split at sentence-final punctuation, fragments dropped."""
    return [s for s in SENTENCE_SPLIT.split(text) if len(s.split()) >= MIN_SENTENCE_WORDS]


def syntax_counts(text: str) -> dict[str, float]:
    """Raw counts (`sentences`, `words`, `words_sq`, one per marker): what pools."""
    sents = sentences(text)
    lengths = [len(s.split()) for s in sents]
    counts = {
        "sentences": len(sents),
        "words": sum(lengths),
        "words_sq": sum(n * n for n in lengths),
    }
    for name, pattern in MARKERS.items():
        counts[name] = sum(len(pattern.findall(s)) for s in sents)
    return counts


def rates(counts: dict[str, float]) -> dict[str, float]:
    """Per-sentence rates of pooled counts; all zero when there is no sentence."""
    n = counts["sentences"]
    if not n:
        return dict.fromkeys(FEATURES, 0.0)
    mean = counts["words"] / n
    out = {"words_per_sentence": mean, "sentence_length_var": counts["words_sq"] / n - mean * mean}
    out.update({name: counts[name] / n for name in MARKERS})
    return out


def syntax_profile(text: str) -> dict[str, float]:
    """`rates(syntax_counts(text))`: the per-text feature vector."""
    return rates(syntax_counts(text))


def is_real(record: dict) -> bool:
    return not (record.get("meta") or {}).get("synthetic")


def meta_value(record: dict, key: str) -> object:
    """`meta[key]`, dotted for nesting (`gen.reasoning_effort`); None when absent."""
    node: object = record.get("meta") or {}
    for part in key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    return node


def facet_of(record: dict, by: str | None) -> str:
    """The stratum label: `real`, else `meta[by]` (`synthetic` with no key)."""
    if is_real(record):
        return REAL
    if by is None:
        return "synthetic"
    value = meta_value(record, by)
    return "(none)" if value is None else str(value)


def read_records(paths: Iterable[str | Path]) -> list[dict]:
    """Every pair record of every corpus named, in order; none named is a refusal."""
    paths = list(paths)
    if not paths:
        raise ConfigError("no corpus named: pass --draws, or one or more --pairs")
    return [rec for path in paths for rec in iter_pairs(path)]


def _load(
    paths: Iterable[str | Path] | str | Path, field: str, same_sources: bool
) -> list[tuple[dict, str]]:
    """`(record, text of field)` per record; `same_sources` keeps only synthetic
    records on a post the real stratum covers."""
    try:
        extract = FIELD_EXTRACTORS[field]
    except KeyError:
        raise ConfigError(f"unknown field {field!r}; one of {sorted(FIELD_EXTRACTORS)}") from None
    records = read_records([paths] if isinstance(paths, (str, Path)) else paths)
    if same_sources:
        sources = {meta_value(r, "source") for r in records if is_real(r)}
        records = [r for r in records if is_real(r) or meta_value(r, "source") in sources]
    return [(rec, extract(rec)) for rec in records]


def profile_pairs(
    paths: Iterable[str | Path] | str | Path,
    *,
    field: str = "slop",
    by: str | None = None,
    same_sources_as_real: bool = False,
) -> dict[str, dict[str, float]]:
    """Facet label -> `{"n": records, **rates}`, counts pooled over `field` of every
    record; `same_sources_as_real` keeps only synthetic records on a post the real
    stratum covers. `real` sorts first, then labels alphabetically."""
    pooled: dict[str, dict[str, float]] = {}
    for rec, text in _load(paths, field, same_sources_as_real):
        bucket = pooled.setdefault(facet_of(rec, by), {"n": 0})
        bucket["n"] += 1
        for key, value in syntax_counts(text).items():
            bucket[key] = bucket.get(key, 0) + value
    return {
        label: {"n": bucket["n"], **rates(bucket)}
        for label, bucket in sorted(pooled.items(), key=lambda kv: (kv[0] != REAL, kv[0]))
    }


_SKLEARN_HINT = (
    "separability needs scikit-learn: install the 'classifier' extra "
    "(`uv add 'stylebot[classifier]'`)"
)


def separability(
    paths: Iterable[str | Path] | str | Path,
    facet_value: str,
    *,
    by: str | None = "slop_strategy",
    field: str = "slop",
    same_sources_as_real: bool = False,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    """Out-of-fold AUC of a logistic head on `SEPARABILITY_FEATURES`, real vs the
    records whose facet is `facet_value`; folds are grouped by `meta.source`
    (`StratifiedGroupKFold`, at most `n_splits`). Fewer than two posts a side
    is a refusal. Returns `{"auc", "n_real", "n_facet", "n_groups", "n_splits"}`."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingExtra(_SKLEARN_HINT) from exc

    rows, labels, groups = [], [], []
    for rec, text in _load(paths, field, same_sources_as_real):
        label = facet_of(rec, by)
        if label not in (REAL, facet_value) or not sentences(text):
            continue
        rows.append([syntax_profile(text)[k] for k in SEPARABILITY_FEATURES])
        labels.append(1 if label == REAL else 0)
        groups.append(str(meta_value(rec, "source")))
    y = np.asarray(labels)
    n_real, n_facet = int(y.sum()), int((y == 0).sum())
    posts_a_side = [
        len({g for g, lab in zip(groups, labels, strict=True) if lab == v}) for v in (1, 0)
    ]
    if min(posts_a_side, default=0) < 2:
        raise ConfigError(
            f"separability needs records on >= 2 posts a side; real {n_real} on "
            f"{posts_a_side[0]}, {facet_value!r} {n_facet} on {posts_a_side[1]}"
        )
    x = np.asarray(rows, dtype=float)
    splits = min(n_splits, len(set(groups)), n_real, n_facet)
    folds = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for train, test in folds.split(x, y, groups):
        if len(set(y[train])) < 2:
            continue
        head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        oof[test] = head.fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    scored = ~np.isnan(oof)
    if len(set(y[scored])) < 2:
        raise ConfigError(f"too few posts to cross-validate real vs {facet_value!r}")
    return {
        "auc": float(roc_auc_score(y[scored], oof[scored])),
        "n_real": n_real,
        "n_facet": n_facet,
        "n_groups": len(set(groups)),
        "n_splits": splits,
    }


def format_table(profiles: dict[str, dict[str, float]], aucs: dict[str, dict] | None = None) -> str:
    """The plan's markdown table: one row per facet, `AUC vs real` when given."""
    cols = ["stratum", "N", "words/sent", "sent-len var", *MARKERS]
    if aucs:
        cols.append("AUC vs real")
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for label, prof in profiles.items():
        cells = [label, str(prof["n"])] + [f"{prof[k]:.3f}" for k in FEATURES]
        if aucs:
            cells.append(f"{aucs[label]['auc']:.3f}" if label in aucs else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


@dataclass(frozen=True)
class FeaturesOpts:
    """The `features` surface: which corpora, which side, which facet."""

    cli_section = "Features"

    pairs_paths: tuple[Path, ...] = opt(
        (),
        "--pairs",
        "pairs_paths",
        multiple=True,
        type=click.Path(exists=True, path_type=Path),
        help="A corpus to profile (pairs.jsonl or a shard directory); repeatable, pooled with "
        "--draws' view — a scratch audition beside the live corpus.",
    )
    field: str = opt(
        "slop",
        "--field",
        type=click.Choice(sorted(FIELD_EXTRACTORS)),
        show_default=True,
        help="Which side of every pair to profile.",
    )
    by: str | None = opt(
        None,
        "--by",
        help="Facet the synthetic stratum by a meta key (dotted for gen.*), e.g. slop_strategy; "
        "real is always its own row.",
    )
    same_sources: bool = opt(
        False,
        "--same-sources",
        is_flag=True,
        help="Keep only synthetic records on posts the real stratum covers (the fair reference).",
    )
    separability: bool = opt(
        False,
        "--separability",
        is_flag=True,
        help="Add each facet's grouped-CV AUC against real (needs scikit-learn).",
    )


FEATURES_PARAMS: dict[str, type] = {"opts": FeaturesOpts, "view": ViewOpts}


def features_options(**kw):
    """The whole `features` surface, one decorator."""
    from stylebot.cli_options import bundle_options

    return bundle_options(list(FEATURES_PARAMS.values()), **kw)


def features_pass(opts: FeaturesOpts, view: ViewOpts) -> str:
    """The command body: the table over `--draws`' view plus every `--pairs`."""
    paths: list[Path] = [view.view()] if view.draws_dir is not None else []
    paths += list(opts.pairs_paths)
    kw = {"field": opts.field, "by": opts.by, "same_sources_as_real": opts.same_sources}
    profiles = profile_pairs(paths, **kw)
    aucs = None
    if opts.separability:
        aucs = {label: separability(paths, label, **kw) for label in profiles if label != REAL}
    return format_table(profiles, aucs)
