"""`stylebot.features`: the per-text vector, the pooled facets, the grouped AUC (D88)."""

from __future__ import annotations

import random

import pytest
from click.testing import CliRunner

from stylebot import features as ft
from stylebot.ai_core import STYLE_SYSTEM
from stylebot.bin.ai_style import main as ai_style
from stylebot.errors import ConfigError
from stylebot.jsonl import dumps_line

# One marker of each kind, placed so every count is known: three sentences
# (a fourth, "So what.", is a two-word fragment and dropped).
FIXTURE = (
    "Although the first sentence concedes something, it still argues, which is why "
    "it runs long. The second sentence is plain — it has a dash; it also has a "
    "semicolon. Yes, the third one might object, rather than agree. So what."
)


def test_syntax_profile_counts_each_marker_once():
    counts = ft.syntax_counts(FIXTURE)
    assert counts["sentences"] == 3
    assert counts["commas"] == 4
    # although, which, rather than / which is why, one might / although, yes,
    assert counts["subordinators"] == 3
    assert counts["connectives"] == 2
    assert counts["concessions"] == 2
    assert counts["em_dash"] == 1 and counts["semicolon"] == 1
    prof = ft.syntax_profile(FIXTURE)
    assert prof["commas"] == pytest.approx(4 / 3)
    assert prof["words_per_sentence"] == pytest.approx(counts["words"] / 3)
    assert set(prof) == set(ft.FEATURES)
    assert ft.syntax_profile("") == dict.fromkeys(ft.FEATURES, 0.0)


def test_markers_match_whole_phrases_only():
    assert ft.MARKERS["subordinators"].findall("whichever sandwich, so thatch") == []
    assert ft.MARKERS["concessions"].findall("Yes, and true, and of course.") == [
        "Yes,",
        "true,",
        "of course",
    ]


def _record(text: str, *, source: str, strategy: str | None) -> dict:
    meta = {"source": source, "captured_at": "2026-09-07T00:00:00+00:00"}
    if strategy:
        meta.update(synthetic=True, slop_strategy=strategy, gen={"reasoning_effort": "off"})
    return {
        "messages": [
            {"role": "system", "content": STYLE_SYSTEM},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "Plain target prose here, unchanged."},
        ],
        "meta": meta,
    }


ARGUED = "Although it is true, the point holds, because the alternative fails, which is why."
PLAIN = "The point holds. The alternative fails. That is all there is to it."


def _write(path, records):
    path.write_text("".join(dumps_line(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.fixture
def six_rows(tmp_path):
    rows = [
        _record(ARGUED, source="post/a.qmd", strategy=None),
        _record(PLAIN, source="post/b.qmd", strategy=None),
        _record(ARGUED, source="post/a.qmd", strategy="measured"),
        _record(ARGUED, source="post/c.qmd", strategy="measured"),
        _record(PLAIN, source="post/b.qmd", strategy="casual"),
        _record(PLAIN, source="post/c.qmd", strategy="casual"),
    ]
    return _write(tmp_path / "pairs.jsonl", rows)


def test_profile_pairs_pools_per_facet_with_real_first(six_rows):
    table = ft.profile_pairs(six_rows, by="slop_strategy")
    assert list(table) == ["real", "casual", "measured"]
    assert [table[k]["n"] for k in table] == [2, 2, 2]
    # Pooled, not a median of per-record rates: real is one argued + one plain text.
    argued, plain = ft.syntax_counts(ARGUED), ft.syntax_counts(PLAIN)
    n = argued["sentences"] + plain["sentences"]
    assert table["real"]["subordinators"] == pytest.approx(
        (argued["subordinators"] + plain["subordinators"]) / n
    )
    assert table["measured"]["subordinators"] == pytest.approx(
        ft.syntax_profile(ARGUED)["subordinators"]
    )
    assert table["casual"]["connectives"] == 0.0
    # No key: one synthetic row. A dotted key reads gen.*.
    assert list(ft.profile_pairs(six_rows)) == ["real", "synthetic"]
    assert list(ft.profile_pairs(six_rows, by="gen.reasoning_effort")) == ["real", "off"]


def test_same_sources_as_real_drops_synthetic_rows_on_other_posts(six_rows):
    table = ft.profile_pairs(six_rows, by="slop_strategy", same_sources_as_real=True)
    assert {k: v["n"] for k, v in table.items()} == {"real": 2, "casual": 1, "measured": 1}


def test_profile_pairs_pools_several_corpora_and_refuses_none(six_rows, tmp_path):
    extra = _write(
        tmp_path / "scratch.jsonl", [_record(PLAIN, source="post/z.qmd", strategy="argued")]
    )
    table = ft.profile_pairs([six_rows, extra], by="slop_strategy")
    assert table["argued"]["n"] == 1 and table["real"]["n"] == 2
    with pytest.raises(ConfigError):
        ft.profile_pairs([])
    with pytest.raises(ConfigError):
        ft.profile_pairs(six_rows, field="nope")


def _separable_corpus(tmp_path, *, separable: bool, n: int = 40):
    """`n` real and `n` facet records over 8 posts each. Separable: the real side
    argues and the facet side is plain; otherwise both sides draw from one bag."""
    rng = random.Random(0)
    argued = [
        "Although the {0} is slow, it works, which is why we keep it, because nothing else does.",
        "The {0} fails unless it is warm, though it rarely is, so that the fix is obvious.",
        "Granted, the {0} is odd; while it runs, it runs, whereas the other never did.",
    ]
    plain = [
        "The {0} is slow. It works. We keep it.",
        "The {0} fails when cold. It is rarely warm. The fix is easy.",
        "The {0} is odd. It runs. The other never did.",
    ]
    words = ["pump", "valve", "gear", "lamp", "belt", "fan", "rail", "cog", "pin", "nut"]

    def text(bag):
        return " ".join(rng.choice(bag).format(rng.choice(words)) for _ in range(3))

    both = argued + plain
    rows = []
    for i in range(n):
        rows.append(
            _record(text(argued if separable else both), source=f"post/r{i % 8}.qmd", strategy=None)
        )
        rows.append(
            _record(text(plain if separable else both), source=f"post/s{i % 8}.qmd", strategy="x")
        )
    return _write(tmp_path / f"{separable}.jsonl", rows)


def test_separability_is_high_on_separable_and_chance_on_inseparable_facets(tmp_path):
    pytest.importorskip("sklearn")
    apart = ft.separability(_separable_corpus(tmp_path, separable=True), "x")
    assert apart["auc"] > 0.95
    assert apart["n_real"] == apart["n_facet"] == 40 and apart["n_splits"] == 5
    same = ft.separability(_separable_corpus(tmp_path, separable=False), "x")
    assert 0.3 < same["auc"] < 0.7


def test_separability_on_an_embedding_takes_the_vector_it_is_handed(tmp_path):
    """`embed` replaces the proxy vector: a one-hot of which bag a text came from
    separates perfectly; a constant vector cannot separate at all."""
    pytest.importorskip("sklearn")
    import numpy as np

    corpus = _separable_corpus(tmp_path, separable=True)
    seen: list[int] = []

    def one_hot(texts):
        seen.append(len(texts))
        return np.array(
            [[1.0, 0.0] if "Although" in s or "Granted" in s else [0.0, 1.0] for s in texts]
        )

    apart = ft.separability(corpus, "x", embed=one_hot)
    assert apart["auc"] > 0.95 and apart["vector"] == "embed" and seen == [80]
    rng = np.random.default_rng(0)
    noise = ft.separability(corpus, "x", embed=lambda texts: rng.standard_normal((len(texts), 3)))
    assert 0.3 < noise["auc"] < 0.7
    # The memoised embedder encodes each distinct text once across calls.
    calls: list[int] = []

    def fake_embed_texts(texts, model_id):
        calls.append(len(texts))
        return np.ones((len(texts), 2))

    embed = ft.memo_embedder("fake")
    ft.embed_texts, real_embed = fake_embed_texts, ft.embed_texts
    try:
        embed(["a", "b"])
        assert embed(["b", "a", "c"]).shape == (3, 2)
    finally:
        ft.embed_texts = real_embed
    assert calls == [2, 1]


def test_separability_refuses_a_side_on_fewer_than_two_posts(six_rows):
    pytest.importorskip("sklearn")
    with pytest.raises(ConfigError, match="2 posts a side"):
        ft.separability(six_rows, "measured", same_sources_as_real=True)
    # Two posts a side pass the first check and fail the second: no fold can train on both.
    with pytest.raises(ConfigError, match="too few posts"):
        ft.separability(six_rows, "measured")


def test_features_cli_prints_the_table_over_pairs_and_draws(six_rows, tmp_path):
    result = CliRunner().invoke(
        ai_style, ["features", "--pairs", str(six_rows), "--by", "slop_strategy"]
    )
    assert result.exit_code == 0, result.output
    header, _, real, *rest = result.output.strip().splitlines()
    assert header.startswith("| stratum | N | words/sent |")
    assert real.startswith("| real | 2 |") and len(rest) == 2
    # No corpus named at all is a usage error, not a traceback.
    result = CliRunner().invoke(ai_style, ["features"])
    assert result.exit_code == 2 and "no corpus named" in result.output
