"""`stylebot.docscale` — a document row says which instrument chose its output.

`rewrite_documents_file` hands `best_of` and `scorer` through to the rewrite;
the row it writes stamps `meta.reranker` with the scorer's D39 identity whenever
a scorer was armed — at `best_of == 1` it still picks input-vs-sample under the
D20 guard — and `null` when nothing chose (C5).
"""

from __future__ import annotations

import json

import pytest

from stylebot import docscale, infer


def _archive(data_dir, capture_id, source, before, after):
    d = data_dir / "captures" / capture_id
    d.mkdir(parents=True)
    (d / "before.md").write_text(before, encoding="utf-8")
    (d / "after.md").write_text(after, encoding="utf-8")
    (d / "meta.json").write_text(
        json.dumps(
            {
                "capture_id": capture_id,
                "source": source,
                "snapped_at": "2026-01-01T00:00:00+00:00",
                "whole": True,
            }
        ),
        encoding="utf-8",
    )


def _fake_rewrite(text, styler, **kw):
    return type("R", (), {"text": text.upper(), "n_chunks": 1, "n_kept_input": 0})()


def _fake_detector(text):
    return {"score": 0.5, "configured": True}


_fake_detector.identity = {
    "name": "stub",
    "embed_model": "e",
    "git_sha": "abc",
    "trained_at": "2026-01-01",
    "n_pairs": 10,
}


def _one_document(tmp_path, **rewrite_kwargs):
    data_dir = tmp_path / "c"
    data_dir.mkdir()
    pairs = data_dir / "pairs.jsonl"
    pairs.write_text("", encoding="utf-8")
    _archive(data_dir, "cap-1", "post/a.qmd", "Before doc.\n", "After doc.\n")
    out = tmp_path / "documents.jsonl"
    result = docscale.rewrite_documents_file(
        pairs, out, styler=object(), rewrite=_fake_rewrite, **rewrite_kwargs
    )
    assert result.written == 1
    return json.loads(out.read_text().splitlines()[0])


def test_the_document_row_carries_the_served_prompt_and_hands_it_to_rewrite(tmp_path):
    from stylebot.ai_core import STYLE_SYSTEM

    seen: dict = {}

    def rewrite(text, styler, **kw):
        seen.update(kw)
        return _fake_rewrite(text, styler)

    data_dir = tmp_path / "mine"
    data_dir.mkdir()
    pairs = data_dir / "pairs.jsonl"
    pairs.write_text("", encoding="utf-8")
    _archive(data_dir, "cap-1", "post/a.qmd", "Before doc.\n", "After doc.\n")
    out = tmp_path / "mine.jsonl"
    docscale.rewrite_documents_file(pairs, out, styler=object(), rewrite=rewrite, system="served")
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["messages"][0]["content"] == seen["system"] == "served"
    assert _one_document(tmp_path)["messages"][0]["content"] == STYLE_SYSTEM


def test_a_best_of_1_document_row_carries_reranker_null(tmp_path):
    rec = _one_document(tmp_path)
    assert "reranker" in rec["meta"]
    assert rec["meta"]["reranker"] is None


def test_a_best_of_3_document_row_carries_the_reranker_identity(tmp_path):
    scorer = infer.detector_scorer(_fake_detector)
    rec = _one_document(tmp_path, best_of=3, scorer=scorer)
    assert rec["meta"]["reranker"] == _fake_detector.identity
    assert rec["meta"]["scale"] == "document"


def test_a_best_of_1_row_with_a_scorer_still_names_the_reranker(tmp_path):
    """One sample is still a choice: the scorer decides input vs sample (D20)."""
    scorer = infer.detector_scorer(_fake_detector)
    rec = _one_document(tmp_path, best_of=1, scorer=scorer)
    assert rec["meta"]["reranker"] == _fake_detector.identity


# --- `document_pass`: the `eval --documents` half as a library (D78) --------


def _shouty_styler(messages, num_samples=1):
    return [messages[-1]["content"].upper()] * num_samples


def _emitter():
    lines: list[tuple[str, str]] = []
    return (lambda message, /, *, level="info": lines.append((level, message))), lines


def test_document_pass_refuses_a_role_with_no_splits_contract(tmp_path):
    from stylebot.errors import ConfigError
    from stylebot.verdict import DocumentOpts

    with pytest.raises(ConfigError) as excinfo:
        docscale.document_pass(
            tmp_path / "pairs.jsonl",
            tmp_path / "scores.jsonl",
            DocumentOpts(documents=True),
            ("slop", "target"),
            manifest={},
            splits=None,
        )
    assert excinfo.value.param_hint == "--documents-role" and "eval" in str(excinfo.value)


def test_document_pass_hands_one_diet_to_both_halves(tmp_path, monkeypatch):
    """The styler is built from the manifest alone; the rewrite and the scoring
    that follows share `chunk_scale`, `div_is_prose` included."""
    from stylebot import backends
    from stylebot.verdict import DocumentOpts

    data_dir = tmp_path / "c"
    data_dir.mkdir()
    pairs = data_dir / "pairs.jsonl"
    pairs.write_text("", encoding="utf-8")
    _archive(data_dir, "cap-1", "post/a.qmd", "Before doc, prose enough.\n", "After doc.\n")
    built = []
    monkeypatch.setattr(backends, "build_styler", lambda opts: built.append(opts) or _shouty_styler)
    emit, lines = _emitter()
    hook = lambda attrs: "callout-" in attrs  # noqa: E731 — the blog's policy, in one line
    opts = DocumentOpts(documents=True, documents_role="all")

    passed = docscale.document_pass(
        pairs,
        data_dir / "scores.jsonl",
        opts,
        ("slop", "target"),
        manifest={"run_id": "test-run"},
        splits=None,
        div_is_prose=hook,
        emit=emit,
    )

    assert built[0].backend == "tinker"
    assert passed.pairs_path == data_dir / "documents.jsonl"  # beside `out`
    assert passed.fields == ("slop", "target", "output") and passed.prune_orphans is False
    assert passed.chunk_scale["div_is_prose"] is hook
    assert set(passed.chunk_scale) == {
        "max_chunk_chars",
        "inline_headings",
        "div_is_prose",
        "html_comments",
    }
    record = json.loads(passed.pairs_path.read_text().splitlines()[0])
    assert record["output"] == "BEFORE DOC, PROSE ENOUGH.\n"
    rewrote = f"rewrote 1 document(s) (1 chunks, 0 kept verbatim) -> {passed.pairs_path}"
    assert ("info", rewrote) in lines
    assert [m for lvl, m in lines if lvl == "status"][0].startswith(
        "rewriting whole all-role documents with run test-run at "
    )
