"""`stylebot.docscale` — a document row says which instrument chose its output.

`rewrite_documents_file` hands `best_of` and `scorer` through to the rewrite;
the row it writes stamps `meta.reranker` with the scorer's D39 identity when
`best_of > 1`, and `null` otherwise (C5).
"""

from __future__ import annotations

import json

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


def test_a_best_of_1_document_row_carries_reranker_null(tmp_path):
    rec = _one_document(tmp_path)
    assert "reranker" in rec["meta"]
    assert rec["meta"]["reranker"] is None


def test_a_best_of_3_document_row_carries_the_reranker_identity(tmp_path):
    scorer = infer.detector_scorer(_fake_detector)
    rec = _one_document(tmp_path, best_of=3, scorer=scorer)
    assert rec["meta"]["reranker"] == _fake_detector.identity
    assert rec["meta"]["scale"] == "document"
