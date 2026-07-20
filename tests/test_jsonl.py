"""Tests for the shared tolerant JSONL reader."""

from __future__ import annotations

from stylebot.jsonl import iter_jsonl, read_jsonl


def _write(tmp_path, text):
    p = tmp_path / "x.jsonl"
    p.write_text(text, encoding="utf-8")
    return p


def test_skips_blank_and_undecodable_lines(tmp_path):
    p = _write(tmp_path, '{"a": 1}\n\nnot json\n{"b": 2}\n')
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]


def test_skips_non_object_json(tmp_path):
    p = _write(tmp_path, '{"a": 1}\n42\n[1, 2]\n')
    assert read_jsonl(p) == [{"a": 1}]


def test_keep_undecodable_wraps_raw(tmp_path):
    # The rewrite paths (ai-style log) must round-trip corrupt lines verbatim,
    # not drop them; non-object JSON gets the same treatment.
    p = _write(tmp_path, '{"a": 1}\nnot json\n42\n')
    assert read_jsonl(p, keep_undecodable=True) == [
        {"a": 1},
        {"_raw": "not json"},
        {"_raw": "42"},
    ]


def test_missing_file_yields_nothing(tmp_path):
    assert list(iter_jsonl(tmp_path / "absent.jsonl")) == []
