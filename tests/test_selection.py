"""Tests for is_human_authored — the lone blog-specific seam.

The corpus's quality depends on this picking pure-human prose correctly, so
the conservative "unknown means exclude" behaviour is locked in here."""

from __future__ import annotations

from datetime import date, datetime

from stylebot.lib import is_human_authored, is_modified_after


def test_automation_zero_is_human():
    assert is_human_authored({"automation": 0}) is True


def test_higher_automation_is_not_human():
    assert is_human_authored({"automation": 2}) is False


def test_missing_field_is_conservative_exclude():
    # Absence means "unknown"; we exclude rather than risk AI-touched text.
    assert is_human_authored({}) is False
    assert is_human_authored({"title": "x"}) is False


def test_unparseable_field_excluded():
    assert is_human_authored({"automation": "maybe"}) is False
    assert is_human_authored({"automation": None}) is False


def test_string_digit_parses():
    # YAML may hand us "0" rather than 0.
    assert is_human_authored({"automation": "0"}) is True


def test_retargetable_field_and_threshold():
    # Another blog can point it at its own marker / tolerance.
    assert is_human_authored({"ai_level": 1}, field="ai_level", max_level=1) is True
    assert is_human_authored({"ai_level": 2}, field="ai_level", max_level=1) is False


def test_modified_after_iso_string():
    # Real blog frontmatter: quoted ISO-8601 string with a timezone offset.
    assert is_modified_after({"date-modified": "2025-11-22T15:10:01+11:00"}) is True
    assert is_modified_after({"date-modified": "2019-06-01T00:00:00+10:00"}) is False
    # Boundary is inclusive (on or after).
    assert is_modified_after({"date-modified": "2021-01-01"}) is True


def test_modified_after_date_object():
    # YAML may parse an unquoted date into a date/datetime object.
    assert is_modified_after({"date-modified": date(2022, 3, 1)}) is True
    assert is_modified_after({"date-modified": datetime(2020, 12, 31, 9, 0)}) is False


def test_modified_after_missing_or_malformed_excluded():
    assert is_modified_after({}) is False
    assert is_modified_after({"date-modified": None}) is False
    assert is_modified_after({"date-modified": "not-a-date"}) is False
    assert is_modified_after({"date-modified": "9999"}) is False  # no YYYY-MM-DD prefix


def test_modified_after_retargetable_field_and_threshold():
    assert is_modified_after({"updated": "2024-01-01"}, field="updated", after="2023-01-01") is True
    assert is_modified_after({"updated": "2022-01-01"}, field="updated", after="2023-01-01") is False
