"""Tests for is_human_authored — the lone blog-specific seam.

The corpus's quality depends on this picking pure-human prose correctly, so
the conservative "unknown means exclude" behaviour is locked in here."""

from __future__ import annotations

from stylebot.lib import is_human_authored


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
