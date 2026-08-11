"""Fakes shared by more than one ``test_synth_*`` module.

Split out of ``test_synth.py`` (2026-08-11) when that file was carved along its
section seams: these two helpers are the only ones with more than one caller.
Helpers used by exactly one module stay in that module.
"""

from __future__ import annotations

from stylebot import synth


def _fake(name: str) -> synth.Generator:
    # Deterministic "slop": tag the text so source != target but stays non-empty.
    return synth.Generator(name=name, generate=lambda text: f"[{name}-slop] {text}")


def _targets(n: int) -> list[synth.Target]:
    return [
        synth.Target(
            text=f"Paragraph {i}: " + "sufficiently long prose about abstraction. " * 3,
            source=f"post/{i % 3}.qmd",
            chunk_index=i,
            chunk_total=n,
        )
        for i in range(n)
    ]
