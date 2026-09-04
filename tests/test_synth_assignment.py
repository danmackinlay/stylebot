"""Phase 2 arm assignment: which generator draws which target.

Carved verbatim out of ``test_synth.py``. Rendezvous (HRW) hashing over the
generator roster — set-stable, order-free, and it must not starve arms that
share a model name.
"""

from __future__ import annotations

from synth_helpers import _targets

from stylebot import synth


def test_hash_assignment_is_set_stable_and_position_free():
    # Each target's arm depends only on its own text (+ salt): dropping the
    # first target must not reassign the rest (round-robin would shift every
    # arm), and a different salt re-randomizes.
    targets = _targets(12)
    gens = [synth.Generator("a"), synth.Generator("b"), synth.Generator("c")]

    full = {t.text: g.name for t, g, _ in synth._assign(targets, gens, per_generator=False)}
    tail = {t.text: g.name for t, g, _ in synth._assign(targets[1:], gens, per_generator=False)}
    assert all(tail[text] == full[text] for text in tail)  # set-stable

    salted = {
        t.text: g.name
        for t, g, _ in synth._assign(targets, gens, per_generator=False, assign_seed="rep2")
    }
    assert salted != full  # a fresh replicate of the design


# --- rendezvous (HRW) assignment: roster stability -------------------------
#
# The 2026-08-10 switch from `hash(text) % n` to highest-random-weight. Under
# modulo, `n` was a length and the winner a position, so a reorder alone moved
# 100% of the corpus; these tests pin the properties that replaced it.


def _arms(n: int, *, name: str | None = None) -> list[synth.Generator]:
    """n distinct arms — same model name when `name` is given, differing strategy."""
    if name is not None:
        return [synth.Generator(name, strategy=f"s{i}") for i in range(n)]
    return [synth.Generator(f"openrouter/m{i}") for i in range(n)]


def _arm_of(targets, gens, **kw) -> dict[str, str]:
    return {
        t.text: synth._arm_identity(g)
        for t, g, _ in synth._assign(targets, gens, per_generator=False, **kw)
    }


def test_assignment_is_independent_of_generator_order():
    # The headline guarantee: the roster is a set, not a list. Reordering it
    # used to reassign every single target.
    import random

    targets = _targets(400)
    gens = _arms(6)
    shuffled = list(gens)
    random.Random(1234).shuffle(shuffled)
    assert shuffled != gens
    assert _arm_of(targets, shuffled) == _arm_of(targets, gens)


def test_dropping_one_arm_churns_about_one_nth():
    # Removing an arm may only move that arm's own share (~1/n). Modulo moved
    # ~83% here; the bound is generous (2/n) so this stays a regression guard,
    # not a distribution test.
    targets = _targets(3000)
    gens = _arms(6)
    before = _arm_of(targets, gens)
    after = _arm_of(targets, gens[:5])
    moved = sum(1 for text, arm in before.items() if after[text] != arm) / len(before)
    assert moved < 2 / 6, moved
    # ...and everything that moved was on the departed arm: nobody else churns.
    dropped = synth._arm_identity(gens[5])
    assert all(before[t] == dropped for t in before if after[t] != before[t])


def test_no_arm_starves_across_a_realistic_rotation():
    # The live shape: 6 models × 6 strategies = 36 arms, only 6 distinct names.
    targets = _targets(3600)
    gens = [
        synth.Generator(f"openrouter/m{m}", strategy=f"s{s}") for m in range(6) for s in range(6)
    ]
    counts: dict[str, int] = {}
    for _, g, _ in synth._assign(targets, gens, per_generator=False):
        ident = synth._arm_identity(g)
        counts[ident] = counts.get(ident, 0) + 1
    assert len(counts) == 36, len(counts)  # nobody starves
    assert min(counts.values()) >= 40, min(counts.values())  # ~100 each


def test_assign_seed_rerandomizes_and_reproduces():
    targets = _targets(400)
    gens = _arms(6)
    base = _arm_of(targets, gens)
    seeded = _arm_of(targets, gens, assign_seed="rep2")
    assert seeded != base
    assert _arm_of(targets, gens, assign_seed="rep2") == seeded  # same seed, same design
    # A re-randomised design is still order-free.
    assert _arm_of(targets, list(reversed(gens)), assign_seed="rep2") == seeded
