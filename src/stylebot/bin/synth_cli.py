"""Back-compat shim: the synth CLI kit now lives at `stylebot.synth_cli`.

It was 1000+ lines of library code sitting in the *executables* package, which
the "library-first, CLI second" rule says it never was: `run_synth` is a public
library entry point, `bin/` is for click wrappers. It moved out on 2026-08-10.

This module exists **only** so livingthing's `from stylebot.bin import
synth_cli` (in `src/livingthing/bin/train_targets.py`) keeps resolving. Delete
it once that import points at `stylebot.synth_cli`; nothing in stylebot itself
imports this path.
"""

from __future__ import annotations

from stylebot.synth_cli import (  # noqa: F401 — re-export surface
    BUNDLE_PARAMS,
    DEFAULT_BUNDLES,
    Arm,
    ChunkOpts,
    CorpusOpts,
    GenOpts,
    InspectOpts,
    SectionedCommand,
    SessionOpts,
    SubstrateOpts,
    build_generators,
    iter_arms,
    pop_bundles,
    run_synth,
    sort_key,
    synth_options,
)
from stylebot.synth_cli import __all__ as __all__
