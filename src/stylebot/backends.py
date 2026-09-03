"""The styler backend surface: one description, one build site, one diet rule.

`BackendOpts` is *how to reach the styler* — the twelve flags `ai-style run`
and `ai-style serve` share (the sidecar's rewrite op is `run` held open), plus
`build_styler` which turns them into a `stylebot.infer.Styler`. This lived in
`bin/ai_style.py` until 2026-09-03, which made it unimportable as a library and
raising `click` exceptions; livingthing's `dan-style` had reimplemented the
construction, and the copy passed no `max_tokens`, so serving the 4500-char
cross-heading diet silently truncated every rewrite at the 2000-token floor.
Hence: mechanism here, click translation in `bin/`.

Three error kinds, and the CLI maps them to exit codes:

* `ValueError` — a flag combination the user must fix (click `UsageError`, 2).
* `RuntimeError` — a fine invocation against an unready environment (a missing
  key; click `ClickException`, 1).
* `ImportError` — an optional extra is not installed (also 1).

`serving_diet` is the other half: which chunk budget and heading mode the
*served adapter* was trained on, read off a training manifest. Serving at the
wrong diet is out-of-distribution and the styler falls back to copying its
input (observed 2026-07-22), so this follows the manifest rather than the
corpus's current policy — the two diverge the moment training policy moves
ahead of the promoted run. It came from livingthing's `chunk_diet`, which read
stylebot's own manifest schema from the wrong repo; the blog keeps only the
fallback constant it supplies as policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import click

from stylebot import config
from stylebot.cli_options import bundle_options, opt

# Minimum share of the training corpus a chunk budget needs before `serving_diet`
# will serve at it. One-sided on purpose — see the function.
MIN_SCALE_SUPPORT = 0.10


@dataclass(frozen=True)
class BackendOpts:
    """How to reach the styler — the surface `run` and `serve` share.

    Also a CLI option bundle (`cli_options.opt`), so the flags are declared
    once and both this repo's commands and livingthing's `dan-style` mirror
    build from the same declarations.
    """

    cli_section = "Styler backend"

    backend: str | None = opt(
        None,
        "--backend",
        type=click.Choice(["tinker", "openai"]),
        show_default=True,
        help="tinker: sample the trained checkpoint (no serving infra; needs TINKER_API_KEY + "
        "trainer extra). openai: any OpenAI-compatible endpoint (Fireworks, mlx_lm.server, "
        "Osaurus). `serve` defaults to neither: omit it there for score-only serving.",
    )
    sampler_path: str | None = opt(
        None,
        "--sampler-path",
        help="[tinker] tinker:// sampler checkpoint (from the training manifest "
        "result.checkpoints.sampler_path).",
    )
    base_model: str | None = opt(
        None,
        "--base-model",
        help="[tinker] Base model id the adapter was trained on (manifest "
        "hyperparameters.base_model).",
    )
    renderer_name: str | None = opt(
        None,
        "--renderer",
        "renderer_name",
        help="[tinker] Cookbook renderer used at training time (manifest result.renderer_name).",
    )
    base_url: str | None = opt(
        None,
        "--base-url",
        help="[openai] Endpoint base URL, e.g. http://localhost:8080/v1.",
    )
    model: str | None = opt(None, "--model", help="[openai] Served model name.")
    api_key_env: str | None = opt(
        None,
        "--api-key-env",
        help="[openai] Env var holding the endpoint key (e.g. FIREWORKS_API_KEY); omit for "
        "keyless local servers.",
    )
    temperature: float = opt(
        0.3,
        "--temperature",
        type=float,
        show_default=True,
        help="Sampling temperature. `serve` runs hotter (0.7) on purpose: interactive best-of "
        "wants DISTINCT candidates; at 0.3 the sampler mode-collapses to near-identical draws.",
    )
    max_tokens: int | None = opt(
        None,
        "--max-tokens",
        type=int,
        help="Output-token budget per chunk. [default: scaled off --max-chunk-chars, floor 2000]",
    )
    max_chunk_chars: int | None = opt(
        None,
        "--max-chunk-chars",
        type=int,
        help="Inference chunk budget; MUST match the adapter's training chunk scale. "
        "[default: the manifest's diet under --manifest, else 1500]",
    )
    # Tri-state, and that is the point: None means "nobody said", so a
    # --manifest may supply the diet, while an explicit --no-inline-headings
    # beats it. `headings()` is the resolved boolean every caller wants.
    inline_headings: bool | None = opt(
        None,
        "--inline-headings/--no-inline-headings",
        help="Match the chunk diet the adapter was TRAINED on (the training manifest records it "
        "under data.chunk_scale). On = the v2 cross-heading diet: headings are chunk content the "
        "styler may rewrite, and chunks span sections. Off = v1, headings pass through verbatim "
        "as context. Serving at the wrong diet is out-of-distribution and makes the styler fall "
        "back to copying its input. [default: off, or the manifest's diet under --manifest]",
    )
    detector_model: Path | None = opt(
        None,
        "--detector-model",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help="Voice-classifier artifact dir for --best-of reranking.",
    )

    def chunk_chars(self) -> int:
        """The inference chunk budget, defaulted from `infer`."""
        from stylebot import infer

        return self.max_chunk_chars or infer.STYLE_CHARS_PER_CHUNK

    def headings(self) -> bool:
        """The resolved heading mode (the tri-state field collapsed to a bool)."""
        return bool(self.inline_headings)

    def tokens(self) -> int:
        """Output-token budget. Fixed at backend construction, so it is scaled
        off the chunk budget — a big --max-chunk-chars would otherwise truncate
        a rewrite mid-chunk."""
        from stylebot import infer

        if self.max_tokens is not None:
            return self.max_tokens
        return infer.max_tokens_for(self.chunk_chars())


BACKEND_PARAMS: dict[str, type] = {"opts": BackendOpts}


def backend_options(**kw):
    """The shared `--backend …` flags. `defaults=`/`exclude=` as elsewhere."""
    return bundle_options([BackendOpts], **kw)


def build_styler(opts: BackendOpts):
    """Construct the styler backend described by `opts` — the one build site.

    Raises `ValueError` for a flag combination the user must fix, `RuntimeError`
    for an unready environment, and `ImportError` for a missing extra. The CLI
    translates those into exit codes; `serve` re-raises them as `ValueError` so
    the failing request gets an `{"error"}` instead of taking the sidecar down.
    """
    from stylebot import infer

    tokens = opts.tokens()
    if opts.backend is None:
        raise ValueError("no --backend given, so there is no styler to build")
    if opts.backend == "tinker":
        if not opts.sampler_path or not opts.base_model:
            raise ValueError(
                "--backend tinker needs --sampler-path and --base-model (see the training manifest)"
            )
        if not config.get_key("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is not set")
        return infer.tinker_backend(
            opts.sampler_path,
            opts.base_model,
            renderer_name=opts.renderer_name,
            temperature=opts.temperature,
            max_tokens=tokens,
        )
    if not opts.base_url or not opts.model:
        raise ValueError("--backend openai needs --base-url and --model")
    api_key = config.get_key(opts.api_key_env) if opts.api_key_env else None
    if opts.api_key_env and not api_key:
        raise RuntimeError(f"{opts.api_key_env} is not set")
    return infer.openai_backend(
        opts.base_url,
        opts.model,
        api_key=api_key,
        temperature=opts.temperature,
        max_tokens=tokens,
    )


# ---------------------------------------------------------------------------
# The served adapter's diet
# ---------------------------------------------------------------------------


def serving_diet(
    manifest: Mapping[str, Any] | None,
    *,
    default_chunk_chars: int | None = None,
    min_support: float = MIN_SCALE_SUPPORT,
) -> tuple[int, bool]:
    """`(max_chunk_chars, inline_headings)` the SERVED adapter was trained on.

    Read from `data.chunk_scale` in a training manifest (recorded since v2).
    A manifest predating that key — or none at all — falls back to
    `default_chunk_chars` (the caller's corpus policy; `infer`'s default if
    omitted), section-bounded, which is what those runs were trained on.

    Serve at the largest budget with REAL support in the diet. The asymmetry
    decides the tie-break: under-chunking merely costs some coherence, while
    over-chunking hits the identity fallback and returns the input verbatim. So
    a thin experimental arm (a handful of 4500s in an otherwise 1500 corpus)
    must not drag serving out to a length the adapter barely saw. The heading
    mode follows the same logic: inline only if the diet was *predominantly*
    cross-heading, so a stray v1 arm cannot flip the serving mode.
    """
    from stylebot import infer

    fallback = default_chunk_chars or infer.STYLE_CHARS_PER_CHUNK
    scale = ((manifest or {}).get("data") or {}).get("chunk_scale") or {}
    budgets = {k: v for k, v in (scale.get("budgets") or {}).items() if k != "None"}
    modes = scale.get("modes") or {}
    total = sum(budgets.values())
    supported = [int(k) for k, v in budgets.items() if total and v / total >= min_support]
    max_chars = max(supported, default=fallback)
    cross = modes.get("cross-heading", 0)
    inline = cross > sum(v for k, v in modes.items() if k != "cross-heading")
    return max_chars, inline


def opts_from_manifest(
    manifest: Mapping[str, Any] | None,
    opts: BackendOpts | None = None,
    *,
    default_chunk_chars: int | None = None,
) -> BackendOpts:
    """Fill `opts` from a training manifest, without overriding what was set.

    The manifest is the committed reproducibility record, so it — not a flag
    soup — is where the checkpoint, base model, renderer and serving diet come
    from. Fields the caller already pinned (a flag the user passed) win; only
    the unset ones are filled, which is why `inline_headings` is tri-state.
    """
    opts = opts or BackendOpts()
    manifest = manifest or {}
    result = manifest.get("result") or {}
    diet_chars, diet_inline = serving_diet(manifest, default_chunk_chars=default_chunk_chars)
    from_manifest = {
        "sampler_path": (result.get("checkpoints") or {}).get("sampler_path"),
        "base_model": (manifest.get("hyperparameters") or {}).get("base_model"),
        "renderer_name": result.get("renderer_name"),
        "max_chunk_chars": diet_chars,
        "inline_headings": diet_inline,
    }
    fill = {k: v for k, v in from_manifest.items() if v is not None and getattr(opts, k) is None}
    return replace(opts, **fill)
