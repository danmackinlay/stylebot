"""The styler backend surface: one description, one build site, one diet rule.

`BackendOpts` is *how to reach the styler* — the twelve flags `ai-style run` and
`ai-style serve` share — plus `build_styler`, which turns them into a
`stylebot.infer.Styler`. Mechanism here, click translation in `bin/`: this lived
in the CLI module until livingthing had reimplemented the construction, and the
copy passed no `max_tokens` (D35). Errors follow the package vocabulary (D47).

`serving_diet` is the other half: which chunk budget and heading mode the
*served adapter* was trained on, read off a training manifest rather than off
the corpus's current policy. Serving at the wrong diet is out-of-distribution
and the styler falls back to copying its input, and the two diverge the moment
training policy moves ahead of the promoted run.

`RunOpts` and the two `run` passes (`rewrite_corpus`, `rewrite_document`) are
the command body as a library, so `dan-style run` calls them with its policy
hooks instead of retyping them (D78).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from stylebot import config
from stylebot.cli_options import Emitter, bundle_options, opt, silent
from stylebot.errors import ConfigError, RunError

if TYPE_CHECKING:
    from stylebot.infer import RewriteResult, Styler

logger = logging.getLogger(__name__)

# Minimum share of the training corpus a chunk budget needs before `serving_diet`
# will serve at it. One-sided on purpose — see the function.
MIN_SCALE_SUPPORT = 0.10

# Samples per chunk under `run`; 1 means nothing reranks and no scorer is needed.
DEFAULT_BEST_OF = 1

MANIFEST_HELP = (
    "A training-run manifest to serve: supplies --sampler-path, --base-model, --renderer and "
    "the chunk diet (--max-chunk-chars/--inline-headings) the adapter was TRAINED on. Explicit "
    "flags win over it. Serving at the wrong diet is out-of-distribution — the styler falls back "
    "to copying its input — so prefer this over retyping four flags."
)


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

    Raises `ConfigError` for a flag combination the caller must fix, `RunError`
    for an unready world (D47); the extras raise `MissingExtra`/`ImportError`.
    """
    from stylebot import infer

    tokens = opts.tokens()
    if opts.backend is None:
        raise ConfigError("no --backend given, so there is no styler to build")
    if opts.backend == "tinker":
        if not opts.sampler_path or not opts.base_model:
            raise ConfigError(
                "--backend tinker needs --sampler-path and --base-model (see the training manifest)"
            )
        if not config.get_key("TINKER_API_KEY"):
            raise RunError("TINKER_API_KEY is not set")
        return infer.tinker_backend(
            opts.sampler_path,
            opts.base_model,
            renderer_name=opts.renderer_name,
            temperature=opts.temperature,
            max_tokens=tokens,
        )
    if not opts.base_url or not opts.model:
        raise ConfigError("--backend openai needs --base-url and --model")
    api_key = config.get_key(opts.api_key_env) if opts.api_key_env else None
    if opts.api_key_env and not api_key:
        raise RunError(f"{opts.api_key_env} is not set")
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

    Serve at the largest budget with REAL support in the diet: under-chunking
    costs some coherence, while over-chunking hits the identity fallback and
    returns the input verbatim, so a thin experimental arm must not drag serving
    out to a length the adapter barely saw. Heading mode follows the same logic.
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


def manifest_renderer(manifest: Mapping[str, Any] | None) -> str | None:
    """The renderer the manifest's run actually trained under.

    `result.renderer_name` is what the run resolved (the cookbook recommendation
    when the flag was None); `hyperparameters.renderer_name` is what was asked
    for, and is all a dry-run manifest has. One reader, so the two spellings
    cannot drift between the filler and the checker below.
    """
    manifest = manifest or {}
    return ((manifest.get("result") or {}).get("renderer_name")
            or (manifest.get("hyperparameters") or {}).get("renderer_name"))


def manifest_mismatches(
    manifest: Mapping[str, Any] | None, opts: BackendOpts | None = None
) -> list[str]:
    """Ways the styler about to be built disagrees with the manifest it claims.

    A manifest is a *pin*, and three of the things it pins are silent when
    violated: the base model, the chat renderer, and the system prompt each
    produce plausible garbage or an out-of-distribution adapter rather than an
    error (D48). So say so at the moment the backend is built.

    `style_system_sha` is the prompt pin (`pairs.system_sha`), checked only when
    the manifest carries one, so an older manifest reads as "cannot answer".
    Warnings, never errors.
    """
    from stylebot.pairs import system_sha

    manifest = manifest or {}
    opts = opts or BackendOpts()
    notes: list[str] = []

    pinned_sha = (manifest.get("data") or {}).get("style_system_sha")
    if pinned_sha and pinned_sha != system_sha():
        notes.append(
            f"STYLE_SYSTEM has changed since run {manifest.get('run_id') or '?'} was trained "
            f"(manifest {pinned_sha}, current {system_sha()}); the styler is being served under "
            f"a system prompt its pairs were never built against"
        )

    trained_renderer = manifest_renderer(manifest)
    if opts.renderer_name and trained_renderer and opts.renderer_name != trained_renderer:
        notes.append(
            f"renderer mismatch: serving with {opts.renderer_name!r}, run "
            f"{manifest.get('run_id') or '?'} trained with {trained_renderer!r}"
        )
    elif opts.backend == "openai" and trained_renderer:
        # The renderer pin is UNCHECKABLE here, which is not the same as
        # checked-and-fine, and the two used to look identical (both silent).
        # D48: the Protocol types the call, not the tokens, so this warning is
        # the mechanism.
        notes.append(
            f"renderer unverifiable: run {manifest.get('run_id') or '?'} trained with "
            f"{trained_renderer!r}, but an OpenAI-compatible endpoint applies its own chat "
            f"template and stylebot cannot see it — score a parity sample against the tinker "
            f"backend before trusting {opts.model or 'this endpoint'}"
        )

    trained_base = (manifest.get("hyperparameters") or {}).get("base_model")
    if opts.base_model and trained_base and opts.base_model != trained_base:
        notes.append(
            f"base-model mismatch: serving with {opts.base_model!r}, run "
            f"{manifest.get('run_id') or '?'} trained on {trained_base!r}"
        )
    return notes


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

    Whatever the caller *did* pin is then checked against the manifest
    (`manifest_mismatches`) and disagreements are logged: filling a field
    silently is right, overriding one silently is not.
    """
    opts = opts or BackendOpts()
    manifest = manifest or {}
    result = manifest.get("result") or {}
    diet_chars, diet_inline = serving_diet(manifest, default_chunk_chars=default_chunk_chars)
    from_manifest = {
        "sampler_path": (result.get("checkpoints") or {}).get("sampler_path"),
        "base_model": (manifest.get("hyperparameters") or {}).get("base_model"),
        "renderer_name": manifest_renderer(manifest),
        "max_chunk_chars": diet_chars,
        "inline_headings": diet_inline,
    }
    fill = {k: v for k, v in from_manifest.items() if v is not None and getattr(opts, k) is None}
    for note in manifest_mismatches(manifest, opts):
        logger.warning("%s", note)
    return replace(opts, **fill)


# ---------------------------------------------------------------------------
# `run` as a library: the option bundle and the two passes (D78)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunOpts:
    """The `run` surface — what `rewrite_corpus` and `rewrite_document` take (D78).

    `BackendOpts` says how to reach the styler; this says what to do with it.
    """

    cli_section = "Rewrite"

    best_of: int = opt(
        DEFAULT_BEST_OF,
        "--best-of",
        show_default=True,
        type=int,
        help="Sample N candidates per chunk and keep the most Dan-scored (needs --detector-model).",
    )
    strict_anchors: bool = opt(
        False,
        "--strict-anchors",
        is_flag=True,
        help="Also require inline code spans and inline math to survive a rewrite, not just links "
        "and citations. Measured over the corpus this roughly triples the rate at which every "
        "candidate is rejected and the input is kept verbatim — hence off by default.",
    )
    anchor_log: Path | None = opt(
        None,
        "--anchor-log",
        type=click.Path(dir_okay=False, path_type=Path),
        help="Append per-chunk anchor accounting to this JSONL (anchor keys and lengths, never "
        "candidate text). The headline counts say whether a run went wrong; this says how the "
        "model behaves, which is what decides whether masking is worth building at all "
        "(D32, D33).",
    )
    write_in_place: bool = opt(
        False,
        "--write",
        "write_in_place",
        is_flag=True,
        help="Rewrite FILE in place (a .bak backup is left beside it). Default prints to stdout.",
    )
    pairs_path: Path | None = opt(
        None,
        "--pairs",
        "pairs_path",
        type=click.Path(exists=True, path_type=Path),  # dir: a sharded corpus
        help="Corpus mode: rewrite the slop side of every pair in this pairs.jsonl instead of a "
        "file, writing an output-bearing JSONL for `ai-style eval --field output`. This is how a "
        "trained adapter gets a number rather than an impression. Needs --out.",
    )
    out_path: Path | None = opt(
        None,
        "--out",
        "out_path",
        type=click.Path(dir_okay=False, path_type=Path),
        help="[--pairs] Where the output-bearing JSONL goes. Appended to and resumable: ids "
        "already present are skipped, so an interrupted sweep continues rather than re-paying "
        "for them.",
    )
    limit: int | None = opt(
        None,
        "--limit",
        type=int,
        help="[--pairs] Cap how many pairs are rewritten.",
    )
    manifest_path: Path | None = opt(
        None,
        "--manifest",
        "manifest_path",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help=MANIFEST_HELP,
    )


RUN_PARAMS: dict[str, type] = {"run": RunOpts, "opts": BackendOpts}


def run_options(**kw):
    """The whole `run` surface — `RUN_PARAMS`'s two bundles, one decorator."""
    return bundle_options(list(RUN_PARAMS.values()), **kw)


def load_manifest(manifest_path: str | Path | None, *, param_hint: str = "--manifest") -> dict:
    """A training manifest as a dict, or `{}` when none was named."""
    if manifest_path is None:
        return {}
    try:
        return json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(
            f"{manifest_path}: not valid JSON ({exc})", param_hint=param_hint
        ) from exc


def corpus_mode(opts: RunOpts, *, source: Path | None = None) -> bool:
    """Whether `opts` asks for corpus mode, after the flag checks both CLIs used to retype.

    Raises `ConfigError` for `--pairs` beside a FILE or without `--out`, and
    for `--out`/`--limit` outside corpus mode.
    """
    if opts.pairs_path is not None:
        if source is not None:
            raise ConfigError(
                "--pairs rewrites a corpus; pass it or a FILE, not both", param_hint="--pairs"
            )
        if opts.out_path is None:
            raise ConfigError(
                "--pairs needs --out (the output-bearing JSONL to write)", param_hint="--out"
            )
        return True
    if opts.out_path is not None or opts.limit is not None:
        raise ConfigError("--out/--limit only apply in corpus mode (--pairs)")
    return False


def rewrite_corpus(
    opts: RunOpts,
    styler: Styler,
    *,
    scorer: Callable[[str], float] | None = None,
    selector: Callable[[dict], bool] | None = None,
    emit: Emitter = silent,
) -> int:
    """Corpus mode: style the slop side of `opts.pairs_path` into `opts.out_path`.

    `selector` is the caller's stratum policy over pair records (the blog keeps
    its role selector); progress and the closing count go to `emit` at
    `status`. Returns the number of pairs written (`infer.rewrite_pairs_file`).
    """
    from stylebot import infer

    if opts.pairs_path is None:
        raise ConfigError("corpus mode needs --pairs (and --out)", param_hint="--pairs")
    corpus_mode(opts)

    def progress(i: int, total: int) -> None:
        if i == 1 or i % 10 == 0 or i == total:
            emit(f"  ... {i}/{total} rewritten", level="status")

    written = infer.rewrite_pairs_file(
        opts.pairs_path,
        opts.out_path,
        styler,
        limit=opts.limit,
        selector=selector,
        best_of=opts.best_of,
        scorer=scorer,
        on_progress=progress,
    )
    emit(f"rewrote {written} pair(s) -> {opts.out_path}", level="status")
    return written


def rewrite_document(
    text: str,
    opts: RunOpts,
    backend: BackendOpts,
    styler: Styler,
    *,
    scorer: Callable[[str], float] | None = None,
    on_decision: Callable[..., None] | None = None,
    div_is_prose=None,
    rewrite: Callable[..., RewriteResult] | None = None,
    source: Path | None = None,
    emit: Emitter = silent,
) -> RewriteResult:
    """Document mode: one text through the styler at the backend's diet.

    `rewrite` replaces `infer.rewrite_text` (same keywords; the blog's masking
    defence); `div_is_prose` is the caller's container policy. Decisions are
    emitted at `status` when something chose (a scorer, or a custom `rewrite`),
    the anchor audit lands in `opts.anchor_log`, and under `opts.write_in_place`
    a `source` file is rewritten beside a `.bak`. Printing the text is the
    caller's.
    """
    from stylebot import infer

    if corpus_mode(opts, source=source):
        raise ConfigError(
            "--pairs rewrites a corpus; pass it or a FILE, not both", param_hint="--pairs"
        )
    rewriter = rewrite or infer.rewrite_text
    result = rewriter(
        text,
        styler,
        max_chunk_chars=backend.chunk_chars(),
        best_of=opts.best_of,
        scorer=scorer,
        on_decision=on_decision,
        inline_headings=backend.headings(),
        inline_anchors=opts.strict_anchors,
        div_is_prose=div_is_prose,
    )
    if scorer is not None or rewrite is not None:
        for line in result.decisions:
            emit(f"  {line}", level="status")
    if opts.anchor_log is not None:
        n = infer.write_anchor_audit(
            opts.anchor_log, result, source=str(source) if source else None
        )
        emit(f"  {n} chunk(s) of anchor accounting -> {opts.anchor_log}", level="status")
    emit(format_rewrite_summary(result), level="status")
    if opts.write_in_place and source:
        source.with_suffix(source.suffix + ".bak").write_text(text, encoding="utf-8")
        source.write_text(result.text, encoding="utf-8")
        emit(f"rewrote {source} (backup at {source}.bak)", level="status")
    return result


def format_rewrite_summary(result: RewriteResult) -> str:
    """The one-line run report: `[N chunk(s), M sample(s), ...]`.

    Anchor rejections are broken out by cause — losing a link wants a better
    sample, inventing one wants a model that stops making things up — and
    truncation is reported apart from them because it usually means the token
    budget, not the sample.
    """
    return (
        f"[{result.n_chunks} chunk(s), {result.n_candidates} sample(s)"
        + (f", {result.n_kept_input} kept as input" if result.n_kept_input else "")
        + (
            f", {result.n_anchor_rejected} sample(s) rejected for anchor damage "
            f"({result.n_anchor_lost} lost, {result.n_anchor_invented} invented)"
            if result.n_anchor_rejected
            else ""
        )
        + (f", {result.n_truncated} truncated (raise --max-tokens?)" if result.n_truncated else "")
        + "]"
    )
