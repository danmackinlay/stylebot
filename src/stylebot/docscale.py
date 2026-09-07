"""Document-scale eval — whole capture archives through `infer.rewrite_text`
(D54), so segmentation, protected blocks and the chunk diet are exercised by
the verdict. `rewrite_documents_file` emits pair-shaped records
`stylebot.scoring` scores unchanged."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from stylebot.cli_options import Emitter, silent
from stylebot.errors import ConfigError
from stylebot.jsonl import dumps_line
from stylebot.pairs import iter_pairs, record_id


@dataclass
class DocumentRewriteResult:
    """Outcome of a `rewrite_documents_file` run."""

    written: int = 0
    skipped_existing: int = 0
    n_chunks: int = 0
    n_kept_input: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)


def iter_document_pairs(pairs_path: str | Path, *, splits: dict | None = None, role: str = "eval"):
    """Whole-document (before, after) captures, optionally restricted to a role.

    Yields `{"capture_id", "source", "before", "after"}` per capture archive
    beside `pairs_path`. With `splits`, only archives whose post resolves to
    `role` are yielded — `role="eval"` being the frozen stratum, the only one a
    verdict may be computed on. `role=None` yields every archive.
    """
    from stylebot.capture import iter_archives

    for archive in iter_archives(Path(pairs_path)):
        source = archive.get("source")
        if splits is not None and role is not None:
            from stylebot import splits as splits_mod

            if splits_mod.role_of(source or "?", splits) != role:
                continue
        yield {
            "capture_id": archive.get("capture_id"),
            "source": source,
            "before": archive.get("before_text") or "",
            "after": archive.get("after_text") or "",
        }


def _document_record(doc: dict, output: str, system: str, reranker: dict | None) -> dict:
    """One whole-document row in `pairs.jsonl` shape, `doc:`-prefixed; `reranker`
    is `infer.reranker_identity`'s answer for the pass (D39)."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": doc["before"]},
            {"role": "assistant", "content": doc["after"]},
        ],
        "output": output,
        "meta": {
            "source": doc["source"],
            "capture_id": f"doc:{doc['capture_id']}",
            "chunk_index": 0,
            "synthetic": False,
            "scale": "document",
            "reranker": reranker,
        },
    }


def rewrite_documents_file(
    pairs_path: str | Path,
    out_path: str | Path,
    styler,
    *,
    splits: dict | None = None,
    role: str = "eval",
    limit: int | None = None,
    rewrite: Callable[..., object] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    system: str | None = None,
    **rewrite_kwargs,
) -> DocumentRewriteResult:
    """Rewrite whole eval-stratum documents; emit a scorable output JSONL.

    Each record is pair-shaped plus a top-level `output`, its `capture_id`
    prefixed `doc:` so it cannot collide with the chunk-level cut. `rewrite`
    is the seam onto `infer.rewrite_text`; `system` (the served prompt; the
    default author's when None) and the diet keywords pass through.

    Resumable by record id; score the output with `prune_orphans=False`.
    """
    from stylebot.ai_core import style_system
    from stylebot.infer import reranker_identity

    if rewrite is None:
        from stylebot.infer import rewrite_text as rewrite
    system = system or style_system()
    reranker = reranker_identity(rewrite_kwargs.get("scorer"))

    out_path = Path(out_path)
    done = {record_id(rec) for rec in iter_pairs(out_path) if rec.get("messages")}

    docs = []
    for doc in iter_document_pairs(pairs_path, splits=splits, role=role):
        if not doc["before"].strip() or not doc["after"].strip():
            continue
        if f"doc:{doc['capture_id']}:0" in done:
            continue
        docs.append(doc)
        if limit is not None and len(docs) >= limit:
            break

    result = DocumentRewriteResult(skipped_existing=len(done))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fp:
        for i, doc in enumerate(docs, start=1):
            try:
                rewritten = rewrite(doc["before"], styler, system=system, **rewrite_kwargs)
            except Exception as exc:  # noqa: BLE001 - one bad post must not end the pass
                result.errors.append((str(doc["capture_id"]), f"{type(exc).__name__}: {exc}"))
                continue
            text = getattr(rewritten, "text", rewritten)
            result.n_chunks += int(getattr(rewritten, "n_chunks", 0) or 0)
            result.n_kept_input += int(getattr(rewritten, "n_kept_input", 0) or 0)
            fp.write(dumps_line(_document_record(doc, text, system, reranker)) + "\n")
            fp.flush()
            result.written += 1
            if on_progress is not None:
                on_progress(i, len(docs))
    return result


@dataclass(frozen=True)
class DocumentPass:
    """What a document-scale rewrite hands the scoring pass that follows it."""

    pairs_path: Path
    fields: tuple[str, ...]
    prune_orphans: bool
    chunk_scale: dict


def document_pass(
    pairs_path: Path,
    out: Path,
    doc_opts,
    score_fields,
    *,
    manifest: dict,
    splits: dict | None,
    div_is_prose=None,
    html_comments: bool = False,
    emit: Emitter = silent,
) -> DocumentPass:
    """`eval --documents` (D78): rewrite whole documents, say what to score.
    The styler and its diet come from `manifest` alone; the one `chunk_scale`
    serves the rewrite and, through `eval.chunked_detector`, the scoring (D54)."""
    from stylebot import backends

    role = None if doc_opts.documents_role == "all" else doc_opts.documents_role
    if role is not None and splits is None:
        raise ConfigError(
            f"--documents-role {role} needs a splits contract, and none was given",
            param_hint="--documents-role",
        )
    opts = backends.opts_from_manifest(manifest, backends.BackendOpts(backend="tinker"))
    styler = backends.build_styler(opts)
    max_chunk_chars, inline_headings = backends.serving_diet(manifest)
    docs_out = Path(doc_opts.documents_out or (Path(out).parent / "documents.jsonl"))
    emit(
        f"rewriting whole {role or 'all'}-role documents with run "
        f"{manifest.get('run_id') or '?'} at {max_chunk_chars} chars"
        f"{', cross-heading' if inline_headings else ''}",
        level="status",
    )
    chunk_scale = {
        "max_chunk_chars": max_chunk_chars,
        "inline_headings": inline_headings,
        "div_is_prose": div_is_prose,
        "html_comments": html_comments,
    }
    result = rewrite_documents_file(
        pairs_path,
        docs_out,
        styler,
        splits=splits,
        role=role,
        system=opts.system(),
        **chunk_scale,
        on_progress=lambda i, total: emit(f"  ... {i}/{total} documents rewritten", level="status"),
    )
    emit(
        f"rewrote {result.written} document(s) ({result.n_chunks} chunks, "
        f"{result.n_kept_input} kept verbatim) -> {docs_out}"
    )
    for capture_id, message in result.errors[:10]:
        emit(f"  {capture_id}: {message}", level="warn")
    fields = (*score_fields, *(() if "output" in score_fields else ("output",)))
    return DocumentPass(docs_out, fields, False, chunk_scale)


def verdict_dir(runs_dir: str | Path, run_id: str) -> Path:
    """Where a run's verdict outputs live: `<runs_dir>/<run_id>/` (D42)."""
    path = Path(runs_dir) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path
