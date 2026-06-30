"""Read-only inspection of synthesis targets — get a *feel* for them.

We make a lot of quality judgments about which prose becomes a training target;
this module makes those judgments eyeball-able without any API spend:

- `render_targets_report(targets, path)` writes ONE self-contained HTML file (no
  server, no network, no external assets, no deps beyond the stdlib) with a
  length histogram, summary stats, and a searchable / sortable / random-sampleable
  table of every target. Open it in a browser.
- `sample_targets` / `format_sample` print a quick random sample to the terminal.

All target text is HTML-escaped; the embedded JS only toggles row visibility on
pre-rendered, pre-escaped rows (never `innerHTML`s target text), so a target
containing markup can't inject anything.
"""

from __future__ import annotations

import html
import json
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

from stylebot.eval import FIELD_EXTRACTORS, load_scores, record_id, summarize_scores
from stylebot.pairs import build_pair_content, iter_pairs
from stylebot.synth import Target


def sample_targets(targets: Sequence[Target], n: int, *, seed: int | None = None) -> list[Target]:
    """Return up to `n` targets chosen at random (deterministic if `seed` given)."""
    rng = random.Random(seed)
    return rng.sample(list(targets), min(n, len(targets)))


def format_sample(targets: Sequence[Target]) -> str:
    """Plain-text rendering of targets for stdout (source · length · text)."""
    out: list[str] = []
    for t in targets:
        context = getattr(t, "context", "") or ""
        out.append("─" * 78)
        out.append(f"{t.source}  ·  {len(t.text)} chars  ·  chunk {t.chunk_index + 1}/{t.chunk_total}")
        out.append("")
        if context:
            out.append(f"  ⌜ {context} ⌟")  # the heading the passage sits under
        out.append(t.text)
    return "\n".join(out)


def _summary_stats(targets: Sequence[Target]) -> dict:
    lengths = sorted(len(t.text) for t in targets)
    n = len(lengths)
    if n == 0:
        return {"count": 0, "median": 0, "p90": 0, "max": 0, "total_chars": 0, "n_sources": 0}
    return {
        "count": n,
        "median": int(statistics.median(lengths)),
        "p90": lengths[min(n - 1, int(0.9 * n))],
        "max": lengths[-1],
        "total_chars": sum(lengths),
        "n_sources": len({t.source for t in targets}),
    }


def _histogram_svg(lengths: Sequence[int], *, bins: int, width: int = 720, height: int = 160, unit: str = "chars") -> str:
    if not lengths:
        return "<svg></svg>"
    hi = max(lengths)
    edge = max(1, hi)
    counts = [0] * bins
    for v in lengths:
        idx = min(bins - 1, int(v / edge * bins))
        counts[idx] += 1
    peak = max(counts) or 1
    bw = width / bins
    bars = []
    for i, c in enumerate(counts):
        h = (c / peak) * (height - 24)
        x = i * bw
        y = height - 20 - h
        lo, hh = int(i * edge / bins), int((i + 1) * edge / bins)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 1:.1f}" height="{h:.1f}" '
            f'fill="var(--bar)"><title>{lo}–{hh} chars: {c}</title></rect>'
        )
    axis = (
        f'<text x="0" y="{height - 4}" class="ax">0</text>'
        f'<text x="{width}" y="{height - 4}" text-anchor="end" class="ax">{hi} {unit}</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" '
        f'role="img" aria-label="length histogram">{"".join(bars)}{axis}</svg>'
    )


def _render_rows(targets: Sequence[Target], *, max_rows: int | None) -> str:
    rows = []
    shown = targets if max_rows is None else targets[:max_rows]
    for t in shown:
        context = getattr(t, "context", "") or ""
        full = html.escape(build_pair_content(context, t.text), quote=True)
        preview = t.text if len(t.text) <= 500 else t.text[:500] + "…"
        # The heading context (the verbatim prefix the pair will carry) shown
        # muted above the body, so heading+passage units are eyeball-able.
        ctx_html = f'<div class=ctx>{html.escape(context)}</div>' if context else ""
        rows.append(
            f'<tr data-src="{html.escape(t.source, quote=True)}" data-len="{len(t.text)}">'
            f"<td class=src>{html.escape(t.source)}</td>"
            f"<td class=len>{len(t.text)}</td>"
            f'<td class=txt title="{full}">{ctx_html}{html.escape(preview)}</td></tr>'
        )
    return "\n".join(rows)


_CSS = """
:root{--bar:#4c8bf5;--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e3e3e3}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;padding:24px;color:var(--fg);background:var(--bg)}
h1{font-size:18px;margin:0 0 4px}
.stats{display:flex;flex-wrap:wrap;gap:18px;margin:12px 0}
.stat b{font-size:20px;display:block}
.stat span{color:var(--muted);font-size:12px}
.controls{margin:14px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{cursor:pointer}
table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{cursor:pointer;user-select:none;position:sticky;top:0;background:var(--bg)}
.src{white-space:nowrap;color:var(--muted);font-size:12px}
.len{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.txt{max-width:60ch}
.ctx{color:var(--muted);font-size:12px;font-weight:600;margin-bottom:3px}
.note{color:var(--muted);font-size:12px;margin:6px 0}
"""

_JS = """
const tbody=document.querySelector('tbody');
const rows=()=>Array.from(tbody.querySelectorAll('tr'));
let sortLen=true;
function applyFilter(){
  const q=document.getElementById('q').value.toLowerCase();
  for(const r of rows()){
    const hay=(r.dataset.src+' '+r.querySelector('.txt').textContent).toLowerCase();
    r.dataset.hit=hay.includes(q)?'1':'0';
    r.style.display=r.dataset.hit==='1'?'':'none';
  }
}
function sortBy(col){
  const rs=rows();
  rs.sort((a,b)=> col==='len'
    ? (+b.dataset.len)-(+a.dataset.len)
    : a.dataset.src.localeCompare(b.dataset.src));
  for(const r of rs) tbody.appendChild(r);
}
function sampleN(){
  const n=+document.getElementById('n').value||20;
  const vis=rows().filter(r=>r.dataset.hit!=='0');
  for(const r of rows()) r.style.display='none';
  for(let i=vis.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[vis[i],vis[j]]=[vis[j],vis[i]];}
  for(const r of vis.slice(0,n)) r.style.display='';
}
function showAll(){for(const r of rows()){r.dataset.hit='1';r.style.display='';}}
document.getElementById('q').addEventListener('input',applyFilter);
"""


def _page(*, title: str, stats: dict, svg: str, rows_html: str, note: str) -> str:
    s = stats
    stat = lambda v, lab: f'<div class=stat><b>{v:,}</b><span>{lab}</span></div>'  # noqa: E731
    stats_html = "".join(
        [
            stat(s["count"], "targets"),
            stat(s["median"], "median chars"),
            stat(s["p90"], "p90 chars"),
            stat(s["max"], "max chars"),
            stat(s["total_chars"], "total chars"),
            stat(s["n_sources"], "sources"),
        ]
    )
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class=stats>{stats_html}</div>
{svg}
<div class=controls>
  <input id=q placeholder="filter by source or text…" size=32>
  <button onclick="sortBy('len')">sort ↑ length</button>
  <button onclick="sortBy('src')">sort by source</button>
  <input id=n type=number value=20 min=1 style="width:70px">
  <button onclick="sampleN()">show N random</button>
  <button onclick="showAll()">show all</button>
</div>
<p class=note>{html.escape(note)}</p>
<table><thead><tr><th onclick="sortBy('src')">source</th><th onclick="sortBy('len')">len</th><th>text</th></tr></thead>
<tbody>
{rows_html}
</tbody></table>
<script>{_JS}</script>
</body></html>"""


def render_targets_report(
    targets: Sequence[Target],
    path: Path | str,
    *,
    title: str = "stylebot targets",
    max_rows: int | None = 2000,
    histogram_bins: int = 30,
) -> Path:
    """Write a self-contained HTML report of `targets`. Returns the path.

    Stats and the histogram cover ALL targets; `max_rows` caps how many rows are
    embedded in the table (None = all). Each target text cell is HTML-escaped and
    truncated for display (full text in an escaped `title=` tooltip).
    """
    path = Path(path)
    stats = _summary_stats(targets)
    svg = _histogram_svg([len(t.text) for t in targets], bins=histogram_bins)
    rows_html = _render_rows(targets, max_rows=max_rows)
    n = stats["count"]
    if max_rows is not None and n > max_rows:
        note = f"showing {max_rows:,} of {n:,} rows in the table (stats + histogram cover all {n:,})"
    else:
        note = f"showing all {n:,} targets"
    path.write_text(_page(title=title, stats=stats, svg=svg, rows_html=rows_html, note=note), encoding="utf-8")
    return path


# `json` is imported for callers who want a machine-readable dump alongside the
# HTML; kept here so a future `--report-json` can reuse `_summary_stats`.
def stats_json(targets: Sequence[Target]) -> str:
    """Return summary stats as a JSON string (for scripting / CI checks)."""
    return json.dumps(_summary_stats(targets), indent=2)


# ---------------------------------------------------------------------------
# Scores report — join pairs.jsonl + scores.jsonl, compare slop ↔ Dan by eye
# ---------------------------------------------------------------------------
#
# The eval sibling of the targets report: read-only, no spend, self-contained
# HTML. Reuses the same _CSS / _histogram_svg / escaping discipline and the eval
# plumbing (load_scores, record_id, FIELD_EXTRACTORS, summarize_scores) +
# pairs.iter_pairs — so scores stay joined to the corpus by id, and the renderer
# is generic over score *fields* (slop/target now; slop/output/target later).


def _judge_score(field_scores: object) -> int | None:
    j = field_scores.get("judge") if isinstance(field_scores, dict) else None
    return j.get("score") if isinstance(j, dict) else None


def _judge_rationale(field_scores: object) -> str:
    j = field_scores.get("judge") if isinstance(field_scores, dict) else None
    return (j.get("rationale") or "") if isinstance(j, dict) else ""


def _detector_score(field_scores: object) -> float | None:
    """The trained detector's P(slop) for a field, or None if unconfigured/absent."""
    d = field_scores.get("detector") if isinstance(field_scores, dict) else None
    return d.get("score") if isinstance(d, dict) else None


def _field_text(pair: dict | None, field: str) -> str:
    """The body text scored for `field`, pulled from the joined pair record."""
    if pair is None:
        return "(text unavailable)"
    extractor = FIELD_EXTRACTORS.get(field)
    if extractor is None:
        return "(unknown field)"
    try:
        return extractor(pair)
    except Exception:  # malformed pair — never break the report
        return "(text unavailable)"


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def _scores_rows_data(records: Sequence[dict], pairs_by_id: dict[str, dict], fields: Sequence[str]) -> list[dict]:
    """Join score records to pairs and flatten to per-row render data."""
    rows = []
    for rec in records:
        pair = pairs_by_id.get(rec.get("id"))
        meta = rec.get("meta") or {}
        scores = rec.get("scores") or {}
        cells = {
            f: {
                "text": _field_text(pair, f),
                "score": _judge_score(scores.get(f)),
                "rationale": _judge_rationale(scores.get(f)),
                "detector": _detector_score(scores.get(f)),
            }
            for f in fields
        }
        s0, s1 = cells[fields[0]]["score"], cells[fields[-1]]["score"]
        delta = (s1 - s0) if (s0 is not None and s1 is not None) else None
        # Generation covariates from the joined pair (synthetic only; empty for real).
        gen = ((pair or {}).get("meta") or {}).get("gen") or {}
        rows.append({
            "id": rec.get("id", ""),
            "strategy": meta.get("slop_strategy") or "—",
            "source": meta.get("source") or "—",
            "cells": cells,
            "delta": delta,
            "gen": gen,
        })
    return rows


def _load_scores_rows(scores, pairs_path, fields: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """`(records, rows)` — score records plus their joined, flattened render data."""
    records = load_scores(scores)
    pairs_by_id = {record_id(p): p for p in iter_pairs(pairs_path)}
    return records, _scores_rows_data(records, pairs_by_id, fields)


def _scores_summary_stats(rows: Sequence[dict], fields: Sequence[str]) -> dict:
    return {
        "count": len(rows),
        "n_strategies": len({r["strategy"] for r in rows}),
        "mean_slop": _mean([r["cells"][fields[0]]["score"] for r in rows]),
        "mean_dan": _mean([r["cells"][fields[-1]]["score"] for r in rows]),
        "mean_delta": _mean([r["delta"] for r in rows]),
        # Trained-detector P(slop) means (None when no detector was wired).
        "mean_det_slop": _mean([r["cells"][fields[0]]["detector"] for r in rows]),
        "mean_det_dan": _mean([r["cells"][fields[-1]]["detector"] for r in rows]),
    }


def _scores_headline_rows(records: Sequence[dict], fields: Sequence[str], *, by: str = "slop_strategy") -> list[dict]:
    """Per-facet means via summarize_scores(by=…) — the compare-flavours headline."""
    buckets = summarize_scores(records, by=by).get("by", {})
    out = []
    for facet, byfields in sorted(buckets.items()):
        sm = (byfields.get(fields[0]) or {}).get("mean_judge_score")
        dm = (byfields.get(fields[-1]) or {}).get("mean_judge_score")
        delta = round(dm - sm, 2) if (sm is not None and dm is not None) else None
        out.append({"strategy": facet, "n": (byfields.get(fields[0]) or {}).get("n", 0), "slop": sm, "dan": dm, "delta": delta})
    return out


_SCORES_CSS = """
.headline{border-collapse:collapse;margin:8px 0 16px;font-size:13px}
.headline th,.headline td{padding:4px 12px;border-bottom:1px solid var(--line);text-align:left}
.num{text-align:right;font-variant-numeric:tabular-nums}
.histos{display:flex;gap:16px;margin:8px 0}
.histos figure{flex:1;margin:0}.histos figcaption{font-size:12px;color:var(--muted);margin-top:2px}
.srcsub{font-size:11px;color:var(--muted)}
.gensub{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;margin-top:2px}
.cmp{display:flex;gap:16px}.cmp>div{flex:1;min-width:0}
.fh{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.fl{font-weight:600;font-size:12px;color:var(--muted)}
.badge{font-size:11px;font-weight:600;padding:1px 7px;border-radius:6px}
.s-slop{background:#fcebeb;color:#791f1f}.s-dan{background:#eaf3de;color:#27500a}.s-mid{background:#e6f1fb;color:#0c447c}.s-na{background:#f1efe8;color:#5f5e5a}
.s-det{background:#eef0fb;color:#34406a;font-variant-numeric:tabular-nums}
.ft{font-size:13px;line-height:1.45}
.rat{color:var(--muted);font-size:12px;margin-top:4px}
.d-pos{color:#27500a}.d-neg{color:#791f1f}.d-na{color:var(--muted)}
"""

_JS_SCORES = """
const tbody=document.querySelector('tbody');
const rows=()=>Array.from(tbody.querySelectorAll('tr'));
function val(r,k){const v=r.dataset[k];return (v===''||v===undefined)?-Infinity:(isNaN(+v)?v:+v);}
function applyFilter(){
  const q=document.getElementById('q').value.toLowerCase();
  const st=document.getElementById('strat').value;
  for(const r of rows()){
    const okText=(r.dataset.src+' '+r.querySelector('.txt').textContent).toLowerCase().includes(q);
    const hit=okText&&(st===''||r.dataset.strategy===st);
    r.dataset.hit=hit?'1':'0';
    r.style.display=hit?'':'none';
  }
}
function sortBy(k,numeric){
  const rs=rows();
  rs.sort((a,b)=>{const x=val(a,k),y=val(b,k);return numeric?(y-x):String(x).localeCompare(String(y));});
  for(const r of rs) tbody.appendChild(r);
}
document.getElementById('q').addEventListener('input',applyFilter);
document.getElementById('strat').addEventListener('change',applyFilter);
sortBy('delta',true);
"""


def _badge(field: str, fields: Sequence[str], score: int | None) -> str:
    cls = "s-na" if score is None else ("s-slop" if field == fields[0] else "s-dan" if field == fields[-1] else "s-mid")
    txt = "—" if score is None else f"{score}/5"
    return f'<span class="badge {cls}">{txt}</span>'


def _det_chip(score: float | None) -> str:
    """A muted chip showing the trained detector's P(slop); omitted if unconfigured."""
    if score is None:
        return ""
    return f'<span class="badge s-det" title="trained detector P(slop)">P(slop) {score:.2f}</span>'


def _gen_subline(gen: dict) -> str:
    """A muted one-line summary of generation covariates (empty for real pairs)."""
    if not gen:
        return ""
    parts: list[str] = []
    if gen.get("model"):
        parts.append(str(gen["model"]))
    if gen.get("reasoning_effort"):
        parts.append(f"reasoning={gen['reasoning_effort']}")
    if gen.get("temperature") is not None:
        parts.append(f"t={gen['temperature']}")
    if gen.get("top_p") is not None:
        parts.append(f"top_p={gen['top_p']}")
    if gen.get("prompt_id"):
        parts.append(f"prompt {gen['prompt_id']}")
    if gen.get("finish_reason"):
        parts.append(str(gen["finish_reason"]))
    return f"<div class=gensub>{html.escape(' · '.join(parts))}</div>" if parts else ""


def _render_scores_rows(rows: Sequence[dict], fields: Sequence[str], *, max_rows: int | None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    out = []
    for r in shown:
        cols = []
        for f in fields:
            c = r["cells"][f]
            text = c["text"]
            preview = text if len(text) <= 400 else text[:400] + "…"
            rat = c["rationale"]
            rat = rat if len(rat) <= 200 else rat[:200] + "…"
            rat_html = f"<div class=rat>{html.escape(rat)}</div>" if rat else ""
            cols.append(
                f'<div><div class=fh><span class=fl>{html.escape(f)}</span>'
                f'{_badge(f, fields, c["score"])}{_det_chip(c["detector"])}</div>'
                f"<div class=ft>{html.escape(preview)}</div>{rat_html}</div>"
            )
        delta = r["delta"]
        d_txt = "—" if delta is None else (f"+{delta}" if delta > 0 else str(delta))
        d_cls = "d-na" if delta is None else ("d-pos" if delta > 0 else "d-neg")
        slop_s, dan_s = r["cells"][fields[0]]["score"], r["cells"][fields[-1]]["score"]
        total_len = sum(len(r["cells"][f]["text"]) for f in fields)
        out.append(
            f'<tr data-strategy="{html.escape(r["strategy"], quote=True)}" '
            f'data-slop="{"" if slop_s is None else slop_s}" data-dan="{"" if dan_s is None else dan_s}" '
            f'data-delta="{"" if delta is None else delta}" '
            f'data-src="{html.escape(r["source"], quote=True)}" data-len="{total_len}" data-hit="1">'
            f"<td class=src>{html.escape(r['strategy'])}<div class=srcsub>{html.escape(r['source'])}</div>"
            f"{_gen_subline(r['gen'])}</td>"
            f'<td class="len {d_cls}">{html.escape(d_txt)}</td>'
            f'<td class=txt><div class=cmp>{"".join(cols)}</div></td></tr>'
        )
    return "\n".join(out)


def _score_histos(rows: Sequence[dict], fields: Sequence[str], *, bins: int) -> str:
    slop = [r["cells"][fields[0]]["score"] for r in rows if r["cells"][fields[0]]["score"] is not None]
    dan = [r["cells"][fields[-1]]["score"] for r in rows if r["cells"][fields[-1]]["score"] is not None]
    sv = _histogram_svg(slop, bins=bins, unit="judge", width=320, height=120) if slop else "<svg></svg>"
    dv = _histogram_svg(dan, bins=bins, unit="judge", width=320, height=120) if dan else "<svg></svg>"
    return (
        f"<div class=histos><figure>{sv}<figcaption>{html.escape(fields[0])} judge scores</figcaption></figure>"
        f"<figure>{dv}<figcaption>{html.escape(fields[-1])} judge scores</figcaption></figure></div>"
    )


def _render_headline(headline_rows: Sequence[dict], fields: Sequence[str], *, by_label: str = "strategy") -> str:
    if not headline_rows:
        return ""
    fmt = lambda x: "—" if x is None else f"{x}"  # noqa: E731
    body = "".join(
        f"<tr><td>{html.escape(str(h['strategy']))}</td><td class=num>{h['n']}</td>"
        f"<td class=num>{fmt(h['slop'])}</td><td class=num>{fmt(h['dan'])}</td><td class=num>{fmt(h['delta'])}</td></tr>"
        for h in headline_rows
    )
    return (
        f"<table class=headline><thead><tr><th>{html.escape(by_label)}</th><th class=num>n</th>"
        f"<th class=num>{html.escape(fields[0])}</th><th class=num>{html.escape(fields[-1])}</th>"
        f"<th class=num>Δ</th></tr></thead><tbody>{body}</tbody></table>"
    )


def _scores_stat_cards(stats: dict, fields: Sequence[str]) -> str:
    fmt = lambda x: "—" if x is None else x  # noqa: E731
    card = lambda v, lab: f"<div class=stat><b>{v}</b><span>{html.escape(lab)}</span></div>"  # noqa: E731
    cards = [
        card(stats["count"], "pairs"),
        card(stats["n_strategies"], "strategies"),
        card(fmt(stats["mean_slop"]), f"mean {fields[0]} judge"),
        card(fmt(stats["mean_dan"]), f"mean {fields[-1]} judge"),
        card(fmt(stats["mean_delta"]), "mean Δ"),
    ]
    # Detector cards only when a trained detector was wired (else both means None).
    if stats.get("mean_det_slop") is not None or stats.get("mean_det_dan") is not None:
        cards += [
            card(fmt(stats["mean_det_slop"]), f"mean {fields[0]} P(slop)"),
            card(fmt(stats["mean_det_dan"]), f"mean {fields[-1]} P(slop)"),
        ]
    return "".join(cards)


def _scores_page(*, title, stat_cards, headline, histos, controls, table, note) -> str:
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}{_SCORES_CSS}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<div class=stats>{stat_cards}</div>
{headline}
{histos}
<div class=controls>{controls}</div>
<p class=note>{html.escape(note)}</p>
{table}
<script>{_JS_SCORES}</script>
</body></html>"""


def render_scores_report(
    scores,
    pairs_path: Path | str,
    out_path: Path | str,
    *,
    title: str = "stylebot eval scores",
    fields: Sequence[str] = ("slop", "target"),
    max_rows: int | None = 2000,
    histogram_bins: int = 5,
    facet_by: str = "slop_strategy",
) -> Path:
    """Write a self-contained HTML scores report. Returns the path.

    Joins `scores` (a `scores.jsonl` path or in-memory records) to `pairs_path`
    by `record_id`, then renders, per pair, each `field`'s body text + judge score
    + rationale side by side (plus a muted generation-covariate sub-line), sortable
    by the slop→Dan delta and filterable by strategy, under a headline grouped by
    `facet_by` (any carried meta covariate — slop_strategy, reasoning_effort,
    prompt_id, …). Generic over `fields` — a Phase-4 styler run renders with no change.
    """
    out_path = Path(out_path)
    records, rows = _load_scores_rows(scores, pairs_path, fields)
    stats = _scores_summary_stats(rows, fields)
    strategies = sorted({r["strategy"] for r in rows})
    strat_opts = '<option value="">all strategies</option>' + "".join(
        f'<option value="{html.escape(s, quote=True)}">{html.escape(s)}</option>' for s in strategies
    )
    controls = (
        '<input id=q placeholder="filter by source or text…" size=28>'
        f"<select id=strat>{strat_opts}</select>"
        "<button onclick=\"sortBy('delta',true)\">sort Δ</button>"
        "<button onclick=\"sortBy('slop',true)\">sort slop</button>"
        "<button onclick=\"sortBy('dan',true)\">sort Dan</button>"
        "<button onclick=\"sortBy('strategy',false)\">sort strategy</button>"
    )
    table = (
        "<table><thead><tr><th onclick=\"sortBy('strategy',false)\">strategy</th>"
        "<th class=num onclick=\"sortBy('delta',true)\">Δ</th>"
        f"<th>comparison ({html.escape(fields[0])} ↔ {html.escape(fields[-1])})</th></tr></thead>"
        f"<tbody>{_render_scores_rows(rows, fields, max_rows=max_rows)}</tbody></table>"
    )
    n = len(rows)
    note = (
        f"showing {max_rows:,} of {n:,} rows (stats + summary cover all {n:,})"
        if (max_rows is not None and n > max_rows)
        else f"showing all {n:,} pairs"
    )
    page = _scores_page(
        title=title,
        stat_cards=_scores_stat_cards(stats, fields),
        headline=_render_headline(_scores_headline_rows(records, fields, by=facet_by), fields, by_label=facet_by),
        histos=_score_histos(rows, fields, bins=histogram_bins),
        controls=controls,
        table=table,
        note=note,
    )
    out_path.write_text(page, encoding="utf-8")
    return out_path


def format_scores_sample(scores, pairs_path: Path | str, n: int, *, fields: Sequence[str] = ("slop", "target"), seed: int | None = None) -> str:
    """Plain-text sample of joined scores for stdout (`eval --sample`)."""
    _, rows = _load_scores_rows(scores, pairs_path, fields)
    pick = random.Random(seed).sample(rows, min(n, len(rows)))
    out: list[str] = []
    for r in pick:
        out.append("─" * 78)
        out.append(f"{r['strategy']}  ·  {r['source']}  ·  Δ {r['delta']}")
        for f in fields:
            c = r["cells"][f]
            sc = "—" if c["score"] is None else f"{c['score']}/5"
            out.append("")
            out.append(f"  [{f} {sc}] {c['text']}")
            if c["rationale"]:
                out.append(f"      ⌜ {c['rationale']} ⌟")
    return "\n".join(out)
