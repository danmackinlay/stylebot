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

from stylebot.pairs import build_pair_content
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


def _histogram_svg(lengths: Sequence[int], *, bins: int, width: int = 720, height: int = 160) -> str:
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
        f'<text x="{width}" y="{height - 4}" text-anchor="end" class="ax">{hi} chars</text>'
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
