"""Read-only HTML inspection of the corpus — get a *feel* for it, with no spend.

Two reports, one machinery:

- **targets** (`render_targets_report`) — which prose was selected for
  synthesis: a length histogram, summary stats, and a searchable / sortable /
  randomly sampleable table of every target.
- **scores** (`render_scores_report`) — the eval sibling: joins `scores.jsonl`
  to the corpus by `record_id` and lays each field's text, judge score,
  detector P(slop) and Vale count side by side, sorted by the slop→Dan delta.

Both write ONE self-contained HTML file (no server, no network, no external
assets, no deps beyond the stdlib) out of the shared scaffolding in the first
section below — `_CSS`, `_JS`, `_page`, `_stat_cards`, `_chip`, `_cap_note`,
`_histogram_svg`. Where the two differ they differ in *data* (which cards,
which columns), not in a second copy of the markup. `sample_targets` /
`format_sample` / `format_scores_sample` are the stdout equivalents.

**Escaping discipline — a security property, not a style preference.** Every
piece of corpus text (body prose, source paths, headings, judge rationales,
facet values, `title=` tooltips) goes through `html.escape`, with `quote=True`
wherever it lands inside an attribute. The embedded JS only reads `data-*`
attributes and toggles `style.display` on pre-rendered, pre-escaped rows; it
never `innerHTML`s report text. A target containing markup therefore renders as
markup-looking prose and can never execute.
"""

from __future__ import annotations

import html
import random
import statistics
from collections.abc import Sequence
from pathlib import Path

from stylebot.pairs import FIELD_EXTRACTORS, build_pair_content, iter_pairs, record_id
from stylebot.scoring import load_scores, summarize_scores
from stylebot.targets import Target

#: `render_scores_report`'s table cap and headline facet; `verdict.EvalOpts` shares them.
DEFAULT_MAX_ROWS = 2000
DEFAULT_FACET_BY = "slop_strategy"

# ---------------------------------------------------------------------------
# Shared scaffolding — what both reports are assembled from
# ---------------------------------------------------------------------------

# Layout colours are `:root` variables so the dark block at the bottom can flip
# them wholesale. The score chips stay literal: they are a semantic palette
# (slop-red / Dan-green / neutral-grey) that wants a hand-picked pair per theme,
# not one hue mechanically darkened.
_CSS = """
:root{--bar:#4c8bf5;--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#e3e3e3}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  margin:0;padding:24px;color:var(--fg);background:var(--bg)}
h1{font-size:18px;margin:0 0 4px}
.stats{display:flex;flex-wrap:wrap;gap:18px;margin:12px 0}
.stat b{font-size:20px;display:block}
.stat span{color:var(--muted);font-size:12px}
.controls{margin:14px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,button,select{font:inherit;padding:6px 10px;border:1px solid var(--line);
  border-radius:6px;background:var(--bg);color:var(--fg)}
button{cursor:pointer}
table{border-collapse:collapse;width:100%;margin-top:8px}
th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{cursor:pointer;user-select:none;position:sticky;top:0;background:var(--bg)}
.src{white-space:nowrap;color:var(--muted);font-size:12px}
.len{text-align:right;font-variant-numeric:tabular-nums;color:var(--muted)}
.txt{max-width:60ch}
.ctx{color:var(--muted);font-size:12px;font-weight:600;margin-bottom:3px}
.note{color:var(--muted);font-size:12px;margin:6px 0}
.headline{border-collapse:collapse;margin:8px 0 16px;font-size:13px}
.headline th,.headline td{padding:4px 12px;border-bottom:1px solid var(--line);text-align:left}
.num{text-align:right;font-variant-numeric:tabular-nums}
.histos{display:flex;gap:16px;margin:8px 0}
.histos figure{flex:1;margin:0}.histos figcaption{font-size:12px;color:var(--muted);margin-top:2px}
.srcsub{font-size:11px;color:var(--muted)}
.facet{font-size:12px;white-space:nowrap;vertical-align:top}
.gensub{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums;margin-top:2px}
.cmp{display:flex;gap:16px}.cmp>div{flex:1;min-width:0}
.fh{display:flex;align-items:center;gap:8px;margin-bottom:4px}
.fl{font-weight:600;font-size:12px;color:var(--muted)}
.badge{font-size:11px;font-weight:600;padding:1px 7px;border-radius:6px}
.s-slop{background:#fcebeb;color:#791f1f}.s-dan{background:#eaf3de;color:#27500a}
.s-mid{background:#e6f1fb;color:#0c447c}.s-na{background:#f1efe8;color:#5f5e5a}
.s-det{background:#eef0fb;color:#34406a;font-variant-numeric:tabular-nums}
.s-vale{background:#f6efe0;color:#6b5416;font-variant-numeric:tabular-nums}
.ft{font-size:13px;line-height:1.45}
.rat{color:var(--muted);font-size:12px;margin-top:4px}
.d-pos{color:#27500a}.d-neg{color:#791f1f}.d-na{color:var(--muted)}
@media (prefers-color-scheme:dark){
:root{--bar:#5b8ff9;--bg:#15171a;--fg:#e6e6e6;--muted:#9a9a9a;--line:#2c3036}
.s-slop{background:#3a2020;color:#f0b4b4}.s-dan{background:#22301a;color:#bcdc9a}
.s-mid{background:#1b2b3d;color:#a8cdf2}.s-na{background:#2a2926;color:#b6b4ae}
.s-det{background:#22263a;color:#b3bbe4}.s-vale{background:#332c1c;color:#dcc98a}
.d-pos{color:#bcdc9a}.d-neg{color:#f0b4b4}
}
"""

# One behaviour script for both reports. It reads only `data-*` attributes and
# writes only `style.display` / `dataset.hit`, never markup, so whatever a given
# report does not wire up is inert. `#rows` is load-bearing, not decoration: the
# scores page's headline table comes FIRST, so a bare `querySelector('tbody')`
# would silently bind every control to it.
_JS = """
const tbody=document.querySelector('#rows tbody');
const rows=()=>Array.from(tbody.querySelectorAll('tr'));
const facetSels=Array.from(document.querySelectorAll('select.facet'));
function val(r,k){const v=r.dataset[k];return (v===''||v===undefined)?-Infinity:(isNaN(+v)?v:+v);}
function applyFilter(){
  const q=document.getElementById('q').value.toLowerCase();
  for(const r of rows()){
    const okText=(r.dataset.src+' '+r.querySelector('.txt').textContent).toLowerCase().includes(q);
    const hit=okText&&facetSels.every(s=>s.value===''||r.dataset[s.dataset.key]===s.value);
    r.dataset.hit=hit?'1':'0';
    r.style.display=hit?'':'none';
  }
}
function sortBy(k,numeric){
  const rs=rows();
  rs.sort((a,b)=>{
    const x=val(a,k),y=val(b,k);
    return numeric?(y-x):String(x).localeCompare(String(y));
  });
  for(const r of rs) tbody.appendChild(r);
}
function sampleN(){
  const n=+document.getElementById('n').value||20;
  const vis=rows().filter(r=>r.dataset.hit!=='0');
  for(const r of rows()) r.style.display='none';
  for(let i=vis.length-1;i>0;i--){
    const j=Math.floor(Math.random()*(i+1));[vis[i],vis[j]]=[vis[j],vis[i]];
  }
  for(const r of vis.slice(0,n)) r.style.display='';
}
function showAll(){for(const r of rows()){r.dataset.hit='1';r.style.display='';}}
document.getElementById('q').addEventListener('input',applyFilter);
for(const s of facetSels) s.addEventListener('change',applyFilter);
if(document.body.dataset.sort) sortBy(document.body.dataset.sort,true);
"""


def _page(
    *,
    title: str,
    cards: str,
    blocks: Sequence[str],
    controls: str,
    note: str,
    table: str,
    sort: str = "",
) -> str:
    """The one HTML scaffold both reports share.

    Head, `<h1>`, the stat-card row, the report's own `blocks` (histogram /
    headline table), the control strip, the row-count note, the table, and the
    shared behaviour script. `sort` names a `data-*` key to sort numerically on
    load. Deliberately a function, not a builder class — two fixed reports.
    """
    body = "\n".join(blocks)
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head>
<body{f" data-sort={sort}" if sort else ""}>
<h1>{html.escape(title)}</h1>
<div class=stats>{cards}</div>
{body}
<div class=controls>{controls}</div>
<p class=note>{html.escape(note)}</p>
{table}
<script>{_JS}</script>
</body></html>"""


def _stat_cards(items: Sequence[tuple[object, str]]) -> str:
    """The headline number row: one card per `(value, label)`. Ints get thousands
    separators, `None` an em dash; labels are escaped because the scores report
    interpolates caller-chosen field names into them."""

    def fmt(v: object) -> str:
        return "—" if v is None else (f"{v:,}" if isinstance(v, int) else str(v))

    return "".join(
        f"<div class=stat><b>{fmt(v)}</b><span>{html.escape(lab)}</span></div>" for v, lab in items
    )


def _chip(cls: str, text: str | None, title: str = "") -> str:
    """A small pill — judge badge, detector P(slop), Vale count. `None` → nothing."""
    if text is None:
        return ""
    tip = f' title="{html.escape(title, quote=True)}"' if title else ""
    return f'<span class="badge {cls}"{tip}>{html.escape(text)}</span>'


def _clip(text: str, limit: int) -> str:
    """Text truncated for display (the untruncated text lives in a tooltip)."""
    return text if len(text) <= limit else text[:limit] + "…"


def _cap_note(n: int, max_rows: int | None, *, noun: str, covered: str) -> str:
    """The line under the controls. `max_rows` caps only the TABLE, so when it
    bites the note has to say what the stats above it still cover."""
    if max_rows is not None and n > max_rows:
        return f"showing {max_rows:,} of {n:,} rows (stats + {covered} cover all {n:,})"
    return f"showing all {n:,} {noun}"


def _histogram_svg(
    lengths: Sequence[int], *, bins: int, width: int = 720, height: int = 160, unit: str = "chars"
) -> str:
    """A bare inline `<svg>` bar chart; an empty series renders as `<svg></svg>`."""
    if not lengths:
        return "<svg></svg>"
    hi = max(lengths)
    edge = max(1, hi)
    counts = [0] * bins
    for v in lengths:
        counts[min(bins - 1, int(v / edge * bins))] += 1
    peak = max(counts) or 1
    bw = width / bins
    top = height - 24  # bar area, leaving room for the axis labels
    bars = "".join(
        f'<rect x="{i * bw:.1f}" y="{height - 20 - c / peak * top:.1f}" '
        f'width="{bw - 1:.1f}" height="{c / peak * top:.1f}" fill="var(--bar)">'
        f"<title>{int(i * edge / bins)}–{int((i + 1) * edge / bins)} chars: {c}</title></rect>"
        for i, c in enumerate(counts)
    )
    axis = (
        f'<text x="0" y="{height - 4}" class="ax">0</text>'
        f'<text x="{width}" y="{height - 4}" text-anchor="end" class="ax">{hi} {unit}</text>'
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="none" '
        f'role="img" aria-label="length histogram">{bars}{axis}</svg>'
    )


# ---------------------------------------------------------------------------
# Targets report — which prose was selected for synthesis
# ---------------------------------------------------------------------------


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
        out.append(
            f"{t.source}  ·  {len(t.text)} chars  ·  chunk {t.chunk_index + 1}/{t.chunk_total}"
        )
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


def _render_rows(targets: Sequence[Target], *, max_rows: int | None) -> str:
    rows = []
    for t in targets if max_rows is None else targets[:max_rows]:
        context = getattr(t, "context", "") or ""
        full = html.escape(build_pair_content(context, t.text), quote=True)
        # The heading context (the verbatim prefix the pair will carry) shown
        # muted above the body, so heading+passage units are eyeball-able.
        ctx_html = f"<div class=ctx>{html.escape(context)}</div>" if context else ""
        rows.append(
            f'<tr data-src="{html.escape(t.source, quote=True)}" data-len="{len(t.text)}">'
            f"<td class=src>{html.escape(t.source)}</td>"
            f"<td class=len>{len(t.text)}</td>"
            f'<td class=txt title="{full}">{ctx_html}{html.escape(_clip(t.text, 500))}</td></tr>'
        )
    return "\n".join(rows)


_TARGET_CONTROLS = (
    '<input id=q placeholder="filter by source or text…" size=32>'
    "<button onclick=\"sortBy('len',true)\">sort ↑ length</button>"
    "<button onclick=\"sortBy('src',false)\">sort by source</button>"
    '<input id=n type=number value=20 min=1 style="width:70px">'
    '<button onclick="sampleN()">show N random</button>'
    '<button onclick="showAll()">show all</button>'
)

_TARGET_HEAD = (
    "<table id=rows><thead><tr><th onclick=\"sortBy('src',false)\">source</th>"
    "<th onclick=\"sortBy('len',true)\">len</th><th>text</th></tr></thead>"
)


def render_targets_report(
    targets: Sequence[Target],
    path: Path | str,
    *,
    title: str = "stylebot targets",
    max_rows: int | None = DEFAULT_MAX_ROWS,
    histogram_bins: int = 30,
) -> Path:
    """Write a self-contained HTML report of `targets`. Returns the path.

    Stats and the histogram cover ALL targets; `max_rows` caps how many rows are
    embedded in the table (None = all). Each target text cell is HTML-escaped and
    truncated for display (full text in an escaped `title=` tooltip).
    """
    path = Path(path)
    s = _summary_stats(targets)
    cards = [(s["count"], "targets"), (s["median"], "median chars"), (s["p90"], "p90 chars")]
    cards += [(s["max"], "max chars"), (s["total_chars"], "total chars")]
    cards += [(s["n_sources"], "sources")]
    page = _page(
        title=title,
        cards=_stat_cards(cards),
        blocks=[_histogram_svg([len(t.text) for t in targets], bins=histogram_bins)],
        controls=_TARGET_CONTROLS,
        note=_cap_note(s["count"], max_rows, noun="targets", covered="histogram"),
        table=f"{_TARGET_HEAD}<tbody>{_render_rows(targets, max_rows=max_rows)}</tbody></table>",
    )
    path.write_text(page, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Scores report — join pairs.jsonl + scores.jsonl, compare slop ↔ Dan by eye
# ---------------------------------------------------------------------------
#
# The eval sibling of the targets report: read-only, self-contained HTML, joined
# to the corpus by `record_id` and generic over score *fields*.


def _signal(field_scores: object, signal: str, key: str = "score"):
    """One eval signal's value for one field — `judge`, `detector` or `vale`.

    `None` whenever the signal is unconfigured, absent or malformed; every
    caller renders that as an em dash rather than failing the whole report.
    """
    s = field_scores.get(signal) if isinstance(field_scores, dict) else None
    return s.get(key) if isinstance(s, dict) else None


def _vale_alerts(field_scores: object) -> int | None:
    """Vale's alert count for a field, or None when Vale wasn't available."""
    if not _signal(field_scores, "vale", "available"):
        return None
    return int(_signal(field_scores, "vale", "alerts") or 0)


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


def _cell_mean(rows: Sequence[dict], field: str, key: str) -> float | None:
    """Mean of one signal across one field's cells; None when nothing scored."""
    return _mean([r["cells"][field][key] for r in rows])


def _scores_rows_data(
    records: Sequence[dict], pairs_by_id: dict[str, dict], fields: Sequence[str]
) -> list[dict]:
    """Join score records to pairs and flatten to per-row render data."""
    rows = []
    for rec in records:
        pair = pairs_by_id.get(rec.get("id"))
        meta = rec.get("meta") or {}
        scores = rec.get("scores") or {}
        cells = {
            f: {
                "text": _field_text(pair, f),
                "score": _signal(scores.get(f), "judge"),
                "rationale": _signal(scores.get(f), "judge", "rationale") or "",
                "unscored": _signal(scores.get(f), "judge", "parse_error") is True,
                "detector": _signal(scores.get(f), "detector"),
                "vale": _vale_alerts(scores.get(f)),
            }
            for f in fields
        }
        s0, s1 = cells[fields[0]]["score"], cells[fields[-1]]["score"]
        delta = (s1 - s0) if (s0 is not None and s1 is not None) else None
        # Generation covariates from the joined pair (synthetic only; empty for real).
        pair_meta = (pair or {}).get("meta") or {}
        gen = pair_meta.get("gen") or {}
        rows.append(
            {
                "strategy": meta.get("slop_strategy") or "—",
                "source": meta.get("source") or "—",
                # Facet covariates: the score record (_CARRIED_GEN flattening), else
                # the joined pair for score files written before that.
                "generator": str(meta.get("generator") or pair_meta.get("generator") or "—"),
                "reasoning": str(
                    meta.get("reasoning_effort") or gen.get("reasoning_effort") or "—"
                ),
                "prompt": str(meta.get("prompt_id") or gen.get("prompt_id") or "—"),
                # Human-readable prompt, "catalogue v1"; the hash id stays the key + tooltip.
                "prompt_label": _prompt_display(
                    meta.get("prompt_label") or gen.get("prompt_label"),
                    meta.get("prompt_version") or gen.get("prompt_version"),
                    meta.get("prompt_id") or gen.get("prompt_id"),
                ),
                "turn": gen.get("session_turn"),
                # Copying ratio slop<->target (1.0 = verbatim no-op), the hygiene
                # covariate synth records; sortable to surface near-identity pairs.
                "sim": meta.get("transform_sim", pair_meta.get("transform_sim")),
                "cells": cells,
                "delta": delta,
                "gen": gen,
            }
        )
    return rows


def _prompt_display(label, version, prompt_id) -> str:
    """ "catalogue v1" when the pair recorded a label; else the bare hash id."""
    if label:
        return f"{label} v{version}" if version else str(label)
    return str(prompt_id) if prompt_id else "—"


def _load_scores_rows(scores, pairs_path, fields: Sequence[str]) -> tuple[list[dict], list[dict]]:
    """`(records, rows)` — score records plus their joined, flattened render data."""
    records = load_scores(scores)
    pairs_by_id = {record_id(p): p for p in iter_pairs(pairs_path)}
    return records, _scores_rows_data(records, pairs_by_id, fields)


def _scores_stat_cards(rows: Sequence[dict], fields: Sequence[str]) -> str:
    """Counts, judge means (with the refusals they exclude) and mean Δ — plus the
    detector means, but only when a trained detector was wired (else noise)."""
    lo, hi = fields[0], fields[-1]
    det = (_cell_mean(rows, lo, "detector"), _cell_mean(rows, hi, "detector"))
    cards: list[tuple[object, str]] = [
        (len(rows), "pairs"),
        (len({r["strategy"] for r in rows}), "strategies"),
        (_cell_mean(rows, lo, "score"), f"mean {lo} judge"),
        (_cell_mean(rows, hi, "score"), f"mean {hi} judge"),
        (sum(r["cells"][f]["unscored"] for r in rows for f in {lo, hi}), "unscored"),
        (_mean([r["delta"] for r in rows]), "mean Δ"),
    ]
    if det[0] is not None or det[1] is not None:
        cards += [(det[0], f"mean {lo} P(slop)"), (det[1], f"mean {hi} P(slop)")]
    return _stat_cards(cards)


def _headline_table(records: Sequence[dict], fields: Sequence[str], *, by: str) -> str:
    """Per-facet judge means and refusals from `summarize_scores(by=…)`, one row per facet."""
    buckets = summarize_scores(records, by=by).get("by", {})
    if not buckets:
        return ""
    fmt = lambda x: "—" if x is None else f"{x}"  # noqa: E731
    body = ""
    for facet, byfields in sorted(buckets.items()):
        lof, hif = byfields.get(fields[0]) or {}, byfields.get(fields[-1]) or {}
        lo, hi = lof.get("mean_judge_score"), hif.get("mean_judge_score")
        delta = round(hi - lo, 2) if (lo is not None and hi is not None) else None
        unscored = lof.get("n_unscored", 0) + hif.get("n_unscored", 0)
        body += (
            f"<tr><td>{html.escape(str(facet))}</td><td class=num>{lof.get('n', 0)}</td>"
            f"<td class=num>{fmt(lo)}</td><td class=num>{fmt(hi)}</td>"
            f"<td class=num>{fmt(delta)}</td><td class=num>{unscored}</td></tr>"
        )
    cols = ("n", fields[0], fields[-1], "Δ", "unscored")
    heads = "".join(f"<th class=num>{html.escape(c)}</th>" for c in cols)
    return (
        f"<table class=headline><thead><tr><th>{html.escape(by)}</th>{heads}</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _badge(field: str, fields: Sequence[str], score: int | None) -> str:
    """The judge chip; its colour says which end of the comparison this field is."""
    cls = (
        "s-na"
        if score is None
        else ("s-slop" if field == fields[0] else "s-dan" if field == fields[-1] else "s-mid")
    )
    return _chip(cls, "—" if score is None else f"{score}/5")


def _gen_subline(gen: dict) -> str:
    """A muted one-line summary of the NON-column covariates (empty for real
    pairs). Model/reasoning/prompt/turn have their own plain-text columns."""
    parts = [
        f"{lab}{gen[k]}"
        for k, lab in (("temperature", "t="), ("top_p", "top_p="), ("window_fill", "fill="))
        if gen.get(k) is not None
    ]
    if gen.get("provider"):
        parts.append(str(gen["provider"]))
    if (gen.get("finish_reason") or "stop") != "stop":
        parts.append(str(gen["finish_reason"]))
    return f"<div class=gensub>{html.escape(' · '.join(parts))}</div>" if parts else ""


def _render_scores_rows(
    rows: Sequence[dict], fields: Sequence[str], *, max_rows: int | None
) -> str:
    out = []
    for r in rows if max_rows is None else rows[:max_rows]:
        cols = []
        for f in fields:
            c = r["cells"][f]
            det, vale = c["detector"], c["vale"]
            chips = (
                _badge(f, fields, c["score"])
                + _chip(
                    "s-det",
                    None if det is None else f"P(slop) {det:.2f}",
                    "trained detector P(slop)",
                )
                + _chip("s-vale", None if vale is None else f"vale {vale}", "Vale style alerts")
            )
            rat = _clip(c["rationale"], 200)
            rat_html = f"<div class=rat>{html.escape(rat)}</div>" if rat else ""
            cols.append(
                f"<div><div class=fh><span class=fl>{html.escape(f)}</span>{chips}</div>"
                f"<div class=ft>{html.escape(_clip(c['text'], 400))}</div>{rat_html}</div>"
            )
        delta = r["delta"]
        d_txt = "—" if delta is None else (f"+{delta}" if delta > 0 else str(delta))
        d_cls = "d-na" if delta is None else ("d-pos" if delta > 0 else "d-neg")
        slop_s, dan_s = r["cells"][fields[0]]["score"], r["cells"][fields[-1]]["score"]
        turn, sim = r.get("turn"), r.get("sim")
        out.append(
            f'<tr data-strategy="{html.escape(r["strategy"], quote=True)}" '
            f'data-generator="{html.escape(r["generator"], quote=True)}" '
            f'data-reasoning="{html.escape(r["reasoning"], quote=True)}" '
            f'data-prompt="{html.escape(r["prompt"], quote=True)}" '
            f'data-turn="{"" if turn is None else turn}" '
            f'data-sim="{"" if sim is None else sim}" '
            f'data-slop="{"" if slop_s is None else slop_s}" '
            f'data-dan="{"" if dan_s is None else dan_s}" '
            f'data-delta="{"" if delta is None else delta}" '
            f'data-src="{html.escape(r["source"], quote=True)}" '
            f'data-len="{sum(len(r["cells"][f]["text"]) for f in fields)}" data-hit="1">'
            f"<td class=facet>{html.escape(r['strategy'])}"
            f"<div class=srcsub>{html.escape(r['source'])}</div></td>"
            f"<td class=facet>{html.escape(r['generator'])}{_gen_subline(r['gen'])}</td>"
            f"<td class=facet>{html.escape(r['reasoning'])}</td>"
            f'<td class=facet title="{html.escape(r["prompt"], quote=True)}">'
            f"{html.escape(r['prompt_label'])}</td>"
            f'<td class="facet num">{"—" if turn is None else turn}</td>'
            f'<td class="facet num">{"—" if sim is None else f"{sim:.2f}"}</td>'
            f'<td class="len {d_cls}">{html.escape(d_txt)}</td>'
            f"<td class=txt><div class=cmp>{''.join(cols)}</div></td></tr>"
        )
    return "\n".join(out)


def _score_histos(rows: Sequence[dict], fields: Sequence[str], *, bins: int) -> str:
    """Side-by-side judge-score histograms for the first and last field."""
    figs = []
    for f in (fields[0], fields[-1]):
        vals = [r["cells"][f]["score"] for r in rows if r["cells"][f]["score"] is not None]
        svg = _histogram_svg(vals, bins=bins, unit="judge", width=320, height=120)
        figs.append(f"<figure>{svg}<figcaption>{html.escape(f)} judge scores</figcaption></figure>")
    return f"<div class=histos>{''.join(figs)}</div>"


# The sortable columns of the scores table, as `<th>` markup; only the trailing
# comparison header varies (it names the fields being compared).
_SCORE_HEAD = (
    "<table id=rows><thead><tr><th onclick=\"sortBy('strategy',false)\">strategy</th>"
    "<th onclick=\"sortBy('generator',false)\">generator</th>"
    "<th onclick=\"sortBy('reasoning',false)\">reasoning</th>"
    "<th onclick=\"sortBy('prompt',false)\">prompt</th>"
    "<th class=num onclick=\"sortBy('turn',true)\">turn</th>"
    "<th class=num onclick=\"sortBy('sim',true)\" "
    'title="copying ratio slop↔target; 1.0 = verbatim no-op">sim</th>'
    "<th class=num onclick=\"sortBy('delta',true)\">Δ</th>"
)


def render_scores_report(
    scores,
    pairs_path: Path | str,
    out_path: Path | str,
    *,
    title: str = "stylebot eval scores",
    fields: Sequence[str] = ("slop", "target"),
    max_rows: int | None = DEFAULT_MAX_ROWS,
    histogram_bins: int = 5,
    facet_by: str = DEFAULT_FACET_BY,
) -> Path:
    """Write a self-contained HTML scores report. Returns the path.

    Joins `scores` (a `scores.jsonl` path or in-memory records) to `pairs_path`
    by `record_id`, then renders, per pair, each `field`'s body text + judge score
    + rationale side by side (plus a muted generation-covariate sub-line), sortable
    by the slop→Dan delta and filterable by strategy, under a headline grouped by
    `facet_by` (any carried meta covariate).
    """
    out_path = Path(out_path)
    records, rows = _load_scores_rows(scores, pairs_path, fields)

    def facet_select(key: str, label: str) -> str:
        # One dropdown per row-level covariate, omitted when it can't discriminate.
        values = sorted({r[key] for r in rows})
        if len(values) < 2:
            return ""
        opts = f'<option value="">all {html.escape(label)}</option>' + "".join(
            f'<option value="{html.escape(v, quote=True)}">{html.escape(v)}</option>'
            for v in values
        )
        return f'<select class=facet data-key="{key}">{opts}</select>'

    controls = (
        '<input id=q placeholder="filter by source or text…" size=28>'
        + facet_select("strategy", "strategies")
        + facet_select("generator", "generators")
        + facet_select("reasoning", "reasoning")
        + facet_select("prompt", "prompts")
        + "<button onclick=\"sortBy('delta',true)\">sort Δ</button>"
        "<button onclick=\"sortBy('slop',true)\">sort slop</button>"
        "<button onclick=\"sortBy('dan',true)\">sort Dan</button>"
        "<button onclick=\"sortBy('strategy',false)\">sort strategy</button>"
    )
    head = (
        f"{_SCORE_HEAD}<th>comparison ({html.escape(fields[0])} ↔ "
        f"{html.escape(fields[-1])})</th></tr></thead>"
    )
    body_rows = _render_scores_rows(rows, fields, max_rows=max_rows)
    page = _page(
        title=title,
        cards=_scores_stat_cards(rows, fields),
        blocks=[
            _headline_table(records, fields, by=facet_by),
            _score_histos(rows, fields, bins=histogram_bins),
        ],
        controls=controls,
        note=_cap_note(len(rows), max_rows, noun="pairs", covered="summary"),
        table=f"{head}<tbody>{body_rows}</tbody></table>",
        sort="delta",
    )
    out_path.write_text(page, encoding="utf-8")
    return out_path


def format_scores_sample(
    scores,
    pairs_path: Path | str,
    n: int,
    *,
    fields: Sequence[str] = ("slop", "target"),
    seed: int | None = None,
) -> str:
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
