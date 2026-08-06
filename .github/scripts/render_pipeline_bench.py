#!/usr/bin/env python3
"""Render the fixture_bench JSON as charts + a markdown PR report.

Two series: `baseline` (the latest released `tokenizers` crate — the bar to beat
AND the correctness reference, drawn gray; the in-tree `Tokenizer`, being removed,
only builds the pipeline) and `pipeline` (the experimental PipelineTokenizer,
blue). `ids_match` compares the pipeline's encode ids against the release's,
`text_match` its decoded text. Leads with
the always-visible charts — per-model geomean ×speedup, its per-fixture twin (the
same speedups decomposed by workload instead of by model), the decode twin of the
overview, the deterministic work
lane (exact allocation counts from the counting allocator) plus a "work vs base"
verdict block naming exactly where work
moved, memory footprint, binary size — then one collapsed <details> per model
(per-fixture speedup, input-size response, thread scaling per fixture group, the
decode chart and its own thread sweeps, numbers
table with an allocs/MB column). Models the pipeline can't build yet
render as "not supported" roadmap cards; a model it can build but not *decode*
yet (WordPiece keeps no id → token map) renders its decode series as "pending"
while the release's is still drawn. Emits light-theme SVGs + pipeline_bench.md;
the CI workflow rasterizes and uploads the SVGs. Input schema: see fixture_bench.rs.

No single cross-model number anywhere on purpose: the models exercise different
execution modes (normalizer-heavy / split-heavy / model-bounded), so the overview
aggregates per model (geomean across fixtures) and the fixture view aggregates per
fixture (geomean across models), never both at once.
"""
import argparse
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from html import escape
from pathlib import Path

INK = {
    "surface": "#fcfcfb", "card": "#ffffff", "primary": "#0b0b0b",
    "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
    "border": "#e1e0d9", "baseline": "#c3c2b7", "critical": "#d03b3b",
}
# The release baseline is context-gray on purpose: in the speedup charts it *is*
# the ×1.0 axis; in memory/binary-size it's the reference bar to read against.
SERIES_INK = {"baseline": "#898781", "pipeline": "#2a78d6"}
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
GUTTER, PLOT_W, PAD_R, COL_W, ROW_H, BAR_H = 190, 540, 110, 150, 26, 16
CHART_W = GUTTER + PLOT_W + PAD_R + COL_W
# Overview / linear charts: wider gutter (model name + desc), same total width.
OV_GUTTER, OV_PLOT = 250, 440
OV_PLOT_W = CHART_W - OV_GUTTER - PAD_R - COL_W  # plot width for 0-anchored linear bars
TICKS = [0.5, 0.67, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
GROUPS = [("lang", "Languages"), ("modalities", "Modalities")]
CARD_W, CARD_H = 470, 150


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def quartile_cell(values):
    """`Q1 · med · Q3` of `values` (linear interpolation), the numbers behind an
    overview row's min–max whisker. A single value collapses to itself three times."""
    s = sorted(values)

    def q(p):
        pos = p * (len(s) - 1)
        lo = math.floor(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    spec = "{:,.0f}" if s[-1] >= 100 else "{:.1f}"
    return " · ".join(spec.format(q(p)) for p in (0.25, 0.5, 0.75))


def fnum(v, spec="{:.1f}"):
    return spec.format(v) if v is not None else "—"


def chain(vals, spec="{:.1f}"):
    return " → ".join(fnum(v, spec) for v in vals)


def bar_path(x0, x1, y, h, r):
    # square at the baseline (x0), rounded at the data end (x1)
    left, right = min(x0, x1), max(x0, x1)
    r = min(r, (right - left) / 2, h / 2)
    if x1 >= x0:
        return (f"M{left:.1f},{y:.1f} H{right - r:.1f} q{r:.1f},0 {r:.1f},{r:.1f} "
                f"V{y + h - r:.1f} q0,{r:.1f} -{r:.1f},{r:.1f} H{left:.1f} Z")
    return (f"M{right:.1f},{y:.1f} H{left + r:.1f} q-{r:.1f},0 -{r:.1f},{r:.1f} "
            f"V{y + h - r:.1f} q0,{r:.1f} {r:.1f},{r:.1f} H{right:.1f} Z")


def hbar(x0, x1, y, h, color):
    if abs(x1 - x0) < 1.5:
        return f'<rect x="{min(x0, x1):.1f}" y="{y:.1f}" width="1.5" height="{h}" fill="{color}"/>'
    return f'<path d="{bar_path(x0, x1, y, h, 4)}" fill="{color}"/>'


def log_x(gutter, plot_w, lo, hi):
    """A log2 x-mapping onto [gutter, gutter+plot_w] — shared by the speedup charts
    so different models plot on the same scale and are directly comparable."""
    span = math.log2(hi) - math.log2(lo)
    return lambda v: gutter + (math.log2(v) - math.log2(lo)) / span * plot_w


def speedup(row):
    b, p = row["mbps"]["baseline"], row["mbps"]["pipeline"]
    return p / b if b and p else None


def model_speedups(model):
    return [v for v in (speedup(r) for r in model["results"]) if v]


def decode_speedup(row):
    m = row.get("decode_mbps") or {}
    b, p = m.get("baseline"), m.get("pipeline")
    return p / b if b and p else None


def decode_model_speedups(model):
    return [v for v in (decode_speedup(r) for r in model["results"]) if v]


def has_decode_baseline(model):
    """True once any fixture carries a released-crate decode number — the decode
    lane ran for this model, whether or not the pipeline can decode it yet."""
    return any((r.get("decode_mbps") or {}).get("baseline") for r in model["results"])


def base_speedup(row, base_lookup, model_name):
    """This PR's pipeline throughput ÷ the base branch's, for the same (model,
    fixture) — the "did this PR help vs base" ratio. `None` when the base branch
    didn't bench that fixture (added model, renamed fixture …)."""
    p = row["mbps"].get("pipeline")
    b = base_lookup.get((model_name, row["fixture"]))
    return p / b if b and p else None


def base_model_speedups(model, base_lookup):
    return [v for v in (base_speedup(r, base_lookup, model["model"])
                        for r in model["results"]) if v]


def alloc_rows(model, impl, lane="encode"):
    """fixture -> that fixture's exact allocation row ({count, bytes,
    input_bytes}) from one implementation's memory child, for the encode or the
    decode pass; {} without the lane (an older run, or a decode the impl can't do)."""
    mem = model.get("memory")
    d = mem.get(impl) if isinstance(mem, dict) else None
    allocs = d.get("allocs") if isinstance(d, dict) else None
    rows = allocs.get(lane) if isinstance(allocs, dict) else None
    return {r["fixture"]: r for r in rows} if rows else {}


def alloc_ratio(model, fixture, lane="encode"):
    """Release allocation count over pipeline allocation count for one fixture:
    ×2 means the pipeline allocates half as often."""
    b = alloc_rows(model, "baseline", lane).get(fixture)
    p = alloc_rows(model, "pipeline", lane).get(fixture)
    return b["count"] / p["count"] if b and p and p["count"] else None


def model_alloc_ratios(model, lane="encode"):
    return [v for v in (alloc_ratio(model, r["fixture"], lane)
                        for r in model["results"]) if v]


def scale(models, speedups_of=model_speedups):
    """Shared log2 x-range across every plotted speedup."""
    vals = [v for m in models for v in speedups_of(m)]
    if not vals:
        return 0.75, 1.5
    return min(0.75, min(vals) / 1.08), max(1.5, max(vals) * 1.08)


def nice_ticks(vmax):
    """Round ticks for a 0-anchored linear axis: a 1/2/2.5/5×10^k step chosen so
    ~5 ticks fit, never exceeding vmax."""
    raw = vmax / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    return [i * step for i in range(int(vmax / step) + 1)]


def thin_ticks(ticks, x, min_px=26, keep=None):
    """Drop axis ticks that would collide at the current scale; `keep` (the ×1.0
    baseline) always survives."""
    kept = [t for t in ticks if t == keep]
    for t in ticks:
        if t != keep and all(abs(x(t) - x(k)) >= min_px for k in kept):
            kept.append(t)
    return sorted(kept)


def footer_text(ink, height, meta, subtitle_base=""):
    """Run metadata as a single muted footer line, kept out of the header so a long
    shape/desc subtitle can't collide with the hardware string."""
    parts = [meta[0], meta[1]] + ([subtitle_base] if subtitle_base else [])
    return (f'<text x="16" y="{height - 12}" fill="{ink["muted"]}" font-size="10.5" '
            f'style="font-variant-numeric:tabular-nums">{escape(" · ".join(parts))}</text>')


def legend_row(ink, sink, y, entries):
    """Bottom legend: [(kind, key, label)] with kind ∈ swatch|tick|dot; `key` is a
    series key into `sink` or a raw color."""
    parts, lx = [], GUTTER
    for kind, key, label in entries:
        color = sink.get(key, key)
        if kind == "swatch":
            parts.append(f'<rect x="{lx}" y="{y - 9}" width="11" height="11" rx="2" fill="{color}"/>')
        elif kind == "dot":
            parts.append(f'<circle cx="{lx + 5.5}" cy="{y - 3.5}" r="5" fill="{color}" '
                         f'stroke="{ink["surface"]}" stroke-width="2"/>')
        else:
            parts.append(f'<line x1="{lx + 5}" y1="{y - 10}" x2="{lx + 5}" y2="{y + 2}" '
                         f'stroke="{color}" stroke-width="1.5"/>')
        parts.append(f'<text x="{lx + 16}" y="{y}" fill="{ink["secondary"]}" '
                     f'font-size="11.5">{escape(label)}</text>')
        lx += 16 + 8 + len(label) * 6.6 + 18
    return "".join(parts)


def svg_doc(ink, height, title, subtitle, body, meta, subtitle_base=""):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}"
  viewBox="0 0 {CHART_W} {height}" font-family="{FONT}">
<rect width="{CHART_W}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">{escape(title)}</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
{body}
{footer_text(ink, height, meta, subtitle_base)}
</svg>'''


def speedup_axis(ink, x, ticks, top, bottom, hints=("← slower", "faster →")):
    grid = []
    for t in ticks:
        strong = t == 1.0
        grid.append(f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{bottom - 4}" '
                    f'stroke="{ink["baseline"] if strong else ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{x(t):.1f}" y="{bottom + 10}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">×{t:g}</text>')
    hints = (f'<text x="{x(1.0) - 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="end">{escape(hints[0])}</text>'
             f'<text x="{x(1.0) + 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="start">{escape(hints[1])}</text>')
    return "".join(grid) + hints


def linear_axis(ink, x, ticks, top, y, unit):
    """0-anchored linear gridlines + labels (`unit` on the last tick), from `top`
    down to `y`. Shared by the memory / binary-size / thread-scaling charts."""
    grid = []
    for tv in ticks:
        u = unit if tv == ticks[-1] else ""
        grid.append(f'<line x1="{x(tv):.1f}" y1="{top - 6}" x2="{x(tv):.1f}" y2="{y - 4}" '
                    f'stroke="{ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{x(tv):.1f}" y="{y + 10}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{tv:g}{u}</text>')
    return "".join(grid)


def overview_svg(models, subtitle_base, meta, lo, hi, baseline_label,
                 speedups_of=model_speedups, title=None, ref_label=None,
                 mark_regressions=False, no_cmp_msg=None,
                 hints=("← slower", "faster →"), quantity="×speedup",
                 tick_set=TICKS, gate_key="ids_match", gate_word="ids",
                 pending_of=None):
    """Headline chart: one row per model — name + workload desc, geomean ×speedup vs
    the reference (×1.0) with a min–max whisker across fixtures. Unsupported models
    show as muted status rows so the overview is the complete state of the world.
    `mark_regressions=True` (the "vs base branch" twin) turns slower-than-base bars red.
    `speedups_of`/`hints`/`quantity`/`tick_set` let the allocation ratio chart
    reuse the layout: any per-fixture "×higher is better" ratio plots here.
    `gate_key`/`gate_word` name the correctness gate counted in the right column
    (encode ids, or decoded text); `pending_of` returns a per-model status line for
    a lane that hasn't landed for that model — the decode twin's "pending" row."""
    ink, sink = INK, SERIES_INK
    title = title or "PipelineTokenizer vs latest release — encode throughput"
    ref_label = ref_label or baseline_label
    x = log_x(OV_GUTTER, OV_PLOT, lo, hi)
    ticks = thin_ticks([t for t in tick_set if lo <= t <= hi], x, min_px=34, keep=1.0)

    top, row_h = 74, 40
    col_x = CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">{escape(quantity)}: Q1 · med · Q3</text>']
    y = top
    for m in models:
        cy = y + row_h / 2
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy - 3:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{escape(m["model"])}</text>')
        desc = m.get("desc") or m["shape"]
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy + 11:.1f}" fill="{ink["muted"]}" '
                    f'font-size="10.5" text-anchor="end">{escape(desc)}</text>')
        vals = speedups_of(m)
        if vals:
            g, mn, mx = geomean(vals), min(vals), max(vals)
            bar_color = ink["critical"] if (mark_regressions and g < 1) else sink["pipeline"]
            body.append(hbar(x(1.0), x(g), cy - 7, 14, bar_color))
            body.append(f'<line x1="{x(mn):.1f}" y1="{cy:.1f}" x2="{x(mx):.1f}" y2="{cy:.1f}" '
                        f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
            for v in (mn, mx):
                body.append(f'<line x1="{x(v):.1f}" y1="{cy - 4:.1f}" x2="{x(v):.1f}" y2="{cy + 4:.1f}" '
                            f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
            anchor, lx = (("start", max(x(mx), x(1.0)) + 8) if g >= 1
                          else ("end", min(x(mn), x(1.0)) - 8))
            if anchor == "end" and lx - 40 < OV_GUTTER + 4:
                anchor, lx = "start", max(x(mx), x(1.0)) + 8
            body.append(f'<text x="{lx:.1f}" y="{cy + 4:.1f}" fill="{ink["primary"]}" font-size="12" '
                        f'font-weight="600" text-anchor="{anchor}" '
                        f'style="font-variant-numeric:tabular-nums">{ratio_label(g)}</text>')
            bad = sum(1 for r in m["results"] if r.get(gate_key) is False)
            right, fill = ((f"⚠ {bad} {gate_word} differ", ink["critical"]) if bad
                           else (quartile_cell(vals), ink["secondary"]))
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{fill}" font-size="12" '
                        f'text-anchor="end" style="font-variant-numeric:tabular-nums">{right}</text>')
        elif m["results"]:
            msg = ((pending_of(m) if pending_of else None) or no_cmp_msg
                   or f"benched, but {baseline_label} can’t load this model — no comparison")
            body.append(f'<text x="{x(1.0) + 8:.1f}" y="{cy + 4:.1f}" fill="{ink["muted"]}" '
                        f'font-size="11.5" font-style="italic">{escape(msg)}</text>')
        else:
            pretok = m["shape"].split("·")[-1].strip()
            why = (m.get("reason") or f"no {pretok} pre-tokenizer")
            body.append(f'<text x="{x(1.0) + 8:.1f}" y="{cy + 4:.1f}" fill="{ink["muted"]}" '
                        f'font-size="11.5" font-style="italic">not supported — {escape(why)}</text>')
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["muted"]}" font-size="12" '
                        f'text-anchor="end">—</text>')
        y += row_h

    axis = speedup_axis(ink, x, ticks, top, y + 4, hints=hints)
    y += 30
    legend = legend_row(ink, sink, y, [
        ("swatch", "pipeline", "PipelineTokenizer"),
        ("tick", ink["baseline"], f"×1.0 = {ref_label}"),
    ])
    height = y + 34
    subtitle = (f"geomean {quantity} per model vs {ref_label} · "
                f"whisker: min–max across fixtures · {subtitle_base}")
    return svg_doc(ink, height, title, subtitle, axis + "".join(body) + legend, meta)


def fixture_rows(models):
    """(group, fixture) -> the per-model speedups of every benched model that ran it."""
    rows = {}
    for m in models:
        for r in m.get("results") or []:
            v = speedup(r)
            if v:
                rows.setdefault((r["group"], r["fixture"]), []).append(v)
    return rows


def fixture_overview_svg(models, subtitle_base, meta, lo, hi, baseline_label):
    """Headline twin of `overview_svg`, cut the other way: one row per fixture,
    geomean ×speedup across every benched model, with a min–max whisker across
    models. The per-model overview answers "which models got faster"; this one
    answers "on which workloads", so a win or regression that only shows on code,
    CJK or the added-token fixtures is visible instead of averaged away. Shares
    its x-range with the per-model overview so bars are comparable between the
    two charts."""
    ink, sink = INK, SERIES_INK
    rows = fixture_rows(models)
    x = log_x(GUTTER, PLOT_W, lo, hi)
    ticks = thin_ticks([t for t in TICKS if lo <= t <= hi], x, min_px=34, keep=1.0)

    top = 74
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">×speedup: Q1 · med · Q3</text>']
    y = top
    for key, title in GROUPS:
        names = sorted(f for (g, f) in rows if g == key)
        if not names:
            continue
        body.append(f'<text x="{GUTTER}" y="{y + 12}" fill="{ink["secondary"]}" font-size="11" '
                    f'font-weight="600" letter-spacing="1.2" text-anchor="end" dx="-10">{title.upper()}</text>')
        y += 22
        for name in names:
            vals = rows[(key, name)]
            g, mn, mx = geomean(vals), min(vals), max(vals)
            cy = y + ROW_H / 2
            body.append(f'<text x="{GUTTER - 10}" y="{cy + 4:.1f}" fill="{ink["secondary"]}" '
                        f'font-size="12.5" text-anchor="end">{escape(name)}</text>')
            body.append(hbar(x(1.0), x(g), y + (ROW_H - BAR_H) / 2, BAR_H, sink["pipeline"]))
            body.append(f'<line x1="{x(mn):.1f}" y1="{cy:.1f}" x2="{x(mx):.1f}" y2="{cy:.1f}" '
                        f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
            for v in (mn, mx):
                body.append(f'<line x1="{x(v):.1f}" y1="{cy - 4:.1f}" x2="{x(v):.1f}" y2="{cy + 4:.1f}" '
                            f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
            anchor, lx = (("start", max(x(mx), x(1.0)) + 8) if g >= 1
                          else ("end", min(x(mn), x(1.0)) - 8))
            if anchor == "end" and lx - 40 < GUTTER + 4:
                anchor, lx = "start", max(x(mx), x(1.0)) + 8
            body.append(f'<text x="{lx:.1f}" y="{cy + 4:.1f}" fill="{ink["primary"]}" font-size="12" '
                        f'font-weight="600" text-anchor="{anchor}" '
                        f'style="font-variant-numeric:tabular-nums">×{g:.2f}</text>')
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["secondary"]}" font-size="12" '
                        f'text-anchor="end" style="font-variant-numeric:tabular-nums">'
                        f'{quartile_cell(vals)}</text>')
            y += ROW_H
        y += 10

    axis = speedup_axis(ink, x, ticks, top, y)
    y += 26
    legend = legend_row(ink, sink, y, [
        ("swatch", "pipeline", "PipelineTokenizer (geomean across models)"),
        ("tick", ink["baseline"], f"×1.0 = {baseline_label}"),
    ])
    height = y + 44
    subtitle = (f"geomean ×speedup per fixture across benched models vs {baseline_label} · "
                f"whisker: min–max across models")
    return svg_doc(ink, height, "PipelineTokenizer vs latest release — encode speedup by fixture",
                   subtitle, axis + "".join(body) + legend, meta, subtitle_base)


def memory_svg(models, meta, baseline_label, subtitle_base="",
               pass_key="encode_bytes", pass_label="encode",
               title="Memory footprint"):
    """Per model: resident-set delta of each implementation — load footprint plus
    the measured pass's delta as stacked segments, peak RSS as a tick. `pass_key`
    picks the encode or the decode pass (same chart, both directions). An impl
    whose child failed — or that can't decode yet — draws no bar and totals "—";
    a model where neither impl ran is dropped."""
    ink, sink = INK, SERIES_INK
    models = [m for m in models if isinstance(m.get("memory"), dict)]

    def mem(m, impl):
        d = m["memory"].get(impl)
        if not isinstance(d, dict):
            return None
        pick = {"load_bytes": "load_bytes", "pass": pass_key, "peak_bytes": "peak_bytes"}
        return {k: max(0, d[src]) / 1e6 if d.get(src) is not None else None
                for k, src in pick.items()}

    models = [m for m in models
              if any((mem(m, impl) or {}).get("pass") is not None
                     for impl in ("baseline", "pipeline"))]

    vals = []
    for m in models:
        for impl in ("baseline", "pipeline"):
            d = mem(m, impl)
            if d and d["load_bytes"] is not None and d["pass"] is not None:
                vals.append(d["load_bytes"] + d["pass"])
                if d["peak_bytes"]:
                    vals.append(d["peak_bytes"])
    if not vals:
        return svg_doc(ink, 120, title, "no data", "", meta)
    max_mb = max(vals) * 1.05

    def x(v):
        return OV_GUTTER + v / max_mb * OV_PLOT_W

    top, bar_h, row_h = 78, 12, 2 * (12 + 3) + 16
    col_x = CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">MB: {escape(baseline_label)} → Pipeline</text>',
            f'<text x="{OV_GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'smaller is better · solid: after load · translucent: '
            f'{escape(pass_label)}-pass delta</text>']
    y = top
    for m in models:
        cy = y + row_h / 2
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{escape(m["model"])}</text>')
        totals = []
        by = y + 8
        for impl in ("baseline", "pipeline"):
            d = mem(m, impl)
            if not d or d["load_bytes"] is None or d["pass"] is None:
                totals.append(None)
                by += bar_h + 3
                continue
            load, enc = d["load_bytes"], d["pass"]
            totals.append(load + enc)
            body.append(f'<rect x="{x(0):.1f}" y="{by:.1f}" width="{max(1.5, x(load) - x(0)):.1f}" '
                        f'height="{bar_h}" rx="2" fill="{sink[impl]}"/>')
            if x(enc) - x(0) > 2:
                body.append(f'<rect x="{x(load) + 2:.1f}" y="{by:.1f}" '
                            f'width="{max(1.5, x(enc) - x(0) - 2):.1f}" height="{bar_h}" rx="2" '
                            f'fill="{sink[impl]}" fill-opacity="0.45"/>')
            if d["peak_bytes"]:
                px = x(d["peak_bytes"])
                body.append(f'<line x1="{px:.1f}" y1="{by - 2:.1f}" x2="{px:.1f}" '
                            f'y2="{by + bar_h + 2:.1f}" stroke="{ink["primary"]}" stroke-width="1.5"/>')
            by += bar_h + 3
        body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["secondary"]}" font-size="12" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">'
                    f'{chain(totals, "{:.0f}")}</text>')
        y += row_h

    grid = linear_axis(ink, x, nice_ticks(max_mb), top, y, " MB")
    y += 30
    legend = legend_row(ink, sink, y, [
        ("swatch", "baseline", baseline_label),
        ("swatch", "pipeline", "PipelineTokenizer"),
        ("tick", ink["primary"], "peak RSS (VmHWM)"),
    ])
    height = y + 34
    subtitle = ("resident-set delta per implementation, one process each · "
                f"load + {pass_label} pass")
    return svg_doc(ink, height, title, subtitle, grid + "".join(body) + legend, meta,
                   subtitle_base)


def chart_svg(model, subtitle_base, meta, lo, hi, baseline_label, lane="encode"):
    """Full-size per-fixture chart: the pipeline's ×speedup vs the release, with the
    `MB/s: release → Pipeline` throughput column. `lane` picks encode or decode —
    same layout, the other direction's numbers and correctness gate."""
    ink, sink = INK, SERIES_INK
    decode = lane == "decode"
    mbps_key, gate_key, gate_word = (("decode_mbps", "text_match", "decode") if decode
                                     else ("mbps", "ids_match", "ids"))
    speedup_of = decode_speedup if decode else speedup
    rows = model["results"]
    x = log_x(GUTTER, PLOT_W, lo, hi)
    ticks = thin_ticks([t for t in TICKS if lo <= t <= hi], x, min_px=34, keep=1.0)

    top = 74
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">MB/s: {escape(baseline_label)} → Pipeline</text>']
    y = top
    for key, title in GROUPS:
        # stable order (alphabetical) so a fixture keeps its row across runs —
        # not sorted by the (run-varying) speedup.
        group_rows = sorted((r for r in rows if r["group"] == key),
                            key=lambda r: r["fixture"])
        if not group_rows:
            continue
        body.append(f'<text x="{GUTTER}" y="{y + 12}" fill="{ink["secondary"]}" font-size="11" '
                    f'font-weight="600" letter-spacing="1.2" text-anchor="end" dx="-10">{title.upper()}</text>')
        y += 22
        for r in group_rows:
            body.append(f'<text x="{GUTTER - 10}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12.5" text-anchor="end">{escape(r["fixture"])}</text>')
            v = speedup_of(r)
            by = y + (ROW_H - BAR_H) / 2
            if v:
                body.append(hbar(x(1.0), x(v), by, BAR_H, sink["pipeline"]))
                txt = f"×{v:.2f}"
                fill = ink["primary"]
                if r.get(gate_key) is False:
                    txt += f"  ⚠ {gate_word} differ"
                    fill = ink["critical"]
                anchor, lx = ("start", max(x(1.0), x(v)) + 6) if v >= 1 else ("end", min(x(1.0), x(v)) - 6)
                # a long label left of a slow bar would run into the fixture
                # names — flip it to the empty space right of the ×1.0 axis
                if anchor == "end" and lx - len(txt) * 6.7 < GUTTER + 4:
                    anchor, lx = "start", x(1.0) + 6
                body.append(f'<text x="{lx:.1f}" y="{y + ROW_H / 2 + 4}" fill="{fill}" '
                            f'font-size="12" font-weight="600" text-anchor="{anchor}" '
                            f'style="font-variant-numeric:tabular-nums">{txt}</text>')
            mb = r.get(mbps_key) or {}
            body.append(f'<text x="{col_x}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12" text-anchor="end" style="font-variant-numeric:tabular-nums">'
                        f'{chain([mb.get("baseline"), mb.get("pipeline")])}</text>')
            y += ROW_H
        y += 10

    axis = speedup_axis(ink, x, ticks, top, y)
    y += 26
    legend = legend_row(ink, sink, y, [
        ("swatch", "pipeline", f"PipelineTokenizer{' decode' if decode else ''}"),
        ("tick", ink["baseline"], f"×1.0 = {baseline_label}"),
    ])
    height = y + 44

    parts = [model["shape"]]
    vals = decode_model_speedups(model) if decode else model_speedups(model)
    if vals:
        parts.append(f"geomean ×{geomean(vals):.2f} vs {baseline_label}")
    elif decode and model.get("decode_reason"):
        parts.append(f"decode pending — {model['decode_reason']}")
    else:
        parts.append(f"{baseline_label} can’t load this model — no comparison")
    if decode:
        parts.append("release-minted ids to both decoders · MB/s over the input bytes")
    return svg_doc(ink, height,
                   f'{model["model"]} — PipelineTokenizer {lane} throughput',
                   " · ".join(parts), axis + "".join(body) + legend, meta, subtitle_base)


def card_svg(model):
    """Compact roadmap card for a model the pipeline can't bench yet (or that failed
    to load). Plain single-chunk text only: cairosvg mis-centers tspans."""
    ink = INK
    pretok = model["shape"].split("·")[-1].strip()
    why = model.get("reason") or f"PipelineTokenizer has no {pretok} pre-tokenizer"
    header = 54
    b = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_W}" height="{CARD_H}" '
         f'viewBox="0 0 {CARD_W} {CARD_H}" font-family="{FONT}">',
         f'<rect x="0.5" y="0.5" width="{CARD_W - 1}" height="{CARD_H - 1}" rx="10" '
         f'fill="{ink["card"]}" stroke="{ink["border"]}" stroke-width="1"/>',
         f'<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">{escape(model["model"])}</text>',
         f'<text x="16" y="44" fill="{ink["muted"]}" font-size="11.5">{escape(model["shape"])}</text>',
         f'<rect x="{CARD_W - 16 - 96}" y="11" width="96" height="19" rx="9.5" fill="none" '
         f'stroke="{ink["border"]}" stroke-width="1"/>',
         f'<text x="{CARD_W - 16 - 48}" y="24" fill="{ink["muted"]}" font-size="11" '
         f'text-anchor="middle">not supported</text>',
         f'<line x1="16" y1="{header - 8}" x2="{CARD_W - 16}" y2="{header - 8}" stroke="{ink["grid"]}" stroke-width="1"/>',
         f'<text x="{CARD_W / 2}" y="{(CARD_H + header) / 2 - 2}" fill="{ink["secondary"]}" font-size="12.5" '
         f'font-weight="600" text-anchor="middle">Not benchmarked yet</text>',
         f'<text x="{CARD_W / 2}" y="{(CARD_H + header) / 2 + 16}" fill="{ink["muted"]}" font-size="11.5" '
         f'text-anchor="middle">{escape(why)}</text>',
         "</svg>"]
    return "".join(b)


def detect_hardware():
    try:
        cpu = Path("/proc/cpuinfo").read_text()
        cpu = next(l.split(":", 1)[1].strip() for l in cpu.splitlines()
                   if l.startswith("model name"))
    except (OSError, StopIteration):
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            import platform
            cpu = platform.processor() or platform.machine() or "unknown cpu"
    return f"{cpu} · {os.cpu_count()} cores"


def img_url(base, run_id, slug):
    if base:
        return f"{base}/pipeline-{run_id}-{slug}.png"
    return f"pipeline_bench_{slug}.png"


def picture(base, run_id, slug, alt, width):
    return f'<img alt="{escape(alt)}" src="{img_url(base, run_id, slug)}" width="{width}">'


def mem_line(model, baseline_label):
    mem = model["memory"]
    def part(impl, label):
        d = mem.get(impl)
        if not isinstance(d, dict):
            return f"{label} —"
        cell = lambda k: ("—" if d.get(k) is None else f"{max(0, d[k]) / 1e6:.0f}")
        return (f"{label} {cell('load_bytes')}+{cell('encode_bytes')}+{cell('decode_bytes')} "
                f"(peak {cell('peak_bytes')})")
    return ("**Memory** (RSS MB, load+encode+decode): "
            + " · ".join(part(i, l) for i, l in
                         (("baseline", baseline_label), ("pipeline", "Pipeline"))))


def human_count(n):
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            return f"{n / div:.1f}{suffix}"
    return f"{n:.0f}"


def ratio_label(v):
    """Decimals only carry meaning near ×1; allocation ratios reach thousands."""
    return f"×{v:,.0f}" if v >= 100 else f"×{v:.2f}"


def alloc_line(model, baseline_label, lane="encode"):
    """One exact-allocations line per model and pass: totals over what that pass
    measured, per implementation, plus the live-heap peak on the encode line (it is
    one number per child process, not per pass). Deterministic, so any movement
    here is a real change."""
    def part(impl, label):
        rows = alloc_rows(model, impl, lane)
        if not rows:
            return f"{label}: —"
        count = sum(r["count"] for r in rows.values())
        alloc_gb = sum(r["bytes"] for r in rows.values()) / 1e9
        cell = f"{label}: {human_count(count)} allocs, {alloc_gb:.2f} GB allocated"
        if lane == "encode":
            a = ((model.get("memory") or {}).get(impl) or {}).get("allocs") or {}
            cell += f", peak live {a['peak_live_bytes'] / 1e6:.0f} MB"
        return cell
    scope = "whole corpus" if lane == "encode" else "decode sample"
    return (f"**Allocations** ({lane} pass, {scope}): "
            + "; ".join(part(i, l) for i, l in
                        (("baseline", baseline_label), ("pipeline", "Pipeline"))))


# How far the canary may drift from its cached number before the report warns
# that the machines no longer match the cached baseline run. Single-number
# medians on shared runners carry a few percent of noise on their own.
CANARY_FENCE_PCT = 10.0


def canary_lines(data, benched):
    """The cached-baseline trust note: when the release's numbers were copied
    from a cached run, each model job re-measured one canary fixture; compare
    every canary against its cached number and clear or warn in one line.
    Empty when this run measured the release itself."""
    if not (data.get("baseline") or {}).get("cached"):
        return []
    drifts, fixture = [], None
    for m in benched:
        c = m.get("baseline_canary") or {}
        cur, old = c.get("measured_mbps"), c.get("cached_mbps")
        if cur and old:
            drifts.append(((cur / old - 1) * 100, m["model"]))
            fixture = c.get("fixture", fixture)
    if not drifts:
        return ["<sub>release numbers from the cached baseline run (no canary data)</sub>", ""]
    pct, model = max(drifts, key=lambda t: abs(t[0]))
    if abs(pct) > CANARY_FENCE_PCT:
        return [f"> ⚠️ **Release numbers are reused from a cached baseline run, and the "
                f"canary ({fixture}, re-measured by every model job) drifted {pct:+.1f}% on "
                f"{model}, beyond ±{CANARY_FENCE_PCT:g}%.** The runners changed since the "
                f"cache was seeded; read the vs-release throughput comparisons with "
                f"suspicion (allocation counts are unaffected). Any edit to "
                f"fixture_bench.rs re-keys the cache and re-measures.", ""]
    return [f"<sub>release numbers from the cached baseline run · canary ({fixture}) "
            f"re-measured by {len(drifts)} model job(s): max drift {pct:+.1f}% "
            f"(fence ±{CANARY_FENCE_PCT:g}%)</sub>", ""]


def base_work_lookup(base_data):
    """(model, fixture) -> the base run's pipeline allocation rows, one per pass.
    Feeds `work_verdict`; entries hold whatever the base run measured (older runs
    carry no allocation lane, and runs before decode landed no decode rows)."""
    out = {}
    for bm in base_data["models"]:
        lanes = {lane: alloc_rows(bm, "pipeline", lane) for lane in ("encode", "decode")}
        for r in bm.get("results") or []:
            out[(bm["model"], r["fixture"])] = {
                lane: rows.get(r["fixture"]) for lane, rows in lanes.items()
            }
    return out


def work_verdict(benched, base_work, base_ref):
    """The allocation lane's vs-base verdict, one line per pass: clear the PR in
    one glance or name exactly where work moved. Allocation counts are exact, so
    any difference is a change."""
    def listing(moves):
        moves.sort(key=lambda t: -abs(t[0]))
        shown = ", ".join(f"{n} {p:+.2f}%" for p, n in moves[:6])
        return shown + (f", and {len(moves) - 6} more" if len(moves) > 6 else "")

    lines = [f"**Work vs base** (`{base_ref}`, allocation lane):", ""]
    for lane in ("encode", "decode"):
        moves, seen = [], 0
        for m in benched:
            rows = alloc_rows(m, "pipeline", lane)
            for r in m["results"]:
                prev = base_work.get((m["model"], r["fixture"])) or {}
                cura, olda = rows.get(r["fixture"]), prev.get(lane)
                if cura and olda:
                    seen += 1
                    if (cura["count"], cura["bytes"]) != (olda["count"], olda["bytes"]):
                        pct = ((cura["count"] / olda["count"] - 1) * 100
                               if olda["count"] else float("inf"))
                        moves.append((pct, f"{m['model']}/{r['fixture']}"))
        if not seen:
            lines.append(f"- {lane} allocations: no data in the base run yet")
        elif moves:
            lines.append(f"- {lane} allocations: ⚠ counts changed on {len(moves)} of "
                         f"{seen} fixtures: {listing(moves)}")
        else:
            lines.append(f"- {lane} allocations: ✓ exactly identical on all {seen} fixtures")
    return lines


def binsize_svg(sizes, meta, baseline_label):
    """Stripped size of a minimal release-built encode program (load a
    tokenizer.json, encode one string) linking each implementation — what the
    library adds to a shipped binary. Bars 0-anchored on a linear MB axis."""
    ink, sink = INK, SERIES_INK
    rows = [("baseline", baseline_label), ("pipeline", "PipelineTokenizer")]
    max_mb = max(sizes.values()) / 1e6 * 1.15

    def x(v):
        return OV_GUTTER + v / max_mb * OV_PLOT_W

    top, bar_h, row_h = 74, 16, 30
    body = [f'<text x="{OV_GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'smaller is better</text>']
    y = top
    for key, label in rows:
        mb = sizes[key] / 1e6
        cy = y + row_h / 2
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{escape(label)}</text>')
        body.append(hbar(x(0), x(mb), cy - bar_h / 2, bar_h, sink[key]))
        txt = f"{mb:.2f} MB"
        if key != "baseline":
            txt += f"  ({(sizes[key] / sizes['baseline'] - 1) * 100:+.0f}% vs {baseline_label})"
        body.append(f'<text x="{x(mb) + 8:.1f}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12" font-weight="600" '
                    f'style="font-variant-numeric:tabular-nums">{escape(txt)}</text>')
        y += row_h

    grid = linear_axis(ink, x, nice_ticks(max_mb), top, y, " MB")
    height = y + 44
    subtitle = ("stripped release build of a minimal encode program "
                "(load a tokenizer.json + encode one string)")
    return svg_doc(ink, height, "Binary size",
                   subtitle, grid + "".join(body), meta)


def group_sweeps(m, key="threads"):
    """The (group_key, sweep) pairs of a model's thread sweep that actually ran, in
    GROUPS order. The sweep is keyed by fixture group, one sweep per group."""
    t = m.get(key)
    if not isinstance(t, dict):
        return []
    return [(g, t[g]) for g, _ in GROUPS
            if isinstance(t.get(g), dict) and t[g].get("counts")]


def has_threads(m, key="threads"):
    return bool(group_sweeps(m, key))


def threads_svg(sweep, meta, baseline_label, title="Thread scaling", subtitle_base=""):
    """One fixture group's sweep: throughput (MB/s) at each swept thread count —
    pipeline vs the release — with a per-row *ideal linear* tick (the first sweep
    point scaled by the thread ratio) on the pipeline bar, so linear vs sub-linear
    scaling is visible at a glance alongside the pipeline↔release gap; the right
    column carries self-scaling % of linear."""
    ink, sink = INK, SERIES_INK
    counts, pipe, base = sweep["counts"], sweep["pipeline_mbps"], sweep["baseline_mbps"]
    n0 = counts[0] if counts else 1  # the sweep's first count anchors the linear reference
    p1 = pipe[0] if pipe and pipe[0] else None
    ideal = [p1 * n / n0 for n in counts] if p1 else []
    vals = [v for v in pipe if v] + [v for v in base if v] + ideal
    max_mb = (max(vals) if vals else 1.0) * 1.08

    def x(v):
        return OV_GUTTER + v / max_mb * OV_PLOT_W

    top, bar_h, gap = 78, 11, 3
    row_h = 2 * (bar_h + gap) + 16
    col_x = CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">Pipeline MB/s · self-scaling (% of linear)</text>',
            f'<text x="{OV_GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'higher is better · top bar {escape(baseline_label)}, bottom Pipeline · tick = ideal linear</text>']
    y = top
    for i, n in enumerate(counts):
        cy = y + row_h / 2
        label = f"{n} thread" + ("" if n == 1 else "s")
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{label}</text>')
        base_y, pipe_y = y + 8, y + 8 + bar_h + gap
        b = base[i] if i < len(base) else None
        if b is not None:
            body.append(hbar(x(0), x(b), base_y, bar_h, sink["baseline"]))
        p = pipe[i] if i < len(pipe) else None
        if p is not None:
            body.append(hbar(x(0), x(p), pipe_y, bar_h, sink["pipeline"]))
        # Ideal-linear reference for the pipeline: the bar reaching this tick == linear.
        if p1 is not None and n > n0:
            ix = x(p1 * n / n0)
            body.append(f'<line x1="{ix:.1f}" y1="{pipe_y - 2:.1f}" x2="{ix:.1f}" '
                        f'y2="{pipe_y + bar_h + 2:.1f}" stroke="{ink["primary"]}" stroke-width="1.5"/>')
        if p is not None and p1:
            sc = p / p1
            right = f"{p:.0f} · {sc:.1f}× ({sc / (n / n0) * 100:.0f}%)"
        else:
            right = f"{p:.0f}" if p is not None else "—"
        body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["secondary"]}" font-size="12" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{right}</text>')
        y += row_h

    grid = linear_axis(ink, x, nice_ticks(max_mb), top, y, " MB/s")
    y += 30
    legend = legend_row(ink, sink, y, [
        ("swatch", "baseline", baseline_label),
        ("swatch", "pipeline", "PipelineTokenizer"),
        ("tick", ink["primary"], f"ideal linear from {n0} threads"),
    ])
    height = y + 34
    scaling = ""
    if p1 and len(pipe) >= 2 and pipe[-1]:
        sc = pipe[-1] / p1
        scaling = (f" · pipeline {sc:.1f}× from {n0} to {counts[-1]} threads "
                   f"({sc / (counts[-1] / n0) * 100:.0f}% of linear)")
    subtitle = (f"throughput at N threads vs {baseline_label}; tick = linear scaling from "
                f"the {n0}-thread point, an upper bound where vCPUs share physical cores{scaling}")
    return svg_doc(ink, height, title,
                   subtitle, grid + "".join(body) + legend, meta, subtitle_base)


def has_sizes(m):
    t = m.get("input_sizes")
    return isinstance(t, dict) and bool(t.get("bytes"))


def size_label(b):
    return f"{b} B" if b < 1024 else f"{b // 1024} kB"


def sizes_svg(sweep, meta, baseline_label, subtitle_base=""):
    """Input-size response: single-thread throughput of both implementations at
    chunk sizes from chat-message (~256 B, per-call overhead dominates) to whole
    document (~256 kB), over one fixed corpus sample re-chunked per size. The
    headline charts measure at ~10 kB; this curve shows how much of that speedup
    survives at either end. Same row layout as the thread sweep; the right column
    carries the pipeline MB/s and the ×speedup at that size."""
    ink, sink = INK, SERIES_INK
    bts, pipe, base = sweep["bytes"], sweep["pipeline_mbps"], sweep["baseline_mbps"]
    vals = [v for v in pipe if v] + [v for v in base if v]
    max_mb = (max(vals) if vals else 1.0) * 1.08

    def x(v):
        return OV_GUTTER + v / max_mb * OV_PLOT_W

    top, bar_h, gap = 78, 11, 3
    row_h = 2 * (bar_h + gap) + 16
    col_x = CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">Pipeline MB/s · ×speedup</text>',
            f'<text x="{OV_GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'higher is better · top bar {escape(baseline_label)}, bottom Pipeline</text>']
    y = top
    for i, bsize in enumerate(bts):
        cy = y + row_h / 2
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{size_label(bsize)}</text>')
        base_y, pipe_y = y + 8, y + 8 + bar_h + gap
        b = base[i] if i < len(base) else None
        if b is not None:
            body.append(hbar(x(0), x(b), base_y, bar_h, sink["baseline"]))
        p = pipe[i] if i < len(pipe) else None
        if p is not None:
            body.append(hbar(x(0), x(p), pipe_y, bar_h, sink["pipeline"]))
        if p is not None and b:
            right = f"{p:.0f} · ×{p / b:.2f}"
        else:
            right = f"{p:.0f}" if p is not None else "—"
        body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["secondary"]}" font-size="12" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{right}</text>')
        y += row_h

    grid = linear_axis(ink, x, nice_ticks(max_mb), top, y, " MB/s")
    y += 30
    legend = legend_row(ink, sink, y, [
        ("swatch", "baseline", baseline_label),
        ("swatch", "pipeline", "PipelineTokenizer"),
    ])
    height = y + 34
    subtitle = (f"single-thread throughput per chunk size vs {baseline_label} · one corpus "
                f"sample re-chunked per size · headline charts measure ~10 kB")
    return svg_doc(ink, height, "Input-size response",
                   subtitle, grid + "".join(body) + legend, meta, subtitle_base)


def render_markdown(data, subtitle_base, meta, base, run_id, sizes,
                    base_lookup=None, base_ref=None, base_work=None, env=None):
    """Overview charts inline; everything per-model inside one <details> block, so
    the PR description stays a single screen."""
    models = data["models"]
    baseline_label = f'v{data["baseline"]["version"]}'
    benched = [m for m in models if m["results"]]
    unsupported = [m for m in models if not m["results"]]
    mismatches = [f"{m['model']}/{r['fixture']}"
                  for m in benched for r in m["results"] if r["ids_match"] is False]
    text_mismatches = [f"{m['model']}/{r['fixture']}"
                       for m in benched for r in m["results"]
                       if r.get("text_match") is False]
    has_allocs = any(model_alloc_ratios(m) for m in benched)
    has_decode = any(has_decode_baseline(m) for m in benched)

    md = ["## PipelineTokenizer benchmark", "",
          f"**{len(benched)} / {len(models)} models supported** — PipelineTokenizer vs "
          f"`tokenizers` {baseline_label} (latest release) · {subtitle_base}", "",
          f"`{meta[0]}` · {meta[1]}", "",
          picture(base, run_id, "overview", "Per-model encode throughput vs latest release", 860), "",
          picture(base, run_id, "fixtures",
                  "Per-fixture encode throughput vs latest release, across models", 860), ""]
    md += canary_lines(data, benched)
    if has_decode:
        md += ["**Decode** — the release encodes each fixture's decode sample "
               "(`add_special_tokens=true`) and both implementations decode those "
               "SAME ids with `skip_special_tokens=false`, so the comparison is "
               "decode alone. MB/s counts the input bytes the ids came from, the "
               "same denominator as the encode charts.", "",
               picture(base, run_id, "decode-overview",
                       "Per-model decode throughput vs latest release", 860), ""]
    if base_lookup:
        md += [f"**vs base branch** (`{escape(base_ref or 'base')}`) — per-model geomean ×speedup of "
               f"this PR's PipelineTokenizer against the base branch's; **regressions in red**.", "",
               picture(base, run_id, "base-overview", "Per-model encode throughput vs base branch", 860), ""]
    if base_work is not None:
        md += work_verdict(benched, base_work, base_ref or "base") + [""]
    if has_allocs:
        md += [picture(base, run_id, "allocs",
                       "Encode allocations vs latest release", 860), ""]
        if isinstance(env, dict):
            bits = " · ".join(str(env[k]) for k in ("cpu", "glibc") if env.get(k))
            if bits:
                md += [f"<sub>allocation lane measured on: {escape(bits)}</sub>", ""]
    md += [picture(base, run_id, "memory", "Per-model memory footprint", 860), ""]
    if has_decode:
        md += [picture(base, run_id, "decode-memory",
                       "Per-model decode memory footprint", 860), ""]
    if sizes:
        md += [picture(base, run_id, "binsize", "Minimal encode binary size", 860), ""]

    if mismatches:
        md += [f"> ⚠️ **Pipeline token ids diverge from `tokenizers` {baseline_label} on: "
               f"{', '.join(mismatches)}** — speedups there are meaningless until fixed.", ""]
    if text_mismatches:
        md += [f"> ⚠️ **Pipeline decode diverges from `tokenizers` {baseline_label} on: "
               f"{', '.join(text_mismatches)}** — decode speedups there are meaningless "
               f"until fixed.", ""]

    for m in benched:
        slug = slugify(m["model"])
        desc = m.get("desc") or m["shape"]
        vals = model_speedups(m)
        summary = (f"×{geomean(vals):.2f} vs {baseline_label}" if vals
                   else f"{baseline_label} can't load — no comparison")
        if base_lookup:
            bvals = base_model_speedups(m, base_lookup)
            if bvals:
                summary += f" · ×{geomean(bvals):.2f} vs base"
        dvals = decode_model_speedups(m)
        if dvals:
            summary += f" · decode ×{geomean(dvals):.2f}"
        elif m.get("decode_reason"):
            summary += " · decode pending"
        flag = " · ⚠ ids differ" if any(r["ids_match"] is False for r in m["results"]) else ""
        if any(r.get("text_match") is False for r in m["results"]):
            flag += " · ⚠ decode differs"
        md += [f"<details><summary><b>{escape(m['model'])}</b> — {escape(desc)} · "
               f"{summary}{flag}</summary>", ""]
        md += [picture(base, run_id, slug, f"{m['model']} speedup", 860), ""]
        if has_sizes(m):
            md += [picture(base, run_id, f"{slug}-sizes",
                           f"{m['model']} input-size response", 860), ""]
        for gkey, _ in group_sweeps(m):
            md += [picture(base, run_id, f"{slug}-threads-{gkey}",
                           f"{m['model']} thread scaling ({gkey})", 860), ""]
        if has_decode_baseline(m):
            md += [picture(base, run_id, f"{slug}-decode",
                           f"{m['model']} decode speedup", 860), ""]
            for gkey, _ in group_sweeps(m, "decode_threads"):
                md += [picture(base, run_id, f"{slug}-decode-threads-{gkey}",
                               f"{m['model']} decode thread scaling ({gkey})", 860), ""]
        md += [mem_line(m, baseline_label), ""]
        p_allocs = alloc_rows(m, "pipeline")
        if p_allocs:
            md += [alloc_line(m, baseline_label), ""]
        if alloc_rows(m, "pipeline", "decode"):
            md += [alloc_line(m, baseline_label, "decode"), ""]
        # Columns beyond the throughput core appear only when their lane ran, so
        # a local render without the memory children matches the old table exactly.
        cols = [("Fixture", "---"), ("Group", "---"),
                (f"{baseline_label} MB/s", "---:"), ("Pipeline MB/s", "---:"),
                ("Speedup", "---:")]
        show_decode = has_decode_baseline(m)
        if show_decode:
            cols.append(("Decode", "---:"))
        if p_allocs:
            cols.append(("allocs/MB", "---:"))
        if base_lookup:
            cols.append(("Δ base", "---:"))
        cols.append(("Ids", ":--"))
        md += ["| " + " | ".join(h for h, _ in cols) + " |",
               "|" + "|".join(a for _, a in cols) + "|"]
        for r in sorted(m["results"], key=lambda r: (r["group"], r["fixture"])):
            mb = r["mbps"]
            cells = [r["fixture"], r["group"], fnum(mb.get("baseline")),
                     fnum(mb.get("pipeline")), fnum(speedup(r), "×{:.2f}")]
            if show_decode:
                dv = decode_speedup(r)
                cells.append("⚠️ ≠ release" if r.get("text_match") is False
                             else fnum(dv, "×{:.2f}") if dv else "pending")
            if p_allocs:
                a, ar = p_allocs.get(r["fixture"]), alloc_ratio(m, r["fixture"])
                cells.append("—" if not a else
                             f"{a['count'] / (a['input_bytes'] / 1e6):,.0f}"
                             + (f" ({ratio_label(ar)})" if ar else ""))
            if base_lookup:
                cells.append(fnum(base_speedup(r, base_lookup, m["model"]), "×{:.2f}"))
            cells.append("⚠️ ≠ release" if r["ids_match"] is False else "match")
            md.append("| " + " | ".join(cells) + " |")
        md += ["", "</details>", ""]

    # Unsupported / failed-to-load models: roadmap cards, collapsed — status, not results.
    if unsupported:
        names = ", ".join(f"<code>{escape(m['model'])}</code>" for m in unsupported)
        md += [f"<details><summary><b>Not yet supported:</b> {names}</summary>", "", "<table>"]
        for i in range(0, len(unsupported), 2):
            md.append("<tr>")
            for m in unsupported[i:i + 2]:
                md += ['<td align="center" valign="top">',
                       picture(base, run_id, slugify(m["model"]), f"{m['model']} not supported", 420),
                       "</td>"]
            md.append("</tr>")
        md += ["</table>", "", "</details>", ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--subtitle",
                    default="whole corpus · ~10 kB chunks · cold caches · add_special_tokens on")
    ap.add_argument("--revision", default="")
    ap.add_argument("--img-base", default="", help="base URL for uploaded PNGs")
    ap.add_argument("--run-id", default="local")
    ap.add_argument("--binary-sizes", default="",
                    help="JSON file {baseline, pipeline} of stripped binary bytes")
    ap.add_argument("--base-bench", default="",
                    help="base branch's cached fixture_bench JSON — enables the 'vs base branch' chart")
    ap.add_argument("--base-ref", default="",
                    help="label for the base branch baseline (e.g. short sha)")
    args = ap.parse_args()

    rev = args.revision
    if not rev:
        try:
            rev = subprocess.run(["git", "rev-parse", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            rev = "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta = (f"{rev[:9]} · {stamp}", detect_hardware())

    data = json.loads(Path(args.results).read_text())
    models = data["models"]
    baseline_label = f'v{data["baseline"]["version"]}'
    sizes = json.loads(Path(args.binary_sizes).read_text()) if args.binary_sizes else None
    benched = [m for m in models if m["results"]]
    lo, hi = scale(benched)

    # "vs base branch": join the PR results against the base branch's cached run by
    # (model, fixture). base_lookup[(model, fixture)] = base pipeline MB/s;
    # base_work carries the allocation lane's rows.
    base_lookup, base_speedups_of, blo, bhi, base_work = None, None, lo, hi, None
    if args.base_bench and Path(args.base_bench).exists():
        base_data = json.loads(Path(args.base_bench).read_text())
        base_lookup = {(bm["model"], r["fixture"]): r["mbps"].get("pipeline")
                       for bm in base_data["models"] for r in bm["results"]}
        base_work = base_work_lookup(base_data)
        if any(base_lookup.values()):
            base_speedups_of = lambda m: base_model_speedups(m, base_lookup)  # noqa: E731
            blo, bhi = scale(benched, base_speedups_of)
        else:
            base_lookup = None

    out = Path(args.out_dir)
    (out / "pipeline_bench_overview.svg").write_text(
        overview_svg(models, args.subtitle, meta, lo, hi, baseline_label))
    (out / "pipeline_bench_fixtures.svg").write_text(
        fixture_overview_svg(models, args.subtitle, meta, lo, hi, baseline_label))
    if base_lookup:
        (out / "pipeline_bench_base-overview.svg").write_text(
            overview_svg(models, args.subtitle, meta, blo, bhi, baseline_label,
                         speedups_of=base_speedups_of,
                         title="PipelineTokenizer vs base branch — encode throughput",
                         ref_label=(args.base_ref or "base branch"), mark_regressions=True,
                         no_cmp_msg="not benched on the base branch"))
    (out / "pipeline_bench_memory.svg").write_text(
        memory_svg(models, meta, baseline_label, subtitle_base=args.subtitle))
    # Decode: the lane ran wherever the release could mint an id stream. A model
    # the pipeline can't decode yet still plots — baseline bar, "pending" pipeline.
    dlo, dhi = scale(benched, decode_model_speedups)
    if any(has_decode_baseline(m) for m in benched):
        (out / "pipeline_bench_decode-overview.svg").write_text(
            overview_svg(models, args.subtitle, meta, dlo, dhi, baseline_label,
                         speedups_of=decode_model_speedups,
                         title="PipelineTokenizer vs latest release — decode throughput",
                         gate_key="text_match", gate_word="decode",
                         pending_of=lambda m: (f"decode pending — {m['decode_reason']}"
                                               if m.get("decode_reason") else None),
                         no_cmp_msg=f"{baseline_label} can’t decode this model — no comparison"))
        (out / "pipeline_bench_decode-memory.svg").write_text(
            memory_svg(models, meta, baseline_label, subtitle_base=args.subtitle,
                       pass_key="decode_bytes", pass_label="decode",
                       title="Memory footprint — decode"))
    if sizes:
        (out / "pipeline_bench_binsize.svg").write_text(
            binsize_svg(sizes, meta, baseline_label))
    # The allocation lane's overview, only when the lane ran (it needs the
    # counting-allocator children, so local renders may skip it).
    if any(model_alloc_ratios(m) for m in benched):
        alo, ahi = scale(benched, model_alloc_ratios)
        (out / "pipeline_bench_allocs.svg").write_text(
            overview_svg(models, args.subtitle, meta, alo, ahi, baseline_label,
                         speedups_of=model_alloc_ratios,
                         title="PipelineTokenizer vs latest release — encode allocations",
                         quantity="×fewer allocations",
                         hints=("← more allocations", "fewer →"),
                         tick_set=[0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0, 10000.0],
                         no_cmp_msg="allocation lane didn't run for this model"))

    for m in models:
        slug = slugify(m["model"])
        svg = (chart_svg(m, args.subtitle, meta, lo, hi, baseline_label)
               if m["results"] else card_svg(m))
        (out / f"pipeline_bench_{slug}.svg").write_text(svg)
        if has_sizes(m):
            (out / f"pipeline_bench_{slug}-sizes.svg").write_text(
                sizes_svg(m["input_sizes"], meta, baseline_label,
                          subtitle_base=args.subtitle))
        group_label = dict(GROUPS)
        for gkey, sweep in group_sweeps(m):
            (out / f"pipeline_bench_{slug}-threads-{gkey}.svg").write_text(
                threads_svg(sweep, meta, baseline_label,
                            title=f"Thread scaling — {group_label[gkey]}",
                            subtitle_base=args.subtitle))
        if has_decode_baseline(m):
            (out / f"pipeline_bench_{slug}-decode.svg").write_text(
                chart_svg(m, args.subtitle, meta, dlo, dhi, baseline_label, lane="decode"))
            for gkey, sweep in group_sweeps(m, "decode_threads"):
                (out / f"pipeline_bench_{slug}-decode-threads-{gkey}.svg").write_text(
                    threads_svg(sweep, meta, baseline_label,
                                title=f"Decode thread scaling — {group_label[gkey]}",
                                subtitle_base=args.subtitle))

    (out / "pipeline_bench.md").write_text(
        render_markdown(data, args.subtitle, meta, args.img_base, args.run_id, sizes,
                        base_lookup=base_lookup, base_ref=args.base_ref,
                        base_work=base_work, env=data.get("env")))

    # per-model geomeans only — a cross-model aggregate would average unrelated modes
    per_model = " · ".join(
        f"{m['model']} x{geomean(model_speedups(m)):.3f}"
        for m in benched if model_speedups(m))
    print(f"{len(benched)}/{len(models)} supported"
          + (f" · vs {baseline_label}: {per_model}" if per_model else ""))
    per_model_decode = " · ".join(
        f"{m['model']} x{geomean(decode_model_speedups(m)):.3f}"
        for m in benched if decode_model_speedups(m))
    if per_model_decode:
        print(f"decode vs {baseline_label}: {per_model_decode}")
    if base_lookup:
        vs_base = " · ".join(
            f"{m['model']} x{geomean(base_speedups_of(m)):.3f}"
            for m in benched if base_speedups_of(m))
        print(f"vs base ({args.base_ref or 'base'}): {vs_base}" if vs_base
              else "vs base: no overlapping fixtures")


if __name__ == "__main__":
    main()
