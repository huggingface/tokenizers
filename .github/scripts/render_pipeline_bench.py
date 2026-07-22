#!/usr/bin/env python3
"""Render the fixture_bench JSON as charts + a markdown PR report.

Two series: `baseline` (the latest released `tokenizers` crate — the bar to beat,
drawn gray; the in-tree `Tokenizer` isn't benched, it's only the id oracle behind
`ids_match`) and `pipeline` (the experimental PipelineTokenizer, blue). Leads with
three always-visible charts — per-model geomean ×speedup, memory footprint, binary
size — then one collapsed <details> per model (per-fixture speedup, thread scaling,
stage mix, numbers table). Models the pipeline can't build yet render as "not
supported" roadmap cards. Emits light-theme SVGs + pipeline_bench.md; the CI
workflow rasterizes and uploads the SVGs. Input schema: see fixture_bench.rs.

No cross-model aggregate anywhere on purpose: the models exercise different
execution modes (normalizer-heavy / split-heavy / model-bounded), so only
per-model geomeans mean anything.
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

# Pipeline encode stages in execution order, keyed to `stage_ns_per_byte`.
# `added_split` = added/special-token scan (AddedVocabulary), `pre_tokenize` =
# pre-tokenizer split — two distinct splitting costs. The four sum to `total`.
STAGES = [("added_split", "added-token"), ("normalize", "normalize"),
          ("pre_tokenize", "pre-tokenize"), ("model", "model")]
STAGE_INK = {"added_split": "#7a5ea8", "normalize": "#2a9d8f",
             "pre_tokenize": "#e0952b", "model": "#2a78d6"}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def fnum(v, spec="{:.1f}"):
    return spec.format(v) if v is not None else "—"


def chain(vals, spec="{:.1f}"):
    return " → ".join(fnum(v, spec) for v in vals)


def text_on(hex_color):
    """Black or white label, whichever reads on `hex_color` (relative luminance)."""
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return "#111111" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.6 else "#ffffff"


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


def speedup_axis(ink, x, ticks, top, bottom):
    grid = []
    for t in ticks:
        strong = t == 1.0
        grid.append(f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{bottom - 4}" '
                    f'stroke="{ink["baseline"] if strong else ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{x(t):.1f}" y="{bottom + 10}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">×{t:g}</text>')
    hints = (f'<text x="{x(1.0) - 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="end">← slower</text>'
             f'<text x="{x(1.0) + 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="start">faster →</text>')
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
                 mark_regressions=False, no_cmp_msg=None):
    """Headline chart: one row per model — name + workload desc, geomean ×speedup vs
    the reference (×1.0) with a min–max whisker across fixtures. Unsupported models
    show as muted status rows so the overview is the complete state of the world.
    `mark_regressions=True` (the "vs base branch" twin) turns slower-than-base bars red."""
    ink, sink = INK, SERIES_INK
    title = title or "PipelineTokenizer vs latest release — encode throughput"
    ref_label = ref_label or baseline_label
    x = log_x(OV_GUTTER, OV_PLOT, lo, hi)
    ticks = thin_ticks([t for t in TICKS if lo <= t <= hi], x, min_px=34, keep=1.0)

    top, row_h = 74, 40
    col_x = CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">fixtures · ids</text>']
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
                        f'style="font-variant-numeric:tabular-nums">×{g:.2f}</text>')
            bad = sum(1 for r in m["results"] if r["ids_match"] is False)
            right, fill = ((f"⚠ {bad} differ", ink["critical"]) if bad
                           else (f'{len(m["results"])} · ids ok', ink["secondary"]))
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{fill}" font-size="12" '
                        f'text-anchor="end" style="font-variant-numeric:tabular-nums">{right}</text>')
        elif m["results"]:
            msg = no_cmp_msg or f"benched, but {baseline_label} can’t load this model — no comparison"
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

    axis = speedup_axis(ink, x, ticks, top, y + 4)
    y += 30
    legend = legend_row(ink, sink, y, [
        ("swatch", "pipeline", "PipelineTokenizer"),
        ("tick", ink["baseline"], f"×1.0 = {ref_label}"),
    ])
    height = y + 34
    subtitle = (f"geomean ×speedup per model vs {ref_label} · "
                f"whisker: min–max across fixtures · {subtitle_base}")
    return svg_doc(ink, height, title, subtitle, axis + "".join(body) + legend, meta)


def memory_svg(models, meta, baseline_label):
    """Per model: resident-set delta of each implementation — load footprint plus
    the encode-pass delta as stacked segments, peak RSS as a tick."""
    ink, sink = INK, SERIES_INK
    models = [m for m in models if isinstance(m.get("memory"), dict)]

    def mem(m, impl):
        d = m["memory"].get(impl)
        if not isinstance(d, dict):
            return None
        return {k: max(0, d[k]) / 1e6 if d.get(k) is not None else None
                for k in ("load_bytes", "encode_bytes", "peak_bytes")}

    vals = []
    for m in models:
        for impl in ("baseline", "pipeline"):
            d = mem(m, impl)
            if d:
                vals.append((d["load_bytes"] or 0) + (d["encode_bytes"] or 0))
                if d["peak_bytes"]:
                    vals.append(d["peak_bytes"])
    if not vals:
        return svg_doc(ink, 120, "Memory footprint", "no data", "", meta)
    max_mb = max(vals) * 1.05

    def x(v):
        return OV_GUTTER + v / max_mb * OV_PLOT_W

    top, bar_h, row_h = 78, 12, 2 * (12 + 3) + 16
    col_x = CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">MB: {escape(baseline_label)} → Pipeline</text>',
            f'<text x="{OV_GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'smaller is better · solid: after load · translucent: encode-pass delta</text>']
    y = top
    for m in models:
        cy = y + row_h / 2
        body.append(f'<text x="{OV_GUTTER - 14}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{escape(m["model"])}</text>')
        totals = []
        by = y + 8
        for impl in ("baseline", "pipeline"):
            d = mem(m, impl)
            if not d:
                totals.append(None)
                by += bar_h + 3
                continue
            load, enc = d["load_bytes"] or 0, d["encode_bytes"] or 0
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
    subtitle = "resident-set delta per implementation, one process each · load + encode pass"
    return svg_doc(ink, height, "Memory footprint",
                   subtitle, grid + "".join(body) + legend, meta)


def chart_svg(model, subtitle_base, meta, lo, hi, baseline_label):
    """Full-size per-fixture chart: the pipeline's ×speedup vs the release, with the
    `MB/s: release → Pipeline` throughput column."""
    ink, sink = INK, SERIES_INK
    rows = model["results"]
    x = log_x(GUTTER, PLOT_W, lo, hi)
    ticks = thin_ticks([t for t in TICKS if lo <= t <= hi], x, min_px=34, keep=1.0)

    top = 74
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">MB/s: {escape(baseline_label)} → Pipeline</text>']
    y = top
    baseline_id_note = False
    for key, title in GROUPS:
        # stable order (alphabetical) so a fixture keeps its row across runs and
        # lines up with the stage chart — not sorted by the (run-varying) speedup.
        group_rows = sorted((r for r in rows if r["group"] == key),
                            key=lambda r: r["fixture"])
        if not group_rows:
            continue
        body.append(f'<text x="{GUTTER}" y="{y + 12}" fill="{ink["secondary"]}" font-size="11" '
                    f'font-weight="600" letter-spacing="1.2" text-anchor="end" dx="-10">{title.upper()}</text>')
        y += 22
        for r in group_rows:
            label = r["fixture"]
            if r.get("ids_match_baseline") is False:
                label += " †"
                baseline_id_note = True
            body.append(f'<text x="{GUTTER - 10}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12.5" text-anchor="end">{escape(label)}</text>')
            v = speedup(r)
            by = y + (ROW_H - BAR_H) / 2
            if v:
                body.append(hbar(x(1.0), x(v), by, BAR_H, sink["pipeline"]))
                txt = f"×{v:.2f}"
                fill = ink["primary"]
                if r["ids_match"] is False:
                    txt += "  ⚠ ids differ"
                    fill = ink["critical"]
                anchor, lx = ("start", max(x(1.0), x(v)) + 6) if v >= 1 else ("end", min(x(1.0), x(v)) - 6)
                # a long label left of a slow bar would run into the fixture
                # names — flip it to the empty space right of the ×1.0 axis
                if anchor == "end" and lx - len(txt) * 6.7 < GUTTER + 4:
                    anchor, lx = "start", x(1.0) + 6
                body.append(f'<text x="{lx:.1f}" y="{y + ROW_H / 2 + 4}" fill="{fill}" '
                            f'font-size="12" font-weight="600" text-anchor="{anchor}" '
                            f'style="font-variant-numeric:tabular-nums">{txt}</text>')
            mb = r["mbps"]
            body.append(f'<text x="{col_x}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12" text-anchor="end" style="font-variant-numeric:tabular-nums">'
                        f'{chain([mb.get("baseline"), mb.get("pipeline")])}</text>')
            y += ROW_H
        y += 10

    axis = speedup_axis(ink, x, ticks, top, y)
    y += 26
    legend = legend_row(ink, sink, y, [
        ("swatch", "pipeline", "PipelineTokenizer"),
        ("tick", ink["baseline"], f"×1.0 = {baseline_label}"),
    ])
    height = y + 44

    parts = [model["shape"]]
    vals = model_speedups(model)
    if vals:
        parts.append(f"geomean ×{geomean(vals):.2f} vs {baseline_label}")
    else:
        parts.append(f"{baseline_label} can’t load this model — no comparison")
    if baseline_id_note:
        parts.append(f"† ids differ from {baseline_label}")
    return svg_doc(ink, height, f'{model["model"]} — PipelineTokenizer encode throughput',
                   " · ".join(parts), axis + "".join(body) + legend, meta, subtitle_base)


def has_stages(model):
    return any("stage_ns_per_byte" in r for r in model["results"])


def stage_mix(rows):
    """Mean per-stage share of total, as a list of (label, fraction), largest first."""
    acc = {k: 0.0 for k, _ in STAGES}
    n = 0
    for r in rows:
        s = r.get("stage_ns_per_byte")
        if not s or not s["total"]:
            continue
        n += 1
        for k, _ in STAGES:
            acc[k] += s.get(k, 0.0) / s["total"]
    if not n:
        return []
    label = dict(STAGES)
    return sorted(((label[k], acc[k] / n) for k in acc), key=lambda kv: -kv[1])


def stage_cell(s, key):
    """One stage's cost as `share% (ns/B)` for the per-fixture table, else `—`."""
    if not s or not s.get("total") or key not in s:
        return "—"
    val, total = s[key], s["total"]
    return f"{100 * val / total:.0f}% ({val:.1f})"


def stage_chart_svg(model, subtitle_base, meta, baseline_label):
    """Per-fixture 100%-normalized stacked bar of where the pipeline spends its OWN
    encode time — each stage as its share of THAT fixture's total, labelled
    `share% · ns/B`. Normalizing per row (not on a shared ns/byte scale anchored to
    the slow release) keeps the mix readable at any absolute cost; the right column
    keeps the total ns/byte and the ×speedup vs the release."""
    ink = INK
    sink = STAGE_INK
    rows = [r for r in model["results"] if "stage_ns_per_byte" in r]

    top = 84
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">total ns/B · ×speedup</text>',
            f'<text x="{GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'share of pipeline encode time (each bar = 100%) · label = share% · ns/B</text>']
    y = top
    for key, title in GROUPS:
        group_rows = sorted((r for r in rows if r["group"] == key),
                            key=lambda r: r["fixture"])  # stable order, matches speedup chart
        if not group_rows:
            continue
        body.append(f'<text x="{GUTTER}" y="{y + 12}" fill="{ink["secondary"]}" font-size="11" '
                    f'font-weight="600" letter-spacing="1.2" text-anchor="end" dx="-10">{title.upper()}</text>')
        y += 22
        for r in group_rows:
            s = r["stage_ns_per_byte"]
            total = s["total"] or 1.0
            by = y + (ROW_H - BAR_H) / 2
            body.append(f'<text x="{GUTTER - 10}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12.5" text-anchor="end">{escape(r["fixture"])}</text>')
            cursor = float(GUTTER)
            for skey, _ in STAGES:
                val = s.get(skey, 0.0)
                frac = val / total
                seg = frac * PLOT_W  # each bar fills PLOT_W (100% of this fixture's total)
                if seg > 0.4:
                    body.append(f'<rect x="{cursor:.1f}" y="{by}" width="{seg:.1f}" '
                                f'height="{BAR_H}" fill="{sink[skey]}"/>')
                    # share% (plus ns/B when the slice is wide enough for both)
                    if seg >= 22:
                        pct = f"{100 * frac:.0f}%"
                        ns = f"{val:.0f}" if val >= 9.5 else f"{val:.1f}"
                        txt = f"{pct} · {ns}" if seg >= 62 else pct
                        body.append(f'<text x="{cursor + seg / 2:.1f}" y="{y + ROW_H / 2 + 4}" '
                                    f'fill="{text_on(sink[skey])}" font-size="10.5" '
                                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">'
                                    f'{txt}</text>')
                cursor += seg
            # magnitude + comparison kept in the right column (the % bar drops both)
            vs = speedup(r)
            body.append(f'<text x="{col_x}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12" text-anchor="end" style="font-variant-numeric:tabular-nums">'
                        f'{s["total"]:.1f} · {fnum(vs, "×{:.2f}")}</text>')
            y += ROW_H
        y += 10

    grid = []
    for pct in (0, 25, 50, 75, 100):
        gx = GUTTER + pct / 100 * PLOT_W
        grid.append(f'<line x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" y2="{y - 6}" '
                    f'stroke="{ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{gx:.1f}" y="{y + 12}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">'
                    f'{pct}{"%" if pct == 100 else ""}</text>')

    y += 26
    legend = legend_row(ink, sink, y, [("swatch", k, lbl) for k, lbl in STAGES])
    height = y + 34

    mix = stage_mix(rows)
    mix_txt = " · ".join(f"{lbl} {100 * frac:.0f}%" for lbl, frac in mix)
    subtitle = f'{model["shape"]} · stage mix: {mix_txt}'
    return svg_doc(ink, height, f'{model["model"]} — Pipeline encode stage mix',
                   subtitle, "".join(grid) + "".join(body) + legend, meta, subtitle_base)


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
        return f"{label} {cell('load_bytes')}+{cell('encode_bytes')} (peak {cell('peak_bytes')})"
    return ("**Memory** (RSS MB, load+encode): "
            + " · ".join(part(i, l) for i, l in
                         (("baseline", baseline_label), ("pipeline", "Pipeline"))))


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


def has_threads(m):
    t = m.get("threads")
    return isinstance(t, dict) and bool(t.get("counts"))


def threads_svg(model, meta, baseline_label):
    """Per model: encode throughput (MB/s) at 1/2/4/8/device-max threads — pipeline
    vs the release — with a per-row *ideal linear* tick (single-thread × N) on the
    pipeline bar, so linear vs sub-linear scaling is visible at a glance alongside
    the pipeline↔release gap; the right column carries self-scaling % of linear."""
    ink, sink = INK, SERIES_INK
    t = model["threads"]
    counts, pipe, base = t["counts"], t["pipeline_mbps"], t["baseline_mbps"]
    p1 = pipe[0] if pipe and pipe[0] else None  # single-thread anchor for the linear reference
    ideal = [p1 * n for n in counts] if p1 else []
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
        # Ideal-linear reference for the pipeline (single-thread × N): the bar reaching this tick == linear.
        if p1 is not None and n > 1:
            ix = x(p1 * n)
            body.append(f'<line x1="{ix:.1f}" y1="{pipe_y - 2:.1f}" x2="{ix:.1f}" '
                        f'y2="{pipe_y + bar_h + 2:.1f}" stroke="{ink["primary"]}" stroke-width="1.5"/>')
        if p is not None and p1:
            sc = p / p1
            right = f"{p:.0f} · {sc:.1f}× ({sc / n * 100:.0f}%)"
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
        ("tick", ink["primary"], "ideal linear (T₁ × N)"),
    ])
    height = y + 34
    scaling = ""
    if p1 and len(pipe) >= 2 and pipe[-1]:
        sc = pipe[-1] / p1
        scaling = f" · pipeline {sc:.1f}× on {counts[-1]} threads ({sc / counts[-1] * 100:.0f}% of linear)"
    subtitle = f"throughput at N threads vs {baseline_label}; tick = perfect linear scaling{scaling}"
    return svg_doc(ink, height, "Thread scaling",
                   subtitle, grid + "".join(body) + legend, meta)


def pretok_compare_md(model):
    """Per-fixture 'classify + fsm vs a regex engine' table — the pre-tokenize split
    vs onig/fancy/pcre2/logos on the model's own regex, WITH and WITHOUT SIMD.
    Rendered only for regex pre-tokenizers (a reference is non-null)."""
    engines = ("onig", "fancy", "pcre2", "logos")

    def has_ref(r):
        pv = r.get("pretok_vs_regex") or {}
        return r.get("stage_ns_per_byte") and any(pv.get(k) is not None for k in engines)
    rows = [r for r in model["results"] if has_ref(r)]
    if not rows:
        return []
    cell = lambda v: fnum(v, "{:.1f}")  # noqa: E731 — "—" for null
    def ratio(v, sp, cp):
        return f"{v / sp:.1f}× / {v / cp:.1f}×" if (v is not None and sp > 0 and cp > 0) else "—"
    cols = 4 + 2 * len(engines)  # cls×2 + pipe×2 + one abs + one ×vs per engine
    md = ["", "**Pre-tokenize: `classify + fsm` vs regex engines** — ns/byte, lower better. The fsm is "
          "the scalar jump-table in both pipe columns; **SIMD / scalar is the classify pass** (regex "
          "pre-tokenizers have no SIMD fsm). `×vs` = engine ÷ our pipeline (SIMD / scalar classify); "
          "`onig` & `pcre2` (JIT) are C, `fancy` is pure-Rust fancy-regex, `logos` is a compile-time DFA "
          "lexer (approximate grammar; n/a for deepseek).", "",
          "| Fixture | classify SIMD | classify scalar | pipe (SIMD cls + fsm) | pipe (scalar cls + fsm) | "
          + " | ".join(engines) + " | " + " | ".join(f"×vs {e}" for e in engines) + " |",
          "|---" + "|---:" * cols + "|"]
    for r in sorted(rows, key=lambda r: (r["group"], r["fixture"])):
        pv = r["pretok_vs_regex"]
        simd_pipe = r["stage_ns_per_byte"]["pre_tokenize"]
        scalar_pipe = simd_pipe + max(0.0, pv["cls_scalar"] - pv["cls_simd"])
        md.append(
            f"| {r['fixture']} | {fnum(pv['cls_simd'], '{:.2f}')} | {fnum(pv['cls_scalar'], '{:.2f}')} "
            f"| {fnum(simd_pipe, '{:.2f}')} | {fnum(scalar_pipe, '{:.2f}')} "
            + "".join(f"| {cell(pv.get(k))} " for k in engines)
            + " ".join(f"| {ratio(pv.get(k), simd_pipe, scalar_pipe)}" for k in engines)
            + " |")
    return md


def render_markdown(data, subtitle_base, meta, base, run_id, sizes,
                    base_lookup=None, base_ref=None):
    """Overview charts inline; everything per-model inside one <details> block, so
    the PR description stays a single screen."""
    models = data["models"]
    baseline_label = f'v{data["baseline"]["version"]}'
    benched = [m for m in models if m["results"]]
    unsupported = [m for m in models if not m["results"]]
    mismatches = [f"{m['model']}/{r['fixture']}"
                  for m in benched for r in m["results"] if r["ids_match"] is False]
    base_mismatch = sorted({m["model"] for m in benched
                            for r in m["results"] if r.get("ids_match_baseline") is False})

    md = ["## PipelineTokenizer benchmark", "",
          f"**{len(benched)} / {len(models)} models supported** — PipelineTokenizer vs "
          f"`tokenizers` {baseline_label} (latest release) · {subtitle_base}", "",
          f"`{meta[0]}` · {meta[1]}", "",
          picture(base, run_id, "overview", "Per-model encode throughput vs latest release", 860), ""]
    if base_lookup:
        md += [f"**vs base branch** (`{escape(base_ref or 'base')}`) — per-model geomean ×speedup of "
               f"this PR's PipelineTokenizer against the base branch's; **regressions in red**.", "",
               picture(base, run_id, "base-overview", "Per-model encode throughput vs base branch", 860), ""]
    md += [picture(base, run_id, "memory", "Per-model memory footprint", 860), ""]
    if sizes:
        md += [picture(base, run_id, "binsize", "Minimal encode binary size", 860), ""]

    if mismatches:
        md += [f"> ⚠️ **Pipeline token ids diverge from this tree's Tokenizer on: "
               f"{', '.join(mismatches)}** — speedups there are meaningless until fixed.", ""]
    if base_mismatch:
        md += [f"> ℹ️ Token ids differ from {baseline_label} on: {', '.join(base_mismatch)} "
               f"(† in the per-model charts) — expected when this branch fixes encode bugs, "
               f"but worth a look.", ""]

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
        flag = " · ⚠ ids differ" if any(r["ids_match"] is False for r in m["results"]) else ""
        md += [f"<details><summary><b>{escape(m['model'])}</b> — {escape(desc)} · "
               f"{summary}{flag}</summary>", ""]
        md += [picture(base, run_id, slug, f"{m['model']} speedup", 860), ""]
        if has_stages(m):
            md += [picture(base, run_id, f"{slug}-stages",
                           f"{m['model']} stage decomposition", 860), ""]
        if has_threads(m):
            md += [picture(base, run_id, f"{slug}-threads",
                           f"{m['model']} thread scaling", 860), ""]
        md += [mem_line(m, baseline_label), ""]
        # Per-stage columns carry each split's share of the pipeline's own encode
        # time with the ns/byte alongside — `share% (ns/B)` — readable as text
        # regardless of how slow the release baseline is.
        base_col = " Δ base |" if base_lookup else ""
        base_sep = "---:|" if base_lookup else ""
        md += [f"| Fixture | Group | {baseline_label} MB/s | Pipeline MB/s | Speedup |{base_col} "
               "added-token | normalize | pre-tokenize | model | Ids |",
               f"|---|---|---:|---:|---:|{base_sep}---:|---:|---:|---:|:--|"]
        for r in sorted(m["results"], key=lambda r: (r["group"], r["fixture"])):
            mb = r["mbps"]
            flags = []
            if r["ids_match"] is False:
                flags.append("⚠️ ≠ tree")
            if r.get("ids_match_baseline") is False:
                flags.append(f"≠ {baseline_label}")
            ids = " · ".join(flags) if flags else "match"
            s = r.get("stage_ns_per_byte")
            stages = " ".join(f"| {stage_cell(s, k)}"
                              for k in ("added_split", "normalize", "pre_tokenize", "model"))
            base_cell = (f"| {fnum(base_speedup(r, base_lookup, m['model']), '×{:.2f}')} "
                         if base_lookup else "")
            md.append(
                f"| {r['fixture']} | {r['group']} "
                f"| {fnum(mb.get('baseline'))} | {fnum(mb.get('pipeline'))} "
                f"| {fnum(speedup(r), '×{:.2f}')} "
                f"{base_cell}{stages} | {ids} |")
        md += pretok_compare_md(m)
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

    # Decode throughput: legacy `Tokenizer::decode` for every model that reports
    # it (the pipeline has no decode), so byte-level BPE is covered here too.
    decode_models = [m for m in models if m.get("decode_results")]
    if decode_models:
        md += ["<details><summary><b>Decode throughput (Tokenizer)</b></summary>", "",
               "Single-thread `Tokenizer::decode`, MB/s over decoded output bytes.", ""]
        for m in decode_models:
            rows = m["decode_results"]
            g = geomean([r["decode_mbps"] for r in rows]) if rows else 0.0
            md += [f"#### {escape(m['model'])} — {escape(m.get('shape', '?'))} · "
                   f"geomean {g:.1f} MB/s", "",
                   "| Fixture | Group | Decode MB/s |", "|---|---|---:|"]
            for r in sorted(rows, key=lambda r: (r["group"], -r["decode_mbps"])):
                md.append(f"| {r['fixture']} | {r['group']} | {r['decode_mbps']:.1f} |")
            md.append("")
        md += ["</details>", ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--subtitle", default="~10 kB inputs · single thread")
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
    # (model, fixture). base_lookup[(model, fixture)] = base pipeline MB/s.
    base_lookup, base_speedups_of, blo, bhi = None, None, lo, hi
    if args.base_bench and Path(args.base_bench).exists():
        base_data = json.loads(Path(args.base_bench).read_text())
        base_lookup = {(bm["model"], r["fixture"]): r["mbps"].get("pipeline")
                       for bm in base_data["models"] for r in bm["results"]}
        if any(base_lookup.values()):
            base_speedups_of = lambda m: base_model_speedups(m, base_lookup)  # noqa: E731
            blo, bhi = scale(benched, base_speedups_of)
        else:
            base_lookup = None

    out = Path(args.out_dir)
    (out / "pipeline_bench_overview.svg").write_text(
        overview_svg(models, args.subtitle, meta, lo, hi, baseline_label))
    if base_lookup:
        (out / "pipeline_bench_base-overview.svg").write_text(
            overview_svg(models, args.subtitle, meta, blo, bhi, baseline_label,
                         speedups_of=base_speedups_of,
                         title="PipelineTokenizer vs base branch — encode throughput",
                         ref_label=(args.base_ref or "base branch"), mark_regressions=True,
                         no_cmp_msg="not benched on the base branch"))
    (out / "pipeline_bench_memory.svg").write_text(
        memory_svg(models, meta, baseline_label))
    if sizes:
        (out / "pipeline_bench_binsize.svg").write_text(
            binsize_svg(sizes, meta, baseline_label))
    for m in models:
        slug = slugify(m["model"])
        svg = (chart_svg(m, args.subtitle, meta, lo, hi, baseline_label)
               if m["results"] else card_svg(m))
        (out / f"pipeline_bench_{slug}.svg").write_text(svg)
        if has_stages(m):
            (out / f"pipeline_bench_{slug}-stages.svg").write_text(
                stage_chart_svg(m, args.subtitle, meta, baseline_label))
        if has_threads(m):
            (out / f"pipeline_bench_{slug}-threads.svg").write_text(
                threads_svg(m, meta, baseline_label))

    (out / "pipeline_bench.md").write_text(
        render_markdown(data, args.subtitle, meta, args.img_base, args.run_id, sizes,
                        base_lookup=base_lookup, base_ref=args.base_ref))

    # per-model geomeans only — a cross-model aggregate would average unrelated modes
    per_model = " · ".join(
        f"{m['model']} x{geomean(model_speedups(m)):.3f}"
        for m in benched if model_speedups(m))
    print(f"{len(benched)}/{len(models)} supported"
          + (f" · vs {baseline_label}: {per_model}" if per_model else ""))
    if base_lookup:
        vs_base = " · ".join(
            f"{m['model']} x{geomean(base_speedups_of(m)):.3f}"
            for m in benched if base_speedups_of(m))
        print(f"vs base ({args.base_ref or 'base'}): {vs_base}" if vs_base
              else "vs base: no overlapping fixtures")


if __name__ == "__main__":
    main()
