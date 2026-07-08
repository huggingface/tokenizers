#!/usr/bin/env python3
"""Render the multi-model fixture_bench JSON as full-size per-model charts +
a markdown report for a PR description.

Input: JSON array from `cargo run --release -p tk-encode --example fixture_bench`,
one object per model:
    {model, shape, supported, [reason], results: [{fixture, group,
     legacy_mbps, pipeline_mbps, speedup, ids_match,
     stage_ns_per_byte: {added_split, normalize, pre_tokenize, model, total}}, ...]}

The report leads with a single overview chart — one row per manifest model:
its workload `desc`, geomean ×speedup bar and a min–max whisker across
fixtures (no cross-model aggregate: the models exercise different execution
modes, so averaging them means nothing). Everything else is collapsed into
per-model <details> blocks.

Each supported model gets two full-size charts inside its block:
  1. a diverging bar chart (log2 around ×1.0): per-fixture ×speedup + the
     `MB/s: Tokenizer → Pipeline` throughput column, with group headers,
     slower/faster hints, ticks and an inline "⚠ ids differ" flag;
  2. a stacked stage-decomposition chart (ns/byte, lower is better — the
     opposite direction from the speedup chart, hence the explicit hint and
     the reference tick): where the pipeline spends its own encode time per
     fixture — added-token split + normalize + pre-tokenize + model — with a
     tick at the reference Tokenizer's total ns/byte on each row, so bar vs
     tick reads the same way as the ×speedup column.
Unsupported models (standalone ByteLevel, Unigram/Metaspace, …) get a compact
"not supported" card. Charts are rendered full-size so readers can zoom.
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
    "light": {
        "surface": "#fcfcfb", "card": "#ffffff", "primary": "#0b0b0b",
        "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
        "border": "#e1e0d9", "baseline": "#c3c2b7",
        "faster": "#2a78d6", "slower": "#e34948", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "card": "#212120", "primary": "#ffffff",
        "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
        "border": "#33332f", "baseline": "#4a4a46",
        "faster": "#3987e5", "slower": "#e66767", "critical": "#e66767",
    },
}
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
GUTTER, PLOT_W, PAD_R, COL_W, ROW_H, BAR_H = 150, 540, 110, 150, 26, 16
CHART_W = GUTTER + PLOT_W + PAD_R + COL_W
# Overview chart: wider gutter (model name + desc), same total width.
OV_GUTTER, OV_PLOT = 250, 440
TICKS = [0.5, 0.67, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
GROUPS = [("lang", "Languages"), ("modalities", "Modalities")]
CARD_W, CARD_H = 470, 150

# Pipeline encode stages, in execution order, keyed to `stage_ns_per_byte` in the
# fixture_bench JSON. `added_split` = the added/special-token scan (AddedVocabulary),
# `pre_tokenize` = the pre-tokenizer split — two distinct splitting costs. The four
# stages sum to the whole-encode total.
STAGES = [("added_split", "added-token"), ("normalize", "normalize"),
          ("pre_tokenize", "pre-tokenize"), ("model", "model")]
STAGE_INK = {
    "light": {"added_split": "#7a5ea8", "normalize": "#2a9d8f",
              "pre_tokenize": "#e0952b", "model": "#2a78d6"},
    "dark": {"added_split": "#a48ad4", "normalize": "#3fb8a8",
             "pre_tokenize": "#f0b45a", "model": "#3987e5"},
}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


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


def scale(models):
    """Shared log2 x-range across every supported fixture speedup, so charts
    for different models are directly comparable."""
    vals = [r["speedup"] for m in models if m["supported"] for r in m["results"]]
    if not vals:
        return 0.75, 1.5
    return min(0.75, min(vals) / 1.08), max(1.5, max(vals) * 1.08)


def thin_ticks(ticks, x, min_px=26):
    """Drop axis ticks that would collide at the current scale (e.g. ×0.75 and
    ×0.8 once a big speedup stretches the log range). ×1.0 always survives."""
    kept = [t for t in ticks if t == 1.0]
    for t in ticks:
        if t != 1.0 and all(abs(x(t) - x(k)) >= min_px for k in kept):
            kept.append(t)
    return sorted(kept)


def footer_text(ink, height, meta, subtitle_base=""):
    """Run metadata as a single muted footer line. Kept out of the header so a
    long shape/desc subtitle can never collide with the hardware string."""
    parts = [meta[0], meta[1]] + ([subtitle_base] if subtitle_base else [])
    return (f'<text x="16" y="{height - 12}" fill="{ink["muted"]}" font-size="10.5" '
            f'style="font-variant-numeric:tabular-nums">{escape(" · ".join(parts))}</text>')


def chart_svg(model, mode, subtitle_base, meta, lo, hi):
    """Full-size per-fixture speedup chart for a supported model."""
    ink = INK[mode]
    rows = model["results"]

    def x(v):
        return GUTTER + (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) * PLOT_W

    ticks = thin_ticks([t for t in TICKS if lo <= t <= hi], x)

    top = 74
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">MB/s: Tokenizer → Pipeline</text>']
    y = top
    for key, title in GROUPS:
        # stable order (alphabetical by fixture) so a fixture keeps its row across
        # runs and lines up with the stage-decomposition chart — not sorted by the
        # (run-varying) speedup.
        group_rows = sorted((r for r in rows if r["group"] == key),
                            key=lambda r: r["fixture"])
        if not group_rows:
            continue
        body.append(f'<text x="{GUTTER}" y="{y + 12}" fill="{ink["secondary"]}" font-size="11" '
                    f'font-weight="600" letter-spacing="1.2" text-anchor="end" dx="-10">{title.upper()}</text>')
        y += 22
        for r in group_rows:
            v, x0, x1 = r["speedup"], x(1.0), x(r["speedup"])
            if abs(math.log2(v)) < math.log2(1.02):  # within noise of the baseline
                color = ink["muted"]
            else:
                color = ink["faster"] if v >= 1 else ink["slower"]
            by = y + (ROW_H - BAR_H) / 2
            body.append(f'<text x="{GUTTER - 10}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12.5" text-anchor="end">{escape(r["fixture"])}</text>')
            if abs(x1 - x0) < 1.5:
                body.append(f'<rect x="{min(x0, x1):.1f}" y="{by}" width="1.5" height="{BAR_H}" fill="{color}"/>')
            else:
                body.append(f'<path d="{bar_path(x0, x1, by, BAR_H, 4)}" fill="{color}"/>')
            label = f"×{v:.2f}"
            anchor, lx = ("start", max(x0, x1) + 6) if v >= 1 else ("end", min(x0, x1) - 6)
            if not r["ids_match"]:
                label += "  ⚠ ids differ"
            fill = ink["primary"] if r["ids_match"] else ink["critical"]
            body.append(f'<text x="{lx:.1f}" y="{y + ROW_H / 2 + 4}" fill="{fill}" font-size="12" '
                        f'font-weight="600" text-anchor="{anchor}" '
                        f'style="font-variant-numeric:tabular-nums">{label}</text>')
            body.append(f'<text x="{col_x}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12" text-anchor="end" style="font-variant-numeric:tabular-nums">'
                        f'{r["legacy_mbps"]:.1f} → {r["pipeline_mbps"]:.1f}</text>')
            y += ROW_H
        y += 10

    height = y + 44
    grid = []
    for t in ticks:
        strong = t == 1.0
        grid.append(f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{y - 6}" '
                    f'stroke="{ink["baseline"] if strong else ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{x(t):.1f}" y="{y + 12}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">×{t:g}</text>')
    hints = (f'<text x="{x(1.0) - 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="end">← slower</text>'
             f'<text x="{x(1.0) + 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="start">faster →</text>')

    g = geomean([r["speedup"] for r in rows])
    subtitle = f'{model["shape"]} · geomean ×{g:.2f}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}"
  viewBox="0 0 {CHART_W} {height}" font-family="{FONT}">
<rect width="{CHART_W}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">{escape(model["model"])} — Pipeline vs Tokenizer encode throughput</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
{"".join(grid)}
{hints}
{"".join(body)}
{footer_text(ink, height, meta, subtitle_base)}
</svg>'''


def has_stages(model):
    return any("stage_ns_per_byte" in r for r in model["results"])


def ref_ns_per_byte(row):
    """The reference Tokenizer's whole-encode cost in the stage chart's unit
    (MB/s → ns/byte), so it can be drawn as a tick on the pipeline's stacked bar."""
    return 1000.0 / row["legacy_mbps"]


def stage_scale(models):
    """Shared linear ns/byte range across every supported fixture's total — and the
    reference Tokenizer's total, so the reference ticks always land on-plot — keeping
    the stacked stage bars comparable across models."""
    vals = [v for m in models if m["supported"]
            for r in m["results"] if "stage_ns_per_byte" in r
            for v in (r["stage_ns_per_byte"]["total"], ref_ns_per_byte(r))]
    vals = [v for v in vals if v > 0]
    return max(vals) * 1.05 if vals else 1.0


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


def stage_chart_svg(model, mode, subtitle_base, meta, max_total):
    """Per-fixture stacked decomposition of the *pipeline's* own encode time
    (ns/byte): normalize + split + model + other. Shows where the pipeline spends
    its time and — via the right column — how that maps onto the ×speedup headline."""
    ink = INK[mode]
    sink = STAGE_INK[mode]
    rows = [r for r in model["results"] if "stage_ns_per_byte" in r]

    def w(v):  # ns/byte -> px width on the shared scale
        return (v / max_total) * PLOT_W if max_total else 0.0

    top = 84
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    # Unlike the speedup chart above it, bars here grow with *time*: call the
    # direction out explicitly so the two charts can't be read the same way.
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">ns/byte · ×speedup</text>',
            f'<text x="{GUTTER}" y="{top - 14}" fill="{ink["muted"]}" font-size="11">'
            f'time per byte — shorter is faster · tick = Tokenizer total</text>']
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
            by = y + (ROW_H - BAR_H) / 2
            body.append(f'<text x="{GUTTER - 10}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12.5" text-anchor="end">{escape(r["fixture"])}</text>')
            cursor = float(GUTTER)
            for skey, _ in STAGES:
                val = s.get(skey, 0.0)
                seg = w(val)
                if seg > 0.4:
                    body.append(f'<rect x="{cursor:.1f}" y="{by}" width="{seg:.1f}" '
                                f'height="{BAR_H}" fill="{sink[skey]}"/>')
                    # ns/byte inside the slice when it's wide enough to read
                    if seg >= 24:
                        txt = f"{val:.0f}" if val >= 9.5 else f"{val:.1f}"
                        body.append(f'<text x="{cursor + seg / 2:.1f}" y="{y + ROW_H / 2 + 4}" '
                                    f'fill="{text_on(sink[skey])}" font-size="10.5" '
                                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">'
                                    f'{txt}</text>')
                cursor += seg
            # reference Tokenizer total on the same scale: bar shorter than the
            # tick ⇔ pipeline faster ⇔ ×speedup ≥ 1
            rx = GUTTER + w(ref_ns_per_byte(r))
            body.append(f'<line x1="{rx:.1f}" y1="{by - 3}" x2="{rx:.1f}" y2="{by + BAR_H + 3}" '
                        f'stroke="{ink["primary"]}" stroke-width="1.5"/>')
            body.append(f'<text x="{col_x}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                        f'font-size="12" text-anchor="end" style="font-variant-numeric:tabular-nums">'
                        f'{s["total"]:.1f} · ×{r["speedup"]:.2f}</text>')
            y += ROW_H
        y += 10

    # x-axis: 4 evenly spaced ns/byte gridlines + ticks
    grid = []
    for i in range(5):
        tv = max_total * i / 4
        gx = GUTTER + w(tv)
        grid.append(f'<line x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" y2="{y - 6}" '
                    f'stroke="{ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{gx:.1f}" y="{y + 12}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">{tv:.0f}</text>')

    # legend row: the stage colors + the reference-tick glyph
    y += 26
    legend = []
    lx = GUTTER
    for skey, lbl in STAGES:
        legend.append(f'<rect x="{lx}" y="{y - 9}" width="11" height="11" rx="2" fill="{sink[skey]}"/>')
        legend.append(f'<text x="{lx + 16}" y="{y}" fill="{ink["secondary"]}" font-size="11.5">{lbl}</text>')
        lx += 16 + 8 + len(lbl) * 7 + 18
    legend.append(f'<line x1="{lx + 5}" y1="{y - 10}" x2="{lx + 5}" y2="{y + 2}" '
                  f'stroke="{ink["primary"]}" stroke-width="1.5"/>')
    legend.append(f'<text x="{lx + 16}" y="{y}" fill="{ink["secondary"]}" font-size="11.5">'
                  f'Tokenizer total (reference)</text>')
    height = y + 34

    mix = stage_mix(rows)
    mix_txt = " · ".join(f"{lbl} {100 * frac:.0f}%" for lbl, frac in mix)
    subtitle = f'{model["shape"]} · stage mix: {mix_txt}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}"
  viewBox="0 0 {CHART_W} {height}" font-family="{FONT}">
<rect width="{CHART_W}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">{escape(model["model"])} — Pipeline encode stage decomposition</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
{"".join(grid)}
{"".join(body)}
{"".join(legend)}
{footer_text(ink, height, meta, subtitle_base)}
</svg>'''


def card_svg(model, mode):
    """Compact 'not supported' card for a model the pipeline can't build.
    Plain single-chunk text only: cairosvg mis-centers text-anchor with tspans."""
    ink = INK[mode]
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


def overview_svg(models, mode, subtitle_base, meta, lo, hi):
    """The one always-visible chart: a row per manifest model — name + workload
    desc, geomean ×speedup bar with a min–max whisker across fixtures — on the
    same log2 axis as the per-model charts. No cross-model aggregate on purpose:
    the models exercise different execution modes. Unsupported models appear as
    muted status rows so the overview is the complete state of the world."""
    ink = INK[mode]

    def x(v):
        return OV_GUTTER + (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) * OV_PLOT

    ticks = thin_ticks([t for t in TICKS if lo <= t <= hi], x)

    top = 74
    row_h = 40
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
        if m["supported"]:
            vals = [r["speedup"] for r in m["results"]]
            g, mn, mx = geomean(vals), min(vals), max(vals)
            if abs(math.log2(g)) < math.log2(1.02):  # within noise of the baseline
                color = ink["muted"]
            else:
                color = ink["faster"] if g >= 1 else ink["slower"]
            body.append(f'<path d="{bar_path(x(1.0), x(g), cy - 7, 14, 4)}" fill="{color}"/>')
            body.append(f'<line x1="{x(mn):.1f}" y1="{cy:.1f}" x2="{x(mx):.1f}" y2="{cy:.1f}" '
                        f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
            for v in (mn, mx):
                body.append(f'<line x1="{x(v):.1f}" y1="{cy - 4:.1f}" x2="{x(v):.1f}" y2="{cy + 4:.1f}" '
                            f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
            anchor, lx = (("start", max(x(mx), x(1.0)) + 8) if g >= 1
                          else ("end", min(x(mn), x(1.0)) - 8))
            body.append(f'<text x="{lx:.1f}" y="{cy + 4:.1f}" fill="{ink["primary"]}" font-size="12" '
                        f'font-weight="600" text-anchor="{anchor}" '
                        f'style="font-variant-numeric:tabular-nums">×{g:.2f}</text>')
            bad = sum(not r["ids_match"] for r in m["results"])
            right, fill = ((f"⚠ {bad} differ", ink["critical"]) if bad
                           else (f'{len(m["results"])} · ids ok', ink["secondary"]))
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{fill}" font-size="12" '
                        f'text-anchor="end" style="font-variant-numeric:tabular-nums">{right}</text>')
        else:
            pretok = m["shape"].split("·")[-1].strip()
            body.append(f'<text x="{x(1.0) + 8:.1f}" y="{cy + 4:.1f}" fill="{ink["muted"]}" '
                        f'font-size="11.5" font-style="italic">not supported — '
                        f'no {escape(pretok)} pre-tokenizer</text>')
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["muted"]}" font-size="12" '
                        f'text-anchor="end">—</text>')
        y += row_h

    grid = []
    for t in ticks:
        strong = t == 1.0
        grid.append(f'<line x1="{x(t):.1f}" y1="{top - 6}" x2="{x(t):.1f}" y2="{y - 2}" '
                    f'stroke="{ink["baseline"] if strong else ink["grid"]}" stroke-width="1"/>')
        grid.append(f'<text x="{x(t):.1f}" y="{y + 14}" fill="{ink["muted"]}" font-size="11" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">×{t:g}</text>')
    hints = (f'<text x="{x(1.0) - 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="end">← slower</text>'
             f'<text x="{x(1.0) + 8:.1f}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
             f'text-anchor="start">faster →</text>')

    height = y + 46
    subtitle = f'geomean ×speedup per model · whisker: min–max across fixtures · {subtitle_base}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}"
  viewBox="0 0 {CHART_W} {height}" font-family="{FONT}">
<rect width="{CHART_W}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">PipelineTokenizer vs Tokenizer — encode throughput by model</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
{"".join(grid)}
{hints}
{"".join(body)}
{footer_text(ink, height, meta)}
</svg>'''


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


def img_url(base, run_id, slug, mode):
    if base:
        return f"{base}/pipeline-{run_id}-{slug}-{mode}.png"
    return f"pipeline_bench_{slug}_{mode}.png"


def picture(base, run_id, slug, alt, width):
    return "\n".join([
        "<picture>",
        f'  <source media="(prefers-color-scheme: dark)" srcset="{img_url(base, run_id, slug, "dark")}">',
        f'  <img alt="{escape(alt)}" src="{img_url(base, run_id, slug, "light")}" width="{width}">',
        "</picture>"])


def render_markdown(models, subtitle_base, meta, base, run_id):
    """Overview chart inline; everything per-model — charts and the per-fixture
    table — inside one <details> block per model, so the PR description stays a
    single screen. No cross-model aggregate number anywhere: the models exercise
    different execution modes, so only per-model geomeans are meaningful."""
    supported = [m for m in models if m["supported"]]
    unsupported = [m for m in models if not m["supported"]]
    mismatches = [f"{m['model']}/{r['fixture']}"
                  for m in supported for r in m["results"] if not r["ids_match"]]

    md = ["## PipelineTokenizer benchmark", "",
          f"**{len(supported)} / {len(models)} models supported** — {subtitle_base}", "",
          f"`{meta[0]}` · {meta[1]}", "",
          picture(base, run_id, "overview", "Per-model encode throughput summary", 860), ""]

    if mismatches:
        md += [f"> ⚠️ **Token ids diverge from the reference on: {', '.join(mismatches)}** "
               f"— speedups there are meaningless until fixed.", ""]

    for m in supported:
        g = geomean([r["speedup"] for r in m["results"]])
        slug = slugify(m["model"])
        desc = m.get("desc") or m["shape"]
        flag = " · ⚠ ids differ" if any(not r["ids_match"] for r in m["results"]) else ""
        md += [f"<details><summary><b>{escape(m['model'])}</b> — {escape(desc)} · "
               f"geomean ×{g:.2f}{flag}</summary>", ""]
        md += [picture(base, run_id, slug, f"{m['model']} speedup", 860), ""]
        if has_stages(m):
            md += [picture(base, run_id, f"{slug}-stages",
                           f"{m['model']} stage decomposition", 860), ""]
        md += ["| Fixture | Group | Tokenizer MB/s | Pipeline MB/s | Speedup "
               "| added ns/B | norm ns/B | pre-tok ns/B | model ns/B | Ids |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|:--|"]
        for r in sorted(m["results"], key=lambda r: (r["group"], r["fixture"])):
            ids = "match" if r["ids_match"] else "⚠️ differ"
            s = r.get("stage_ns_per_byte", {})
            cells = " ".join(f"| {s[k]:.2f}" if k in s else "| —"
                             for k in ("added_split", "normalize", "pre_tokenize", "model"))
            md.append(f"| {r['fixture']} | {r['group']} | {r['legacy_mbps']:.1f} "
                      f"| {r['pipeline_mbps']:.1f} | ×{r['speedup']:.2f} {cells} | {ids} |")
        md += ["", "</details>", ""]

    # Unsupported models: roadmap cards, collapsed — they're status, not results.
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
    ap.add_argument("--subtitle", default="~10 kB inputs · single thread")
    ap.add_argument("--revision", default="")
    ap.add_argument("--img-base", default="", help="base URL for uploaded PNGs")
    ap.add_argument("--run-id", default="local")
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

    models = json.loads(Path(args.results).read_text())
    lo, hi = scale(models)
    max_total = stage_scale(models)
    out = Path(args.out_dir)
    for mode in ("light", "dark"):
        (out / f"pipeline_bench_overview_{mode}.svg").write_text(
            overview_svg(models, mode, args.subtitle, meta, lo, hi))
    for m in models:
        slug = slugify(m["model"])
        for mode in ("light", "dark"):
            svg = (chart_svg(m, mode, args.subtitle, meta, lo, hi)
                   if m["supported"] else card_svg(m, mode))
            (out / f"pipeline_bench_{slug}_{mode}.svg").write_text(svg)
            if m["supported"] and has_stages(m):
                (out / f"pipeline_bench_{slug}-stages_{mode}.svg").write_text(
                    stage_chart_svg(m, mode, args.subtitle, meta, max_total))

    (out / "pipeline_bench.md").write_text(
        render_markdown(models, args.subtitle, meta, args.img_base, args.run_id))

    supported = [m for m in models if m["supported"]]
    # per-model geomeans only — a cross-model aggregate would average unrelated
    # execution modes (normalizer-heavy vs split-heavy vs model-bounded)
    per_model = " · ".join(
        f"{m['model']} x{geomean([r['speedup'] for r in m['results']]):.3f}"
        for m in supported)
    print(f"{len(supported)}/{len(models)} supported"
          + (f" · {per_model}" if per_model else ""))


if __name__ == "__main__":
    main()
