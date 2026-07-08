#!/usr/bin/env python3
"""Render the multi-model fixture_bench JSON as full-size per-model charts +
a markdown report for a PR description.

Input: JSON array from `cargo run --release -p tk-encode --example fixture_bench`,
one object per model:
    {model, shape, supported, [reason], results: [{fixture, group,
     legacy_mbps, pipeline_mbps, speedup, ids_match,
     stage_ns_per_byte: {normalize, pre_tokenize, model, other, total}}, ...]}

Each supported model gets two full-size charts:
  1. a diverging bar chart (log2 around ×1.0): per-fixture ×speedup + the
     `MB/s: Tokenizer → Pipeline` throughput column, with group headers,
     slower/faster hints, ticks and an inline "⚠ ids differ" flag;
  2. a stacked stage-decomposition chart (ns/byte): where the pipeline spends
     its own encode time — normalize + split + model + other — per fixture,
     with each fixture's ×speedup alongside.
Unsupported models (byte-level BPE, Unigram/Metaspace, …) get a compact
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
TICKS = [0.5, 0.67, 0.75, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0]
GROUPS = [("lang", "Languages"), ("modalities", "Modalities")]
CARD_W, CARD_H = 470, 150

# Pipeline encode stages, in execution order, keyed to `stage_ns_per_byte` in the
# fixture_bench JSON. "other" is the untimed remainder (special-token scan + glue).
STAGES = [("normalize", "normalize"), ("pre_tokenize", "split"),
          ("model", "model"), ("other", "other")]
STAGE_INK = {
    "light": {"normalize": "#2a9d8f", "pre_tokenize": "#e0952b",
              "model": "#2a78d6", "other": "#c3c2b7"},
    "dark": {"normalize": "#3fb8a8", "pre_tokenize": "#f0b45a",
             "model": "#3987e5", "other": "#54544e"},
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


def chart_svg(model, mode, subtitle_base, meta, lo, hi):
    """Full-size per-fixture speedup chart for a supported model."""
    ink = INK[mode]
    rows = model["results"]
    ticks = [t for t in TICKS if lo <= t <= hi]

    def x(v):
        return GUTTER + (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) * PLOT_W

    top = 74
    col_x = GUTTER + PLOT_W + PAD_R + COL_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">MB/s: Tokenizer → Pipeline</text>']
    y = top
    for key, title in GROUPS:
        group_rows = sorted((r for r in rows if r["group"] == key),
                            key=lambda r: -r["speedup"])
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

    height = y + 34
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
    subtitle = f'{model["shape"]} · geomean ×{g:.2f} · {subtitle_base}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}"
  viewBox="0 0 {CHART_W} {height}" font-family="{FONT}">
<rect width="{CHART_W}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">{escape(model["model"])} — Pipeline vs Tokenizer encode throughput</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
<text x="{CHART_W - 16}" y="26" fill="{ink["muted"]}" font-size="11" text-anchor="end"
  style="font-variant-numeric:tabular-nums">{escape(meta[0])}</text>
<text x="{CHART_W - 16}" y="44" fill="{ink["muted"]}" font-size="11" text-anchor="end">{escape(meta[1])}</text>
{"".join(grid)}
{hints}
{"".join(body)}
</svg>'''


def has_stages(model):
    return any("stage_ns_per_byte" in r for r in model["results"])


def stage_scale(models):
    """Shared linear ns/byte range across every supported fixture's total, so the
    stacked stage bars are comparable across models."""
    vals = [r["stage_ns_per_byte"]["total"] for m in models if m["supported"]
            for r in m["results"] if "stage_ns_per_byte" in r]
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
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">ns/byte · ×speedup</text>']
    y = top
    for key, title in GROUPS:
        group_rows = sorted((r for r in rows if r["group"] == key),
                            key=lambda r: -r["stage_ns_per_byte"]["total"])
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

    # legend row
    y += 26
    legend = []
    lx = GUTTER
    for skey, lbl in STAGES:
        legend.append(f'<rect x="{lx}" y="{y - 9}" width="11" height="11" rx="2" fill="{sink[skey]}"/>')
        legend.append(f'<text x="{lx + 16}" y="{y}" fill="{ink["secondary"]}" font-size="11.5">{lbl}</text>')
        lx += 16 + 8 + len(lbl) * 7 + 18
    height = y + 18

    # Keep this subtitle short: the stage-mix string is long, and `subtitle_base`
    # (the "~10 kB inputs · single thread" run context) already rides in the markdown
    # header and the speedup chart above — appending it here collides with the
    # right-aligned hardware line.
    mix = stage_mix(rows)
    mix_txt = " · ".join(f"{lbl} {100 * frac:.0f}%" for lbl, frac in mix)
    subtitle = f'{model["shape"]} · stage mix: {mix_txt}'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{height}"
  viewBox="0 0 {CHART_W} {height}" font-family="{FONT}">
<rect width="{CHART_W}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="700">{escape(model["model"])} — Pipeline encode stage decomposition</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
<text x="{CHART_W - 16}" y="26" fill="{ink["muted"]}" font-size="11" text-anchor="end"
  style="font-variant-numeric:tabular-nums">{escape(meta[0])}</text>
<text x="{CHART_W - 16}" y="44" fill="{ink["muted"]}" font-size="11" text-anchor="end">{escape(meta[1])}</text>
{"".join(grid)}
{"".join(body)}
{"".join(legend)}
</svg>'''


def card_svg(model, mode):
    """Compact 'not supported' card for a model the pipeline can't build."""
    ink = INK[mode]
    pretok = model["shape"].split("·")[-1].strip()
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
         f'text-anchor="middle">PipelineTokenizer has no <tspan font-weight="600">{escape(pretok)}</tspan> pre-tokenizer</text>',
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
    supported = [m for m in models if m["supported"]]
    unsupported = [m for m in models if not m["supported"]]
    mismatches = [f"{m['model']}/{r['fixture']}"
                  for m in supported for r in m["results"] if not r["ids_match"]]

    md = ["## PipelineTokenizer benchmark", ""]
    head = f"**{len(supported)} / {len(models)} models supported**"
    if supported:
        g = geomean([r["speedup"] for m in supported for r in m["results"]])
        nfix = len(supported[0]["results"])
        head += f" · geomean ×{g:.2f} across supported · {nfix} fixtures each"
    md += [f"{head} — {subtitle_base}", "", f"`{meta[0]}` · {meta[1]}", ""]

    staged_rows = [r for m in supported for r in m["results"] if "stage_ns_per_byte" in r]
    if staged_rows:
        mix = ", ".join(f"{lbl} {100 * frac:.0f}%" for lbl, frac in stage_mix(staged_rows))
        md += [f"Pipeline stage mix (mean share of encode time): {mix}. "
               f"Per-model decomposition charts below.", ""]

    if unsupported:
        items = ", ".join(f"`{m['model']}` ({m['shape']})" for m in unsupported)
        md += [f"> **Not yet supported:** {items} — roadmap cards below.", ""]
    if mismatches:
        md += [f"> ⚠️ **Token ids diverge from the reference on: {', '.join(mismatches)}** "
               f"— speedups there are meaningless until fixed.", ""]

    # Supported models: speedup chart + stage-decomposition chart each
    # (readable inline, click to zoom).
    for m in supported:
        slug = slugify(m["model"])
        md += [picture(base, run_id, slug, f"{m['model']} speedup", 860), ""]
        if has_stages(m):
            md += [picture(base, run_id, f"{slug}-stages",
                           f"{m['model']} stage decomposition", 860), ""]

    # Unsupported models: compact roadmap cards, two per row.
    if unsupported:
        md += ["### Not yet supported", "", "<table>"]
        for i in range(0, len(unsupported), 2):
            md.append("<tr>")
            for m in unsupported[i:i + 2]:
                md += ['<td align="center" valign="top">',
                       picture(base, run_id, slugify(m["model"]), f"{m['model']} not supported", 420),
                       "</td>"]
            md.append("</tr>")
        md += ["</table>", ""]

    if supported:
        md += ["<details><summary>Per-fixture results</summary>", ""]
        for m in supported:
            md += [f"#### {m['model']} — {m['shape']}", "",
                   "| Fixture | Group | Tokenizer MB/s | Pipeline MB/s | Speedup "
                   "| norm ns/B | split ns/B | model ns/B | other ns/B | Ids |",
                   "|---|---|---:|---:|---:|---:|---:|---:|---:|:--|"]
            for r in sorted(m["results"], key=lambda r: (r["group"], -r["speedup"])):
                ids = "match" if r["ids_match"] else "⚠️ differ"
                s = r.get("stage_ns_per_byte", {})
                cells = " ".join(f"| {s[k]:.2f}" if k in s else "| —"
                                 for k in ("normalize", "pre_tokenize", "model", "other"))
                md.append(f"| {r['fixture']} | {r['group']} | {r['legacy_mbps']:.1f} "
                          f"| {r['pipeline_mbps']:.1f} | ×{r['speedup']:.2f} {cells} | {ids} |")
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
    g = geomean([r["speedup"] for m in supported for r in m["results"]]) if supported else 0
    print(f"{len(supported)}/{len(models)} supported, geomean x{g:.3f}")


if __name__ == "__main__":
    main()
