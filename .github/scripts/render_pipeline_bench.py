#!/usr/bin/env python3
"""Render the fixture_bench JSON comparison as an SVG chart + markdown report.

Input: JSON array from `cargo run --release -p tk-encode --example fixture_bench`
(one row per fixture: legacy_mbps, pipeline_mbps, speedup, ids_match).

The chart is a diverging horizontal bar chart of per-fixture speedup on a log2
scale around the x1.0 baseline: blue = PipelineTokenizer faster, red = slower.
Colors follow the validated reference palette (light + dark variants).
"""
import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from html import escape
from pathlib import Path

INK = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
        "faster": "#2a78d6", "slower": "#e34948", "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835",
        "faster": "#3987e5", "slower": "#e66767", "critical": "#e66767",
    },
}
FONT = "-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
GUTTER, PLOT_W, PAD_R, COL_W, ROW_H, BAR_H = 150, 540, 110, 150, 26, 16
TICKS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
GROUPS = [("lang", "Languages"), ("modalities", "Modalities")]


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def bar_path(x0, x1, y, h, r):
    # square at the baseline (x0), rounded at the data end (x1)
    left, right = min(x0, x1), max(x0, x1)
    r = min(r, (right - left) / 2, h / 2)
    if x1 >= x0:
        return (f"M{left:.1f},{y:.1f} H{right - r:.1f} q{r:.1f},0 {r:.1f},{r:.1f} "
                f"V{y + h - r:.1f} q0,{r:.1f} -{r:.1f},{r:.1f} H{left:.1f} Z")
    return (f"M{right:.1f},{y:.1f} H{left + r:.1f} q-{r:.1f},0 -{r:.1f},{r:.1f} "
            f"V{y + h - r:.1f} q0,{r:.1f} {r:.1f},{r:.1f} H{right:.1f} Z")


def render_svg(rows, mode, subtitle, meta):
    ink = INK[mode]
    lo = min(0.75, min(r["speedup"] for r in rows) / 1.08)
    hi = max(1.5, max(r["speedup"] for r in rows) * 1.08)
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

    width = GUTTER + PLOT_W + PAD_R + COL_W
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
  viewBox="0 0 {width} {height}" font-family="{FONT}">
<rect width="{width}" height="{height}" fill="{ink["surface"]}"/>
<text x="16" y="26" fill="{ink["primary"]}" font-size="15" font-weight="600">PipelineTokenizer vs Tokenizer — encode throughput</text>
<text x="16" y="44" fill="{ink["secondary"]}" font-size="12">{escape(subtitle)}</text>
<text x="{width - 16}" y="26" fill="{ink["muted"]}" font-size="11" text-anchor="end"
  style="font-variant-numeric:tabular-nums">{escape(meta[0])}</text>
<text x="{width - 16}" y="44" fill="{ink["muted"]}" font-size="11" text-anchor="end">{escape(meta[1])}</text>
{"".join(grid)}
{hints}
{"".join(body)}
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


def render_markdown(rows, subtitle, meta, img_light, img_dark):
    overall = geomean([r["speedup"] for r in rows])
    parts = []
    for key, title in GROUPS:
        vals = [r["speedup"] for r in rows if r["group"] == key]
        if vals:
            parts.append(f"{title.lower()} ×{geomean(vals):.2f}")
    mismatches = [r["fixture"] for r in rows if not r["ids_match"]]

    md = ["## PipelineTokenizer benchmark",
          "",
          f"**Geomean speedup ×{overall:.2f}** across {len(rows)} fixtures "
          f"({', '.join(parts)}) — {subtitle}",
          "",
          f"`{meta[0]}` · {meta[1]}",
          ""]
    if mismatches:
        md += [f"> ⚠️ **Token ids diverge from the reference on: "
               f"{', '.join(mismatches)}** — speedups there are meaningless until fixed.",
               ""]
    if img_light:
        md += ["<picture>",
               f'  <source media="(prefers-color-scheme: dark)" srcset="{img_dark or img_light}">',
               f'  <img alt="Per-fixture speedup chart" src="{img_light}">',
               "</picture>",
               ""]
    md += ["<details><summary>Per-fixture results</summary>", "",
           "| Fixture | Group | Tokenizer MB/s | Pipeline MB/s | Speedup | Ids |",
           "|---|---|---:|---:|---:|:--|"]
    for r in sorted(rows, key=lambda r: (r["group"], -r["speedup"])):
        ids = "match" if r["ids_match"] else "⚠️ differ"
        md.append(f"| {r['fixture']} | {r['group']} | {r['legacy_mbps']:.1f} "
                  f"| {r['pipeline_mbps']:.1f} | ×{r['speedup']:.2f} | {ids} |")
    md += ["", "</details>", ""]
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--subtitle", default="bert-wiki WordPiece · ~10 kB inputs · single thread")
    ap.add_argument("--revision", default="")
    ap.add_argument("--img-light-url", default="")
    ap.add_argument("--img-dark-url", default="")
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

    rows = json.loads(Path(args.results).read_text())
    out = Path(args.out_dir)
    for mode in ("light", "dark"):
        (out / f"pipeline_bench_{mode}.svg").write_text(render_svg(rows, mode, args.subtitle, meta))
    (out / "pipeline_bench.md").write_text(
        render_markdown(rows, args.subtitle, meta, args.img_light_url, args.img_dark_url))
    print(f"geomean x{geomean([r['speedup'] for r in rows]):.3f}, "
          f"{sum(not r['ids_match'] for r in rows)} id mismatches")


if __name__ == "__main__":
    main()
