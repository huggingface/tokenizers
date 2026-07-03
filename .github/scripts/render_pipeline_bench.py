#!/usr/bin/env python3
"""Render the multi-model fixture_bench JSON as a grid of per-model SVG cards
+ a markdown report for a PR description.

Input: JSON array from `cargo run --release -p tk-encode --example fixture_bench`,
one object per model:
    {model, repo, shape, supported, [reason], results: [{fixture, group,
     legacy_mbps, pipeline_mbps, speedup, ids_match}, ...]}

Each model becomes one fixed-size card (light + dark SVG):
  • supported  → a diverging bar chart of per-fixture speedup (log2 around ×1.0):
                 blue = PipelineTokenizer faster, red = slower.
  • unsupported → a "not supported yet" placeholder showing the pipeline shape.
All cards share one x-scale (comparable) and one height (a tidy grid). The
markdown lays them out two-per-row and links each to its uploaded PNG.
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
TICKS = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
GROUPS = [("lang", "Languages"), ("modalities", "Modalities")]

CARD_W = 470
PAD = 16
HEADER_H = 56
GROUP_H = 22
ROW_H = 17
BAR_H = 10
AXIS_H = 28
LABEL_R = PAD + 94          # right edge of fixture labels
PLOT_L = LABEL_R + 10       # bars start here
VAL_W = 62                  # room for the ×value label
PLOT_R = CARD_W - PAD - VAL_W
PLOT_W = PLOT_R - PLOT_L


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def geomean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def bar_path(x0, x1, y, h, r):
    left, right = min(x0, x1), max(x0, x1)
    r = min(r, (right - left) / 2, h / 2)
    if x1 >= x0:
        return (f"M{left:.1f},{y:.1f} H{right - r:.1f} q{r:.1f},0 {r:.1f},{r:.1f} "
                f"V{y + h - r:.1f} q0,{r:.1f} -{r:.1f},{r:.1f} H{left:.1f} Z")
    return (f"M{right:.1f},{y:.1f} H{left + r:.1f} q-{r:.1f},0 -{r:.1f},{r:.1f} "
            f"V{y + h - r:.1f} q0,{r:.1f} {r:.1f},{r:.1f} H{right:.1f} Z")


def layout_of(models):
    """Ordered {group: fixture_count} from the first model that has results."""
    for m in models:
        if m["results"]:
            counts = {}
            for r in m["results"]:
                counts[r["group"]] = counts.get(r["group"], 0) + 1
            return counts
    return {}


def body_height(layout):
    h = sum(GROUP_H + layout[k] * ROW_H for k, _ in GROUPS if layout.get(k))
    return (h + AXIS_H) if h else 180


def scale(models):
    """Shared log2 x-range across every supported fixture speedup."""
    vals = [r["speedup"] for m in models if m["supported"] for r in m["results"]]
    if not vals:
        return 0.75, 1.5
    lo = min(0.75, min(vals) / 1.08)
    hi = max(1.5, max(vals) * 1.08)
    return lo, hi


def card_svg(model, mode, lo, hi, card_h):
    ink = INK[mode]
    b = [f'<rect x="0.5" y="0.5" width="{CARD_W - 1}" height="{card_h - 1}" rx="10" '
         f'fill="{ink["card"]}" stroke="{ink["border"]}" stroke-width="1"/>']

    # ── header ──────────────────────────────────────────────────────────
    b.append(f'<text x="{PAD}" y="26" fill="{ink["primary"]}" font-size="15" '
             f'font-weight="700">{escape(model["model"])}</text>')
    b.append(f'<text x="{PAD}" y="44" fill="{ink["muted"]}" font-size="11.5">'
             f'{escape(model["shape"])}</text>')

    if model["supported"]:
        rows = model["results"]
        g = geomean([r["speedup"] for r in rows])
        gc = ink["faster"] if g >= 1 else ink["slower"]
        b.append(f'<text x="{CARD_W - PAD}" y="24" fill="{gc}" font-size="19" '
                 f'font-weight="700" text-anchor="end" '
                 f'style="font-variant-numeric:tabular-nums">×{g:.2f}</text>')
        b.append(f'<text x="{CARD_W - PAD}" y="42" fill="{ink["muted"]}" font-size="10.5" '
                 f'text-anchor="end">geomean</text>')
    else:
        pill_w = 96
        b.append(f'<rect x="{CARD_W - PAD - pill_w}" y="11" width="{pill_w}" height="19" rx="9.5" '
                 f'fill="none" stroke="{ink["border"]}" stroke-width="1"/>')
        b.append(f'<text x="{CARD_W - PAD - pill_w / 2}" y="24" fill="{ink["muted"]}" '
                 f'font-size="11" text-anchor="middle">not supported</text>')

    b.append(f'<line x1="{PAD}" y1="{HEADER_H - 8}" x2="{CARD_W - PAD}" y2="{HEADER_H - 8}" '
             f'stroke="{ink["grid"]}" stroke-width="1"/>')

    # ── unsupported placeholder (compact card) ──────────────────────────
    if not model["supported"]:
        pretok = model["shape"].split("·")[-1].strip()
        cy = HEADER_H + (card_h - HEADER_H) / 2
        b.append(f'<text x="{CARD_W / 2}" y="{cy - 2}" fill="{ink["secondary"]}" font-size="12.5" '
                 f'font-weight="600" text-anchor="middle">Not benchmarked yet</text>')
        b.append(f'<text x="{CARD_W / 2}" y="{cy + 16}" fill="{ink["muted"]}" font-size="11.5" '
                 f'text-anchor="middle">PipelineTokenizer has no <tspan font-weight="600">'
                 f'{escape(pretok)}</tspan> pre-tokenizer</text>')
        return "".join(b)

    # ── supported: diverging bars ───────────────────────────────────────
    def x(v):
        return PLOT_L + (math.log2(v) - math.log2(lo)) / (math.log2(hi) - math.log2(lo)) * PLOT_W

    ticks = [t for t in TICKS if lo <= t <= hi]
    body_top = HEADER_H + 6
    y = body_top
    for key, title in GROUPS:
        group_rows = sorted((r for r in model["results"] if r["group"] == key),
                            key=lambda r: -r["speedup"])
        if not group_rows:
            continue
        b.append(f'<text x="{LABEL_R}" y="{y + 13}" fill="{ink["secondary"]}" font-size="10" '
                 f'font-weight="700" letter-spacing="1.1" text-anchor="end">{title.upper()}</text>')
        y += GROUP_H
        for r in group_rows:
            v, x0, x1 = r["speedup"], x(1.0), x(r["speedup"])
            if abs(math.log2(v)) < math.log2(1.02):
                color = ink["muted"]
            else:
                color = ink["faster"] if v >= 1 else ink["slower"]
            by = y + (ROW_H - BAR_H) / 2
            b.append(f'<text x="{LABEL_R}" y="{y + ROW_H / 2 + 4}" fill="{ink["secondary"]}" '
                     f'font-size="11.5" text-anchor="end">{escape(r["fixture"])}</text>')
            if abs(x1 - x0) < 1.5:
                b.append(f'<rect x="{min(x0, x1):.1f}" y="{by}" width="1.5" height="{BAR_H}" fill="{color}"/>')
            else:
                b.append(f'<path d="{bar_path(x0, x1, by, BAR_H, 3)}" fill="{color}"/>')
            label = f"×{v:.2f}"
            anchor, lx = ("start", max(x0, x1) + 5) if v >= 1 else ("end", min(x0, x1) - 5)
            fill = ink["primary"] if r["ids_match"] else ink["critical"]
            if not r["ids_match"]:
                label += " ⚠"
            b.append(f'<text x="{lx:.1f}" y="{y + ROW_H / 2 + 4}" fill="{fill}" font-size="11" '
                     f'font-weight="600" text-anchor="{anchor}" '
                     f'style="font-variant-numeric:tabular-nums">{label}</text>')
            y += ROW_H
        y += 4

    axis_bottom = y
    grid = []
    for t in ticks:
        strong = t == 1.0
        grid.append(f'<line x1="{x(t):.1f}" y1="{body_top}" x2="{x(t):.1f}" y2="{axis_bottom}" '
                    f'stroke="{ink["baseline"] if strong else ink["grid"]}" '
                    f'stroke-width="1"{"" if strong else ""}/>')
        grid.append(f'<text x="{x(t):.1f}" y="{axis_bottom + 15}" fill="{ink["muted"]}" font-size="10" '
                    f'text-anchor="middle" style="font-variant-numeric:tabular-nums">×{t:g}</text>')
    # frame (b[0]) first, gridlines behind the bars/labels (b[1:])
    return b[0] + "".join(grid) + "".join(b[1:])


def wrap_svg(inner, card_h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_W}" height="{card_h}" '
            f'viewBox="0 0 {CARD_W} {card_h}" font-family="{FONT}">{inner}</svg>')


def img_url(base, run_id, slug, mode):
    if base:
        return f"{base}/pipeline-{run_id}-{slug}-{mode}.png"
    return f"pipeline_bench_{slug}_{mode}.png"


def render_markdown(models, subtitle, meta, base, run_id):
    supported = [m for m in models if m["supported"]]
    ordered = supported + [m for m in models if not m["supported"]]
    mismatches = [f"{m['model']}/{r['fixture']}"
                  for m in supported for r in m["results"] if not r["ids_match"]]

    md = ["## PipelineTokenizer benchmark", ""]
    head = f"**{len(supported)} / {len(models)} models supported**"
    if supported:
        g = geomean([r["speedup"] for m in supported for r in m["results"]])
        nfix = len(supported[0]["results"])
        head += f" · geomean ×{g:.2f} across supported · {nfix} fixtures each"
    md += [f"{head} — {subtitle}", "", f"`{meta[0]}` · {meta[1]}", ""]

    unsupported = [m for m in models if not m["supported"]]
    if unsupported:
        items = ", ".join(f"`{m['model']}` ({m['shape']})" for m in unsupported)
        md += [f"> **Not yet supported:** {items} — shown as roadmap cards below.", ""]
    if mismatches:
        md += [f"> ⚠️ **Token ids diverge from the reference on: {', '.join(mismatches)}** "
               f"— speedups there are meaningless until fixed.", ""]

    # two-per-row grid of picture embeds
    md.append("<table>")
    for i in range(0, len(ordered), 2):
        md.append("<tr>")
        for m in ordered[i:i + 2]:
            slug = slugify(m["model"])
            light = img_url(base, run_id, slug, "light")
            dark = img_url(base, run_id, slug, "dark")
            md += ['<td align="center" valign="top">',
                   "<picture>",
                   f'  <source media="(prefers-color-scheme: dark)" srcset="{dark}">',
                   f'  <img alt="{escape(m["model"])} speedup" src="{light}" width="430">',
                   "</picture>",
                   "</td>"]
        md.append("</tr>")
    md += ["</table>", ""]

    if supported:
        md += ["<details><summary>Per-fixture results (supported models)</summary>", ""]
        for m in supported:
            md += [f"#### {m['model']} — {m['shape']}", "",
                   "| Fixture | Group | Tokenizer MB/s | Pipeline MB/s | Speedup | Ids |",
                   "|---|---|---:|---:|---:|:--|"]
            for r in sorted(m["results"], key=lambda r: (r["group"], -r["speedup"])):
                ids = "match" if r["ids_match"] else "⚠️ differ"
                md.append(f"| {r['fixture']} | {r['group']} | {r['legacy_mbps']:.1f} "
                          f"| {r['pipeline_mbps']:.1f} | ×{r['speedup']:.2f} | {ids} |")
            md.append("")
        md += ["</details>", ""]
    return "\n".join(md)


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
    tall_h = HEADER_H + body_height(layout_of(models)) + PAD
    mini_h = HEADER_H + 62

    out = Path(args.out_dir)
    for m in models:
        slug = slugify(m["model"])
        h = tall_h if m["supported"] else mini_h
        for mode in ("light", "dark"):
            svg = wrap_svg(card_svg(m, mode, lo, hi, h), h)
            (out / f"pipeline_bench_{slug}_{mode}.svg").write_text(svg)

    (out / "pipeline_bench.md").write_text(
        render_markdown(models, args.subtitle, meta, args.img_base, args.run_id))

    supported = [m for m in models if m["supported"]]
    g = geomean([r["speedup"] for m in supported for r in m["results"]]) if supported else 0
    print(f"{len(supported)}/{len(models)} supported, geomean x{g:.3f}, "
          f"cards {CARD_W}x{tall_h} / {CARD_W}x{mini_h}")


if __name__ == "__main__":
    main()
