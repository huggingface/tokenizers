#!/usr/bin/env python3
"""Render the Python-bindings bench JSON (bench_vs_release.py) as a chart +
markdown section for the PR description.

A sibling of render_pipeline_bench.py's overview chart, reusing its helpers so
the two read as one report: one row per model, log-scale ×speedup vs the
released `tokenizers` wheel (×1.0), two bars per row — single-thread (geomean
across fixtures, with a min–max whisker) and multi-thread (one sweep over the
flattened corpus). Skipped models render as muted rows. Emits
pipeline_bench_pybindings.svg (picked up by the workflow's existing
rasterize/upload globs) and python_bench_section.md (appended to
pipeline_bench.md before the PR upsert).
"""
import argparse
import json
import re
from html import escape
from pathlib import Path

import render_pipeline_bench as rpb

# Same-system series colors: pipeline blue for single-thread, the catalog teal
# for multi-thread (CVD ΔE 16.8 protan, both >= 3:1 on the chart surface).
PY_SINK = {"st": "#2a78d6", "mt": "#2a9d8f"}

SLUG = "pybindings"
ROW_H, BAR_H, GAP = 58, 12, 2


def st_speedups(model):
    return [f["speedup"] for f in model.get("fixtures", [])]


def agg_mbps(fixtures):
    """Byte-weighted single-thread throughput across all fixtures."""
    nbytes = sum(f["bytes"] for f in fixtures)
    t = sum(f["bytes"] / (f["pipeline_mbps"] * 1e6) for f in fixtures)
    return nbytes / t / 1e6


def skip_reason(message):
    """Short form of the compile error ("no Metaspace pre-tokenizer"); the raw
    message embeds Rust debug output too noisy for a chart row."""
    m = re.search(r"does not support PreTokenizer: (\w+)", message)
    if m:
        return f"no {m.group(1)} pre-tokenizer"
    return message if len(message) <= 60 else message[:57] + "…"


def chart_svg(report, meta):
    ink, sink = rpb.INK, PY_SINK
    models = report["models"]
    ref = f"tokenizers {report['release_version']} (PyPI)"

    vals = [v for m in models for v in st_speedups(m)]
    vals += [m["multi_thread"]["speedup"] for m in models if "multi_thread" in m]
    lo = min(0.75, min(vals) / 1.08) if vals else 0.75
    hi = max(1.5, max(vals) * 1.08) if vals else 1.5
    x = rpb.log_x(rpb.OV_GUTTER, rpb.OV_PLOT, lo, hi)
    ticks = rpb.thin_ticks([t for t in rpb.TICKS if lo <= t <= hi], x, min_px=34, keep=1.0)

    top = 74
    col_x = rpb.CHART_W - 16
    body = [f'<text x="{col_x}" y="{top - 14}" fill="{ink["muted"]}" font-size="11" '
            f'text-anchor="end">fixtures · ids</text>']
    y = top
    for m in models:
        cy = y + ROW_H / 2
        body.append(f'<text x="{rpb.OV_GUTTER - 14}" y="{cy + 4:.1f}" fill="{ink["primary"]}" '
                    f'font-size="12.5" font-weight="600" text-anchor="end">{escape(m["model"])}</text>')
        if "skipped" in m:
            body.append(f'<text x="{x(1.0) + 8:.1f}" y="{cy + 4:.1f}" fill="{ink["muted"]}" '
                        f'font-size="11.5" font-style="italic">not supported — '
                        f'{escape(skip_reason(m["skipped"]))}</text>')
            body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{ink["muted"]}" '
                        f'font-size="12" text-anchor="end">—</text>')
            y += ROW_H
            continue

        st = st_speedups(m)
        g, mn, mx = rpb.geomean(st), min(st), max(st)
        y_st = cy - BAR_H - GAP / 2
        y_mt = cy + GAP / 2
        body.append(rpb.hbar(x(1.0), x(g), y_st, BAR_H, sink["st"]))
        wy = y_st + BAR_H / 2
        body.append(f'<line x1="{x(mn):.1f}" y1="{wy:.1f}" x2="{x(mx):.1f}" y2="{wy:.1f}" '
                    f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
        for v in (mn, mx):
            body.append(f'<line x1="{x(v):.1f}" y1="{wy - 4:.1f}" x2="{x(v):.1f}" y2="{wy + 4:.1f}" '
                        f'stroke="{ink["secondary"]}" stroke-width="1.5"/>')
        body.append(f'<text x="{max(x(mx), x(g), x(1.0)) + 8:.1f}" y="{wy + 4:.1f}" '
                    f'fill="{ink["primary"]}" font-size="12" font-weight="600" '
                    f'style="font-variant-numeric:tabular-nums">×{g:.2f}'
                    f'<tspan fill="{ink["muted"]}" font-weight="400" font-size="11"> '
                    f'· {agg_mbps(m["fixtures"]):.0f} MB/s</tspan></text>')

        mt = m["multi_thread"]["speedup"]
        body.append(rpb.hbar(x(1.0), x(mt), y_mt, BAR_H, sink["mt"]))
        body.append(f'<text x="{max(x(mt), x(1.0)) + 8:.1f}" y="{y_mt + BAR_H / 2 + 4:.1f}" '
                    f'fill="{ink["primary"]}" font-size="12" font-weight="600" '
                    f'style="font-variant-numeric:tabular-nums">×{mt:.2f}'
                    f'<tspan fill="{ink["muted"]}" font-weight="400" font-size="11"> '
                    f'· {m["multi_thread"]["pipeline_mbps"]:.0f} MB/s</tspan></text>')

        bad = sum(1 for f in m["fixtures"] if not f["ids_match"])
        right, fill = ((f"⚠ {bad} differ", ink["critical"]) if bad
                       else (f'{len(m["fixtures"])} · ids ok', ink["secondary"]))
        body.append(f'<text x="{col_x}" y="{cy + 4:.1f}" fill="{fill}" font-size="12" '
                    f'text-anchor="end" style="font-variant-numeric:tabular-nums">{right}</text>')
        y += ROW_H

    axis = rpb.speedup_axis(ink, x, ticks, top, y + 4)
    y += 30
    legend = rpb.legend_row(ink, sink, y, [
        ("swatch", "st", "single thread (geomean, min–max whisker)"),
        ("swatch", "mt", "all threads"),
        ("tick", ink["baseline"], f"×1.0 = {ref}"),
    ])
    height = y + 34
    subtitle = (f"×speedup per model vs {ref} · timed end-to-end through Python · "
                "~10 KiB fixture chunks")
    return rpb.svg_doc(ink, height, "Python bindings vs released wheel — encode_batch",
                       subtitle, axis + "".join(body) + legend, meta)


def section_md(report, img_base, run_id):
    ref = f"`tokenizers` {report['release_version']} (PyPI)"
    lines = [
        f"### Python bindings — this branch vs {ref}",
        "",
        rpb.picture(img_base, run_id, SLUG, "Python bindings encode_batch speedup "
                    "vs the released tokenizers wheel", 860),
        "",
        "<details><summary>numbers (MB/s)</summary>", "",
        "| model | pipeline 1t | release 1t | speedup 1t | pipeline mt | release mt | speedup mt |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in report["models"]:
        if "skipped" in m:
            lines.append(f"| {m['model']} | not supported | | | | | |")
            continue
        fx = m["fixtures"]
        nbytes = sum(f["bytes"] for f in fx)
        t_ours = sum(f["bytes"] / (f["pipeline_mbps"] * 1e6) for f in fx)
        t_theirs = sum(f["bytes"] / (f["release_mbps"] * 1e6) for f in fx)
        mt = m["multi_thread"]
        lines.append(
            f"| {m['model']} | {nbytes / t_ours / 1e6:.0f} | {nbytes / t_theirs / 1e6:.0f} "
            f"| {t_theirs / t_ours:.2f}× "
            f"| {mt['pipeline_mbps']:.0f} | {mt['release_mbps']:.0f} | {mt['speedup']:.2f}× |"
        )
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bench_json", type=Path)
    parser.add_argument("--img-base", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    report = json.loads(args.bench_json.read_text())
    meta = [
        f"tokenizers {report['pipeline_version']} (this branch) vs "
        f"{report['release_version']} (PyPI)",
        f"{report['cpus']} CPUs · median of {report['iters']} runs · "
        f"{report['fixture_count']} fixtures",
    ]
    out = args.out_dir / f"pipeline_bench_{SLUG}.svg"
    out.write_text(chart_svg(report, meta))
    print(f"wrote {out}")
    out = args.out_dir / "python_bench_section.md"
    out.write_text(section_md(report, args.img_base, args.run_id))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
