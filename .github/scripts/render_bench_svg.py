#!/usr/bin/env python3
"""
Render benchmark comparison as an SVG image with red/green color coding.

Usage:
    python render_bench_svg.py --baseline baseline.txt --current output.txt --output bench.svg
    python render_bench_svg.py --baseline-json python-baseline.json --current-json bench_output.json --output bench.svg

Supports both Rust (bencher text format) and Python (pytest-benchmark JSON).
"""

import argparse
import json
import re
import sys
from pathlib import Path


def parse_bencher(path: str) -> dict[str, float]:
    """Parse criterion bencher output: 'test name ... bench: N ns/iter (+/- M)'"""
    results = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"^test (.+?) \.\.\. bench:\s+([\d,]+) ns/iter", line)
        if m:
            name = m.group(1).strip()
            ns = int(m.group(2).replace(",", ""))
            results[name] = ns
    return results


def parse_pytest_json(path: str) -> dict[str, float]:
    """Parse pytest-benchmark JSON → {name: mean_ns}"""
    data = json.loads(Path(path).read_text())
    results = {}
    for b in data["benchmarks"]:
        results[b["name"]] = b["stats"]["mean"] * 1e9  # seconds → ns
    return results


def format_time(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns / 1e9:.2f}s"
    if ns >= 1e6:
        return f"{ns / 1e6:.1f}ms"
    if ns >= 1e3:
        return f"{ns / 1e3:.1f}us"
    return f"{ns:.0f}ns"


def render_svg(
    baseline: dict[str, float],
    current: dict[str, float],
    title: str = "Benchmark Comparison",
) -> str:
    ROW_H = 28
    HEADER_H = 50
    PAD = 16
    NAME_W = 320
    TIME_W = 100
    BAR_W = 200
    DELTA_W = 80
    TOTAL_W = NAME_W + TIME_W * 2 + BAR_W + DELTA_W + PAD * 2

    names = sorted(set(baseline) & set(current))
    if not names:
        return "<svg></svg>"

    total_h = HEADER_H + len(names) * ROW_H + PAD * 2 + 30  # +30 for title

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{TOTAL_W}" height="{total_h}" '
        f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="12">'
    )
    # Background
    lines.append(
        f'<rect width="{TOTAL_W}" height="{total_h}" fill="#0d1117" rx="8"/>'
    )

    # Title
    lines.append(
        f'<text x="{TOTAL_W // 2}" y="24" fill="#e6edf3" font-size="14" '
        f'font-weight="bold" text-anchor="middle">{title}</text>'
    )

    # Header
    y0 = 30 + PAD
    cols = [
        (PAD, "Benchmark", "start"),
        (PAD + NAME_W, "Baseline", "end"),
        (PAD + NAME_W + TIME_W, "Current", "end"),
        (PAD + NAME_W + TIME_W * 2 + BAR_W // 2, "", "middle"),  # bar area
        (PAD + NAME_W + TIME_W * 2 + BAR_W + DELTA_W // 2, "Δ", "middle"),
    ]
    for x, label, anchor in cols:
        if label:
            lines.append(
                f'<text x="{x}" y="{y0}" fill="#8b949e" font-size="11" '
                f'text-anchor="{anchor}">{label}</text>'
            )
    lines.append(
        f'<line x1="{PAD}" y1="{y0 + 6}" x2="{TOTAL_W - PAD}" y2="{y0 + 6}" '
        f'stroke="#30363d" stroke-width="1"/>'
    )

    max_ratio = max(
        abs(current[n] / baseline[n] - 1) for n in names if baseline[n] > 0
    )
    max_ratio = max(max_ratio, 0.01)  # avoid division by zero

    for i, name in enumerate(names):
        y = y0 + HEADER_H - 14 + i * ROW_H
        base_ns = baseline[name]
        cur_ns = current[name]

        if base_ns > 0:
            delta = (cur_ns - base_ns) / base_ns
        else:
            delta = 0.0

        # Alternating row background
        if i % 2 == 0:
            lines.append(
                f'<rect x="{PAD}" y="{y - 14}" width="{TOTAL_W - PAD * 2}" '
                f'height="{ROW_H}" fill="#161b22" rx="4"/>'
            )

        # Name (truncate if too long)
        display_name = name if len(name) <= 38 else name[:35] + "..."
        lines.append(
            f'<text x="{PAD + 4}" y="{y}" fill="#e6edf3">{display_name}</text>'
        )

        # Baseline time
        lines.append(
            f'<text x="{PAD + NAME_W}" y="{y}" fill="#8b949e" '
            f'text-anchor="end">{format_time(base_ns)}</text>'
        )

        # Current time
        color = "#3fb950" if delta <= -0.02 else "#f85149" if delta >= 0.02 else "#e6edf3"
        lines.append(
            f'<text x="{PAD + NAME_W + TIME_W}" y="{y}" fill="{color}" '
            f'text-anchor="end">{format_time(cur_ns)}</text>'
        )

        # Bar
        bar_x = PAD + NAME_W + TIME_W * 2 + BAR_W // 2
        bar_len = int(abs(delta) / max_ratio * (BAR_W // 2 - 4))
        bar_color = "#3fb950" if delta < 0 else "#f85149"

        if delta < 0:
            # Green bar goes left from center
            lines.append(
                f'<rect x="{bar_x - bar_len}" y="{y - 10}" width="{bar_len}" '
                f'height="14" fill="{bar_color}" rx="2" opacity="0.8"/>'
            )
        else:
            # Red bar goes right from center
            lines.append(
                f'<rect x="{bar_x}" y="{y - 10}" width="{bar_len}" '
                f'height="14" fill="{bar_color}" rx="2" opacity="0.8"/>'
            )

        # Center line
        lines.append(
            f'<line x1="{bar_x}" y1="{y - 12}" x2="{bar_x}" y2="{y + 4}" '
            f'stroke="#30363d" stroke-width="1"/>'
        )

        # Delta text
        sign = "+" if delta > 0 else ""
        delta_str = f"{sign}{delta * 100:.1f}%"
        lines.append(
            f'<text x="{PAD + NAME_W + TIME_W * 2 + BAR_W + DELTA_W // 2}" y="{y}" '
            f'fill="{color}" text-anchor="middle" font-weight="bold">{delta_str}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="Baseline bencher text file")
    parser.add_argument("--current", help="Current bencher text file")
    parser.add_argument("--baseline-json", help="Baseline pytest-benchmark JSON")
    parser.add_argument("--current-json", help="Current pytest-benchmark JSON")
    parser.add_argument("--output", required=True, help="Output path (.svg or .png)")
    parser.add_argument("--title", default="Benchmark Comparison", help="Chart title")
    args = parser.parse_args()

    if args.baseline and args.current:
        baseline = parse_bencher(args.baseline)
        current = parse_bencher(args.current)
    elif args.baseline_json and args.current_json:
        baseline = parse_pytest_json(args.baseline_json)
        current = parse_pytest_json(args.current_json)
    else:
        print("Provide either --baseline/--current or --baseline-json/--current-json")
        sys.exit(1)

    svg = render_svg(baseline, current, title=args.title)

    output = Path(args.output)
    if output.suffix == ".png":
        import cairosvg
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(output), scale=2)
    else:
        output.write_text(svg)

    print(f"Wrote {output} ({len(baseline)} baseline, {len(current)} current benchmarks)")


if __name__ == "__main__":
    main()
