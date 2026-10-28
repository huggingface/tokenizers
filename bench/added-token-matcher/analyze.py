#!/usr/bin/env python3
"""Fetch a job's log and write results/<job>/summary.md: max-of-rounds throughput per arm and cell, the
aho-corasick/buckets ratio, the identical-output gate, and memory. `analyze.py <job-id> [<job-id> ...]`."""
import collections
import pathlib
import statistics
import subprocess
import sys

here = pathlib.Path(__file__).parent
ARMS = ("buckets", "aho-corasick")


def load(job):
    out = here / "results" / job
    out.mkdir(parents=True, exist_ok=True)
    log = out / "log.txt"
    if not log.exists() or "== " not in log.read_text().splitlines()[-1]:
        log.write_text(subprocess.run(["hf", "jobs", "logs", job], check=True, capture_output=True, text=True).stdout)
    return out, log.read_text().splitlines()


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def summarize(job):
    out, lines = load(job)
    speed = collections.defaultdict(list)  # (kind, model, corpus, mode, arm) -> [MB/s]
    gate = collections.defaultdict(set)    # (kind, model, corpus, mode) -> {(arm, hash)}
    matches, build, loads = {}, collections.defaultdict(list), collections.defaultdict(list)
    for line in lines:
        f = line.split("\t")
        if f[0] == "MATCHER":
            _, _, arm, model, corpus, mode, mbs, count, h = f
            speed["matcher", model, corpus, mode, arm].append(float(mbs))
            gate["matcher", model, corpus, mode].add((arm, h))
            matches[model, corpus] = int(count)
        elif f[0] == "E2E":
            _, _, arm, model, corpus, mbs, h = f
            speed["e2e", model, corpus, "4k", arm].append(float(mbs))
            gate["e2e", model, corpus, "4k"].add((arm, h))
        elif f[0] == "BUILD":
            _, _, arm, model, tokens, us, live, peak = f
            build[model, arm].append((int(tokens), float(us), int(live), int(peak)))
        elif f[0] == "LOAD":
            _, _, arm, model, ms, live = f
            loads[model, arm].append((float(ms), int(live)))

    mismatched = [k for k, v in gate.items() if len({h for _, h in v}) != 1 or len({a for a, _ in v}) != 2]
    header = [l for l in lines if l.startswith("== machine") or l.startswith("== branch") or "steal" in l]
    md = [f"# Added-token matcher, job {job}", "", *[f"- {l[3:]}" for l in header], "",
          "Throughput is input MB/s, one thread, the best of the rounds (n = rounds per arm). Ratio = aho-corasick / buckets: above 1 means aho-corasick is faster.",
          "Spread = (max - min) / max over the rounds, worst arm.", "",
          f"Identical outputs between arms: {'all cells' if not mismatched else 'MISMATCH in ' + str(mismatched)}", ""]
    for kind, title in [("matcher", "Matcher alone: every match, restarting after each, like `extract_next`"),
                        ("e2e", "End to end: `encode` (no post-processor specials) on 4 KB pieces")]:
        rows = []
        for (k, model, corpus, mode) in sorted({key[:4] for key in speed if key[0] == kind}):
            b, a = (speed[kind, model, corpus, mode, arm] for arm in ARMS)
            spread = max((max(v) - min(v)) / max(v) for v in (a, b))
            row = [model, corpus]
            if kind == "matcher":
                row += [mode, matches.get((model, corpus), "")]
            rows.append(row + [f"{max(b):.0f}", f"{max(a):.0f}", f"{max(a) / max(b):.2f}", f"{spread:.0%}", min(len(a), len(b))])
        cols = ["model", "corpus"] + (["calls", "matches"] if kind == "matcher" else []) + ["buckets", "aho-corasick", "ratio", "spread", "n"]
        md += [f"## {title}", "", table(cols, rows), ""]
    rows = []
    for model in sorted({m for m, _ in build}):
        (tokens, _, bl, bp), (_, _, al, ap) = (build[model, arm][0] for arm in ARMS)
        bus, aus = (statistics.median(r[1] for r in build[model, arm]) for arm in ARMS)
        (bms, blive), (ams, alive) = (loads[model, arm][0] for arm in ARMS)
        rows.append([model, tokens, f"{bl / 1024:.0f}", f"{al / 1024:.0f}", f"{bp / 1024:.0f}", f"{ap / 1024:.0f}",
                     f"{bus:.0f}", f"{aus:.0f}", f"{blive / 2**20:.1f}", f"{alive / 2**20:.1f}"])
    md += ["## Memory and build time", "",
           "Matcher: bytes left allocated after building it from every added token (and the peak during the build), "
           "build time median of rounds. Tokenizer: bytes left allocated by `from_json`, both matchers and the model included.", "",
           table(["model", "added tokens", "matcher KiB buckets", "matcher KiB a-c", "peak KiB buckets", "peak KiB a-c",
                  "build µs buckets", "build µs a-c", "tokenizer MiB buckets", "tokenizer MiB a-c"], rows), ""]
    (out / "summary.md").write_text("\n".join(md))
    print(f"wrote {out / 'summary.md'}")


for job in sys.argv[1:]:
    summarize(job)
