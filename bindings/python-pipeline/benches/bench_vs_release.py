"""Benchmark tokenizers_pipeline against the released `tokenizers` wheel.

Mirrors tk-encode/examples/fixture_bench.rs: every `.txt` corpus under
data/fixtures/{lang,modalities}, cut into ~10 KiB multi-line chunks (at most
100 per fixture), single-thread throughput per fixture plus one multi-thread
sweep over all fixtures flattened. Timing is end-to-end through Python —
input conversion, encode, and output objects all count, because that is what
a user pays. Ids are checked to match on every fixture before anything is
timed; a mismatch fails the run.

Usage:
    python benches/bench_vs_release.py [--manifest bench_models.json]
        [--data-dir ../../tokenizers/data] [--iters 3]
        [--json out.json] [--markdown out.md]
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import tokenizers as release
import tokenizers_pipeline as pipeline

# Keep in sync with fixture_bench.rs (CHUNK_BYTES, MAX_CHUNKS).
CHUNK_BYTES = 10 * 1024
MAX_CHUNKS = 100

DEFAULT_MODELS = [
    {"name": "gpt2", "file": "gpt2.json"},
    {"name": "llama-3", "file": "llama-3-tokenizer.json"},
    {"name": "llama-2", "file": "llama-2.json"},
    {"name": "bert-base-uncased", "file": "bert-base-uncased.json"},
]


def make_chunks(text: str) -> list[str]:
    """~10 KiB multi-line chunks, same construction as fixture_bench.rs."""
    chunks: list[str] = []
    cur: list[str] = []
    cur_bytes = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        cur_bytes += len(line.encode()) + bool(cur)
        cur.append(line)
        if cur_bytes >= CHUNK_BYTES:
            chunks.append("\n".join(cur))
            if len(chunks) == MAX_CHUNKS:
                return chunks
            cur, cur_bytes = [], 0
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def load_fixtures(data_dir: Path) -> list[dict]:
    fixtures = []
    for group in ["lang", "modalities"]:
        directory = data_dir / "fixtures" / group
        if not directory.is_dir():
            sys.exit(f"{directory} not found — run `make fixtures` in tokenizers/ first")
        for path in sorted(directory.glob("*.txt")):
            chunks = make_chunks(path.read_text(encoding="utf-8"))
            fixtures.append(
                {
                    "group": group,
                    "name": path.stem,
                    "chunks": chunks,
                    "bytes": sum(len(c.encode()) for c in chunks),
                }
            )
    return fixtures


def timed(fn, iters: int) -> float:
    fn()  # warmup: compile the pipeline, fill caches, spin up thread pools
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def bench_model(model: dict, fixtures: list[dict], iters: int) -> dict:
    row = {"model": model["name"], "fixtures": []}
    try:
        ours = pipeline.Tokenizer.from_file(model["path"])
        ours.encode("warmup", add_special_tokens=False)
    except (pipeline.TokenizersError, NotImplementedError) as e:
        row["skipped"] = str(e)
        return row
    theirs = release.Tokenizer.from_file(str(model["path"]))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for fixture in fixtures:
        chunks = fixture["chunks"]
        parity = ours.encode_batch(chunks, add_special_tokens=False)
        reference = theirs.encode_batch_fast(chunks, add_special_tokens=False)
        t_ours = timed(lambda: ours.encode_batch(chunks, add_special_tokens=False), iters)
        t_theirs = timed(
            lambda: theirs.encode_batch_fast(chunks, add_special_tokens=False), iters
        )
        row["fixtures"].append(
            {
                "fixture": fixture["name"],
                "group": fixture["group"],
                "bytes": fixture["bytes"],
                "ids_match": all(
                    a.tolist() == b.ids for a, b in zip(parity, reference, strict=True)
                ),
                "pipeline_mbps": fixture["bytes"] / t_ours / 1e6,
                "release_mbps": fixture["bytes"] / t_theirs / 1e6,
                "speedup": t_theirs / t_ours,
            }
        )

    # Multi-thread: one sweep over all fixtures flattened, like fixture_bench.rs.
    all_chunks = [c for f in fixtures for c in f["chunks"]]
    nbytes = sum(f["bytes"] for f in fixtures)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    t_ours = timed(lambda: ours.encode_batch(all_chunks, add_special_tokens=False), iters)
    t_theirs = timed(
        lambda: theirs.encode_batch_fast(all_chunks, add_special_tokens=False), iters
    )
    row["multi_thread"] = {
        "bytes": nbytes,
        "pipeline_mbps": nbytes / t_ours / 1e6,
        "release_mbps": nbytes / t_theirs / 1e6,
        "speedup": t_theirs / t_ours,
    }
    return row


def render_markdown(report: dict) -> str:
    lines = [
        "## Python bindings: `tokenizers_pipeline` vs released `tokenizers` "
        f"{report['release_version']}",
        "",
        f"{report['fixture_count']} fixtures (~10 KiB chunks, ≤100/fixture), median of "
        f"{report['iters']} runs, {report['cpus']} CPUs. Single-thread numbers aggregate "
        "all fixtures (speedup range = slowest…fastest fixture); multi-thread runs the "
        "flattened corpus. Speedup >1 means the pipeline bindings are faster.",
        "",
        "| model | ids | pipeline 1t (MB/s) | release 1t (MB/s) | speedup 1t (range) "
        "| pipeline mt (MB/s) | release mt (MB/s) | speedup mt |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["models"]:
        if "skipped" in row:
            lines.append(f"| {row['model']} | — | skipped: {row['skipped']} | | | | | |")
            continue
        fixtures = row["fixtures"]
        nbytes = sum(f["bytes"] for f in fixtures)
        t_ours = sum(f["bytes"] / (f["pipeline_mbps"] * 1e6) for f in fixtures)
        t_theirs = sum(f["bytes"] / (f["release_mbps"] * 1e6) for f in fixtures)
        speedups = [f["speedup"] for f in fixtures]
        ids = "✓" if all(f["ids_match"] for f in fixtures) else "✗ MISMATCH"
        mt = row["multi_thread"]
        lines.append(
            f"| {row['model']} | {ids} "
            f"| {nbytes / t_ours / 1e6:.0f} | {nbytes / t_theirs / 1e6:.0f} "
            f"| {t_theirs / t_ours:.2f}× ({min(speedups):.2f}…{max(speedups):.2f}) "
            f"| {mt['pipeline_mbps']:.0f} | {mt['release_mbps']:.0f} | {mt['speedup']:.2f}× |"
        )
    lines += ["", "Per-fixture numbers are in the `python-bindings-bench` artifact."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="bench_models.json to take the model list from")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[3] / "tokenizers" / "data")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--markdown", type=Path, help="write the summary table here")
    args = parser.parse_args()

    models = json.load(open(args.manifest)) if args.manifest else DEFAULT_MODELS
    for model in models:
        model["path"] = args.data_dir / model.get("file", model["name"] + ".json")
    models = [m for m in models if m["path"].is_file()] or sys.exit("no model files found")

    fixtures = load_fixtures(args.data_dir)
    report = {
        "release_version": release.__version__,
        "pipeline_version": pipeline.__version__,
        "iters": args.iters,
        "fixture_count": len(fixtures),
        "cpus": os.cpu_count(),
        "models": [bench_model(m, fixtures, args.iters) for m in models],
    }

    markdown = render_markdown(report)
    print(markdown)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    if args.markdown:
        args.markdown.write_text(markdown)

    mismatches = [
        f"{row['model']}/{f['fixture']}"
        for row in report["models"]
        for f in row.get("fixtures", [])
        if not f["ids_match"]
    ]
    if mismatches:
        print(f"ids diverge on: {mismatches}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
