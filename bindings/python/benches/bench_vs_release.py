"""Benchmark these bindings against the released `tokenizers` wheel from PyPI.

Mirrors tk-encode/examples/fixture_bench.rs: every `.txt` corpus under
data/fixtures/{lang,modalities}, cut into ~10 KiB multi-line chunks (at most
100 per fixture), single-thread throughput per fixture plus one multi-thread
sweep over all fixtures flattened. Timing is end-to-end through Python —
input conversion, encode, and output objects all count, because that is what
a user pays. Ids are checked to match on every fixture; a mismatch fails the
run.

The local build and the released wheel share the package name `tokenizers`,
so they cannot be imported into one process. The released wheel lives in its
own directory (`pip install --target <dir> tokenizers`) and this script
re-runs itself in a subprocess with PYTHONPATH pointing there — PYTHONPATH
wins over site-packages, so the subprocess sees the release while the main
process sees the local build. Ids cross the process boundary as one SHA-1
digest per chunk.

Usage:
    python benches/bench_vs_release.py [--manifest bench_models.json]
        [--data-dir ../../tokenizers/data] [--iters 3]
        [--release-dir .release] [--json out.json] [--markdown out.md]

Set up the release directory once with:
    uv pip install --target .release tokenizers
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import tokenizers

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


def digests(batch) -> list[str]:
    return [hashlib.sha1(np.asarray(ids, dtype=np.uint32).tobytes()).hexdigest() for ids in batch]


def bench_release_side(models: list[dict], fixtures: list[dict], iters: int) -> dict:
    """Runs in the subprocess where `tokenizers` is the released wheel."""
    all_chunks = [c for f in fixtures for c in f["chunks"]]
    nbytes = sum(f["bytes"] for f in fixtures)
    out = {"version": tokenizers.__version__, "models": {}}
    for model in models:
        tok = tokenizers.Tokenizer.from_file(model["path"])
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        rows = []
        for fixture in fixtures:
            chunks = fixture["chunks"]
            encoded = tok.encode_batch_fast(chunks, add_special_tokens=False)
            t = timed(lambda: tok.encode_batch_fast(chunks, add_special_tokens=False), iters)
            rows.append(
                {
                    "mbps": fixture["bytes"] / t / 1e6,
                    "digests": digests([e.ids for e in encoded]),
                }
            )
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        t = timed(lambda: tok.encode_batch_fast(all_chunks, add_special_tokens=False), iters)
        out["models"][model["name"]] = {"fixtures": rows, "multi_thread_mbps": nbytes / t / 1e6}
    return out


def bench_local_side(tok, fixtures: list[dict], release_row: dict, iters: int) -> dict:
    row = {"fixtures": []}
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for fixture, rel in zip(fixtures, release_row["fixtures"], strict=True):
        chunks = fixture["chunks"]
        encoded = tok.encode_batch(chunks, add_special_tokens=False)
        t = timed(lambda: tok.encode_batch(chunks, add_special_tokens=False), iters)
        mbps = fixture["bytes"] / t / 1e6
        row["fixtures"].append(
            {
                "fixture": fixture["name"],
                "group": fixture["group"],
                "bytes": fixture["bytes"],
                "ids_match": digests(encoded) == rel["digests"],
                "pipeline_mbps": mbps,
                "release_mbps": rel["mbps"],
                "speedup": mbps / rel["mbps"],
            }
        )

    # Multi-thread: one sweep over all fixtures flattened, like fixture_bench.rs.
    all_chunks = [c for f in fixtures for c in f["chunks"]]
    nbytes = sum(f["bytes"] for f in fixtures)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    t = timed(lambda: tok.encode_batch(all_chunks, add_special_tokens=False), iters)
    mbps = nbytes / t / 1e6
    row["multi_thread"] = {
        "bytes": nbytes,
        "pipeline_mbps": mbps,
        "release_mbps": release_row["multi_thread_mbps"],
        "speedup": mbps / release_row["multi_thread_mbps"],
    }
    return row


def render_markdown(report: dict) -> str:
    lines = [
        "## Python bindings: this branch vs `tokenizers` "
        f"{report['release_version']} (PyPI)",
        "",
        f"{report['fixture_count']} fixtures (~10 KiB chunks, ≤100/fixture), median of "
        f"{report['iters']} runs, {report['cpus']} CPUs. Single-thread numbers aggregate "
        "all fixtures (speedup range = slowest…fastest fixture); multi-thread runs the "
        "flattened corpus. Speedup >1 means this branch is faster.",
        "",
        "| model | ids | branch 1t (MB/s) | release 1t (MB/s) | speedup 1t (range) "
        "| branch mt (MB/s) | release mt (MB/s) | speedup mt |",
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
    parser.add_argument("--release-dir", type=Path, default=Path(__file__).parents[1] / ".release")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--markdown", type=Path, help="write the summary table here")
    parser.add_argument("--side", choices=["release"], help=argparse.SUPPRESS)
    parser.add_argument("--models-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.side == "release":
        models = json.loads(args.models_json.read_text())
        fixtures = load_fixtures(args.data_dir)
        args.out.write_text(json.dumps(bench_release_side(models, fixtures, args.iters)))
        return 0

    if not (args.release_dir / "tokenizers").is_dir():
        sys.exit(
            f"released wheel not found in {args.release_dir} — run "
            f"`uv pip install --target {args.release_dir} tokenizers`"
        )

    models = json.load(open(args.manifest)) if args.manifest else DEFAULT_MODELS
    for model in models:
        model["path"] = str(args.data_dir / model.get("file", model["name"] + ".json"))
    models = [m for m in models if Path(m["path"]).is_file()] or sys.exit("no model files found")

    fixtures = load_fixtures(args.data_dir)

    compiled: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for m in models:
        try:
            tok = tokenizers.Tokenizer.from_file(m["path"])
            tok.encode("warmup", add_special_tokens=False)
            compiled[m["name"]] = tok
        except (tokenizers.TokenizersError, NotImplementedError) as e:
            skipped[m["name"]] = str(e)

    with tempfile.TemporaryDirectory() as td:
        models_json = Path(td) / "models.json"
        models_json.write_text(
            json.dumps([{"name": m["name"], "path": m["path"]} for m in models if m["name"] in compiled])
        )
        release_out = Path(td) / "release.json"
        subprocess.run(
            [
                sys.executable, __file__,
                "--side", "release",
                "--models-json", str(models_json),
                "--out", str(release_out),
                "--data-dir", str(args.data_dir),
                "--iters", str(args.iters),
            ],
            env=os.environ | {"PYTHONPATH": str(args.release_dir)},
            check=True,
        )
        release = json.loads(release_out.read_text())

    rows = []
    for m in models:
        if m["name"] in skipped:
            rows.append({"model": m["name"], "skipped": skipped[m["name"]]})
            continue
        row = bench_local_side(compiled[m["name"]], fixtures, release["models"][m["name"]], args.iters)
        rows.append({"model": m["name"], **row})

    report = {
        "release_version": release["version"],
        "pipeline_version": tokenizers.__version__,
        "iters": args.iters,
        "fixture_count": len(fixtures),
        "cpus": os.cpu_count(),
        "models": rows,
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
