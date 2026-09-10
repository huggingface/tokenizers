r"""Compare Python list conversion with native Arrow batch tokenization.

Run with PyArrow and a release build of Tokenizers installed (for example,
``maturin develop --release`` from ``bindings/python``)::

    python bindings/python/benches/benchmark_arrow.py
    python bindings/python/benches/benchmark_arrow.py \
        --batch-sizes 32 --text-lengths 64 --repeats 1 --min-duration 0.01
    python bindings/python/benches/benchmark_arrow.py \
        --batch-sizes 1000 10000 --text-lengths 32 256 2048 \
        --arrow-types string large_string --threads 4 --json
    python bindings/python/benches/benchmark_arrow.py \
        --batch-sizes 1 16 256 4096 --text-lengths 32 256 2048 8192 \
        --arrow-types string --threads 1 --format markdown
    python bindings/python/benches/benchmark_arrow.py --format markdown \
        --metrics pylist_ms encode_ms total_ms peak_rss_mib

No model downloads are needed. Each method, Arrow type, batch size, text length,
and repetition runs in a fresh subprocess with one full-batch warmup. List and
Arrow trials are paired, alternating their order across repetitions and cases.
Both methods start with repeated Unicode text in Arrow and use the same WordLevel
tokenizer. Each worker repeats calls until their cumulative measured duration
reaches ``--min-duration`` (default 0.2 seconds), then reports per-call means.
Run on an otherwise idle machine and increase repetitions to assess timing noise.

``pylist_ms`` measures explicit ``to_pylist()`` conversion. It is zero for native
Arrow input; native capsule import and validation are included in ``encode_ms``.
``total_ms`` includes conversion and the encoding call, including construction of
the returned Encoding objects. Throughput is based on that same call-return time:
documents/second, UTF-8 input MiB/second, and output tokens/second. Fixture setup,
warmup, validation, and destruction of previous inputs and outputs are outside
the timers; the latter are discarded before the next call. The timing loop and
its bookkeeping are also excluded, but each call includes Python dispatch and
timer overhead. This measures repeated calls with warm allocator/model state,
not complete application throughput. The pipeline copies text into Rust-owned inputs; tokenization and outputs also allocate.

``peak_rss_mib`` is the absolute process peak resident memory through encoding,
including imports, fixture construction, and full-batch warmup. The final inputs
and outputs remain alive through this measurement; earlier batches do not
accumulate. It is not a measurement of only tokenizer allocations.
``setup_peak_rss_mib`` in JSON reports the high-water mark before measurement for
context. These are separate high-water marks, so
subtracting them does not measure the operation's allocated memory. Memory uses
``resource.getrusage`` and this benchmark supports Linux and macOS.

Printed results are medians across fresh-process repetitions. JSON also includes
individual samples, their iteration counts and measured duration, and interpreter,
platform, and package versions. Markdown renders List/Arrow matrices for all 12
measured fields by default; ``--metrics`` selects specific matrices. Performance,
memory context, and diagnostic fields are labeled separately. Document throughput
also includes median paired Arrow/list ratios (values above 1x favor Arrow).
Medians are computed independently per metric, so median conversion and encoding
times need not sum to median total time; the timing fields add within each sample.
Record the source commit and build configuration separately when comparing local
Tokenizers builds; their package version alone does not identify the code or
compiler options. These
synthetic measurements do not establish performance for other tokenizer models
or text distributions.
"""

import argparse
import gc
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

MARKDOWN_METRICS = {
    "pylist_ms": ("Python list conversion (ms/call)", 3),
    "encode_ms": ("Encoding call (ms/call)", 3),
    "total_ms": ("Conversion and encoding (ms/call)", 3),
    "documents_per_second": ("Document throughput (documents/s)", 0),
    "utf8_mib_per_second": ("UTF-8 input throughput (MiB/s)", 2),
    "output_tokens_per_second": ("Output token throughput (tokens/s)", 0),
    "peak_rss_mib": ("Process peak RSS through encoding (MiB)", 2),
    "setup_peak_rss_mib": ("Context: process peak RSS before timed calls (MiB)", 2),
    "iterations": ("Diagnostic: timed calls per worker (iterations)", None),
    "timed_duration_ms": ("Diagnostic: cumulative measured duration per worker (ms)", 1),
    "input_utf8_bytes": ("Diagnostic: UTF-8 input size per call (bytes)", None),
    "output_tokens": ("Diagnostic: output tokens per call (tokens)", None),
}


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_duration(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number of seconds")
    return value


def peak_rss_mib():
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def environment_info():
    # Inspect package metadata without importing their native runtimes in the
    # parent, keeping subprocess memory measurements independent of those imports.
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "tokenizers_version": version("tokenizers"),
        "tokenizers_module": find_spec("tokenizers").origin,
        "pyarrow_version": version("pyarrow"),
    }


def run_worker(case):
    import pyarrow as pa

    from tokenizers import Tokenizer

    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "unk_token": "[UNK]",
            "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "arrow": 3, "café": 4, "東京": 5},
        },
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        path.write_text(json.dumps(config))
        tokenizer = Tokenizer.from_file(path)
    convert_to_list = case["method"] == "list"
    encode = tokenizer.encode_batch if convert_to_list else tokenizer.encode_batch_arrow
    arrow_type = pa.string() if case["arrow_type"] == "string" else pa.large_string()

    phrase = "hello world arrow café 東京 "
    text = (phrase * ((case["text_length"] + len(phrase) - 1) // len(phrase)))[: case["text_length"]]
    # The temporary list repeats references to one string; it does not construct
    # a Python string for every row. Both methods use identical fixture setup.
    values = pa.array([text] * case["batch_size"], type=arrow_type)

    expected = tokenizer.encode(text)
    inputs = values.to_pylist() if convert_to_list else values
    encodings = encode(inputs)
    del encodings, inputs
    gc.collect()
    setup_peak = peak_rss_mib()

    iterations = 0
    conversion_seconds = 0.0
    total_seconds = 0.0
    while True:
        start = time.perf_counter()
        inputs = values.to_pylist() if convert_to_list else values
        # Sample the same three timestamps for both methods. For Arrow, all of
        # this call's time belongs to encode_ms; there is no to_pylist conversion.
        converted = time.perf_counter()
        encodings = encode(inputs)
        finished = time.perf_counter()

        iterations += 1
        total_seconds += finished - start
        if convert_to_list:
            conversion_seconds += converted - start
        if total_seconds >= case["min_duration_s"]:
            break
        # Do not charge the next call for previous result destruction or keep
        # two batches of outputs/converted strings alive during that call.
        del encodings, inputs
    peak = peak_rss_mib()

    # Keep inputs and outputs alive through the memory measurement. Validate
    # representative rows outside the timed section; all fixture rows are equal.
    assert len(encodings) == case["batch_size"]
    for encoding in (encodings[0], encodings[-1]):
        assert encoding == expected
    input_utf8_bytes = len(text.encode("utf-8")) * len(values)
    output_tokens = sum(map(len, encodings))
    return {
        "pylist_ms": conversion_seconds / iterations * 1000,
        "encode_ms": (total_seconds - conversion_seconds) / iterations * 1000,
        "total_ms": total_seconds / iterations * 1000,
        "documents_per_second": len(values) * iterations / total_seconds,
        "utf8_mib_per_second": input_utf8_bytes * iterations / total_seconds / (1024 * 1024),
        "output_tokens_per_second": output_tokens * iterations / total_seconds,
        "iterations": iterations,
        "timed_duration_ms": total_seconds * 1000,
        "peak_rss_mib": peak,
        "setup_peak_rss_mib": setup_peak,
        "input_utf8_bytes": input_utf8_bytes,
        "output_tokens": output_tokens,
    }


def format_metric(value, precision):
    if 0 < value < 1:
        return f"{value:.4g}"
    if precision is None:
        precision = 0 if value == int(value) else 1
    return f"{value:,.{precision}f}"


def print_markdown(results, args):
    metrics = getattr(args, "metrics", ["all"])
    metrics = list(MARKDOWN_METRICS) if "all" in metrics else list(dict.fromkeys(metrics))
    print(
        "Each cell shows List → Arrow medians across fresh-process repetitions. "
        "Document throughput also shows median paired Arrow/list speedup (above 1x favors Arrow)."
    )
    print(
        "Metric medians are independent: median conversion + median encoding need not equal median total. "
        "The timing fields add within each raw sample."
    )
    print(
        "Call-return throughput includes list conversion and encoding, excluding previous result destruction "
        "and loop bookkeeping. Arrow import is included in encoding; its explicit Python conversion is zero."
    )
    print(
        "Both RSS fields include imports and full-batch warmup; subtracting their high-water marks does not "
        "measure allocated memory. Iteration counts, timed duration, and per-call volumes are diagnostics."
    )
    print(
        f"Rayon threads: {args.threads}; fresh-process repetitions: {args.repeats}; "
        f"minimum timed duration per worker: {args.min_duration:g}s."
    )
    index = {
        (result["arrow_type"], result["batch_size"], result["text_length"], result["method"]): result
        for result in results
    }
    for metric in metrics:
        title, precision = MARKDOWN_METRICS[metric]
        print(f"\n{title} (`{metric}`)")
        for arrow_type in args.arrow_types:
            print(f"\n`{arrow_type}` arrays (column labels are Unicode characters per document):\n")
            print("| Batch size | " + " | ".join(map(str, args.text_lengths)) + " |")
            print("| ---: | " + " | ".join("---:" for _ in args.text_lengths) + " |")
            for batch_size in args.batch_sizes:
                cells = []
                for text_length in args.text_lengths:
                    arrow = index[arrow_type, batch_size, text_length, "arrow"]
                    baseline = index[arrow_type, batch_size, text_length, "list"]
                    cell = f"{format_metric(baseline[metric], precision)} → {format_metric(arrow[metric], precision)}"
                    if metric == "documents_per_second":
                        speedup = statistics.median(
                            a[metric] / b[metric] for a, b in zip(arrow["samples"], baseline["samples"])
                        )
                        cell += f" ({speedup:.2f}x)"
                    cells.append(cell)
                print(f"| {batch_size:,} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", nargs="+", type=positive_int, default=[1000, 10000])
    parser.add_argument(
        "--text-lengths", nargs="+", type=positive_int, default=[32, 256, 2048], help="Unicode characters"
    )
    parser.add_argument(
        "--arrow-types", nargs="+", choices=["string", "large_string"], default=["string", "large_string"]
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument(
        "--min-duration",
        type=nonnegative_duration,
        default=0.2,
        help="minimum cumulative timed seconds per worker; zero makes one measured call (default: 0.2)",
    )
    parser.add_argument("--threads", type=positive_int, default=1, help="Rayon threads per fresh subprocess")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--format", dest="output_format", choices=["text", "json", "markdown"], default="text")
    output.add_argument(
        "--json", dest="output_format", action="store_const", const="json", help="alias for --format json"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["all", *MARKDOWN_METRICS],
        default=["all"],
        help="select Markdown matrices (default: all); measurements and JSON are unaffected",
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        parser.error("peak RSS measurement currently supports Linux and macOS")
    if args.worker:
        print(json.dumps(run_worker(json.loads(args.worker))))
        return

    results = []
    environment = environment_info()
    env = dict(os.environ, RAYON_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="true")
    if args.output_format == "text":
        print(
            f"Python {platform.python_version()}; Tokenizers {environment['tokenizers_version']}; "
            f"PyArrow {environment['pyarrow_version']}; {platform.system()} {environment['machine']}; "
            f"threads={args.threads}; repeats={args.repeats}; min_duration={args.min_duration:g}s",
            flush=True,
        )
        print("Median milliseconds; peak RSS includes imports, input fixture, warmup, and outputs.", flush=True)
        print("Arrow capsule import is included in encode_ms; pylist_ms measures only to_pylist().", flush=True)
        print("Call-return throughput excludes previous result destruction and loop bookkeeping.", flush=True)
        print(
            f"{'type':12} {'rows':>8} {'chars':>6} {'method':>6} "
            f"{'pylist_ms':>10} {'encode_ms':>10} {'total_ms':>10} "
            f"{'docs/s':>12} {'UTF8_MiB/s':>11} {'tokens/s':>12} {'peak_MiB':>10}",
            flush=True,
        )
    cases = itertools.product(args.arrow_types, args.batch_sizes, args.text_lengths)
    case_count = len(args.arrow_types) * len(args.batch_sizes) * len(args.text_lengths)
    for case_index, (arrow_type, batch_size, text_length) in enumerate(cases):
        if args.output_format != "text":
            print(
                f"[{case_index + 1}/{case_count}] {arrow_type}: batch={batch_size}, chars={text_length}, "
                f"threads={args.threads}, repeats={args.repeats}",
                file=sys.stderr,
                flush=True,
            )
        case = {
            "arrow_type": arrow_type,
            "batch_size": batch_size,
            "text_length": text_length,
            "min_duration_s": args.min_duration,
        }
        samples = {"list": [], "arrow": []}
        for repetition in range(args.repeats):
            methods = ("list", "arrow") if (case_index + repetition) % 2 == 0 else ("arrow", "list")
            for method in methods:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        json.dumps({**case, "method": method}),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode:
                    parser.exit(1, f"Benchmark failed for {case}, method={method}:\n{completed.stderr}")
                samples[method].append(json.loads(completed.stdout))
        for method in ("list", "arrow"):
            result = {**case, "method": method, "threads": args.threads, "repeats": args.repeats}
            for key in samples[method][0]:
                result[key] = statistics.median(sample[key] for sample in samples[method])
            result["samples"] = samples[method]
            results.append(result)
            if args.output_format == "text":
                print(
                    f"{arrow_type:12} {batch_size:8d} {text_length:6d} {method:>6} "
                    f"{result['pylist_ms']:10.3f} {result['encode_ms']:10.3f} "
                    f"{result['total_ms']:10.3f} {result['documents_per_second']:12,.0f} "
                    f"{result['utf8_mib_per_second']:11.2f} {result['output_tokens_per_second']:12,.0f} "
                    f"{result['peak_rss_mib']:10.2f}",
                    flush=True,
                )
    if args.output_format == "json":
        print(json.dumps({"environment": environment, "results": results}, indent=2))
    elif args.output_format == "markdown":
        print_markdown(results, args)


if __name__ == "__main__":
    main()
