"""Demonstrates that encode runs without the interpreter lock (the GIL):
Python threads calling encode() scale, and encode_batch spreads one batch
across Rust's thread pool (rayon).

Scaling is asserted unless TOKENIZERS_SCALING_ASSERTS=0 (CI sets it on shared
macOS runners, whose noisy hosts make parallel speedup unmeasurable there);
result-correctness asserts always apply.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parents[3] / "tokenizers" / "data"
N_THREADS = min(4, os.cpu_count() or 1)
STRICT = os.environ.get("TOKENIZERS_SCALING_ASSERTS", "1") != "0"


def check_scaling(cond, msg):
    if STRICT:
        assert cond, msg
    elif not cond:
        print(f"WARNING (not asserted here): {msg}")


def best_of(n, fn):
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


tok = Tokenizer.from_file(DATA / "llama-3-tokenizer.json")
with open(DATA / "big.txt", encoding="utf-8") as f:
    text = f.read(2_000_000)
lines = [line for line in text.splitlines() if line.strip()] * 4

tok.encode(text, add_special_tokens=False)  # warmup + compile


def encode_once():
    return tok.encode(text, add_special_tokens=False)


# 1. Python threads: with the GIL held during encode this could not scale
with ThreadPoolExecutor(N_THREADS) as pool:  # warmup thread pool
    list(pool.map(lambda _: None, range(N_THREADS)))
    sequential = best_of(3, lambda: [encode_once() for _ in range(N_THREADS)])
    threaded = best_of(3, lambda: list(pool.map(lambda _: encode_once(), range(N_THREADS))))

speedup = sequential / threaded
print(
    f"{N_THREADS} encodes of {len(text) / 1e6:.1f}MB: "
    f"sequential {sequential:.2f}s, {N_THREADS} threads {threaded:.2f}s "
    f"({speedup:.1f}x)"
)
check_scaling(speedup > 1.5, f"threads did not scale ({speedup:.2f}x): is the GIL held?")

# 2. encode_batch: rayon parallelism inside one call, toggled by env var
os.environ["TOKENIZERS_PARALLELISM"] = "false"
serial_ids = tok.encode_batch(lines, add_special_tokens=False)
serial = best_of(3, lambda: tok.encode_batch(lines, add_special_tokens=False))

os.environ["TOKENIZERS_PARALLELISM"] = "true"
parallel_ids = tok.encode_batch(lines, add_special_tokens=False)  # warmup: spins up the pool
parallel = best_of(3, lambda: tok.encode_batch(lines, add_special_tokens=False))

assert all(a.tolist() == b.tolist() for a, b in zip(serial_ids, parallel_ids, strict=True))
mbps = sum(len(line) for line in lines) / parallel / 1e6
print(
    f"encode_batch {len(lines)} lines: serial {serial:.2f}s, "
    f"rayon {parallel:.2f}s ({serial / parallel:.1f}x, {mbps:.0f} MB/s)"
)
check_scaling(serial / parallel > 1.5, f"rayon batch did not scale ({serial / parallel:.2f}x)")

print("OK")
