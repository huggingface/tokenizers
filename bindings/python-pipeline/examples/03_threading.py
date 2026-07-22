"""Demonstrates that encode runs without the GIL: Python threads calling
encode() scale, and encode_batch parallelizes in Rust via rayon."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tokenizers_pipeline import Tokenizer

DATA = Path(__file__).resolve().parents[3] / "tokenizers" / "data"
N_THREADS = 4

tok = Tokenizer.from_file(DATA / "llama-3-tokenizer.json")
with open(DATA / "big.txt", encoding="utf-8") as f:
    text = f.read(2_000_000)
lines = [line for line in text.splitlines() if line.strip()]

tok.encode(text, add_special_tokens=False)  # warmup + compile


def encode_once():
    return tok.encode(text, add_special_tokens=False)


# 1. Python threads: with the GIL held during encode this could not scale
start = time.perf_counter()
for _ in range(N_THREADS):
    encode_once()
sequential = time.perf_counter() - start

with ThreadPoolExecutor(N_THREADS) as pool:  # warmup thread pool
    list(pool.map(lambda _: None, range(N_THREADS)))
    start = time.perf_counter()
    results = list(pool.map(lambda _: encode_once(), range(N_THREADS)))
    threaded = time.perf_counter() - start

speedup = sequential / threaded
print(f"{N_THREADS} encodes of {len(text) / 1e6:.1f}MB: "
      f"sequential {sequential:.2f}s, {N_THREADS} threads {threaded:.2f}s "
      f"({speedup:.1f}x)")
assert speedup > 1.5, f"threads did not scale ({speedup:.2f}x): is the GIL held?"

# 2. encode_batch: rayon parallelism inside one call, toggled by env var
os.environ["TOKENIZERS_PARALLELISM"] = "false"
start = time.perf_counter()
serial_ids = tok.encode_batch(lines, add_special_tokens=False)
serial = time.perf_counter() - start

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tok.encode_batch(lines[:100], add_special_tokens=False)  # spin up the pool
start = time.perf_counter()
parallel_ids = tok.encode_batch(lines, add_special_tokens=False)
parallel = time.perf_counter() - start

assert all(a.tolist() == b.tolist() for a, b in zip(serial_ids, parallel_ids, strict=True))
mbps = len(text) / parallel / 1e6
print(f"encode_batch {len(lines)} lines: serial {serial:.2f}s, "
      f"rayon {parallel:.2f}s ({serial / parallel:.1f}x, {mbps:.0f} MB/s)")
assert serial / parallel > 1.5, "rayon batch did not scale"

print("OK")
