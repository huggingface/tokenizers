"""The simplest starting point: load real tokenizer.json files and encode a
real corpus. Also demonstrates the two loud failure modes: pre-tokenizers the
pipeline does not support yet, and post-processing (special-token insertion),
which is not implemented yet. (Id parity against the released wheel is
checked by benches/bench_vs_release.py — the released package shares our
name, so the comparison needs two processes.)"""

from pathlib import Path

import numpy as np

from tokenizers import Tokenizer, TokenizersError

DATA = Path(__file__).resolve().parents[3] / "tokenizers" / "data"

with open(DATA / "big.txt", encoding="utf-8") as f:
    LINES = [line for line in f.read(500_000).splitlines() if line.strip()]
print(f"corpus: {len(LINES)} lines")

for name, file in [
    ("gpt2", "gpt2.json"),
    ("llama-3", "llama-3-tokenizer.json"),
    ("llama-2", "llama-2.json"),
    ("bert-base-uncased", "bert-base-uncased.json"),
]:
    tok = Tokenizer.from_file(DATA / file)
    batch = tok.encode_batch(LINES, add_special_tokens=False)
    assert len(batch) == len(LINES)
    total = sum(len(enc) for enc in batch)
    first = batch[0]
    assert first.ids_array().dtype == np.uint32
    assert first.attention_mask == [1] * len(first)
    print(f"{name}: {total} tokens, first token {first.tokens[0]!r}")

# Expected failure 1: post-processor would add special tokens -> loud error,
# not silently wrong ids
bert = Tokenizer.from_file(DATA / "bert-base-uncased.json")
try:
    bert.encode_ids("hello")
    raise AssertionError("should have raised")
except NotImplementedError as e:
    print(f"bert with add_special_tokens=True: NotImplementedError({e})")

# Expected failure 2: pipeline-unsupported component (Metaspace) -> loud error
# at compile time, with the reason
t5 = Tokenizer.from_file(DATA / "t5-base.json")
try:
    t5.encode_ids("hello", add_special_tokens=False)
    raise AssertionError("should have raised")
except TokenizersError as e:
    print(f"t5-base (Metaspace): TokenizersError({e})")

print("OK")
