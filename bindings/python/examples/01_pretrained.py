"""The simplest starting point: load real tokenizer.json files and encode a
real corpus, with and without the special tokens the post-processor inserts.
Also shows the one remaining loud failure mode: a pre-tokenizer the pipeline
does not support yet (Metaspace). (Id parity against the released wheel is
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

# Post-processing runs: add_special_tokens=True wraps the input with the
# template's special tokens ([CLS]/[SEP] for BERT); False leaves them off.
bert = Tokenizer.from_file(DATA / "bert-base-uncased.json")
wrapped = bert.encode("hello world")
plain = bert.encode("hello world", add_special_tokens=False)
assert wrapped.tokens[0] == "[CLS]" and wrapped.tokens[-1] == "[SEP]"
assert len(wrapped) == len(plain) + 2
print(f"bert add_special_tokens: {wrapped.tokens}")

# Loud failure mode: a pipeline-unsupported component (Metaspace) -> error at
# compile time, with the reason
t5 = Tokenizer.from_file(DATA / "t5-base.json")
try:
    t5.encode_ids("hello", add_special_tokens=False)
    raise AssertionError("should have raised")
except TokenizersError as e:
    print(f"t5-base (Metaspace): TokenizersError({e})")

print("OK")
