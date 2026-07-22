"""Load real tokenizer.json files and check id parity against the released
`tokenizers` package on a real corpus. Also demonstrates the two loud failure
modes: unsupported pre-tokenizers and unwired post-processing."""

from pathlib import Path

import tokenizers as reference

from tokenizers_pipeline import Tokenizer, TokenizersError

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
    ref = reference.Tokenizer.from_file(str(DATA / file))

    ours = tok.encode_batch(LINES, add_special_tokens=False)
    theirs = ref.encode_batch_fast(LINES, add_special_tokens=False)
    mismatches = sum(
        1 for a, b in zip(ours, theirs, strict=True) if a.tolist() != b.ids
    )
    total = sum(len(a) for a in ours)
    assert mismatches == 0, f"{name}: {mismatches} mismatching lines"
    print(f"{name}: {total} tokens, ids identical to `tokenizers` {reference.__version__}")

# Expected failure 1: post-processor would add special tokens -> loud error,
# not silently wrong ids
bert = Tokenizer.from_file(DATA / "bert-base-uncased.json")
try:
    bert.encode("hello")
    raise AssertionError("should have raised")
except NotImplementedError as e:
    print(f"bert with add_special_tokens=True: NotImplementedError({e})")

# Expected failure 2: pipeline-unsupported component (Metaspace) -> loud error
# at compile time, with the reason
t5 = Tokenizer.from_file(DATA / "t5-base.json")
try:
    t5.encode("hello", add_special_tokens=False)
    raise AssertionError("should have raised")
except TokenizersError as e:
    print(f"t5-base (Metaspace): TokenizersError({e})")

print("OK")
