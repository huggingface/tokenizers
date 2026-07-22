"""End-to-end: build a tokenizer from scratch, train it, mutate its components
in place, encode, serialize, pickle, and hit the decode stub."""

import pickle
import tempfile
from pathlib import Path

import numpy as np

from tokenizers_pipeline import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parents[3] / "tokenizers" / "data"


def corpus():
    with open(DATA / "big.txt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20_000:
                break
            yield line


# 1. Build: model + components, assigned as plain values
tok = Tokenizer(models.BPE())
tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
tok.pre_tokenizer = pre_tokenizers.Whitespace()

# 2. Train from a Python iterator (GIL only taken to refill 256-line buffers)
trainer = trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=["<s>", "</s>", AddedToken("<pad>", special=True)],
    show_progress=False,
)
tok.train_from_iterator(corpus(), trainer=trainer)
print(f"trained: {tok!r}")
assert tok.get_vocab_size() == 1000, tok.get_vocab_size()
assert tok.token_to_id("<s>") == 0

# 3. Encode -> numpy uint32 array; special tokens are matched in the text
ids = tok.encode("The quick brown fox <s> jumps over the lazy dog")
print(f"ids: {ids.dtype} {ids}")
assert isinstance(ids, np.ndarray) and ids.dtype == np.uint32
assert tok.token_to_id("<s>") in ids
assert [tok.id_to_token(int(i)) for i in ids[:2]] is not None

# 4. Mutate a component in place: dropping the lowercasing normalizer changes ids
tok_ids_lower = tok.encode("HELLO WORLD")
tok.normalizer = normalizers.NFKC()
tok_ids_upper = tok.encode("HELLO WORLD")
assert not np.array_equal(tok_ids_lower, tok_ids_upper), "normalizer change must affect ids"
tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
assert np.array_equal(tok.encode("HELLO WORLD"), tok_ids_lower)
print(f"component swap: {tok.normalizer!r}")

# 5. Post-hoc vocabulary extension
added = tok.add_special_tokens(["<mask>"])
assert added == 1 and tok.token_to_id("<mask>") is not None
assert tok.token_to_id("<mask>") in tok.encode("a <mask> b")

# 6. Serialize / reload round-trip
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "tokenizer.json"
    tok.save(path)
    reloaded = Tokenizer.from_file(path)
text = "Round-trip: 42 tokens?"
assert np.array_equal(tok.encode(text), reloaded.encode(text))
print("save/load round-trip: identical ids")

# 7. Pickle round-trip (multiprocessing readiness)
unpickled = pickle.loads(pickle.dumps(tok))
assert np.array_equal(tok.encode(text), unpickled.encode(text))
print("pickle round-trip: identical ids")

# 8. decode is a stub for now
try:
    tok.decode(ids)
    raise AssertionError("decode should not be implemented yet")
except NotImplementedError as e:
    print(f"decode stub: NotImplementedError({e})")

print("OK")
