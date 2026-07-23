"""End-to-end: build a tokenizer from scratch, train it, swap its components,
encode to an `Encoding` (and to a bare id array), serialize, pickle, and see
that decode raises for now."""

import pickle
import tempfile
from pathlib import Path

import numpy as np
from tokenizers import AddedToken, Tokenizer, models, normalizers, pre_tokenizers, trainers

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

# 2. Train from a Python iterator. Training runs in Rust threads; the
#    interpreter lock is only taken briefly to pull lines from the iterator.
trainer = trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=["<s>", "</s>", AddedToken("<pad>", special=True)],
    show_progress=False,
)
tok.train_from_iterator(corpus(), trainer=trainer)
print(f"trained: {tok!r}")
assert tok.get_vocab_size() == 1000, tok.get_vocab_size()
assert tok.token_to_id("<s>") == 0

# 3. Encode -> Encoding: ids plus the masks and metadata a model consumes.
enc = tok.encode("The quick brown fox <s> jumps over the lazy dog")
print(f"encoding: {enc!r}")
print(f"  tokens: {enc.tokens}")
print(f"  type_ids / attention_mask: {enc.type_ids} / {enc.attention_mask}")
assert enc.ids == tok.encode_ids("The quick brown fox <s> jumps over the lazy dog").tolist()
assert tok.token_to_id("<s>") in enc.ids
assert enc.type_ids == [0] * len(enc) and enc.attention_mask == [1] * len(enc)
# .ids is a list (models pad with `ids + [pad_id] * n`); .ids_array is numpy.
assert isinstance(enc.ids, list)
assert enc.ids_array().dtype == np.uint32

# Word ids and character offsets are not emitted by the encode pipeline yet;
# they raise rather than returning a plausible-looking guess.
for feature in (lambda: enc.word_ids, lambda: enc.offsets, lambda: enc.char_to_token(0)):
    try:
        feature()
        raise AssertionError("unavailable feature should raise")
    except NotImplementedError:
        pass

# encode_ids skips the Encoding and returns the ids array directly (no copy).
ids = tok.encode_ids("The quick brown fox <s> jumps over the lazy dog")
assert isinstance(ids, np.ndarray) and ids.dtype == np.uint32

# 4. Mutate a component in place: dropping the lowercasing normalizer changes ids
tok_ids_lower = tok.encode_ids("HELLO WORLD")
tok.normalizer = normalizers.NFKC()
tok_ids_upper = tok.encode_ids("HELLO WORLD")
assert not np.array_equal(tok_ids_lower, tok_ids_upper), "normalizer change must affect ids"
tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
assert np.array_equal(tok.encode_ids("HELLO WORLD"), tok_ids_lower)
print(f"component swap: {tok.normalizer!r}")

# 5. Post-hoc vocabulary extension
added = tok.add_special_tokens(["<mask>"])
assert added == 1 and tok.token_to_id("<mask>") is not None
assert tok.token_to_id("<mask>") in tok.encode_ids("a <mask> b")

# 6. Serialize / reload round-trip
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "tokenizer.json"
    tok.save(path)
    reloaded = Tokenizer.from_file(path)
text = "Round-trip: 42 tokens?"
assert np.array_equal(tok.encode_ids(text), reloaded.encode_ids(text))
print("save/load round-trip: identical ids")

# 7. Pickle round-trip (multiprocessing readiness)
unpickled = pickle.loads(pickle.dumps(tok))
assert np.array_equal(tok.encode_ids(text), unpickled.encode_ids(text))
print("pickle round-trip: identical ids")

# 8. decode is not implemented yet — it raises instead of guessing
try:
    tok.decode(ids)
    raise AssertionError("decode should not be implemented yet")
except NotImplementedError as e:
    print(f"decode stub: NotImplementedError({e})")

print("OK")
