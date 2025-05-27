import json
import timeit
from tokenizers import Tokenizer, AddedToken
from tokenizers.models import WordLevel
from tokenizers.normalizers import (
    ByteLevel, Lowercase, NFC, NFD, NFKC, NFKD, Nmt, Strip, Replace, Prepend, BertNormalizer
)
import pytest

def build_tokenizer_json(size, normalizer=None, special_tokens=True):
    # Build vocab and WordLevel model
    vocab = {"a": 0}
    model = WordLevel(vocab=vocab, unk_token="[UNK]")
    # Add normalizer if specified
    tokenizer = Tokenizer(model)
    if normalizer:
        tokenizer.normalizer = normalizer
    tokens = [AddedToken(f"tok{i}", special=special_tokens) for i in range(size)]
    tokenizer.add_tokens(tokens)
    # Return serialized tokenizer JSON
    return tokenizer.to_str()


normalizer_factories = {
    "none": None,
    "byte_level": ByteLevel,
    "lowercase": Lowercase,
    "nfc": NFC,
    "nfd": NFD,
    "nfkc": NFKC,
    "nfkd": NFKD,
    "nmt": Nmt,
    "strip": lambda: Strip(strip_left=True, strip_right=True),
    "replace": lambda: Replace("a", "b"),
    "prepend": lambda: Prepend("pre_"),
    "bert": BertNormalizer
}



normalizer_factories = {
    "none": None,
    "byte_level": ByteLevel,
    "lowercase": Lowercase,
    "nfc": NFC,
    "nfd": NFD,
    "nfkc": NFKC,
    "nfkd": NFKD,
    "nmt": Nmt,
    "strip": lambda: Strip(True, True),
    "replace": lambda: Replace("a", "b"),
    "prepend": lambda: Prepend("pre_"),
    "bert": BertNormalizer,
}


@pytest.mark.parametrize("size", [10_000, 100_000])
@pytest.mark.parametrize("special_tokens", [True, False])
@pytest.mark.parametrize("norm_name,norm_factory", normalizer_factories.items())
def test_tokenizer_deserialization(benchmark, size, special_tokens, norm_name, norm_factory):
    """Benchmark Tokenizer.from_str deserialization with different vocab sizes and normalizers."""
    normalizer = norm_factory() if norm_factory else None
    tok_json = build_tokenizer_json(size, normalizer, special_tokens)

    def deserialize():
        tok = Tokenizer.from_str(tok_json)
        _ = tok

    benchmark.group = f"deserialize_{size}_{'special' if special_tokens else 'non_special'}"
    benchmark.name = f"norm_{norm_name}"
    benchmark(deserialize)

# some example usage
# pytest  benches/test_deserialize.py --benchmark-enable
# pytest test_deserialization_benchmark.py --benchmark-save=baseline
# pytest test_deserialization_benchmark.py --benchmark-compare=baseline
# pytest test_deserialization_benchmark.py --benchmark-compare=baseline --benchmark-save=baseline2