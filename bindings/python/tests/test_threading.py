import concurrent.futures
import os

import numpy as np

from .conftest import SENTENCES, train_word_tokenizer


def test_concurrent_encode_matches_serial():
    tok = train_word_tokenizer()
    expected = [tok.encode(line, add_special_tokens=False) for line in SENTENCES]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda s: tok.encode(s, add_special_tokens=False), SENTENCES))
    for got, want in zip(results, expected):
        assert np.array_equal(got, want)


def test_parallel_encode_batch_matches_serial():
    tok = train_word_tokenizer()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    serial = tok.encode_batch(SENTENCES * 32, add_special_tokens=False)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    try:
        parallel = tok.encode_batch(SENTENCES * 32, add_special_tokens=False)
    finally:
        del os.environ["TOKENIZERS_PARALLELISM"]
    for got, want in zip(parallel, serial):
        assert np.array_equal(got, want)


def test_concurrent_mutation_and_encode_is_safe():
    tok = train_word_tokenizer()

    def encode_some(_):
        return tok.encode_batch(SENTENCES, add_special_tokens=False)

    def add_some(i):
        return tok.add_tokens([f"<extra_{i}>"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        encoded = pool.map(encode_some, range(8))
        added = pool.map(add_some, range(8))
        list(encoded)
        assert sum(added) == 8
