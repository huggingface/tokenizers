import multiprocessing
import os
import pickle

import pytest

from conftest import GPT2
from tokenizers import Padding, Tokenizer, Truncation

TEXTS = ["Hello there", "General Kenobi", "You are a bold one"]


def test_pickle_tokenizer(bert):
    restored = pickle.loads(pickle.dumps(bert))

    assert [e.ids for e in restored.encode_batch(TEXTS)] == [e.ids for e in bert.encode_batch(TEXTS)]


def test_pickle_tokenizer_padding(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    tokenizer.padding = Padding(direction="left", length=8, pad_id=3)
    assert pickle.loads(pickle.dumps(tokenizer)).padding == Padding(direction="left", length=8, pad_id=3)

    tokenizer.padding = None
    assert pickle.loads(pickle.dumps(tokenizer)).padding is None


def test_pickle_tokenizer_truncation(truncated_wiki):
    tokenizer = Tokenizer.from_file(truncated_wiki)

    tokenizer.truncation = Truncation(2, direction="left")
    assert pickle.loads(pickle.dumps(tokenizer)).truncation == Truncation(2, direction="left")

    tokenizer.truncation = None
    assert pickle.loads(pickle.dumps(tokenizer)).truncation is None


def test_pickle_encoding(bert):
    encoding = bert.encode("Hello there", padding=Padding(length=8, pad_id=3))

    restored = pickle.loads(pickle.dumps(encoding))

    assert restored.ids == encoding.ids == [1, 27462, 7495, 2, 3, 3, 3, 3]
    assert restored.type_ids == encoding.type_ids
    assert restored.attention_mask == encoding.attention_mask == [1, 1, 1, 1, 0, 0, 0, 0]


def test_pickle_encoding_equals_original(bert):
    encoding = bert.encode("Hello there")

    assert pickle.loads(pickle.dumps(encoding)) == encoding


def test_pickle_padding():
    padding = Padding(direction="left", pad_id=50256, pad_token="<|endoftext|>", pad_to_multiple_of=8)

    assert pickle.loads(pickle.dumps(padding)) == padding


def test_pickle_truncation():
    truncation = Truncation(7, strategy="only_second", direction="left")

    assert pickle.loads(pickle.dumps(truncation)) == truncation


def _encode_batch(tokenizer, texts):
    return tokenizer.encode_batch(texts)


def test_multiprocessing(gpt2_spec):
    # spawn, not fork: the tokenizer travels to the worker as a pickle.
    gpt2_spec.padding = Padding(pad_id=50256)
    gpt2 = gpt2_spec.compile()

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        encoded = pool.apply(_encode_batch, (gpt2, TEXTS))

    assert [e.ids for e in encoded] == [e.ids for e in gpt2.encode_batch(TEXTS)]
    assert [e.attention_mask for e in encoded] == [e.attention_mask for e in gpt2.encode_batch(TEXTS)]


fork_only = pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
fork_warning = pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")


def _load_and_encode(path):
    return Tokenizer.from_file(path).compile().encode("Hello there").ids


# Regression test for issue 2559
@fork_only
@fork_warning
def test_fork_encode(gpt2):
    path = str(GPT2)
    with multiprocessing.get_context("fork").Pool(4) as pool:
        ids = pool.map_async(_load_and_encode, [path] * 4).get(timeout=60)
    assert ids == [gpt2.encode("Hello there").ids] * 4


# Regression test for issue 2559
@fork_only
@fork_warning
def test_fork_encode_pickle(gpt2):
    with multiprocessing.get_context("fork").Pool(4) as pool:
        encoded = pool.starmap_async(_encode_batch, [(gpt2, TEXTS)] * 4).get(timeout=60)
    assert [[e.ids for e in batch] for batch in encoded] == [[e.ids for e in gpt2.encode_batch(TEXTS)]] * 4
