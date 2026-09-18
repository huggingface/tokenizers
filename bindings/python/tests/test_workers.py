import multiprocessing
import pickle

from tokenizers import Padding, Tokenizer

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


def test_pickle_encoding(bert):
    bert.padding = Padding(length=8, pad_id=3)
    encoding = bert.encode("Hello there")

    restored = pickle.loads(pickle.dumps(encoding))

    assert restored.ids == encoding.ids == [1, 27462, 7495, 2, 3, 3, 3, 3]
    assert restored.type_ids == encoding.type_ids
    assert restored.attention_mask == encoding.attention_mask == [1, 1, 1, 1, 0, 0, 0, 0]


def test_pickle_padding():
    padding = Padding(direction="left", pad_id=50256, pad_token="<|endoftext|>", pad_to_multiple_of=8)

    assert pickle.loads(pickle.dumps(padding)) == padding


def _encode_batch(tokenizer, texts):
    return tokenizer.encode_batch(texts)


def test_multiprocessing(gpt2):
    # spawn, not fork: the tokenizer travels to the worker as a pickle.
    gpt2.padding = Padding(pad_id=50256)

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        encoded = pool.apply(_encode_batch, (gpt2, TEXTS))

    assert [e.ids for e in encoded] == [e.ids for e in gpt2.encode_batch(TEXTS)]
    assert [e.attention_mask for e in encoded] == [e.attention_mask for e in gpt2.encode_batch(TEXTS)]
