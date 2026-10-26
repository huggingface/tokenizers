import multiprocessing
import pickle

from conftest import GPT2
from tokenizers import CompiledTokenizer, Padding, TokenizerSpec, Truncation

TEXTS = ["Hello there", "General Kenobi", "You are a bold one"]


def test_pickle_compiled_tokenizer(bert):
    restored = pickle.loads(pickle.dumps(bert))

    assert [e.ids for e in restored.encode_batch(TEXTS)] == [e.ids for e in bert.encode_batch(TEXTS)]


def test_pickle_spec_padding(padded_wiki):
    spec = TokenizerSpec.from_file(padded_wiki)

    spec.padding = Padding(direction="left", length=8, pad_id=3)
    assert pickle.loads(pickle.dumps(spec)).padding == Padding(direction="left", length=8, pad_id=3)

    spec.padding = None
    assert pickle.loads(pickle.dumps(spec)).padding is None


def test_pickle_spec_truncation(truncated_wiki):
    spec = TokenizerSpec.from_file(truncated_wiki)

    spec.truncation = Truncation(2, direction="left")
    assert pickle.loads(pickle.dumps(spec)).truncation == Truncation(2, direction="left")

    spec.truncation = None
    assert pickle.loads(pickle.dumps(spec)).truncation is None


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


compiled: CompiledTokenizer


def _compile(spec):
    global compiled
    compiled = spec.compile()


def _encode_batch(texts):
    return compiled.encode_batch(texts)


def test_multiprocessing():
    # spawn, not fork: the spec travels to the worker as a pickle, and the worker compiles it.
    spec = TokenizerSpec.from_file(GPT2)
    spec.padding = Padding(pad_id=50256)

    with multiprocessing.get_context("spawn").Pool(1, initializer=_compile, initargs=(spec,)) as pool:
        encoded = pool.apply(_encode_batch, (TEXTS,))

    expected = spec.compile().encode_batch(TEXTS)
    assert [e.ids for e in encoded] == [e.ids for e in expected]
    assert [e.attention_mask for e in encoded] == [e.attention_mask for e in expected]
