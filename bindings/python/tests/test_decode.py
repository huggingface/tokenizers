import numpy as np
import pytest

ROUND_TRIPS = [
    "Hello there, how are you today?",
    " Hello there",
    "naïve café",
    "日本語",
    "👋🏽",
    "line one\nline two",
    "   ",
    "",
]


@pytest.mark.parametrize("text", ROUND_TRIPS)
def test_gpt2_decode(gpt2, text):
    assert gpt2.decode(gpt2.encode(text).ids) == text


@pytest.mark.parametrize("text", ROUND_TRIPS)
def test_llama3_decode(llama3, text):
    assert llama3.decode(llama3.encode(text).ids) == text


def test_llama3_decode_special_tokens(llama3):
    ids = llama3.encode("Hello<|eot_id|>world").ids

    assert llama3.decode(ids) == "Helloworld"
    assert llama3.decode(ids, skip_special_tokens=False) == "<|begin_of_text|>Hello<|eot_id|>world"


def test_bert_decode(bert):
    assert bert.decode(bert.encode("Hello there").ids) == "hello there"
    assert bert.decode(bert.encode("Café").ids) == "cafe"


# Every token below is `Encoding.tokens` from released tokenizers 0.23.1 on the same fixture.
def test_gpt2_decode_tokens(gpt2):
    assert gpt2.decode_tokens([15496, 612]) == ["Hello", "Ġthere"]
    assert gpt2.decode_tokens(gpt2.encode("日本語").ids) == ["æĹ", "¥", "æľ", "¬", "èª", "ŀ"]
    assert gpt2.decode_tokens([]) == []


def test_decode_tokens_takes_arrays(gpt2):
    assert gpt2.decode_tokens(gpt2.encode("Hello there").ids_array) == ["Hello", "Ġthere"]
    assert gpt2.decode_tokens(np.array([15496, 612], dtype=np.int32)) == ["Hello", "Ġthere"]


def test_decode_tokens_drops_unknown_ids(gpt2):
    assert gpt2.decode_tokens([15496, 999_999]) == ["Hello"]


def test_llama3_decode_tokens_special_tokens(llama3):
    ids = [128000, 9906, 128009, 14957]

    assert llama3.decode_tokens(ids) == ["<|begin_of_text|>", "Hello", "<|eot_id|>", "world"]
    assert llama3.decode_tokens(ids, skip_special_tokens=True) == ["Hello", "world"]


def test_bert_decode_tokens(bert):
    ids = bert.encode("Hello there").ids

    assert bert.decode_tokens(ids) == ["[CLS]", "hello", "there", "[SEP]"]
    assert bert.decode_tokens(ids, skip_special_tokens=True) == ["hello", "there"]


def test_wiki_decode_tokens_skips_pad(wiki):
    assert wiki.decode_tokens([3, 3], skip_special_tokens=True) == []
    assert wiki.decode_tokens([3, 3]) == ["[PAD]", "[PAD]"]
