import numpy as np
import pytest


@pytest.mark.parametrize(
    "text",
    [
        "Hello there, how are you today?",
        " Hello there",
        "naïve café",
        "日本語",
        "👋🏽",
        "line one\nline two",
        "   ",
        "",
    ],
)
def test_gpt2_decode(gpt2, text):
    assert gpt2.decode(gpt2.encode(text).ids) == text


def test_gpt2_decode_bytes_preserves_partial_utf8(gpt2):
    # In GPT-2, these two token IDs hold the individual UTF-8 bytes of "é".
    assert gpt2.decode_bytes([127]) == b"\xc3"
    assert gpt2.decode_bytes([102]) == b"\xa9"
    assert gpt2.decode_bytes([127, 102]) == "é".encode("utf-8")


def test_gpt2_decode_bytes_skip_special_tokens(gpt2):
    end_id = gpt2.token_to_id("<|endoftext|>")
    assert end_id is not None
    ids = [127, end_id, 102]

    assert gpt2.decode_bytes(ids) == b"\xc3\xa9"
    assert gpt2.decode_bytes(ids, skip_special_tokens=False) == b"\xc3<|endoftext|>\xa9"


def test_bert_decode_bytes_requires_byte_level_bpe(bert):
    with pytest.raises(ValueError, match="byte-level BPE"):
        bert.decode_bytes(bert.encode("hello").ids)


def test_bert_decode(bert):
    assert bert.decode(bert.encode("Hello there").ids) == "hello there"
    assert bert.decode(bert.encode("Café").ids) == "cafe"


def test_decode_skip_special_tokens(llama3):
    ids = llama3.encode("Hello there").ids

    assert llama3.decode(ids) == "Hello there"
    assert llama3.decode(ids, skip_special_tokens=False) == "<|begin_of_text|>Hello there"


def test_decode_tokens(gpt2):
    assert gpt2.decode_tokens([15496, 612]) == ["Hello", "Ġthere"]
    assert gpt2.decode_tokens([]) == []


def test_decode_tokens_takes_arrays(gpt2):
    encoding = gpt2.encode("Hello there")
    tokens = gpt2.decode_tokens(encoding.ids)

    assert gpt2.decode_tokens(encoding.ids_array) == tokens
    assert gpt2.decode_tokens(np.array(encoding.ids, dtype=np.int32)) == tokens


def test_decode_tokens_skip_special_tokens(llama3):
    ids = llama3.encode("Hello there").ids
    tokens = llama3.decode_tokens(ids)

    assert tokens[0] == "<|begin_of_text|>"
    assert llama3.decode_tokens(ids, skip_special_tokens=True) == tokens[1:]


def test_decode_takes_an_encoding(llama3):
    encoding = llama3.encode("Hello there")

    assert llama3.decode(encoding) == "Hello there"
    assert llama3.decode(encoding, skip_special_tokens=False) == llama3.decode(encoding.ids, skip_special_tokens=False)


def test_decode_tokens_takes_an_encoding(llama3):
    encoding = llama3.encode("Hello there")

    assert llama3.decode_tokens(encoding) == llama3.decode_tokens(encoding.ids)
