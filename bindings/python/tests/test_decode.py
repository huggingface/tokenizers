import json

import numpy as np
import pytest

from tokenizers import Tokenizer


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


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["Hello", "Ġthere"], "Hello there"),
        (["Hello", "Ġnever_registered_piece_2524"], "Hello never_registered_piece_2524"),
        (["Ã", "©"], "é"),
        (["Ã"], "�"),
        (["<|endoftext|>"], "<|endoftext|>"),
        (["[PA D]"], "[PA D]"),
        ([], ""),
    ],
)
def test_decode_token_strings_byte_level(gpt2, tokens, expected):
    assert gpt2.decode_token_strings(tokens) == expected


@pytest.mark.parametrize(
    ("decoder", "tokens", "expected"),
    [
        ({"type": "WordPiece", "prefix": "@@", "cleanup": True}, ["novel", "@@piece", "."], "novelpiece."),
        (
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
            ["▁hello", "▁world"],
            "hello world",
        ),
        (
            {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
            ["<0xC3>", "<0xA9>"],
            "é",
        ),
        (None, ["novel", "piece"], "novel piece"),
    ],
)
def test_decode_token_strings_uses_configured_decoder(tmp_path, decoder, tokens, expected):
    config = {
        "version": "2.0",
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": decoder,
        "model": {"type": "BPE", "byte_level": False, "vocab": {"a": 0}, "merges": []},
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    tokenizer = Tokenizer.from_file(path)

    # None of these pieces has an id in this vocabulary.
    assert tokenizer.decode_token_strings(tokens) == expected
    assert tokenizer.decode_token_strings([]) == ""
