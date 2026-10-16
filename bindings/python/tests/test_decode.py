import copy
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
    "ids, prefill, inputs, expected",
    [
        ([15496, 612], [], [15496, 612], ["Hello", " there"]),
        ([15496, 612], [], [[15496, 612]], ["Hello there"]),
        ([15496, 612], [15496], [[], 612], [None, " there"]),
        (
            [19526, 254, 25001, 121, 30325, 222],
            [],
            [19526, 254, 25001, 121, 30325, 222],
            [None, "你", None, "好", None, " 😀"],
        ),
        ([47249, 222], [47249], [[], 222], [None, "😀"]),
        ([41840, 233, 8582, 237, 121], [], [41840, 233, [8582, 237], 121], [None, "👋", None, "🏽"]),
    ],
)
def test_decode_stream_gpt2(gpt2, ids, prefill, inputs, expected):
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(ids=prefill)
    chunks = [stream.step(gpt2, item) for item in inputs]
    assert chunks == expected
    if not prefill or gpt2.decode(prefill).endswith("�"):
        assert "".join(chunk for chunk in chunks if chunk is not None) == gpt2.decode(ids)


@pytest.mark.parametrize("skip_special_tokens", [False, True])
def test_decode_stream_special_tokens(gpt2, skip_special_tokens):
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream(skip_special_tokens=skip_special_tokens)
    assert stream.step(gpt2, 50256) == (None if skip_special_tokens else "<|endoftext|>")
    assert stream.step(gpt2, 15496) == "Hello"
    assert stream.step(gpt2, 50256) == (None if skip_special_tokens else "<|endoftext|>")
    assert stream.step(gpt2, 612) == " there"


@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
def test_decode_stream_copy_preserves_pending_bytes(gpt2, clone):
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream()
    assert stream.step(gpt2, 47249) is None
    copied = clone(stream)
    assert stream.step(gpt2, 222) == "😀"
    assert stream.step(gpt2, 15496) == "Hello"
    assert copied.step(gpt2, 222) == "😀"
    assert copied.step(gpt2, 612) == " there"


@pytest.mark.parametrize("value", ["bad", -1, 2**32, [15496, "bad"]])
def test_decode_stream_rejects_invalid_input_without_changing_state(gpt2, value):
    from tokenizers.decoders import DecodeStream

    stream = DecodeStream([15496])
    with pytest.raises(TypeError):
        stream.step(gpt2, value)
    assert stream.step(gpt2, 612) == " there"


@pytest.mark.parametrize(
    "tokens, decoder, expected",
    [
        (
            ["hello", "##s", ".", "world"],
            {"type": "WordPiece", "prefix": "##", "cleanup": True},
            ["hello", "s", ".", " world"],
        ),
        (
            ["▁This", "▁is", "▁fine"],
            {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always"},
            ["This", " is", " fine"],
        ),
        (
            ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"],
            {"type": "Sequence", "decoders": [{"type": "ByteFallback"}, {"type": "Fuse"}]},
            [None, None, None, "😀"],
        ),
        (["hello", "world"], None, ["hello", " world"]),
    ],
)
def test_decode_stream_configured_decoder(tmp_path, tokens, decoder, expected):
    from tokenizers.decoders import DecodeStream

    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": decoder,
        "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, **{t: i for i, t in enumerate(tokens, 1)}},
            "unk_token": "<unk>",
        },
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    tokenizer = Tokenizer.from_file(str(path))
    ids = list(range(1, len(tokens) + 1))
    stream = DecodeStream()
    assert [stream.step(tokenizer, item) for item in ids] == expected
    assert "".join(chunk for chunk in expected if chunk is not None) == tokenizer.decode(ids)
