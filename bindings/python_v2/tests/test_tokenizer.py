import json
import math
from pathlib import Path

import pytest

from tokenizers import Encoding, Padding, Tokenizer, __version__

# `make test` fetches these; see the Makefile's TESTS_RESOURCES.
DATA = Path(__file__).parent.parent / "data"
# BPE with a ByteLevel pre-tokenizer, post-processor and decoder.
GPT2 = str(DATA / "gpt2.json")
# BPE with a Whitespace pre-tokenizer and nothing else.
WIKI = str(DATA / "tokenizer-wiki.json")
# WordPiece with a `[CLS] $A [SEP]` template.
BERT = str(DATA / "bert-wiki.json")

SHORT_AND_LONG = ["Hello", "Hello there, how are you today?"]


@pytest.fixture
def padded_wiki(tmp_path):
    declared = json.loads(Path(WIKI).read_text())
    declared["padding"] = {
        "strategy": "BatchLongest",
        "direction": "Right",
        "pad_to_multiple_of": None,
        "pad_id": 0,
        "pad_type_id": 0,
        "pad_token": "[PAD]",
    }
    path = tmp_path / "padded.json"
    path.write_text(json.dumps(declared))
    return str(path)


def test_module_reports_a_version():
    assert __version__


def test_from_file_reads_a_legacy_tokenizer_json():
    tokenizer = Tokenizer.from_file(WIKI)
    assert isinstance(tokenizer, Tokenizer)


def test_from_file_reports_a_missing_file():
    with pytest.raises(ValueError):
        Tokenizer.from_file(str(DATA / "does-not-exist.json"))


def test_encode_returns_an_encoding():
    encoding = Tokenizer.from_file(BERT).encode("Hello there")

    assert isinstance(encoding, Encoding)
    assert encoding.ids[0] == 1 and encoding.ids[-1] == 2
    assert len(encoding) == len(encoding.ids)
    assert encoding.type_ids == [0] * len(encoding)
    assert encoding.attention_mask == [1] * len(encoding)


def test_add_special_tokens_is_honoured():
    tokenizer = Tokenizer.from_file(BERT)

    with_specials = tokenizer.encode("Hello", add_special_tokens=True)
    without = tokenizer.encode("Hello", add_special_tokens=False)

    assert len(with_specials) == len(without) + 2


def test_encode_batch_keeps_input_order():
    tokenizer = Tokenizer.from_file(GPT2)
    texts = ["Hello there", "General Kenobi", "You are a bold one"]

    batch = tokenizer.encode_batch(texts)

    assert [e.ids for e in batch] == [tokenizer.encode(t).ids for t in texts]


def test_encode_batch_of_nothing_is_empty():
    assert Tokenizer.from_file(GPT2).encode_batch([]) == []


def test_decode_round_trips_through_gpt2():
    tokenizer = Tokenizer.from_file(GPT2)
    text = "Hello there, how are you?"

    assert tokenizer.decode(tokenizer.encode(text).ids) == text


def test_decode_skips_special_tokens_by_default():
    # This BERT lowercases, so the round trip lands on the normalized text.
    tokenizer = Tokenizer.from_file(BERT)
    ids = tokenizer.encode("Hello there").ids

    assert tokenizer.decode(ids) == "hello there"
    assert tokenizer.decode(ids, skip_special_tokens=False) == "[CLS] hello there [SEP]"


def test_padding_defaults_match_the_released_enable_padding():
    padding = Padding()

    assert padding.direction == "right"
    assert padding.pad_id == 0
    assert padding.pad_type_id == 0
    assert padding.pad_token == "[PAD]"
    assert padding.length is None
    assert padding.pad_to_multiple_of is None


def test_padding_refuses_an_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        Padding(direction="up")


def test_no_padding_unless_asked():
    tokenizer = Tokenizer.from_file(GPT2)
    assert tokenizer.padding is None

    short, long = tokenizer.encode_batch(SHORT_AND_LONG)

    assert len(short) < len(long)


def test_padding_to_the_longest_in_the_batch():
    padding = Padding(direction="left", pad_id=50256, pad_token="<|endoftext|>")
    tokenizer = Tokenizer.from_file(GPT2, padding=padding)

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.direction == "left"
    assert in_force.pad_id == 50256
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)

    assert len(short) == len(long)
    pad = len(long) - 1
    assert short.ids[:pad] == [50256] * pad
    assert short.attention_mask == [0] * pad + [1]
    assert long.attention_mask == [1] * len(long)


def test_padding_to_a_fixed_length():
    tokenizer = Tokenizer.from_file(GPT2, padding=Padding(length=8, pad_id=50256))

    encoding = tokenizer.encode("Hello")

    assert len(encoding) == 8
    assert encoding.ids[1:] == [50256] * 7
    assert encoding.attention_mask == [1] + [0] * 7


def test_padding_to_a_multiple():
    unpadded = Tokenizer.from_file(GPT2)
    padded = Tokenizer.from_file(GPT2, padding=Padding(pad_to_multiple_of=8))

    assert len(padded.encode("Hello")) == 8
    for text in ["Hello there, how are you today?", "word " * 20]:
        assert len(padded.encode(text)) == math.ceil(len(unpadded.encode(text)) / 8) * 8


def test_padding_declared_in_the_file_is_applied(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.direction == "right"
    assert in_force.length is None
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)
    assert short.ids[-1] == 0 and short.attention_mask[-1] == 0


def test_padding_passed_to_from_file_replaces_the_files(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki, padding=Padding(direction="left"))

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.direction == "left"
    short, _ = tokenizer.encode_batch(SHORT_AND_LONG)
    assert short.ids[0] == 0 and short.attention_mask[0] == 0


def test_padding_can_be_set_after_construction():
    tokenizer = Tokenizer.from_file(GPT2)

    tokenizer.padding = Padding(pad_id=50256)

    in_force = tokenizer.padding
    assert in_force is not None
    assert in_force.pad_id == 50256
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)
    assert short.ids[-1] == 50256


def test_setting_padding_to_none_switches_it_off(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    tokenizer.padding = None

    assert tokenizer.padding is None
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) < len(long)


def test_padding_and_encoding_are_read_only():
    padding = Padding()
    encoding = Tokenizer.from_file(GPT2).encode("Hello")

    with pytest.raises(AttributeError):
        padding.pad_id = 1  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        encoding.ids = []  # ty: ignore[invalid-assignment]
