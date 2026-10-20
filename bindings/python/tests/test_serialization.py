import json

import pytest

from conftest import BERT, GPT2, LLAMA3, WIKI
from tokenizers import Padding, Tokenizer, Truncation


@pytest.fixture
def source(tmp_path):
    config = {
        "version": "1.0",
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "padding": None,
        "truncation": None,
        "model": {
            "type": "WordLevel",
            "vocab": {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3, "café": 4},
            "unk_token": "[UNK]",
        },
    }
    path = tmp_path / "source.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_from_str_loads_legacy_and_canonical_json(source):
    tokenizer = Tokenizer.from_str(source.read_text(encoding="utf-8"))
    assert tokenizer.encode("hello café").ids == [2, 4]
    restored = Tokenizer.from_str(tokenizer.to_str())
    assert restored.encode("hello café").ids == [2, 4]


@pytest.mark.parametrize("save_to_file", [False, True])
def test_round_trip_preserves_changed_and_disabled_config(source, tmp_path, save_to_file):
    tokenizer = Tokenizer.from_file(source)
    tokenizer.padding = Padding(length=4, pad_id=1, pad_token="[PAD]")
    tokenizer.truncation = Truncation(max_length=2)
    text = "hello café world"
    assert tokenizer.encode(text).ids == [2, 4, 1, 1]

    def round_trip(tokenizer):
        if save_to_file:
            path = tmp_path / "saved-é.json"
            tokenizer.save(path)
            return Tokenizer.from_file(path)
        return Tokenizer.from_str(tokenizer.to_str())

    restored = round_trip(tokenizer)
    assert restored.encode(text).ids == [2, 4, 1, 1]
    # The loaded pipeline now has configuration of its own. Disabling it must
    # also survive serialization, instead of reverting to the file's defaults.
    restored.padding = None
    restored.truncation = None
    assert round_trip(restored).encode(text).ids == [2, 4, 3]
    assert tokenizer.encode(text).ids == [2, 4, 1, 1]


def test_serialization_errors(source, tmp_path):
    with pytest.raises(ValueError):
        Tokenizer.from_str("{")
    with pytest.raises(FileNotFoundError):
        Tokenizer.from_file(source).save(tmp_path / "missing" / "tokenizer.json")


@pytest.mark.parametrize("path", [BERT, GPT2, LLAMA3, WIKI])
def test_real_tokenizer_round_trip(path):
    tokenizer = Tokenizer.from_file(path)
    restored = Tokenizer.from_str(tokenizer.to_str())
    for text in ["hello world", "naïve café 你好", "", "<|endoftext|>"]:
        expected = tokenizer.encode(text)
        actual = restored.encode(text)
        assert actual.ids == expected.ids
        assert actual.type_ids == expected.type_ids
        assert actual.attention_mask == expected.attention_mask
        assert restored.decode(actual.ids) == tokenizer.decode(expected.ids)
