import json

import pytest

from conftest import BERT, DATA
from tokenizers import Tokenizer


@pytest.mark.parametrize("path", [BERT, str(BERT)])
def test_from_file(path):
    assert Tokenizer.from_file(path).encode("Hello there").ids == [1, 27462, 7495, 2]


def test_missing_file():
    with pytest.raises(FileNotFoundError, match="does-not-exist.json"):
        Tokenizer.from_file(DATA / "does-not-exist.json")


@pytest.mark.parametrize(
    ("component", "value"),
    [
        ("normalizer", {"type": "Replace", "pattern": {"Regex": " +"}, "content": " "}),
        ("pre_tokenizer", {"type": "Split", "pattern": {"Regex": " +"}, "behavior": "Removed", "invert": False}),
    ],
)
def test_from_file_with_regex_component(tmp_path, component, value):
    config = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"a": 0, "b": 1, "<unk>": 2}, "unk_token": "<unk>"},
    }
    config[component] = value
    path = tmp_path / "regex-tokenizer.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    assert Tokenizer.from_file(str(path)).encode("a    b").ids == [0, 1]
