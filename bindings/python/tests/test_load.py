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


def test_wordpiece_with_empty_vocabulary_token(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": None,
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordPiece",
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100,
                    "vocab": {"[UNK]": 0, "a": 1, "##b": 2, "": 7},
                },
            }
        ),
        encoding="utf-8",
    )
    tokenizer = Tokenizer.from_file(path)
    assert tokenizer.encode("ab").ids == [1, 2]
    assert tokenizer.encode("ac").ids == [0]
    assert tokenizer.encode("").ids == []
