import json

import pytest

from conftest import BERT, DATA
from tokenizers import Tokenizer


@pytest.mark.parametrize("path", [BERT, str(BERT)])
def test_from_file(path):
    assert Tokenizer.from_file(path).compile().encode("Hello there").ids == [1, 27462, 7495, 2]


def test_missing_file():
    with pytest.raises(FileNotFoundError, match="does-not-exist.json"):
        Tokenizer.from_file(DATA / "does-not-exist.json")


def test_refuses_an_invalid_file(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({**json.loads(BERT.read_text()), "foo": 1}))

    with pytest.raises(ValueError, match="`foo`"):
        Tokenizer.from_file(path)
