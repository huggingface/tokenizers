import json
import pickle

import pytest

from conftest import BERT, DATA
from tokenizers import Tokenizer


@pytest.fixture
def load_bert_with(tmp_path):
    def load(**keys):
        path = tmp_path / "tokenizer.json"
        path.write_text(json.dumps({**json.loads(BERT.read_text()), **keys}))
        return Tokenizer.from_file(path)

    return load


@pytest.mark.parametrize("path", [BERT, str(BERT)])
def test_from_file(path):
    assert Tokenizer.from_file(path).compile().encode("Hello there").ids == [1, 27462, 7495, 2]


def test_missing_file():
    with pytest.raises(FileNotFoundError, match="does-not-exist.json"):
        Tokenizer.from_file(DATA / "does-not-exist.json")


@pytest.mark.parametrize(
    ("keys", "error"),
    [
        ({"foo": 1, "bar": 2}, "`bar`, `foo`"),
        # 0.23 refuses a version it does not know.
        ({"version": "3.0"}, "3.0"),
        # 0.23 refuses `role_to_token`, so a "1.0" file can't carry it.
        ({"role_to_token": None}, "`role_to_token`"),
    ],
)
def test_refuses(load_bert_with, keys, error):
    with pytest.raises(ValueError, match=error):
        load_bert_with(**keys)


# "2.0" is the canonical format the Rust writer emits, with `role_to_token` in every file.
def test_keeps_a_2_0_file_through_pickle(load_bert_with):
    tokenizer = load_bert_with(version="2.0", role_to_token={"sep_token": "[SEP]"})

    restored = pickle.loads(pickle.dumps(tokenizer))

    assert 'version="2.0"' in repr(restored)
    assert 'role_to_token={"sep_token": "[SEP]"}' in repr(restored)
