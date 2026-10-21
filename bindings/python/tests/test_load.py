import json
import pickle

import pytest

from conftest import BERT, DATA
from tokenizers import Tokenizer


@pytest.mark.parametrize("path", [BERT, str(BERT)])
def test_from_file(path):
    assert Tokenizer.from_file(path).encode("Hello there").ids == [1, 27462, 7495, 2]


def test_missing_file():
    with pytest.raises(FileNotFoundError, match="does-not-exist.json"):
        Tokenizer.from_file(DATA / "does-not-exist.json")


# 0.23 refuses a file with keys it does not know.
def test_refuses_unknown_top_level_keys(tmp_path):
    file = json.loads(BERT.read_text())
    file["foo"] = 1
    file["bar"] = 2
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(file))
    with pytest.raises(ValueError, match="`foo`, `bar`"):
        Tokenizer.from_file(path)


def write(tmp_path, file):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(file))
    return path


# "2.0" is the canonical format the Rust writer emits; the file keeps the version it came with.
def test_keeps_the_file_version(tmp_path):
    file = {**json.loads(BERT.read_text()), "version": "2.0"}

    tokenizer = Tokenizer.from_file(write(tmp_path, file))

    assert 'version="2.0"' in repr(tokenizer)


# 0.23 refuses a version it does not know.
def test_refuses_unknown_version(tmp_path):
    file = {**json.loads(BERT.read_text()), "version": "3.0"}

    with pytest.raises(ValueError, match="3.0"):
        Tokenizer.from_file(write(tmp_path, file))


# The Rust writer adds `role_to_token` to every "2.0" file.
def test_keeps_role_to_token_in_a_2_0_file(tmp_path):
    file = {**json.loads(BERT.read_text()), "version": "2.0"}
    file["role_to_token"] = {"sep_token": "[SEP]"}

    restored = pickle.loads(pickle.dumps(Tokenizer.from_file(write(tmp_path, file))))

    assert 'role_to_token={"sep_token": "[SEP]"}' in repr(restored)


# 0.23 refuses `role_to_token`, so a "1.0" file can't carry it.
def test_refuses_role_to_token_in_a_1_0_file(tmp_path):
    file = {**json.loads(BERT.read_text()), "role_to_token": None}

    with pytest.raises(ValueError, match="`role_to_token`"):
        Tokenizer.from_file(write(tmp_path, file))
