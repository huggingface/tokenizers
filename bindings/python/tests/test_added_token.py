import copy
import json
import pickle

import pytest

from conftest import GPT2
from tokenizers import AddedToken, Tokenizer


def test_normalized_default():
    assert AddedToken("<x>").normalized is True
    assert AddedToken("<x>", special=True).normalized is False
    assert AddedToken("<x>", normalized=True, special=True).normalized is True


def test_state_rebuilds_the_token():
    # transformers' `save_pretrained` writes this dict to `tokenizer_config.json` and loads it back
    # with `AddedToken(**state)`. normalized=True differs from its default for a special token.
    token = AddedToken("<x>", normalized=True, special=True)
    assert AddedToken(**token.__getstate__()) == token


FLAGS = ["single_word", "lstrip", "rstrip", "normalized", "special"]


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_reaches_its_own_getter(flag):
    token = AddedToken("<x>", **{"normalized": False, flag: True})
    assert {f: getattr(token, f) for f in FLAGS} == {f: f == flag for f in FLAGS}


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle(protocol):
    # normalized=True differs from its default for a special token, so it must travel in the pickle.
    token = AddedToken("<x>", normalized=True, special=True)
    assert pickle.loads(pickle.dumps(token, protocol=protocol)) == token


@pytest.fixture
def spec():
    return Tokenizer.from_file(GPT2)


def test_add_tokens_ids(spec):
    hello = json.loads(GPT2.read_text())["model"]["vocab"]["hello"]
    endoftext = AddedToken("<|endoftext|>", normalized=True, special=True)

    # "hello" is in the model vocab. It keeps its id there and still counts as added.
    assert spec.add_tokens(["<new>", "hello"]) == 2
    assert spec.compile().added_tokens == {
        hello: AddedToken("hello"),
        50256: endoftext,
        50257: AddedToken("<new>"),
    }


def test_add_tokens_counts_changes(spec):
    assert spec.add_tokens(["<new>", "<new>", ""]) == 1
    assert spec.add_tokens(["<new>"]) == 0
    assert spec.add_tokens([AddedToken("<new>", lstrip=True)]) == 1
    assert spec.compile().added_tokens[50257] == AddedToken("<new>", lstrip=True)


def test_add_special_tokens(spec):
    assert spec.add_special_tokens(["<str>", AddedToken("<token>", normalized=True)]) == 2
    assert spec.compile().added_tokens == {
        50256: AddedToken("<|endoftext|>", normalized=True, special=True),
        50257: AddedToken("<str>", special=True),
        50258: AddedToken("<token>", normalized=True, special=True),
    }


def test_added_tokens_edits_leave_the_compiled_tokenizer_unchanged(spec):
    compiled = spec.compile()
    compiled.added_tokens.clear()
    assert 50256 in compiled.added_tokens


def test_add_tokens_leaves_copies_unchanged(spec):
    copied = copy.copy(spec)
    spec.add_tokens(["<new>"])
    assert 50257 not in copied.compile().added_tokens


def test_from_file_refuses_added_tokens_not_an_array(tmp_path):
    # `add_tokens` appends to the array, so the spec must not hold anything else.
    declared = json.loads(GPT2.read_text())
    declared["added_tokens"] = {}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(declared))
    with pytest.raises(ValueError, match="added_tokens"):
        Tokenizer.from_file(path)
