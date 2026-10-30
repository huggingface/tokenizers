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


@pytest.mark.parametrize("protocol", [0, pickle.HIGHEST_PROTOCOL])
def test_pickle(protocol):
    # normalized=True differs from its default for a special token, so it must travel in the pickle.
    token = AddedToken("<x>", normalized=True, special=True)
    assert pickle.loads(pickle.dumps(token, protocol=protocol)) == token


def compile_gpt2_with(added_tokens, tmp_path):
    """Compiles GPT-2 with `added_tokens`, a `{id: AddedToken}` dict, in place of its own."""
    declared = json.loads(GPT2.read_text())
    declared["added_tokens"] = [{"id": id, **token.__getstate__()} for id, token in added_tokens.items()]
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(declared))
    # TODO: to re-write when we land Tokenizer.from_str to avoid round-trip to the filesystem
    return Tokenizer.from_file(path).compile()


def test_compiled_flags(tmp_path):
    # Each token sets a different flag, so a flag read into the wrong field shows up.
    added_tokens = {
        50257: AddedToken("<single_word>", single_word=True, normalized=False),
        50258: AddedToken("<lstrip>", lstrip=True, normalized=False),
        50259: AddedToken("<rstrip>", rstrip=True, normalized=False),
        50260: AddedToken("<normalized>", normalized=True),
        50261: AddedToken("<special>", special=True),
    }

    assert compile_gpt2_with(added_tokens, tmp_path).added_tokens == added_tokens


@pytest.fixture
def spec():
    return Tokenizer.from_file(GPT2)


def test_add_tokens_ids(spec):
    hello = json.loads(GPT2.read_text())["model"]["vocab"]["hello"]
    endoftext = AddedToken("<|endoftext|>", normalized=True, special=True)

    # "hello" is in the model vocab: it keeps its id but is still added, as in 0.23.
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


def test_add_tokens_leaves_copies_unchanged(spec):
    before = spec.compile().added_tokens
    copied = copy.copy(spec)
    compiled = spec.compile()

    spec.add_tokens(["<new>"])

    assert copied.compile().added_tokens == before
    assert compiled.added_tokens == before
