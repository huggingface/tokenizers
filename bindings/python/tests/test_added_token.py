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
