import json
import pickle

import huggingface_hub
import pytest

from conftest import WIKI
from tokenizers import Tokenizer

DECLARED = {"cls_token": "[CLS]", "sep_token": "[SEP]"}
REPLACEMENT = {"eos_token": "[SEP]", "image_token": "<image>"}


@pytest.fixture
def bert_config_with_roles(tmp_path):
    declared = json.loads(WIKI.read_text())
    declared["role_to_token"] = DECLARED
    path = tmp_path / "roled.json"
    path.write_text(json.dumps(declared))
    return path


def test_a_file_without_roles_has_an_empty_map(wiki):
    assert wiki.role_to_token == {}


def test_reads_json_config(bert_config_with_roles):
    assert Tokenizer.from_file(bert_config_with_roles).role_to_token == DECLARED


@pytest.mark.parametrize("override", [REPLACEMENT, {}])
def test_from_file(bert_config_with_roles, override):
    assert Tokenizer.from_file(bert_config_with_roles, role_to_token=override).role_to_token == override


def test_from_pretrained(monkeypatch, bert_config_with_roles):
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **_: str(bert_config_with_roles))

    tokenizer = Tokenizer.from_pretrained("some-org/some-model", role_to_token=REPLACEMENT)

    assert tokenizer.role_to_token == REPLACEMENT


def test_with_role_to_token(bert_config_with_roles):
    tokenizer = Tokenizer.from_file(bert_config_with_roles)

    rebuilt = tokenizer.with_role_to_token(REPLACEMENT)

    assert rebuilt is not tokenizer
    assert rebuilt.role_to_token == REPLACEMENT
    assert tokenizer.role_to_token == DECLARED
    assert rebuilt.encode("Hello there").ids == tokenizer.encode("Hello there").ids


def test_pickle(wiki):
    rebuilt = wiki.with_role_to_token(REPLACEMENT)

    assert pickle.loads(pickle.dumps(rebuilt)).role_to_token == REPLACEMENT
