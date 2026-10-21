import json

import pytest

from conftest import BERT
from tokenizers import Padding, Tokenizer


def test_from_str_reads_what_to_str_writes(bert_spec):
    bert_spec.padding = Padding(length=8, pad_id=0)

    restored = Tokenizer.from_str(bert_spec.to_str())

    assert restored.padding == bert_spec.padding
    assert restored.compile().encode("Hello there").ids == bert_spec.compile().encode("Hello there").ids


def test_from_str_refuses_invalid_json():
    with pytest.raises(ValueError, match="line 1 column 2"):
        Tokenizer.from_str("nope")


# The keys and order of the 2.0 format.
def test_to_str_writes_the_keys_in_file_order(bert_spec):
    assert list(json.loads(bert_spec.to_str())) == [
        "version",
        "truncation",
        "padding",
        "role_to_token",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "model",
    ]


def test_to_str_indents_only_when_pretty(bert_spec):
    compact = bert_spec.to_str()

    assert "\n" not in compact
    assert bert_spec.to_str(pretty=True) == json.dumps(json.loads(compact), indent=2, ensure_ascii=False)


def test_save_writes_pretty_json_by_default(bert_spec, tmp_path):
    bert_spec.save(tmp_path / "pretty.json")
    bert_spec.save(str(tmp_path / "compact.json"), pretty=False)

    assert (tmp_path / "pretty.json").read_text() == bert_spec.to_str(pretty=True)
    assert (tmp_path / "compact.json").read_text() == bert_spec.to_str()


def test_save_into_a_missing_directory(bert_spec, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        bert_spec.save(tmp_path / "missing" / "tokenizer.json")
