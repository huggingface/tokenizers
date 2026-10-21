import json

import pytest

from tokenizers import Padding, Tokenizer


def test_from_str_reads_what_to_str_writes(bert):
    bert.padding = Padding(length=8, pad_id=0)

    restored = Tokenizer.from_str(bert.to_str())

    assert restored.padding == bert.padding
    assert restored.encode("Hello there").ids == bert.encode("Hello there").ids


def test_from_str_refuses_invalid_json():
    with pytest.raises(ValueError, match="line 1 column 2"):
        Tokenizer.from_str("nope")


# The keys and order 0.23 writes.
def test_to_str_writes_the_keys_in_file_order(bert):
    assert list(json.loads(bert.to_str())) == [
        "version",
        "truncation",
        "padding",
        "added_tokens",
        "normalizer",
        "pre_tokenizer",
        "post_processor",
        "decoder",
        "model",
    ]


def test_to_str_indents_only_when_pretty(bert):
    compact = bert.to_str()

    assert "\n" not in compact
    assert bert.to_str(pretty=True) == json.dumps(json.loads(compact), indent=2, ensure_ascii=False)


def test_save_writes_pretty_json_by_default(bert, tmp_path):
    bert.save(tmp_path / "pretty.json")
    bert.save(str(tmp_path / "compact.json"), pretty=False)

    assert (tmp_path / "pretty.json").read_text() == bert.to_str(pretty=True)
    assert (tmp_path / "compact.json").read_text() == bert.to_str()


def test_save_into_a_missing_directory(bert, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        bert.save(tmp_path / "missing" / "tokenizer.json")
