import json

import pytest

from tokenizers import Tokenizer


def load_tokenizer(tmp_path, model, added_tokens=()):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"version": "2.0", "model": model, "added_tokens": list(added_tokens)}))
    return Tokenizer.from_file(path)


def added_token(content, token_id, special=False):
    return {
        "id": token_id,
        "content": content,
        "special": special,
        "normalized": False,
        "single_word": False,
        "lstrip": False,
        "rstrip": False,
    }


@pytest.mark.parametrize(
    "model",
    [
        {"type": "BPE", "byte_level": False, "vocab": {"<unk>": 0, "a": 1}, "merges": []},
        {
            "type": "WordPiece",
            "vocab": {"<unk>": 0, "a": 1},
            "unk_token": "<unk>",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
        },
        {"type": "WordLevel", "vocab": {"<unk>": 0, "a": 1}, "unk_token": "<unk>"},
        {"type": "Unigram", "vocab": [["<unk>", 0.0], ["a", -1.0]], "unk_id": 0},
    ],
    ids=["bpe", "wordpiece", "wordlevel", "unigram"],
)
def test_vocab_lookups_include_added_tokens_without_double_counting(tmp_path, model):
    tokenizer = load_tokenizer(tmp_path, model, [added_token("a", 1), added_token("<pad>", 2, special=True)])
    assert tokenizer.get_vocab(False) == {"<unk>": 0, "a": 1}
    assert tokenizer.get_vocab() == {"<unk>": 0, "a": 1, "<pad>": 2}
    assert tokenizer.get_vocab_size(False) == 2
    assert tokenizer.get_vocab_size() == 3
    for token, token_id in tokenizer.get_vocab().items():
        assert tokenizer.token_to_id(token) == token_id
        assert tokenizer.id_to_token(token_id) == token
    assert tokenizer.token_to_id("missing") is None
    assert tokenizer.token_to_id("") is None
    assert tokenizer.id_to_token(99) is None
    assert tokenizer.id_to_token(2**32 - 1) is None
    for invalid_id in [-1, 2**32]:
        with pytest.raises(OverflowError):
            tokenizer.id_to_token(invalid_id)
    assert tokenizer.get_vocab(with_added_tokens=False) == tokenizer.get_vocab(False)
    assert tokenizer.get_vocab_size(with_added_tokens=False) == tokenizer.get_vocab_size(False)
    vocab = tokenizer.get_vocab()
    vocab["a"] = 99
    assert tokenizer.token_to_id("a") == 1


def test_wordlevel_looks_up_an_empty_token_in_its_vocabulary(tmp_path):
    tokenizer = load_tokenizer(
        tmp_path,
        {"type": "WordLevel", "vocab": {"<unk>": 0, "": 1}, "unk_token": "<unk>"},
        [added_token("<pad>", 2, special=True)],
    )
    assert tokenizer.token_to_id("") == 1
    assert tokenizer.id_to_token(1) == ""
    assert tokenizer.get_vocab()[""] == 1


def test_wordlevel_looks_up_tokens_that_share_an_id(tmp_path):
    vocab = {"<unk>": 0, "a": 1, "b": 1}
    tokenizer = load_tokenizer(tmp_path, {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"})
    assert tokenizer.get_vocab() == vocab
    assert tokenizer.get_vocab_size(False) == tokenizer.get_vocab_size() == 3
    assert tokenizer.token_to_id("a") == tokenizer.token_to_id("b") == 1


def test_duplicate_unigram_pieces_keep_model_size_and_last_lookup_id(tmp_path):
    tokenizer = load_tokenizer(
        tmp_path,
        {"type": "Unigram", "vocab": [["<unk>", 0.0], ["a", -1.0], ["a", -2.0]], "unk_id": 0},
    )
    assert tokenizer.get_vocab(False) == {"<unk>": 0, "a": 2}
    assert tokenizer.get_vocab() == tokenizer.get_vocab(False)
    assert tokenizer.get_vocab_size(False) == 3
    assert tokenizer.get_vocab_size() == 2
    assert tokenizer.token_to_id("a") == 2
    assert tokenizer.id_to_token(1) == tokenizer.id_to_token(2) == "a"


@pytest.mark.parametrize("model_type", ["BPE", "WordPiece", "WordLevel"])
def test_vocab_lookups_preserve_sparse_ids(tmp_path, model_type):
    model = {"type": model_type, "vocab": {"<unk>": 0, "a": 7}, "unk_token": "<unk>"}
    if model_type == "BPE":
        model.update(byte_level=False, merges=[])
    elif model_type == "WordPiece":
        model.update(continuing_subword_prefix="##", max_input_chars_per_word=100)
    tokenizer = load_tokenizer(tmp_path, model)
    assert tokenizer.get_vocab() == {"<unk>": 0, "a": 7}
    assert tokenizer.get_vocab_size(False) == tokenizer.get_vocab_size() == 2
    assert tokenizer.token_to_id("a") == 7
    assert tokenizer.id_to_token(7) == "a"
    assert tokenizer.id_to_token(1) is None


def test_byte_level_lookups_use_vocabulary_spelling(tmp_path):
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {byte: chr(byte) for byte in printable}
    alphabet.update({byte: chr(256 + index) for index, byte in enumerate(b for b in range(256) if b not in printable)})
    vocab = {alphabet[byte]: byte for byte in range(256)}
    vocab["Ġa"] = 256
    tokenizer = load_tokenizer(
        tmp_path,
        {"type": "BPE", "byte_level": True, "vocab": vocab, "merges": [["Ġ", "a"]]},
    )
    assert tokenizer.get_vocab() == vocab
    assert tokenizer.get_vocab_size(False) == tokenizer.get_vocab_size() == 257
    for token, token_id in vocab.items():
        assert tokenizer.token_to_id(token) == token_id
        assert tokenizer.id_to_token(token_id) == token
    assert tokenizer.token_to_id(" a") is None
    assert tokenizer.token_to_id("漢") is None
    assert tokenizer.id_to_token(256) == "Ġa"
    assert tokenizer.decode([256]) == " a"
