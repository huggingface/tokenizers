import json

import pytest

from tokenizers import Padding, Tokenizer, Truncation

TEXTS = ["Hello there, how are you?", "naïve café", "Hello<|eot_id|>world", ""]


@pytest.mark.parametrize("text", TEXTS)
def test_tokenize_is_encode_then_decode_tokens(gpt2, llama3, bert, text):
    for tokenizer in (gpt2, llama3, bert):
        assert tokenizer.tokenize(text) == tokenizer.decode_tokens(tokenizer.encode(text).ids)


def test_tokenize_forwards_add_special_tokens(llama3):
    encoding = llama3.encode("Hello there", add_special_tokens=False)

    assert llama3.tokenize("Hello there", add_special_tokens=False) == llama3.decode_tokens(encoding.ids)
    assert llama3.tokenize("Hello there") != llama3.decode_tokens(encoding.ids)


def test_tokenize_forwards_padding(wiki):
    padding = Padding(length=4, pad_id=3)
    encoding = wiki.encode("Hello there", padding=padding)

    assert wiki.tokenize("Hello there", padding=padding) == wiki.decode_tokens(encoding.ids)
    assert len(wiki.tokenize("Hello there")) == 2


def test_tokenize_forwards_truncation(gpt2):
    truncation = Truncation(max_length=2)
    encoding = gpt2.encode("Hello there, how are you?", truncation=truncation)

    assert gpt2.tokenize("Hello there, how are you?", truncation=truncation) == gpt2.decode_tokens(encoding.ids)
    assert len(gpt2.tokenize("Hello there, how are you?")) > 2


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("你好", ["你", "好"]),
        ("a\U0002b820\U0002b91fb", ["a", "[UNK]", "[UNK]", "b"]),
    ],
)
@pytest.mark.parametrize("handle_chinese_chars", [True, False])
def test_bert_chinese_boundaries(tmp_path, text, tokens, handle_chinese_chars):
    config = {
        "version": "1.0",
        "added_tokens": [],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": False,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": False,
            "lowercase": False,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "a": 1, "b": 2, "你": 3, "好": 4},
        },
    }
    path = tmp_path / "bert.json"
    path.write_text(json.dumps(config))
    tokenizer = Tokenizer.from_file(path)

    assert tokenizer.tokenize(text) == (tokens if handle_chinese_chars else ["[UNK]"])
