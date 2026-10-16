import copy
import json
import pickle

import pytest

from tokenizers import Padding, Tokenizer, Truncation


@pytest.fixture
def unigram(tmp_path):
    # An unknown run keeps its text in Encoding.tokens, even when several runs share id 0.
    config = {
        "version": "1.0",
        "added_tokens": [
            {
                "id": 3,
                "content": "<s>",
                "special": True,
                "normalized": False,
                "single_word": False,
                "lstrip": True,
                "rstrip": True,
            }
        ],
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<s>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [
                {"SpecialToken": {"id": "<s>", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "<s>", "type_id": 1}},
                {"Sequence": {"id": "B", "type_id": 1}},
            ],
            "special_tokens": {"<s>": {"id": "<s>", "ids": [3], "tokens": ["<s>"]}},
        },
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["a", -1.0], ["b", -1.0]]},
    }
    path = tmp_path / "unigram.json"
    path.write_text(json.dumps(config))
    return Tokenizer.from_file(path)


def test_unknown_text_and_added_token_match(unigram):
    for _ in range(2):
        encoding = unigram.encode("A😀  <s> \tB😁")
        assert encoding.ids == [3, 1, 0, 3, 2, 0]
        assert encoding.tokens == ["<s>", "a", "😀", "  <s> \t", "b", "😁"]
        assert unigram.tokenize("A😀  <s> \tB😁") == encoding.tokens
        assert unigram.decode_tokens(encoding.ids) == ["<s>", "a", "<unk>", "<s>", "b", "<unk>"]
    as_text = unigram.encode("a<s>b", encode_special_tokens=True)
    assert as_text.ids == [3, 1, 0, 2]
    assert as_text.tokens == ["<s>", "a", "<s>", "b"]


def test_normalized_added_token_text(unigram, tmp_path):
    path = tmp_path / "unigram.json"
    config = json.loads(path.read_text())
    config["added_tokens"][0].update(normalized=True, content="<S>")
    path.write_text(json.dumps(config))
    tokenizer = Tokenizer.from_file(path)
    assert tokenizer.encode("A😀  <S> \tB😁").tokens == ["<s>", "a", "😀", "  <s> \t", "b", "😁"]


@pytest.mark.parametrize("direction", ["left", "right"])
def test_tokens_follow_pair_truncation_and_padding(unigram, direction):
    truncation = Truncation(max_length=5, direction=direction)
    padding = Padding(length=7, direction=direction, pad_id=0, pad_token="<pad>")
    encoding = unigram.encode(("a😀b", "b😁a"), truncation=truncation, padding=padding)
    if direction == "left":
        assert encoding.tokens == ["<pad>", "<pad>", "<s>", "b", "<s>", "😁", "a"]
    else:
        assert encoding.tokens == ["<s>", "a", "<s>", "b", "😁", "<pad>", "<pad>"]
    assert len(encoding.tokens) == len(encoding.ids) == 7


def test_tokens_survive_copy_and_pickle(unigram):
    encoding = unigram.encode("😀")
    for copied in [copy.copy(encoding), copy.deepcopy(encoding), pickle.loads(pickle.dumps(encoding))]:
        assert copied == encoding
        assert copied.tokens == ["<s>", "😀"]
    # IDs alone cannot distinguish these encodings.
    assert unigram.encode("😀").ids == unigram.encode("😁").ids
    assert unigram.encode("😀") != unigram.encode("😁")
    exposed = encoding.tokens
    exposed.clear()
    assert encoding.tokens == ["<s>", "😀"]


def test_parallel_chunks_preserve_token_text(unigram):
    left, right = "a😀 " * 5000, "b😁 " * 5000
    text = left + "<s> " + right
    encoding = unigram.encode(text)
    assert encoding.tokens == ["<s>"] + ["a", "😀"] * 5000 + [" <s> "] + ["b", "😁"] * 5000
    batch = unigram.encode_batch([text, (text, "😀")])
    assert batch[0].tokens == encoding.tokens
    assert batch[1].tokens == encoding.tokens + ["<s>", "😀"]


def test_empty_text_and_no_specials(unigram):
    assert unigram.encode("").tokens == ["<s>"]
    assert unigram.encode("", add_special_tokens=False).tokens == []
    assert unigram.encode_batch(["😀", "😁"], add_special_tokens=False)[1].tokens == ["😁"]
