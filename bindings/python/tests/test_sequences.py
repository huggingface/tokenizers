import copy
import json
import pickle

import pytest

from tokenizers import decoders, normalizers, pre_tokenizers, processors

TEXT = "Hello world 2024, how are you?"


# The JSON 0.23 writes for each.
@pytest.mark.parametrize(
    ("key", "sequence", "written"),
    [
        (
            "normalizer",
            normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()]),
            '{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Lowercase"}]}',
        ),
        (
            "pre_tokenizer",
            pre_tokenizers.Sequence(pre_tokenizers=[pre_tokenizers.Whitespace(), pre_tokenizers.WhitespaceSplit()]),
            '{"type":"Sequence","pretokenizers":[{"type":"Whitespace"},{"type":"WhitespaceSplit"}]}',
        ),
        (
            "decoder",
            decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()]),
            '{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]}',
        ),
    ],
    ids=["normalizers", "pre_tokenizers", "decoders"],
)
def test_sequence_json(wiki, key, sequence, written):
    assert sequence.__getstate__().decode() == written
    assert type(pickle.loads(pickle.dumps(sequence))) is type(sequence)
    tokenizer = copy.deepcopy(wiki)
    setattr(tokenizer, key, sequence)
    assert type(getattr(tokenizer, key)) is type(sequence)
    tokenizer.decode(tokenizer.encode(TEXT).ids)


def test_processor_sequence_holds_its_items_json(gpt2):
    sequence = processors.Sequence([gpt2.post_processor])
    assert json.loads(sequence.__getstate__()) == {
        "type": "Sequence",
        "processors": [json.loads(gpt2.post_processor.__getstate__())],
    }
    tokenizer = copy.deepcopy(gpt2)
    tokenizer.post_processor = sequence
    assert type(tokenizer.post_processor) is processors.Sequence
    tokenizer.decode(tokenizer.encode(TEXT).ids)


# sentence-transformers and vLLM iterate a tokenizer's normalizer and rebuild it.
def test_normalizer_sequence_iterates_its_components():
    sequence = normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()])
    assert [type(n) for n in sequence] == [normalizers.NFC, normalizers.Lowercase]
    assert len(sequence) == 2
    assert not normalizers.Sequence([])
    with pytest.raises(IndexError):
        sequence[2]


# transformers' Mistral regex fix replaces the first pre-tokenizer in place.
def test_pre_tokenizer_assignment_writes_through(wiki):
    tokenizer = copy.deepcopy(wiki)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.UnicodeScripts()])
    tokenizer.pre_tokenizer[0] = pre_tokenizers.BertPreTokenizer()
    written = json.loads(tokenizer.to_str())["pre_tokenizer"]["pretokenizers"]
    assert [p["type"] for p in written] == ["BertPreTokenizer", "UnicodeScripts"]
    assert type(tokenizer.pre_tokenizer[0]) is pre_tokenizers.BertPreTokenizer
    assert not hasattr(tokenizer.pre_tokenizer, "add_prefix_space")


def test_sequence_refuses_another_kind():
    with pytest.raises(TypeError):
        normalizers.Sequence([pre_tokenizers.Whitespace()])  # ty: ignore[invalid-argument-type]
