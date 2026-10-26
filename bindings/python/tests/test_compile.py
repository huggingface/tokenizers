import copy
import json
import pickle

import pytest

from conftest import BERT, GPT2
from tokenizers import Padding, TokenizerSpec, Truncation

TEXT = "Hello there, how are you today?"
EOT = 50256


def test_frozen(gpt2_spec):
    compiled = gpt2_spec.compile()

    gpt2_spec.padding = Padding(length=12, pad_id=EOT)
    gpt2_spec.truncation = Truncation(4)

    assert len(compiled.encode(TEXT)) == 8
    assert len(gpt2_spec.compile().encode(TEXT)) == 12


# transformers deepcopies every tokenizer it loads, and a real copy would mean a recompile.
def test_copy(gpt2):
    assert copy.copy(gpt2) is gpt2
    assert copy.deepcopy(gpt2) is gpt2


def test_pickle(gpt2_spec):
    gpt2_spec.padding = Padding(length=12, pad_id=EOT)
    compiled = gpt2_spec.compile()

    restored = pickle.loads(pickle.dumps(compiled))

    assert restored.encode(TEXT) == compiled.encode(TEXT)
    assert len(restored.encode(TEXT)) == 12


def test_compile_error(tmp_path):
    declared = json.loads(GPT2.read_text())
    declared["pre_tokenizer"]["type"] = "NotAPreTokenizer"
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(declared))
    spec = TokenizerSpec.from_file(path)

    with pytest.raises(ValueError, match="NotAPreTokenizer"):
        spec.compile()


# Rendering the model's JSON would mean writing the whole vocabulary back out on every call.
def test_repr():
    spec = TokenizerSpec.from_file(BERT)
    spec.padding = Padding(length=12, pad_id=3)
    spec.truncation = Truncation(10)

    rendered = repr(spec.compile())

    assert rendered.startswith("CompiledTokenizer(\n    added_tokens=[\n")
    assert rendered[rendered.index("    normalizer=") :] == "\n".join(
        [
            "    normalizer=Sequence(normalizers=[NFD(), Lowercase(), StripAccents()]),",
            "    pre_tokenizer=Whitespace(),",
            "    model=WordPiece,",
            '    post_processor=Template(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1"),',
            "    decoder=None,",
            "    padding=Padding(",
            '        direction="right",',
            "        pad_id=3,",
            "        pad_type_id=0,",
            '        pad_token="[PAD]",',
            "        length=12,",
            "        pad_to_multiple_of=None,",
            "    ),",
            '    truncation=Truncation(max_length=10, strategy="longest_first", direction="right"),',
            ")",
        ]
    )


def test_repr_nested(llama3):
    rendered = repr(llama3)

    assert '            "id": 128004,' in rendered
    assert '            "id": 128005,' not in rendered
    assert "        ...,\n    ],\n    normalizer=None," in rendered
    assert "    pre_tokenizer=Sequence(\n        pretokenizers=[\n            Split(\n" in rendered
    assert '                behavior="Isolated",\n                invert=False,\n' in rendered


def test_spec_repr_charsmap(tmp_path):
    declared = json.loads(GPT2.read_text())
    declared["normalizer"] = {"type": "Precompiled", "precompiled_charsmap": "A" * 4096}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(declared))

    assert "    normalizer=Precompiled(precompiled_charsmap=...)," in repr(TokenizerSpec.from_file(path))
