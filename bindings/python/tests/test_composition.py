import copy
import json
import pickle
import weakref

from conftest import GPT2
from tokenizers import Padding, Tokenizer

TEXT = "Hello world, how are you?"


def test_spec_dict_copy(gpt2_spec):
    gpt2_spec.custom = "original only"

    for twin in (copy.copy(gpt2_spec), copy.deepcopy(gpt2_spec)):
        assert twin.compile().encode(TEXT).ids == gpt2_spec.compile().encode(TEXT).ids
        assert twin.__dict__ == {}
        twin.padding = Padding()
        assert gpt2_spec.padding is None


def test_v1_file_loads_canonical(gpt2_spec):
    spec = repr(gpt2_spec)

    assert "pre_tokenizer=Split(" in spec
    assert "byte_level=True" in spec


def test_pickle_keeps_the_spec(gpt2_spec):
    restored = pickle.loads(pickle.dumps(gpt2_spec))

    assert restored.compile().encode(TEXT).ids == gpt2_spec.compile().encode(TEXT).ids
    assert repr(restored) == repr(gpt2_spec)


def test_weakref(gpt2_spec):
    assert weakref.ref(gpt2_spec)() is gpt2_spec


def test_spec_repr_charsmap(tmp_path):
    declared = json.loads(GPT2.read_text())
    declared["normalizer"] = {"type": "Precompiled", "precompiled_charsmap": "A" * 4096}
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(declared))

    assert "    normalizer=Precompiled(precompiled_charsmap=...)," in repr(Tokenizer.from_file(path))
