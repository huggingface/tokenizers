import copy
import pickle
import weakref

from tokenizers import Padding

TEXT = "Hello world, how are you?"


def test_spec_dict(gpt2_spec):
    gpt2_spec.custom = "original"

    assert gpt2_spec.custom == "original"


def test_spec_dict_copy(gpt2_spec):
    gpt2_spec.custom = "original only"

    for twin in (copy.copy(gpt2_spec), copy.deepcopy(gpt2_spec)):
        assert twin.compile().encode(TEXT).ids == gpt2_spec.compile().encode(TEXT).ids
        assert twin.__dict__ == {}
        twin.padding = Padding()
        assert gpt2_spec.padding is None


def test_pickle_keeps_v1_json(gpt2_spec):
    restored = pickle.loads(pickle.dumps(gpt2_spec))

    assert restored.compile().encode(TEXT).ids == gpt2_spec.compile().encode(TEXT).ids
    assert "pre_tokenizer=ByteLevel(" in repr(restored)
    assert "model=BPE(" in repr(restored)


def test_weakref(gpt2_spec):
    assert weakref.ref(gpt2_spec)() is gpt2_spec
