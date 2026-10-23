import copy
import pickle
import weakref

from tokenizers import Padding

TEXT = "Hello world, how are you?"


def test_tokenizer_dict(gpt2):
    gpt2.decode = lambda ids, skip_special_tokens=False: "replaced"
    assert gpt2.decode([0]) == "replaced"


def test_tokenizer_dict_copy(gpt2):
    gpt2.custom = "original only"

    for twin in (copy.copy(gpt2), copy.deepcopy(gpt2)):
        assert twin.encode(TEXT).ids == gpt2.encode(TEXT).ids
        assert twin.__dict__ == {}
        twin.padding = Padding()
        assert gpt2.padding is None


def test_pickle_keeps_v1_json(gpt2):
    restored = pickle.loads(pickle.dumps(gpt2))

    assert restored.encode(TEXT).ids == gpt2.encode(TEXT).ids
    assert "pre_tokenizer=ByteLevel(" in repr(restored)
    assert "model=BPE(" in repr(restored)


def test_weakref(gpt2):
    assert weakref.ref(gpt2)() is gpt2
