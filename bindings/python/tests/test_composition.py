import copy
import pickle
import weakref

from tokenizers import Padding

TEXT = "Hello world, how are you?"


# CLIP replaces `decode` on the instance at every load.
def test_tokenizer_accepts_instance_attributes(gpt2):
    gpt2.decode = lambda ids, skip_special_tokens=False: "replaced"

    assert gpt2.decode([0]) == "replaced"


# 0.23 drops the instance's attributes on copy and pickle too.
def test_copies_are_independent_and_drop_attributes(gpt2):
    gpt2.custom = "original only"

    for twin in (copy.copy(gpt2), copy.deepcopy(gpt2)):
        assert twin.encode(TEXT).ids == gpt2.encode(TEXT).ids
        assert twin.__dict__ == {}
        twin.padding = Padding()
        assert gpt2.padding is None


# Loading rebuilds gpt2.json's ByteLevel pre-tokenizer as a Split; pickle keeps the file's.
# gpt2.json's model has no "type"; 0.23 fills it in.
def test_pickle_keeps_the_components_as_the_file_wrote_them(gpt2):
    restored = pickle.loads(pickle.dumps(gpt2))

    assert restored.encode(TEXT).ids == gpt2.encode(TEXT).ids
    assert "pre_tokenizer=ByteLevel(" in repr(restored)
    assert "model=BPE(" in repr(restored)


def test_tokenizer_supports_weak_references(gpt2):
    assert weakref.ref(gpt2)() is gpt2
