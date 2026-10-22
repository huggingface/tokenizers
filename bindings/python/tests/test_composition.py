import copy
import importlib
import json
import pickle
import weakref

from conftest import GPT2
from tokenizers import Component, Padding, decoders, models, normalizers, pre_tokenizers

TEXT = "Hello world, how are you?"


# --------------------------------------------------------------------------------------------------
# backwards-compatible behavior
# --------------------------------------------------------------------------------------------------


def test_submodules_import_by_name():
    assert importlib.import_module("tokenizers.models") is models


def test_tokenizer_supports_weak_references(gpt2):
    assert weakref.ref(gpt2)() is gpt2


def test_copies_are_independent_and_drop_dict_attributes(gpt2):
    gpt2.custom = "original only"
    for twin in (copy.copy(gpt2), copy.deepcopy(gpt2)):
        assert twin.encode(TEXT).ids == gpt2.encode(TEXT).ids
        assert twin.__dict__ == {}
        twin.padding = Padding()
        assert gpt2.padding is None


def test_pickle_keeps_the_json_spec_verbatim(gpt2):
    restored = pickle.loads(pickle.dumps(gpt2))
    assert restored.encode(TEXT).ids == gpt2.encode(TEXT).ids
    assert "pre_tokenizer=ByteLevel(" in repr(restored)
    assert "model=BPE(" in repr(restored)


# Checks every component kind defines _unpickle
def test_all_components_pickle(gpt2, bert):
    kinds = (bert.normalizer, gpt2.pre_tokenizer, gpt2.model, gpt2.post_processor, gpt2.decoder)
    for component in kinds:
        for twin in (pickle.loads(pickle.dumps(component)), copy.deepcopy(component)):
            assert type(twin) is type(component)
            assert twin.__getstate__() == component.__getstate__()


def test_getters_class(gpt2, bert):
    assert isinstance(gpt2.model, models.Model)
    assert isinstance(gpt2.pre_tokenizer, pre_tokenizers.PreTokenizer)
    assert isinstance(gpt2.decoder, decoders.Decoder)
    assert isinstance(bert.normalizer, normalizers.Normalizer)
    assert isinstance(gpt2.model, Component)
    assert gpt2.normalizer is None


def test_bpe_repr(gpt2):
    shown = repr(gpt2.model)
    assert shown.startswith("BPE(")
    # Does not print the whole vocab
    assert len(shown) < 1000


def test_assigning_none_removes_a_component(gpt2):
    ids = gpt2.encode(TEXT).ids
    gpt2.pre_tokenizer = None
    assert gpt2.pre_tokenizer is None
    assert gpt2.encode(TEXT).ids != ids


def test_handle_outlives_its_slot(gpt2):
    pre_tokenizer = gpt2.pre_tokenizer
    gpt2.pre_tokenizer = None
    assert repr(pre_tokenizer).startswith("ByteLevel(")


# --------------------------------------------------------------------------------------------------
# transformers
# --------------------------------------------------------------------------------------------------


# CLIP replaces `decode` on the instance at every load.
def test_replace_attribute(gpt2):
    gpt2.decode = lambda ids, skip_special_tokens=False: "replaced"
    assert gpt2.decode([0]) == "replaced"


# Stop strings parse a decoder's `__getstate__` as JSON bytes.
def test_getstate(gpt2):
    state = gpt2.pre_tokenizer.__getstate__()
    assert isinstance(state, bytes)
    declared = json.loads(GPT2.read_text())["pre_tokenizer"]
    assert list(json.loads(state)) == list(declared)


# --------------------------------------------------------------------------------------------------
# sglang and unsloth
# --------------------------------------------------------------------------------------------------


def test_move_component(gpt2, bert):
    lowercased = gpt2.encode("hello world").ids
    gpt2.normalizer = bert.normalizer
    assert gpt2.normalizer.__getstate__() == bert.normalizer.__getstate__()
    assert gpt2.encode("HELLO WORLD").ids == lowercased
