import copy
import importlib
import json
import pickle
import weakref

from tokenizers import Component, Padding, decoders, models, normalizers, pre_tokenizers

TEXT = "Hello world, how are you?"


# --------------------------------------------------------------------------------------------------
# backwards-compatible behavior
# --------------------------------------------------------------------------------------------------


def test_submodules_import_by_name():
    assert importlib.import_module("tokenizers.models") is models


def test_spec_dict(gpt2_spec):
    gpt2_spec.custom = "original"

    assert gpt2_spec.custom == "original"


def test_spec_dict_copy(gpt2_spec):
    gpt2_spec.custom = "original only"
    assert gpt2_spec.__dict__ == {"custom": "original only"}

    for twin in (copy.copy(gpt2_spec), copy.deepcopy(gpt2_spec)):
        assert twin.compile().encode(TEXT).ids == gpt2_spec.compile().encode(TEXT).ids
        assert twin.__dict__ == {}
        twin.padding = Padding()
        assert gpt2_spec.padding is None


def test_weakref(gpt2_spec):
    assert weakref.ref(gpt2_spec)() is gpt2_spec


# Checks every component kind defines _unpickle
def test_all_components_pickle(gpt2_spec, bert_spec):
    kinds = (
        bert_spec.normalizer,
        gpt2_spec.pre_tokenizer,
        gpt2_spec.model,
        gpt2_spec.post_processor,
        gpt2_spec.decoder,
    )
    for component in kinds:
        for twin in (pickle.loads(pickle.dumps(component)), copy.deepcopy(component)):
            assert type(twin) is type(component)
            assert twin.__getstate__() == component.__getstate__()


def test_getters_class(gpt2_spec, bert_spec):
    assert isinstance(gpt2_spec.model, models.Model)
    assert isinstance(gpt2_spec.pre_tokenizer, pre_tokenizers.PreTokenizer)
    assert isinstance(gpt2_spec.decoder, decoders.Decoder)
    assert isinstance(bert_spec.normalizer, normalizers.Normalizer)
    assert isinstance(gpt2_spec.model, Component)
    assert gpt2_spec.normalizer is None


def test_bpe_repr(gpt2_spec):
    shown = repr(gpt2_spec.model)
    assert shown.startswith("BPE(")
    # Does not print the whole vocab
    assert len(shown) < 1000


def test_assigning_none_removes_a_component(gpt2_spec):
    ids = gpt2_spec.compile().encode(TEXT).ids
    gpt2_spec.pre_tokenizer = None
    assert gpt2_spec.pre_tokenizer is None
    assert gpt2_spec.compile().encode(TEXT).ids != ids


def test_handle_outlives_its_slot(gpt2_spec):
    pre_tokenizer = gpt2_spec.pre_tokenizer
    gpt2_spec.pre_tokenizer = None
    assert repr(pre_tokenizer).startswith("ByteLevel(")


# --------------------------------------------------------------------------------------------------
# transformers
# --------------------------------------------------------------------------------------------------


# Stop strings parse a decoder's `__getstate__` as JSON bytes.
def test_getstate(gpt2_spec):
    state = gpt2_spec.decoder.__getstate__()
    assert isinstance(state, bytes)
    assert json.loads(state)["type"] == "ByteLevel"


# --------------------------------------------------------------------------------------------------
# sglang and unsloth
# --------------------------------------------------------------------------------------------------


def test_move_component(gpt2_spec, bert_spec):
    lowercased = gpt2_spec.compile().encode("hello world").ids
    gpt2_spec.normalizer = bert_spec.normalizer
    assert gpt2_spec.normalizer.__getstate__() == bert_spec.normalizer.__getstate__()
    assert gpt2_spec.compile().encode("HELLO WORLD").ids == lowercased


# --------------------------------------------------------------------------------------------------
# concrete components
# --------------------------------------------------------------------------------------------------


def test_editing_a_component_writes_through(gpt2_spec):
    unchanged = copy.deepcopy(gpt2_spec)
    prepend = normalizers.Prepend("a")
    gpt2_spec.normalizer = prepend
    prepend.prepend = "b"
    assert type(gpt2_spec.normalizer) is normalizers.Prepend
    assert gpt2_spec.normalizer.prepend == "b"
    assert gpt2_spec.compile().encode(TEXT).ids == unchanged.compile().encode("b" + TEXT).ids
