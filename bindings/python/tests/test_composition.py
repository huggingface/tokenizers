import copy
import importlib
import json
import pickle
import weakref

import pytest

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


# As in 0.23, two tokenizers given the same component share it.
def test_component_moved_between_tokenizers_is_shared(gpt2):
    other = copy.deepcopy(gpt2)
    other.model = gpt2.model
    other.model.ignore_merges = True
    assert gpt2.model.ignore_merges is True


def test_edit_after_assigning_reaches_the_tokenizer(gpt2):
    bpe = pickle.loads(pickle.dumps(gpt2.model))
    gpt2.model = bpe
    bpe.ignore_merges = True
    assert gpt2.model.ignore_merges is True


def test_assign_component(gpt2):
    bpe = pickle.loads(pickle.dumps(gpt2.model))
    bpe.ignore_merges = True
    assert gpt2.model.ignore_merges is False
    gpt2.model = bpe
    assert gpt2.model.ignore_merges is True


def test_handle_keeps_its_component_after_reassignment(gpt2, bert):
    model = gpt2.model
    gpt2.model = bert.model
    model.ignore_merges = True
    assert repr(model).startswith("BPE(")
    assert "ignore_merges" not in json.loads(gpt2.model.__getstate__())


# 0.23 writes each of these as its type alone.
@pytest.mark.parametrize(
    ("key", "cls"),
    [
        ("normalizer", normalizers.Lowercase),
        ("normalizer", normalizers.ByteLevel),
        ("normalizer", normalizers.StripAccents),
        ("normalizer", normalizers.NFC),
        ("normalizer", normalizers.NFD),
        ("normalizer", normalizers.NFKC),
        ("normalizer", normalizers.NFKD),
        ("normalizer", normalizers.Nmt),
        ("pre_tokenizer", pre_tokenizers.Whitespace),
        ("pre_tokenizer", pre_tokenizers.WhitespaceSplit),
        ("pre_tokenizer", pre_tokenizers.BertPreTokenizer),
        ("pre_tokenizer", pre_tokenizers.UnicodeScripts),
        ("decoder", decoders.ByteFallback),
        ("decoder", decoders.Fuse),
    ],
    ids=lambda param: param if isinstance(param, str) else param.__name__,
)
def test_stateless_component(wiki, key, cls):
    tokenizer = copy.deepcopy(wiki)
    component = cls()
    assert json.loads(component.__getstate__()) == {"type": cls.__name__}
    assert type(pickle.loads(pickle.dumps(component))) is cls
    setattr(tokenizer, key, component)
    assert type(getattr(tokenizer, key)) is cls
    tokenizer.decode(tokenizer.encode(TEXT).ids)


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


# transformers checks `type(tok.model).__name__`.
def test_model_type(gpt2):
    assert type(gpt2.model) is models.BPE
    assert type(gpt2.model).__name__ == "BPE"


# transformers sets it on tekken and tiktoken models at every load.
def test_write_through_reference(gpt2):
    gpt2.model.ignore_merges = True
    assert gpt2.model.ignore_merges is True
    assert json.loads(gpt2.to_str())["model"]["ignore_merges"] is True


def test_field_of_the_wrong_json_type():
    bpe = models.Model._unpickle(b'{"type": "BPE", "ignore_merges": "yes"}')
    assert isinstance(bpe, models.BPE)
    with pytest.raises(ValueError, match="`ignore_merges`: invalid type"):
        bpe.ignore_merges


# Stop strings parse a decoder's `__getstate__` as JSON bytes.
def test_getstate(gpt2):
    state = gpt2.pre_tokenizer.__getstate__()
    assert isinstance(state, bytes)
    declared = json.loads(GPT2.read_text())["pre_tokenizer"]
    assert list(json.loads(state)) == list(declared)


# --------------------------------------------------------------------------------------------------
# vLLM
# --------------------------------------------------------------------------------------------------


# vLLM gives each thread its own deep copy.
def test_deepcopy_write(gpt2):
    twin = copy.deepcopy(gpt2)
    twin.model.ignore_merges = True
    # Ensure both copies are independent
    assert gpt2.model.ignore_merges is False


# --------------------------------------------------------------------------------------------------
# sglang and unsloth
# --------------------------------------------------------------------------------------------------


def test_move_component(gpt2, bert):
    lowercased = gpt2.encode("hello world").ids
    gpt2.normalizer = bert.normalizer
    assert gpt2.normalizer.__getstate__() == bert.normalizer.__getstate__()
    assert gpt2.encode("HELLO WORLD").ids == lowercased
