import copy
import json
import pickle
import weakref

import pytest

from tokenizers import Padding, decoders, models, normalizers, pre_tokenizers, processors

TEXT = "Hello world, how are you?"


# --------------------------------------------------------------------------------------------------
# backwards-compatible behavior
# --------------------------------------------------------------------------------------------------


def test_submodules_import_by_name():
    from tokenizers.decoders import Decoder
    from tokenizers.models import Model
    from tokenizers.normalizers import Normalizer
    from tokenizers.pre_tokenizers import PreTokenizer
    from tokenizers.processors import PostProcessor

    assert Decoder is decoders.Decoder
    assert Model is models.Model
    assert Normalizer is normalizers.Normalizer
    assert PreTokenizer is pre_tokenizers.PreTokenizer
    assert PostProcessor is processors.PostProcessor


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
        bert_spec.post_processor,
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
    assert gpt2_spec.normalizer is None


def test_bpe_repr(gpt2_spec):
    shown = repr(gpt2_spec.model)
    assert shown.startswith("BPE(")
    assert "...}" in shown


def test_assigning_none_removes_a_component(gpt2_spec):
    gpt2_spec.pre_tokenizer = None
    assert gpt2_spec.pre_tokenizer is None
    assert "    pre_tokenizer=None,\n" in repr(gpt2_spec)


def test_setters_refuse_another_kind(gpt2_spec):
    with pytest.raises(TypeError):
        gpt2_spec.normalizer = gpt2_spec.decoder
    with pytest.raises(TypeError):
        gpt2_spec.model = None


def test_deepcopy_is_independent(gpt2_spec):
    prepend = normalizers.Prepend("a")
    gpt2_spec.normalizer = prepend
    for twin in (copy.deepcopy(prepend), copy.deepcopy(gpt2_spec).normalizer):
        twin.prepend = "b"
        assert prepend.prepend == "a"


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


def test_shared_component_edits_reach_both_tokenizers(gpt2_spec, bert_spec):
    gpt2_spec.normalizer = normalizers.Prepend("a")
    bert_spec.normalizer = gpt2_spec.normalizer
    gpt2_spec.normalizer.prepend = "b"
    assert bert_spec.normalizer.prepend == "b"


# --------------------------------------------------------------------------------------------------
# concrete components
# --------------------------------------------------------------------------------------------------


def test_editing_a_component_writes_through(gpt2_spec):
    prepend = normalizers.Prepend("a")
    gpt2_spec.normalizer = prepend
    prepend.prepend = "b"
    assert type(gpt2_spec.normalizer) is normalizers.Prepend
    assert gpt2_spec.normalizer.prepend == "b"


def test_sequence_setitem(gpt2_spec):
    gpt2_spec.normalizer = normalizers.Sequence([normalizers.Prepend("a")])
    gpt2_spec.normalizer[0] = normalizers.Prepend("b")
    assert gpt2_spec.normalizer[0].prepend == "b"  # ty: ignore[unresolved-attribute]


def test_sequence_iter():
    sequence = normalizers.Sequence([normalizers.Prepend("a"), normalizers.Prepend("b")])
    assert len(sequence) == 2
    for item in sequence:
        assert isinstance(item, normalizers.Normalizer)


def test_sequence_item_is_shared():
    sequence = normalizers.Sequence([normalizers.Prepend("a")])
    item = sequence[0]
    assert isinstance(item, normalizers.Prepend)
    item.prepend = "b"
    assert sequence[0].prepend == "b"  # ty: ignore[unresolved-attribute]


def test_sequence_shares_its_arguments():
    prepend = normalizers.Prepend("a")
    sequence = normalizers.Sequence([prepend])
    prepend.prepend = "b"
    assert sequence[0].prepend == "b"  # ty: ignore[unresolved-attribute]


def test_sequence_negative_index():
    sequence = normalizers.Sequence([normalizers.Prepend("a"), normalizers.Prepend("b")])
    assert sequence[-1].__getstate__() == sequence[1].__getstate__()
    sequence[-2] = normalizers.Prepend("c")
    assert [item.prepend for item in sequence] == ["c", "b"]  # ty: ignore[unresolved-attribute]


def test_sequence_index_out_of_range():
    sequence = normalizers.Sequence([normalizers.Prepend("a")])
    with pytest.raises(IndexError):
        sequence[-2]
    with pytest.raises(IndexError):
        sequence[1] = normalizers.Prepend("b")


# A Sequence item that is a Sequence lets two sequences hold each other, and their JSON never ends.
def test_sequence_setitem_refuses_a_sequence():
    outer = normalizers.Sequence([normalizers.Prepend("a")])
    inner = normalizers.Sequence([normalizers.Prepend("b")])
    with pytest.raises(ValueError, match="can't be a Sequence"):
        outer[0] = inner


# transformers' pocket_tts converter wraps a loaded normalizer that may already be a Sequence.
def test_sequence_flattens_a_sequence_argument():
    inner = normalizers.Sequence([normalizers.Prepend("a"), normalizers.Prepend("b")])
    outer = normalizers.Sequence([inner, normalizers.Prepend("c")])
    flat = normalizers.Sequence([normalizers.Prepend(text) for text in "abc"])
    assert outer.__getstate__() == flat.__getstate__()
    item = inner[0]
    assert isinstance(item, normalizers.Prepend)
    item.prepend = "x"
    assert outer[0].prepend == "x"  # ty: ignore[unresolved-attribute]


# A file can nest a Sequence in a Sequence.
def test_loading_flattens_a_nested_sequence(gpt2_spec):
    file = json.loads(gpt2_spec.to_str())
    prepend = [{"type": "Prepend", "prepend": text} for text in "abc"]
    inner = {"type": "Sequence", "normalizers": prepend[:2]}
    file["normalizer"] = {"type": "Sequence", "normalizers": [inner, prepend[2]]}
    nested = type(gpt2_spec).from_str(json.dumps(file))
    assert json.loads(nested.to_str())["normalizer"]["normalizers"] == prepend
