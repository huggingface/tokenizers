import copy
import json
import pickle
import weakref

import pytest

from tokenizers import Padding, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors

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
        assert twin.to_str() == gpt2_spec.to_str()
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


def test_handle_keeps_its_component_after_reassignment(gpt2_spec, bert_spec):
    model = gpt2_spec.model
    gpt2_spec.model = bert_spec.model
    model.ignore_merges = True
    assert repr(model).startswith("BPE(")
    assert "ignore_merges" not in json.loads(gpt2_spec.model.__getstate__())


# transformers builds most tokenizers this way, then sets the other components.
def test_tokenizer_from_parts(gpt2_spec):
    tokenizer = Tokenizer(model=gpt2_spec.model)
    tokenizer.pre_tokenizer = gpt2_spec.pre_tokenizer
    tokenizer.post_processor = gpt2_spec.post_processor
    tokenizer.decoder = gpt2_spec.decoder
    written, declared = json.loads(tokenizer.to_str()), json.loads(gpt2_spec.to_str())
    for key in ("model", "pre_tokenizer", "post_processor", "decoder"):
        assert written[key] == declared[key]


def test_tokenizer_json_from_a_model():
    tokenizer = Tokenizer(models.BPE())
    written = json.loads(tokenizer.to_str())
    assert written["version"] == "2.0"
    assert written["added_tokens"] == []


# The keys 0.23 writes, plus 2.0's `byte_level`.
@pytest.mark.parametrize(
    ("model", "state"),
    [
        (
            models.BPE(
                {"a": 0, "b": 1, "ab": 2},
                [("a", "b")],
                dropout=0.5,
                unk_token="a",
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=True,
                byte_fallback=True,
                ignore_merges=True,
                byte_level=True,
            ),
            {
                "type": "BPE",
                "dropout": 0.5,
                "unk_token": "a",
                "continuing_subword_prefix": "",
                "end_of_word_suffix": "</w>",
                "fuse_unk": True,
                "byte_fallback": True,
                "ignore_merges": True,
                "byte_level": True,
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": [["a", "b"]],
            },
        ),
        (
            models.WordPiece({"a": 0}, unk_token="a", continuing_subword_prefix="@@", max_input_chars_per_word=5),
            {
                "type": "WordPiece",
                "unk_token": "a",
                "continuing_subword_prefix": "@@",
                "max_input_chars_per_word": 5,
                "vocab": {"a": 0},
            },
        ),
        (
            models.Unigram([("<unk>", 0.0), ("a", -1.0)], unk_id=0, byte_fallback=True),
            {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["a", -1.0]], "byte_fallback": True},
        ),
        (
            models.WordLevel({"a": 0}, unk_token="a"),
            {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"},
        ),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, models.Model) else None,
)
def test_model_writes_its_arguments(model, state):
    assert json.loads(model.__getstate__()) == state
    assert type(pickle.loads(pickle.dumps(model))) is type(model)


def test_bpe_from_files(gpt2_spec, tmp_path):
    declared = json.loads(gpt2_spec.to_str())["model"]
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(declared["vocab"]))
    merges = tmp_path / "merges.txt"
    merges.write_text("#version: 0.2\n" + "\n".join(" ".join(merge) for merge in declared["merges"]) + "\n")
    state = json.loads(models.BPE(vocab=str(vocab), merges=str(merges)).__getstate__())
    assert (state["vocab"], state["merges"]) == (declared["vocab"], declared["merges"])


def test_bpe_refuses_a_bad_merges_file(tmp_path):
    merges = tmp_path / "merges.txt"
    merges.write_text("#version: 0.2\na b\na b c\n")
    with pytest.raises(ValueError, match="line 3"):
        models.BPE({}, str(merges))


def test_bpe_getters():
    bpe = models.BPE(unk_token="a", continuing_subword_prefix="", end_of_word_suffix="</w>")
    assert (bpe.unk_token, bpe.continuing_subword_prefix, bpe.end_of_word_suffix) == ("a", "", "</w>")
    assert models.BPE().unk_token is None


def test_wordpiece_from_a_vocab_txt(bert_spec, tmp_path):
    declared = json.loads(bert_spec.to_str())["model"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("".join(f"{token}\n" for token in sorted(declared["vocab"], key=declared["vocab"].get)))
    assert json.loads(models.WordPiece(str(vocab)).__getstate__())["vocab"] == declared["vocab"]


# --------------------------------------------------------------------------------------------------
# transformers
# --------------------------------------------------------------------------------------------------


# transformers checks `type(tok.model).__name__`.
def test_model_type(gpt2_spec):
    assert type(gpt2_spec.model) is models.BPE
    assert type(gpt2_spec.model).__name__ == "BPE"


# transformers sets it on tekken and tiktoken models at every load.
def test_write_through_reference(gpt2_spec):
    gpt2_spec.model.ignore_merges = True
    assert gpt2_spec.model.ignore_merges is True
    assert json.loads(gpt2_spec.to_str())["model"]["ignore_merges"] is True


def test_field_of_the_wrong_json_type():
    bpe = models.Model._unpickle(b'{"type": "BPE", "ignore_merges": "yes"}')
    assert isinstance(bpe, models.BPE)
    with pytest.raises(ValueError, match="`ignore_merges`: invalid type"):
        bpe.ignore_merges


# Stop strings parse a decoder's `__getstate__` as JSON bytes.
def test_getstate(gpt2_spec):
    state = gpt2_spec.decoder.__getstate__()
    assert isinstance(state, bytes)
    assert json.loads(state)["type"] == "ByteLevel"


# --------------------------------------------------------------------------------------------------
# vLLM
# --------------------------------------------------------------------------------------------------


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


# Each kind lists a Sequence's items under its own key, and "pretokenizers" has no underscore.
@pytest.mark.parametrize(
    ("field", "items_key", "item", "sequence_class"),
    [
        ("normalizer", "normalizers", {"type": "Lowercase"}, normalizers.Sequence),
        ("pre_tokenizer", "pretokenizers", {"type": "Whitespace"}, pre_tokenizers.Sequence),
        ("decoder", "decoders", {"type": "Fuse"}, decoders.Sequence),
    ],
)
def test_loads_each_kinds_sequence(gpt2_spec, field, items_key, item, sequence_class):
    file = json.loads(gpt2_spec.to_str())
    file[field] = {"type": "Sequence", items_key: [item]}
    loaded = type(gpt2_spec).from_str(json.dumps(file))
    assert type(getattr(loaded, field)) is sequence_class


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
