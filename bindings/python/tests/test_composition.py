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


# As in 0.23, two tokenizers given the same component share it.
def test_component_moved_between_tokenizers_is_shared(gpt2_spec):
    other = copy.deepcopy(gpt2_spec)
    other.model = gpt2_spec.model
    other.model.ignore_merges = True
    assert gpt2_spec.model.ignore_merges is True


def test_edit_after_assigning_reaches_the_tokenizer(gpt2_spec):
    bpe = pickle.loads(pickle.dumps(gpt2_spec.model))
    gpt2_spec.model = bpe
    bpe.ignore_merges = True
    assert gpt2_spec.model.ignore_merges is True


def test_assign_component(gpt2_spec):
    bpe = pickle.loads(pickle.dumps(gpt2_spec.model))
    bpe.ignore_merges = True
    assert gpt2_spec.model.ignore_merges is False
    gpt2_spec.model = bpe
    assert gpt2_spec.model.ignore_merges is True


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
    ids = tokenizer.compile().encode(TEXT).ids
    assert ids == gpt2_spec.compile().encode(TEXT).ids
    assert tokenizer.compile().decode(ids) == gpt2_spec.compile().decode(ids)


def test_tokenizer_json_from_a_model():
    tokenizer = Tokenizer(models.BPE())
    written = json.loads(tokenizer.to_str())
    assert written["version"] == "2.0"
    assert written["added_tokens"] == []
    assert written["model"] == json.loads(tokenizer.model.__getstate__())


# The keys 0.23 writes, plus 2.0's `byte_level`.
@pytest.mark.parametrize(
    ("model", "keys"),
    [
        (
            models.BPE(),
            ["type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix"]
            + ["fuse_unk", "byte_fallback", "ignore_merges", "byte_level", "vocab", "merges"],
        ),
        (
            models.WordPiece(),
            ["type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word", "vocab"],
        ),
        (models.WordLevel(), ["type", "vocab", "unk_token"]),
        (models.Unigram([("<unk>", 0.0)]), ["type", "unk_id", "vocab", "byte_fallback"]),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, models.Model) else None,
)
def test_model_json_keys(model, keys):
    assert sorted(json.loads(model.__getstate__())) == sorted(keys)
    assert type(pickle.loads(pickle.dumps(model))) is type(model)


@pytest.mark.parametrize(("state", "shown"), [(b'{"type": "Bogus"}', '"Bogus"'), (b"{}", "null")])
def test_unpickle_refuses_an_unknown_model_type(state, shown):
    with pytest.raises(ValueError, match=f'^Unknown model "type": {shown}$'):
        models.Model._unpickle(state)


def test_bpe_from_dicts(wiki_spec):
    declared = json.loads(wiki_spec.to_str())["model"]
    vocab = dict(reversed(declared["vocab"].items()))
    merges = [tuple(merge) for merge in declared["merges"]]
    tokenizer = copy.deepcopy(wiki_spec)
    tokenizer.model = models.BPE(vocab, merges, unk_token="[UNK]", byte_level=declared["byte_level"])
    text = f"{TEXT} 😀"
    assert tokenizer.compile().encode(text).ids == wiki_spec.compile().encode(text).ids


def test_bpe_from_files(gpt2_spec, tmp_path):
    declared = json.loads(gpt2_spec.to_str())["model"]
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(declared["vocab"]))
    merges = tmp_path / "merges.txt"
    merges.write_text("#version: 0.2\n" + "\n".join(" ".join(merge) for merge in declared["merges"]) + "\n")
    tokenizer = copy.deepcopy(gpt2_spec)
    tokenizer.model = models.BPE(vocab=str(vocab), merges=str(merges), byte_level=declared["byte_level"])
    assert tokenizer.compile().encode(TEXT).ids == gpt2_spec.compile().encode(TEXT).ids


def test_bpe_refuses_a_bad_merges_file(tmp_path):
    merges = tmp_path / "merges.txt"
    merges.write_text("#version: 0.2\na b\na b c\n")
    with pytest.raises(ValueError, match="line 3"):
        models.BPE({}, str(merges))


def test_bpe_options():
    bpe = models.BPE(
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
    )
    assert json.loads(bpe.__getstate__()) == {
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
    }
    assert (bpe.unk_token, bpe.continuing_subword_prefix, bpe.end_of_word_suffix) == ("a", "", "</w>")
    assert models.BPE().unk_token is None


def test_wordpiece(bert_spec, tmp_path):
    declared = json.loads(bert_spec.to_str())["model"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("".join(f"{token}\n" for token in sorted(declared["vocab"], key=declared["vocab"].get)))
    for source in (declared["vocab"], str(vocab)):
        tokenizer = copy.deepcopy(bert_spec)
        tokenizer.model = models.WordPiece(source, unk_token="[UNK]")
        assert tokenizer.model.continuing_subword_prefix == "##"
        text = f"{TEXT} 😀"
        assert tokenizer.compile().encode(text).ids == bert_spec.compile().encode(text).ids


def test_unigram(wiki_spec):
    tokenizer = copy.deepcopy(wiki_spec)
    vocab = [("<unk>", 0.0), ("a", -1.0), ("b", -1.0), ("ab", -1.5)]
    tokenizer.model = models.Unigram(vocab, unk_id=0, byte_fallback=False)
    assert tokenizer.compile().encode("ab c", add_special_tokens=False).ids == [3, 0]


def test_wordlevel(wiki_spec):
    tokenizer = copy.deepcopy(wiki_spec)
    tokenizer.model = models.WordLevel({"hello": 0, "[UNK]": 1}, unk_token="[UNK]")
    assert tokenizer.compile().encode("hello world", add_special_tokens=False).ids == [0, 1]


def test_handle_outlives_its_slot(gpt2_spec):
    decoder = gpt2_spec.decoder
    gpt2_spec.decoder = None
    assert repr(decoder).startswith("ByteLevel(")


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


# vLLM gives each thread its own deep copy.
def test_deepcopy_write(gpt2_spec):
    twin = copy.deepcopy(gpt2_spec)
    twin.model.ignore_merges = True
    # Ensure both copies are independent
    assert gpt2_spec.model.ignore_merges is False


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
    unchanged = copy.deepcopy(gpt2_spec)
    prepend = normalizers.Prepend("a")
    gpt2_spec.normalizer = prepend
    prepend.prepend = "b"
    assert type(gpt2_spec.normalizer) is normalizers.Prepend
    assert gpt2_spec.normalizer.prepend == "b"
    assert gpt2_spec.compile().encode(TEXT).ids == unchanged.compile().encode("b" + TEXT).ids
