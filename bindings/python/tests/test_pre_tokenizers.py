import copy
import json
import pickle

import pytest

from tokenizers import Regex, pre_tokenizers

TEXT = "Hello world, how are you? 1234"


# The JSON 0.23 writes for each, key order included, and a fixture whose model accepts it.
@pytest.mark.parametrize(
    ("pre_tokenizer", "written", "fixture"),
    [
        (
            pre_tokenizers.ByteLevel(),
            '{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}',
            # The core refuses `add_prefix_space=True` when it compiles a tokenizer.
            None,
        ),
        (
            pre_tokenizers.ByteLevel(add_prefix_space=False),
            '{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}',
            "gpt2",
        ),
        (
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            '{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}',
            "llama3",
        ),
        (
            pre_tokenizers.Metaspace(),
            '{"type":"Metaspace","replacement":"▁","prepend_scheme":"always","split":true}',
            "wiki",
        ),
        (
            pre_tokenizers.Metaspace(replacement="_", prepend_scheme="first", split=False),
            '{"type":"Metaspace","replacement":"_","prepend_scheme":"first","split":false}',
            "wiki",
        ),
        (
            pre_tokenizers.Split("a", "isolated"),
            '{"type":"Split","pattern":{"String":"a"},"behavior":"Isolated","invert":false}',
            "gpt2",
        ),
        (
            pre_tokenizers.Split(pattern=Regex(r"\s+"), behavior="contiguous"),
            '{"type":"Split","pattern":{"Regex":"\\\\s+"},"behavior":"Contiguous","invert":false}',
            "gpt2",
        ),
        (
            pre_tokenizers.Split(pattern=" ", behavior="merged_with_previous", invert=True),
            '{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":true}',
            "gpt2",
        ),
        (
            pre_tokenizers.Split(" ", "merged_with_next"),
            '{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithNext","invert":false}',
            "gpt2",
        ),
        (
            pre_tokenizers.Split(" ", "removed"),
            '{"type":"Split","pattern":{"String":" "},"behavior":"Removed","invert":false}',
            "gpt2",
        ),
        (pre_tokenizers.Digits(), '{"type":"Digits","individual_digits":false}', "wiki"),
        (pre_tokenizers.Digits(individual_digits=True), '{"type":"Digits","individual_digits":true}', "wiki"),
        (pre_tokenizers.Punctuation(), '{"type":"Punctuation","behavior":"Isolated"}', "wiki"),
        (pre_tokenizers.Punctuation("contiguous"), '{"type":"Punctuation","behavior":"Contiguous"}', "wiki"),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, pre_tokenizers.PreTokenizer) else None,
)
def test_pre_tokenizer_json(request, pre_tokenizer, written, fixture):
    assert pre_tokenizer.__getstate__().decode() == written
    twin = pickle.loads(pickle.dumps(pre_tokenizer))
    assert type(twin) is type(pre_tokenizer)
    if fixture is None:
        return
    tokenizer = copy.deepcopy(request.getfixturevalue(fixture))
    tokenizer.pre_tokenizer = pre_tokenizer
    assert type(tokenizer.pre_tokenizer) is type(pre_tokenizer)
    tokenizer.decode(tokenizer.encode(TEXT).ids)


def test_byte_level_add_prefix_space():
    assert pre_tokenizers.ByteLevel().add_prefix_space is True
    assert pre_tokenizers.ByteLevel(add_prefix_space=False).add_prefix_space is False


# transformers passes it to byte-level trainers as `initial_alphabet`.
def test_byte_level_alphabet_has_one_char_per_byte():
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    assert len(set(alphabet)) == 256
    assert all(len(char) == 1 for char in alphabet)


def test_metaspace_replacement_and_split():
    metaspace = pre_tokenizers.Metaspace(replacement="_", split=False)
    assert metaspace.replacement == "_"
    assert metaspace.split is False


def test_metaspace_refuses_an_unknown_prepend_scheme():
    with pytest.raises(ValueError):
        pre_tokenizers.Metaspace(prepend_scheme="sometimes")


@pytest.mark.parametrize(
    "build",
    [lambda: pre_tokenizers.Split("a", "dropped"), lambda: pre_tokenizers.Punctuation("dropped")],
    ids=["Split", "Punctuation"],
)
def test_refuses_an_unknown_behavior(build):
    with pytest.raises(ValueError):
        build()


# 0.23 writes each of these as its type alone.
@pytest.mark.parametrize(
    ("key", "cls"),
    [
        ("pre_tokenizer", pre_tokenizers.Whitespace),
        ("pre_tokenizer", pre_tokenizers.WhitespaceSplit),
        ("pre_tokenizer", pre_tokenizers.BertPreTokenizer),
        ("pre_tokenizer", pre_tokenizers.UnicodeScripts),
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


# The JSON 0.23 writes for each.
@pytest.mark.parametrize(
    ("key", "sequence", "written"),
    [
        (
            "pre_tokenizer",
            pre_tokenizers.Sequence(pre_tokenizers=[pre_tokenizers.Whitespace(), pre_tokenizers.Digits()]),
            '{"type":"Sequence","pretokenizers":[{"type":"Whitespace"},{"type":"Digits","individual_digits":false}]}',
        ),
    ],
    ids=["pre_tokenizers"],
)
def test_sequence_json(wiki, key, sequence, written):
    assert sequence.__getstate__().decode() == written
    assert type(pickle.loads(pickle.dumps(sequence))) is type(sequence)
    tokenizer = copy.deepcopy(wiki)
    setattr(tokenizer, key, sequence)
    assert type(getattr(tokenizer, key)) is type(sequence)
    tokenizer.decode(tokenizer.encode(TEXT).ids)


# transformers' Mistral regex fix replaces the first pre-tokenizer in place.
def test_pre_tokenizer_assignment_writes_through(wiki):
    tokenizer = copy.deepcopy(wiki)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits()])
    tokenizer.pre_tokenizer[0] = pre_tokenizers.Punctuation()
    written = json.loads(tokenizer.to_str())["pre_tokenizer"]["pretokenizers"]
    assert [p["type"] for p in written] == ["Punctuation", "Digits"]
    assert type(tokenizer.pre_tokenizer[0]) is pre_tokenizers.Punctuation
    assert not hasattr(tokenizer.pre_tokenizer, "add_prefix_space")
