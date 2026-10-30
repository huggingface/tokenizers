import json
import pickle

import pytest

from tokenizers import Regex, pre_tokenizers

TEXT = "Hello world, how are you? 1234"


# The JSON 0.23 writes for each, and a fixture whose model accepts it.
@pytest.mark.parametrize(
    ("pre_tokenizer", "written", "fixture"),
    [
        (
            pre_tokenizers.Split("a", "isolated"),
            '{"type":"Split","pattern":{"String":"a"},"behavior":"Isolated","invert":false}',
            "gpt2_spec",
        ),
        (
            pre_tokenizers.Split(pattern=Regex(r"\s+"), behavior="contiguous"),
            '{"type":"Split","pattern":{"Regex":"\\\\s+"},"behavior":"Contiguous","invert":false}',
            "gpt2_spec",
        ),
        (
            pre_tokenizers.Split(pattern=" ", behavior="merged_with_previous", invert=True),
            '{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":true}',
            "gpt2_spec",
        ),
        (
            pre_tokenizers.Split(" ", "merged_with_next"),
            '{"type":"Split","pattern":{"String":" "},"behavior":"MergedWithNext","invert":false}',
            "gpt2_spec",
        ),
        (
            pre_tokenizers.Split(" ", "removed"),
            '{"type":"Split","pattern":{"String":" "},"behavior":"Removed","invert":false}',
            "gpt2_spec",
        ),
        (pre_tokenizers.Digits(), '{"type":"Digits","individual_digits":false}', "wiki_spec"),
        (pre_tokenizers.Digits(individual_digits=True), '{"type":"Digits","individual_digits":true}', "wiki_spec"),
        (pre_tokenizers.Punctuation(), '{"type":"Punctuation","behavior":"Isolated"}', "wiki_spec"),
        (pre_tokenizers.Punctuation("contiguous"), '{"type":"Punctuation","behavior":"Contiguous"}', "wiki_spec"),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, pre_tokenizers.PreTokenizer) else None,
)
def test_pre_tokenizer_json(request, pre_tokenizer, written, fixture):
    assert json.loads(pre_tokenizer.__getstate__()) == json.loads(written)
    twin = pickle.loads(pickle.dumps(pre_tokenizer))
    assert type(twin) is type(pre_tokenizer)
    tokenizer = request.getfixturevalue(fixture)
    tokenizer.pre_tokenizer = pre_tokenizer
    assert type(tokenizer.pre_tokenizer) is type(pre_tokenizer)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


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
def test_stateless_component(wiki_spec, key, cls):
    tokenizer = wiki_spec
    component = cls()
    assert json.loads(component.__getstate__()) == {"type": cls.__name__}
    assert type(pickle.loads(pickle.dumps(component))) is cls
    setattr(tokenizer, key, component)
    assert type(getattr(tokenizer, key)) is cls
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


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
def test_sequence_json(wiki_spec, key, sequence, written):
    assert json.loads(sequence.__getstate__()) == json.loads(written)
    assert type(pickle.loads(pickle.dumps(sequence))) is type(sequence)
    tokenizer = wiki_spec
    setattr(tokenizer, key, sequence)
    assert type(getattr(tokenizer, key)) is type(sequence)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


# transformers' Mistral regex fix replaces the first pre-tokenizer in place.
def test_pre_tokenizer_assignment_writes_through(wiki_spec):
    tokenizer = wiki_spec
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Whitespace(), pre_tokenizers.Digits()])
    tokenizer.pre_tokenizer[0] = pre_tokenizers.Punctuation()
    written = json.loads(tokenizer.to_str())["pre_tokenizer"]["pretokenizers"]
    assert [p["type"] for p in written] == ["Punctuation", "Digits"]
    assert type(tokenizer.pre_tokenizer[0]) is pre_tokenizers.Punctuation
    assert not hasattr(tokenizer.pre_tokenizer, "add_prefix_space")
