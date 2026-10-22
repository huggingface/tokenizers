import base64
import json
import pickle

import pytest

from conftest import ALBERT
from tokenizers import Regex, normalizers

TEXT = "Hello world, how are you?"


# The JSON 0.23 writes for each, key order included.
@pytest.mark.parametrize(
    ("normalizer", "written"),
    [
        (normalizers.Replace("a", "b"), '{"type":"Replace","pattern":{"String":"a"},"content":"b"}'),
        (
            normalizers.Replace(pattern=Regex(r"\s+"), content=" "),
            '{"type":"Replace","pattern":{"Regex":"\\\\s+"},"content":" "}',
        ),
        (
            normalizers.BertNormalizer(),
            '{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,'
            '"lowercase":true}',
        ),
        (
            normalizers.BertNormalizer(
                clean_text=False, handle_chinese_chars=False, strip_accents=True, lowercase=False
            ),
            '{"type":"BertNormalizer","clean_text":false,"handle_chinese_chars":false,"strip_accents":true,'
            '"lowercase":false}',
        ),
        (normalizers.Strip(), '{"type":"Strip","strip_left":true,"strip_right":true}'),
        (normalizers.Strip(left=False), '{"type":"Strip","strip_left":false,"strip_right":true}'),
        (normalizers.Prepend("▁"), '{"type":"Prepend","prepend":"▁"}'),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, normalizers.Normalizer) else None,
)
def test_normalizer_json(wiki_spec, normalizer, written):
    assert normalizer.__getstate__().decode() == written
    twin = pickle.loads(pickle.dumps(normalizer))
    assert type(twin) is type(normalizer)
    tokenizer = wiki_spec
    tokenizer.normalizer = normalizer
    assert type(tokenizer.normalizer) is type(normalizer)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


# T5 and ALBERT ship their SentencePiece charsmap as base64; transformers decodes it and passes bytes.
def test_precompiled_writes_the_charsmap_as_base64():
    declared = json.loads(ALBERT.read_text())["normalizer"]["normalizers"][-1]
    charsmap = base64.b64decode(declared["precompiled_charsmap"])
    precompiled = normalizers.Precompiled(charsmap)
    assert json.loads(precompiled.__getstate__()) == declared


# spacy-curated-transformers reads it.
def test_bert_normalizer_strip_accents():
    assert normalizers.BertNormalizer().strip_accents is None
    assert normalizers.BertNormalizer(strip_accents=False).strip_accents is False


def test_regex_refuses_a_pattern_that_does_not_compile():
    with pytest.raises(ValueError):
        Regex("(")


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
            "normalizer",
            normalizers.Sequence([normalizers.NFC(), normalizers.Lowercase()]),
            '{"type":"Sequence","normalizers":[{"type":"NFC"},{"type":"Lowercase"}]}',
        ),
    ],
    ids=["normalizers"],
)
def test_sequence_json(wiki_spec, key, sequence, written):
    assert sequence.__getstate__().decode() == written
    assert type(pickle.loads(pickle.dumps(sequence))) is type(sequence)
    tokenizer = wiki_spec
    setattr(tokenizer, key, sequence)
    assert type(getattr(tokenizer, key)) is type(sequence)
    compiled = tokenizer.compile()
    compiled.decode(compiled.encode(TEXT).ids)


# sentence-transformers and vLLM iterate a tokenizer's normalizer and rebuild it.
def test_normalizer_sequence_iterates_its_components():
    sequence = normalizers.Sequence([normalizers.NFC(), normalizers.Strip(left=False)])
    assert [type(n) for n in sequence] == [normalizers.NFC, normalizers.Strip]
    assert len(sequence) == 2
    assert not normalizers.Sequence([])
    with pytest.raises(IndexError):
        sequence[2]


def test_sequence_refuses_another_kind(gpt2_spec):
    with pytest.raises(TypeError):
        normalizers.Sequence([gpt2_spec.pre_tokenizer])
