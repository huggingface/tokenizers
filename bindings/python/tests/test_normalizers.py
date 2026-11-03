import base64
import json
import pickle

import pytest

from conftest import ALBERT
from tokenizers import Regex, normalizers

TEXT = "Hello world, how are you?"


# The JSON 0.23 writes for each, and 2.0's for MetaspaceNormalizer.
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
        (
            normalizers.MetaspaceNormalizer(),
            '{"type":"MetaspaceNormalizer","replacement":"▁","prepend":"always","drop_whitespace":false}',
        ),
        (
            normalizers.MetaspaceNormalizer(replacement="_", prepend="first", drop_whitespace=True),
            '{"type":"MetaspaceNormalizer","replacement":"_","prepend":"first","drop_whitespace":true}',
        ),
    ],
    ids=lambda param: type(param).__name__ if isinstance(param, normalizers.Normalizer) else None,
)
def test_normalizer_json(wiki_spec, normalizer, written):
    assert json.loads(normalizer.__getstate__()) == json.loads(written)
    twin = pickle.loads(pickle.dumps(normalizer))
    assert type(twin) is type(normalizer)
    wiki_spec.normalizer = normalizer
    assert type(wiki_spec.normalizer) is type(normalizer)
    compiled = wiki_spec.compile()
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
    "cls",
    [
        normalizers.Lowercase,
        normalizers.ByteLevel,
        normalizers.StripAccents,
        normalizers.NFC,
        normalizers.NFD,
        normalizers.NFKC,
        normalizers.NFKD,
        normalizers.Nmt,
    ],
    ids=lambda cls: cls.__name__,
)
def test_stateless_normalizer(wiki_spec, cls):
    normalizer = cls()
    assert json.loads(normalizer.__getstate__()) == {"type": cls.__name__}
    assert type(pickle.loads(pickle.dumps(normalizer))) is cls
    wiki_spec.normalizer = normalizer
    assert type(wiki_spec.normalizer) is cls
    compiled = wiki_spec.compile()
    compiled.decode(compiled.encode(TEXT).ids)
