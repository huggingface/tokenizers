import base64
import json
import pickle

import pytest

from conftest import ALBERT
from tokenizers import Regex, decoders, normalizers, pre_tokenizers

TEXT = "Hello world, how are you? 1234"

# T5 and ALBERT ship their SentencePiece charsmap as base64; transformers decodes it and passes bytes.
ALBERT_PRECOMPILED = json.loads(ALBERT.read_text())["normalizer"]["normalizers"][-1]
CHARSMAP = base64.b64decode(ALBERT_PRECOMPILED["precompiled_charsmap"])


# The generator writes one template per field type, and one file per kind. One component per field
# type, and per kind, covers them all.
@pytest.mark.parametrize(
    ("component", "written"),
    [
        (normalizers.NFC(), {"type": "NFC"}),
        (normalizers.Strip(strip_left=False), {"type": "Strip", "strip_left": False, "strip_right": True}),
        (normalizers.Prepend("▁"), {"type": "Prepend", "prepend": "▁"}),
        (
            normalizers.BertNormalizer(),
            {
                "type": "BertNormalizer",
                "clean_text": True,
                "handle_chinese_chars": True,
                "strip_accents": None,
                "lowercase": True,
            },
        ),
        (normalizers.Replace("a", "b"), {"type": "Replace", "pattern": {"String": "a"}, "content": "b"}),
        (
            normalizers.Replace(Regex(r"\s+"), " "),
            {"type": "Replace", "pattern": {"Regex": r"\s+"}, "content": " "},
        ),
        (normalizers.Precompiled(CHARSMAP), ALBERT_PRECOMPILED),
        (pre_tokenizers.FixedLength(), {"type": "FixedLength", "length": 5}),
        (pre_tokenizers.Punctuation(), {"type": "Punctuation", "behavior": "Isolated"}),
        (decoders.Strip(), {"type": "Strip", "content": " ", "start": 0, "stop": 0}),
    ],
    ids=lambda param: None if isinstance(param, dict) else type(param).__name__,
)
def test_component_json(wiki_spec, component, written):
    assert json.loads(component.__getstate__()) == written
    assert type(pickle.loads(pickle.dumps(component))) is type(component)
    if isinstance(component, decoders.Decoder):
        field = "decoder"
    elif isinstance(component, pre_tokenizers.PreTokenizer):
        field = "pre_tokenizer"
    else:
        field = "normalizer"
    setattr(wiki_spec, field, component)
    assert type(getattr(wiki_spec, field)) is type(component)
    compiled = wiki_spec.compile()
    compiled.decode(compiled.encode(TEXT).ids)


# The getters that convert the JSON back: base64 to bytes, `{"Regex": ...}` to a Regex, `null` to
# None. spacy-curated-transformers reads `BertNormalizer.strip_accents`.
def test_getters_return_the_constructor_types():
    assert normalizers.Precompiled(CHARSMAP).precompiled_charsmap == CHARSMAP
    assert isinstance(normalizers.Replace(Regex(r"\s+"), " ").pattern, Regex)
    assert normalizers.Replace("a", "b").pattern == "a"
    assert normalizers.BertNormalizer().strip_accents is None


# A choice outside the JSON's strings fails at once, not at `compile()`.
def test_refuses_a_choice_tokenizer_json_does_not_take():
    with pytest.raises(ValueError, match="behavior"):
        pre_tokenizers.Punctuation("isolated")
    punctuation = pre_tokenizers.Punctuation()
    with pytest.raises(ValueError, match="behavior"):
        punctuation.behavior = "isolated"
    assert punctuation.behavior == "Isolated"
