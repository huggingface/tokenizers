import base64
import inspect
import json
import pickle

import pytest

from conftest import ALBERT
from tokenizers import Regex, decoders, normalizers, pre_tokenizers

TEXT = "Hello world, how are you? 1234"

# T5 and ALBERT ship their SentencePiece charsmap as base64; transformers decodes it and passes bytes.
ALBERT_PRECOMPILED = json.loads(ALBERT.read_text())["normalizer"]["normalizers"][-1]
CHARSMAP = base64.b64decode(ALBERT_PRECOMPILED["precompiled_charsmap"])


COMPONENT_FIELD = {
    normalizers.Normalizer: "normalizer",
    pre_tokenizers.PreTokenizer: "pre_tokenizer",
    decoders.Decoder: "decoder",
}
COMPONENT_CLASSES = [
    component_class
    for module, kind in (
        (normalizers, normalizers.Normalizer),
        (pre_tokenizers, pre_tokenizers.PreTokenizer),
        (decoders, decoders.Decoder),
    )
    for component_class in vars(module).values()
    if isinstance(component_class, type) and issubclass(component_class, kind)
    if component_class not in (kind, module.Sequence)
]

# The fields whose type and default don't give a sample value the core accepts.
SAMPLES = {
    "precompiled_charsmap": CHARSMAP,
    "behavior": "merged_with_previous",
    "prepend": "never",
    "prepend_scheme": "never",
}


def constructor_defaults(component_class):
    # inspect.signature refuses a signature that isn't ASCII, such as Metaspace's "▁" default.
    namespace = {}
    exec(f"def new{component_class.__text_signature__}: pass", namespace)
    return {name: parameter.default for name, parameter in inspect.signature(namespace["new"]).parameters.items()}


def sample(name, default):
    if name in SAMPLES:
        return SAMPLES[name]
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    # Only an `OptionalBool` defaults to None.
    if default is None:
        return True
    return "x"


# Every class takes a value other than its default in each field, through the constructor and
# through the setter, gives it back from the getter, and writes JSON the core compiles.
@pytest.mark.parametrize("component_class", COMPONENT_CLASSES, ids=lambda cls: f"{cls.__module__}.{cls.__name__}")
def test_every_field_round_trips(wiki_spec, component_class):
    defaults = constructor_defaults(component_class)
    values = {name: sample(name, default) for name, default in defaults.items()}
    built = component_class(**values)
    assert {name: getattr(built, name) for name in values} == values

    required = {name: values[name] for name, default in defaults.items() if default is inspect.Parameter.empty}
    edited = component_class(**required)
    for name, value in values.items():
        setattr(edited, name, value)
    assert edited.__getstate__() == built.__getstate__()

    restored = pickle.loads(pickle.dumps(built))
    assert type(restored) is component_class
    assert restored.__getstate__() == built.__getstate__()

    field = next(field for kind, field in COMPONENT_FIELD.items() if isinstance(built, kind))
    setattr(wiki_spec, field, built)
    assert type(getattr(wiki_spec, field)) is component_class
    compiled = wiki_spec.compile()
    compiled.decode(compiled.encode(TEXT).ids)


# The getters that convert the JSON back: base64 to bytes, `{"Regex": ...}` to a Regex, `null` to
# None. spacy-curated-transformers reads `BertNormalizer.strip_accents`.
def test_getters_return_the_constructor_types():
    assert normalizers.Precompiled(CHARSMAP).precompiled_charsmap == CHARSMAP
    assert isinstance(normalizers.Replace(Regex(r"\s+"), " ").pattern, Regex)
    assert normalizers.Replace("a", "b").pattern == "a"
    assert normalizers.BertNormalizer().strip_accents is None


# Python spells a choice in snake_case and refuses the JSON's spelling, so a value has one
# spelling. A refused choice fails at once, not at `compile()`.
def test_refuses_a_choice_in_the_json_spelling():
    with pytest.raises(ValueError, match="behavior"):
        pre_tokenizers.Punctuation("Isolated")
    punctuation = pre_tokenizers.Punctuation()
    with pytest.raises(ValueError, match="behavior"):
        punctuation.behavior = "Isolated"
    assert punctuation.behavior == "isolated"
    assert json.loads(punctuation.__getstate__())["behavior"] == "Isolated"


# The core reads a missing field with a default. Files that predate Punctuation's `behavior` leave it out.
def test_a_missing_field_reads_as_its_default(wiki_spec):
    file = json.loads(wiki_spec.to_str())
    file["pre_tokenizer"] = {"type": "Punctuation"}
    loaded = type(wiki_spec).from_str(json.dumps(file))
    assert loaded.pre_tokenizer.behavior == "isolated"
