import pytest

from conftest import GPT2
from tokenizers import TokenizerSpec, Truncation

LONG = "Hello there, how are you today?"


def test_defaults():
    truncation = Truncation(512)

    assert truncation.max_length == 512
    assert truncation.strategy == "longest_first"
    assert truncation.direction == "right"


@pytest.mark.parametrize("strategy", ["longest_first", "only_first", "only_second"])
def test_strategy_round_trips(strategy):
    assert Truncation(4, strategy=strategy).strategy == strategy


@pytest.mark.parametrize("direction", ["left", "right"])
def test_direction_round_trips(direction):
    assert Truncation(4, direction=direction).direction == direction


def test_unknown_strategy():
    with pytest.raises(ValueError, match='strategy must be "longest_first", "only_first" or "only_second"'):
        Truncation(4, strategy="sideways")  # ty: ignore[invalid-argument-type]


def test_unknown_direction():
    with pytest.raises(ValueError, match='direction must be "left" or "right"'):
        Truncation(4, direction="sideways")  # ty: ignore[invalid-argument-type]


def test_max_length_is_required():
    with pytest.raises(TypeError):
        Truncation()  # ty: ignore[missing-argument]


def test_max_length(gpt2):
    assert len(gpt2.encode(LONG, truncation=Truncation(4))) == 4


def test_left_keeps_the_tail(gpt2):
    full = gpt2.encode(LONG).ids

    assert gpt2.encode(LONG, truncation=Truncation(4, direction="left")).ids == full[-4:]


# CompiledTokenizer.encode takes one sequence, so only_second never has the sequence it cuts.
def test_encode_error_is_a_value_error(gpt2):
    with pytest.raises(ValueError, match="Second sequence not provided"):
        gpt2.encode(LONG, truncation=Truncation(4, strategy="only_second"))


def test_truncation_from_file(truncated_wiki):
    spec = TokenizerSpec.from_file(truncated_wiki)

    assert spec.truncation == Truncation(4)
    assert len(spec.compile().encode(LONG)) == 4


def test_set_truncation(truncated_wiki, wiki):
    spec = TokenizerSpec.from_file(truncated_wiki)

    spec.truncation = Truncation(2)
    assert spec.truncation == Truncation(2)
    assert len(spec.compile().encode(LONG)) == 2

    spec.truncation = None
    assert spec.truncation is None
    assert spec.compile().encode(LONG).ids == wiki.encode(LONG).ids


@pytest.fixture(scope="module")
def truncated_gpt2():
    spec = TokenizerSpec.from_file(GPT2)
    spec.truncation = Truncation(4)
    return spec.compile()


def test_encode_truncation_off(gpt2, truncated_gpt2):
    assert truncated_gpt2.encode(LONG, truncation=None).ids == gpt2.encode(LONG).ids
    assert len(truncated_gpt2.encode(LONG)) == 4


def test_encode_truncation_override(truncated_gpt2):
    assert len(truncated_gpt2.encode(LONG, truncation=Truncation(2))) == 2
    assert len(truncated_gpt2.encode(LONG)) == 4


def test_encode_batch_truncation(gpt2):
    assert [len(e) for e in gpt2.encode_batch([LONG] * 2, truncation=Truncation(4))] == [4, 4]


def test_encode_batch_truncation_off(gpt2, truncated_gpt2):
    full = gpt2.encode(LONG).ids

    assert [e.ids for e in truncated_gpt2.encode_batch([LONG] * 2, truncation=None)] == [full, full]
    assert [len(e) for e in truncated_gpt2.encode_batch([LONG] * 2)] == [4, 4]
