import pytest

from tokenizers import Tokenizer, Truncation

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


def test_no_truncation(gpt2):
    assert gpt2.truncation is None


def test_max_length(gpt2):
    gpt2.truncation = Truncation(4)

    assert len(gpt2.encode(LONG)) == 4


def test_left_keeps_the_tail(gpt2):
    full = gpt2.encode(LONG).ids
    gpt2.truncation = Truncation(4, direction="left")

    assert gpt2.encode(LONG).ids == full[-4:]


# Tokenizer.encode takes one sequence, so only_second never has the sequence it cuts.
def test_encode_error_is_a_value_error(gpt2):
    gpt2.truncation = Truncation(4, strategy="only_second")

    with pytest.raises(ValueError, match="Second sequence not provided"):
        gpt2.encode(LONG)


def test_truncation_from_file(truncated_wiki):
    tokenizer = Tokenizer.from_file(truncated_wiki)

    assert tokenizer.truncation == Truncation(4)
    assert len(tokenizer.encode(LONG)) == 4


def test_set_truncation(truncated_wiki, wiki):
    tokenizer = Tokenizer.from_file(truncated_wiki)

    tokenizer.truncation = Truncation(2)
    assert tokenizer.truncation == Truncation(2)
    assert len(tokenizer.encode(LONG)) == 2

    tokenizer.truncation = None
    assert tokenizer.truncation is None
    assert tokenizer.encode(LONG).ids == wiki.encode(LONG).ids


def test_encode_truncation_override(gpt2):
    assert len(gpt2.encode(LONG, truncation=Truncation(4))) == 4
    assert gpt2.truncation is None


def test_encode_truncation_off(gpt2):
    full = gpt2.encode(LONG).ids
    gpt2.truncation = Truncation(4)

    assert gpt2.encode(LONG, truncation=None).ids == full
    assert len(gpt2.encode(LONG)) == 4


def test_encode_truncation_override_replaces_config(gpt2):
    gpt2.truncation = Truncation(4)

    assert len(gpt2.encode(LONG, truncation=Truncation(2))) == 2
    assert len(gpt2.encode(LONG)) == 4


def test_encode_batch_truncation_override(gpt2):
    assert [len(e) for e in gpt2.encode_batch([LONG] * 2, truncation=Truncation(4))] == [4, 4]
    assert gpt2.truncation is None


def test_encode_batch_truncation_off(gpt2):
    full = gpt2.encode(LONG).ids
    gpt2.truncation = Truncation(4)

    assert [e.ids for e in gpt2.encode_batch([LONG] * 2, truncation=None)] == [full, full]
    assert [len(e) for e in gpt2.encode_batch([LONG] * 2)] == [4, 4]
