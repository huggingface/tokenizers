import pickle

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


def test_stride_defaults_to_zero():
    assert Truncation(512).stride == 0


def test_stride_round_trips():
    assert Truncation(4, stride=2).stride == 2
    assert Truncation(4, "only_second", "left", 3).stride == 3


def test_repr_includes_stride():
    assert repr(Truncation(4, "only_second", "left", 2)) == (
        'Truncation(max_length=4, strategy="only_second", direction="left", stride=2)'
    )


def test_equality_compares_stride():
    assert Truncation(4, stride=0) != Truncation(4, stride=1)
    assert Truncation(4, "only_first", "left", 2) == Truncation(4, "only_first", "left", 2)


def test_pickle_round_trips_stride():
    truncation = Truncation(4, "only_second", "left", 2)

    assert pickle.loads(pickle.dumps(truncation)) == truncation


# A pickle written before `stride` existed reduces to the old three constructor arguments, so it
# has to keep loading with `stride` defaulting to 0.
def test_old_pickle_defaults_stride_to_zero():
    old = pickle.dumps((Truncation, (4, "only_first", "left")))
    cls, args = pickle.loads(old)

    assert cls(*args) == Truncation(4, "only_first", "left")
    assert cls(*args).stride == 0


# Windows tile the input from the start, `max_length` long and overlapping the previous one by
# `stride` (`max_length - stride` new tokens each), and stop at the first window that reaches the
# end of the input. The main encoding is the first window.
def test_overflowing_windows_overlap_by_stride(gpt2):
    full = gpt2.encode(LONG).ids
    gpt2.truncation = Truncation(4, stride=1)
    encoding = gpt2.encode(LONG)
    rows = [encoding.ids] + [window.ids for window in encoding.overflowing]

    starts, start = [], 0
    while True:
        starts.append(start)
        if start + 4 >= len(full):
            break
        start += 4 - 1

    assert rows == [full[start : start + 4] for start in starts]


def test_no_overflowing_when_the_sequence_fits(gpt2):
    gpt2.truncation = Truncation(len(LONG) + 10, stride=3)

    assert gpt2.encode(LONG).overflowing == []
