import pickle

import pytest

from tokenizers import Padding, Truncation

LONG = "Hello there, how are you today? I am fine, thank you very much."
QUESTION = "Who is fine?"


def test_stride_defaults_to_zero():
    assert Truncation(4).stride == 0


def test_stride_survives_pickle():
    truncation = Truncation(4, strategy="only_second", direction="left", stride=2)

    assert pickle.loads(pickle.dumps(truncation)).stride == 2


def test_first_window_is_what_encode_returns(bert):
    bert.truncation = Truncation(6, stride=2)

    windows = bert.encode_windows(LONG)

    assert len(windows) > 1
    assert windows[0] == bert.encode(LONG)


def test_encode_windows_takes_the_encode_options(bert):
    windows = bert.encode_windows(LONG, truncation=Truncation(6, stride=2), padding=Padding(length=8))

    assert [len(window) for window in windows] == [8] * len(windows)


def test_pair_windows_cut_the_sequence_the_strategy_names(bert):
    truncation = Truncation(12, strategy="only_second", stride=2)

    windows = bert.encode_windows((QUESTION, LONG), truncation=truncation)

    assert len(windows) > 1
    assert windows[0] == bert.encode((QUESTION, LONG), truncation=truncation)


def test_encode_batch_windows_gives_one_list_per_input_in_order(bert):
    truncation = Truncation(6, stride=2)

    rows = bert.encode_batch_windows([LONG, "Hello"], truncation=truncation)

    assert rows == [
        bert.encode_windows(LONG, truncation=truncation),
        bert.encode_windows("Hello", truncation=truncation),
    ]


def test_windows_error_is_a_value_error(bert):
    with pytest.raises(ValueError, match="`only_first` or `only_second`"):
        bert.encode_windows((QUESTION, LONG), truncation=Truncation(12, stride=2))


def test_input_that_fits_is_one_window_whatever_the_strategy(bert):
    pair = bert.encode_windows(("Hello", "there"), truncation=Truncation(12))
    single = bert.encode_windows("Hello", truncation=Truncation(12, strategy="only_second"))

    assert pair == [bert.encode(("Hello", "there"))]
    assert single == [bert.encode("Hello")]


def test_stride_too_large_is_a_value_error(bert):
    with pytest.raises(ValueError, match="stride 4 must be smaller"):
        bert.encode_windows(LONG, truncation=Truncation(6, stride=4))
