import copy
import gc
import pickle

import numpy as np
import pytest

from tokenizers import Padding, Truncation


def test_array_list_equivalence(bert):
    bert.padding = Padding(length=8, pad_id=3)
    encoding = bert.encode("Hello there")

    assert encoding.ids_array.tolist() == encoding.ids == [1, 27462, 7495, 2, 3, 3, 3, 3]
    assert encoding.type_ids_array.tolist() == encoding.type_ids
    assert encoding.attention_mask_array.tolist() == encoding.attention_mask == [1, 1, 1, 1, 0, 0, 0, 0]
    for field in (encoding.ids_array, encoding.type_ids_array, encoding.attention_mask_array):
        assert field.dtype == np.uint32
        assert field.shape == (8,)


def test_array_copy(bert):
    encoding = bert.encode("Hello there")

    assert not np.shares_memory(encoding.ids_array, encoding.ids_array)
    assert encoding.ids_array.base is None


def test_array_write_leaves_encoding_unchanged(bert):
    encoding = bert.encode("Hello there")
    ids = encoding.ids_array

    ids[0] = 0

    assert ids.tolist() == [0, 27462, 7495, 2]
    assert encoding.ids == encoding.ids_array.tolist() == [1, 27462, 7495, 2]


def test_array_outlives_encoding(bert):
    ids = bert.encode("Hello there").ids_array

    gc.collect()

    assert ids.tolist() == [1, 27462, 7495, 2]


def test_empty(gpt2):
    encoding = gpt2.encode("")

    assert encoding.ids_array.shape == (0,)
    assert encoding.ids_array.dtype == np.uint32


def test_equality_ignores_which_fields_were_read(bert):
    read, unread = bert.encode("Hello there"), bert.encode("Hello there")
    read.type_ids, read.attention_mask

    assert read == unread


def test_equality_compares_tokens(bert):
    assert bert.encode("Hello there") != bert.encode("General Kenobi")


@pytest.mark.parametrize("add_special_tokens", [False, True])
def test_special_tokens_mask_tracks_origin(bert, add_special_tokens):
    encoding = bert.encode("Hello [SEP] there", add_special_tokens=add_special_tokens)
    expected = [1, 0, 0, 0, 1] if add_special_tokens else [0, 0, 0]

    assert encoding.special_tokens_mask == expected
    array = encoding.special_tokens_mask_array
    assert array.tolist() == expected
    assert array.dtype == np.uint32
    array[:] = 0
    assert encoding.special_tokens_mask == expected


@pytest.mark.parametrize("direction", ["left", "right"])
def test_special_tokens_mask_after_pair_truncation_and_padding(bert, direction):
    encoding = bert.encode(
        ("Hello [SEP] there", "Hello there"),
        truncation=Truncation(7, direction=direction),
        padding=Padding(length=9, direction=direction, pad_id=3),
    )
    expected = [1, 0, 0, 1, 0, 0, 1]
    expected = [1, 1] + expected if direction == "left" else expected + [1, 1]

    assert encoding.special_tokens_mask == expected
    for restored in [pickle.loads(pickle.dumps(encoding)), copy.copy(encoding), copy.deepcopy(encoding)]:
        assert restored == encoding
        assert restored.special_tokens_mask == expected


def test_special_tokens_mask_without_post_processor(gpt2):
    assert gpt2.encode("").special_tokens_mask == []
    assert gpt2.encode("Hello<|endoftext|>world").special_tokens_mask == [0, 0, 0]
    short, long = gpt2.encode_batch(["", "Hello there"], padding=Padding(pad_id=50256))
    assert short.special_tokens_mask == [1, 1]
    assert long.special_tokens_mask == [0, 0]


def test_special_tokens_mask_distinguishes_equal_ids(bert):
    inserted = bert.encode("Hello")
    literal = bert.encode("[CLS] Hello [SEP]", add_special_tokens=False)

    assert inserted.ids == literal.ids
    assert inserted.type_ids == literal.type_ids
    assert inserted.attention_mask == literal.attention_mask
    assert inserted.special_tokens_mask == [1, 0, 1]
    assert literal.special_tokens_mask == [0, 0, 0]
    assert inserted != literal
