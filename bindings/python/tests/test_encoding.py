import gc

import numpy as np

from tokenizers import Padding


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
