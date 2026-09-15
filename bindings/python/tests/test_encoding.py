import gc

import numpy as np
import pytest

from tokenizers import Encoding, Padding


def test_array_list_equivalence(bert):
    bert.padding = Padding(length=8, pad_id=3)
    encoding = bert.encode("Hello there")

    assert encoding.ids_array.tolist() == encoding.ids == [1, 27462, 7495, 2, 3, 3, 3, 3]
    assert encoding.type_ids_array.tolist() == encoding.type_ids
    assert encoding.attention_mask_array.tolist() == encoding.attention_mask == [1, 1, 1, 1, 0, 0, 0, 0]
    for field in (encoding.ids_array, encoding.type_ids_array, encoding.attention_mask_array):
        assert field.dtype == np.uint32
        assert field.shape == (8,)


def test_array_zero_copy(bert):
    encoding = bert.encode("Hello there")

    assert np.shares_memory(encoding.ids_array, encoding.ids_array)
    assert not np.shares_memory(encoding.ids_array, encoding.type_ids_array)
    assert encoding.ids_array.base is encoding


def test_array_read_only(bert):
    encoding = bert.encode("Hello there")

    for field in (encoding.ids_array, encoding.type_ids_array, encoding.attention_mask_array):
        assert not field.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        encoding.ids_array[0] = 0


def test_array_keep_encoding_alive(bert):
    ids = bert.encode("Hello there").ids_array

    gc.collect()

    assert isinstance(ids.base, Encoding)
    assert ids.tolist() == [1, 27462, 7495, 2]


def test_empty(gpt2):
    encoding = gpt2.encode("")

    assert encoding.ids_array.shape == (0,)
    assert encoding.ids_array.dtype == np.uint32
