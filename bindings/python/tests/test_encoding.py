import gc

import numpy as np

from tokenizers import Padding


def test_buffer_list_equivalence(bert):
    encoding = bert.encode("Hello there", padding=Padding(length=8, pad_id=3))

    assert encoding.ids_buffer.tolist() == encoding.ids == [1, 27462, 7495, 2, 3, 3, 3, 3]
    assert encoding.type_ids_buffer.tolist() == encoding.type_ids
    assert encoding.attention_mask_buffer.tolist() == encoding.attention_mask == [1, 1, 1, 1, 0, 0, 0, 0]
    for field in (encoding.ids_buffer, encoding.type_ids_buffer, encoding.attention_mask_buffer):
        assert np.asarray(field).dtype == np.uint32
        assert field.shape == (8,)


def test_buffer_copy(bert):
    encoding = bert.encode("Hello there")

    assert not np.shares_memory(encoding.ids_buffer, encoding.ids_buffer)


def test_buffer_write_leaves_encoding_unchanged(bert):
    encoding = bert.encode("Hello there")
    ids = encoding.ids_buffer

    ids[0] = 0

    assert ids.tolist() == [0, 27462, 7495, 2]
    assert encoding.ids == encoding.ids_buffer.tolist() == [1, 27462, 7495, 2]


def test_buffer_outlives_encoding(bert):
    ids = bert.encode("Hello there").ids_buffer

    gc.collect()

    assert ids.tolist() == [1, 27462, 7495, 2]


def test_empty(gpt2):
    encoding = gpt2.encode("")

    assert encoding.ids_buffer.shape == (0,)
    assert np.asarray(encoding.ids_buffer).dtype == np.uint32


def test_equality_ignores_which_fields_were_read(bert):
    read, unread = bert.encode("Hello there"), bert.encode("Hello there")
    read.type_ids, read.attention_mask

    assert read == unread


def test_equality_compares_tokens(bert):
    assert bert.encode("Hello there") != bert.encode("General Kenobi")
