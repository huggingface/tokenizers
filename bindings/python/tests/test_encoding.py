import gc
import pickle

import numpy as np

from tokenizers import Encoding, Padding, Truncation


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


def test_overflowing_defaults_to_empty(bert):
    assert bert.encode("Hello there").overflowing == []


def test_overflowing_rows_carry_the_specials(bert):
    # "Hello there my friend" is four tokens, so `max_length` 4 leaves room for two and windows.
    bert.truncation = Truncation(4, stride=1)
    encoding = bert.encode("Hello there my friend")
    rows = [encoding.ids] + [window.ids for window in encoding.overflowing]

    assert len(rows) > 1
    # Every row is the same `[CLS] window [SEP]` arrangement as the encoding itself.
    assert all(row[0] == 1 and row[-1] == 2 for row in rows)
    assert all(len(row) <= 4 for row in rows)


def test_overflowing_rows_carry_the_type_ids(bert):
    a, b = "Hello there", "my friend how are you today my dear friend"
    bert.truncation = Truncation(10, strategy="only_second", stride=2)
    untruncated = bert.encode((a, b), truncation=None)
    encoding = bert.encode((a, b))
    rows = [encoding, *encoding.overflowing]

    assert len(rows) > 1
    for row in rows:
        # `[CLS] A [SEP]` carries type id 0 and is identical in every row; the B window and its
        # trailing `[SEP]` carry type id 1.
        split = row.type_ids.index(1)
        assert row.ids[:split] == untruncated.ids[:split]
        assert row.type_ids == [0] * split + [1] * (len(row) - split)


def test_overflowing_equality_compares_rows(bert):
    bert.truncation = Truncation(4, stride=1)

    assert bert.encode("Hello there my friend") == bert.encode("Hello there my friend")
    assert bert.encode("Hello there my friend") != bert.encode("General Kenobi my friend")


def test_overflowing_survives_pickle(bert):
    bert.truncation = Truncation(4, stride=1)
    encoding = bert.encode("Hello there my friend")

    back = pickle.loads(pickle.dumps(encoding))

    assert back == encoding
    assert [window.ids for window in back.overflowing] == [window.ids for window in encoding.overflowing]


# A pickle written before `overflowing` existed reduces to the old three arguments, so it has to
# keep loading with no overflowing encodings.
def test_old_encoding_pickle_still_loads(bert):
    encoding = bert.encode("Hello there")
    rebuilt = Encoding._unpickle(encoding.ids, encoding.type_ids, encoding.attention_mask)

    assert rebuilt.overflowing == []
    assert rebuilt == encoding
