import pickle

import numpy as np
import pytest

from tokenizers import Encoding, Padding

TEXTS = ["Hello there", "General Kenobi", "You are a bold one"]


def test_sequence_protocol(gpt2):
    batch = gpt2.encode_batch(TEXTS)

    assert len(batch) == 3
    assert isinstance(batch[0], Encoding)
    assert batch[-1].ids == batch[2].ids
    assert [e.ids for e in batch] == [gpt2.encode(t).ids for t in TEXTS]
    with pytest.raises(IndexError):
        batch[3]


def test_unpadded_is_ragged(gpt2):
    batch = gpt2.encode_batch(TEXTS)

    assert batch.stride is None
    with pytest.raises(ValueError, match="different lengths"):
        batch.ids_array


def test_padded_is_2d(gpt2):
    gpt2.padding = Padding(pad_id=50256)
    batch = gpt2.encode_batch(TEXTS)

    ids = batch.ids_array
    assert batch.stride == ids.shape[1]
    assert ids.shape == (3, batch.stride)
    assert ids.dtype == np.uint32
    assert ids.tolist() == [e.ids for e in batch]


def test_2d_is_a_view(gpt2):
    gpt2.padding = Padding(pad_id=50256)
    ids = gpt2.encode_batch(TEXTS).ids_array

    # The batch owns the buffer and is kept alive as the array's base, so no copy is made.
    assert type(ids.base).__name__ == "Batch"
    assert not ids.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        ids[0, 0] = 0


def test_attention_mask_2d(gpt2):
    gpt2.padding = Padding(pad_id=50256)
    batch = gpt2.encode_batch(TEXTS)

    mask = batch.attention_mask_array
    assert mask.shape == (3, batch.stride)
    assert mask.tolist() == [e.attention_mask for e in batch]
    # One masked-in token per real token, padding masked out.
    assert int(mask.sum()) == sum(len(gpt2.encode(t)) for t in TEXTS)


def test_offsets(gpt2):
    batch = gpt2.encode_batch(TEXTS)

    offsets = batch.offsets
    assert len(offsets) == len(batch) + 1
    assert offsets[0] == 0
    assert [int(offsets[i + 1] - offsets[i]) for i in range(len(batch))] == [len(e) for e in batch]


def test_ids_list_of_lists(gpt2):
    batch = gpt2.encode_batch(TEXTS)

    assert batch.ids == [e.ids for e in batch]


def test_equality_and_pickle(gpt2):
    batch = gpt2.encode_batch(TEXTS)

    assert batch == [gpt2.encode(t) for t in TEXTS]
    # A batch cannot carry its shared buffer across a pickle, so it travels as its rows.
    restored = pickle.loads(pickle.dumps(batch))
    assert [e.ids for e in restored] == [e.ids for e in batch]


def test_empty(gpt2):
    batch = gpt2.encode_batch([])

    assert len(batch) == 0
    assert batch == []
