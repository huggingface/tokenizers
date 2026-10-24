import numpy as np
import pytest

from tokenizers import Padding, Truncation

SHORT_AND_LONG = ["Hello", "Hello there, how are you today?"]
EOT = 50256
TRUNCATION = Truncation(4)
PADDING = Padding(length=6, pad_id=EOT)


def ids(batch):
    return [encoding.ids for encoding in batch]


def encode_configured(gpt2, truncation, padding):
    gpt2.truncation = truncation
    gpt2.padding = padding
    expected = ids(gpt2.encode_batch(SHORT_AND_LONG))
    gpt2.truncation = None
    gpt2.padding = None
    return expected


# Omitting a kwarg inherits the configured value, None switches it off, a value replaces it.
# The expected ids come from configuring the effective pair on the tokenizer itself.
@pytest.mark.parametrize(
    ("configured", "overrides", "effective"),
    [
        pytest.param(
            (None, None),
            {"truncation": TRUNCATION, "padding": PADDING},
            (TRUNCATION, PADDING),
            id="both_overridden",
        ),
        pytest.param(
            (None, PADDING),
            {"truncation": TRUNCATION},
            (TRUNCATION, PADDING),
            id="truncation_override_keeps_configured_padding",
        ),
        pytest.param(
            (TRUNCATION, None),
            {"padding": PADDING},
            (TRUNCATION, PADDING),
            id="padding_override_keeps_configured_truncation",
        ),
        pytest.param(
            (TRUNCATION, PADDING),
            {"padding": None},
            (TRUNCATION, None),
            id="padding_off_keeps_configured_truncation",
        ),
        pytest.param(
            (TRUNCATION, PADDING),
            {"truncation": None},
            (None, PADDING),
            id="truncation_off_keeps_configured_padding",
        ),
        pytest.param(
            (TRUNCATION, PADDING),
            {"truncation": None, "padding": None},
            (None, None),
            id="both_off",
        ),
    ],
)
def test_overrides_resolve_independently(gpt2, configured, overrides, effective):
    expected = encode_configured(gpt2, *effective)
    gpt2.truncation, gpt2.padding = configured

    assert ids(gpt2.encode_batch(SHORT_AND_LONG, **overrides)) == expected
    assert (gpt2.truncation, gpt2.padding) == configured


# An overflowing window is a row of its own and is padded together with the encoding it came
# from: rows stay rectangular, and the padding is the tail of each row.
def test_windows_pad_with_their_encoding(gpt2):
    gpt2.truncation = Truncation(4)
    gpt2.padding = Padding(length=6, pad_id=EOT)
    batch = gpt2.encode_batch(SHORT_AND_LONG)

    rows = [row for encoding in batch for row in [encoding, *encoding.overflowing]]
    assert len(rows) > len(batch)
    assert [len(row.ids) for row in rows] == [6] * len(rows)
    for row in rows:
        assert len(row.attention_mask) == 6
        assert row.attention_mask == [1] * row.attention_mask.count(1) + [0] * row.attention_mask.count(0)


# A truncation with no room at all leaves the sequence as one window longer than its encoding.
# With `BatchLongest` every row has to end up the same length, or the batch cannot become an
# array at all.
def test_windows_make_the_batch_rectangular(bert):
    bert.truncation = Truncation(2)
    bert.padding = Padding()
    batch = bert.encode_batch(["Hello there my friend"])

    rows = [row for encoding in batch for row in [encoding, *encoding.overflowing]]
    assert [len(row.ids) for row in rows] == [6, 6]
    for row in rows:
        assert len(row.type_ids) == len(row.ids)
        assert len(row.attention_mask) == len(row.ids)
    # numpy raises ValueError on ragged rows
    assert np.stack([row.ids_array for row in rows]).shape == (2, 6)
    assert np.stack([row.type_ids_array for row in rows]).shape == (2, 6)
    masks = np.stack([row.attention_mask_array for row in rows])
    assert masks.tolist() == [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
