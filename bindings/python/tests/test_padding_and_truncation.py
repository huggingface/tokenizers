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
