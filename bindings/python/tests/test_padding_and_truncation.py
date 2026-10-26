import pytest

from tokenizers import Padding, Truncation

SHORT_AND_LONG = ["Hello", "Hello there, how are you today?"]
EOT = 50256
TRUNCATION = Truncation(4)
PADDING = Padding(length=6, pad_id=EOT)


def ids(batch):
    return [encoding.ids for encoding in batch]


def compile_with(spec, truncation, padding):
    spec.truncation = truncation
    spec.padding = padding
    return spec.compile()


# Omitting a kwarg inherits the configured value, None switches it off, a value replaces it.
# The expected ids come from compiling the effective pair into the tokenizer itself.
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
def test_overrides_resolve_independently(gpt2_spec, configured, overrides, effective):
    expected = ids(compile_with(gpt2_spec, *effective).encode_batch(SHORT_AND_LONG))
    configured_tokenizer = compile_with(gpt2_spec, *configured)

    assert ids(configured_tokenizer.encode_batch(SHORT_AND_LONG, **overrides)) == expected
