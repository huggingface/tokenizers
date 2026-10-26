import pytest

from conftest import GPT2
from tokenizers import Padding, TokenizerSpec

SHORT_AND_LONG = ["Hello", "Hello there, how are you today?"]
EOT = 50256


def test_defaults():
    padding = Padding()

    assert padding.direction == "right"
    assert padding.pad_id == 0
    assert padding.pad_type_id == 0
    assert padding.pad_token == "[PAD]"
    assert padding.length is None
    assert padding.pad_to_multiple_of is None


def test_getters_return_the_arguments():
    padding = Padding(
        direction="left", pad_id=EOT, pad_type_id=1, pad_token="<|endoftext|>", length=4, pad_to_multiple_of=8
    )

    assert padding.direction == "left"
    assert padding.pad_id == EOT
    assert padding.pad_type_id == 1
    assert padding.pad_token == "<|endoftext|>"
    assert padding.length == 4
    assert padding.pad_to_multiple_of == 8


def test_unknown_direction():
    with pytest.raises(ValueError, match='direction must be "left" or "right"'):
        Padding(direction="sideways")  # ty: ignore[invalid-argument-type]


def test_no_padding(gpt2):
    short, long = gpt2.encode_batch(SHORT_AND_LONG)

    assert len(short) < len(long)


def test_batch_longest(gpt2):
    short, long = gpt2.encode_batch(SHORT_AND_LONG, padding=Padding(pad_id=EOT))

    assert len(short) == len(long)
    assert short.ids[-1] == EOT


def test_left(gpt2):
    short, _ = gpt2.encode_batch(SHORT_AND_LONG, padding=Padding(direction="left", pad_id=EOT))

    assert short.ids[0] == EOT
    assert short.ids[-1] != EOT


def test_fixed_length(gpt2):
    assert len(gpt2.encode("Hello", padding=Padding(length=4, pad_id=EOT))) == 4


def test_pad_to_multiple_of(gpt2):
    assert len(gpt2.encode("Hello", padding=Padding(pad_to_multiple_of=8, pad_id=EOT))) == 8


def test_pad_type_id(gpt2):
    assert gpt2.encode("Hello", padding=Padding(length=4, pad_id=EOT, pad_type_id=1)).type_ids[-1] == 1


def test_padding_from_file(padded_wiki):
    spec = TokenizerSpec.from_file(padded_wiki)

    assert spec.padding == Padding(pad_id=3, pad_token="[PAD]")
    short, long = spec.compile().encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)


def test_set_padding(padded_wiki):
    spec = TokenizerSpec.from_file(padded_wiki)

    spec.padding = Padding(direction="left", pad_id=3)
    assert spec.padding == Padding(direction="left", pad_id=3)
    assert spec.compile().encode_batch(SHORT_AND_LONG)[0].ids[0] == 3

    spec.padding = None
    assert spec.padding is None
    short, long = spec.compile().encode_batch(SHORT_AND_LONG)
    assert len(short) < len(long)


@pytest.fixture(scope="module")
def padded_gpt2():
    spec = TokenizerSpec.from_file(GPT2)
    spec.padding = Padding(length=4, pad_id=EOT)
    return spec.compile()


def test_encode_padding_off(gpt2, padded_gpt2):
    assert padded_gpt2.encode("Hello", padding=None).ids == gpt2.encode("Hello").ids
    assert len(padded_gpt2.encode("Hello")) == 4


def test_encode_padding_override(padded_gpt2):
    assert len(padded_gpt2.encode("Hello", padding=Padding(length=6, pad_id=EOT))) == 6
    assert len(padded_gpt2.encode("Hello")) == 4


def test_encode_batch_padding(gpt2):
    short, long = gpt2.encode_batch(SHORT_AND_LONG, padding=Padding(pad_id=EOT))

    assert len(short) == len(long)


def test_encode_batch_padding_off(gpt2, padded_gpt2):
    unpadded = [e.ids for e in gpt2.encode_batch(SHORT_AND_LONG)]

    assert [e.ids for e in padded_gpt2.encode_batch(SHORT_AND_LONG, padding=None)] == unpadded
    assert [len(e) for e in padded_gpt2.encode_batch(SHORT_AND_LONG)] == [4, 8]
