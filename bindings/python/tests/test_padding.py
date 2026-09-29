import pytest

from tokenizers import Padding, Tokenizer

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
    assert gpt2.padding is None

    short, long = gpt2.encode_batch(SHORT_AND_LONG)
    assert len(short) < len(long)


def test_batch_longest(gpt2):
    gpt2.padding = Padding(pad_id=EOT)

    short, long = gpt2.encode_batch(SHORT_AND_LONG)

    assert len(short) == len(long)
    assert short.ids[-1] == EOT


def test_left(gpt2):
    gpt2.padding = Padding(direction="left", pad_id=EOT)

    short, _ = gpt2.encode_batch(SHORT_AND_LONG)

    assert short.ids[0] == EOT
    assert short.ids[-1] != EOT


def test_fixed_length(gpt2):
    gpt2.padding = Padding(length=4, pad_id=EOT)

    assert len(gpt2.encode("Hello")) == 4


def test_pad_to_multiple_of(gpt2):
    gpt2.padding = Padding(pad_to_multiple_of=8, pad_id=EOT)

    assert len(gpt2.encode("Hello")) == 8


def test_pad_type_id(gpt2):
    gpt2.padding = Padding(length=4, pad_id=EOT, pad_type_id=1)

    assert gpt2.encode("Hello").type_ids[-1] == 1


def test_padding_from_file(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    assert tokenizer.padding == Padding(pad_id=3, pad_token="[PAD]")
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)


def test_set_padding(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    tokenizer.padding = Padding(direction="left", pad_id=3)
    assert tokenizer.padding == Padding(direction="left", pad_id=3)
    assert tokenizer.encode_batch(SHORT_AND_LONG)[0].ids[0] == 3

    tokenizer.padding = None
    assert tokenizer.padding is None
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) < len(long)


def test_encode_padding_override(gpt2):
    assert len(gpt2.encode("Hello", padding=Padding(length=4, pad_id=EOT))) == 4
    assert gpt2.padding is None


def test_encode_padding_off(gpt2):
    unpadded = gpt2.encode("Hello").ids
    gpt2.padding = Padding(length=4, pad_id=EOT)

    assert gpt2.encode("Hello", padding=None).ids == unpadded
    assert len(gpt2.encode("Hello")) == 4


def test_encode_padding_override_replaces_config(gpt2):
    gpt2.padding = Padding(length=4, pad_id=EOT)

    assert len(gpt2.encode("Hello", padding=Padding(length=6, pad_id=EOT))) == 6
    assert len(gpt2.encode("Hello")) == 4


def test_encode_batch_padding_override(gpt2):
    short, long = gpt2.encode_batch(SHORT_AND_LONG, padding=Padding(pad_id=EOT))

    assert len(short) == len(long)
    assert gpt2.padding is None


def test_encode_batch_padding_off(gpt2):
    unpadded = [e.ids for e in gpt2.encode_batch(SHORT_AND_LONG)]
    gpt2.padding = Padding(pad_id=EOT)

    assert [e.ids for e in gpt2.encode_batch(SHORT_AND_LONG, padding=None)] == unpadded
    short, long = gpt2.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)
