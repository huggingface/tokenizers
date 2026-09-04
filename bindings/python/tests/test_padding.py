from tokenizers import Padding, Tokenizer

SHORT_AND_LONG = ["Hello", "Hello there, how are you today?"]
EOT = 50256


def test_no_padding(gpt2):
    assert gpt2.padding is None
    assert [len(e) for e in gpt2.encode_batch(SHORT_AND_LONG)] == [1, 8]


def test_batch_longest(bert):
    bert.padding = Padding(pad_id=3)

    short, long = bert.encode_batch(SHORT_AND_LONG)

    assert long.ids == [1, 27462, 7495, 16, 7510, 7268, 7989, 9819, 35, 2]
    assert short.ids == [1, 27462, 2] + [3] * 7
    assert short.type_ids == [0] * 10
    assert short.attention_mask == [1] * 3 + [0] * 7
    assert long.attention_mask == [1] * 10


def test_left(gpt2):
    gpt2.padding = Padding(direction="left", pad_id=EOT, pad_token="<|endoftext|>")

    short, long = gpt2.encode_batch(SHORT_AND_LONG)

    assert short.ids == [EOT] * 7 + [15496]
    assert short.attention_mask == [0] * 7 + [1]
    assert long.ids == [15496, 612, 11, 703, 389, 345, 1909, 30]


def test_fixed_length(gpt2):
    gpt2.padding = Padding(length=4, pad_id=EOT)

    short, long = gpt2.encode_batch(SHORT_AND_LONG)

    assert short.ids == [15496] + [EOT] * 3
    assert short.attention_mask == [1] + [0] * 3
    assert long.ids == [15496, 612, 11, 703, 389, 345, 1909, 30]


def test_pad_to_multiple_of(gpt2):
    gpt2.padding = Padding(pad_to_multiple_of=8, pad_id=EOT)
    assert len(gpt2.encode("Hello")) == 8
    assert len(gpt2.encode("Hello " * 12)) == 16
    assert [len(e) for e in gpt2.encode_batch(SHORT_AND_LONG)] == [8, 8]

    gpt2.padding = Padding(length=5, pad_to_multiple_of=8, pad_id=EOT)
    assert len(gpt2.encode("Hello")) == 8


def test_pad_type_id(bert):
    bert.padding = Padding(length=8, pad_id=3, pad_type_id=1)

    encoding = bert.encode("Hello")

    assert encoding.ids == [1, 27462, 2, 3, 3, 3, 3, 3]
    assert encoding.type_ids == [0, 0, 0, 1, 1, 1, 1, 1]
    assert encoding.attention_mask == [1, 1, 1, 0, 0, 0, 0, 0]


def test_padding_from_file(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    assert tokenizer.padding == Padding(pad_id=3, pad_token="[PAD]")
    short, long = tokenizer.encode_batch(SHORT_AND_LONG)
    assert len(short) == len(long)
    assert short.ids[-1] == 3


def test_from_file_padding_override(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki, padding=Padding(direction="left", pad_id=3))

    assert tokenizer.padding == Padding(direction="left", pad_id=3)
    assert tokenizer.encode_batch(SHORT_AND_LONG)[0].ids[0] == 3


def test_set_padding(padded_wiki):
    tokenizer = Tokenizer.from_file(padded_wiki)

    tokenizer.padding = Padding(direction="left", pad_id=3)
    assert tokenizer.padding == Padding(direction="left", pad_id=3)
    assert tokenizer.encode_batch(SHORT_AND_LONG)[0].ids[0] == 3

    tokenizer.padding = None
    assert tokenizer.padding is None
    assert [len(e) for e in tokenizer.encode_batch(SHORT_AND_LONG)] == [1, 8]


def test_defaults():
    padding = Padding()

    assert padding.direction == "right"
    assert padding.pad_id == 0
    assert padding.pad_type_id == 0
    assert padding.pad_token == "[PAD]"
    assert padding.length is None
    assert padding.pad_to_multiple_of is None
