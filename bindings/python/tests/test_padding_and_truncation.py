from tokenizers import Padding, Truncation

# Every id in this file comes from released tokenizers 0.23.1 on the same fixture.
LONG = "Hello there, how are you today?"
SHORT_AND_LONG = ["Hello", LONG]
EOT = 50256
GPT2_IDS = [15496, 612, 11, 703, 389, 345, 1909, 30]

# Truncating to 4 and then padding to 6 leaves the long text with 4 ids and 2 pads. Padding first
# would leave its 8 ids alone, and the cut would then give 4 ids and no pad.
TRUNCATION = Truncation(4)
PADDING = Padding(length=6, pad_id=EOT)
LONG_TRUNCATED_THEN_PADDED = GPT2_IDS[:4] + [EOT, EOT]
SHORT_PADDED = [15496] + [EOT] * 5


def test_configured(gpt2):
    gpt2.truncation = TRUNCATION
    gpt2.padding = PADDING

    short, long = gpt2.encode_batch(SHORT_AND_LONG)

    assert short.ids == SHORT_PADDED
    assert long.ids == LONG_TRUNCATED_THEN_PADDED
    assert long.attention_mask == [1, 1, 1, 1, 0, 0]


def test_overridden(gpt2):
    short, long = gpt2.encode_batch(SHORT_AND_LONG, truncation=TRUNCATION, padding=PADDING)

    assert short.ids == SHORT_PADDED
    assert long.ids == LONG_TRUNCATED_THEN_PADDED
    assert gpt2.truncation is None
    assert gpt2.padding is None


def test_truncation_override_keeps_configured_padding(gpt2):
    gpt2.padding = PADDING

    short, long = gpt2.encode_batch(SHORT_AND_LONG, truncation=TRUNCATION)

    assert short.ids == SHORT_PADDED
    assert long.ids == LONG_TRUNCATED_THEN_PADDED


def test_padding_override_keeps_configured_truncation(gpt2):
    gpt2.truncation = TRUNCATION

    short, long = gpt2.encode_batch(SHORT_AND_LONG, padding=PADDING)

    assert short.ids == SHORT_PADDED
    assert long.ids == LONG_TRUNCATED_THEN_PADDED


def test_padding_off_keeps_configured_truncation(gpt2):
    gpt2.truncation = TRUNCATION
    gpt2.padding = PADDING

    short, long = gpt2.encode_batch(SHORT_AND_LONG, padding=None)

    assert short.ids == [15496]
    assert long.ids == GPT2_IDS[:4]


def test_truncation_off_keeps_configured_padding(gpt2):
    gpt2.truncation = TRUNCATION
    gpt2.padding = PADDING

    short, long = gpt2.encode_batch(SHORT_AND_LONG, truncation=None)

    assert short.ids == SHORT_PADDED
    assert long.ids == GPT2_IDS


def test_both_off(gpt2):
    gpt2.truncation = TRUNCATION
    gpt2.padding = PADDING

    short, long = gpt2.encode_batch(SHORT_AND_LONG, truncation=None, padding=None)

    assert short.ids == [15496]
    assert long.ids == GPT2_IDS


def test_truncate_then_specials_then_pad(bert):
    bert.truncation = Truncation(5)
    bert.padding = Padding(length=8, pad_id=3, pad_type_id=1)

    encoding = bert.encode(LONG)

    assert encoding.ids == [1, 27462, 7495, 16, 2, 3, 3, 3]
    assert encoding.type_ids == [0, 0, 0, 0, 0, 1, 1, 1]
    assert encoding.attention_mask == [1, 1, 1, 1, 1, 0, 0, 0]
