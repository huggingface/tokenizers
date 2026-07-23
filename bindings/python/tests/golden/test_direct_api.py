"""Layer 1: the `tokenizers` API driven directly.

The library's bread and butter: load a pretrained tokenizer, turn raw strings
into padded/truncated id batches a model can consume, then map ids back to
text (decode) and tokens back to source positions (offsets).
"""

import pytest

from tokenizers import Tokenizer


@pytest.fixture
def bert():
    # Function-scoped: enable_truncation/enable_padding mutate the tokenizer,
    # so every test gets a fresh instance (loaded from the local Hub cache).
    return Tokenizer.from_pretrained("bert-base-uncased")


def test_encode_wraps_with_special_tokens(bert):
    enc = bert.encode("Hello world")

    assert enc.tokens == ["[CLS]", "hello", "world", "[SEP]"]
    assert enc.ids == [101, 7592, 2088, 102]
    assert enc.type_ids == [0, 0, 0, 0]
    assert enc.attention_mask == [1, 1, 1, 1]
    assert enc.special_tokens_mask == [1, 0, 0, 1]


def test_question_context_pair(bert):
    # QA and reranking models take two sequences in one encoding, told apart
    # by type_ids; sequence_ids gives None on the template's special tokens.
    enc = bert.encode("Where is Paris?", "Paris is in France.")

    assert enc.tokens.count("[SEP]") == 2
    boundary = enc.tokens.index("[SEP]")
    assert all(t == 0 for t in enc.type_ids[: boundary + 1])
    assert all(t == 1 for t in enc.type_ids[boundary + 1 :])
    assert enc.sequence_ids[0] is None
    assert enc.sequence_ids[1] == 0
    assert enc.sequence_ids[-2] == 1
    assert enc.sequence_ids[-1] is None


def test_batch_padded_and_truncated_to_fixed_shape(bert):
    # Fixed-shape batches: long inputs truncated (keeping the special-token
    # template), short ones padded, attention_mask flagging real tokens.
    bert.enable_truncation(max_length=6)
    bert.enable_padding(length=6, pad_token="[PAD]", pad_id=0)

    short, long = bert.encode_batch(["One two", "one two three four five six seven eight"])

    assert short.tokens == ["[CLS]", "one", "two", "[SEP]", "[PAD]", "[PAD]"]
    assert short.attention_mask == [1, 1, 1, 1, 0, 0]
    assert len(long.ids) == 6
    assert long.tokens[0] == "[CLS]"
    assert long.tokens[-1] == "[SEP]"
    assert long.attention_mask == [1, 1, 1, 1, 1, 1]


def test_offsets_point_back_into_the_source(bert):
    # Offsets power everything that maps tokens to source positions, e.g.
    # highlighting an answer span. Special tokens have no source, WordPiece
    # continuations drop their "##" marker.
    text = "Tokenization highlights substrings."
    enc = bert.encode(text)

    for token, (start, end), special in zip(enc.tokens, enc.offsets, enc.special_tokens_mask):
        if not special:
            assert text[start:end].lower() == token.removeprefix("##")


def test_decode_round_trips(bert):
    enc = bert.encode("hello world")

    assert bert.decode(enc.ids, skip_special_tokens=True) == "hello world"


def test_byte_level_round_trip():
    # GPT-2's byte-level BPE is the other big archetype: no [UNK], any text
    # survives encode/decode byte for byte.
    gpt2 = Tokenizer.from_pretrained("openai-community/gpt2")
    text = "Byte-level BPE round-trips emoji 🤗 and accents é!"

    assert gpt2.encode("Hello world").ids == [15496, 995]
    assert gpt2.decode(gpt2.encode(text).ids) == text
