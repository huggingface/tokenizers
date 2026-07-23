import numpy as np
import pytest

from tokenizers import Encoding, EncodingBatch, Tokenizer, models, pre_tokenizers, trainers

from .conftest import SENTENCES, train_word_tokenizer


@pytest.fixture(scope="module")
def special_tokenizer():
    """A BPE tokenizer whose vocabulary contains a special token (`<s>`), so
    tests can feed that token in the input text."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(
        ["hello world foo bar baz", "the quick brown fox"] * 50,
        trainer=trainers.BpeTrainer(vocab_size=120, special_tokens=["<unk>", "<s>"], show_progress=False),
    )
    return tok


def test_encode_returns_encoding(word_tokenizer):
    enc = word_tokenizer.encode(SENTENCES[0], add_special_tokens=False)
    assert isinstance(enc, Encoding)
    assert len(enc) == len(SENTENCES[0].split())
    assert repr(enc) == f"Encoding(length={len(enc)})"


def test_ids_match_encode_ids(word_tokenizer):
    enc = word_tokenizer.encode(SENTENCES[0], add_special_tokens=False)
    ids = word_tokenizer.encode_ids(SENTENCES[0], add_special_tokens=False)
    assert enc.ids == ids.tolist()
    assert np.array_equal(enc.ids_array(), ids)
    assert enc.ids_array().dtype == np.uint32


def test_ids_is_a_list_not_an_array(word_tokenizer):
    # transformers' _pad does `input_ids + [pad] * n`; a numpy array would
    # broadcast-add the padding into the token ids instead of concatenating.
    enc = word_tokenizer.encode(SENTENCES[0], add_special_tokens=False)
    assert isinstance(enc.ids, list)
    assert enc.ids + [0, 0] == list(enc.ids) + [0, 0]


def test_tokens(word_tokenizer):
    enc = word_tokenizer.encode(SENTENCES[0], add_special_tokens=False)
    assert enc.tokens == SENTENCES[0].split()


def test_constant_metadata_fields_for_a_single_sequence(special_tokenizer):
    # type_ids and attention_mask are constant for one unpadded sequence.
    enc = special_tokenizer.encode("hello <s> world", add_special_tokens=False)
    n = len(enc)
    assert "<s>" in enc.tokens
    assert enc.type_ids == [0] * n
    assert enc.attention_mask == [1] * n
    assert enc.n_sequences == 1


@pytest.mark.parametrize(
    "access",
    [
        lambda e: e.special_tokens_mask,
        lambda e: e.sequence_ids,
        lambda e: e.token_to_sequence(0),
        lambda e: e.word_ids,
        lambda e: e.offsets,
        lambda e: e.char_to_token(0),
        lambda e: e.char_to_word(0),
        lambda e: e.token_to_chars(0),
        lambda e: e.token_to_word(0),
        lambda e: e.word_to_tokens(0),
        lambda e: e.word_to_chars(0),
    ],
)
def test_unavailable_features_raise(word_tokenizer, access):
    enc = word_tokenizer.encode(SENTENCES[0], add_special_tokens=False)
    with pytest.raises(NotImplementedError):
        access(enc)


def test_encode_batch_returns_encoding_batch(word_tokenizer):
    batch = word_tokenizer.encode_batch(SENTENCES, add_special_tokens=False)
    assert isinstance(batch, EncodingBatch)
    assert len(batch) == len(SENTENCES)
    assert isinstance(batch[0], Encoding)


def test_batch_rows_match_single_encode(word_tokenizer):
    batch = word_tokenizer.encode_batch(SENTENCES, add_special_tokens=False)
    for i, line in enumerate(SENTENCES):
        assert batch[i].ids == word_tokenizer.encode(line, add_special_tokens=False).ids


def test_batch_matches_encode_batch_ids(word_tokenizer):
    batch = word_tokenizer.encode_batch(SENTENCES, add_special_tokens=False)
    ids = word_tokenizer.encode_batch_ids(SENTENCES, add_special_tokens=False)
    assert [batch[i].ids for i in range(len(batch))] == [row.tolist() for row in ids]


def test_batch_indexing_and_iteration(word_tokenizer):
    batch = word_tokenizer.encode_batch(SENTENCES, add_special_tokens=False)
    assert batch[-1].ids == batch[len(batch) - 1].ids
    assert len(list(batch)) == len(SENTENCES)
    with pytest.raises(IndexError):
        _ = batch[len(batch)]


def test_encode_and_encode_ids_do_the_same_work():
    tok = train_word_tokenizer()
    for line in SENTENCES:
        assert (
            tok.encode(line, add_special_tokens=False).ids == tok.encode_ids(line, add_special_tokens=False).tolist()
        )
