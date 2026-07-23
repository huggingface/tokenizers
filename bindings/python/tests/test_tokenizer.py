import pickle

import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer, models

from .conftest import SENTENCES, train_word_tokenizer


def test_encode_returns_uint32_array(word_tokenizer):
    ids = word_tokenizer.encode_ids(SENTENCES[0], add_special_tokens=False)
    assert isinstance(ids, np.ndarray)
    assert ids.dtype == np.uint32
    words = [word_tokenizer.id_to_token(int(i)) for i in ids]
    assert words == SENTENCES[0].split()


def test_encode_batch_ids_matches_encode_ids(word_tokenizer):
    batch = word_tokenizer.encode_batch_ids(SENTENCES, add_special_tokens=False)
    assert len(batch) == len(SENTENCES)
    for line, ids in zip(SENTENCES, batch):
        single = word_tokenizer.encode_ids(line, add_special_tokens=False)
        assert np.array_equal(ids, single)


def test_unknown_words_map_to_unk(word_tokenizer):
    ids = word_tokenizer.encode_ids("supercalifragilistic", add_special_tokens=False)
    assert [word_tokenizer.id_to_token(int(i)) for i in ids] == ["[UNK]"]


def test_vocab_and_lookups(word_tokenizer):
    vocab = word_tokenizer.get_vocab()
    assert len(vocab) == word_tokenizer.get_vocab_size()
    token, id_ = next(iter(vocab.items()))
    assert word_tokenizer.token_to_id(token) == id_
    assert word_tokenizer.id_to_token(id_) == token
    assert word_tokenizer.token_to_id("definitely-not-in-vocab") is None


def test_add_tokens_and_encode_them():
    tok = train_word_tokenizer()
    assert tok.add_tokens(["procrastination"]) == 1
    assert tok.add_tokens(["procrastination"]) == 0
    ids = tok.encode_ids("the procrastination", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in ids] == ["the", "procrastination"]


def test_add_special_tokens_marks_special():
    tok = train_word_tokenizer()
    assert tok.add_special_tokens(["<eos>"]) == 1
    ids = tok.encode_ids("the <eos>", add_special_tokens=False)
    assert [tok.id_to_token(int(i)) for i in ids] == ["the", "<eos>"]


def test_add_tokens_accepts_added_token():
    tok = train_word_tokenizer()
    assert tok.add_tokens([AddedToken("<pipe>", single_word=True)]) == 1
    assert tok.token_to_id("<pipe>") is not None


def test_save_and_from_file_round_trip(word_tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    word_tokenizer.save(path)
    reloaded = Tokenizer.from_file(path)
    for line in SENTENCES[:4]:
        assert np.array_equal(
            reloaded.encode_ids(line, add_special_tokens=False),
            word_tokenizer.encode_ids(line, add_special_tokens=False),
        )


def test_to_str_and_from_buffer_round_trip(word_tokenizer):
    reloaded = Tokenizer.from_buffer(word_tokenizer.to_str().encode())
    assert np.array_equal(
        reloaded.encode_ids(SENTENCES[0], add_special_tokens=False),
        word_tokenizer.encode_ids(SENTENCES[0], add_special_tokens=False),
    )


def test_pickle_round_trip(word_tokenizer):
    reloaded = pickle.loads(pickle.dumps(word_tokenizer))
    assert np.array_equal(
        reloaded.encode_ids(SENTENCES[0], add_special_tokens=False),
        word_tokenizer.encode_ids(SENTENCES[0], add_special_tokens=False),
    )


def test_repr_names_model(word_tokenizer):
    assert "WordLevel" in repr(word_tokenizer)


def test_decode_raises():
    tok = Tokenizer(models.BPE())
    with pytest.raises(NotImplementedError):
        tok.decode([1, 2, 3])


def test_model_getter_returns_typed_copy(word_tokenizer):
    assert isinstance(word_tokenizer.model, models.WordLevel)


def test_train_iterator_error_propagates():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))

    def broken():
        yield "fine"
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        tok.train_from_iterator(broken())


def test_train_iterator_rejects_non_str():
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    with pytest.raises(TypeError):
        tok.train_from_iterator([b"bytes are not str"])
