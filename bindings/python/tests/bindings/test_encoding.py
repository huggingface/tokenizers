import pytest
from ..utils import data_dir, bert_files

from tokenizers import BertWordPieceTokenizer


class TestEncoding:
    @pytest.fixture(scope="class")
    def encodings(self, bert_files):
        tokenizer = BertWordPieceTokenizer.from_file(bert_files["vocab"])
        single_encoding = tokenizer.encode("I love HuggingFace")
        pair_encoding = tokenizer.encode("I love HuggingFace", "Do you?")
        return single_encoding, pair_encoding

    def test_sequence_ids(self, encodings):
        single, pair = encodings

        assert single.sequence_ids == [None, 0, 0, 0, 0, None]
        assert pair.sequence_ids == [None, 0, 0, 0, 0, None, 1, 1, 1, None]

    def test_n_sequences(self, encodings):
        single, pair = encodings
        assert single.n_sequences == 1
        assert pair.n_sequences == 2

    def test_word_to_tokens(self, encodings):
        single, pair = encodings

        assert single.tokens == ["[CLS]", "i", "love", "hugging", "##face", "[SEP]"]
        assert single.word_to_tokens(0) == (1, 2)

        assert pair.tokens == [
            "[CLS]",
            "i",
            "love",
            "hugging",
            "##face",
            "[SEP]",
            "do",
            "you",
            "?",
            "[SEP]",
        ]
        assert pair.word_to_tokens(0) == (1, 2)
        assert pair.word_to_tokens(0, 0) == (1, 2)
        assert pair.word_to_tokens(6, 0) == None
        assert pair.word_to_tokens(0, 1) == (6, 7)

    def test_word_to_chars(self, encodings):
        single, pair = encodings

        assert single.word_to_chars(2) == (7, 18)
        assert pair.word_to_chars(2) == (7, 18)
        assert pair.word_to_chars(2, 0) == (7, 18)
        assert pair.word_to_chars(2, 1) == (6, 7)

    def test_token_to_sequence(self, encodings):
        single, pair = encodings

        assert single.token_to_sequence(2) == 0
        assert pair.token_to_sequence(2) == 0
        assert pair.token_to_sequence(0) == None
        assert pair.token_to_sequence(5) == None
        assert pair.token_to_sequence(6) == 1
        assert pair.token_to_sequence(8) == 1
        assert pair.token_to_sequence(9) == None
        assert pair.token_to_sequence(1200) == None

    def test_token_to_chars(self, encodings):
        single, pair = encodings

        assert single.token_to_chars(0) == None
        assert single.token_to_chars(2) == (2, 6)
        assert pair.token_to_chars(2) == (2, 6)
        assert pair.token_to_chars(5) == None
        assert pair.token_to_chars(6) == (0, 2)

    def test_token_to_word(self, encodings):
        single, pair = encodings

        assert single.token_to_word(0) == None
        assert single.token_to_word(1) == 0
        assert single.token_to_word(4) == 2
        assert pair.token_to_word(1) == 0
        assert pair.token_to_word(4) == 2
        assert pair.token_to_word(5) == None
        assert pair.token_to_word(6) == 0
        assert pair.token_to_word(7) == 1

    def test_char_to_token(self, encodings):
        single, pair = encodings

        assert single.char_to_token(0) == 1
        assert pair.char_to_token(0) == 1
        assert pair.char_to_token(0, 0) == 1
        assert pair.char_to_token(1, 0) == None
        assert pair.char_to_token(0, 1) == 6
        assert pair.char_to_token(2, 1) == None

    def test_char_to_word(self, encodings):
        single, pair = encodings

        assert single.char_to_word(0) == 0
        assert single.char_to_word(1) == None
        assert pair.char_to_word(2) == 1
        assert pair.char_to_word(2, 0) == 1
        assert pair.char_to_word(2, 1) == None
        assert pair.char_to_word(3, 1) == 1
