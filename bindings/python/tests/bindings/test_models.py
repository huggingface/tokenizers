import pickle

import pytest

from tokenizers.models import BPE, Model, WordLevel, WordPiece, Unigram
from ..utils import bert_files, data_dir, roberta_files


class TestBPE:
    def test_can_modify(self):
        model = BPE(
            dropout=0.5,
            unk_token="[UNK]",
            continuing_subword_prefix="__prefix__",
            end_of_word_suffix="__suffix__",
            fuse_unk=False,
        )

        assert model.dropout == 0.5
        assert model.unk_token == "[UNK]"
        assert model.continuing_subword_prefix == "__prefix__"
        assert model.end_of_word_suffix == "__suffix__"
        assert model.fuse_unk == False
        assert model.byte_fallback == False

        # Modify these
        model.dropout = 0.1
        assert pytest.approx(model.dropout) == 0.1
        model.unk_token = "<unk>"
        assert model.unk_token == "<unk>"
        model.continuing_subword_prefix = None
        assert model.continuing_subword_prefix == None
        model.end_of_word_suffix = "suff"
        assert model.end_of_word_suffix == "suff"
        model.fuse_unk = True
        assert model.fuse_unk == True
        model.byte_fallback = True
        assert model.byte_fallback == True

    def test_dropout_zero(self):
        model = BPE(dropout=0.0)
        assert model.dropout == 0.0


class TestWordPiece:
    def test_instantiate(self, bert_files):
        assert isinstance(WordPiece(), Model)
        assert isinstance(WordPiece(), WordPiece)

        vocab = {"a": 0, "b": 1, "ab": 2}
        assert isinstance(WordPiece(vocab), Model)
        assert isinstance(WordPiece(vocab), WordPiece)
        assert isinstance(WordPiece.from_file(bert_files["vocab"]), WordPiece)
        assert isinstance(pickle.loads(pickle.dumps(WordPiece(vocab))), WordPiece)

        assert isinstance(WordPiece(bert_files["vocab"]), Model)
        assert isinstance(pickle.loads(pickle.dumps(WordPiece(bert_files["vocab"]))), WordPiece)

    def test_can_modify(self):
        model = WordPiece(
            unk_token="<oov>",
            continuing_subword_prefix="__prefix__",
            max_input_chars_per_word=200,
        )

        assert model.unk_token == "<oov>"
        assert model.continuing_subword_prefix == "__prefix__"
        assert model.max_input_chars_per_word == 200

        # Modify these
        model.unk_token = "<unk>"
        assert model.unk_token == "<unk>"
        model.continuing_subword_prefix = "$$$"
        assert model.continuing_subword_prefix == "$$$"
        model.max_input_chars_per_word = 10
        assert model.max_input_chars_per_word == 10


class TestWordLevel:
    def test_instantiate(self, roberta_files):
        assert isinstance(WordLevel(), Model)
        assert isinstance(WordLevel(), WordLevel)

        vocab = {"a": 0, "b": 1, "ab": 2}
        assert isinstance(WordLevel(vocab), Model)
        assert isinstance(WordLevel(vocab), WordLevel)
        assert isinstance(WordLevel.from_file(roberta_files["vocab"]), WordLevel)

        # The WordLevel model expects a vocab.json using the same format as roberta
        # so we can just try to load with this file
        assert isinstance(WordLevel(roberta_files["vocab"]), Model)
        assert isinstance(WordLevel(roberta_files["vocab"]), WordLevel)

    def test_can_modify(self):
        model = WordLevel(unk_token="<oov>")

        assert model.unk_token == "<oov>"

        # Modify these
        model.unk_token = "<unk>"
        assert model.unk_token == "<unk>"


class TestUnigram:
    def test_unk_token_property(self):
        # Create Unigram with vocab containing <unk>
        vocab = [
            ("<unk>", 0.0),
            ("hello", -1.0),
            ("world", -1.5),
        ]
        model = Unigram(vocab, unk_id=0)

        # Test unk_token getter returns str
        assert model.unk_token == "<unk>"
        assert isinstance(model.unk_token, str)

        # Test unk_id getter returns int
        assert model.unk_id == 0
        assert isinstance(model.unk_id, int)

        # Test unk_token setter - set to existing token
        model.unk_token = "hello"
        assert model.unk_token == "hello"
        assert model.unk_id == 1

        # Test unk_token setter - non-existent token should raise
        with pytest.raises(ValueError, match="not found in vocabulary"):
            model.unk_token = "nonexistent"

        # Test unk_token setter - None should raise
        with pytest.raises(ValueError, match="Cannot set unk_token to None"):
            model.unk_token = None

    def test_unk_token_without_unk_id(self):
        # Create Unigram without unk_id
        vocab = [
            ("hello", -1.0),
            ("world", -1.5),
        ]
        model = Unigram(vocab, unk_id=None)

        # unk_token should be None when unk_id is not set
        assert model.unk_token is None
        assert model.unk_id is None
