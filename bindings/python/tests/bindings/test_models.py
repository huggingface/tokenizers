import pickle

import pytest

from tokenizers.models import BPE, Model, WordLevel, WordPiece
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
