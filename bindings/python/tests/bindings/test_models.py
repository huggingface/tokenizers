import pytest
import pickle

from ..utils import data_dir, roberta_files, bert_files

from tokenizers.models import Model, BPE, WordPiece, WordLevel


class TestBPE:
    def test_instantiate(self, roberta_files):
        assert isinstance(BPE(), Model)
        assert isinstance(BPE(), BPE)
        assert isinstance(BPE(roberta_files["vocab"], roberta_files["merges"]), Model)
        with pytest.raises(ValueError, match="`vocab` and `merges` must be both specified"):
            BPE(vocab=roberta_files["vocab"])
            BPE(merges=roberta_files["merges"])
        assert isinstance(
            pickle.loads(pickle.dumps(BPE(roberta_files["vocab"], roberta_files["merges"]))), BPE
        )


class TestWordPiece:
    def test_instantiate(self, bert_files):
        assert isinstance(WordPiece(), Model)
        assert isinstance(WordPiece(), WordPiece)
        assert isinstance(WordPiece(bert_files["vocab"]), Model)
        assert isinstance(pickle.loads(pickle.dumps(WordPiece(bert_files["vocab"]))), WordPiece)


class TestWordLevel:
    def test_instantiate(self, roberta_files):
        assert isinstance(WordLevel(), Model)
        assert isinstance(WordLevel(), WordLevel)
        # The WordLevel model expects a vocab.json using the same format as roberta
        # so we can just try to load with this file
        assert isinstance(WordLevel(roberta_files["vocab"]), Model)
        assert isinstance(WordLevel(roberta_files["vocab"]), WordLevel)
