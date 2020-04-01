from ..utils import data_dir, roberta_files, bert_files

from tokenizers.models import Model, BPE, WordPiece, WordLevel


class TestBPE:
    def test_instantiate(self, roberta_files):
        assert isinstance(BPE.empty(), Model)
        assert isinstance(BPE.from_files(roberta_files["vocab"], roberta_files["merges"]), Model)


class TestWordPiece:
    def test_instantiate(self, bert_files):
        assert isinstance(WordPiece.empty(), Model)
        assert isinstance(WordPiece.from_files(bert_files["vocab"]), Model)


class TestWordLevel:
    def test_instantiate(self, roberta_files):
        assert isinstance(WordLevel.empty(), Model)
        # The WordLevel model expects a vocab.json using the same format as roberta
        # so we can just try to load with this file
        assert isinstance(WordLevel.from_files(roberta_files["vocab"]), Model)
