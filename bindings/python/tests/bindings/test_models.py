import pytest
import pickle

from ..utils import data_dir, roberta_files, bert_files

from tokenizers.models import Model, BPE, WordPiece, WordLevel


class TestBPE:
    def test_instantiate(self, roberta_files):
        assert isinstance(BPE(), Model)
        assert isinstance(BPE(), BPE)

        vocab = {"a": 0, "b": 1, "ab": 2}
        merges = {(0, 1): (0, 2)}
        assert isinstance(BPE(vocab, merges), Model)
        with pytest.raises(ValueError, match="`vocab` and `merges` must be both specified"):
            BPE(vocab=vocab)
            BPE(merges=merges)

        assert isinstance(
            pickle.loads(pickle.dumps(BPE(vocab, merges))),
            BPE,
        )

        # Deprecated calls in 0.9
        with pytest.deprecated_call():
            assert isinstance(BPE(roberta_files["vocab"], roberta_files["merges"]), Model)

        with pytest.raises(ValueError, match="`vocab` and `merges` must be both specified"):
            BPE(vocab=roberta_files["vocab"])
            BPE(merges=roberta_files["merges"])
        with pytest.deprecated_call():
            assert isinstance(
                pickle.loads(pickle.dumps(BPE(roberta_files["vocab"], roberta_files["merges"]))),
                BPE,
            )


class TestWordPiece:
    def test_instantiate(self, bert_files):
        assert isinstance(WordPiece(), Model)
        assert isinstance(WordPiece(), WordPiece)

        vocab = {"a": 0, "b": 1, "ab": 2}
        assert isinstance(WordPiece(vocab), Model)
        assert isinstance(WordPiece(vocab), WordPiece)
        assert isinstance(pickle.loads(pickle.dumps(WordPiece(vocab))), WordPiece)

        # Deprecated calls in 0.9
        with pytest.deprecated_call():
            assert isinstance(WordPiece(bert_files["vocab"]), Model)
        with pytest.deprecated_call():
            assert isinstance(pickle.loads(pickle.dumps(WordPiece(bert_files["vocab"]))), WordPiece)


class TestWordLevel:
    def test_instantiate(self, roberta_files):
        assert isinstance(WordLevel(), Model)
        assert isinstance(WordLevel(), WordLevel)

        vocab = {"a": 0, "b": 1, "ab": 2}
        assert isinstance(WordLevel(vocab), Model)
        assert isinstance(WordLevel(vocab), WordLevel)

        # The WordLevel model expects a vocab.json using the same format as roberta
        # so we can just try to load with this file
        with pytest.deprecated_call():
            assert isinstance(WordLevel(roberta_files["vocab"]), Model)
        with pytest.deprecated_call():
            assert isinstance(WordLevel(roberta_files["vocab"]), WordLevel)
