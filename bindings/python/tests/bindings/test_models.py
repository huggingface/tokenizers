import pytest
import pickle

from ..utils import data_dir, roberta_files, bert_files

from tokenizers.models import Model, BPE, WordPiece, WordLevel


class TestBPE:
    def test_instantiate(self, roberta_files):
        assert isinstance(BPE(), Model)
        assert isinstance(BPE(), BPE)

        vocab = {"a": 0, "b": 1, "ab": 2}
        merges = [("a", "b")]
        assert isinstance(BPE(vocab, merges), Model)
        assert isinstance(BPE.from_file(roberta_files["vocab"], roberta_files["merges"]), BPE)
        with pytest.raises(ValueError, match="`vocab` and `merges` must be both specified"):
            BPE(vocab=vocab)
        with pytest.raises(ValueError, match="`vocab` and `merges` must be both specified"):
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
        with pytest.raises(ValueError, match="`vocab` and `merges` must be both specified"):
            BPE(merges=roberta_files["merges"])
        with pytest.deprecated_call():
            assert isinstance(
                pickle.loads(pickle.dumps(BPE(roberta_files["vocab"], roberta_files["merges"]))),
                BPE,
            )

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


class TestWordPiece:
    def test_instantiate(self, bert_files):
        assert isinstance(WordPiece(), Model)
        assert isinstance(WordPiece(), WordPiece)

        vocab = {"a": 0, "b": 1, "ab": 2}
        assert isinstance(WordPiece(vocab), Model)
        assert isinstance(WordPiece(vocab), WordPiece)
        assert isinstance(WordPiece.from_file(bert_files["vocab"]), WordPiece)
        assert isinstance(pickle.loads(pickle.dumps(WordPiece(vocab))), WordPiece)

        # Deprecated calls in 0.9
        with pytest.deprecated_call():
            assert isinstance(WordPiece(bert_files["vocab"]), Model)
        with pytest.deprecated_call():
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
        with pytest.deprecated_call():
            assert isinstance(WordLevel(roberta_files["vocab"]), Model)
        with pytest.deprecated_call():
            assert isinstance(WordLevel(roberta_files["vocab"]), WordLevel)

    def test_can_modify(self):
        model = WordLevel(unk_token="<oov>")

        assert model.unk_token == "<oov>"

        # Modify these
        model.unk_token = "<unk>"
        assert model.unk_token == "<unk>"
