import os
import pytest


from tokenizers import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer


class TestSentencePieceBPE:
    def test_train_from_iterator(self):
        text = ["A first sentence", "Another sentence", "And a last one"]
        tokenizer = SentencePieceBPETokenizer()
        tokenizer.train_from_iterator(text, show_progress=False)

        output = tokenizer.encode("A sentence")
        assert output.tokens == ["▁A", "▁sentence"]


class TestSentencePieceUnigram:
    def test_train(self, tmpdir):
        p = tmpdir.mkdir("tmpdir").join("file.txt")
        p.write("A first sentence\nAnother sentence\nAnd a last one")

        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train(files=str(p), show_progress=False)

        output = tokenizer.encode("A sentence")
        assert output.tokens == ["▁A", "▁", "s", "en", "t", "en", "c", "e"]

        with pytest.raises(Exception) as excinfo:
            _ = tokenizer.encode("A sentence 🤗")
        assert str(excinfo.value) == "Encountered an unknown token but `unk_id` is missing"

    def test_train_with_unk_token(self, tmpdir):
        p = tmpdir.mkdir("tmpdir").join("file.txt")
        p.write("A first sentence\nAnother sentence\nAnd a last one")

        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train(
            files=str(p), show_progress=False, special_tokens=["<unk>"], unk_token="<unk>"
        )
        output = tokenizer.encode("A sentence 🤗")
        assert output.ids[-1] == 0
        assert output.tokens == ["▁A", "▁", "s", "en", "t", "en", "c", "e", "▁", "🤗"]

    def test_train_from_iterator(self):
        text = ["A first sentence", "Another sentence", "And a last one"]
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train_from_iterator(text, show_progress=False)

        output = tokenizer.encode("A sentence")
        assert output.tokens == ["▁A", "▁", "s", "en", "t", "en", "c", "e"]

        with pytest.raises(Exception) as excinfo:
            _ = tokenizer.encode("A sentence 🤗")
        assert str(excinfo.value) == "Encountered an unknown token but `unk_id` is missing"

    def test_train_from_iterator_with_unk_token(self):
        text = ["A first sentence", "Another sentence", "And a last one"]
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train_from_iterator(
            text, vocab_size=100, show_progress=False, special_tokens=["<unk>"], unk_token="<unk>"
        )
        output = tokenizer.encode("A sentence 🤗")
        assert output.ids[-1] == 0
        assert output.tokens == ["▁A", "▁", "s", "en", "t", "en", "c", "e", "▁", "🤗"]
