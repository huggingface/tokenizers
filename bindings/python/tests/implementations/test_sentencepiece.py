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
    def test_train_from_iterator(self):
        text = ["A first sentence", "Another sentence", "And a last one"]
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train_from_iterator(text, show_progress=False)

        output = tokenizer.encode("A sentence")
        assert output.tokens == ["▁A", "▁", "s", "en", "t", "en", "c", "e"]
