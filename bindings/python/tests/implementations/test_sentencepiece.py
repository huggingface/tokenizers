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
    @pytest.mark.parametrize("unk_id, expected_ids", [(0, [1, 0]), (1, [0, 1])])
    def test_init_with_vocab_and_unk_id(self, unk_id, expected_ids):
        vocab = [("a", -1.0)]
        vocab.insert(unk_id, ("<unk>", 0.0))

        with pytest.raises(Exception, match="Encountered an unknown token but `unk_id` is missing"):
            SentencePieceUnigramTokenizer(vocab, "_", False).encode("ab")

        tokenizer = SentencePieceUnigramTokenizer(vocab, "_", False, unk_id=unk_id)
        output = tokenizer.encode("ab")
        assert output.tokens == ["a", "b"]
        assert output.ids == expected_ids

        with pytest.raises(ValueError, match="`vocab` and `unk_id` must be both specified"):
            SentencePieceUnigramTokenizer(unk_id=unk_id)

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
        tokenizer.train(files=str(p), show_progress=False, special_tokens=["<unk>"], unk_token="<unk>")
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
