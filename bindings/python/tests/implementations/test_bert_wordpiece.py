import pytest
from tokenizers import BertWordPieceTokenizer

from ..utils import bert_files, data_dir, multiprocessing_with_parallelism


class TestBertWordPieceTokenizer:
    @pytest.mark.parametrize("wordpieces_prefix", ["##", "@@"])
    def test_wordpieces_prefix(self, wordpieces_prefix):
        vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "h": 3, f"{wordpieces_prefix}ello": 4}
        tokenizer = BertWordPieceTokenizer(vocab, wordpieces_prefix=wordpieces_prefix)

        output = tokenizer.encode("hello", add_special_tokens=False)
        assert output.tokens == ["h", f"{wordpieces_prefix}ello"]
        assert output.ids == [3, 4]
        assert output.offsets == [(0, 1), (1, 5)]
        assert tokenizer.decode(output.ids) == "hello"

    @pytest.mark.parametrize("wordpieces_prefix", ["##", "@@"])
    def test_train_save_reload_wordpieces_prefix(self, wordpieces_prefix, tmp_path):
        tokenizer = BertWordPieceTokenizer(wordpieces_prefix=wordpieces_prefix)
        tokenizer.train_from_iterator(
            ["hello", "help", "hello", "help"],
            vocab_size=11,
            min_frequency=2,
            wordpieces_prefix=wordpieces_prefix,
            show_progress=False,
        )
        output = tokenizer.encode("hello", add_special_tokens=False)
        assert len(output.tokens) > 1
        assert all(token.startswith(wordpieces_prefix) for token in output.tokens[1:])

        vocab_path = tokenizer.save_model(str(tmp_path))[0]
        reloaded = BertWordPieceTokenizer.from_file(vocab_path, wordpieces_prefix=wordpieces_prefix)
        reloaded_output = reloaded.encode("hello", add_special_tokens=False)
        assert reloaded_output.tokens == output.tokens
        assert reloaded_output.ids == output.ids
        assert reloaded.decode(reloaded_output.ids) == "hello"

    @pytest.mark.network
    def test_basic_encode(self, bert_files):
        tokenizer = BertWordPieceTokenizer.from_file(bert_files["vocab"])

        # Encode with special tokens by default
        output = tokenizer.encode("My name is John", "pair")
        assert output.ids == [101, 2026, 2171, 2003, 2198, 102, 3940, 102]
        assert output.tokens == [
            "[CLS]",
            "my",
            "name",
            "is",
            "john",
            "[SEP]",
            "pair",
            "[SEP]",
        ]
        assert output.offsets == [
            (0, 0),
            (0, 2),
            (3, 7),
            (8, 10),
            (11, 15),
            (0, 0),
            (0, 4),
            (0, 0),
        ]
        assert output.type_ids == [0, 0, 0, 0, 0, 0, 1, 1]

        # Can encode without the special tokens
        output = tokenizer.encode("My name is John", "pair", add_special_tokens=False)
        assert output.ids == [2026, 2171, 2003, 2198, 3940]
        assert output.tokens == ["my", "name", "is", "john", "pair"]
        assert output.offsets == [(0, 2), (3, 7), (8, 10), (11, 15), (0, 4)]
        assert output.type_ids == [0, 0, 0, 0, 1]

    @pytest.mark.network
    def test_multiprocessing_with_parallelism(self, bert_files):
        tokenizer = BertWordPieceTokenizer.from_file(bert_files["vocab"])
        multiprocessing_with_parallelism(tokenizer, False)
        multiprocessing_with_parallelism(tokenizer, True)

    def test_train_from_iterator(self):
        text = ["A first sentence", "Another sentence", "And a last one"]
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train_from_iterator(text, show_progress=False)

        output = tokenizer.encode("A sentence")
        assert output.tokens == ["a", "sentence"]
