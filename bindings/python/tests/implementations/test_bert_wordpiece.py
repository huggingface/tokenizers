import pytest

from ..utils import data_dir, bert_files, multiprocessing_with_parallelism
from tokenizers import BertWordPieceTokenizer


class TestBertWordPieceBPE:
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

    def test_multiprocessing_with_parallelism(self, bert_files):
        tokenizer = BertWordPieceTokenizer.from_file(bert_files["vocab"])
        multiprocessing_with_parallelism(tokenizer, False)
        multiprocessing_with_parallelism(tokenizer, True)
