import os
import pytest
import pickle

from tokenizers import SentencePieceUnigramTokenizer


class TestUnigram:
    def test_train(self):
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train("tests/data/unigram_wagahaiwa_nekodearu.txt", show_progress=False)

        filename = "tests/data/unigram_trained.json"
        tokenizer.save(filename)
        os.remove(filename)
