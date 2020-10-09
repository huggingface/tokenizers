import os
import pytest
import pickle

from tokenizers import SentencePieceUnigramTokenizer
from ..utils import data_dir, train_files


class TestUnigram:
    def test_train(self, train_files):
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train(train_files["small"], show_progress=False)

        filename = "tests/data/unigram_trained.json"
        tokenizer.save(filename)
        os.remove(filename)
