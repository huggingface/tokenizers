import os
import pytest
import pickle

from tokenizers import SentencePieceUnigramTokenizer
from ..utils import data_dir, train_files


class TestUnigram:
    def test_train(self, train_files):
        # TODO: This is super *slow* fix it before merging.
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train(train_files["wagahaiwa"], show_progress=False)

        filename = "tests/data/unigram_trained.json"
        tokenizer.save(filename)
        os.remove(filename)
