import os
import pytest
import pickle

from tokenizers import (
    SentencePieceUnigramTokenizer,
    models,
    pre_tokenizers,
    normalizers,
    Tokenizer,
    trainers,
)
from ..utils import data_dir, train_files


class TestUnigram:
    def test_train(self, train_files):
        tokenizer = SentencePieceUnigramTokenizer()
        tokenizer.train(train_files["small"], show_progress=False)

        filename = "tests/data/unigram_trained.json"
        tokenizer.save(filename)
        os.remove(filename)

    def test_train_parallelism_with_custom_pretokenizer(self, train_files):
        class GoodCustomPretok:
            def split(self, n, normalized):
                #  Here we just test that we can return a List[NormalizedString], it
                # does not really make sense to return twice the same otherwise
                return [normalized, normalized]

            def pre_tokenize(self, pretok):
                pretok.split(self.split)

        custom = pre_tokenizers.PreTokenizer.custom(GoodCustomPretok())
        bpe_tokenizer = Tokenizer(models.BPE())
        bpe_tokenizer.normalizer = normalizers.Lowercase()
        bpe_tokenizer.pre_tokenizer = custom

        if "TOKENIZERS_PARALLELISM" in os.environ:
            del os.environ["TOKENIZERS_PARALLELISM"]

        trainer = trainers.BpeTrainer(special_tokens=["<unk>"], show_progress=False)
        bpe_tokenizer.train([train_files["small"]], trainer=trainer)

    def test_train_with_special_tokens(self):
        filename = "tests/data/dummy-unigram-special_tokens-train.txt"
        with open(filename, "w") as f:
            f.write(
                """
[CLS] The Zen of Python, by Tim Peters [SEP]
[CLS] Beautiful is better than ugly. [SEP]
[CLS] Explicit is better than implicit. [SEP]
[CLS] Simple is better than complex. [SEP]
[CLS] Complex is better than complicated. [SEP]
[CLS] Flat is better than nested. [SEP]
[CLS] Sparse is better than dense. [SEP]
[CLS] Readability counts. [SEP]
[CLS] Special cases aren't special enough to break the rules. [SEP]
[CLS] Although practicality beats purity. [SEP]
[CLS] Errors should never pass silently. [SEP]
[CLS] Unless explicitly silenced. [SEP]
[CLS] In the face of ambiguity, refuse the temptation to guess. [SEP]
[CLS] There should be one-- and preferably only one --obvious way to do it. [SEP]
[CLS] Although that way may not be obvious at first unless you're Dutch. [SEP]
[CLS] Now is better than never. [SEP]
[CLS] Although never is often better than *right* now. [SEP]
[CLS] If the implementation is hard to explain, it's a bad idea. [SEP]
[CLS] If the implementation is easy to explain, it may be a good idea. [SEP]
[CLS] Namespaces are one honking great idea -- let's do more of those! [SEP]
            """
            )

        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            show_progress=False, special_tokens=["[PAD]", "[SEP]", "[CLS]"], unk_token="[UNK]"
        )

        tokenizer.train([filename], trainer=trainer)

        assert tokenizer.encode("[CLS] This is a test [SEP]").tokens == [
            "[CLS]",
            " T",
            "h",
            "i",
            "s",
            " is ",
            "a",
            " ",
            "te",
            "s",
            "t ",
            "[SEP]",
        ]
