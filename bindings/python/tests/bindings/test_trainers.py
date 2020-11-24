import os
import pytest
import pickle

from tokenizers import (
    SentencePieceUnigramTokenizer,
    AddedToken,
    models,
    pre_tokenizers,
    normalizers,
    Tokenizer,
    trainers,
)
from ..utils import data_dir, train_files


class TestBPETrainer:
    def test_can_modify(self):
        trainer = trainers.BpeTrainer(
            vocab_size=12345,
            min_frequency=12,
            show_progress=False,
            special_tokens=["1", "2"],
            limit_alphabet=13,
            initial_alphabet=["a", "b", "c"],
            continuing_subword_prefix="pref",
            end_of_word_suffix="suf",
        )

        assert trainer.vocab_size == 12345
        assert trainer.min_frequency == 12
        assert trainer.show_progress == False
        assert trainer.special_tokens == [
            AddedToken("1"),
            AddedToken("2"),
        ]
        assert trainer.limit_alphabet == 13
        assert sorted(trainer.initial_alphabet) == ["a", "b", "c"]
        assert trainer.continuing_subword_prefix == "pref"
        assert trainer.end_of_word_suffix == "suf"

        # Modify these
        trainer.vocab_size = 20000
        assert trainer.vocab_size == 20000
        trainer.min_frequency = 1
        assert trainer.min_frequency == 1
        trainer.show_progress = True
        assert trainer.show_progress == True
        trainer.special_tokens = []
        assert trainer.special_tokens == []
        trainer.limit_alphabet = None
        assert trainer.limit_alphabet == None
        trainer.initial_alphabet = ["d", "z"]
        assert sorted(trainer.initial_alphabet) == ["d", "z"]
        trainer.continuing_subword_prefix = None
        assert trainer.continuing_subword_prefix == None
        trainer.end_of_word_suffix = None
        assert trainer.continuing_subword_prefix == None


class TestWordPieceTrainer:
    def test_can_modify(self):
        trainer = trainers.WordPieceTrainer(
            vocab_size=12345,
            min_frequency=12,
            show_progress=False,
            special_tokens=["1", "2"],
            limit_alphabet=13,
            initial_alphabet=["a", "b", "c"],
            continuing_subword_prefix="pref",
            end_of_word_suffix="suf",
        )

        assert trainer.vocab_size == 12345
        assert trainer.min_frequency == 12
        assert trainer.show_progress == False
        assert trainer.special_tokens == [
            AddedToken("1"),
            AddedToken("2"),
        ]
        assert trainer.limit_alphabet == 13
        assert sorted(trainer.initial_alphabet) == ["a", "b", "c"]
        assert trainer.continuing_subword_prefix == "pref"
        assert trainer.end_of_word_suffix == "suf"

        # Modify these
        trainer.vocab_size = 20000
        assert trainer.vocab_size == 20000
        trainer.min_frequency = 1
        assert trainer.min_frequency == 1
        trainer.show_progress = True
        assert trainer.show_progress == True
        trainer.special_tokens = []
        assert trainer.special_tokens == []
        trainer.limit_alphabet = None
        assert trainer.limit_alphabet == None
        trainer.initial_alphabet = ["d", "z"]
        assert sorted(trainer.initial_alphabet) == ["d", "z"]
        trainer.continuing_subword_prefix = None
        assert trainer.continuing_subword_prefix == None
        trainer.end_of_word_suffix = None
        assert trainer.continuing_subword_prefix == None


class TestWordLevelTrainer:
    def test_can_modify(self):
        trainer = trainers.WordLevelTrainer(
            vocab_size=12345, min_frequency=12, show_progress=False, special_tokens=["1", "2"]
        )

        assert trainer.vocab_size == 12345
        assert trainer.min_frequency == 12
        assert trainer.show_progress == False
        assert trainer.special_tokens == [
            AddedToken("1"),
            AddedToken("2"),
        ]

        # Modify these
        trainer.vocab_size = 20000
        assert trainer.vocab_size == 20000
        trainer.min_frequency = 1
        assert trainer.min_frequency == 1
        trainer.show_progress = True
        assert trainer.show_progress == True
        trainer.special_tokens = []
        assert trainer.special_tokens == []


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

    def test_cannot_train_different_model(self):
        tokenizer = Tokenizer(models.BPE())
        trainer = trainers.UnigramTrainer(show_progress=False)

        with pytest.raises(Exception, match="UnigramTrainer can only train a Unigram"):
            tokenizer.train([], trainer)

    def test_can_modify(self):
        trainer = trainers.UnigramTrainer(
            vocab_size=12345,
            show_progress=False,
            special_tokens=["1", AddedToken("2", lstrip=True)],
            initial_alphabet=["a", "b", "c"],
        )

        assert trainer.vocab_size == 12345
        assert trainer.show_progress == False
        assert trainer.special_tokens == [
            AddedToken("1", normalized=False),
            AddedToken("2", lstrip=True, normalized=False),
        ]
        assert sorted(trainer.initial_alphabet) == ["a", "b", "c"]

        # Modify these
        trainer.vocab_size = 20000
        assert trainer.vocab_size == 20000
        trainer.show_progress = True
        assert trainer.show_progress == True
        trainer.special_tokens = []
        assert trainer.special_tokens == []
        trainer.initial_alphabet = ["d", "z"]
        assert sorted(trainer.initial_alphabet) == ["d", "z"]
