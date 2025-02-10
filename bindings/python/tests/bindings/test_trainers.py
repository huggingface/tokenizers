import copy
import os
import pickle

import pytest

from tokenizers import (
    AddedToken,
    SentencePieceUnigramTokenizer,
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)

from ..utils import data_dir, train_files


class TestBpeTrainer:
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
            AddedToken("1", special=True),
            AddedToken("2", special=True),
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

    def test_can_pickle(self):
        assert (
            trainers.BpeTrainer(min_frequency=12).__getstate__()
            == b"""{"BpeTrainer":{"min_frequency":12,"vocab_size":30000,"show_progress":true,"special_tokens":[],"limit_alphabet":null,"initial_alphabet":[],"continuing_subword_prefix":null,"end_of_word_suffix":null,"max_token_length":null,"words":{}}}"""
        )
        assert isinstance(pickle.loads(pickle.dumps(trainers.BpeTrainer(min_frequency=12))), trainers.BpeTrainer)

        assert isinstance(copy.deepcopy(trainers.BpeTrainer(min_frequency=12)), trainers.BpeTrainer)
        # Make sure everything is correct
        assert pickle.dumps(pickle.loads(pickle.dumps(trainers.BpeTrainer(min_frequency=12)))) == pickle.dumps(
            trainers.BpeTrainer(min_frequency=12)
        )


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
            AddedToken("1", special=True),
            AddedToken("2", special=True),
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

    def test_can_pickle(self):
        assert isinstance(pickle.loads(pickle.dumps(trainers.WordPieceTrainer())), trainers.WordPieceTrainer)


class TestWordLevelTrainer:
    def test_can_modify(self):
        trainer = trainers.WordLevelTrainer(
            vocab_size=12345, min_frequency=12, show_progress=False, special_tokens=["1", "2"]
        )

        assert trainer.vocab_size == 12345
        assert trainer.min_frequency == 12
        assert trainer.show_progress == False
        assert trainer.special_tokens == [
            AddedToken("1", special=True),
            AddedToken("2", special=True),
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

    def test_can_pickle(self):
        assert isinstance(pickle.loads(pickle.dumps(trainers.WordLevelTrainer())), trainers.WordLevelTrainer)


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

    def test_can_pickle(self):
        assert isinstance(pickle.loads(pickle.dumps(trainers.UnigramTrainer())), trainers.UnigramTrainer)

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

        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            show_progress=False,
            special_tokens=["[PAD]", "[SEP]", "[CLS]"],
            unk_token="[UNK]",
            vocab_size=100,
        )
        tokenizer.train([filename], trainer=trainer)

        assert tokenizer.get_vocab_size() == 100

        tokenizer = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(
            show_progress=False,
            special_tokens=["[PAD]", "[SEP]", "[CLS]", "[UNK]"],
            unk_token="[UNK]",
            vocab_size=100,
        )
        tokenizer.train([filename], trainer=trainer)

        assert tokenizer.get_vocab_size() == 100

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
            AddedToken("1", normalized=False, special=True),
            AddedToken("2", lstrip=True, normalized=False, special=True),
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

    def test_continuing_prefix_trainer_mismatch(self):
        UNK = "[UNK]"
        special_tokens = [UNK]
        tokenizer = Tokenizer(models.BPE(unk_token=UNK, continuing_subword_prefix="##"))
        trainer = trainers.BpeTrainer(special_tokens=special_tokens)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=True)]
        )
        tokenizer.train(files=["data/big.txt"], trainer=trainer)

        tokenizer.save("data/tokenizer.json")

        tokenizer.from_file("data/tokenizer.json")
