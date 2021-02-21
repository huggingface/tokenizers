import pytest

from ..utils import data_dir, roberta_files, multiprocessing_with_parallelism
from tokenizers import ByteLevelBPETokenizer


class TestByteLevelBPE:
    def test_basic_encode(self, roberta_files):
        tokenizer = ByteLevelBPETokenizer.from_file(roberta_files["vocab"], roberta_files["merges"])
        output = tokenizer.encode("The quick brown fox jumps over the lazy dog")

        assert output.ids == [133, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]
        assert output.tokens == [
            "The",
            "Ġquick",
            "Ġbrown",
            "Ġfox",
            "Ġjumps",
            "Ġover",
            "Ġthe",
            "Ġlazy",
            "Ġdog",
        ]
        assert output.offsets == [
            (0, 3),
            (3, 9),
            (9, 15),
            (15, 19),
            (19, 25),
            (25, 30),
            (30, 34),
            (34, 39),
            (39, 43),
        ]

    def test_add_prefix_space(self, roberta_files):
        tokenizer = ByteLevelBPETokenizer.from_file(
            roberta_files["vocab"], roberta_files["merges"], add_prefix_space=True
        )
        output = tokenizer.encode("The quick brown fox jumps over the lazy dog")

        assert output.ids == [20, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]
        assert output.tokens == [
            "ĠThe",
            "Ġquick",
            "Ġbrown",
            "Ġfox",
            "Ġjumps",
            "Ġover",
            "Ġthe",
            "Ġlazy",
            "Ġdog",
        ]
        assert output.offsets == [
            (0, 3),
            (3, 9),
            (9, 15),
            (15, 19),
            (19, 25),
            (25, 30),
            (30, 34),
            (34, 39),
            (39, 43),
        ]

    def test_lowerspace(self, roberta_files):
        tokenizer = ByteLevelBPETokenizer.from_file(
            roberta_files["vocab"],
            roberta_files["merges"],
            add_prefix_space=True,
            lowercase=True,
        )
        output = tokenizer.encode("The Quick Brown Fox Jumps Over The Lazy Dog")

        assert output.ids == [5, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335]
        assert output.tokens == [
            "Ġthe",
            "Ġquick",
            "Ġbrown",
            "Ġfox",
            "Ġjumps",
            "Ġover",
            "Ġthe",
            "Ġlazy",
            "Ġdog",
        ]

    def test_multiprocessing_with_parallelism(self, roberta_files):
        tokenizer = ByteLevelBPETokenizer.from_file(roberta_files["vocab"], roberta_files["merges"])
        multiprocessing_with_parallelism(tokenizer, False)
        multiprocessing_with_parallelism(tokenizer, True)
