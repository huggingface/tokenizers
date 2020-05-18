import pickle

from ..utils import data_dir, roberta_files

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.processors import PostProcessor, BertProcessing, RobertaProcessing, ByteLevel


class TestBertProcessing:
    def test_instantiate(self):
        processor = BertProcessing(("[SEP]", 0), ("[CLS]", 1))
        assert processor is not None
        assert isinstance(processor, PostProcessor)
        assert isinstance(processor, BertProcessing)
        assert isinstance(
            pickle.loads(pickle.dumps(BertProcessing(("[SEP]", 0), ("[CLS]", 1)))), BertProcessing
        )

    def test_processing(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_special_tokens(["[SEP]", "[CLS]"])
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])
        tokenizer.post_processor = BertProcessing(("[SEP]", 0), ("[CLS]", 1))

        output = tokenizer.encode("my name", "pair")
        assert output.tokens == ["[CLS]", "my", "name", "[SEP]", "pair", "[SEP]"]
        assert output.ids == [1, 2, 3, 0, 6, 0]


class TestRobertaProcessing:
    def test_instantiate(self):
        processor = RobertaProcessing(("</s>", 1), ("<s>", 0))
        assert processor is not None
        assert isinstance(processor, PostProcessor)
        assert isinstance(processor, RobertaProcessing)
        assert isinstance(
            pickle.loads(pickle.dumps(RobertaProcessing(("</s>", 1), ("<s>", 0)))),
            RobertaProcessing,
        )

    def test_processing(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_special_tokens(["<s>", "</s>"])
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])
        tokenizer.post_processor = RobertaProcessing(("</s>", 1), ("<s>", 0))

        output = tokenizer.encode("my name", "pair")
        assert output.tokens == ["<s>", "my", "name", "</s>", "</s>", "pair", "</s>"]
        assert output.ids == [0, 2, 3, 1, 1, 6, 1]


class TestByteLevelProcessing:
    def test_instantiate(self):
        assert ByteLevel() is not None
        assert ByteLevel(trim_offsets=True) is not None
        assert isinstance(ByteLevel(), PostProcessor)
        assert isinstance(ByteLevel(), ByteLevel)
        assert isinstance(pickle.loads(pickle.dumps(ByteLevel())), ByteLevel)

    def test_processing(self, roberta_files):
        tokenizer = Tokenizer(BPE(roberta_files["vocab"], roberta_files["merges"]))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=True)

        # Keeps original offsets
        output = tokenizer.encode("My name is John")
        assert output.tokens == ["ĠMy", "Ġname", "Ġis", "ĠJohn"]
        assert output.offsets == [(0, 2), (2, 7), (7, 10), (10, 15)]

        # Trims offsets when activated
        tokenizer.post_processor = ByteLevel(trim_offsets=True)
        output = tokenizer.encode("My name is John")
        assert output.tokens == ["ĠMy", "Ġname", "Ġis", "ĠJohn"]
        assert output.offsets == [(0, 2), (3, 7), (8, 10), (11, 15)]
