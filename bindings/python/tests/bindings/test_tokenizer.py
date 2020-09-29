import numpy as np
import pickle
import pytest
from ..utils import (
    data_dir,
    roberta_files,
    bert_files,
    multiprocessing_with_parallelism,
)

from tokenizers import AddedToken, Tokenizer, Encoding
from tokenizers.models import Model, BPE, WordPiece
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import RobertaProcessing, BertProcessing
from tokenizers.normalizers import Lowercase
from tokenizers.implementations import BertWordPieceTokenizer


class TestAddedToken:
    def test_instantiate_with_content_only(self):
        added_token = AddedToken("<mask>")
        assert type(added_token) == AddedToken
        assert str(added_token) == "<mask>"
        assert (
            repr(added_token)
            == 'AddedToken("<mask>", rstrip=False, lstrip=False, single_word=False, normalized=True)'
        )
        assert added_token.rstrip == False
        assert added_token.lstrip == False
        assert added_token.single_word == False
        assert added_token.normalized == True
        assert isinstance(pickle.loads(pickle.dumps(added_token)), AddedToken)

    def test_can_set_rstrip(self):
        added_token = AddedToken("<mask>", rstrip=True)
        assert added_token.rstrip == True
        assert added_token.lstrip == False
        assert added_token.single_word == False
        assert added_token.normalized == True

    def test_can_set_lstrip(self):
        added_token = AddedToken("<mask>", lstrip=True)
        assert added_token.rstrip == False
        assert added_token.lstrip == True
        assert added_token.single_word == False
        assert added_token.normalized == True

    def test_can_set_single_world(self):
        added_token = AddedToken("<mask>", single_word=True)
        assert added_token.rstrip == False
        assert added_token.lstrip == False
        assert added_token.single_word == True
        assert added_token.normalized == True

    def test_can_set_normalized(self):
        added_token = AddedToken("<mask>", normalized=False)
        assert added_token.rstrip == False
        assert added_token.lstrip == False
        assert added_token.single_word == False
        assert added_token.normalized == False


class TestTokenizer:
    def test_has_expected_type_and_methods(self):
        tokenizer = Tokenizer(BPE())
        assert type(tokenizer) == Tokenizer
        assert callable(tokenizer.num_special_tokens_to_add)
        assert callable(tokenizer.get_vocab)
        assert callable(tokenizer.get_vocab_size)
        assert callable(tokenizer.enable_truncation)
        assert callable(tokenizer.no_truncation)
        assert callable(tokenizer.enable_padding)
        assert callable(tokenizer.no_padding)
        assert callable(tokenizer.encode)
        assert callable(tokenizer.encode_batch)
        assert callable(tokenizer.decode)
        assert callable(tokenizer.decode_batch)
        assert callable(tokenizer.token_to_id)
        assert callable(tokenizer.id_to_token)
        assert callable(tokenizer.add_tokens)
        assert callable(tokenizer.add_special_tokens)
        assert callable(tokenizer.train)
        assert callable(tokenizer.post_process)
        assert isinstance(tokenizer.model, Model)
        assert tokenizer.normalizer is None
        assert tokenizer.pre_tokenizer is None
        assert tokenizer.post_processor is None
        assert tokenizer.decoder is None
        assert isinstance(pickle.loads(pickle.dumps(Tokenizer(BPE()))), Tokenizer)

    def test_add_tokens(self):
        tokenizer = Tokenizer(BPE())
        added = tokenizer.add_tokens(["my", "name", "is", "john"])
        assert added == 4

        tokens = [AddedToken("the"), AddedToken("quick", normalized=False), AddedToken()]
        assert tokens[0].normalized == True
        added = tokenizer.add_tokens(tokens)
        assert added == 2
        assert tokens[0].normalized == True
        assert tokens[1].normalized == False

    def test_add_special_tokens(self):
        tokenizer = Tokenizer(BPE())

        # Can add special tokens as `str`
        added = tokenizer.add_special_tokens(["my", "name", "is", "john"])
        assert added == 4

        # Can add special tokens as `AddedToken`
        tokens = [AddedToken("the"), AddedToken("quick", normalized=True), AddedToken()]
        assert tokens[0].normalized == True
        added = tokenizer.add_special_tokens(tokens)
        assert added == 2
        assert tokens[0].normalized == False
        assert tokens[1].normalized == True

    def test_encode(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can encode single sequence
        output = tokenizer.encode("my name is john")
        assert output.tokens == ["my", "name", "is", "john"]
        assert type(output.ids) == list
        assert type(output.type_ids) == list
        assert type(output.offsets) == list
        assert type(output.words) == list
        assert type(output.special_tokens_mask) == list
        assert type(output.attention_mask) == list
        assert type(output.overflowing) == list

        # Can encode a pair of sequences
        output = tokenizer.encode("my name is john", "pair")
        assert output.tokens == ["my", "name", "is", "john", "pair"]
        assert isinstance(pickle.loads(pickle.dumps(output)), Encoding)

        # Can encode a single pre-tokenized sequence
        output = tokenizer.encode(["my", "name", "is", "john"], is_pretokenized=True)
        assert output.tokens == ["my", "name", "is", "john"]

        # Can encode a batch with both a single sequence and a pair of sequences
        output = tokenizer.encode_batch(["my name is john", ("my name is john", "pair")])
        assert len(output) == 2

    def test_encode_formats(self, bert_files):
        with pytest.deprecated_call():
            tokenizer = BertWordPieceTokenizer(bert_files["vocab"])

        # Encode
        output = tokenizer.encode("my name is john")
        assert output.tokens == ["[CLS]", "my", "name", "is", "john", "[SEP]"]
        output = tokenizer.encode("my name is john", "pair")
        assert output.tokens == ["[CLS]", "my", "name", "is", "john", "[SEP]", "pair", "[SEP]"]
        output = tokenizer.encode(["my", "name", "is", "john"], is_pretokenized=True)
        assert output.tokens == ["[CLS]", "my", "name", "is", "john", "[SEP]"]
        output = tokenizer.encode(["my", "name", "is", "john"], ["pair"], is_pretokenized=True)
        assert output.tokens == ["[CLS]", "my", "name", "is", "john", "[SEP]", "pair", "[SEP]"]

        # Encode batch
        result_single = [
            ["[CLS]", "my", "name", "is", "john", "[SEP]"],
            ["[CLS]", "my", "name", "is", "georges", "[SEP]"],
        ]
        result_pair = [
            ["[CLS]", "my", "name", "is", "john", "[SEP]", "pair", "[SEP]"],
            ["[CLS]", "my", "name", "is", "georges", "[SEP]", "pair", "[SEP]"],
        ]

        def format(encodings):
            return [e.tokens for e in encodings]

        def test_single(input, is_pretokenized=False):
            output = tokenizer.encode_batch(input, is_pretokenized=is_pretokenized)
            assert format(output) == result_single

        def test_pair(input, is_pretokenized=False):
            output = tokenizer.encode_batch(input, is_pretokenized=is_pretokenized)
            assert format(output) == result_pair

        # Classic inputs

        # Lists
        test_single(["My name is John", "My name is Georges"])
        test_pair([("my name is john", "pair"), ("my name is georges", "pair")])
        test_pair([["my name is john", "pair"], ["my name is georges", "pair"]])

        # Tuples
        test_single(("My name is John", "My name is Georges"))
        test_pair((("My name is John", "pair"), ("My name is Georges", "pair")))

        # Numpy
        test_single(np.array(["My name is John", "My name is Georges"]))
        test_pair(np.array([("My name is John", "pair"), ("My name is Georges", "pair")]))
        test_pair(np.array([["My name is John", "pair"], ["My name is Georges", "pair"]]))

        # PreTokenized inputs

        # Lists
        test_single([["My", "name", "is", "John"], ["My", "name", "is", "Georges"]], True)
        test_pair(
            [
                (["My", "name", "is", "John"], ["pair"]),
                (["My", "name", "is", "Georges"], ["pair"]),
            ],
            True,
        )
        test_pair(
            [
                [["My", "name", "is", "John"], ["pair"]],
                [["My", "name", "is", "Georges"], ["pair"]],
            ],
            True,
        )

        # Tuples
        test_single((("My", "name", "is", "John"), ("My", "name", "is", "Georges")), True)
        test_pair(
            (
                (("My", "name", "is", "John"), ("pair",)),
                (("My", "name", "is", "Georges"), ("pair",)),
            ),
            True,
        )
        test_pair(
            (
                (["My", "name", "is", "John"], ["pair"]),
                (["My", "name", "is", "Georges"], ["pair"]),
            ),
            True,
        )

        # Numpy
        test_single(
            np.array([["My", "name", "is", "John"], ["My", "name", "is", "Georges"]]),
            True,
        )
        test_single(
            np.array((("My", "name", "is", "John"), ("My", "name", "is", "Georges"))),
            True,
        )
        test_pair(
            np.array(
                [
                    [["My", "name", "is", "John"], ["pair"]],
                    [["My", "name", "is", "Georges"], ["pair"]],
                ],
                dtype=object,
            ),
            True,
        )
        test_pair(
            np.array(
                (
                    (("My", "name", "is", "John"), ("pair",)),
                    (("My", "name", "is", "Georges"), ("pair",)),
                ),
                dtype=object,
            ),
            True,
        )

        # Mal formed
        with pytest.raises(TypeError, match="TextInputSequence must be str"):
            tokenizer.encode([["my", "name"]])
        with pytest.raises(TypeError, match="TextInputSequence must be str"):
            tokenizer.encode("My name is john", [["pair"]])
        with pytest.raises(TypeError, match="TextInputSequence must be str"):
            tokenizer.encode("my name is john", ["pair"])

        with pytest.raises(TypeError, match="InputSequence must be Union[List[str]"):
            tokenizer.encode("My name is john", is_pretokenized=True)
        with pytest.raises(TypeError, match="InputSequence must be Union[List[str]"):
            tokenizer.encode("My name is john", ["pair"], is_pretokenized=True)
        with pytest.raises(TypeError, match="InputSequence must be Union[List[str]"):
            tokenizer.encode(["My", "name", "is", "John"], "pair", is_pretokenized=True)

    def test_encode_add_special_tokens(self, roberta_files):
        with pytest.deprecated_call():
            tokenizer = Tokenizer(BPE(roberta_files["vocab"], roberta_files["merges"]))
        tokenizer.add_special_tokens(["<s>", "</s>"])

        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.post_processor = RobertaProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )

        # Can encode with special tokens
        output_with_specials = tokenizer.encode("My name is John", add_special_tokens=True)
        assert output_with_specials.tokens == ["<s>", "ĠMy", "Ġname", "Ġis", "ĠJohn", "</s>"]

        # Can encode without special tokens
        output_without_specials = tokenizer.encode("My name is John", add_special_tokens=False)
        assert output_without_specials.tokens == ["ĠMy", "Ġname", "Ġis", "ĠJohn"]

    def test_truncation(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])
        tokenizer.enable_truncation(2)

        # Can truncate single sequences
        output = tokenizer.encode("my name is john")
        assert output.tokens == ["my", "name"]

        # Can truncate pair sequences as well
        output = tokenizer.encode("my name is john", "pair")
        assert output.tokens == ["my", "pair"]

        # Can get the params and give them to enable_truncation
        trunc = tokenizer.truncation
        tokenizer.enable_truncation(**trunc)

    def test_padding(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # By default it does nothing when encoding single sequence
        tokenizer.enable_padding()
        output = tokenizer.encode("my name")
        assert output.tokens == ["my", "name"]

        # Can pad to the longest in a batch
        output = tokenizer.encode_batch(["my name", "my name is john"])
        assert all([len(encoding) == 4 for encoding in output])

        # Can pad to the specified length otherwise
        tokenizer.enable_padding(length=4)
        output = tokenizer.encode("my name")
        assert output.tokens == ["my", "name", "[PAD]", "[PAD]"]
        output = tokenizer.encode("my name", "pair")
        assert output.tokens == ["my", "name", "pair", "[PAD]"]

        # Can get the params and give them to enable_padding
        padding = tokenizer.padding
        tokenizer.enable_padding(**padding)

    def test_decode(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can decode single sequences
        output = tokenizer.decode([0, 1, 2, 3])
        assert output == "my name is john"

        # Can decode batch
        output = tokenizer.decode_batch([[0, 1, 2, 3], [4]])
        assert output == ["my name is john", "pair"]

    def test_get_vocab(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can retrieve vocab with added tokens
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        assert vocab == {"is": 2, "john": 3, "my": 0, "name": 1, "pair": 4}

        # Can retrieve vocab without added tokens
        vocab = tokenizer.get_vocab(with_added_tokens=False)
        assert vocab == {}

    def test_get_vocab_size(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can retrieve vocab's size with added tokens
        size = tokenizer.get_vocab_size(with_added_tokens=True)
        assert size == 5

        # Can retrieve vocab's size without added tokens
        size = tokenizer.get_vocab_size(with_added_tokens=False)
        assert size == 0

    def test_post_process(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])
        tokenizer.enable_truncation(2)
        tokenizer.enable_padding(length=4)

        encoding = tokenizer.encode("my name is john")
        pair_encoding = tokenizer.encode("pair")

        # Can post process a single encoding
        output = tokenizer.post_process(encoding)
        assert output.tokens == ["my", "name", "[PAD]", "[PAD]"]

        # Can post process a pair of encodings
        output = tokenizer.post_process(encoding, pair_encoding)
        assert output.tokens == ["my", "pair", "[PAD]", "[PAD]"]

    def test_multiprocessing_with_parallelism(self):
        tokenizer = Tokenizer(BPE())
        multiprocessing_with_parallelism(tokenizer, False)
        multiprocessing_with_parallelism(tokenizer, True)
