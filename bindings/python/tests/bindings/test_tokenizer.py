import pickle
import concurrent.futures
import pytest
import numpy as np
import asyncio
from tokenizers import AddedToken, Encoding, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.models import BPE, Model, Unigram
from tokenizers.pre_tokenizers import ByteLevel, Metaspace
from tokenizers.processors import RobertaProcessing, TemplateProcessing
from tokenizers.normalizers import Strip, Lowercase, Sequence
from tokenizers.decoders import ByteFallback, DecodeStream, Metaspace as DecoderMetaspace
import time

from ..utils import bert_files, data_dir, multiprocessing_with_parallelism, roberta_files


class TestAddedToken:
    def test_instantiate_with_content_only(self):
        added_token = AddedToken("<mask>")
        added_token.content = "<MASK>"
        assert added_token.content == "<MASK>"
        assert type(added_token) == AddedToken
        added_token.content = added_token.content.lower()

        assert added_token.special == False
        added_token.special = True
        assert added_token.special == True
        added_token.special = False
        assert str(added_token) == "<mask>"
        assert (
            repr(added_token)
            == 'AddedToken("<mask>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False)'
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
        assert callable(tokenizer.async_encode_batch)
        assert callable(tokenizer.decode)
        assert callable(tokenizer.decode_batch)
        assert callable(tokenizer.async_decode_batch)
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
        with pytest.warns(DeprecationWarning):
            assert type(output.words) == list
        assert type(output.word_ids) == list
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

        # Left truncation direction
        tokenizer.enable_truncation(2, direction="left")
        output = tokenizer.encode("my name is john")
        assert output.tokens == ["is", "john"]

        output = tokenizer.encode("my name is john", "pair")
        assert output.tokens == ["john", "pair"]

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

        # Can decode stream
        stream = DecodeStream(skip_special_tokens=False)
        assert stream.step(tokenizer, 0) == "my"
        assert stream.step(tokenizer, 1) == " name"
        assert stream.step(tokenizer, 2) == " is"
        assert stream.step(tokenizer, 3) == " john"

        stream = DecodeStream(ids=[0, 1, 2])
        assert stream.step(tokenizer, 3) == " john"

    def test_decode_stream_fallback(self):
        tokenizer = Tokenizer.from_pretrained("gpt2")
        # tokenizer.decode([255]) fails because its a fallback
        # tokenizer.encode("อั").ids = [19567, 255, 19567, 109]
        stream = DecodeStream()
        stream.step(tokenizer, [19567])
        stream.step(tokenizer, [255])
        stream.step(tokenizer, [19567])
        out = stream.step(tokenizer, [109])
        assert out == "ั"

        stream = DecodeStream()
        out = stream.step(tokenizer, [19567, 255, 19567, 109])
        assert out == "อั"
        stream = DecodeStream()
        stream.step(tokenizer, [19567])
        out = stream.step(tokenizer, [255, 19567, 109])
        assert out == "อั"

        stream = DecodeStream()
        stream.step(tokenizer, [19567])
        first_out = stream.step(tokenizer, [255])
        assert first_out == "อ"
        # since we emitted the 'อ', we can't produce 'อั'
        out = stream.step(tokenizer, [19567, 109])
        assert out == "ั"

        stream = DecodeStream([19567, 255, 19567])
        # the stream's prefix is 'อ�' which is invalid, thus all ids are kept for the next step
        out = stream.step(tokenizer, [109])
        assert out == "อั"

    def test_decode_skip_special_tokens(self):
        tokenizer = Tokenizer.from_pretrained("hf-internal-testing/Llama-3.1-8B-Instruct")

        stream = DecodeStream([40])
        out = stream.step(tokenizer, [2846, 40, 40, 40])
        assert out == "'mIII"

        stream = DecodeStream(
            [
                128000,
                128006,
                9125,
                128007,
                271,
                38766,
                1303,
                33025,
                2696,
                25,
                6790,
                220,
                2366,
                18,
                198,
                15724,
                2696,
                25,
                220,
                1627,
                10263,
                220,
                2366,
                19,
                271,
                9514,
                527,
                264,
                11190,
                18328,
                13,
                128009,
                128006,
                882,
                128007,
                271,
                15339,
                11,
                1268,
                527,
                499,
                30,
                128009,
                128006,
                78191,
                128007,
                271,
            ]
        )
        out = stream.step(tokenizer, 40)
        assert out == "I"

        stream = DecodeStream([40])
        out = stream.step(tokenizer, 2846)
        assert out == "'m"

        stream = DecodeStream([40])
        out = stream.step(tokenizer, [2846, 40, 40, 40])
        assert out == "'mIII"

    def test_decode_stream(self):
        vocab = [
            ("<unk>", 0.0),
            ("<0x20>", -0.1),
            ("<0xC3>", -0.2),
            ("<0xA9>", -0.3),
        ]
        tokenizer = Tokenizer(Unigram(vocab, 0, byte_fallback=True))
        tokenizer.decoder = ByteFallback()
        stream = DecodeStream(skip_special_tokens=False)
        assert stream.step(tokenizer, 1) == " "
        assert stream.step(tokenizer, 2) == None
        assert stream.step(tokenizer, 3) == "é"

        vocab = [
            ("<unk>", 0.0),
            ("▁This", -0.1),
        ]
        tokenizer = Tokenizer(Unigram(vocab, 0, byte_fallback=False))
        tokenizer.decoder = DecoderMetaspace()
        stream = DecodeStream(skip_special_tokens=False)
        assert stream.step(tokenizer, 1) == "This"
        assert stream.step(tokenizer, 1) == " This"

    def test_get_vocab(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can retrieve vocab with added tokens
        vocab = tokenizer.get_vocab(with_added_tokens=True)
        assert vocab == {"is": 2, "john": 3, "my": 0, "name": 1, "pair": 4}

        # Can retrieve vocab without added tokens
        vocab = tokenizer.get_vocab(with_added_tokens=False)
        assert vocab == {}

        # Can retrieve added token decoder
        vocab = tokenizer.get_added_tokens_decoder()
        assert vocab == {
            0: AddedToken("my", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
            1: AddedToken("name", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
            2: AddedToken("is", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
            3: AddedToken("john", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
            4: AddedToken("pair", rstrip=False, lstrip=False, single_word=False, normalized=True, special=False),
        }

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

    def test_from_pretrained(self):
        tokenizer = Tokenizer.from_pretrained("bert-base-cased")
        output = tokenizer.encode("Hey there dear friend!", add_special_tokens=False)
        assert output.tokens == ["Hey", "there", "dear", "friend", "!"]

    def test_from_pretrained_revision(self):
        tokenizer = Tokenizer.from_pretrained("anthony/tokenizers-test")
        output = tokenizer.encode("Hey there dear friend!", add_special_tokens=False)
        assert output.tokens == ["hey", "there", "dear", "friend", "!"]

        tokenizer = Tokenizer.from_pretrained("anthony/tokenizers-test", revision="gpt-2")
        output = tokenizer.encode("Hey there dear friend!", add_special_tokens=False)
        assert output.tokens == ["Hey", "Ġthere", "Ġdear", "Ġfriend", "!"]

    def test_unigram_byte_fallback(self):
        vocab = [
            ("<unk>", 0.0),
            ("A", -0.03),
            ("sen", -0.02),
            ("te", -0.03),
            ("n", -0.04),
            ("ce", -0.05),
            ("<0xF0>", -0.06),
            ("<0x9F>", -0.06),
            ("<0xA4>", -0.06),
            ("<0x97>", -0.06),
            (" ", -0.4),
        ]
        tokenizer = tokenizer = Tokenizer(Unigram(vocab, 0, byte_fallback=False))

        output = tokenizer.encode("A sentence 🤗")
        assert output.ids == [1, 10, 2, 3, 4, 5, 10, 0]
        assert output.tokens == ["A", " ", "sen", "te", "n", "ce", " ", "🤗"]

        tokenizer = Tokenizer(Unigram(vocab, 0, byte_fallback=True))

        output = tokenizer.encode("A sentence 🤗")
        assert output.ids == [1, 10, 2, 3, 4, 5, 10, 6, 7, 8, 9]
        assert output.tokens == ["A", " ", "sen", "te", "n", "ce", " ", "<0xF0>", "<0x9F>", "<0xA4>", "<0x97>"]

    def test_encode_special_tokens(self):
        tokenizer = Tokenizer.from_pretrained("t5-base")
        tokenizer.add_tokens(["<eot>"])
        tokenizer.add_special_tokens(["<end_of_text>"])
        output = tokenizer.encode("Hey there<end_of_text> dear<eot>friend!", add_special_tokens=False)
        assert output.tokens == ["▁Hey", "▁there", "<end_of_text>", "▁dear", "<eot>", "▁friend", "!"]

        tokenizer.encode_special_tokens = True
        assert tokenizer.encode_special_tokens == True

        output = tokenizer.encode("Hey there<end_of_text> dear<eot>friend!", add_special_tokens=False)
        assert output.tokens == [
            "▁Hey",
            "▁there",
            "<",
            "end",
            "_",
            "of",
            "_",
            "text",
            ">",
            "▁dear",
            "<eot>",
            "▁friend",
            "!",
        ]

        tokenizer.add_tokens(["of_text>"])
        output = tokenizer.encode("Hey there<end_of_text> dear<eot>friend!", add_special_tokens=False)
        assert output.tokens == ["▁Hey", "▁there", "<", "end", "_", "of_text>", "▁dear", "<eot>", "▁friend", "!"]

    def test_splitting(self):
        tokenizer = Tokenizer.from_pretrained("hf-internal-testing/llama-new-metaspace")
        tokenizer.pre_tokenizer.split = False
        tokenizer.add_tokens([AddedToken("<REPR_END>", rstrip=True, lstrip=True)])
        assert tokenizer.encode("<REPR_END>inform<s>. Hey.       .", add_special_tokens=False).tokens == [
            "<REPR_END>",
            "in",
            "form",
            "<s>",
            ".",
            "▁Hey",
            ".",
            "▁▁▁▁▁▁",
            "▁.",
        ]

        assert tokenizer.encode("<REPR_END>inform<s>. Hey.       .", add_special_tokens=False).ids == [
            32000,
            262,
            689,
            1,
            29889,
            18637,
            29889,
            539,
            869,
        ]

        assert tokenizer.encode("inform<s>. Hey.       .").tokens == [
            "<s>",
            "▁inform",
            "<s>",
            ".",
            "▁Hey",
            ".",
            "▁▁▁▁▁▁",
            "▁.",
        ]
        assert tokenizer.encode("inform<s>. Hey.       .", add_special_tokens=False).tokens == [
            "▁inform",
            "<s>",
            ".",
            "▁Hey",
            ".",
            "▁▁▁▁▁▁",
            "▁.",
        ]

    def test_decode_special(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens([AddedToken("my", special=True), AddedToken("name", special=False), "is", "john", "pair"])

        # Can decode single sequences
        output = tokenizer.decode([0, 1, 2, 3], skip_special_tokens=False)
        assert output == "my name is john"

        output = tokenizer.decode([0, 1, 2, 3], skip_special_tokens=True)
        assert output == "name is john"
        assert tokenizer.get_added_tokens_decoder()[0] == AddedToken("my", special=True)

    def test_setting_to_none(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.normalizer = Strip()
        tokenizer.normalizer = None
        assert tokenizer.normalizer == None

        tokenizer.pre_tokenizer = Metaspace()
        tokenizer.pre_tokenizer = None
        assert tokenizer.pre_tokenizer == None


class TestTokenizerRepr:
    def test_repr(self):
        tokenizer = Tokenizer(BPE())
        out = repr(tokenizer)
        assert (
            out
            == 'Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[], normalizer=None, pre_tokenizer=None, post_processor=None, decoder=None, model=BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[]))'
        )

    def test_repr_complete(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.post_processor = TemplateProcessing(
            single=["[CLS]", "$0", "[SEP]"],
            pair=["[CLS]:0", "$A", "[SEP]:0", "$B:1", "[SEP]:1"],
            special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
        )
        tokenizer.normalizer = Sequence([Lowercase(), Strip()])
        out = repr(tokenizer)
        assert (
            out
            == 'Tokenizer(version="1.0", truncation=None, padding=None, added_tokens=[], normalizer=Sequence(normalizers=[Lowercase(), Strip(strip_left=True, strip_right=True)]), pre_tokenizer=ByteLevel(add_prefix_space=True, trim_offsets=True, use_regex=True), post_processor=TemplateProcessing(single=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0)], pair=[SpecialToken(id="[CLS]", type_id=0), Sequence(id=A, type_id=0), SpecialToken(id="[SEP]", type_id=0), Sequence(id=B, type_id=1), SpecialToken(id="[SEP]", type_id=1)], special_tokens={"[CLS]":SpecialToken(id="[CLS]", ids=[1], tokens=["[CLS]"]), "[SEP]":SpecialToken(id="[SEP]", ids=[0], tokens=["[SEP]"])}), decoder=None, model=BPE(dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=False, byte_fallback=False, ignore_merges=False, vocab={}, merges=[]))'
        )


class TestAsyncTokenizer:
    """Tests for async methods of the Tokenizer class."""

    def setup_method(self):
        """Setup a basic tokenizer before each test."""
        self.tokenizer = Tokenizer.from_pretrained("hf-internal-testing/gpt-oss-20b")

    async def _compare_sync_async(self, input_data, is_pretokenized=False, add_special_tokens=True):
        """Helper to compare sync and async results for both normal and fast encoding."""
        # Normal encoding
        sync_result = self.tokenizer.encode_batch(input_data, is_pretokenized, add_special_tokens)
        async_result = await self.tokenizer.async_encode_batch(input_data, is_pretokenized, add_special_tokens)

        assert len(sync_result) == len(async_result)
        for s, a in zip(sync_result, async_result):
            assert s.tokens == a.tokens
            assert s.ids == a.ids
            assert s.offsets == a.offsets
            assert s.attention_mask == a.attention_mask
            assert s.special_tokens_mask == a.special_tokens_mask
            assert s.type_ids == a.type_ids

        # Fast encoding
        sync_fast_result = self.tokenizer.encode_batch_fast(input_data, is_pretokenized, add_special_tokens)
        async_fast_result = await self.tokenizer.async_encode_batch_fast(
            input_data, is_pretokenized, add_special_tokens
        )

        assert len(sync_fast_result) == len(async_fast_result)
        for s, a in zip(sync_fast_result, async_fast_result):
            assert s.tokens == a.tokens
            assert s.ids == a.ids
            assert s.attention_mask == a.attention_mask
            assert s.special_tokens_mask == a.special_tokens_mask
            assert s.type_ids == a.type_ids

    @pytest.mark.asyncio
    async def test_basic_encoding(self):
        """Test basic encoding functionality."""
        # Single sequences
        await self._compare_sync_async(["my name is john", "my pair"])

        # Pair sequences
        await self._compare_sync_async([("my name", "is john"), ("my", "pair")])

        # Empty batch
        await self._compare_sync_async([])

    @pytest.mark.asyncio
    async def test_encode(self):
        out = await self.tokenizer.async_encode("my name is john", "my pair")
        no_async_out = self.tokenizer.encode("my name is john", "my pair")
        assert out.ids == no_async_out.ids

        out = await self.tokenizer.async_encode("my name is john")
        no_async_out = self.tokenizer.encode("my name is john")
        assert out.ids == no_async_out.ids

    @pytest.mark.asyncio
    async def test_with_special_tokens(self):
        """Test with special tokens handling."""
        self.tokenizer.add_special_tokens(["[CLS]", "[SEP]"])
        self.tokenizer.post_processor = TemplateProcessing(
            single=["[CLS]", "$0", "[SEP]"],
            pair=["[CLS]", "$A", "[SEP]", "$B", "[SEP]"],
            special_tokens=[
                ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
                ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            ],
        )

        # With special tokens
        await self._compare_sync_async(["my name is john", "my pair"], add_special_tokens=True)

        # Without special tokens
        await self._compare_sync_async(["my name is john", "my pair"], add_special_tokens=False)

    @pytest.mark.asyncio
    async def test_with_truncation_padding(self):
        """Test with truncation and padding enabled."""
        self.tokenizer.enable_truncation(2)
        self.tokenizer.enable_padding(length=4)

        # Single sequences
        await self._compare_sync_async(["my name is john", "pair longer"])

        # Pair sequences
        await self._compare_sync_async([("my name", "is john"), ("pair", "longer")])

    @pytest.mark.asyncio
    async def test_various_input_formats(self):
        """Test with various input formats."""
        # Lists
        await self._compare_sync_async(["my name", "is john"])

        # Tuples
        await self._compare_sync_async(("my name", "is john"))

        # Numpy arrays
        # await self._compare_sync_async(np.array(["my name", "is john"]))

        # Mixed pairs
        await self._compare_sync_async([("my name", "is john"), ["my", "pair"]])

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors are handled consistently."""
        # Invalid input type for single item
        with pytest.raises(TypeError):
            await self.tokenizer.async_encode_batch(123)

        with pytest.raises(TypeError):
            self.tokenizer.encode_batch(123)

        # Invalid pre-tokenized input
        with pytest.raises(TypeError):
            await self.tokenizer.async_encode_batch("my name", is_pretokenized=True)

        with pytest.raises(TypeError):
            self.tokenizer.encode_batch("my name", is_pretokenized=True)

    @pytest.mark.asyncio
    async def test_concurrency(self):
        """Test concurrent encoding operations."""
        # Create some significant workload
        large_batch = ["my name is john " * 50] * 20

        # Run multiple encoding operations concurrently
        tasks = [
            self.tokenizer.async_encode_batch(large_batch),
            self.tokenizer.async_encode_batch_fast(large_batch),
            self.tokenizer.async_encode_batch(large_batch),
        ]

        # They should all complete successfully
        results = await asyncio.gather(*tasks)

        # Verify results
        assert len(results) == 3
        assert all(len(result) == 20 for result in results)

    @pytest.mark.asyncio
    async def test_decode(self):
        tokenizer = Tokenizer(BPE())
        tokenizer.add_tokens(["my", "name", "is", "john", "pair"])

        # Can decode single sequences
        output = tokenizer.decode([0, 1, 2, 3])
        assert output == "my name is john"

        output = tokenizer.decode_batch([[0, 1, 2, 3], [4]])
        assert output == ["my name is john", "pair"]

        output = await tokenizer.async_decode_batch([[0, 1, 2, 3], [4]])
        assert output == ["my name is john", "pair"]

    @pytest.mark.asyncio
    async def test_large_batch(self):
        """Test encoding a large batch of sequences."""
        large_batch = ["my name is john"] * 1000

        # Encode large batch both ways
        async_result = await self.tokenizer.async_encode_batch_fast(large_batch)
        sync_result = self.tokenizer.encode_batch_fast(large_batch)

        # Results should be identical
        assert len(async_result) == len(sync_result)
        assert all(a.tokens == s.tokens for a, s in zip(async_result[:10], sync_result[:10]))

    @pytest.mark.asyncio
    async def test_numpy_inputs(self):
        """Test with numpy array inputs."""
        # Single numpy array
        input_array = np.array(["my name", "is john", "pair longer"])
        await self._compare_sync_async(input_array)

        # Pre-tokenized numpy array
        pretok_array = np.array([["my", "name"], ["is", "john"]], dtype=object)
        await self._compare_sync_async(pretok_array, is_pretokenized=True)

    def test_async_methods_existence(self):
        """Test that the async methods exist on the Tokenizer class."""
        assert hasattr(self.tokenizer, "async_encode_batch")
        assert hasattr(self.tokenizer, "async_encode_batch_fast")
        assert callable(self.tokenizer.async_encode_batch)
        assert callable(self.tokenizer.async_encode_batch_fast)

    @pytest.mark.asyncio
    async def test_performance_comparison(self):
        """Compare performance between sync and async methods (informational)."""
        # Create a large batch for performance comparison
        large_batch = [
            "short text",
            "Sometimes it helps to have a better idea",
            "More short",
            "Let's not delve into that habbit sir",
            "I believe we can get to",
            "I am going to do it. I have made up my mind. These are the first few words of the new… the best … the Longest Text In The Entire History Of The Known Universe! This Has To Have Over 35,000 words the beat the current world record set by that person who made that flaming chicken handbooky thingy. I might just be saying random things the whole time I type in this so you might get confused a lot. I just discovered something terrible. autocorrect is on!! no!!! this has to be crazy, so I will have to break all the English language rules and the basic knowledge of the average human being. I am not an average human being, however I am special. no no no, not THAT kind of special ;). Why do people send that wink face! it always gives me nightmares! it can make a completely normal sentence creepy. imagine you are going to a friend’s house, so you text this: [ see you soon 🙂 ] seems normal, right? But what is you add the word semi to that colon? (Is that right? or is it the other way around) what is you add a lorry to that briquettes? (Semi-truck to that coal-on) anyway, back to the point: [ see you soon 😉 ]THAT IS JUST SO CREEPY! is that really your friend, or is it a creepy stalker watching your every move? Or even worse, is it your friend who is a creepy stalker? maybe you thought it was your friend, but it was actually your fri end (let me explain: you are happily in McDonalds, getting fat while eating yummy food and some random dude walks up and blots out the sun (he looks like a regular here) you can’t see anything else than him, so you can’t try to avoid eye contact. he finishes eating his cheeseburger (more like horseburgher(I learned that word from the merchant of Venice(which is a good play(if you can understand it(I can cause I got a special book with all the words in readable English written on the side of the page(which is kinda funny because Shakespeare was supposed to be a good poet but no-one can understand him(and he’s racist in act 2 scene1 of the play too))))))) and sits down beside you , like you are old pals (you’ve never met him before but he looks like he could be in some weird cult) he clears his throat and asks you a very personal question. “can i have some French fries?” (I don’t know why there called French fries when I’ve never seen a French person eat fries! all they eat it is stuff like baguettes and crêpes and rats named ratty-two-ee which is a really fun game on the PlayStation 2) And you think {bubbly cloud thinking bubble} “Hahahahahhahahahahahahahaha!!!!!!!!!!!! Hehheheheheh…..heeeheehe..hehe… sigh. I remember that i was just about to eat one of my fries when I noticed something mushy and moist and [insert gross color like green or brown] on the end of one of my fries! now I can give it to this NERD!! ” (yes he is a nerd because all he does all day is watch the extended editions of the hobbit, lord of the rings and star wars and eat fat cakes (what the heck is a fat cake? I think it might be like a Twinkie or something)and twinkies(wow so is doesn’t really matter which is which because he eats both(i may have just done that so I didn’t have to Google what a fat cake is (right now I am typing on my iPhone 3gs anyway, which has a broken antenna so i can’t get internet anyway (it’s actually a really funny story that i’ll tell you sometime)))and sit in his man cave with his friend named Joe (an ACTUAL friend, not a fri end)and all Joe does is watch sports like football with bob and all bob does is gamble ferociously (don’t ask(it means he buys all those bags of chips that say “win a free monkey or something if you find a banana in your bag*”(if there is a little star it means there is fine print so I always check the back of the package) *flips over the package* okay, it says: “one of our workers accidentally threw a banana in the packing machine and we don’t want to get sued so we did this promotion thing” cool. Oh wow, this is salt and vinegar! my favourite! i hate cheese and onion.))and that’s pretty much his life, he lives in Jamaica with Naruto and his friends) so you give him that gross fri end he throws up all over you and me and the worker behind the counter who was still making an onion, and THAT is the story of the fri end, not a friend who somehow remembered your name and your phone number / email so he could text you saying he would come to your house soon. *finally takes a breath after typing a few hundred words about fri-ends* so what now? i know, i know, you think i ramble too much and use too many brackets (i don’t) but now i am going to talk about my amAZEing day. first i woke up, ate choco pops for breakfast even tho i always hate it when people say that cause i get jealous and super hungry. then i… umm… yea! that was my day. you know that other person i mentioned before? that flaming chicken person? WELL. i will steal something from that person but do it better. i will… drum roll please … badabadabadabadabadabadabummmmmmmmmmmchshchshchshchshbadabadboumboumpoopoopichypichypichypowpow-crash! *a drum roll was just playing in the background* that drumroll was so long i forget what i was talking about. *scrolls up to see what he was writing about* oh yea! i will make my own FLAMING CHICKEN HANDBOOK! what things do i like? instead of flaming it could be rainbow, instead of chicken it could be fluffysheep and instead of handbook it could be handbook (not very creative, i know) but the total complete name is now to rainbow fluffysheep handbook! to make life easier for you guys, instead of taking random rules out of book willy nilly, i will take them out using my favourite numbers! so, section 5040 of the rainbow fluffysheep handbook states that the king of all oddly coloured farm animals (thats me!) is allowed to tell you any part out of this book randomly or if it is his one of his favorite numbers! 5040 is a great number because it is divisible by 60 integers which i don’t know. i’m tired. it is 10:41 and i am getting sleepy… hey hey hey! an intruder! remember that from pokepals rulers of time and darkness or something like that! with piplup and sunflora and chimchar! whaoh piplup is really hard to write on a tiny qwerty keyboard! try it! i realised that asdf is actually written in order on the qwerty keyboard! (just in case you didn’t know, asdf is an amazing short video clips cartoony thing on youtube i first learned bout on flipnote hatena, which is now shut down 😦 ) what if one day they get rid of the qwerty keyboard completely! i will type it out for you just in case one day they get rid of it.",
        ]
        results_sync = []
        results_async = []

        # Pre-initialize a thread pool executor with a reasonable number of workers
        # This avoids the overhead of creating the pool for each task

        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=2048)
            loop = asyncio.get_running_loop()

            async def encode_sync_with_executor(_):
                # Use the pre-initialized executor
                return await loop.run_in_executor(executor, lambda: self.tokenizer.encode_batch_fast(large_batch))

            async def encode_to_thread_sync(_):
                return await asyncio.to_thread(self.tokenizer.encode_batch_fast, large_batch)

            async def encode_async(_):
                return await self.tokenizer.async_encode_batch_fast(large_batch)

            await asyncio.gather(*[encode_sync_with_executor(i) for i in range(2048)])
            await asyncio.gather(*[encode_async(i) for i in range(2048)])

            for n_tasks in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
                # Measure sync performance with pre-initialized executor
                # Warm up
                await asyncio.gather(*[encode_sync_with_executor(i) for i in range(10)])
                time.sleep(0.03)
                # Actual measurement
                start = time.perf_counter()
                await asyncio.gather(*[encode_sync_with_executor(i) for i in range(n_tasks)])
                sync_time = time.perf_counter() - start

                # Measure async performance
                # Warm up
                await asyncio.gather(*[encode_async(i) for i in range(10)])

                # Actual measurement
                time.sleep(0.03)
                start = time.perf_counter()
                await asyncio.gather(*[encode_async(i) for i in range(n_tasks)])
                async_time = time.perf_counter() - start

                # Log times
                print(f"sync vs async processing times: {sync_time:.4f}s vs {async_time:.4f}s for {n_tasks} tasks")
                results_sync.append(sync_time)
                results_async.append(async_time)
        finally:
            # Make sure we shut down the executor properly
            executor.shutdown(wait=False)

        # assert async_time < sync_time, ("Async processing was faster than sync processing")
        assert any(a < s for a, s in zip(results_async, results_sync)), (
            "Async processing was faster than sync processing"
        )
