import pickle

import numpy as np
import pytest

from tokenizers import AddedToken, Encoding, Tokenizer
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.models import BPE, Model, Unigram
from tokenizers.pre_tokenizers import ByteLevel, Metaspace
from tokenizers.processors import RobertaProcessing, TemplateProcessing
from tokenizers.normalizers import Strip, Lowercase, Sequence
from tokenizers.decoders import ByteFallback, DecodeStream, Metaspace as DecoderMetaspace


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
            ("A", -0.01),
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
