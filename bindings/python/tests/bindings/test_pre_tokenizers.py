import pytest
import pickle

from tokenizers.pre_tokenizers import (
    PreTokenizer,
    ByteLevel,
    Whitespace,
    WhitespaceSplit,
    BertPreTokenizer,
    Metaspace,
    CharDelimiterSplit,
    Punctuation,
    Sequence,
    Digits,
)


class TestByteLevel:
    def test_instantiate(self):
        assert ByteLevel() is not None
        assert ByteLevel(add_prefix_space=True) is not None
        assert ByteLevel(add_prefix_space=False) is not None
        assert isinstance(ByteLevel(), PreTokenizer)
        assert isinstance(ByteLevel(), ByteLevel)
        assert isinstance(pickle.loads(pickle.dumps(ByteLevel())), ByteLevel)

    def test_has_alphabet(self):
        assert isinstance(ByteLevel.alphabet(), list)
        assert len(ByteLevel.alphabet()) == 256


class TestWhitespace:
    def test_instantiate(self):
        assert Whitespace() is not None
        assert isinstance(Whitespace(), PreTokenizer)
        assert isinstance(Whitespace(), Whitespace)
        assert isinstance(pickle.loads(pickle.dumps(Whitespace())), Whitespace)


class TestWhitespaceSplit:
    def test_instantiate(self):
        assert WhitespaceSplit() is not None
        assert isinstance(WhitespaceSplit(), PreTokenizer)
        assert isinstance(WhitespaceSplit(), WhitespaceSplit)
        assert isinstance(pickle.loads(pickle.dumps(WhitespaceSplit())), WhitespaceSplit)


class TestBertPreTokenizer:
    def test_instantiate(self):
        assert BertPreTokenizer() is not None
        assert isinstance(BertPreTokenizer(), PreTokenizer)
        assert isinstance(BertPreTokenizer(), BertPreTokenizer)
        assert isinstance(pickle.loads(pickle.dumps(BertPreTokenizer())), BertPreTokenizer)


class TestMetaspace:
    def test_instantiate(self):
        assert Metaspace() is not None
        assert Metaspace(replacement="-") is not None
        with pytest.raises(Exception, match="replacement must be a character"):
            Metaspace(replacement="")
        assert Metaspace(add_prefix_space=True) is not None
        assert isinstance(Metaspace(), PreTokenizer)
        assert isinstance(Metaspace(), Metaspace)
        assert isinstance(pickle.loads(pickle.dumps(Metaspace())), Metaspace)


class TestCharDelimiterSplit:
    def test_instantiate(self):
        assert CharDelimiterSplit("-") is not None
        with pytest.raises(Exception, match="delimiter must be a single character"):
            CharDelimiterSplit("")
        assert isinstance(CharDelimiterSplit(" "), PreTokenizer)
        assert isinstance(CharDelimiterSplit(" "), CharDelimiterSplit)
        assert isinstance(pickle.loads(pickle.dumps(CharDelimiterSplit("-"))), CharDelimiterSplit)


class TestPunctuation:
    def test_instantiate(self):
        assert Punctuation() is not None
        assert isinstance(Punctuation(), PreTokenizer)
        assert isinstance(Punctuation(), Punctuation)
        assert isinstance(pickle.loads(pickle.dumps(Punctuation())), Punctuation)


class TestSequence:
    def test_instantiate(self):
        assert Sequence([]) is not None
        assert isinstance(Sequence([]), PreTokenizer)
        assert isinstance(Sequence([]), Sequence)
        dumped = pickle.dumps(Sequence([]))
        assert isinstance(pickle.loads(dumped), Sequence)

    def test_bert_like(self):
        pre_tokenizer = Sequence([WhitespaceSplit(), Punctuation()])
        assert isinstance(Sequence([]), PreTokenizer)
        assert isinstance(Sequence([]), Sequence)
        assert isinstance(pickle.loads(pickle.dumps(pre_tokenizer)), Sequence)

        result = pre_tokenizer.pre_tokenize("Hey friend!     How are you?!?")
        assert result == [
            ("Hey", (0, 3)),
            ("friend", (4, 10)),
            ("!", (10, 11)),
            ("How", (16, 19)),
            ("are", (20, 23)),
            ("you", (24, 27)),
            ("?", (27, 28)),
            ("!", (28, 29)),
            ("?", (29, 30)),
        ]


class TestDigits:
    def test_instantiate(self):
        assert Digits() is not None
        assert isinstance(Digits(), PreTokenizer)
        assert isinstance(Digits(), Digits)
        assert isinstance(Digits(True), Digits)
        assert isinstance(Digits(False), Digits)
        assert isinstance(pickle.loads(pickle.dumps(Digits())), Digits)
