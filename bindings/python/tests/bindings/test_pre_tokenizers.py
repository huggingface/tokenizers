import pytest

from tokenizers.pre_tokenizers import (
    PreTokenizer,
    ByteLevel,
    Whitespace,
    WhitespaceSplit,
    BertPreTokenizer,
    Metaspace,
    CharDelimiterSplit,
)


class TestByteLevel:
    def test_instantiate(self):
        assert ByteLevel() is not None
        assert ByteLevel(add_prefix_space=True) is not None
        assert ByteLevel(add_prefix_space=False) is not None
        assert isinstance(ByteLevel(), PreTokenizer)
        assert isinstance(ByteLevel(), ByteLevel)

    def test_has_alphabet(self):
        assert isinstance(ByteLevel.alphabet(), list)
        assert len(ByteLevel.alphabet()) == 256


class TestWhitespace:
    def test_instantiate(self):
        assert Whitespace() is not None
        assert isinstance(Whitespace(), PreTokenizer)
        assert isinstance(Whitespace(), Whitespace)


class TestWhitespaceSplit:
    def test_instantiate(self):
        assert WhitespaceSplit() is not None
        assert isinstance(WhitespaceSplit(), PreTokenizer)
        assert isinstance(WhitespaceSplit(), WhitespaceSplit)


class TestBertPreTokenizer:
    def test_instantiate(self):
        assert BertPreTokenizer() is not None
        assert isinstance(BertPreTokenizer(), PreTokenizer)
        assert isinstance(BertPreTokenizer(), BertPreTokenizer)


class TestMetaspace:
    def test_instantiate(self):
        assert Metaspace() is not None
        assert Metaspace(replacement="-") is not None
        with pytest.raises(Exception, match="replacement must be a character"):
            Metaspace(replacement="")
        assert Metaspace(add_prefix_space=True) is not None
        assert isinstance(Metaspace(), PreTokenizer)
        assert isinstance(Metaspace(), Metaspace)


class TestCharDelimiterSplit:
    def test_instantiate(self):
        assert CharDelimiterSplit("-") is not None
        with pytest.raises(Exception, match="delimiter must be a single character"):
            CharDelimiterSplit("")
        assert isinstance(CharDelimiterSplit(" "), PreTokenizer)
        assert isinstance(CharDelimiterSplit(" "), CharDelimiterSplit)
