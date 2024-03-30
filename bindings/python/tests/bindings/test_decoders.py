import json
import pickle

import pytest

from tokenizers.decoders import (
    CTC,
    BPEDecoder,
    ByteLevel,
    Decoder,
    Metaspace,
    Sequence,
    WordPiece,
    ByteFallback,
    Replace,
    Strip,
    Fuse,
)


class TestByteLevel:
    def test_instantiate(self):
        assert ByteLevel() is not None
        assert isinstance(ByteLevel(), Decoder)
        assert isinstance(ByteLevel(), ByteLevel)
        assert isinstance(pickle.loads(pickle.dumps(ByteLevel())), ByteLevel)

    def test_decoding(self):
        decoder = ByteLevel()
        assert decoder.decode(["My", "Ġname", "Ġis", "ĠJohn"]) == "My name is John"

    def test_manual_reload(self):
        byte_level = ByteLevel()
        state = json.loads(byte_level.__getstate__())
        reloaded = ByteLevel(**state)
        assert isinstance(reloaded, ByteLevel)


class TestReplace:
    def test_instantiate(self):
        assert Replace("_", " ") is not None
        assert isinstance(Replace("_", " "), Decoder)
        assert isinstance(Replace("_", " "), Replace)
        # assert isinstance(pickle.loads(pickle.dumps(Replace("_", " "))), Replace)

    def test_decoding(self):
        decoder = Replace("_", " ")
        assert decoder.decode(["My", "_name", "_is", "_John"]) == "My name is John"


class TestWordPiece:
    def test_instantiate(self):
        assert WordPiece() is not None
        assert WordPiece(prefix="__") is not None
        assert WordPiece(cleanup=True) is not None
        assert isinstance(WordPiece(), Decoder)
        assert isinstance(WordPiece(), WordPiece)
        assert isinstance(pickle.loads(pickle.dumps(WordPiece())), WordPiece)

    def test_decoding(self):
        decoder = WordPiece()
        assert decoder.decode(["My", "na", "##me", "is", "Jo", "##hn"]) == "My name is John"
        assert decoder.decode(["I", "'m", "Jo", "##hn"]) == "I'm John"
        decoder = WordPiece(prefix="__", cleanup=False)
        assert decoder.decode(["My", "na", "__me", "is", "Jo", "__hn"]) == "My name is John"
        assert decoder.decode(["I", "'m", "Jo", "__hn"]) == "I 'm John"

    def test_can_modify(self):
        decoder = WordPiece(prefix="$$", cleanup=False)

        assert decoder.prefix == "$$"
        assert decoder.cleanup == False

        # Modify these
        decoder.prefix = "__"
        assert decoder.prefix == "__"
        decoder.cleanup = True
        assert decoder.cleanup == True


class TestByteFallback:
    def test_instantiate(self):
        assert ByteFallback() is not None
        assert isinstance(ByteFallback(), Decoder)
        assert isinstance(ByteFallback(), ByteFallback)
        assert isinstance(pickle.loads(pickle.dumps(ByteFallback())), ByteFallback)

    def test_decoding(self):
        decoder = ByteFallback()
        assert decoder.decode(["My", " na", "me"]) == "My name"
        assert decoder.decode(["<0x61>"]) == "a"
        assert decoder.decode(["<0xE5>"]) == "�"
        assert decoder.decode(["<0xE5>", "<0x8f>"]) == "��"
        assert decoder.decode(["<0xE5>", "<0x8f>", "<0xab>"]) == "叫"
        assert decoder.decode(["<0xE5>", "<0x8f>", "a"]) == "��a"
        assert decoder.decode(["<0xE5>", "<0x8f>", "<0xab>", "a"]) == "叫a"


class TestFuse:
    def test_instantiate(self):
        assert Fuse() is not None
        assert isinstance(Fuse(), Decoder)
        assert isinstance(Fuse(), Fuse)
        assert isinstance(pickle.loads(pickle.dumps(Fuse())), Fuse)

    def test_decoding(self):
        decoder = Fuse()
        assert decoder.decode(["My", " na", "me"]) == "My name"


class TestStrip:
    def test_instantiate(self):
        assert Strip(left=0, right=0) is not None
        assert isinstance(Strip(content="_", left=0, right=0), Decoder)
        assert isinstance(Strip(content="_", left=0, right=0), Strip)
        assert isinstance(pickle.loads(pickle.dumps(Strip(content="_", left=0, right=0))), Strip)

    def test_decoding(self):
        decoder = Strip(content="_", left=1, right=0)
        assert decoder.decode(["_My", " na", "me", " _-", "__-"]) == "My name _-_-"


class TestMetaspace:
    def test_instantiate(self):
        assert Metaspace() is not None
        assert Metaspace(replacement="-") is not None
        with pytest.raises(ValueError, match="expected a string of length 1"):
            Metaspace(replacement="")
        assert Metaspace(prepend_scheme="always") is not None
        assert isinstance(Metaspace(), Decoder)
        assert isinstance(Metaspace(), Metaspace)
        assert isinstance(pickle.loads(pickle.dumps(Metaspace())), Metaspace)

    def test_decoding(self):
        decoder = Metaspace()
        assert decoder.decode(["▁My", "▁name", "▁is", "▁John"]) == "My name is John"
        decoder = Metaspace(replacement="-", prepend_scheme="never")
        assert decoder.decode(["-My", "-name", "-is", "-John"]) == " My name is John"

    def test_can_modify(self):
        decoder = Metaspace(replacement="*", prepend_scheme="never")

        assert decoder.replacement == "*"
        assert decoder.prepend_scheme == "never"

        # Modify these
        decoder.replacement = "&"
        assert decoder.replacement == "&"
        decoder.prepend_scheme = "first"
        assert decoder.prepend_scheme == "first"


class TestBPEDecoder:
    def test_instantiate(self):
        assert BPEDecoder() is not None
        assert BPEDecoder(suffix="_") is not None
        assert isinstance(BPEDecoder(), Decoder)
        assert isinstance(BPEDecoder(), BPEDecoder)
        assert isinstance(pickle.loads(pickle.dumps(BPEDecoder())), BPEDecoder)

    def test_decoding(self):
        decoder = BPEDecoder()
        assert decoder.decode(["My</w>", "na", "me</w>", "is</w>", "Jo", "hn</w>"]) == "My name is John"
        decoder = BPEDecoder(suffix="_")
        assert decoder.decode(["My_", "na", "me_", "is_", "Jo", "hn_"]) == "My name is John"

    def test_can_modify(self):
        decoder = BPEDecoder(suffix="123")

        assert decoder.suffix == "123"

        # Modify these
        decoder.suffix = "</w>"
        assert decoder.suffix == "</w>"


class TestCTCDecoder:
    def test_instantiate(self):
        assert CTC() is not None
        assert CTC(pad_token="[PAD]") is not None
        assert isinstance(CTC(), Decoder)
        assert isinstance(CTC(), CTC)
        assert isinstance(pickle.loads(pickle.dumps(CTC())), CTC)

    def test_decoding(self):
        decoder = CTC()
        assert (
            decoder.decode(["<pad>", "<pad>", "h", "e", "e", "l", "l", "<pad>", "l", "o", "o", "o", "<pad>"])
            == "hello"
        )
        decoder = CTC(pad_token="[PAD]")
        assert (
            decoder.decode(["[PAD]", "[PAD]", "h", "e", "e", "l", "l", "[PAD]", "l", "o", "o", "o", "[PAD]"])
            == "hello"
        )

    def test_can_modify(self):
        decoder = CTC(pad_token="[PAD]")

        assert decoder.pad_token == "[PAD]"
        assert decoder.word_delimiter_token == "|"
        assert decoder.cleanup == True

        # Modify these
        decoder.pad_token = "{pad}"
        assert decoder.pad_token == "{pad}"

        decoder.word_delimiter_token = "_"
        assert decoder.word_delimiter_token == "_"

        decoder.cleanup = False
        assert decoder.cleanup == False


class TestSequenceDecoder:
    def test_instantiate(self):
        assert Sequence([]) is not None
        assert Sequence([CTC()]) is not None
        assert isinstance(Sequence([]), Decoder)
        assert isinstance(Sequence([]), Sequence)
        serialized = pickle.dumps(Sequence([]))
        assert isinstance(pickle.loads(serialized), Sequence)

    def test_decoding(self):
        decoder = Sequence([CTC(), Metaspace()])
        initial = ["▁", "▁", "H", "H", "i", "i", "▁", "y", "o", "u"]
        expected = "Hi you"
        assert decoder.decode(initial) == expected
