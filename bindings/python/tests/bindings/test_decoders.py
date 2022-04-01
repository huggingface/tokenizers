import pytest
import pickle
import json

from tokenizers.decoders import Decoder, ByteLevel, WordPiece, Metaspace, BPEDecoder, CTC


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


class TestMetaspace:
    def test_instantiate(self):
        assert Metaspace() is not None
        assert Metaspace(replacement="-") is not None
        with pytest.raises(ValueError, match="expected a string of length 1"):
            Metaspace(replacement="")
        assert Metaspace(add_prefix_space=True) is not None
        assert isinstance(Metaspace(), Decoder)
        assert isinstance(Metaspace(), Metaspace)
        assert isinstance(pickle.loads(pickle.dumps(Metaspace())), Metaspace)

    def test_decoding(self):
        decoder = Metaspace()
        assert decoder.decode(["▁My", "▁name", "▁is", "▁John"]) == "My name is John"
        decoder = Metaspace(replacement="-", add_prefix_space=False)
        assert decoder.decode(["-My", "-name", "-is", "-John"]) == " My name is John"

    def test_can_modify(self):
        decoder = Metaspace(replacement="*", add_prefix_space=False)

        assert decoder.replacement == "*"
        assert decoder.add_prefix_space == False

        # Modify these
        decoder.replacement = "&"
        assert decoder.replacement == "&"
        decoder.add_prefix_space = True
        assert decoder.add_prefix_space == True


class TestBPEDecoder:
    def test_instantiate(self):
        assert BPEDecoder() is not None
        assert BPEDecoder(suffix="_") is not None
        assert isinstance(BPEDecoder(), Decoder)
        assert isinstance(BPEDecoder(), BPEDecoder)
        assert isinstance(pickle.loads(pickle.dumps(BPEDecoder())), BPEDecoder)

    def test_decoding(self):
        decoder = BPEDecoder()
        assert (
            decoder.decode(["My</w>", "na", "me</w>", "is</w>", "Jo", "hn</w>"])
            == "My name is John"
        )
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
            decoder.decode(
                ["<pad>", "<pad>", "h", "e", "e", "l", "l", "<pad>", "l", "o", "o", "o", "<pad>"]
            )
            == "hello"
        )
        decoder = CTC(pad_token="[PAD]")
        assert (
            decoder.decode(
                ["[PAD]", "[PAD]", "h", "e", "e", "l", "l", "[PAD]", "l", "o", "o", "o", "[PAD]"]
            )
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
