import pytest
import pickle

from tokenizers.decoders import Decoder, ByteLevel, WordPiece, Metaspace, BPEDecoder


class TestByteLevel:
    def test_instantiate(self):
        assert ByteLevel() is not None
        assert isinstance(ByteLevel(), Decoder)
        assert isinstance(ByteLevel(), ByteLevel)
        assert isinstance(pickle.loads(pickle.dumps(ByteLevel())), ByteLevel)

    def test_decoding(self):
        decoder = ByteLevel()
        assert decoder.decode(["My", "Ġname", "Ġis", "ĠJohn"]) == "My name is John"


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


class TestMetaspace:
    def test_instantiate(self):
        assert Metaspace() is not None
        assert Metaspace(replacement="-") is not None
        with pytest.raises(Exception, match="replacement must be a character"):
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
