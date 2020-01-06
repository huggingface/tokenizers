from .. import decoders

class Decoder:
    """Decoder
    """

    @staticmethod
    def custom():
        pass

    def decode(tokens: List[str]) -> str:
        pass

class ByteLevel:
    """ByteLevel
    """

    @staticmethod
    def new() -> Decoder:
        pass

class WordPiece:
    """WordPiece
    """

    @staticmethod
    def new() -> Decoder:
        pass
