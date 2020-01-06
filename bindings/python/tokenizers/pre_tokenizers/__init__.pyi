from .. import pre_tokenizers

Offsets = Tuple[int, int]

class PreTokenizer:
    """PreTokenizer
    """

    def pre_tokenize(self, sequence: str) -> List[Tuple[str, Offsets]]:
        pass

class ByteLevel:
    """ByteLevel
    """

    @staticmethod
    def new() -> PreTokenizer:
        pass

    @staticmethod
    def alphabet() -> List[str]:
        pass

class Whitespace:
    """Whitespace
    """

    @staticmethod
    def new() -> PreTokenizer:
        pass

class BertPreTokenizer:
    """BertPreTokenizer
    """

    @staticmethod
    def new() -> PreTokenizer:
        pass
