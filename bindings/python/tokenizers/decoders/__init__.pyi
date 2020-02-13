from typing import List

class Decoder:
    """ Base class for all decoders

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Decoder will return an instance of this class when instantiated.
    """

    def decode(self, tokens: List[str]) -> str:
        """ Decode the given list of string to a final string """
        pass

class ByteLevel(Decoder):
    """ ByteLevel Decoder """

    def __init__(self) -> None:
        """ Instantiate a new ByteLevel Decoder """
        pass

class WordPiece(Decoder):
    """ WordPiece Decoder """

    @staticmethod
    def __init__(self, prefix: str = "##", cleanup: bool = True) -> Decoder:
        """ Instantiate a new WordPiece Decoder

        Args:
            prefix: str:
                The prefix to use for subwords that are not a beginning-of-word
            cleanup: bool:
                Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation,
                and some abbreviated english forms.
        """
        pass

class Metaspace(Decoder):
    """ Metaspace decoder """

    def __init__(self, replacement: str = "▁", add_prefix_space: bool = True) -> None:
        """ Instantiate a new Metaspace

        Args:
            replacement: str:
                The replacement character. Must be exactly one character. By default we
                use the `▁` (U+2581) meta symbol (Same as in SentencePiece).

            add_prefix_space: boolean:
                Whether to add a space to the first word if there isn't already one. This
                lets us treat `hello` exactly like `say hello`.
        """
        pass

class BPEDecoder(Decoder):
    """ BPEDecoder """

    def __init__(self, suffix: str = "</w>") -> None:
        """ Instantiate a new BPEDecoder

        Args:
            suffix: str:
                The suffix that was used to caracterize an end-of-word. This suffix will
                be replaced by whitespaces during the decoding
        """
        pass
