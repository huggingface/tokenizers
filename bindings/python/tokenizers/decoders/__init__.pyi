from typing import List

class Decoder:
    """ Base class for all decoders

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Decoder will return an instance of this class when instantiated.
    """

    def decode(self, tokens: List[str]) -> str:
        """ Decode the given list of string to a final string """
        pass

class ByteLevel:
    """ ByteLevel Decoder """

    @staticmethod
    def new() -> Decoder:
        """ Instantiate a new ByteLevel Decoder """
        pass

class WordPiece:
    """ WordPiece Decoder """

    @staticmethod
    def new(prefix: str="##") -> Decoder:
        """ Instantiate a new WordPiece Decoder

        Args:
            prefix: str:
                The prefix to use for subwords that are not a beginning-of-word
        """
        pass
