from typing import Optional, List, Tuple

Offsets = Tuple[int, int]

class PreTokenizer:
    """ Base class for all pre-tokenizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    PreTokenizer will return an instance of this class when instantiated.
    """

    def pre_tokenize(self, sequence: str) -> List[Tuple[str, Offsets]]:
        """ Pre tokenize the given sequence """
        pass

class ByteLevel:
    """ ByteLevel PreTokenizer

    This pre-tokenizer takes care of replacing all bytes of the given string
    with a corresponding representation, as well as splitting into words.
    """

    @staticmethod
    def new(add_prefix_space: Optional[bool]=True) -> PreTokenizer:
        """ Instantiate a new ByteLevel PreTokenizer

        Args:
            add_prefix_space: (`optional`) boolean:
                Whether a space should be added at the very beginning of the sequence
                if there isn't one already.

        Returns:
            PreTokenizer
        """
        pass

    @staticmethod
    def alphabet() -> List[str]:
        """ Returns the alphabet used by this PreTokenizer.

        Since the ByteLevel works as its name suggests, at the byte level, it
        encodes any byte to one visible character. This means that there is a
        total of 256 different characters composing this alphabet.
        """
        pass

class Whitespace:
    """ Whitespace PreTokenizer

    This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
    """

    @staticmethod
    def new() -> PreTokenizer:
        """ Instantiate a new Whitespace PreTokenizer """
        pass

class BertPreTokenizer:
    """ BertPreTokenizer

    This pre-tokenizer splits tokens on spaces, and also on punctuation.
    Each occurence of a punctuation character will be treated separately.
    """

    @staticmethod
    def new() -> PreTokenizer:
        """ Instantiate a new BertPreTokenizer """
        pass
