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

class ByteLevel(PreTokenizer):
    """ ByteLevel PreTokenizer

    This pre-tokenizer takes care of replacing all bytes of the given string
    with a corresponding representation, as well as splitting into words.
    """

    def __init__(self, add_prefix_space: bool = True) -> None:
        """ Instantiate a new ByteLevel PreTokenizer
        Args:
            add_prefix_space: (`optional`) boolean:
                Whether to add a space to the first word if there isn't already one. This
                lets us treat `hello` exactly like `say hello`.
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

class Whitespace(PreTokenizer):
    """ Whitespace PreTokenizer

    This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
    """

    def __init__(self) -> None:
        """ Instantiate a new Whitespace PreTokenizer """
        pass

class WhitespaceSplit(PreTokenizer):
    """ Whitespace PreTokenizer

    This pre-tokenizer simply splits on the whitespace. Works like `.split()`
    """

    def __init__(self) -> None:
        """ Instantiate a new WhitespaceSplit PreTokenizer """
        pass

class BertPreTokenizer(PreTokenizer):
    """ BertPreTokenizer

    This pre-tokenizer splits tokens on spaces, and also on punctuation.
    Each occurence of a punctuation character will be treated separately.
    """

    def __init__(self) -> None:
        """ Instantiate a new BertPreTokenizer """
        pass

class Metaspace(PreTokenizer):
    """ Metaspace pre-tokenizer

    This pre-tokenizer replaces any whitespace by the provided replacement character.
    It then tries to split on these spaces.
    """

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

class CharDelimiterSplit(PreTokenizer):
    """ CharDelimiterSplit PreTokenizer

    This pre-tokenizer simply splits on the provided char. Works like `.split(delimiter)`
    """

    @staticmethod
    def __init__(self, delimiter: str) -> None:
        """ Instantiate a new CharDelimiterSplit PreTokenizer

        Args:
            delimiter: str:
                The delimiter char that will be used to split input
        """
        pass
