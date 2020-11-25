# Generated content DO NOT EDIT
class PreTokenizer:
    """
    Base class for all pre-tokenizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    PreTokenizer will return an instance of this class when instantiated.
    """

    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class BertPreTokenizer(PreTokenizer):
    """
    BertPreTokenizer

    This pre-tokenizer splits tokens on spaces, and also on punctuation.
    Each occurence of a punctuation character will be treated separately.
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class ByteLevel(PreTokenizer):
    """
    ByteLevel PreTokenizer

    This pre-tokenizer takes care of replacing all bytes of the given string
    with a corresponding representation, as well as splitting into words.

    Args:
        add_prefix_space: (`optional`) boolean:
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
    Returns:
        PreTokenizer
    """

    def __init__(self, add_prefix_space=True):
        pass
    @staticmethod
    def alphabet():
        """
        Returns the alphabet used by this PreTokenizer.

        Since the ByteLevel works as its name suggests, at the byte level, it
        encodes any byte to one visible character. This means that there is a
        total of 256 different characters composing this alphabet.
        """
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class CharDelimiterSplit(PreTokenizer):
    """
    This pre-tokenizer simply splits on the provided char. Works like `.split(delimiter)`

    Args:
        delimiter: str:
            The delimiter char that will be used to split input
    """

    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class Digits(PreTokenizer):
    """
    This pre-tokenizer simply splits using the digits in separate tokens
    Args:
        individual_digits: bool:
            If set to True, digits will each be separated "Call 123 please" -> "Call ", "1", "2", "3", " please"
            If set to False, digits will grouped "Call 123 please" -> "Call ", "123", " please"
    """

    def __init__(self, individual_digits=False):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class Metaspace(PreTokenizer):
    """
    Metaspace pre-tokenizer

    This pre-tokenizer replaces any whitespace by the provided replacement character.
    It then tries to split on these spaces.
    Args:
        replacement: str:
            The replacement character. Must be exactly one character. By default we
            use the `▁` (U+2581) meta symbol (Same as in SentencePiece).

        add_prefix_space: boolean:
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
    """

    def __init__(self, replacement="▁", add_prefix_space=True):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class Punctuation(PreTokenizer):
    """
    This pre-tokenizer simply splits on punctuation as individual characters.`
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class Sequence(PreTokenizer):
    """
    This pre-tokenizer composes other pre_tokenizers and applies them in sequence
    """

    def __init__(self, pretokenizers):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class Split(PreTokenizer):
    """
    Split PreTokenizer

    This versatile pre-tokenizer splits using the provided pattern and
    according to the provided behavior. The pattern can be inverted by
    making use of the invert flag.

    Args:
        pattern: Pattern:
            A pattern used to split the string. Usually a string or a Regex

        behavior: SplitDelimiterBehavior:
            The behavior to use when splitting.
            Choices: "removed", "isolated", "merged_with_previous", "merged_with_next",
            "contiguous"

        invert: bool:
            Whether to invert the pattern.
    """

    def __init__(self, pattern, behavior, invert=False):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class UnicodeScripts(PreTokenizer):
    """
    This pre-tokenizer splits on characters that belong to different language family
    It roughly follows https://github.com/google/sentencepiece/blob/master/data/Scripts.txt
    Actually Hiragana and Katakana are fused with Han, and 0x30FC is Han too.
    This mimicks SentencePiece Unigram implementation.
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class Whitespace(PreTokenizer):
    """
    This pre-tokenizer simply splits using the following regex: `\w+|[^\w\s]+`
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass

class WhitespaceSplit(PreTokenizer):
    """
    This pre-tokenizer simply splits on the whitespace. Works like `.split()`
    """

    def __init__(self):
        pass
    def pre_tokenize(self, pretok):
        """
        Pre tokenize the given PreTokenizedString in-place
        """
        pass
    def pre_tokenize_str(self, sequence):
        """
        Pre tokenize the given sequence
        """
        pass
