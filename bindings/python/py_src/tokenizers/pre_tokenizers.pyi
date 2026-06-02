"""
PreTokenizers Module
"""

from _typeshed import Incomplete
from tokenizers import PreTokenizedString, Regex
from typing import Any, final

@final
class BertPreTokenizer(PreTokenizer):
    """
    BertPreTokenizer

    This pre-tokenizer splits tokens on whitespace and punctuation. Each occurrence of
    a punctuation character will be treated as a separate token. This is the pre-tokenizer
    used by the original BERT model.

    Example::

        >>> from tokenizers.pre_tokenizers import BertPreTokenizer
        >>> pre_tokenizer = BertPreTokenizer()
        >>> pre_tokenizer.pre_tokenize_str("Hello, I'm a single sentence!")
        [('Hello', (0, 5)), (',', (5, 6)), ('I', (7, 8)), ("'", (8, 9)), ('m', (9, 10)), ('a', (11, 12)), ('single', (13, 19)), ('sentence', (20, 28)), ('!', (28, 29))]
    """
    def __new__(cls, /) -> BertPreTokenizer: ...

@final
class ByteLevel(PreTokenizer):
    """
    ByteLevel PreTokenizer

    This pre-tokenizer takes care of replacing all bytes of the given string
    with a corresponding representation, as well as splitting into words.

    Args:
        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
        use_regex (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Set this to :obj:`False` to prevent this `pre_tokenizer` from using
            the GPT2 specific regexp for spliting on whitespace.

    Example::

        >>> from tokenizers.pre_tokenizers import ByteLevel
        >>> pre_tokenizer = ByteLevel()
        >>> pre_tokenizer.pre_tokenize_str("Hello my friend, how is it going?")
        [('ĠHello', (0, 5)), ('Ġmy', (5, 8)), ('Ġfriend,', (8, 15)), ('Ġhow', (15, 19)), ('Ġis', (19, 22)), ('Ġit', (22, 25)), ('Ġgoing?', (25, 32))]
    """
    def __new__(
        cls, /, add_prefix_space: bool = True, trim_offsets: bool = True, use_regex: bool = True, **_kwargs
    ) -> ByteLevel: ...
    @property
    def add_prefix_space(self, /) -> bool: ...
    @add_prefix_space.setter
    def add_prefix_space(self, /, add_prefix_space: bool) -> None: ...
    @staticmethod
    def alphabet() -> list[str]:
        """
        Returns the alphabet used by this PreTokenizer.

        Since the ByteLevel works as its name suggests, at the byte level, it
        encodes each byte value to a unique visible character. This means that there is a
        total of 256 different characters composing this alphabet.

        Returns:
            :obj:`List[str]`: A list of characters that compose the alphabet
        """
    @property
    def trim_offsets(self, /) -> bool: ...
    @trim_offsets.setter
    def trim_offsets(self, /, trim_offsets: bool) -> None: ...
    @property
    def use_regex(self, /) -> bool: ...
    @use_regex.setter
    def use_regex(self, /, use_regex: bool) -> None: ...

@final
class CharDelimiterSplit(PreTokenizer):
    """
    This pre-tokenizer simply splits on the provided char. Works like :meth:`str.split`
    with a single-character delimiter.

    Args:
        delimiter (:obj:`str`):
            The single character that will be used to split the input. The delimiter
            is removed from the output.

    Example::

        >>> from tokenizers.pre_tokenizers import CharDelimiterSplit
        >>> pre_tokenizer = CharDelimiterSplit("x")
        >>> pre_tokenizer.pre_tokenize_str("helloxthere")
        [('hello', (0, 5)), ('there', (6, 11))]
    """
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, delimiter: str) -> CharDelimiterSplit: ...
    @property
    def delimiter(self, /) -> str: ...
    @delimiter.setter
    def delimiter(self, /, delimiter: str) -> None: ...

@final
class Digits(PreTokenizer):
    """
    This pre-tokenizer simply splits using the digits in separate tokens

    Args:
        individual_digits (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If set to True, digits will each be separated as follows::

                "Call 123 please" -> "Call ", "1", "2", "3", " please"

            If set to False, digits will grouped as follows::

                "Call 123 please" -> "Call ", "123", " please"
    """
    def __new__(cls, /, individual_digits: bool = False) -> Digits: ...
    @property
    def individual_digits(self, /) -> bool: ...
    @individual_digits.setter
    def individual_digits(self, /, individual_digits: bool) -> None: ...

@final
class FixedLength(PreTokenizer):
    """
    This pre-tokenizer splits the text into fixed length chunks as used
    [here](https://www.biorxiv.org/content/10.1101/2023.01.11.523679v1.full)

    Args:
        length (:obj:`int`, `optional`, defaults to :obj:`5`):
            The length of the chunks to split the text into.

            Strings are split on the character level rather than the byte level to avoid
            splitting unicode characters consisting of multiple bytes.

    Example::

        >>> from tokenizers.pre_tokenizers import FixedLength
        >>> pre_tokenizer = FixedLength(length=3)
        >>> pre_tokenizer.pre_tokenize_str("Hello")
        [('Hel', (0, 3)), ('lo', (3, 5))]
    """
    def __new__(cls, /, length: int = 5) -> FixedLength: ...
    @property
    def length(self, /) -> int: ...
    @length.setter
    def length(self, /, length: int) -> None: ...

@final
class Metaspace(PreTokenizer):
    """
    Metaspace pre-tokenizer

    This pre-tokenizer replaces any whitespace by the provided replacement character.
    It then tries to split on these spaces.

    Args:
        replacement (:obj:`str`, `optional`, defaults to :obj:`▁`):
            The replacement character. Must be exactly one character. By default we
            use the `▁` (U+2581) meta symbol (Same as in SentencePiece).

        prepend_scheme (:obj:`str`, `optional`, defaults to :obj:`"always"`):
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
            Choices: "always", "never", "first". First means the space is only added on the first
            token (relevant when special tokens are used or other pre_tokenizer are used).

    Example::

        >>> from tokenizers.pre_tokenizers import Metaspace
        >>> pre_tokenizer = Metaspace()
        >>> pre_tokenizer.pre_tokenize_str("Hello my friend")
        [('▁Hello', (0, 5)), ('▁my', (6, 8)), ('▁friend', (9, 15))]
    """
    def __new__(cls, /, replacement: str = "▁", prepend_scheme: str = ..., split: bool = True) -> Metaspace: ...
    @property
    def prepend_scheme(self, /) -> str: ...
    @prepend_scheme.setter
    def prepend_scheme(self, /, prepend_scheme: str) -> None: ...
    @property
    def replacement(self, /) -> str: ...
    @replacement.setter
    def replacement(self, /, replacement: str) -> None: ...
    @property
    def split(self, /) -> bool: ...
    @split.setter
    def split(self, /, split: bool) -> None: ...

class PreTokenizer:
    """
    Base class for all pre-tokenizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    PreTokenizer will return an instance of this class when instantiated.
    """
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str: ...
    @staticmethod
    def custom(pretok: Any) -> PreTokenizer: ...
    def pre_tokenize(self, /, pretok: PreTokenizedString) -> None:
        """
        Pre-tokenize a :class:`~tokenizers.PyPreTokenizedString` in-place

        This method allows to modify a :class:`~tokenizers.PreTokenizedString` to
        keep track of the pre-tokenization, and leverage the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you just want to see the result of
        the pre-tokenization of a raw string, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str`

        Args:
            pretok (:class:`~tokenizers.PreTokenizedString):
                The pre-tokenized string on which to apply this
                :class:`~tokenizers.pre_tokenizers.PreTokenizer`
        """
    def pre_tokenize_str(self, /, s: str) -> list[tuple[str, tuple[int, int]]]:
        """
        Pre tokenize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` but it does not keep track of the
        alignment, nor does it provide all the capabilities of the
        :class:`~tokenizers.PreTokenizedString`. If you need some of these, you can use
        :meth:`~tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize`

        Args:
            sequence (:obj:`str`):
                A string to pre-tokeize

        Returns:
            :obj:`List[Tuple[str, Offsets]]`:
                A list of tuple with the pre-tokenized parts and their offsets
        """

@final
class Punctuation(PreTokenizer):
    """
    This pre-tokenizer simply splits on punctuation as individual characters.

    Args:
        behavior (:class:`~tokenizers.SplitDelimiterBehavior`):
            The behavior to use when splitting.
            Choices: "removed", "isolated" (default), "merged_with_previous", "merged_with_next",
            "contiguous"

    Example::

        >>> from tokenizers.pre_tokenizers import Punctuation
        >>> pre_tokenizer = Punctuation()
        >>> pre_tokenizer.pre_tokenize_str("Hello, how are you?")
        [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (15, 18)), ('?', (18, 19))]
    """
    def __new__(cls, /, behavior: Incomplete = ...) -> Punctuation: ...
    @property
    def behavior(self, /) -> str: ...
    @behavior.setter
    def behavior(self, /, behavior: str) -> None: ...

@final
class Sequence(PreTokenizer):
    """
    This pre-tokenizer composes other pre-tokenizers and applies them in sequence.
    Each pre-tokenizer in the list is applied to the output of the previous one,
    allowing complex tokenization strategies to be built by chaining simpler components.

    Args:
        pretokenizers (:obj:`List[PreTokenizer]`):
            A list of :class:`~tokenizers.pre_tokenizers.PreTokenizer` to be applied
            in sequence.

    Example::

        >>> from tokenizers.pre_tokenizers import Punctuation, Whitespace, Sequence
        >>> pre_tokenizer = Sequence([Whitespace(), Punctuation()])
        >>> pre_tokenizer.pre_tokenize_str("Hello, world!")
        [('Hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ('!', (12, 13))]
    """
    def __getitem__(self, /, index: int) -> Any: ...
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, pre_tokenizers: list) -> Sequence: ...
    def __setitem__(self, /, index: int, value: Any) -> None: ...

@final
class Split(PreTokenizer):
    """
    Split PreTokenizer

    This versatile pre-tokenizer splits using the provided pattern and
    according to the provided behavior. The pattern can be inverted by
    making use of the invert flag.

    Args:
        pattern (:obj:`str` or :class:`~tokenizers.Regex`):
            A pattern used to split the string. Usually a string or a regex built with `tokenizers.Regex`.
            If you want to use a regex pattern, it has to be wrapped around a `tokenizers.Regex`,
            otherwise we consider is as a string pattern. For example `pattern="|"`
            means you want to split on `|` (imagine a csv file for example), while
            `pattern=tokenizers.Regex("1|2")` means you split on either '1' or '2'.
        behavior (:class:`~tokenizers.SplitDelimiterBehavior`):
            The behavior to use when splitting.
            Choices: "removed", "isolated", "merged_with_previous", "merged_with_next",
            "contiguous"

        invert (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to invert the pattern.

    Example::

        >>> from tokenizers import Regex
        >>> from tokenizers.pre_tokenizers import Split
        >>> # Split on commas, removing them
        >>> pre_tokenizer = Split(",", behavior="removed")
        >>> pre_tokenizer.pre_tokenize_str("one,two,three")
        [('one', (0, 3)), ('two', (4, 7)), ('three', (8, 13))]
        >>> # Split using a regex, keeping the delimiter isolated
        >>> Split(Regex(r"\s+"), behavior="isolated").pre_tokenize_str("hello   world")
        [('hello', (0, 5)), ('   ', (5, 8)), ('world', (8, 13))]
    """
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, pattern: str | Regex, behavior: Incomplete, invert: bool = False) -> Split: ...
    @property
    def behavior(self, /) -> str: ...
    @behavior.setter
    def behavior(self, /, behavior: str) -> None: ...
    @property
    def invert(self, /) -> bool: ...
    @invert.setter
    def invert(self, /, invert: bool) -> None: ...
    @property
    def pattern(self, /) -> None: ...
    @pattern.setter
    def pattern(self, /, _pattern: str | Regex) -> None: ...

@final
class UnicodeScripts(PreTokenizer):
    """
    This pre-tokenizer splits on characters that belong to different language families.
    It roughly follows the SentencePiece script boundaries, with Hiragana and Katakana
    fused into the Han script category. This mimics the SentencePiece Unigram
    implementation and is useful for multilingual models that need to handle CJK text.

    Example::

        >>> from tokenizers.pre_tokenizers import UnicodeScripts
        >>> pre_tokenizer = UnicodeScripts()
        >>> pre_tokenizer.pre_tokenize_str("どこ Where")
        [('どこ', (0, 2)), ('Where', (3, 8))]
    """
    def __new__(cls, /) -> UnicodeScripts: ...

@final
class Whitespace(PreTokenizer):
    """
    This pre-tokenizer splits on word boundaries according to the ``\w+|[^\w\s]+``
    regex pattern. It splits on word characters or characters that aren't words or
    whitespaces (punctuation such as hyphens, apostrophes, commas, etc.).

    Example::

        >>> from tokenizers.pre_tokenizers import Whitespace
        >>> pre_tokenizer = Whitespace()
        >>> pre_tokenizer.pre_tokenize_str("Hello, world! Let's tokenize.")
        [('Hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ('!', (12, 13)), ('Let', (14, 17)), ("'", (17, 18)), ('s', (18, 19)), ('tokenize', (20, 28)), ('.', (28, 29))]
    """
    def __new__(cls, /) -> Whitespace: ...

@final
class WhitespaceSplit(PreTokenizer):
    """
    This pre-tokenizer simply splits on whitespace. Works like :meth:`str.split` with no
    arguments — it splits on any whitespace and discards the whitespace tokens. Unlike
    :class:`~tokenizers.pre_tokenizers.Whitespace`, it does not split on punctuation.

    Example::

        >>> from tokenizers.pre_tokenizers import WhitespaceSplit
        >>> pre_tokenizer = WhitespaceSplit()
        >>> pre_tokenizer.pre_tokenize_str("Hello, world! How are you?")
        [('Hello,', (0, 6)), ('world!', (7, 13)), ('How', (14, 17)), ('are', (18, 21)), ('you?', (22, 26))]
    """
    def __new__(cls, /) -> WhitespaceSplit: ...
