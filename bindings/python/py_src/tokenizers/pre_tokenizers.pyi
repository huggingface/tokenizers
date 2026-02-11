import tokenizers
import tokenizers.pre_tokenizers
import typing

class BertPreTokenizer:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class ByteLevel:
    def __new__(
        cls, /, add_prefix_space: bool = True, trim_offsets: bool = True, use_regex: bool = True, **_kwargs
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def add_prefix_space(self, /) -> bool: ...
    @add_prefix_space.setter
    def add_prefix_space(self, /, add_prefix_space: bool) -> None: ...
    @staticmethod
    def alphabet() -> typing.Any:
        """
        Returns the alphabet used by this PreTokenizer.

        Since the ByteLevel works as its name suggests, at the byte level, it
        encodes each byte value to a unique visible character. This means that there is a
        total of 256 different characters composing this alphabet.

        Returns:
            :obj:`List[str]`: A list of characters that compose the alphabet
        """
        ...
    @property
    def trim_offsets(self, /) -> bool: ...
    @trim_offsets.setter
    def trim_offsets(self, /, trim_offsets: bool) -> None: ...
    @property
    def use_regex(self, /) -> bool: ...
    @use_regex.setter
    def use_regex(self, /, use_regex: bool) -> None: ...

class CharDelimiterSplit:
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(cls, /, delimiter: str) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def delimiter(self, /) -> str: ...
    @delimiter.setter
    def delimiter(self, /, delimiter: str) -> None: ...

class Digits:
    def __new__(cls, /, individual_digits: bool = False) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def individual_digits(self, /) -> bool: ...
    @individual_digits.setter
    def individual_digits(self, /, individual_digits: bool) -> None: ...

class FixedLength:
    def __new__(cls, /, length: int = 5) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def length(self, /) -> int: ...
    @length.setter
    def length(self, /, length: int) -> None: ...

class Metaspace:
    def __new__(cls, /, replacement: str = "▁", prepend_scheme: str = ..., split: bool = True) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def prepend_scheme(self, /) -> str: ...
    @prepend_scheme.setter
    def prepend_scheme(self, /, prepend_scheme: str) -> typing.Any: ...
    @property
    def replacement(self, /) -> str: ...
    @replacement.setter
    def replacement(self, /, replacement: str) -> None: ...
    @property
    def split(self, /) -> bool: ...
    @split.setter
    def split(self, /, split: bool) -> None: ...

class PreTokenizer:
    def __getstate__(self, /) -> typing.Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    @staticmethod
    def custom(pretok: typing.Any) -> tokenizers.pre_tokenizers.PreTokenizer: ...
    def pre_tokenize(self, /, pretok: tokenizers.PreTokenizedString) -> typing.Any:
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
        ...
    def pre_tokenize_str(self, /, s: str) -> typing.Any:
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
        ...

class Punctuation:
    def __new__(cls, /, behavior: typing.Any = ...) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def behavior(self, /) -> str: ...
    @behavior.setter
    def behavior(self, /, behavior: str) -> typing.Any: ...

class Sequence:
    def __getitem__(self, /, index: int) -> typing.Any:
        """Return self[key]."""
        ...
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(cls, /, pre_tokenizers: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __setitem__(self, /, index: int, value: typing.Any) -> typing.Any:
        """Set self[key] to value."""
        ...

class Split:
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(cls, /, pattern: str | tokenizers.Regex, behavior: typing.Any, invert: bool = False) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def behavior(self, /) -> str: ...
    @behavior.setter
    def behavior(self, /, behavior: str) -> typing.Any: ...
    @property
    def invert(self, /) -> bool: ...
    @invert.setter
    def invert(self, /, invert: bool) -> None: ...
    @property
    def pattern(self, /) -> typing.Any: ...
    @pattern.setter
    def pattern(self, /, _pattern: str | tokenizers.Regex) -> typing.Any: ...

class UnicodeScripts:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Whitespace:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class WhitespaceSplit:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
