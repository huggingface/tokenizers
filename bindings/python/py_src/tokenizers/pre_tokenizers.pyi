from _typeshed import Incomplete
from tokenizers import PreTokenizedString, Regex
from typing import Any, final

@final
class BertPreTokenizer(PreTokenizer):
    def __new__(cls, /) -> BertPreTokenizer:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class ByteLevel(PreTokenizer):
    def __new__(
        cls, /, add_prefix_space: bool = True, trim_offsets: bool = True, use_regex: bool = True, **_kwargs
    ) -> ByteLevel:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
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
        ...
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
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, delimiter: str) -> CharDelimiterSplit:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def delimiter(self, /) -> str: ...
    @delimiter.setter
    def delimiter(self, /, delimiter: str) -> None: ...

@final
class Digits(PreTokenizer):
    def __new__(cls, /, individual_digits: bool = False) -> Digits:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def individual_digits(self, /) -> bool: ...
    @individual_digits.setter
    def individual_digits(self, /, individual_digits: bool) -> None: ...

@final
class FixedLength(PreTokenizer):
    def __new__(cls, /, length: int = 5) -> FixedLength:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def length(self, /) -> int: ...
    @length.setter
    def length(self, /, length: int) -> None: ...

@final
class Metaspace(PreTokenizer):
    def __new__(cls, /, replacement: str = "▁", prepend_scheme: str = ..., split: bool = True) -> Metaspace:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
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
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
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
        ...
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
        ...

@final
class Punctuation(PreTokenizer):
    def __new__(cls, /, behavior: Incomplete = ...) -> Punctuation:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def behavior(self, /) -> str: ...
    @behavior.setter
    def behavior(self, /, behavior: str) -> None: ...

@final
class Sequence(PreTokenizer):
    def __getitem__(self, /, index: int) -> Any:
        """Return self[key]."""
        ...
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, pre_tokenizers: list) -> Sequence:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __setitem__(self, /, index: int, value: Any) -> None:
        """Set self[key] to value."""
        ...

@final
class Split(PreTokenizer):
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, pattern: str | Regex, behavior: Incomplete, invert: bool = False) -> Split:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
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
    def __new__(cls, /) -> UnicodeScripts:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class Whitespace(PreTokenizer):
    def __new__(cls, /) -> Whitespace:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class WhitespaceSplit(PreTokenizer):
    def __new__(cls, /) -> WhitespaceSplit:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
