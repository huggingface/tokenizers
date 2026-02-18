import tokenizers
import tokenizers.normalizers
import typing

class BertNormalizer:
    def __new__(
        cls,
        /,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        lowercase: bool = True,
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def clean_text(self, /) -> bool: ...
    @clean_text.setter
    def clean_text(self, /, clean_text: bool) -> None: ...
    @property
    def handle_chinese_chars(self, /) -> bool: ...
    @handle_chinese_chars.setter
    def handle_chinese_chars(self, /, handle_chinese_chars: bool) -> None: ...
    @property
    def lowercase(self, /) -> bool: ...
    @lowercase.setter
    def lowercase(self, /, lowercase: bool) -> None: ...
    @property
    def strip_accents(self, /) -> typing.Any: ...
    @strip_accents.setter
    def strip_accents(self, /, strip_accents: bool | None) -> None: ...

class ByteLevel:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Lowercase:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class NFC:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class NFD:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class NFKC:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class NFKD:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Nmt:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Normalizer:
    def __getstate__(self, /) -> typing.Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    @staticmethod
    def custom(obj: typing.Any) -> tokenizers.normalizers.Normalizer: ...
    def normalize(self, /, normalized: tokenizers.NormalizedString | tokenizers.NormalizedStringRefMut) -> typing.Any:
        """
        Normalize a :class:`~tokenizers.NormalizedString` in-place

        This method allows to modify a :class:`~tokenizers.NormalizedString` to
        keep track of the alignment information. If you just want to see the result
        of the normalization on a raw string, you can use
        :meth:`~tokenizers.normalizers.Normalizer.normalize_str`

        Args:
            normalized (:class:`~tokenizers.NormalizedString`):
                The normalized string on which to apply this
                :class:`~tokenizers.normalizers.Normalizer`
        """
        ...
    def normalize_str(self, /, sequence: str) -> str:
        """
        Normalize the given string

        This method provides a way to visualize the effect of a
        :class:`~tokenizers.normalizers.Normalizer` but it does not keep track of the alignment
        information. If you need to get/convert offsets, you can use
        :meth:`~tokenizers.normalizers.Normalizer.normalize`

        Args:
            sequence (:obj:`str`):
                A string to normalize

        Returns:
            :obj:`str`: A string after normalization
        """
        ...

class Precompiled:
    def __new__(cls, /, precompiled_charsmap: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Prepend:
    def __new__(cls, /, prepend: str = ...) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def prepend(self, /) -> str: ...
    @prepend.setter
    def prepend(self, /, prepend: str) -> None: ...

class Replace:
    def __new__(cls, /, pattern: str | tokenizers.Regex, content: str) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def content(self, /) -> str: ...
    @content.setter
    def content(self, /, content: str) -> None: ...
    @property
    def pattern(self, /) -> typing.Any: ...
    @pattern.setter
    def pattern(self, /, _pattern: str | tokenizers.Regex) -> typing.Any: ...

class Sequence:
    def __getitem__(self, /, index: int) -> typing.Any:
        """Return self[key]."""
        ...
    def __getnewargs__(self, /) -> typing.Any: ...
    def __len__(self, /) -> int:
        """Return len(self)."""
        ...
    def __new__(cls, /, normalizers: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __setitem__(self, /, index: int, value: typing.Any) -> typing.Any:
        """Set self[key] to value."""
        ...

class Strip:
    def __new__(cls, /, left: bool = True, right: bool = True) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def left(self, /) -> bool: ...
    @left.setter
    def left(self, /, left: bool) -> None: ...
    @property
    def right(self, /) -> bool: ...
    @right.setter
    def right(self, /, right: bool) -> None: ...

class StripAccents:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
