from collections.abc import Sequence as Sequence2
from tokenizers import NormalizedString, Regex
from typing import Any, final

@final
class BertNormalizer(Normalizer):
    def __new__(
        cls,
        /,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        lowercase: bool = True,
    ) -> BertNormalizer:
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
    def strip_accents(self, /) -> bool | None: ...
    @strip_accents.setter
    def strip_accents(self, /, strip_accents: bool | None) -> None: ...

@final
class ByteLevel(Normalizer):
    def __new__(cls, /) -> ByteLevel:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class Lowercase(Normalizer):
    def __new__(cls, /) -> Lowercase:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class NFC(Normalizer):
    def __new__(cls, /) -> NFC:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class NFD(Normalizer):
    def __new__(cls, /) -> NFD:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class NFKC(Normalizer):
    def __new__(cls, /) -> NFKC:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class NFKD(Normalizer):
    def __new__(cls, /) -> NFKD:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class Nmt(Normalizer):
    def __new__(cls, /) -> Nmt:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Normalizer:
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    @staticmethod
    def custom(obj: Any) -> Normalizer: ...
    def normalize(self, /, normalized: NormalizedString | Any) -> None:
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

@final
class Precompiled(Normalizer):
    def __new__(cls, /, precompiled_charsmap: Sequence2[int]) -> Precompiled:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

@final
class Prepend(Normalizer):
    def __new__(cls, /, prepend: str = ...) -> Prepend:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def prepend(self, /) -> str: ...
    @prepend.setter
    def prepend(self, /, prepend: str) -> None: ...

@final
class Replace(Normalizer):
    def __new__(cls, /, pattern: str | Regex, content: str) -> Replace:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def content(self, /) -> str: ...
    @content.setter
    def content(self, /, content: str) -> None: ...
    @property
    def pattern(self, /) -> None: ...
    @pattern.setter
    def pattern(self, /, _pattern: str | Regex) -> None: ...

@final
class Sequence(Normalizer):
    def __getitem__(self, /, index: int) -> Any:
        """Return self[key]."""
        ...
    def __getnewargs__(self, /) -> tuple: ...
    def __len__(self, /) -> int:
        """Return len(self)."""
        ...
    def __new__(cls, /, normalizers: list) -> Sequence:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __setitem__(self, /, index: int, value: Any) -> None:
        """Set self[key] to value."""
        ...

@final
class Strip(Normalizer):
    def __new__(cls, /, left: bool = True, right: bool = True) -> Strip:
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

@final
class StripAccents(Normalizer):
    def __new__(cls, /) -> StripAccents:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
