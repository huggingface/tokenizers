"""
Normalizers Module
"""

from collections.abc import Sequence as Sequence2
from tokenizers import NormalizedString, Regex
from typing import Any, final

@final
class BertNormalizer(Normalizer):
    """
    BertNormalizer

    Takes care of normalizing raw text before giving it to a Bert model.
    This includes cleaning the text, handling accents, chinese chars and lowercasing

    Args:
        clean_text (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to clean the text, by removing any control characters
            and replacing all whitespaces by the classic one.

        handle_chinese_chars (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to handle chinese chars by putting spaces around them.

        strip_accents (:obj:`bool`, `optional`):
            Whether to strip all accents. If this option is not specified (ie == None),
            then it will be determined by the value for `lowercase` (as in the original Bert).

        lowercase (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to lowercase.

    Example::

        >>> from tokenizers.normalizers import BertNormalizer
        >>> normalizer = BertNormalizer(lowercase=True)
        >>> normalizer.normalize_str("Héllo WORLD")
        'hello world'
    """
    def __new__(
        cls,
        /,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        lowercase: bool = True,
    ) -> BertNormalizer: ...
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
    """
    Bytelevel Normalizer

    Converts all bytes in the input to their Unicode representation using the GPT-2
    byte-to-unicode mapping. Every byte value (0–255) is mapped to a unique visible
    character so that any arbitrary binary input can be tokenized without needing a
    special unknown token.

    This normalizer is used together with the
    :class:`~tokenizers.pre_tokenizers.ByteLevel` pre-tokenizer and
    :class:`~tokenizers.decoders.ByteLevel` decoder.

    Example::

        >>> from tokenizers.normalizers import ByteLevel
        >>> normalizer = ByteLevel()
        >>> normalizer.normalize_str("hello\nworld")
        'helloĊworld'
    """
    def __new__(cls, /) -> ByteLevel: ...

@final
class Lowercase(Normalizer):
    """
    Lowercase Normalizer

    Converts all text to lowercase using Unicode-aware lowercasing. This is equivalent
    to calling :meth:`str.lower` on the input.

    Example::

        >>> from tokenizers.normalizers import Lowercase
        >>> normalizer = Lowercase()
        >>> normalizer.normalize_str("Hello World")
        'hello world'
    """
    def __new__(cls, /) -> Lowercase: ...

@final
class NFC(Normalizer):
    """
    NFC Unicode Normalizer

    Applies Unicode NFC (Canonical Decomposition, followed by Canonical Composition)
    normalization. First decomposes characters, then recomposes them using canonical
    composition rules. This produces the canonical composed form.

    Example::

        >>> from tokenizers.normalizers import NFC
        >>> normalizer = NFC()
        >>> normalizer.normalize_str("e\u0301")  # 'e' + combining accent
        'é'
    """
    def __new__(cls, /) -> NFC: ...

@final
class NFD(Normalizer):
    """
    NFD Unicode Normalizer

    Applies Unicode NFD (Canonical Decomposition) normalization. Decomposes characters into
    their canonical components. For example, accented characters like ``é`` (U+00E9) are
    decomposed into ``e`` (U+0065) + combining accent (U+0301).

    This is often used as a first step before stripping accents with
    :class:`~tokenizers.normalizers.StripAccents`.

    Example::

        >>> from tokenizers.normalizers import NFD
        >>> normalizer = NFD()
        >>> normalizer.normalize_str("Héllo")
        'He\u0301llo'
    """
    def __new__(cls, /) -> NFD: ...

@final
class NFKC(Normalizer):
    """
    NFKC Unicode Normalizer

    Applies Unicode NFKC (Compatibility Decomposition, followed by Canonical Composition)
    normalization. Like NFC but also maps compatibility characters to their canonical
    equivalents. This is the normalization used by Python's :func:`str.casefold` and
    by many NLP pipelines.

    Example::

        >>> from tokenizers.normalizers import NFKC
        >>> normalizer = NFKC()
        >>> normalizer.normalize_str("ﬁne caf\u00e9")
        'fine café'
    """
    def __new__(cls, /) -> NFKC: ...

@final
class NFKD(Normalizer):
    """
    NFKD Unicode Normalizer

    Applies Unicode NFKD (Compatibility Decomposition) normalization. Like NFD but also
    decomposes compatibility characters. For example, the ligature ``ﬁ`` (U+FB01) is
    decomposed into ``f`` + ``i``.

    Example::

        >>> from tokenizers.normalizers import NFKD
        >>> normalizer = NFKD()
        >>> normalizer.normalize_str("ﬁne")
        'fine'
    """
    def __new__(cls, /) -> NFKD: ...

@final
class Nmt(Normalizer):
    """
    Nmt normalizer

    Normalizer used in the Google NMT pipeline. It handles various text cleaning tasks
    including removing control characters, normalizing whitespace, and replacing certain
    Unicode characters. This is equivalent to the normalization done in the original
    SentencePiece NMT preprocessing.

    Example::

        >>> from tokenizers.normalizers import Nmt
        >>> normalizer = Nmt()
        >>> normalizer.normalize_str("Hello\x00World")
        'Hello World'
    """
    def __new__(cls, /) -> Nmt: ...

class Normalizer:
    """
    Base class for all normalizers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Normalizer will return an instance of this class when instantiated.
    """
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str: ...
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

@final
class Precompiled(Normalizer):
    """
    Precompiled normalizer

    A normalizer that uses a precompiled character map built from a SentencePiece model.
    This normalizer is automatically extracted from SentencePiece ``.model`` files and
    should not be constructed manually — it is used internally for full compatibility
    with SentencePiece-based tokenizers.

    Args:
        precompiled_charsmap (:obj:`bytes`):
            The raw bytes of the precompiled character map, as found inside a
            SentencePiece ``.model`` file.
    """
    def __new__(cls, /, precompiled_charsmap: Sequence2[int]) -> Precompiled: ...

@final
class Prepend(Normalizer):
    """
    Prepend normalizer

    Prepends a given string to the beginning of the input. This is typically used to
    add a meta-symbol such as ``▁`` (U+2581) at the start of each sequence, which is
    the convention used by SentencePiece-based models to indicate that a token appears
    at the start of a word.

    Args:
        prepend (:obj:`str`, defaults to :obj:`"▁"`):
            The string to prepend to the input.

    Example::

        >>> from tokenizers.normalizers import Prepend
        >>> normalizer = Prepend("▁")
        >>> normalizer.normalize_str("hello")
        '▁hello'
    """
    def __new__(cls, /, prepend: str = ...) -> Prepend: ...
    @property
    def prepend(self, /) -> str: ...
    @prepend.setter
    def prepend(self, /, prepend: str) -> None: ...

@final
class Replace(Normalizer):
    """
    Replace normalizer

    Replaces occurrences of a pattern in the input string with the given content.
    The pattern can be either a plain string or a regular expression wrapped in
    :class:`~tokenizers.Regex`.

    Args:
        pattern (:obj:`str` or :class:`~tokenizers.Regex`):
            The pattern to search for. Use a plain string for literal replacement,
            or wrap a regex pattern in :class:`~tokenizers.Regex` for regex replacement.

        content (:obj:`str`):
            The string to replace each match with.

    Example::

        >>> from tokenizers import Regex
        >>> from tokenizers.normalizers import Replace
        >>> # Replace a literal string
        >>> Replace(".", " ").normalize_str("hello.world")
        'hello world'
        >>> # Replace using a regex
        >>> Replace(Regex(r"\s+"), " ").normalize_str("hello   world")
        'hello world'
    """
    def __new__(cls, /, pattern: str | Regex, content: str) -> Replace: ...
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
    """
    Allows concatenating multiple other Normalizer as a Sequence.
    All the normalizers run in sequence in the given order

    Args:
        normalizers (:obj:`List[Normalizer]`):
            A list of Normalizer to be run as a sequence

    Example::

        >>> from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
        >>> normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        >>> normalizer.normalize_str("Héllo Wörld")
        'hello world'
    """
    def __getitem__(self, /, index: int) -> Any: ...
    def __getnewargs__(self, /) -> tuple: ...
    def __len__(self, /) -> int: ...
    def __new__(cls, /, normalizers: list) -> Sequence: ...
    def __setitem__(self, /, index: int, value: Any) -> None: ...

@final
class Strip(Normalizer):
    """
    Strip normalizer

    Removes leading and/or trailing whitespace from the input string.

    Args:
        left (:obj:`bool`, defaults to :obj:`True`):
            Whether to strip leading (left) whitespace.

        right (:obj:`bool`, defaults to :obj:`True`):
            Whether to strip trailing (right) whitespace.

    Example::

        >>> from tokenizers.normalizers import Strip
        >>> normalizer = Strip()
        >>> normalizer.normalize_str("  hello world  ")
        'hello world'
        >>> Strip(right=False).normalize_str("  hello  ")
        'hello  '
    """
    def __new__(cls, /, left: bool = True, right: bool = True) -> Strip: ...
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
    """
    StripAccents normalizer

    Strips all accent marks (combining diacritical characters) from the input. This
    normalizer should typically be used after applying :class:`~tokenizers.normalizers.NFD`
    or :class:`~tokenizers.normalizers.NFKD` decomposition, which separates base
    characters from their combining accents.

    Example::

        >>> from tokenizers.normalizers import NFD, StripAccents, Sequence
        >>> normalizer = Sequence([NFD(), StripAccents()])
        >>> normalizer.normalize_str("café")
        'cafe'
    """
    def __new__(cls, /) -> StripAccents: ...
