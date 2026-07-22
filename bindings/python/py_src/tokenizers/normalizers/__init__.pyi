"""
Text cleanup that runs before the text is split.
"""

from collections.abc import Sequence as Sequence2
from typing import final

@final
class BertNormalizer(Normalizer):
    """
    The BERT cleanup: removes control characters, puts spaces around CJK
    characters, and optionally strips accents and lowercases.
    `strip_accents=None` means "follow the lowercase setting", like the
    original BERT.
    """
    def __new__(cls, /, *, clean_text: bool = True, handle_chinese_chars: bool = True, strip_accents: bool |None = None, lowercase: bool = True) -> BertNormalizer: ...

@final
class Lowercase(Normalizer):
    """
    Lowercases everything.
    """
    def __new__(cls, /) -> Lowercase: ...

@final
class NFC(Normalizer):
    """
    Unicode NFC: recombines split characters (e + ´ becomes é).
    """
    def __new__(cls, /) -> NFC: ...

@final
class NFD(Normalizer):
    """
    Unicode NFD: splits characters into base + accents (é becomes e + ´).
    """
    def __new__(cls, /) -> NFD: ...

@final
class NFKC(Normalizer):
    """
    Unicode NFKC: NFC, plus compatibility replacements (ﬁ becomes fi).
    """
    def __new__(cls, /) -> NFKC: ...

@final
class NFKD(Normalizer):
    """
    Unicode NFKD: NFD, plus compatibility replacements (ﬁ becomes fi).
    """
    def __new__(cls, /) -> NFKD: ...

class Normalizer:
    """
    Base class for all normalizers.
    
    A normalizer rewrites text before it is split: cleanup, case-folding,
    Unicode normalization. Normalizers are immutable values — assigning one to
    a tokenizer copies it.
    """
    def __repr__(self, /) -> str: ...

@final
class Prepend(Normalizer):
    """
    Puts a fixed string in front of the text (SentencePiece prepends "▁").
    """
    def __new__(cls, /, prepend: str) -> Prepend: ...

@final
class Replace(Normalizer):
    """
    Replaces every occurrence of `pattern` with `content`. With `regex=True`
    the pattern is a regular expression.
    """
    def __new__(cls, /, pattern: str, content: str, *, regex: bool = False) -> Replace: ...

@final
class Sequence(Normalizer):
    """
    Runs several normalizers in order.
    """
    def __new__(cls, /, normalizers: Sequence2[Normalizer]) -> Sequence: ...

@final
class Strip(Normalizer):
    """
    Removes whitespace at the start and/or end of the text.
    """
    def __new__(cls, /, *, left: bool = True, right: bool = True) -> Strip: ...

@final
class StripAccents(Normalizer):
    """
    Removes accents (é becomes e). Only works on decomposed text: put NFD before it.
    """
    def __new__(cls, /) -> StripAccents: ...

__all__ = ["BertNormalizer", "Lowercase", "NFC", "NFD", "NFKC", "NFKD", "Normalizer", "Prepend", "Replace", "Sequence", "Strip", "StripAccents"]
