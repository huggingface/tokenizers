from collections.abc import Sequence as Sequence2
from typing import final

@final
class BertNormalizer(Normalizer):
    def __new__(cls, /, *, clean_text: bool = True, handle_chinese_chars: bool = True, strip_accents: bool |None = None, lowercase: bool = True) -> BertNormalizer: ...

@final
class Lowercase(Normalizer):
    def __new__(cls, /) -> Lowercase: ...

@final
class NFC(Normalizer):
    def __new__(cls, /) -> NFC: ...

@final
class NFD(Normalizer):
    def __new__(cls, /) -> NFD: ...

@final
class NFKC(Normalizer):
    def __new__(cls, /) -> NFKC: ...

@final
class NFKD(Normalizer):
    def __new__(cls, /) -> NFKD: ...

class Normalizer:
    """
    Base class for all normalizers. Immutable value: assigning it to a
    Tokenizer copies the configuration, there is no shared state.
    """
    def __repr__(self, /) -> str: ...

@final
class Prepend(Normalizer):
    def __new__(cls, /, prepend: str) -> Prepend: ...

@final
class Replace(Normalizer):
    def __new__(cls, /, pattern: str, content: str, *, regex: bool = False) -> Replace: ...

@final
class Sequence(Normalizer):
    def __new__(cls, /, normalizers: Sequence2[Normalizer]) -> Sequence: ...

@final
class Strip(Normalizer):
    def __new__(cls, /, *, left: bool = True, right: bool = True) -> Strip: ...

@final
class StripAccents(Normalizer):
    def __new__(cls, /) -> StripAccents: ...
