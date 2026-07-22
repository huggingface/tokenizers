from collections.abc import Sequence as Sequence2
from typing import final

@final
class BertPreTokenizer(PreTokenizer):
    def __new__(cls, /) -> BertPreTokenizer: ...

@final
class ByteLevel(PreTokenizer):
    def __new__(cls, /, *, use_regex: bool = True) -> ByteLevel:
        """
        `add_prefix_space` is not supported by the pipeline and is always false.
        """

@final
class CharDelimiterSplit(PreTokenizer):
    def __new__(cls, /, delimiter: str) -> CharDelimiterSplit: ...

@final
class Digits(PreTokenizer):
    def __new__(cls, /, *, individual_digits: bool = False) -> Digits: ...

@final
class FixedLength(PreTokenizer):
    def __new__(cls, /, *, length: int = 5) -> FixedLength: ...

class PreTokenizer:
    """
    Base class for all pre-tokenizers. Immutable value: assigning it to a
    Tokenizer copies the configuration, there is no shared state.
    
    Only pre-tokenizers supported by the encode pipeline are constructible here;
    notably `Metaspace` is not available yet.
    """
    def __repr__(self, /) -> str: ...

@final
class Punctuation(PreTokenizer):
    def __new__(cls, /, behavior: str = ...) -> Punctuation: ...

@final
class Sequence(PreTokenizer):
    def __new__(cls, /, pre_tokenizers: Sequence2[PreTokenizer]) -> Sequence: ...

@final
class Split(PreTokenizer):
    def __new__(cls, /, pattern: str, behavior: str = ..., *, invert: bool = False, regex: bool = False) -> Split: ...

@final
class UnicodeScripts(PreTokenizer):
    def __new__(cls, /) -> UnicodeScripts: ...

@final
class Whitespace(PreTokenizer):
    def __new__(cls, /) -> Whitespace: ...

@final
class WhitespaceSplit(PreTokenizer):
    def __new__(cls, /) -> WhitespaceSplit: ...
