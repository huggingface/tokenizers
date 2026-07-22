"""
How text is cut into pieces before the model runs.
"""

from collections.abc import Sequence as Sequence2
from typing import final

@final
class BertPreTokenizer(PreTokenizer):
    """
    The BERT split: on whitespace, and each punctuation character becomes its own piece.
    """
    def __new__(cls, /) -> BertPreTokenizer: ...

@final
class ByteLevel(PreTokenizer):
    """
    GPT-2 style byte-level splitting: cuts with the GPT-2 regex unless
    `use_regex=False`. The pipeline does not support `add_prefix_space`, so it
    is always off.
    """
    def __new__(cls, /, *, use_regex: bool = True) -> ByteLevel: ...
    @staticmethod
    def alphabet() -> list[str]:
        """
        The 256 characters byte-level tokens are spelled with, one per byte
        value. Pass it as a trainer's `initial_alphabet` so every byte gets a
        token even if it never appears in the training data.
        """

@final
class CharDelimiterSplit(PreTokenizer):
    """
    Splits on one fixed character, dropping it.
    """
    def __new__(cls, /, delimiter: str) -> CharDelimiterSplit: ...

@final
class Digits(PreTokenizer):
    """
    Separates digits from everything else. With `individual_digits=True`,
    every digit becomes its own piece.
    """
    def __new__(cls, /, *, individual_digits: bool = False) -> Digits: ...

@final
class FixedLength(PreTokenizer):
    """
    Cuts the text into pieces of exactly `length` characters (the last one may
    be shorter).
    """
    def __new__(cls, /, *, length: int = 5) -> FixedLength: ...

class PreTokenizer:
    """
    Base class for all pre-tokenizers.
    
    A pre-tokenizer cuts text into pieces (usually words); the model then turns
    each piece into token ids. Pre-tokenizers are immutable values — assigning
    one to a tokenizer copies it. Only pre-tokenizers the encode pipeline can
    run are constructible here; `Metaspace` is not available yet.
    """
    def __repr__(self, /) -> str: ...

@final
class Punctuation(PreTokenizer):
    """
    Splits on punctuation. `behavior` says what happens to the punctuation
    itself — see `Split` for the options; the default, "isolated", keeps each
    punctuation character as its own piece.
    """
    def __new__(cls, /, behavior: str = ...) -> Punctuation: ...

@final
class Sequence(PreTokenizer):
    """
    Runs several pre-tokenizers in order, each one further splitting the
    pieces left by the previous.
    """
    def __new__(cls, /, pre_tokenizers: Sequence2[PreTokenizer]) -> Sequence: ...

@final
class Split(PreTokenizer):
    """
    Splits on a pattern: a literal string, or a regular expression with
    `regex=True`. `behavior` says what to do with each match — "removed" drops
    it, "isolated" keeps it as its own piece, "merged_with_previous" /
    "merged_with_next" glue it to a neighbor, "contiguous" merges runs of
    matches. `invert=True` keeps the matches and splits everything else.
    """
    def __new__(cls, /, pattern: str, behavior: str = ..., *, invert: bool = False, regex: bool = False) -> Split: ...

@final
class UnicodeScripts(PreTokenizer):
    """
    Splits where the script changes (Latin to Han, for example), so a piece never mixes alphabets.
    """
    def __new__(cls, /) -> UnicodeScripts: ...

@final
class Whitespace(PreTokenizer):
    """
    Splits into runs of letters/digits/underscore or runs of other symbols (the pattern `\w+|[^\w\s]+`).
    """
    def __new__(cls, /) -> Whitespace: ...

@final
class WhitespaceSplit(PreTokenizer):
    """
    Splits on whitespace only.
    """
    def __new__(cls, /) -> WhitespaceSplit: ...

__all__ = ["BertPreTokenizer", "ByteLevel", "CharDelimiterSplit", "Digits", "FixedLength", "PreTokenizer", "Punctuation", "Sequence", "Split", "UnicodeScripts", "Whitespace", "WhitespaceSplit"]
