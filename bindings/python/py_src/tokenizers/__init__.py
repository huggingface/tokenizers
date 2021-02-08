__version__ = "0.10.1"

from typing import Tuple, Union, Tuple, List
from enum import Enum

Offsets = Tuple[int, int]

TextInputSequence = str
"""A :obj:`str` that represents an input sequence """

PreTokenizedInputSequence = Union[List[str], Tuple[str]]
"""A pre-tokenized input sequence. Can be one of:

    - A :obj:`List` of :obj:`str`
    - A :obj:`Tuple` of :obj:`str`
"""

TextEncodeInput = Union[
    TextInputSequence,
    Tuple[TextInputSequence, TextInputSequence],
    List[TextInputSequence],
]
"""Represents a textual input for encoding. Can be either:

    - A single sequence: :data:`~tokenizers.TextInputSequence`
    - A pair of sequences:

      - A :obj:`Tuple` of :data:`~tokenizers.TextInputSequence`
      - Or a :obj:`List` of :data:`~tokenizers.TextInputSequence` of size 2
"""

PreTokenizedEncodeInput = Union[
    PreTokenizedInputSequence,
    Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence],
    List[PreTokenizedInputSequence],
]
"""Represents a pre-tokenized input for encoding. Can be either:

    - A single sequence: :data:`~tokenizers.PreTokenizedInputSequence`
    - A pair of sequences:

      - A :obj:`Tuple` of :data:`~tokenizers.PreTokenizedInputSequence`
      - Or a :obj:`List` of :data:`~tokenizers.PreTokenizedInputSequence` of size 2
"""

InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]
"""Represents all the possible types of input sequences for encoding. Can be:

    - When ``is_pretokenized=False``: :data:`~TextInputSequence`
    - When ``is_pretokenized=True``: :data:`~PreTokenizedInputSequence`
"""

EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]
"""Represents all the possible types of input for encoding. Can be:

    - When ``is_pretokenized=False``: :data:`~TextEncodeInput`
    - When ``is_pretokenized=True``: :data:`~PreTokenizedEncodeInput`
"""


class OffsetReferential(Enum):
    ORIGINAL = "original"
    NORMALIZED = "normalized"


class OffsetType(Enum):
    BYTE = "byte"
    CHAR = "char"


class SplitDelimiterBehavior(Enum):
    REMOVED = "removed"
    ISOLATED = "isolated"
    MERGED_WITH_PREVIOUS = "merged_with_previous"
    MERGED_WITH_NEXT = "merged_with_next"
    CONTIGUOUS = "contiguous"


from .tokenizers import (
    Tokenizer,
    Encoding,
    AddedToken,
    Regex,
    NormalizedString,
    PreTokenizedString,
    Token,
)
from .tokenizers import decoders
from .tokenizers import models
from .tokenizers import normalizers
from .tokenizers import pre_tokenizers
from .tokenizers import processors
from .tokenizers import trainers

from .implementations import (
    ByteLevelBPETokenizer,
    CharBPETokenizer,
    SentencePieceBPETokenizer,
    SentencePieceUnigramTokenizer,
    BertWordPieceTokenizer,
)
