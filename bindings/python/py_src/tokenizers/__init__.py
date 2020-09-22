__version__ = "0.9.0.dev1"

from typing import Tuple, Union, Tuple, List

Offsets = Tuple[int, int]

TextInputSequence = str
PreTokenizedInputSequence = Union[List[str], Tuple[str]]
TextEncodeInput = Union[TextInputSequence, Tuple[TextInputSequence, TextInputSequence]]
PreTokenizedEncodeInput = Union[
    PreTokenizedInputSequence, Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence],
]

InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]
EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]

from .tokenizers import Tokenizer, Encoding, AddedToken, Regex, NormalizedString
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
