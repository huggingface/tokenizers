"""
The algorithms that turn pre-tokenized pieces into token ids.
"""

from typing import final

@final
class BPE(Model):
    """
    Byte-Pair Encoding: builds tokens by applying the merges learned during
    training. `unk_token` stands in for characters the vocabulary cannot
    represent; `byte_fallback` encodes them as raw bytes instead. `dropout`
    randomly skips merges (a training-time regularization). `ignore_merges`
    looks whole pieces up in the vocabulary before merging.
    """
    def __new__(cls, /, *, unk_token: str |None = None, dropout: float |None = None, fuse_unk: bool = False, byte_fallback: bool = False, ignore_merges: bool = False) -> BPE: ...
    @staticmethod
    def from_file(vocab: str, merges: str, *, unk_token: str |None = None) -> "BPE":
        """
        Load a BPE from the legacy vocab.json + merges.txt format.
        """

class Model:
    """
    Base class for all models.
    
    The model is the trained part of a tokenizer: it turns each pre-tokenized
    piece into token ids using its vocabulary. Models are immutable values —
    assigning one to a tokenizer copies it.
    """
    def __repr__(self, /) -> str: ...

@final
class Unigram(Model):
    """
    The SentencePiece Unigram model: picks the most probable segmentation
    under a learned piece vocabulary. Starts empty — train it, or load a
    tokenizer.json.
    """
    def __new__(cls, /) -> Unigram: ...

@final
class WordLevel(Model):
    """
    The simplest model: one whole word, one id. Words outside the vocabulary
    become `unk_token`.
    """
    def __new__(cls, /, *, unk_token: str = ...) -> WordLevel: ...

@final
class WordPiece(Model):
    """
    The BERT model: greedily matches the longest vocabulary entry, marking
    word continuations with a prefix ("##" by default). A piece longer than
    `max_input_chars_per_word` becomes `unk_token` outright.
    """
    def __new__(cls, /, *, unk_token: str = ..., continuing_subword_prefix: str = ..., max_input_chars_per_word: int = 100) -> WordPiece: ...
