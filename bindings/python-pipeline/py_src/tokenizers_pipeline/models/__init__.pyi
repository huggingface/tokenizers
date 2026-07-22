from typing import final

@final
class BPE(Model):
    def __new__(cls, /, *, unk_token: str |None = None, dropout: float |None = None, fuse_unk: bool = False, byte_fallback: bool = False, ignore_merges: bool = False) -> BPE: ...
    @staticmethod
    def from_file(vocab: str, merges: str, *, unk_token: str |None = None) -> "BPE":
        """
        Load a BPE from the legacy vocab.json + merges.txt format.
        """

class Model:
    """
    Base class for all models. Not constructible from Python; holds the actual
    Rust model by value (no sharing with the Tokenizer — assignment copies).
    """
    def __repr__(self, /) -> str: ...

@final
class Unigram(Model):
    def __new__(cls, /) -> Unigram: ...

@final
class WordLevel(Model):
    def __new__(cls, /, *, unk_token: str = ...) -> WordLevel: ...

@final
class WordPiece(Model):
    def __new__(cls, /, *, unk_token: str = ..., continuing_subword_prefix: str = ..., max_input_chars_per_word: int = 100) -> WordPiece: ...
