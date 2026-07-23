"""
Recipes for learning a vocabulary from text.
"""

from tokenizers import AddedToken, Tokenizer
from collections.abc import Sequence
from typing import Any, final

@final
class BpeTrainer(Trainer):
    """
    Learns a BPE vocabulary: keeps merging the most frequent pair until
    `vocab_size` is reached, ignoring pairs seen fewer than `min_frequency`
    times. `special_tokens` get the first ids. `limit_alphabet` caps how many
    distinct characters are kept; `initial_alphabet` forces characters in even
    if the data never shows them; `max_token_length` caps merged token length.
    """
    def __new__(cls, /, *, vocab_size: int = 30000, min_frequency: int = 0, special_tokens: Sequence[str |AddedToken] = ..., limit_alphabet: int |None = None, initial_alphabet: Sequence[str] = ..., continuing_subword_prefix: str |None = None, end_of_word_suffix: str |None = None, max_token_length: int |None = None, show_progress: bool = True) -> BpeTrainer: ...

@final
class ParityBpeTrainer:
    """
    Learns a BPE vocabulary from several languages at once, keeping the
    compression rate fair across them (parity-aware BPE, Foroutan et al. 2026).
    Unlike the other trainers it is not passed to `Tokenizer.train`: call its
    own `train_from_iterator` with one iterator of text per language.
    `variant` picks how merges are selected — "base" enforces parity on every
    merge, "window" relaxes it to every `window_size` merges. Fairness is
    measured on the per-language `dev_iterators`, or against target
    compression `ratio`s when no dev data is given. `num_merges` replaces
    `vocab_size`; the remaining knobs match `BpeTrainer`.
    """
    def __new__(cls, /, *, num_merges: int = 32000, variant: str = "base", min_frequency: int = 0, ratio: Sequence[float] |None = None, global_merges: int = 0, window_size: int = 100, alpha: float = 2.0, total_symbols: bool = False, special_tokens: Sequence[str |AddedToken] = ..., show_progress: bool = True, limit_alphabet: int |None = None, initial_alphabet: Sequence[str] = ..., continuing_subword_prefix: str |None = None, end_of_word_suffix: str |None = None, max_token_length: int |None = None) -> ParityBpeTrainer: ...
    def __repr__(self, /) -> str: ...
    def train_from_iterator(self, /, tokenizer: Tokenizer, train_iterators: Sequence[Any], *, dev_iterators: Sequence[Any] = ..., ratio: Sequence[float] |None = None) -> None:
        """
        Train `tokenizer`'s vocabulary with parity-aware BPE. `train_iterators`
        holds one iterator of `str` per language; `dev_iterators` (same length)
        drives the fairness measurement, or pass per-language `ratio` targets
        instead. The tokenizer's normalizer and pre-tokenizer are applied to
        every sequence, and its model is replaced by the trained BPE.
        """

class Trainer:
    """
    Base class for all trainers.
    
    A trainer is the recipe for learning a model's vocabulary from text; pass
    one to `Tokenizer.train` or `train_from_iterator`. Trainers are plain
    configuration values — training copies them and writes nothing back.
    """
    def __repr__(self, /) -> str: ...

@final
class UnigramTrainer(Trainer):
    """
    Learns a Unigram vocabulary: starts from a large candidate set and prunes
    it by `shrinking_factor` each round until `vocab_size` pieces remain.
    `unk_token` names the fallback piece for unknown characters.
    """
    def __new__(cls, /, *, vocab_size: int = 8000, special_tokens: Sequence[str |AddedToken] = ..., initial_alphabet: Sequence[str] = ..., unk_token: str |None = None, shrinking_factor: float = 0.75, max_piece_length: int = 16, n_sub_iterations: int = 2, show_progress: bool = True) -> UnigramTrainer: ...

@final
class WordLevelTrainer(Trainer):
    """
    Learns a WordLevel vocabulary: the `vocab_size` most frequent words,
    keeping only those seen at least `min_frequency` times.
    """
    def __new__(cls, /, *, vocab_size: int = 30000, min_frequency: int = 0, special_tokens: Sequence[str |AddedToken] = ..., show_progress: bool = True) -> WordLevelTrainer: ...

@final
class WordPieceTrainer(Trainer):
    """
    Learns a WordPiece vocabulary. Same knobs as `BpeTrainer`, plus the
    continuation prefix ("##" by default).
    """
    def __new__(cls, /, *, vocab_size: int = 30000, min_frequency: int = 0, special_tokens: Sequence[str |AddedToken] = ..., limit_alphabet: int |None = None, initial_alphabet: Sequence[str] = ..., continuing_subword_prefix: str = ..., end_of_word_suffix: str |None = None, show_progress: bool = True) -> WordPieceTrainer: ...

__all__ = ["BpeTrainer", "ParityBpeTrainer", "Trainer", "UnigramTrainer", "WordLevelTrainer", "WordPieceTrainer"]
