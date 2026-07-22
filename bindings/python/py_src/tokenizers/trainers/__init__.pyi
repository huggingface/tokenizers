"""
Recipes for learning a vocabulary from text.
"""

from tokenizers import AddedToken
from collections.abc import Sequence
from typing import final

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
