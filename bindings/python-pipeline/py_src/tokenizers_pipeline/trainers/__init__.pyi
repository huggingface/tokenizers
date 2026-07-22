from tokenizers_pipeline import AddedToken
from collections.abc import Sequence
from typing import final

@final
class BpeTrainer(Trainer):
    def __new__(cls, /, *, vocab_size: int = 30000, min_frequency: int = 0, special_tokens: Sequence[str |AddedToken] = ..., limit_alphabet: int |None = None, initial_alphabet: Sequence[str] = ..., continuing_subword_prefix: str |None = None, end_of_word_suffix: str |None = None, max_token_length: int |None = None, show_progress: bool = True) -> BpeTrainer: ...

class Trainer:
    """
    Base class for all trainers. A trainer is a plain configuration value:
    `Tokenizer.train*` copies it, no state is shared or written back.
    """
    def __repr__(self, /) -> str: ...

@final
class UnigramTrainer(Trainer):
    def __new__(cls, /, *, vocab_size: int = 8000, special_tokens: Sequence[str |AddedToken] = ..., initial_alphabet: Sequence[str] = ..., unk_token: str |None = None, shrinking_factor: float = 0.75, max_piece_length: int = 16, n_sub_iterations: int = 2, show_progress: bool = True) -> UnigramTrainer: ...

@final
class WordLevelTrainer(Trainer):
    def __new__(cls, /, *, vocab_size: int = 30000, min_frequency: int = 0, special_tokens: Sequence[str |AddedToken] = ..., show_progress: bool = True) -> WordLevelTrainer: ...

@final
class WordPieceTrainer(Trainer):
    def __new__(cls, /, *, vocab_size: int = 30000, min_frequency: int = 0, special_tokens: Sequence[str |AddedToken] = ..., limit_alphabet: int |None = None, initial_alphabet: Sequence[str] = ..., continuing_subword_prefix: str = ..., end_of_word_suffix: str |None = None, show_progress: bool = True) -> WordPieceTrainer: ...
