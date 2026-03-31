"""
Trainers Module
"""

from collections.abc import Sequence
from tokenizers import AddedToken
from typing import Any, final

@final
class BpeTrainer(Trainer):
    """
    Trainer capable of training a BPE model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.

        max_token_length (:obj:`int`, `optional`):
            Prevents creating tokens longer than the specified size.
            This can help with reducing polluting your vocabulary with
            highly repetitive tokens like `======` for wikipedia

    Example::

        >>> from tokenizers.models import BPE
        >>> from tokenizers.trainers import BpeTrainer
        >>> trainer = BpeTrainer(
        ...     vocab_size=30000,
        ...     special_tokens=["<unk>", "<s>", "</s>"],
        ...     min_frequency=2,
        ... )
        >>> tokenizer = Tokenizer(BPE())
        >>> tokenizer.train(["path/to/corpus.txt"], trainer)
    """
    def __new__(cls, /, **kwargs) -> BpeTrainer: ...
    @property
    def continuing_subword_prefix(self, /) -> str | None: ...
    @continuing_subword_prefix.setter
    def continuing_subword_prefix(self, /, prefix: str | None) -> None: ...
    @property
    def end_of_word_suffix(self, /) -> str | None: ...
    @end_of_word_suffix.setter
    def end_of_word_suffix(self, /, suffix: str | None) -> None: ...
    def get_word_count(self, /) -> int:
        """
        Get the number of unique words after feeding the corpus
        """
    @property
    def initial_alphabet(self, /) -> list[str]: ...
    @initial_alphabet.setter
    def initial_alphabet(self, /, alphabet: Sequence[str]) -> None: ...
    @property
    def limit_alphabet(self, /) -> int | None: ...
    @limit_alphabet.setter
    def limit_alphabet(self, /, limit: int | None) -> None: ...
    @property
    def max_token_length(self, /) -> int | None: ...
    @max_token_length.setter
    def max_token_length(self, /, limit: int | None) -> None: ...
    @property
    def min_frequency(self, /) -> int: ...
    @min_frequency.setter
    def min_frequency(self, /, freq: int) -> None: ...
    @property
    def progress_format(self, /) -> str:
        """
        Get the progress output format ("indicatif", "json", or "silent")
        """
    @progress_format.setter
    def progress_format(self, /, format: str) -> None:
        """
        Set the progress output format ("indicatif", "json", or "silent")
        """
    @property
    def show_progress(self, /) -> bool: ...
    @show_progress.setter
    def show_progress(self, /, show_progress: bool) -> None: ...
    @property
    def special_tokens(self, /) -> list[AddedToken]: ...
    @special_tokens.setter
    def special_tokens(self, /, special_tokens: list) -> None: ...
    @property
    def vocab_size(self, /) -> int: ...
    @vocab_size.setter
    def vocab_size(self, /, vocab_size: int) -> None: ...

class Trainer:
    """
    Base class for all trainers

    This class is not supposed to be instantiated directly. Instead, any implementation of a
    Trainer will return an instance of this class when instantiated.
    """
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str: ...

@final
class UnigramTrainer(Trainer):
    """
    Trainer capable of training a Unigram model

    Args:
        vocab_size (:obj:`int`):
            The size of the final vocabulary, including all tokens and alphabet.

        show_progress (:obj:`bool`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.

        initial_alphabet (:obj:`List[str]`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        shrinking_factor (:obj:`float`):
            The shrinking factor used at each step of the training to prune the
            vocabulary.

        unk_token (:obj:`str`):
            The token used for out-of-vocabulary tokens.

        max_piece_length (:obj:`int`):
            The maximum length of a given token.

        n_sub_iterations (:obj:`int`):
            The number of iterations of the EM algorithm to perform before
            pruning the vocabulary.

    Example::

        >>> from tokenizers.models import Unigram
        >>> from tokenizers.trainers import UnigramTrainer
        >>> trainer = UnigramTrainer(
        ...     vocab_size=8000,
        ...     special_tokens=["<unk>", "<s>", "</s>"],
        ...     unk_token="<unk>",
        ... )
        >>> tokenizer = Tokenizer(Unigram())
        >>> tokenizer.train(["path/to/corpus.txt"], trainer)
    """
    def __new__(cls, /, **kwargs) -> UnigramTrainer: ...
    @property
    def initial_alphabet(self, /) -> list[str]: ...
    @initial_alphabet.setter
    def initial_alphabet(self, /, alphabet: Sequence[str]) -> None: ...
    @property
    def show_progress(self, /) -> bool: ...
    @show_progress.setter
    def show_progress(self, /, show_progress: bool) -> None: ...
    @property
    def special_tokens(self, /) -> list[AddedToken]: ...
    @special_tokens.setter
    def special_tokens(self, /, special_tokens: list) -> None: ...
    @property
    def vocab_size(self, /) -> int: ...
    @vocab_size.setter
    def vocab_size(self, /, vocab_size: int) -> None: ...

@final
class WordLevelTrainer(Trainer):
    """
    Trainer capable of training a WordLevel model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`):
            A list of special tokens the model should know of.

    Example::

        >>> from tokenizers.models import WordLevel
        >>> from tokenizers.trainers import WordLevelTrainer
        >>> trainer = WordLevelTrainer(
        ...     vocab_size=10000,
        ...     special_tokens=["<unk>"],
        ...     min_frequency=1,
        ... )
        >>> tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        >>> tokenizer.train(["path/to/corpus.txt"], trainer)
    """
    def __new__(cls, /, **kwargs) -> WordLevelTrainer: ...
    @property
    def min_frequency(self, /) -> int: ...
    @min_frequency.setter
    def min_frequency(self, /, freq: int) -> None: ...
    @property
    def show_progress(self, /) -> bool: ...
    @show_progress.setter
    def show_progress(self, /, show_progress: bool) -> None: ...
    @property
    def special_tokens(self, /) -> list[AddedToken]: ...
    @special_tokens.setter
    def special_tokens(self, /, special_tokens: list) -> None: ...
    @property
    def vocab_size(self, /) -> int: ...
    @vocab_size.setter
    def vocab_size(self, /, vocab_size: int) -> None: ...

@final
class WordPieceTrainer(Trainer):
    """
    Trainer capable of training a WordPiece model

    Args:
        vocab_size (:obj:`int`, `optional`):
            The size of the final vocabulary, including all tokens and alphabet.

        min_frequency (:obj:`int`, `optional`):
            The minimum frequency a pair should have in order to be merged.

        show_progress (:obj:`bool`, `optional`):
            Whether to show progress bars while training.

        special_tokens (:obj:`List[Union[str, AddedToken]]`, `optional`):
            A list of special tokens the model should know of.

        limit_alphabet (:obj:`int`, `optional`):
            The maximum different characters to keep in the alphabet.

        initial_alphabet (:obj:`List[str]`, `optional`):
            A list of characters to include in the initial alphabet, even
            if not seen in the training dataset.
            If the strings contain more than one character, only the first one
            is kept.

        continuing_subword_prefix (:obj:`str`, `optional`):
            A prefix to be used for every subword that is not a beginning-of-word.

        end_of_word_suffix (:obj:`str`, `optional`):
            A suffix to be used for every subword that is a end-of-word.

    Example::

        >>> from tokenizers.models import WordPiece
        >>> from tokenizers.trainers import WordPieceTrainer
        >>> trainer = WordPieceTrainer(
        ...     vocab_size=30000,
        ...     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        ... )
        >>> tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        >>> tokenizer.train(["path/to/corpus.txt"], trainer)
    """
    def __new__(cls, /, **kwargs) -> WordPieceTrainer: ...
    @property
    def continuing_subword_prefix(self, /) -> str | None: ...
    @continuing_subword_prefix.setter
    def continuing_subword_prefix(self, /, prefix: str | None) -> None: ...
    @property
    def end_of_word_suffix(self, /) -> str | None: ...
    @end_of_word_suffix.setter
    def end_of_word_suffix(self, /, suffix: str | None) -> None: ...
    @property
    def initial_alphabet(self, /) -> list[str]: ...
    @initial_alphabet.setter
    def initial_alphabet(self, /, alphabet: Sequence[str]) -> None: ...
    @property
    def limit_alphabet(self, /) -> int | None: ...
    @limit_alphabet.setter
    def limit_alphabet(self, /, limit: int | None) -> None: ...
    @property
    def min_frequency(self, /) -> int: ...
    @min_frequency.setter
    def min_frequency(self, /, freq: int) -> None: ...
    @property
    def show_progress(self, /) -> bool: ...
    @show_progress.setter
    def show_progress(self, /, show_progress: bool) -> None: ...
    @property
    def special_tokens(self, /) -> list[AddedToken]: ...
    @special_tokens.setter
    def special_tokens(self, /, special_tokens: list) -> None: ...
    @property
    def vocab_size(self, /) -> int: ...
    @vocab_size.setter
    def vocab_size(self, /, vocab_size: int) -> None: ...
