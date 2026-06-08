"""
Models Module
"""

from collections.abc import Sequence
from tokenizers import Token
from typing import Any, final

@final
class BPE(Model):
    """
    An implementation of the BPE (Byte-Pair Encoding) algorithm

    Args:
        vocab (:obj:`Dict[str, int]`, `optional`):
            A dictionary of string keys and their ids :obj:`{"am": 0,...}`

        merges (:obj:`List[Tuple[str, str]]`, `optional`):
            A list of pairs of tokens (:obj:`Tuple[str, str]`) :obj:`[("a", "b"),...]`

        cache_capacity (:obj:`int`, `optional`):
            The number of words that the BPE cache can contain. The cache allows
            to speed-up the process by keeping the result of the merge operations
            for a number of words.

        dropout (:obj:`float`, `optional`):
            A float between 0 and 1 that represents the BPE dropout to use.

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

        continuing_subword_prefix (:obj:`str`, `optional`):
            The prefix to attach to subword units that don't represent a beginning of word.

        end_of_word_suffix (:obj:`str`, `optional`):
            The suffix to attach to subword units that represent an end of word.

        fuse_unk (:obj:`bool`, `optional`):
            Whether to fuse any subsequent unknown tokens into a single one

        byte_fallback (:obj:`bool`, `optional`):
            Whether to use spm byte-fallback trick (defaults to False)

        ignore_merges (:obj:`bool`, `optional`):
            Whether or not to match tokens with the vocab before using merges.

    Example::

        >>> from tokenizers.models import BPE
        >>> # Build an empty model (to be trained)
        >>> model = BPE(unk_token="<unk>")
        >>> # Load from vocabulary and merges files
        >>> model = BPE.from_file("vocab.json", "merges.txt")
    """
    def __new__(
        cls,
        /,
        vocab: dict[str, int] | str | None = None,
        merges: Sequence[tuple[str, str]] | str | None = None,
        **kwargs,
    ) -> BPE: ...
    def _clear_cache(self, /) -> "None":
        """
        Clears the internal cache
        """
    def _resize_cache(self, /, capacity: int) -> "None":
        """
        Resize the internal cache
        """
    @property
    def byte_fallback(self, /) -> bool: ...
    @byte_fallback.setter
    def byte_fallback(self, /, byte_fallback: bool) -> None: ...
    @property
    def continuing_subword_prefix(self, /) -> str | None: ...
    @continuing_subword_prefix.setter
    def continuing_subword_prefix(self, /, continuing_subword_prefix: str | None) -> None: ...
    @property
    def dropout(self, /) -> float | None: ...
    @dropout.setter
    def dropout(self, /, dropout: float | None) -> None: ...
    @property
    def end_of_word_suffix(self, /) -> str | None: ...
    @end_of_word_suffix.setter
    def end_of_word_suffix(self, /, end_of_word_suffix: str | None) -> None: ...
    @classmethod
    def from_file(cls, /, vocab: str, merges: str, **kwargs) -> "BPE":
        """
        Instantiate a BPE model from the given files.

        This method is roughly equivalent to doing::

           vocab, merges = BPE.read_file(vocab_filename, merges_filename)
           bpe = BPE(vocab, merges)

        If you don't need to keep the :obj:`vocab, merges` values lying around,
        this method is more optimized than manually calling
        :meth:`~tokenizers.models.BPE.read_file` to initialize a :class:`~tokenizers.models.BPE`

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

            merges (:obj:`str`):
                The path to a :obj:`merges.txt` file

        Returns:
            :class:`~tokenizers.models.BPE`: An instance of BPE loaded from these files
        """
    @property
    def fuse_unk(self, /) -> bool: ...
    @fuse_unk.setter
    def fuse_unk(self, /, fuse_unk: bool) -> None: ...
    @property
    def ignore_merges(self, /) -> bool: ...
    @ignore_merges.setter
    def ignore_merges(self, /, ignore_merges: bool) -> None: ...
    @staticmethod
    def read_file(vocab: str, merges: str) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """
        Read a :obj:`vocab.json` and a :obj:`merges.txt` files

        This method provides a way to read and parse the content of these files,
        returning the relevant data structures. If you want to instantiate some BPE models
        from memory, this method gives you the expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

            merges (:obj:`str`):
                The path to a :obj:`merges.txt` file

        Returns:
            A :obj:`Tuple` with the vocab and the merges:
                The vocabulary and merges loaded into memory
        """
    @property
    def unk_token(self, /) -> str | None: ...
    @unk_token.setter
    def unk_token(self, /, unk_token: str | None) -> None: ...

class Model:
    """
    Base class for all models

    The model represents the actual tokenization algorithm. This is the part that
    will contain and manage the learned vocabulary.

    This class cannot be constructed directly. Please use one of the concrete models.
    """
    def __getstate__(self, /) -> Any: ...
    def __new__(cls, /) -> "Model": ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str: ...
    def get_trainer(self, /) -> Any:
        """
        Get the associated :class:`~tokenizers.trainers.Trainer`

        Retrieve the :class:`~tokenizers.trainers.Trainer` associated to this
        :class:`~tokenizers.models.Model`.

        Returns:
            :class:`~tokenizers.trainers.Trainer`: The Trainer used to train this model
        """
    def id_to_token(self, /, id: int) -> str | None:
        """
        Get the token associated to an ID

        Args:
            id (:obj:`int`):
                An ID to convert to a token

        Returns:
            :obj:`str`: The token associated to the ID
        """
    def save(self, /, folder: str, prefix: str | None = None, name: str | None = None) -> "list[str]":
        """
        Save the current model

        Save the current model in the given folder, using the given prefix for the various
        files that will get created.
        Any file with the same name that already exists in this folder will be overwritten.

        Args:
            folder (:obj:`str`):
                The path to the target folder in which to save the various files

            prefix (:obj:`str`, `optional`):
                An optional prefix, used to prefix each file name

        Returns:
            :obj:`List[str]`: The list of saved files
        """
    def token_to_id(self, /, token: str) -> int | None:
        """
        Get the ID associated to a token

        Args:
            token (:obj:`str`):
                A token to convert to an ID

        Returns:
            :obj:`int`: The ID associated to the token
        """
    def tokenize(self, /, sequence: str) -> list[Token]:
        """
        Tokenize a sequence

        Args:
            sequence (:obj:`str`):
                A sequence to tokenize

        Returns:
            A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
        """

@final
class Unigram(Model):
    """
    An implementation of the Unigram algorithm

    The Unigram algorithm is a subword tokenization algorithm based on unigram language
    models, as used in SentencePiece. It learns a vocabulary by starting with a large
    initial vocabulary and iteratively pruning it using the EM algorithm.

    Args:
        vocab (:obj:`List[Tuple[str, float]]`, `optional`):
            A list of vocabulary items and their log-probability scores,
            e.g. ``[("am", -0.2442), ...]``. If not provided, an empty model is created.

        unk_id (:obj:`int`, `optional`):
            The index of the unknown token in the vocabulary list.

        byte_fallback (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use SentencePiece byte fallback for characters not in the vocabulary.

        alpha (:obj:`float`, `optional`):
            A float between 0 and 1 that represents the smoothing parameter (temperature) to use.

        nbest_size (:obj:`int`, `optional`):
            An integer greater than 0 that represents the maximum number of best paths to consider.
            If not set, it samples from the full lattice (i.e. all valid subword segmentations).

    Example::

        >>> from tokenizers.models import Unigram
        >>> # Build an empty model (to be trained)
        >>> model = Unigram()
        >>> # Build from a vocabulary list
        >>> vocab = [("<unk>", 0.0), ("hello", -1.0), ("world", -1.5)]
        >>> model = Unigram(vocab=vocab, unk_id=0)
    """
    def __new__(
        cls,
        /,
        vocab: Sequence[tuple[str, float]] | None = None,
        unk_id: int | None = None,
        byte_fallback: bool | None = None,
        alpha: float | None = None,
        nbest_size: int | None = None,
    ) -> Unigram: ...
    def _clear_cache(self, /) -> "None":
        """
        Clears the internal cache
        """
    def _resize_cache(self, /, capacity: int) -> "None":
        """
        Resize the internal cache
        """
    @property
    def alpha(self, /) -> float | None: ...
    @alpha.setter
    def alpha(self, /, alpha: float | None) -> None: ...
    @property
    def nbest_size(self, /) -> int | None: ...
    @nbest_size.setter
    def nbest_size(self, /, nbest_size: int | None) -> None: ...

@final
class WordLevel(Model):
    """
    An implementation of the WordLevel algorithm

    Most simple tokenizer model based on mapping tokens to their corresponding id.

    Args:
        vocab (:obj:`str`, `optional`):
            A dictionary of string keys and their ids :obj:`{"am": 0,...}`

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

    Example::

        >>> from tokenizers.models import WordLevel
        >>> # Build from a vocabulary dictionary
        >>> vocab = {"hello": 0, "world": 1, "<unk>": 2}
        >>> model = WordLevel(vocab=vocab, unk_token="<unk>")
        >>> # Load from file
        >>> model = WordLevel.from_file("vocab.json", unk_token="<unk>")
    """
    def __new__(cls, /, vocab: dict[str, int] | str | None = None, unk_token: str | None = None) -> WordLevel: ...
    @classmethod
    def from_file(cls, /, vocab: str, unk_token: str | None = None) -> "WordLevel":
        """
        Instantiate a WordLevel model from the given file

        This method is roughly equivalent to doing::

            vocab = WordLevel.read_file(vocab_filename)
            wordlevel = WordLevel(vocab)

        If you don't need to keep the :obj:`vocab` values lying around, this method is
        more optimized than manually calling :meth:`~tokenizers.models.WordLevel.read_file` to
        initialize a :class:`~tokenizers.models.WordLevel`

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            :class:`~tokenizers.models.WordLevel`: An instance of WordLevel loaded from file
        """
    @staticmethod
    def read_file(vocab: str) -> dict[str, int]:
        """
        Read a :obj:`vocab.json`

        This method provides a way to read and parse the content of a vocabulary file,
        returning the relevant data structures. If you want to instantiate some WordLevel models
        from memory, this method gives you the expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.json` file

        Returns:
            :obj:`Dict[str, int]`: The vocabulary as a :obj:`dict`
        """
    @property
    def unk_token(self, /) -> str: ...
    @unk_token.setter
    def unk_token(self, /, unk_token: str) -> None: ...

@final
class WordPiece(Model):
    """
    An implementation of the WordPiece algorithm

    Args:
        vocab (:obj:`Dict[str, int]`, `optional`):
            A dictionary of string keys and their ids :obj:`{"am": 0,...}`

        unk_token (:obj:`str`, `optional`):
            The unknown token to be used by the model.

        max_input_chars_per_word (:obj:`int`, `optional`):
            The maximum number of characters to authorize in a single word.

    Example::

        >>> from tokenizers.models import WordPiece
        >>> # Build an empty model (to be trained)
        >>> model = WordPiece(unk_token="[UNK]")
        >>> # Load from a vocabulary file
        >>> model = WordPiece.from_file("vocab.txt")
    """
    def __new__(cls, /, vocab: dict[str, int] | str | None = None, **kwargs) -> WordPiece: ...
    @property
    def continuing_subword_prefix(self, /) -> str: ...
    @continuing_subword_prefix.setter
    def continuing_subword_prefix(self, /, continuing_subword_prefix: str) -> None: ...
    @classmethod
    def from_file(cls, /, vocab: str, **kwargs) -> "WordPiece":
        """
        Instantiate a WordPiece model from the given file

        This method is roughly equivalent to doing::

            vocab = WordPiece.read_file(vocab_filename)
            wordpiece = WordPiece(vocab)

        If you don't need to keep the :obj:`vocab` values lying around, this method is
        more optimized than manually calling :meth:`~tokenizers.models.WordPiece.read_file` to
        initialize a :class:`~tokenizers.models.WordPiece`

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.txt` file

        Returns:
            :class:`~tokenizers.models.WordPiece`: An instance of WordPiece loaded from file
        """
    @property
    def max_input_chars_per_word(self, /) -> int: ...
    @max_input_chars_per_word.setter
    def max_input_chars_per_word(self, /, max: int) -> None: ...
    @staticmethod
    def read_file(vocab: str) -> dict[str, int]:
        """
        Read a :obj:`vocab.txt` file

        This method provides a way to read and parse the content of a standard `vocab.txt`
        file as used by the WordPiece Model, returning the relevant data structures. If you
        want to instantiate some WordPiece models from memory, this method gives you the
        expected input from the standard files.

        Args:
            vocab (:obj:`str`):
                The path to a :obj:`vocab.txt` file

        Returns:
            :obj:`Dict[str, int]`: The vocabulary as a :obj:`dict`
        """
    @property
    def unk_token(self, /) -> str: ...
    @unk_token.setter
    def unk_token(self, /, unk_token: str) -> None: ...
