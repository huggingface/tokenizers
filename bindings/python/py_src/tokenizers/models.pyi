import typing

class BPE:
    def __new__(
        cls, /, vocab: typing.Any | str | None = None, merges: typing.Any | str | None = None, **kwargs
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def _clear_cache(self, /) -> None:
        """Clears the internal cache"""
        ...
    def _resize_cache(self, /, capacity: int) -> None:
        """Resize the internal cache"""
        ...
    @property
    def byte_fallback(self, /) -> bool: ...
    @byte_fallback.setter
    def byte_fallback(self, /, byte_fallback: bool) -> None: ...
    @property
    def continuing_subword_prefix(self, /) -> typing.Any: ...
    @continuing_subword_prefix.setter
    def continuing_subword_prefix(self, /, continuing_subword_prefix: str | None) -> None: ...
    @property
    def dropout(self, /) -> typing.Any: ...
    @dropout.setter
    def dropout(self, /, dropout: float | None) -> None: ...
    @property
    def end_of_word_suffix(self, /) -> typing.Any: ...
    @end_of_word_suffix.setter
    def end_of_word_suffix(self, /, end_of_word_suffix: str | None) -> None: ...
    @classmethod
    def from_file(cls, /, vocab: str, merges: str, **kwargs) -> BPE:
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
        ...
    @property
    def fuse_unk(self, /) -> bool: ...
    @fuse_unk.setter
    def fuse_unk(self, /, fuse_unk: bool) -> None: ...
    @property
    def ignore_merges(self, /) -> bool: ...
    @ignore_merges.setter
    def ignore_merges(self, /, ignore_merges: bool) -> None: ...
    @staticmethod
    def read_file(vocab: str, merges: str) -> typing.Any:
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
        ...
    @property
    def unk_token(self, /) -> typing.Any: ...
    @unk_token.setter
    def unk_token(self, /, unk_token: str | None) -> None: ...

class Model:
    def __getstate__(self, /) -> typing.Any: ...
    def __new__(cls, /) -> Model:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    def get_trainer(self, /) -> typing.Any:
        """
        Get the associated :class:`~tokenizers.trainers.Trainer`

        Retrieve the :class:`~tokenizers.trainers.Trainer` associated to this
        :class:`~tokenizers.models.Model`.

        Returns:
            :class:`~tokenizers.trainers.Trainer`: The Trainer used to train this model
        """
        ...
    def id_to_token(self, /, id: int) -> typing.Any:
        """
        Get the token associated to an ID

        Args:
            id (:obj:`int`):
                An ID to convert to a token

        Returns:
            :obj:`str`: The token associated to the ID
        """
        ...
    def save(self, /, folder: str, prefix: str | None = None, name: str | None = None) -> list[str]:
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
        ...
    def token_to_id(self, /, token: str) -> typing.Any:
        """
        Get the ID associated to a token

        Args:
            token (:obj:`str`):
                A token to convert to an ID

        Returns:
            :obj:`int`: The ID associated to the token
        """
        ...
    def tokenize(self, /, sequence: str) -> typing.Any:
        """
        Tokenize a sequence

        Args:
            sequence (:obj:`str`):
                A sequence to tokenize

        Returns:
            A :obj:`List` of :class:`~tokenizers.Token`: The generated tokens
        """
        ...

class Unigram:
    def __new__(
        cls, /, vocab: typing.Any | None = None, unk_id: int | None = None, byte_fallback: bool | None = None
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def _clear_cache(self, /) -> None:
        """Clears the internal cache"""
        ...
    def _resize_cache(self, /, capacity: int) -> None:
        """Resize the internal cache"""
        ...

class WordLevel:
    def __new__(cls, /, vocab: typing.Any | str | None = None, unk_token: str | None = None) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @classmethod
    def from_file(cls, /, vocab: str, unk_token: str | None = None) -> WordLevel:
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
        ...
    @staticmethod
    def read_file(vocab: str) -> typing.Any:
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
        ...
    @property
    def unk_token(self, /) -> str: ...
    @unk_token.setter
    def unk_token(self, /, unk_token: str) -> None: ...

class WordPiece:
    def __new__(cls, /, vocab: typing.Any | str | None = None, **kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def continuing_subword_prefix(self, /) -> str: ...
    @continuing_subword_prefix.setter
    def continuing_subword_prefix(self, /, continuing_subword_prefix: str) -> None: ...
    @classmethod
    def from_file(cls, /, vocab: str, **kwargs) -> WordPiece:
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
        ...
    @property
    def max_input_chars_per_word(self, /) -> int: ...
    @max_input_chars_per_word.setter
    def max_input_chars_per_word(self, /, max: int) -> None: ...
    @staticmethod
    def read_file(vocab: str) -> typing.Any:
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
        ...
    @property
    def unk_token(self, /) -> str: ...
    @unk_token.setter
    def unk_token(self, /, unk_token: str) -> None: ...
