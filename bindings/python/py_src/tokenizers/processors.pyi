import tokenizers
import typing

class BertProcessing:
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(cls, /, sep: typing.Any, cls_token: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def cls(self, /) -> typing.Any: ...
    @cls.setter
    def cls(self, /, cls: typing.Any) -> typing.Any: ...
    @property
    def sep(self, /) -> typing.Any: ...
    @sep.setter
    def sep(self, /, sep: typing.Any) -> typing.Any: ...

class ByteLevel:
    def __new__(
        cls,
        /,
        add_prefix_space: bool | None = None,
        trim_offsets: bool | None = None,
        use_regex: bool | None = None,
        **_kwargs,
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def add_prefix_space(self, /) -> bool: ...
    @add_prefix_space.setter
    def add_prefix_space(self, /, add_prefix_space: bool) -> None: ...
    @property
    def trim_offsets(self, /) -> bool: ...
    @trim_offsets.setter
    def trim_offsets(self, /, trim_offsets: bool) -> None: ...
    @property
    def use_regex(self, /) -> bool: ...
    @use_regex.setter
    def use_regex(self, /, use_regex: bool) -> None: ...

class PostProcessor:
    def __getstate__(self, /) -> typing.Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    def num_special_tokens_to_add(self, /, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.

        Args:
            is_pair (:obj:`bool`):
                Whether the input would be a pair of sequences

        Returns:
            :obj:`int`: The number of tokens to add
        """
        ...
    def process(
        self,
        /,
        encoding: tokenizers.Encoding,
        pair: tokenizers.Encoding | None = None,
        add_special_tokens: bool = True,
    ) -> tokenizers.Encoding:
        """
        Post-process the given encodings, generating the final one

        Args:
            encoding (:class:`~tokenizers.Encoding`):
                The encoding for the first sequence

            pair (:class:`~tokenizers.Encoding`, `optional`):
                The encoding for the pair sequence

            add_special_tokens (:obj:`bool`):
                Whether to add the special tokens

        Return:
            :class:`~tokenizers.Encoding`: The final encoding
        """
        ...

class RobertaProcessing:
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(
        cls, /, sep: typing.Any, cls_token: typing.Any, trim_offsets: bool = True, add_prefix_space: bool = True
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def add_prefix_space(self, /) -> bool: ...
    @add_prefix_space.setter
    def add_prefix_space(self, /, add_prefix_space: bool) -> None: ...
    @property
    def cls(self, /) -> typing.Any: ...
    @cls.setter
    def cls(self, /, cls: typing.Any) -> typing.Any: ...
    @property
    def sep(self, /) -> typing.Any: ...
    @sep.setter
    def sep(self, /, sep: typing.Any) -> typing.Any: ...
    @property
    def trim_offsets(self, /) -> bool: ...
    @trim_offsets.setter
    def trim_offsets(self, /, trim_offsets: bool) -> None: ...

class Sequence:
    def __getitem__(self, /, index: int) -> typing.Any:
        """Return self[key]."""
        ...
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(cls, /, processors_py: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __setitem__(self, /, index: int, value: typing.Any) -> typing.Any:
        """Set self[key] to value."""
        ...

class TemplateProcessing:
    def __new__(
        cls,
        /,
        single: typing.Any | None = None,
        pair: typing.Any | None = None,
        special_tokens: typing.Any | None = None,
    ) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def single(self, /) -> str: ...
    @single.setter
    def single(self, /, single: typing.Any) -> typing.Any: ...
