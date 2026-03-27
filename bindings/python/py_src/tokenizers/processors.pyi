from _typeshed import Incomplete
from collections.abc import Sequence as Sequence2
from tokenizers import Encoding
from typing import Any, final

@final
class BertProcessing(PostProcessor):
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, sep: tuple[str, int], cls_token: tuple[str, int]) -> BertProcessing:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def cls(self, /) -> tuple: ...
    @cls.setter
    def cls(self, /, cls: tuple) -> None: ...
    @property
    def sep(self, /) -> tuple: ...
    @sep.setter
    def sep(self, /, sep: tuple) -> None: ...

@final
class ByteLevel(PostProcessor):
    def __new__(
        cls,
        /,
        add_prefix_space: bool | None = None,
        trim_offsets: bool | None = None,
        use_regex: bool | None = None,
        **_kwargs,
    ) -> ByteLevel:
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
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: Any) -> None: ...
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
        self, /, encoding: Encoding, pair: Encoding | None = None, add_special_tokens: bool = True
    ) -> "Encoding":
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

@final
class RobertaProcessing(PostProcessor):
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(
        cls,
        /,
        sep: tuple[str, int],
        cls_token: tuple[str, int],
        trim_offsets: bool = True,
        add_prefix_space: bool = True,
    ) -> RobertaProcessing:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def add_prefix_space(self, /) -> bool: ...
    @add_prefix_space.setter
    def add_prefix_space(self, /, add_prefix_space: bool) -> None: ...
    @property
    def cls(self, /) -> tuple: ...
    @cls.setter
    def cls(self, /, cls: tuple) -> None: ...
    @property
    def sep(self, /) -> tuple: ...
    @sep.setter
    def sep(self, /, sep: tuple) -> None: ...
    @property
    def trim_offsets(self, /) -> bool: ...
    @trim_offsets.setter
    def trim_offsets(self, /, trim_offsets: bool) -> None: ...

@final
class Sequence(PostProcessor):
    def __getitem__(self, /, index: int) -> Any:
        """Return self[key]."""
        ...
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, processors_py: list) -> Sequence:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def __setitem__(self, /, index: int, value: Any) -> None:
        """Set self[key] to value."""
        ...

@final
class TemplateProcessing(PostProcessor):
    def __new__(
        cls,
        /,
        single: Incomplete | None = None,
        pair: Incomplete | None = None,
        special_tokens: Sequence2[Incomplete] | None = None,
    ) -> TemplateProcessing:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def single(self, /) -> str: ...
    @single.setter
    def single(self, /, single: Incomplete) -> None: ...
