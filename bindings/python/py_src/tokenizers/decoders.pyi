import tokenizers
import tokenizers.decoders
import typing

class BPEDecoder:
    def __new__(cls, /, suffix: str = ...) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def suffix(self, /) -> str: ...
    @suffix.setter
    def suffix(self, /, suffix: str) -> None: ...

class ByteFallback:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class ByteLevel:
    def __new__(cls, /, **_kwargs) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class CTC:
    def __new__(cls, /, pad_token: str = ..., word_delimiter_token: str = ..., cleanup: bool = True) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def cleanup(self, /) -> bool: ...
    @cleanup.setter
    def cleanup(self, /, cleanup: bool) -> None: ...
    @property
    def pad_token(self, /) -> str: ...
    @pad_token.setter
    def pad_token(self, /, pad_token: str) -> None: ...
    @property
    def word_delimiter_token(self, /) -> str: ...
    @word_delimiter_token.setter
    def word_delimiter_token(self, /, word_delimiter_token: str) -> None: ...

class DecodeStream:
    def __new__(cls, /, ids: typing.Any | None = None, skip_special_tokens: bool | None = False) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    def step(self, /, tokenizer: tokenizers.Tokenizer, id: typing.Any) -> typing.Any:
        """
        Streaming decode step

        Args:
            tokenizer (:class:`~tokenizers.Tokenizer`):
               The tokenizer to use for decoding
           id (:obj:`int` or `List[int]`):
              The next token id or list of token ids to add to the stream


        Returns:
            :obj:`Optional[str]`: The next decoded string chunk, or None if not enough
                tokens have been provided yet.
        """
        ...

class Decoder:
    def __getstate__(self, /) -> typing.Any: ...
    def __repr__(self, /) -> str:
        """Return repr(self)."""
        ...
    def __setstate__(self, /, state: typing.Any) -> typing.Any: ...
    def __str__(self, /) -> str:
        """Return str(self)."""
        ...
    @staticmethod
    def custom(decoder: typing.Any) -> tokenizers.decoders.Decoder: ...
    def decode(self, /, tokens: typing.Any) -> str:
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """
        ...

class Fuse:
    def __new__(cls, /) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Metaspace:
    def __new__(cls, /, replacement: str = "▁", prepend_scheme: str = ..., split: bool = True) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def prepend_scheme(self, /) -> str: ...
    @prepend_scheme.setter
    def prepend_scheme(self, /, prepend_scheme: str) -> typing.Any: ...
    @property
    def replacement(self, /) -> str: ...
    @replacement.setter
    def replacement(self, /, replacement: str) -> None: ...
    @property
    def split(self, /) -> bool: ...
    @split.setter
    def split(self, /, split: bool) -> None: ...

class Replace:
    def __new__(cls, /, pattern: str | tokenizers.Regex, content: str) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Sequence:
    def __getnewargs__(self, /) -> typing.Any: ...
    def __new__(cls, /, decoders_py: typing.Any) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...

class Strip:
    def __new__(cls, /, content: str = " ", left: int = 0, right: int = 0) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def content(self, /) -> str: ...
    @content.setter
    def content(self, /, content: str) -> None: ...
    @property
    def start(self, /) -> int: ...
    @start.setter
    def start(self, /, start: int) -> None: ...
    @property
    def stop(self, /) -> int: ...
    @stop.setter
    def stop(self, /, stop: int) -> None: ...

class WordPiece:
    def __new__(cls, /, prefix: str = ..., cleanup: bool = True) -> None:
        """Create and return a new object.  See help(type) for accurate signature."""
        ...
    @property
    def cleanup(self, /) -> bool: ...
    @cleanup.setter
    def cleanup(self, /, cleanup: bool) -> None: ...
    @property
    def prefix(self, /) -> str: ...
    @prefix.setter
    def prefix(self, /, prefix: str) -> None: ...
