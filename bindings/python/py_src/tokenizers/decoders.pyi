"""
Decoders Module
"""

from _typeshed import Incomplete
from collections.abc import Sequence as Sequence2
from tokenizers import Regex, Tokenizer
from typing import Any, final

@final
class BPEDecoder(Decoder):
    """
    BPEDecoder Decoder

    Args:
        suffix (:obj:`str`, `optional`, defaults to :obj:`</w>`):
            The suffix that was used to characterize an end-of-word. This suffix will
            be replaced by whitespaces during the decoding

    Example::

        >>> from tokenizers.decoders import BPEDecoder
        >>> decoder = BPEDecoder()
        >>> decoder.decode(["Hello</w>", "world</w>"])
        'Hello world'
    """
    def __new__(cls, /, suffix: str = ...) -> BPEDecoder: ...
    @property
    def suffix(self, /) -> str: ...
    @suffix.setter
    def suffix(self, /, suffix: str) -> None: ...

@final
class ByteFallback(Decoder):
    """
    ByteFallback Decoder

    ByteFallback is a decoder that handles tokens representing raw bytes in the
    ``<0xNN>`` format (e.g., ``<0x61>`` for the byte ``0x61`` = ``'a'``). It converts
    such tokens to their corresponding bytes and attempts to decode the resulting byte
    sequence as UTF-8. This is used in LLaMA/SentencePiece models that use byte fallback
    for unknown characters. Inconvertible byte tokens are replaced with the Unicode
    replacement character (U+FFFD).

    Example::

        >>> from tokenizers.decoders import ByteFallback, Fuse, Sequence
        >>> decoder = Sequence([ByteFallback(), Fuse()])
        >>> decoder.decode(["<0x48>", "<0x65>", "<0x6C>", "<0x6C>", "<0x6F>"])
        'Hello'
    """
    def __new__(cls, /) -> ByteFallback: ...

@final
class ByteLevel(Decoder):
    """
    ByteLevel Decoder

    This decoder is to be used in tandem with the
    :class:`~tokenizers.pre_tokenizers.ByteLevel` pre-tokenizer. It reverses the
    byte-to-unicode mapping applied during pre-tokenization, converting the special
    Unicode characters back into the original bytes to reconstruct the original string.

    Example::

        >>> from tokenizers.decoders import ByteLevel
        >>> decoder = ByteLevel()
        >>> decoder.decode(["ĠHello", "Ġworld"])
        ' Hello world'
    """
    def __new__(cls, /, **_kwargs) -> ByteLevel: ...

@final
class CTC(Decoder):
    """
    CTC Decoder

    Args:
        pad_token (:obj:`str`, `optional`, defaults to :obj:`<pad>`):
            The pad token used by CTC to delimit a new token.
        word_delimiter_token (:obj:`str`, `optional`, defaults to :obj:`|`):
            The word delimiter token. It will be replaced by a <space>
        cleanup (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to cleanup some tokenization artifacts.
            Mainly spaces before punctuation, and some abbreviated english forms.

    Example::

        >>> from tokenizers.decoders import CTC
        >>> decoder = CTC()
        >>> decoder.decode(["h", "e", "e", "<pad>", "l", "l", "o", "|", "w", "o", "r", "l", "d"])
        'hello world'
    """
    def __new__(cls, /, pad_token: str = ..., word_delimiter_token: str = ..., cleanup: bool = True) -> CTC: ...
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

@final
class DecodeStream:
    """
    Provides incremental decoding of token IDs as they are generated, yielding
    decoded text chunks as soon as they are available.

    Unlike batch decoding, streaming decode is designed for use with autoregressive
    generation — tokens arrive one at a time and the decoder needs to handle
    multi-byte sequences (e.g., UTF-8 characters split across token boundaries) and
    byte-fallback tokens gracefully.

    The decoder internally buffers tokens until it can produce a valid UTF-8 string
    chunk, then yields that chunk and advances its internal state. This means
    individual calls to :meth:`~tokenizers.decoders.DecodeStream.step` may return
    :obj:`None` when the current token completes a partial sequence that cannot yet
    be decoded.

    Args:
        skip_special_tokens (:obj:`bool`, defaults to :obj:`False`):
            Whether to skip special tokens (e.g. ``[CLS]``, ``[SEP]``, ``<s>``) when
            decoding.

    Example::

        >>> from tokenizers import Tokenizer
        >>> from tokenizers.decoders import DecodeStream
        >>> tokenizer = Tokenizer.from_pretrained("gpt2")
        >>> stream = DecodeStream(skip_special_tokens=True)
        >>> # Simulate streaming token-by-token generation
        >>> token_ids = tokenizer.encode("Hello, streaming world!").ids
        >>> for token_id in token_ids:
        ...     chunk = stream.step(tokenizer, token_id)
        ...     if chunk is not None:
        ...         print(chunk, end="", flush=True)
    """
    def __copy__(self, /) -> DecodeStream: ...
    def __deepcopy__(self, /, _memo: dict) -> DecodeStream: ...
    def __new__(
        cls, /, ids: Sequence2[int] | None = None, skip_special_tokens: bool | None = False
    ) -> DecodeStream: ...
    def step(self, /, tokenizer: Tokenizer, id: Incomplete) -> str | None:
        """
        Add the next token ID (or list of IDs) to the stream and return the next
        decoded text chunk if one is available.

        Because some characters span multiple tokens (e.g. multi-byte UTF-8
        sequences or byte-fallback tokens), this method may return :obj:`None`
        when the provided token does not yet complete a decodable unit. Callers
        should simply continue feeding tokens until a non-:obj:`None` value is
        returned.

        Args:
            tokenizer (:class:`~tokenizers.Tokenizer`):
                The tokenizer whose decoder pipeline will be used.

            id (:obj:`int` or :obj:`List[int]`):
                The next token ID, or a list of token IDs to append to the stream.

        Returns:
            :obj:`Optional[str]`: The next decoded text chunk if enough tokens have
                accumulated, or :obj:`None` if more tokens are still needed.
        """

class Decoder:
    """
    Base class for all decoders

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a Decoder will return an instance of this class when instantiated.
    """
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str: ...
    @staticmethod
    def custom(decoder: Any) -> Decoder: ...
    def decode(self, /, tokens: Sequence2[str]) -> str:
        """
        Decode the given list of tokens to a final string

        Args:
            tokens (:obj:`List[str]`):
                The list of tokens to decode

        Returns:
            :obj:`str`: The decoded string
        """

@final
class Fuse(Decoder):
    """
    Fuse Decoder

    Fuse simply concatenates every token into a single string without any separator.
    This is typically the last step in a decoder chain when other decoders need to
    operate on individual tokens before they are joined together.

    Example::

        >>> from tokenizers.decoders import Fuse
        >>> decoder = Fuse()
        >>> decoder.decode(["Hello", ",", " ", "world", "!"])
        'Hello, world!'
    """
    def __new__(cls, /) -> Fuse: ...

@final
class Metaspace(Decoder):
    """
    Metaspace Decoder

    Args:
        replacement (:obj:`str`, `optional`, defaults to :obj:`▁`):
            The replacement character. Must be exactly one character. By default we
            use the `▁` (U+2581) meta symbol (Same as in SentencePiece).

        prepend_scheme (:obj:`str`, `optional`, defaults to :obj:`"always"`):
            Whether to add a space to the first word if there isn't already one. This
            lets us treat `hello` exactly like `say hello`.
            Choices: "always", "never", "first". First means the space is only added on the first
            token (relevant when special tokens are used or other pre_tokenizer are used).

    Example::

        >>> from tokenizers.decoders import Metaspace
        >>> decoder = Metaspace()
        >>> decoder.decode(["▁Hello", "▁my", "▁friend"])
        'Hello my friend'
    """
    def __new__(cls, /, replacement: str = "▁", prepend_scheme: str = ..., split: bool = True) -> Metaspace: ...
    @property
    def prepend_scheme(self, /) -> str: ...
    @prepend_scheme.setter
    def prepend_scheme(self, /, prepend_scheme: str) -> None: ...
    @property
    def replacement(self, /) -> str: ...
    @replacement.setter
    def replacement(self, /, replacement: str) -> None: ...
    @property
    def split(self, /) -> bool: ...
    @split.setter
    def split(self, /, split: bool) -> None: ...

@final
class Replace(Decoder):
    """
    Replace Decoder

    This decoder is to be used in tandem with the
    :class:`~tokenizers.normalizers.Replace` normalizer or a similar replace operation.
    It reverses a string replacement by substituting the replacement content back
    with the original pattern.

    Args:
        pattern (:obj:`str` or :class:`~tokenizers.Regex`):
            The pattern that was used as the replacement target during encoding.

        content (:obj:`str`):
            The string to replace each match of the pattern with during decoding.

    Example::

        >>> from tokenizers.decoders import Replace
        >>> decoder = Replace("▁", " ")
        >>> decoder.decode(["▁Hello", "▁world"])
        ' Hello world'
    """
    def __new__(cls, /, pattern: str | Regex, content: str) -> Replace: ...

@final
class Sequence(Decoder):
    """
    Sequence Decoder

    Chains multiple decoders together, applying them in order. Each decoder in the
    sequence processes the output of the previous one, allowing complex decoding
    pipelines to be built from simpler components.

    Args:
        decoders (:obj:`List[Decoder]`):
            The list of decoders to chain together.

    Example::

        >>> from tokenizers.decoders import ByteFallback, Fuse, Metaspace, Sequence
        >>> decoder = Sequence([ByteFallback(), Fuse(), Metaspace()])
        >>> decoder.decode(["▁Hello", "▁world"])
        'Hello world'
    """
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, decoders_py: list) -> Sequence: ...

@final
class Strip(Decoder):
    """
    Strip Decoder

    Strips a given number of occurrences of a character from the left and/or right
    side of each token. This is useful for removing padding characters or special
    prefix/suffix markers added during tokenization.

    Args:
        content (:obj:`str`, defaults to :obj:`" "`):
            The character to strip from each token.

        left (:obj:`int`, defaults to :obj:`0`):
            The number of occurrences of :obj:`content` to remove from the left
            side of each token.

        right (:obj:`int`, defaults to :obj:`0`):
            The number of occurrences of :obj:`content` to remove from the right
            side of each token.

    Example::

        >>> from tokenizers.decoders import Strip
        >>> decoder = Strip(content="▁", left=1)
        >>> decoder.decode(["▁Hello", "▁world"])
        'Hello world'
    """
    def __new__(cls, /, content: str = " ", left: int = 0, right: int = 0) -> Strip: ...
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

@final
class WordPiece(Decoder):
    """
    WordPiece Decoder

    Args:
        prefix (:obj:`str`, `optional`, defaults to :obj:`##`):
            The prefix to use for subwords that are not a beginning-of-word

        cleanup (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation,
            and some abbreviated english forms.

    Example::

        >>> from tokenizers.decoders import WordPiece
        >>> decoder = WordPiece()
        >>> decoder.decode(["Hello", ",", "##world", "!"])
        'Hello, world!'
    """
    def __new__(cls, /, prefix: str = ..., cleanup: bool = True) -> WordPiece: ...
    @property
    def cleanup(self, /) -> bool: ...
    @cleanup.setter
    def cleanup(self, /, cleanup: bool) -> None: ...
    @property
    def prefix(self, /) -> str: ...
    @prefix.setter
    def prefix(self, /, prefix: str) -> None: ...
