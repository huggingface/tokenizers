from collections.abc import Sequence
from numpy import integer, uint32
from numpy.typing import NDArray
from os import PathLike
from typing import Any, Final, Literal, final

__version__: Final[str]

@final
class Encoding:
    """
    Text encoded to token ids by a tokenizer.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __len__(self, /) -> int:
        """
        The number of tokens in the encoding
        """
    def __ne__(self, value: object, /) -> bool: ...
    def __reduce__(self, /) -> tuple[Any, tuple[list[int], list[int], list[int]]]: ...
    def __repr__(self, /) -> str: ...
    @property
    def attention_mask(self, /) -> list[int]:
        """
        Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
        A list of ints.
        """
    @property
    def attention_mask_array(self, /) -> NDArray[uint32]:
        """
        Attention mask when the encoding is padded: 1 for token ids, 0 for padding tokens.
        A read-only `uint32` numpy array, a view over the encoding, not a copy.
        """
    @property
    def ids(self, /) -> list[int]:
        """
        The id of each token, as a list of ints.
        """
    @property
    def ids_array(self, /) -> NDArray[uint32]:
        """
        The id of each token, as a read-only `uint32` numpy array.
        A view over the encoding, not a copy.
        """
    @property
    def type_ids(self, /) -> list[int]:
        """
        The type id of each token, as a list of ints.
        """
    @property
    def type_ids_array(self, /) -> NDArray[uint32]:
        """
        The type id of each token, as a read-only `uint32` numpy array.
        A view over the encoding, not a copy.
        """

@final
class Padding:
    """
    Padding parameters for Tokenizer.encode
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __hash__(self, /) -> int: ...
    def __ne__(self, value: object, /) -> bool: ...
    def __new__(
        cls,
        /,
        direction: Literal["left", "right"] = ...,
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
        length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> Padding:
        """
        Args:
            direction: `"right"` (the default) or `"left"`
                whether padding tokens are appended to the right or
                prepended to the left  of encoded tokens.
            pad_id: int
                The id of the padding token.
            pad_type_id: int
                The type id of the padding token.
            pad_token: str
                The text of the padding token.
            length: int (optional)
                Pads every encoding to exactly this many tokens. `None` pads each batch to its
                longest item.
            pad_to_multiple_of: int (optional)
                Rounds the padded length up to a multiple of this.
        """
    def __reduce__(self, /) -> tuple[Any, tuple[Literal["left", "right"], int, int, str, int | None, int | None]]: ...
    def __repr__(self, /) -> str: ...
    @property
    def direction(self, /) -> Literal["left", "right"]:
        """
        `"left"` or `"right"`.
        Whether padding tokens are appended to the right or prepended to the left  of encoded tokens.
        """
    @property
    def length(self, /) -> int | None:
        """
        The fixed length padded to, or `None` when padding to the longest item in the batch.
        """
    @property
    def pad_id(self, /) -> int:
        """
        The id of the padding token.
        """
    @property
    def pad_to_multiple_of(self, /) -> int | None:
        """
        The multiple the padded length is rounded up to, or `None`.
        """
    @property
    def pad_token(self, /) -> str:
        """
        The text of the padding token.
        """
    @property
    def pad_type_id(self, /) -> int:
        """
        The type id of the padding token.
        """

@final
class Tokenizer:
    """
    A tokenizer. Encodes text into token ids, and decodes token ids back into text.
    """
    def __reduce__(self, /) -> tuple[Any, tuple[str, Padding | None]]: ...
    def __repr__(self, /) -> str: ...
    def decode(self, /, ids: Sequence[int] | NDArray[integer[Any]], skip_special_tokens: bool = True) -> str:
        """
        Decodes token ids back into text

        Args:
            ids:
                The ids to decode, a numpy array or any sequence of ints.
            skip_special_tokens: bool
                Whether special tokens should not be added to the decoded text.

        Returns:
            str
        """
    def encode(self, /, text: str, add_special_tokens: bool = True) -> Encoding:
        """
        Encodes the given text to token ids.

        Args:
            text: str
                The text to encode.
            add_special_tokens: bool
                 Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.

        Returns:
            Encoding
        """
    def encode_batch(self, /, texts: Sequence[str], add_special_tokens: bool = True) -> list[Encoding]:
        """
        Encodes a batch of text.
        The encodings come back in input order.

        Args:
            texts: List[str]
                The batch of text to encode.
            add_special_tokens: bool
                Whether the post-processor adds its special tokens, such as `[CLS]` and `[SEP]`.

        Returns:
            List[Encoding]
        """
    @staticmethod
    def from_file(path: str | PathLike[str], padding: Padding | None = None) -> Tokenizer:
        """
        Loads a `tokenizer.json`.

        Args:
            path:
                The file to read.
            padding:
                Replaces the padding configuration the file declares.
                None` means use the file's padding configuration.
        """
    @property
    def padding(self, /) -> Padding | None:
        """
        The padding applied to every encode, or `None`.
        Assign `None` to switch padding off.
        """
    @padding.setter
    def padding(self, /, padding: Padding | None) -> None: ...

def _unpickle_encoding(ids: Sequence[int], type_ids: Sequence[int], attention_mask: Sequence[int]) -> Encoding:
    """
    Rebuilds the `Encoding` that `Encoding.__reduce__` took apart. Private, because an `Encoding`
    comes out of `Tokenizer.encode` and is not built by hand.
    """

def _unpickle_tokenizer(json: str, padding: Padding | None) -> Tokenizer:
    """
    Rebuilds the `Tokenizer` that `Tokenizer.__reduce__` took apart. Private, because reading a
    `tokenizer.json` from a string is not public API yet.

    `__reduce__` writes the canonical `tokenizer.json` the pipeline holds, so unlike
    `Tokenizer.from_file` this skips the conversion pass for older files.
    """
