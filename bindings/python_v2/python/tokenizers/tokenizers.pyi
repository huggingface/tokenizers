"""
Tokenizers backed by Rust.

`Tokenizer.from_file` loads a `tokenizer.json`, `encode` and `encode_batch` turn text into
`Encoding`s, `decode` turns ids back into text. `Padding` says how encodings are padded.
"""

from collections.abc import Sequence
from numpy import integer, uint32
from numpy.typing import NDArray
from os import PathLike
from typing import Any, Final, Literal, final

__version__: Final[str]

@final
class Encoding:
    """
    One encoded text: `ids`, `type_ids` and `attention_mask`, the fields the pipeline computes.

    The pipeline only stores `type_ids` and `attention_mask` when a post-processor or padding
    set them. When it did not, every token is of type 0 and attended to, so that is what the
    two arrays report.

    Each field reads as a read-only numpy array over the encoding's own memory. Nothing is
    copied, and the array keeps the encoding alive for as long as the array exists.
    """
    def __eq__(self, value: object, /) -> bool: ...
    def __len__(self, /) -> int:
        """
        The number of tokens, padding included.
        """
    def __ne__(self, value: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...
    @property
    def attention_mask(self, /) -> NDArray[uint32]:
        """
        1 for every token, 0 for padding.
        """
    @property
    def ids(self, /) -> NDArray[uint32]:
        """
        The id of each token.
        """
    @property
    def type_ids(self, /) -> NDArray[uint32]:
        """
        The type id of each token, 0 unless a post-processor or padding set it.
        """

@final
class Padding:
    """
    How a `Tokenizer` pads what it encodes.

    Args:
        direction: `"right"` (the default) or `"left"`, the side the padding goes on.
        pad_id: The id of the padding token.
        pad_type_id: The type id of the padding token.
        pad_token: The text of the padding token.
        length: Pads every encoding to exactly this many tokens. `None` pads each batch to its
            longest item.
        pad_to_multiple_of: Rounds the padded length up to a multiple of this.

    The defaults are those of the released `Tokenizer.enable_padding`.
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
    ) -> Padding: ...
    def __repr__(self, /) -> str: ...
    @property
    def direction(self, /) -> Literal["left", "right"]:
        """
        `"left"` or `"right"`.
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
    A tokenizer loaded from a `tokenizer.json`.

    `encode` and `encode_batch` turn text into `Encoding`s, `decode` turns ids back into text.
    `padding` is the only attribute that can be changed after `from_file`.
    """
    def __repr__(self, /) -> str: ...
    def decode(self, /, ids: Sequence[int] | NDArray[integer[Any]], skip_special_tokens: bool = True) -> str:
        """
        Turns ids back into text.

        Args:
            ids: The ids to decode, a numpy array or any sequence of ints.
            skip_special_tokens: Whether special tokens are left out of the text.
        """
    def encode(self, /, text: str, add_special_tokens: bool = True) -> Encoding:
        """
        Encodes one text.

        Args:
            text: The text to encode.
            add_special_tokens: Whether the post-processor adds its special tokens, such as
                `[CLS]` and `[SEP]`.
        """
    def encode_batch(self, /, texts: Sequence[str], add_special_tokens: bool = True) -> list[Encoding]:
        """
        Encodes every text in parallel. The encodings come back in input order.

        Args:
            texts: The texts to encode.
            add_special_tokens: Whether the post-processor adds its special tokens, such as
                `[CLS]` and `[SEP]`.
        """
    @staticmethod
    def from_file(path: str | PathLike[str], padding: Padding | None = None) -> Tokenizer:
        """
        Loads a `tokenizer.json`. Files written by older `tokenizers` versions are upgraded on
        the way in, so anything already on disk loads.

        Args:
            path: The file to read.
            padding: Replaces the padding the file declares. `None` keeps the file's, which for
                most files means no padding.
        """
    @property
    def padding(self, /) -> Padding | None:
        """
        The padding applied to every encode, or `None`. Assign `None` to switch padding off.
        """
    @padding.setter
    def padding(self, /, padding: Padding | None) -> None: ...
