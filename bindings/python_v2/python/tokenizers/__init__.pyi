from collections.abc import Sequence
from typing import Final, final

__version__: Final[str]

@final
class Encoding:
    """
    One encoded text: `ids`, `type_ids` and `attention_mask`, the fields the pipeline computes.

    The pipeline only stores `type_ids` and `attention_mask` when a post-processor or padding
    set them. When it did not, every token is of type 0 and attended to, so that is what the
    two lists report.
    """
    def __len__(self, /) -> int: ...
    @property
    def attention_mask(self, /) -> list[int]: ...
    @property
    def ids(self, /) -> list[int]: ...
    @property
    def type_ids(self, /) -> list[int]: ...

@final
class Padding:
    """
    How a `Tokenizer` pads what it encodes.

    `length=None` pads every batch to its longest item; a number pads to exactly that.
    The defaults are those of the released `Tokenizer.enable_padding`.
    """
    def __new__(
        cls,
        /,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = ...,
        length: int | None = None,
        pad_to_multiple_of: int | None = None,
    ) -> Padding: ...
    def __repr__(self, /) -> str: ...
    @property
    def direction(self, /) -> str:
        """
        `"left"` or `"right"`.
        """
    @property
    def length(self, /) -> int | None:
        """
        The fixed length padded to, or `None` when padding to the longest item in the batch.
        """
    @property
    def pad_id(self, /) -> int: ...
    @property
    def pad_to_multiple_of(self, /) -> int | None: ...
    @property
    def pad_token(self, /) -> str: ...
    @property
    def pad_type_id(self, /) -> int: ...

@final
class Tokenizer:
    """
    The pipeline encode path. `padding` is the only thing that can be changed after `from_file`.
    """
    def decode(self, /, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...
    def encode(self, /, text: str, add_special_tokens: bool = True) -> Encoding: ...
    def encode_batch(self, /, texts: Sequence[str], add_special_tokens: bool = True) -> list[Encoding]:
        """
        Encodes every text in parallel and returns the encodings in input order.
        """
    @staticmethod
    def from_file(path: str, padding: Padding | None = None) -> Tokenizer:
        """
        Read a `tokenizer.json`. The file is put through the legacy "1.0" -> canonical "2.0"
        upgrade first, so the tokenizers already on disk keep loading; `tk_serialize` itself
        only reads the canonical form.

        `padding` replaces the padding the file declares; `None` keeps the file's.
        """
    @property
    def padding(self, /) -> Padding | None:
        """
        The padding applied to every encode, or `None`. Assign `None` to switch padding off.
        """
    @padding.setter
    def padding(self, /, padding: Padding | None) -> None: ...
