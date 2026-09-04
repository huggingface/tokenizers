from collections.abc import Sequence
from os import PathLike
from pathlib import Path
from typing import Any, Final, Literal, final

from numpy import integer, uint32
from numpy.typing import NDArray

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
    def __reduce__(self, /) -> tuple[Any, tuple[list[int], list[int], list[int]]]:
        """
        Pickle rebuilds an `Encoding` by calling `_unpickle` with these arguments.
        """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def _unpickle(ids: Sequence[int], type_ids: Sequence[int], attention_mask: Sequence[int]) -> Encoding:
        """
        Unpickles an `Encoding`
        """
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
    def __reduce__(self, /) -> tuple[type, tuple[Literal["left", "right"], int, int, str, int | None, int | None]]:
        """
        Pickle rebuilds a `Padding` by calling the class with these constructor arguments.
        """
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
    def __reduce__(self, /) -> tuple[Any, tuple[str, Padding | None]]:
        """
        Pickle rebuilds a `Tokenizer` by calling `_unpickle` with these arguments.
        """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def _unpickle(json: str, padding: Padding | None) -> Tokenizer:
        """
        Unpickles a `Tokenizer`
        """
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
    @staticmethod
    def from_pretrained(
        identifier: str,
        revision: str = "main",
        token: str | bool | None = None,
        *,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        subfolder: str | None = None,
        padding: Padding | None = None,
    ) -> Tokenizer:
        """
        Instantiate a new `Tokenizer` from an existing file on the Hugging Face Hub.

        Defers downloading and caching to `huggingface_hub.hf_hub_download`.

        Args:
            identifier (`str`):
                The *model id* of a repo hosted on huggingface.co that contains a `tokenizer.json` file,
                e.g., `"openai-community/gpt2"`.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id,
                since we use a git-based system for storing models and other artifacts on huggingface.co,
                so `revision` can be any identifier allowed by git.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the
                token generated when running `hf auth login`. If `False`, will send no token. If `None`,
                will use the stored token when there is one.
            cache_dir (`str` or `Path`, *optional*):
                Path to a directory in which the downloaded file should be cached if the standard cache
                should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to download the file again and override the cached version if it exists.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether or not to only rely on the cache and not to attempt to download anything.
            subfolder (`str`, *optional*):
                In case `tokenizer.json` is located inside a subfolder of the model repo on huggingface.co,
                specify it here.
            padding (`Padding`, *optional*):
                Replaces the padding configuration the file declares. `None` keeps the file's.

        Returns:
            `Tokenizer`: The tokenizer the file describes.

        Examples:

        ```python
        # Download tokenizer.json from huggingface.co and cache it.
        tokenizer = Tokenizer.from_pretrained("openai-community/gpt2")

        # Pin a revision: a branch name, a tag name or a commit id.
        tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")

        # A private or gated repo, with the token `hf auth login` stored.
        tokenizer = Tokenizer.from_pretrained("my-org/my-model", token=True)

        # Read the cache without contacting the Hub.
        tokenizer = Tokenizer.from_pretrained("openai-community/gpt2", local_files_only=True)
        ```
        """
