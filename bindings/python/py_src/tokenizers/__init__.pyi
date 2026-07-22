from collections.abc import Coroutine
from typing import Any

import numpy as np
import numpy.typing as npt

"""
Fast tokenizers built on the pipeline encode path. Start with `Tokenizer`.
"""

from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.trainers import Trainer
from _typeshed import Incomplete
from collections.abc import Sequence
from os import PathLike
from typing import Any, Final, final
__version__: Final[str]

@final
class AddedToken:
    """
    A token added to the vocabulary after training, with options for how it
    is matched in text: `single_word` only matches when it stands alone (not
    inside a word); `lstrip`/`rstrip` also swallow the whitespace before/after
    it; `normalized` matches against normalized instead of raw text (defaults
    to the opposite of `special`); `special` marks template tokens like "<s>"
    that decoding should be able to skip.
    """
    def __new__(cls, /, content: str, *, single_word: bool = False, lstrip: bool = False, rstrip: bool = False, normalized: bool |None = None, special: bool = False) -> AddedToken: ...
    def __repr__(self, /) -> str: ...
    @property
    def content(self, /) -> str: ...
    @property
    def lstrip(self, /) -> bool: ...
    @property
    def normalized(self, /) -> bool: ...
    @property
    def rstrip(self, /) -> bool: ...
    @property
    def single_word(self, /) -> bool: ...
    @property
    def special(self, /) -> bool: ...

@final
class Tokenizer:
    """
    A tokenizer: a model plus its optional normalizer and pre-tokenizer.
    
    Create one from a model (`Tokenizer(models.BPE())`), a file
    (`Tokenizer.from_file`), or the Hub (`Tokenizer.from_pretrained`).
    Changes — assigning components, training, adding tokens — apply to the
    serializable definition; encoding runs a compiled pipeline that is rebuilt
    automatically after any change. A definition the pipeline cannot run
    raises `TokenizersError` at that point, with the reason.
    """
    def __new__(cls, /, model: Model) -> Tokenizer:
        """
        Create an untrained tokenizer from a model.
        """
    def __reduce__(self, /) -> tuple[Any, tuple[bytes]]: ...
    def __repr__(self, /) -> str: ...
    def add_special_tokens(self, /, tokens: Sequence[str |AddedToken]) -> int:
        """
        Add special tokens ("<s>", "[CLS]", …) to the vocabulary. Same as
        `add_tokens`, but every token is marked `special`. Returns how many
        were actually new.
        """
    def add_tokens(self, /, tokens: Sequence[str |AddedToken]) -> int:
        """
        Add tokens to the vocabulary and match them in the input text from now
        on. Plain strings match with default options; pass `AddedToken` to
        control matching. Returns how many were actually new.
        """
    def async_encode(self, /, text: Any, *, add_special_tokens: bool = True) -> "Coroutine[Any, Any, npt.NDArray[np.uint32]]":
        """
        Awaitable `encode`: same arguments and result, run in a worker thread
        (`asyncio.to_thread`) so the event loop stays free. The thread releases
        the interpreter lock while Rust encodes, so encodes genuinely overlap.
        """
    def async_encode_batch(self, /, texts: Any, *, add_special_tokens: bool = True) -> "Coroutine[Any, Any, list[npt.NDArray[np.uint32]]]":
        """
        Awaitable `encode_batch`: same arguments and result, run in a worker
        thread (`asyncio.to_thread`) so the event loop stays free while the
        batch encodes on Rust threads.
        """
    def decode(self, /, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """
        Not implemented yet: decoding is not part of the encode pipeline.
        """
    def encode(self, /, text: str, *, add_special_tokens: bool = True) -> "npt.NDArray[np.uint32]":
        """
        Encode `text` into token ids.
        
        Runs entirely outside the interpreter lock and returns a `numpy.uint32`
        array backed by the Rust output buffer (no copy).
        """
    def encode_batch(self, /, texts: Sequence[str], *, add_special_tokens: bool = True) -> "list[npt.NDArray[np.uint32]]":
        """
        Encode a batch of texts, in parallel across Rust threads (respects
        `TOKENIZERS_PARALLELISM`), without holding the interpreter lock.
        Input strings are borrowed, not copied; each output is a `numpy.uint32`
        array backed by its Rust buffer.
        """
    @staticmethod
    def from_buffer(buffer: Sequence[int]) -> "Tokenizer":
        """
        Load a tokenizer from the bytes of a `tokenizer.json` file.
        """
    @staticmethod
    def from_file(path: str |PathLike[str]) -> "Tokenizer":
        """
        Load a tokenizer from a `tokenizer.json` file.
        """
    @staticmethod
    def from_pretrained(identifier: str, *, revision: str = ..., token: str |None = None) -> "Tokenizer":
        """
        Download `tokenizer.json` from a model on the Hugging Face Hub (requires
        the `huggingface_hub` package) and load it.
        """
    def get_vocab(self, /, *, with_added_tokens: bool = True) -> dict[str, int]:
        """
        The whole vocabulary as a dict. This copies every entry; prefer
        `token_to_id` for lookups.
        """
    def get_vocab_size(self, /, *, with_added_tokens: bool = True) -> int:
        """
        Number of entries in the vocabulary. `with_added_tokens=False` counts
        only what the model was trained with.
        """
    def id_to_token(self, /, id: int) -> str |None:
        """
        The token behind `id`, or None if the id is out of range.
        """
    @property
    def model(self, /) -> Model:
        """
        The model in use by this tokenizer (a copy: reassign to change it).
        """
    @model.setter
    def model(self, /, model: Model) -> None: ...
    @property
    def normalizer(self, /) -> Normalizer |None:
        """
        The optional normalizer in use by this tokenizer (a copy: reassign to
        change it).
        """
    @normalizer.setter
    def normalizer(self, /, normalizer: Normalizer |None) -> None: ...
    @property
    def pre_tokenizer(self, /) -> PreTokenizer |None:
        """
        The optional pre-tokenizer in use by this tokenizer (a copy: reassign
        to change it).
        """
    @pre_tokenizer.setter
    def pre_tokenizer(self, /, pre_tokenizer: PreTokenizer |None) -> None: ...
    def save(self, /, path: str |PathLike[str], *, pretty: bool = True) -> None:
        """
        Save the tokenizer definition to a `tokenizer.json` file.
        """
    def to_str(self, /, *, pretty: bool = False) -> str:
        """
        Serialize the tokenizer definition as a `tokenizer.json` string.
        """
    def token_to_id(self, /, token: str) -> int |None:
        """
        The id of `token`, or None if it is not in the vocabulary.
        """
    def train(self, /, files: Sequence[str], *, trainer: Trainer |None = None) -> None:
        """
        Train the model's vocabulary on text files (one sequence per line).
        Without a `trainer`, the model's default trainer is used.
        """
    def train_from_iterator(self, /, iterator: Any, *, trainer: Trainer |None = None) -> None:
        """
        Train the model's vocabulary from any iterator of `str`. Without a
        `trainer`, the model's default trainer is used.
        
        The interpreter lock is only re-acquired to refill an internal buffer
        (256 sequences at a time); the training itself runs multi-threaded in
        Rust with the lock released.
        """

def __getattr__(name: str) -> Incomplete: ...

class TokenizersError(Exception): ...
