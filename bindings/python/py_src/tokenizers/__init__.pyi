"""
Fast tokenizers: turn text into the token ids models consume.
Start with `Tokenizer`.
"""

from collections.abc import Coroutine
from typing import Any

import numpy as np
import numpy.typing as npt

from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.trainers import Trainer
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
class Encoding:
    """
    The result of encoding one sequence: token ids plus the masks and metadata
    a model consumes. The fields are derived from the ids on access, so an
    `Encoding` costs the same to produce as a bare id array — `Tokenizer.encode`
    runs exactly the work `encode_ids` does.
    
    `encode` only produces an `Encoding` for a single sequence with no
    post-processor-inserted special tokens (it raises otherwise), so the
    segment, attention, special-token and sequence values are constant: one
    sequence numbered 0, nothing padded, nothing special. Anything that would
    need per-token provenance the pipeline does not compute — word ids and
    character offsets — raises rather than returning a plausible-looking guess.
    """
    def __len__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...
    @property
    def attention_mask(self, /) -> list[int]:
        """
        Attention mask, one entry per token: all 1 (nothing padded).
        """
    def char_to_token(self, /, char_pos: int, sequence_index: int = 0) -> int |None: ...
    def char_to_word(self, /, char_pos: int, sequence_index: int = 0) -> int |None: ...
    @property
    def ids(self, /) -> list[int]:
        """
        The token ids, as a list.
        """
    def ids_array(self, /) -> "npt.NDArray[np.uint32]":
        """
        The token ids as a `numpy.uint32` array. This copies; for the copy-free
        array use `Tokenizer.encode_ids`, which hands ownership of the buffer
        straight to numpy.
        """
    @property
    def n_sequences(self, /) -> int:
        """
        Number of sequences in this encoding: always 1.
        """
    @property
    def offsets(self, /) -> list[tuple[int, int]]:
        """
        Character span per token — not available: the pipeline does not track
        offsets yet.
        """
    @property
    def sequence_ids(self, /) -> list[int |None]:
        """
        The sequence each token belongs to: all 0 (single sequence).
        """
    @property
    def special_tokens_mask(self, /) -> list[int]:
        """
        Special-tokens mask, one entry per token: all 0 (no post-processing).
        """
    def token_to_chars(self, /, token_index: int) -> tuple[int, int] |None: ...
    def token_to_sequence(self, /, token_index: int) -> int |None:
        """
        The sequence a token belongs to (0), or None for an out-of-range index.
        """
    def token_to_word(self, /, token_index: int) -> int |None: ...
    @property
    def tokens(self, /) -> list[str]:
        """
        The token strings behind the ids.
        """
    @property
    def type_ids(self, /) -> list[int]:
        """
        Segment id per token: all 0 (single sequence).
        """
    @property
    def word_ids(self, /) -> list[int |None]:
        """
        Word id per token — not available: the pipeline does not emit word
        boundaries yet.
        """
    def word_to_chars(self, /, word_index: int, sequence_index: int = 0) -> tuple[int, int] |None: ...
    def word_to_tokens(self, /, word_index: int, sequence_index: int = 0) -> tuple[int, int] |None: ...

@final
class EncodingBatch:
    """
    The result of encoding a batch: a sequence of `Encoding`s. Index it
    (`batch[0]`) or iterate it.
    """
    def __getitem__(self, /, index: int) -> Encoding: ...
    def __len__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...

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
    def async_encode(self, /, text: Any, *, add_special_tokens: bool = True) -> "Coroutine[Any, Any, Encoding]":
        """
        Awaitable `encode`: same arguments and result, run in a worker thread.
        """
    def async_encode_batch(self, /, texts: Any, *, add_special_tokens: bool = True) -> "Coroutine[Any, Any, EncodingBatch]":
        """
        Awaitable `encode_batch`: same arguments and result, run in a worker
        thread while the batch encodes on Rust threads.
        """
    def async_encode_batch_ids(self, /, texts: Any, *, add_special_tokens: bool = True) -> "Coroutine[Any, Any, list[npt.NDArray[np.uint32]]]":
        """
        Awaitable `encode_batch_ids`: same arguments and result, run in a
        worker thread (`asyncio.to_thread`) so the event loop stays free while
        the batch encodes on Rust threads.
        """
    def async_encode_ids(self, /, text: Any, *, add_special_tokens: bool = True) -> "Coroutine[Any, Any, npt.NDArray[np.uint32]]":
        """
        Awaitable `encode_ids`: same arguments and result, run in a worker
        thread (`asyncio.to_thread`) so the event loop stays free. The thread
        releases the interpreter lock while Rust encodes, so encodes genuinely
        overlap.
        """
    def decode(self, /, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        """
        Not implemented yet: decoding is not part of the encode pipeline.
        """
    def encode(self, /, text: str, *, add_special_tokens: bool = True) -> "Encoding":
        """
        Encode `text` into an `Encoding`: token ids plus the masks and metadata
        a model consumes. Same encode work as `encode_ids` (GIL released, no
        copies); the `Encoding` wraps the ids and derives its fields on access.
        """
    def encode_batch(self, /, texts: Sequence[str], *, add_special_tokens: bool = True) -> "EncodingBatch":
        """
        Encode a batch of texts into an `EncodingBatch`, in parallel across Rust
        threads (respects `TOKENIZERS_PARALLELISM`), without holding the
        interpreter lock. The batch version of `encode`.
        """
    def encode_batch_ids(self, /, texts: Sequence[str], *, add_special_tokens: bool = True) -> "list[npt.NDArray[np.uint32]]":
        """
        Encode a batch of texts, in parallel across Rust threads (respects
        `TOKENIZERS_PARALLELISM`), without holding the interpreter lock.
        Input strings are borrowed, not copied; each output is a `numpy.uint32`
        array backed by its Rust buffer.
        """
    def encode_ids(self, /, text: str, *, add_special_tokens: bool = True) -> "npt.NDArray[np.uint32]":
        """
        Encode `text` into token ids.
        
        Runs entirely outside the interpreter lock and returns a `numpy.uint32`
        array backed by the Rust output buffer (no copy). For ids plus the
        masks and metadata models consume, use `encode`, which returns an
        `Encoding` from the same work.
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


class TokenizersError(Exception): ...

from tokenizers import models as models
from tokenizers import normalizers as normalizers
from tokenizers import pre_tokenizers as pre_tokenizers
from tokenizers import trainers as trainers

__all__ = ["AddedToken", "Encoding", "EncodingBatch", "Tokenizer", "TokenizersError", "__version__", "models", "normalizers", "pre_tokenizers", "trainers"]
