"""
Processors Module
"""

from _typeshed import Incomplete
from collections.abc import Sequence as Sequence2
from tokenizers import Encoding
from typing import Any, final

@final
class BertProcessing(PostProcessor):
    """
    This post-processor takes care of adding the special tokens needed by
    a Bert model:

        - a SEP token
        - a CLS token

    Args:
        sep (:obj:`Tuple[str, int]`):
            A tuple with the string representation of the SEP token, and its id

        cls (:obj:`Tuple[str, int]`):
            A tuple with the string representation of the CLS token, and its id

    Example::

        >>> from tokenizers.processors import BertProcessing
        >>> processor = BertProcessing(("[SEP]", 102), ("[CLS]", 101))
        >>> processor.process(encoding)
        # Encoding with [CLS] at start and [SEP] at end
    """
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, sep: tuple[str, int], cls_token: tuple[str, int]) -> BertProcessing: ...
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
    """
    This post-processor takes care of trimming the offsets.

    By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
    want the offsets to include these whitespaces, then this PostProcessor must be used.

    Args:
        trim_offsets (:obj:`bool`):
            Whether to trim the whitespaces from the produced offsets.

        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If :obj:`True`, keeps the first token's offset as is. If :obj:`False`, increments
            the start of the first token's offset by 1. Only has an effect if :obj:`trim_offsets`
            is set to :obj:`True`.

    Example::

        >>> from tokenizers.processors import ByteLevel
        >>> processor = ByteLevel(trim_offsets=True)
        >>> # Offsets will be trimmed to exclude leading whitespace bytes
    """
    def __new__(
        cls,
        /,
        add_prefix_space: bool | None = None,
        trim_offsets: bool | None = None,
        use_regex: bool | None = None,
        **_kwargs,
    ) -> ByteLevel: ...
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
    """
    Base class for all post-processors

    This class is not supposed to be instantiated directly. Instead, any implementation of
    a PostProcessor will return an instance of this class when instantiated.
    """
    def __getstate__(self, /) -> Any: ...
    def __repr__(self, /) -> str: ...
    def __setstate__(self, /, state: Any) -> None: ...
    def __str__(self, /) -> str: ...
    def num_special_tokens_to_add(self, /, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.

        Args:
            is_pair (:obj:`bool`):
                Whether the input would be a pair of sequences

        Returns:
            :obj:`int`: The number of tokens to add
        """
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

@final
class RobertaProcessing(PostProcessor):
    """
    This post-processor takes care of adding the special tokens needed by
    a Roberta model:

        - a SEP token
        - a CLS token

    It also takes care of trimming the offsets.
    By default, the ByteLevel BPE might include whitespaces in the produced tokens. If you don't
    want the offsets to include these whitespaces, then this PostProcessor should be initialized
    with :obj:`trim_offsets=True`

    Args:
        sep (:obj:`Tuple[str, int]`):
            A tuple with the string representation of the SEP token, and its id

        cls (:obj:`Tuple[str, int]`):
            A tuple with the string representation of the CLS token, and its id

        trim_offsets (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to trim the whitespaces from the produced offsets.

        add_prefix_space (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether the add_prefix_space option was enabled during pre-tokenization. This
            is relevant because it defines the way the offsets are trimmed out.

    Example::

        >>> from tokenizers.processors import RobertaProcessing
        >>> processor = RobertaProcessing(("</s>", 2), ("<s>", 0))
        >>> processor.process(encoding)
        # Encoding with <s> at start and </s> at end
    """
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(
        cls,
        /,
        sep: tuple[str, int],
        cls_token: tuple[str, int],
        trim_offsets: bool = True,
        add_prefix_space: bool = True,
    ) -> RobertaProcessing: ...
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
    """
    Sequence Processor

    Chains multiple post-processors together, applying them in order. Each processor
    in the sequence processes the output of the previous one.

    Args:
        processors (:obj:`List[PostProcessor]`):
            The list of post-processors to chain together.

    Example::

        >>> from tokenizers.processors import BertProcessing, ByteLevel, Sequence
        >>> processor = Sequence([ByteLevel(trim_offsets=True), BertProcessing(("[SEP]", 102), ("[CLS]", 101))])
    """
    def __getitem__(self, /, index: int) -> Any: ...
    def __getnewargs__(self, /) -> tuple: ...
    def __new__(cls, /, processors_py: list) -> Sequence: ...
    def __setitem__(self, /, index: int, value: Any) -> None: ...

@final
class TemplateProcessing(PostProcessor):
    """
    Provides a way to specify templates in order to add the special tokens to each
    input sequence as relevant.

    Let's take :obj:`BERT` tokenizer as an example. It uses two special tokens, used to
    delimitate each sequence. :obj:`[CLS]` is always used at the beginning of the first
    sequence, and :obj:`[SEP]` is added at the end of both the first, and the pair
    sequences. The final result looks like this:

        - Single sequence: :obj:`[CLS] Hello there [SEP]`
        - Pair sequences: :obj:`[CLS] My name is Anthony [SEP] What is my name? [SEP]`

    With the type ids as following::

        [CLS]   ...   [SEP]   ...   [SEP]
          0      0      0      1      1

    You can achieve such behavior using a TemplateProcessing::

        TemplateProcessing(
            single="[CLS] $0 [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", 1), ("[SEP]", 0)],
        )

    In this example, each input sequence is identified using a ``$`` construct. This identifier
    lets us specify each input sequence, and the type_id to use. When nothing is specified,
    it uses the default values. Here are the different ways to specify it:

        - Specifying the sequence, with default ``type_id == 0``: ``$A`` or ``$B``
        - Specifying the `type_id` with default ``sequence == A``: ``$0``, ``$1``, ``$2``, ...
        - Specifying both: ``$A:0``, ``$B:1``, ...

    The same construct is used for special tokens: ``<identifier>(:<type_id>)?``.

    **Warning**: You must ensure that you are giving the correct tokens/ids as these
    will be added to the Encoding without any further check. If the given ids correspond
    to something totally different in a `Tokenizer` using this `PostProcessor`, it
    might lead to unexpected results.

    Args:
        single (:obj:`Template`):
            The template used for single sequences

        pair (:obj:`Template`):
            The template used when both sequences are specified

        special_tokens (:obj:`Tokens`):
            The list of special tokens used in each sequences

    Types:

        Template (:obj:`str` or :obj:`List`):
            - If a :obj:`str` is provided, the whitespace is used as delimiter between tokens
            - If a :obj:`List[str]` is provided, a list of tokens

        Tokens (:obj:`List[Union[Tuple[int, str], Tuple[str, int], dict]]`):
            - A :obj:`Tuple` with both a token and its associated ID, in any order
            - A :obj:`dict` with the following keys:
                - "id": :obj:`str` => The special token id, as specified in the Template
                - "ids": :obj:`List[int]` => The associated IDs
                - "tokens": :obj:`List[str]` => The associated tokens

             The given dict expects the provided :obj:`ids` and :obj:`tokens` lists to have
             the same length.
    """
    def __new__(
        cls,
        /,
        single: Incomplete | None = None,
        pair: Incomplete | None = None,
        special_tokens: Sequence2[Incomplete] | None = None,
    ) -> TemplateProcessing: ...
    @property
    def single(self, /) -> str: ...
    @single.setter
    def single(self, /, single: Incomplete) -> None: ...
