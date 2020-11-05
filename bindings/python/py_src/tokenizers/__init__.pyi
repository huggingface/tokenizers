from .decoders import *
from .models import *
from .normalizers import *
from .pre_tokenizers import *
from .processors import *
from .trainers import *

from .implementations import (
    ByteLevelBPETokenizer as ByteLevelBPETokenizer,
    CharBPETokenizer as CharBPETokenizer,
    SentencePieceBPETokenizer as SentencePieceBPETokenizer,
    BertWordPieceTokenizer as BertWordPieceTokenizer,
)

from typing import Optional, Union, List, Tuple, Callable
from enum import Enum

Offsets = Tuple[int, int]

TextInputSequence = str
PreTokenizedInputSequence = Union[List[str], Tuple[str]]
TextEncodeInput = Union[TextInputSequence, Tuple[TextInputSequence, TextInputSequence]]
PreTokenizedEncodeInput = Union[
    PreTokenizedInputSequence,
    Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence],
]

InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]
EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]

class OffsetReferential(Enum):
    ORIGINAL = "original"
    NORMALIZED = "normalized"

class OffsetType(Enum):
    BYTE = "byte"
    CHAR = "char"

class SplitDelimiterBehavior(Enum):
    REMOVED = "removed"
    ISOLATED = "isolated"
    MERGED_WITH_PREVIOUS = "merged_with_previous"
    MERGED_WITH_NEXT = "merged_with_next"
    CONTIGUOUS = "contiguous"

class Token:
    id: int
    token: str
    offsets: Offsets

Split = Tuple[str, Offsets, List[Token]]
Range = Union[int, Tuple[int, int], slice]
Pattern = Union[str, Regex]

class PreTokenizedString:
    """PreTokenizedString

    Wrapper over a string, that provides a way to normalize, pre-tokenize, tokenize the
    underlying string, while keeping track of the alignment information (offsets).

    The PreTokenizedString manages what we call `splits`. Each split represents a substring
    which is a subpart of the original string, with the relevant offsets and tokens.

    When calling one of the methods used to modify the PreTokenizedString (namely one of
    `split`, `normalize` or `tokenize), only the `splits` that don't have any associated
    tokens will get modified.
    """

    def __new__(sequence: str) -> PreTokenizedString:
        """Instantiate a new PreTokenizedString using the given str

        Args:
            sequence: str:
                The string sequence used to initialize this PreTokenizedString
        """
        pass
    def split(self, func: Callable[[index, NormalizedString], List[NormalizedString]]):
        """Split the PreTokenizedString using the given `func`

        Args:
            func: Callable[[index, NormalizedString], List[NormalizedString]]:
                The function used to split each underlying split.
                It is expected to return a list of `NormalizedString`, that represent the new
                splits. If the given `NormalizedString` does not need any splitting, we can
                just return it directly.
                In order for the offsets to be tracked accurately, any returned `NormalizedString`
                should come from calling either `.split` or `.slice` on the received one.
        """
        pass
    def normalize(self, func: Callable[[NormalizedString], None]):
        """Normalize each split of the `PreTokenizedString` using the given `func`

        Args:
            func: Callable[[NormalizedString], None]:
                The function used to normalize each underlying split. This function
                does not need to return anything, just calling the methods on the provided
                NormalizedString allow its modification.
        """
        pass
    def tokenize(self, func: Callable[[str], List[Token]]):
        """Tokenize each split of the `PreTokenizedString` using the given `func`

        Args:
            func: Callable[[str], List[Token]]:
                The function used to tokenize each underlying split. This function must return
                a list of Token generated from the input str.
        """
        pass
    def to_encoding(self, type_id: int = 0, word_idx: Optional[int] = None) -> Encoding:
        """Return an Encoding generated from this PreTokenizedString

        Args:
            type_id: int = 0:
                The type_id to be used on the generated Encoding.

            word_idx: Optional[int] = None:
                An optional word index to be used for each token of this Encoding. If provided,
                all the word indices in the generated Encoding will use this value, instead
                of the one automatically tracked during pre-tokenization.

        Returns:
            An Encoding
        """
        pass
    def get_splits(
        self,
        offset_referential: OffsetReferential = OffsetReferential.ORIGINAL,
        offset_type: OffsetType = OffsetType.CHAR,
    ) -> List[Split]:
        """Get the splits currently managed by the PreTokenizedString

        Args:
            offset_referential: OffsetReferential:
                Whether the returned splits should have offsets expressed relative
                to the original string, or the normalized one.

            offset_type: OffsetType:
                Whether the returned splits should have offsets expressed in bytes or chars.
                When slicing an str, we usually want to use chars, which is the default value.
                Now in some cases it might be interesting to get these offsets expressed in bytes,
                so it is possible to change this here.

        Returns
            A list of splits
        """
        pass

class NormalizedString:
    """NormalizedString

    A NormalizedString takes care of modifying an "original" string, to obtain a "normalized" one.
    While making all the requested modifications, it keeps track of the alignment information
    between the two versions of the string.
    """

    def __new__(sequence: str) -> NormalizedString:
        """Instantiate a new NormalizedString using the given str

        Args:
            sequence: str:
                The string sequence used to initialize this NormalizedString
        """
        pass
    @property
    def normalized(self) -> str:
        """ The normalized part of the string """
        pass
    @property
    def original(self) -> str:
        """ The original part of the string """
        pass
    def nfd(self):
        """ Runs the NFD normalization """
        pass
    def nfkd(self):
        """ Runs the NFKD normalization """
        pass
    def nfc(self):
        """ Runs the NFC normalization """
        pass
    def nfkc(self):
        """ Runs the NFKC normalization """
        pass
    def lowercase(self):
        """ Lowercase the string """
        pass
    def uppercase(self):
        """ Uppercase the string """
        pass
    def prepend(self, s: str):
        """ Prepend the given sequence to the string """
        pass
    def append(self, s: str):
        """ Append the given sequence to the string """
        pass
    def lstrip(self):
        """ Strip the left of the string """
        pass
    def rstrip(self):
        """ Strip the right of the string """
        pass
    def strip(self):
        """ Strip both ends of the string """
        pass
    def clear(self):
        """ Clear the string """
        pass
    def slice(self, range: Range) -> Optional[NormalizedString]:
        """ Slice the string using the given range """
        pass
    def filter(self, func: Callable[[str], bool]):
        """ Filter each character of the string using the given func """
        pass
    def for_each(self, func: Callable[[str], None]):
        """ Calls the given function for each character of the string """
        pass
    def map(self, func: Callable[[str], str]):
        """Calls the given function for each character of the string

        Replaces each character of the string using the returned value. Each
        returned value **must** be a str of length 1 (ie a character).
        """
        pass
    def split(self, pattern: Pattern, behavior: SplitDelimiterBehavior) -> List[NormalizedString]:
        """Split the NormalizedString using the given pattern and the specified behavior

        Args:
            pattern: Pattern:
                A pattern used to split the string. Usually a string or a Regex

            behavior: SplitDelimiterBehavior:
                The behavior to use when splitting

        Returns:
            A list of NormalizedString, representing each split
        """
        pass
    def replace(self, pattern: Pattern, content: str):
        """Replace the content of the given pattern with the provided content

        Args:
            pattern: Pattern:
                A pattern used to match the string. Usually a string or a Regex

            content: str:
                The content to be used as replacement
        """
        pass

class Regex:
    """ A Regex """

    def __new__(pattern: str) -> Regex:
        """ Instantiate a new Regex with the given pattern """
        pass

class Encoding:
    """
    The :class:`~tokenizers.Encoding` represents the output of a :class:`~tokenizers.Tokenizer`.
    """

    @staticmethod
    def merge(encodings: List[Encoding], growing_offsets: bool = True) -> Encoding:
        """Merge the list of encodings into one final :class:`~tokenizers.Encoding`

        Args:
            encodings (A :obj:`List` of :class:`~tokenizers.Encoding`):
                The list of encodings that should be merged in one

            growing_offsets (:obj:`bool`, defaults to :obj:`True`):
                Whether the offsets should accumulate while merging

        Returns:
            :class:`~tokenizers.Encoding`: The resulting Encoding
        """
        pass
    @property
    def n_sequences(self) -> int:
        """The number of sequences represented

        Returns:
            :obj:`int`: The number of sequences in this :class:`~tokenizers.Encoding`
        """
        pass
    def set_sequence_id(self, sequence_index: int):
        """Set the given sequence index

        Set the given sequence index for the whole range of tokens contained in this
        :class:`~tokenizers.Encoding`.
        """
        pass
    @property
    def ids(self) -> List[int]:
        """The generated IDs

        The IDs are the main input to a Language Model. They are the token indices,
        the numerical representations that a LM understands.

        Returns:
            :obj:`List[int]`: The list of IDs
        """
        pass
    @property
    def tokens(self) -> List[str]:
        """The generated tokens

        They are the string representation of the IDs.

        Returns:
            :obj:`List[str]`: The list of tokens
        """
        pass
    @property
    def words(self) -> List[Optional[int]]:
        """The generated word indices.

        They represent the index of the word associated to each token.
        When the input is pre-tokenized, they correspond to the ID of the given input label,
        otherwise they correspond to the words indices as defined by the
        :class:`~tokenizers.pre_tokenizers.PreTokenizer` that was used.

        For special tokens and such (any token that was generated from something that was
        not part of the input), the output is :obj:`None`

        Returns:
            A :obj:`List` of :obj:`Optional[int]`: A list of optional word index.
        """
        pass
    @property
    def sequences(self) -> List[Optional[int]]:
        """The generated sequence indices.

        They represent the index of the input sequence associated to each token.
        The sequence id can be None if the token is not related to any input sequence,
        like for example with special tokens.

        Returns:
            A :obj:`List` of :obj:`Optional[int]`: A list of optional sequence index.
        """
    @property
    def type_ids(self) -> List[int]:
        """The generated type IDs

        Generally used for tasks like sequence classification or question answering,
        these tokens let the LM know which input sequence corresponds to each tokens.

        Returns:
            :obj:`List[int]`: The list of type ids
        """
        pass
    @property
    def offsets(self) -> List[Offsets]:
        """The offsets associated to each token

        These offsets let's you slice the input string, and thus retrieve the original
        part that led to producing the corresponding token.

        Returns:
            A :obj:`List` of :obj:`Tuple[int, int]`: The list of offsets
        """
        pass
    @property
    def special_tokens_mask(self) -> List[int]:
        """The special token mask

        This indicates which tokens are special tokens, and which are not.

        Returns:
            :obj:`List[int]`: The special tokens mask
        """
        pass
    @property
    def attention_mask(self) -> List[int]:
        """The attention mask

        This indicates to the LM which tokens should be attended to, and which should not.
        This is especially important when batching sequences, where we need to applying
        padding.

        Returns:
           :obj:`List[int]`: The attention mask
        """
        pass
    @property
    def overflowing(self) -> Optional[Encoding]:
        """A :obj:`List` of overflowing :class:`~tokenizers.Encoding`

        When using truncation, the :class:`~tokenizers.Tokenizer` takes care of splitting
        the output into as many pieces as required to match the specified maximum length.
        This field lets you retrieve all the subsequent pieces.

        When you use pairs of sequences, the overflowing pieces will contain enough
        variations to cover all the possible combinations, while respecting the provided
        maximum length.
        """
        pass
    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> Optional[Tuple[int, int]]:
        """Get the encoded tokens corresponding to the word at the given index
        in one of the input sequences.

        Args:
            word_index (:obj:`int`):
                The index of a word in one of the input sequences.
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target word

        Returns:
            :obj:`Tuple[int, int]`: The range of tokens: :obj:`(first, last + 1)`
        """
        pass
    def word_to_chars(self, word_index: int, sequence_index: int = 0) -> Optional[Offsets]:
        """Get the offsets of the word at the given index in one of the input sequences.

        Args:
            word_index (:obj:`int`):
                The index of a word in one of the input sequences.
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target word

        Returns:
            :obj:`Tuple[int, int]`: The range of characters (span) :obj:`(first, last + 1)`
        """
        pass
    def token_to_sequence(self, token_index: int) -> Optional[int]:
        """Get the index of the sequence represented by the given token.

        In the general use case, this method returns :obj:`0` for a single sequence or
        the first sequence of a pair, and :obj:`1` for the second sequence of a pair

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`int`: The sequence id of the given token
        """
        pass
    def token_to_chars(self, token_index: int) -> Optional[Offsets]:
        """Get the offsets of the token at the given index.

        The returned offsets are related to the input sequence that contains the
        token.  In order to determine in which input sequence it belongs, you
        must call :meth:`~tokenizers.Encoding.token_to_sequence()`.

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`Tuple[int, int]`: The token offsets :obj:`(first, last + 1)`
        """
        pass
    def token_to_word(self, token_index: int) -> Optional[int]:
        """Get the index of the word that contains the token in one of the input sequences.

        The returned word index is related to the input sequence that contains
        the token.  In order to determine in which input sequence it belongs, you
        must call :meth:`~tokenizers.Encoding.token_to_sequence()`.

        Args:
            token_index (:obj:`int`):
                The index of a token in the encoded sequence.

        Returns:
            :obj:`int`: The index of the word in the relevant input sequence.
        """
        pass
    def char_to_token(self, pos: int, sequence_index: int = 0) -> Optional[int]:
        """Get the token that contains the char at the given position in the input sequence.

        Args:
            char_pos (:obj:`int`):
                The position of a char in the input string
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target char

        Returns:
            :obj:`int`: The index of the token that contains this char in the encoded sequence
        """
        pass
    def char_to_word(self, pos: int, sequence_index: int = 0) -> Optional[int]:
        """Get the word that contains the char at the given position in the input sequence.

        Args:
            char_pos (:obj:`int`):
                The position of a char in the input string
            sequence_index (:obj:`int`, defaults to :obj:`0`):
                The index of the sequence that contains the target char

        Returns:
            :obj:`int`: The index of the word that contains this char in the input sequence
        """
        pass
    def pad(
        self,
        length: int,
        pad_id: Optional[int] = 0,
        pad_type_id: Optional[int] = 0,
        pad_token: Optional[str] = "[PAD]",
        direction: Optional[str] = "right",
    ):
        """Pad the :class:`~tokenizers.Encoding` at the given length

        Args:
            length (:obj:`int`):
                The desired length

            direction: (:obj:`str`, defaults to :obj:`right`):
                The expected padding direction. Can be either :obj:`right` or :obj:`left`

            pad_id (:obj:`int`, defaults to :obj:`0`):
                The ID corresponding to the padding token

            pad_type_id (:obj:`int`, defaults to :obj:`0`):
                The type ID corresponding to the padding token

            pad_token (:obj:`str`, defaults to `[PAD]`):
                The pad token to use
        """
        pass
    def truncate(self, max_length: int, stride: Optional[int] = 0):
        """Truncate the :class:`~tokenizers.Encoding` at the given length

        If this :class:`~tokenizers.Encoding` represents multiple sequences, when truncating
        this information is lost. It will be considered as representing a single sequence.

        Args:
            max_length (:obj:`int`):
                The desired length

            stride (:obj:`int`, defaults to :obj:`0`):
                The length of previous content to be included in each overflowing piece
        """
        pass

class AddedToken:
    """AddedToken

    Represents a token that can be be added to a :class:`~tokenizers.Tokenizer`.
    It can have special options that defines the way it should behave.

    Args:
        content (:obj:`str`): The content of the token

        single_word (:obj:`bool`, defaults to :obj:`False`):
            Defines whether this token should only match single words. If :obj:`True`, this
            token will never match inside of a word. For example the token ``ing`` would match
            on ``tokenizing`` if this option is :obj:`False`, but not if it is :obj:`True`.
            The notion of "`inside of a word`" is defined by the word boundaries pattern in
            regular expressions (ie. the token should start and end with word boundaries).

        lstrip (:obj:`bool`, defaults to :obj:`False`):
            Defines whether this token should strip all potential whitespaces on its left side.
            If :obj:`True`, this token will greedily match any whitespace on its left. For
            example if we try to match the token ``[MASK]`` with ``lstrip=True``, in the text
            ``"I saw a [MASK]"``, we would match on ``" [MASK]"``. (Note the space on the left).

        rstrip (:obj:`bool`, defaults to :obj:`False`):
            Defines whether this token should strip all potential whitespaces on its right
            side. If :obj:`True`, this token will greedily match any whitespace on its right.
            It works just like :obj:`lstrip` but on the right.

        normalized (:obj:`bool`, defaults to :obj:`True` with :meth:`~tokenizers.Tokenizer.add_tokens` and :obj:`False` with :meth:`~tokenizers.Tokenizer.add_special_tokens`):
            Defines whether this token should match against the normalized version of the input
            text. For example, with the added token ``"yesterday"``, and a normalizer in charge of
            lowercasing the text, the token could be extract from the input ``"I saw a lion
            Yesterday"``.
    """

    def __new__(
        cls,
        content: str = "",
        single_word: bool = False,
        lstrip: bool = False,
        rstrip: bool = False,
        normalized: bool = True,
    ) -> AddedToken:
        """Instantiate a new AddedToken

        Args:
            content (:obj:`str`): The content of the token

            single_word (:obj:`bool`, defaults to :obj:`False`):
                Defines whether this token should only match single words. If :obj:`True`, this
                token will never match inside of a word. For example the token ``ing`` would match
                on ``tokenizing`` if this option is :obj:`False`, but not if it is :obj:`True`.
                The notion of "`inside of a word`" is defined by the word boundaries pattern in
                regular expressions (ie. the token should start and end with word boundaries).

            lstrip (:obj:`bool`, defaults to :obj:`False`):
                Defines whether this token should strip all potential whitespaces on its left side.
                If :obj:`True`, this token will greedily match any whitespace on its left. For
                example if we try to match the token ``[MASK]`` with ``lstrip=True``, in the text
                ``"I saw a [MASK]"``, we would match on ``" [MASK]"``. (Note the space on the left).

            rstrip (:obj:`bool`, defaults to :obj:`False`):
                Defines whether this token should strip all potential whitespaces on its right
                side. If :obj:`True`, this token will greedily match any whitespace on its right.
                It works just like :obj:`lstrip` but on the right.

            normalized (:obj:`bool`, defaults to :obj:`True` with :meth:`~tokenizers.Tokenizer.add_tokens` and :obj:`False` with :meth:`~tokenizers.Tokenizer.add_special_tokens`):
                Defines whether this token should match against the normalized version of the input
                text. For example, with the added token ``"yesterday"``, and a normalizer in charge of
                lowercasing the text, the token could be extract from the input ``"I saw a lion
                Yesterday"``.
        """
        pass

class Tokenizer:
    """Tokenizer

    A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
    and outputs an :class:`~tokenizers.Encoding`.

    Args:
        model (:class:`~tokenizers.models.Model`):
            The core algorithm that this :obj:`Tokenizer` should be using.
    """

    def __new__(cls, model: models.Model) -> Tokenizer:
        """Instantiate a new Tokenizer using the given Model

        A :obj:`Tokenizer` works as a pipeline. It processes some raw text as input
        and outputs an :class:`~tokenizers.Encoding`.

        Args:
            model (:class:`~tokenizers.models.Model`):
                The core algorithm that this :obj:`Tokenizer` should be using.

        Returns:
            Tokenizer
        """
        pass
    @staticmethod
    def from_str(s: str) -> Tokenizer:
        """Instantiate a new :class:`~tokenizers.Tokenizer` from the given JSON string.

        Args:
            json (:obj:`str`):
                A valid JSON string representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        pass
    @staticmethod
    def from_file(path: str) -> Tokenizer:
        """Instantiate a new :class:`~tokenizers.Tokenizer` from the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a local JSON file representing a previously serialized
                :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        pass
    @staticmethod
    def from_buffer(buffer: bytes) -> Tokenizer:
        """Instantiate a new :class:`~tokenizers.Tokenizer` from the given buffer.

        Args:
            buffer (:obj:`bytes`):
                A buffer containing a previously serialized :class:`~tokenizers.Tokenizer`

        Returns:
            :class:`~tokenizers.Tokenizer`: The new tokenizer
        """
        pass
    def to_str(self, pretty: bool = False) -> str:
        """Gets a serialized string representing this :class:`~tokenizers.Tokenizer`.

        Args:
            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON string should be pretty formatted.

        Returns:
            :obj:`str`: A string representing the serialized Tokenizer
        """
        pass
    def save(self, path: str, pretty: bool = False):
        """Save the :class:`~tokenizers.Tokenizer` to the file at the given path.

        Args:
            path (:obj:`str`):
                A path to a file in which to save the serialized tokenizer.

            pretty (:obj:`bool`, defaults to :obj:`False`):
                Whether the JSON file should be pretty formatted.
        """
        pass
    @property
    def model(self) -> Model:
        """ Get the model in use with this Tokenizer """
        pass
    @model.setter
    def model(self, model: models.Model):
        """ Change the model to use with this Tokenizer """
        pass
    @property
    def pre_tokenizer(self) -> Optional[PreTokenizer]:
        """ Get the pre-tokenizer in use with this model """
        pass
    @pre_tokenizer.setter
    def pre_tokenizer(self, pre_tokenizer: pre_tokenizers.PreTokenizer):
        """ Change the pre tokenizer to use with this Tokenizer """
        pass
    @property
    def decoder(self) -> Optional[Decoder]:
        """ Get the decoder in use with this model """
        pass
    @decoder.setter
    def decoder(self, decoder: decoders.Decoder):
        """ Change the decoder to use with this Tokenizer """
        pass
    @property
    def post_processor(self) -> Optional[PostProcessor]:
        """ Get the post-processor in use with this Tokenizer """
        pass
    @post_processor.setter
    def post_processor(self, processor: processors.PostProcessor):
        """ Change the post processor to use with this Tokenizer """
    @property
    def normalizer(self) -> Optional[Normalizer]:
        """ Get the normalizer in use with this Tokenizer """
        pass
    @normalizer.setter
    def normalizer(self, normalizer: normalizers.Normalizer):
        """ Change the normalizer to use with this Tokenizer """
    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        """
        Return the number of special tokens that would be added for single/pair sentences.
        :param is_pair: Boolean indicating if the input would be a single sentence or a pair
        :return:
        """
        pass
    def get_vocab(self, with_added_tokens: bool = True) -> Dict[str, int]:
        """Get the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`Dict[str, int]`: The vocabulary
        """
        pass
    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """Get the size of the underlying vocabulary

        Args:
            with_added_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to include the added tokens

        Returns:
            :obj:`int`: The size of the vocabulary
        """
        pass
    def enable_truncation(self, max_length: int, stride: Optional[int], strategy: Optional[str]):
        """Enable truncation

        Args:
            max_length (:obj:`int`):
                The max length at which to truncate

            stride (:obj:`int`, `optional`):
                The length of the previous first sequence to be included in the overflowing
                sequence

            strategy (:obj:`str`, `optional`, defaults to :obj:`longest_first`):
                The strategy used to truncation. Can be one of ``longest_first``, ``only_first`` or
                ``only_second``.
        """
        pass
    def no_truncation(self):
        """ Disable truncation """
        pass
    @property
    def truncation(self) -> Optional[dict]:
        """Get the currently set truncation parameters

        `Cannot set, use` :meth:`~tokenizers.Tokenizer.enable_truncation` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current truncation parameters if truncation is enabled
        """
        pass
    def enable_padding(
        self,
        direction: Optional[str] = "right",
        pad_to_multiple_of: Optional[int] = None,
        pad_id: Optional[int] = 0,
        pad_type_id: Optional[int] = 0,
        pad_token: Optional[str] = "[PAD]",
        length: Optional[int] = None,
    ):
        """Enable the padding

        Args:
            direction (:obj:`str`, `optional`, defaults to :obj:`right`):
                The direction in which to pad. Can be either ``right`` or ``left``

            pad_to_multiple_of (:obj:`int`, `optional`):
                If specified, the padding length should always snap to the next multiple of the
                given value. For example if we were going to pad witha length of 250 but
                ``pad_to_multiple_of=8`` then we will pad to 256.

            pad_id (:obj:`int`, defaults to 0):
                The id to be used when padding

            pad_type_id (:obj:`int`, defaults to 0):
                The type id to be used when padding

            pad_token (:obj:`str`, defaults to :obj:`[PAD]`):
                The pad token to be used when padding

            length (:obj:`int`, `optional`):
                If specified, the length at which to pad. If not specified we pad using the size of
                the longest sequence in a batch.
        """
        pass
    def no_padding(self):
        """ Disable padding """
        pass
    @property
    def padding(self) -> Optional[dict]:
        """Get the current padding parameters

        `Cannot be set, use` :meth:`~tokenizers.Tokenizer.enable_padding` `instead`

        Returns:
            (:obj:`dict`, `optional`):
                A dict with the current padding parameters if padding is enabled
        """
        pass
    def encode(
        self,
        sequence: InputSequence,
        pair: Optional[InputSequence],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Encode the given sequence and pair. This method can process raw text sequences
        as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode("A single sequence")`
                encode("A sequence", "And its pair")`
                encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
                encode(
                    [ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
                    is_pretokenized=True
                )

        Args:
            sequence (:obj:`~tokenizers.InputSequence`):
                The main input sequence we want to encode. This sequence can be either raw
                text or pre-tokenized, according to the ``is_pretokenized`` argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextInputSequence`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedInputSequence`

            pair (:obj:`~tokenizers.InputSequence`, `optional`):
                An optional input sequence. The expected format is the same that for ``sequence``.

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The encoded result
        """
        pass
    def encode_batch(
        self,
        inputs: List[EncodeInput],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List[Encoding]:
        """Encode the given batch of inputs. This method accept both raw text sequences
        as well as already pre-tokenized sequences.

        Example:
            Here are some examples of the inputs that are accepted::

                encode_batch([
                    "A single sequence",
                    ("A tuple with a sequence", "And its pair"),
                    [ "A", "pre", "tokenized", "sequence" ],
                    ([ "A", "pre", "tokenized", "sequence" ], "And its pair")
                ])

        Args:
            input (A :obj:`List`/:obj:`Tuple` of :obj:`~tokenizers.EncodeInput`):
                A list of single sequences or pair sequences to encode. Each sequence
                can be either raw text or pre-tokenized, according to the ``is_pretokenized``
                argument:

                - If ``is_pretokenized=False``: :class:`~tokenizers.TextEncodeInput`
                - If ``is_pretokenized=True``: :class:`~tokenizers.PreTokenizedEncodeInput`

            is_pretokenized (:obj:`bool`, defaults to :obj:`False`):
                Whether the input is already pre-tokenized

            add_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether to add the special tokens

        Returns:
            A :obj:`List` of :class:`~tokenizers.Encoding`: The encoded batch

        """
        pass
    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids back to a string

        This is used to decode anything coming back from a Language Model

        Args:
            ids (A :obj:`List/Tuple` of :obj:`int`):
                The list of ids that we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded string

        Returns:
            :obj:`str`: The decoded string
        """
        pass
    def decode_batch(
        self, sequences: List[List[int]], skip_special_tokens: Optional[bool] = True
    ) -> str:
        """Decode a batch of ids back to their corresponding string

        Args:
            sequences (:obj:`List` of :obj:`List[int]`):
                The batch of sequences we want to decode

            skip_special_tokens (:obj:`bool`, defaults to :obj:`True`):
                Whether the special tokens should be removed from the decoded strings

        Returns:
            :obj:`List[str]`: A list of decoded strings
        """
        pass
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert the given token to its corresponding id if it exists

        Args:
            token (:obj:`str`):
                The token to convert

        Returns:
            :obj:`Optional[int]`: An optional id, :obj:`None` if out of vocabulary
        """
        pass
    def id_to_token(self, id: int) -> Optional[str]:
        """Convert the given id to its corresponding token if it exists

        Args:
            id (:obj:`int`):
                The id to convert

        Returns:
            :obj:`Optional[str]`: An optional token, :obj:`None` if out of vocabulary
        """
        pass
    def add_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """Add the given tokens to the vocabulary

        The given tokens are added only if they don't already exist in the vocabulary.
        Each token then gets a new attributed id.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of tokens we want to add to the vocabulary. Each token can be either a
                string or an instance of :class:`~tokenizers.AddedToken` for more customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        pass
    def add_special_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """Add the given special tokens to the Tokenizer.

        If these tokens are already part of the vocabulary, it just let the Tokenizer know about
        them. If they don't exist, the Tokenizer creates them, giving them a new id.

        These special tokens will never be processed by the model (ie won't be split into
        multiple tokens), and they can be removed from the output when decoding.

        Args:
            tokens (A :obj:`List` of :class:`~tokenizers.AddedToken` or :obj:`str`):
                The list of special tokens we want to add to the vocabulary. Each token can either
                be a string or an instance of :class:`~tokenizers.AddedToken` for more
                customization.

        Returns:
            :obj:`int`: The number of tokens that were created in the vocabulary
        """
        pass
    def post_process(
        self,
        encoding: Encoding,
        pair: Optional[Encoding] = None,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Apply all the post-processing steps to the given encodings.

        The various steps are:

            1. Truncate according to the set truncation params (provided with
               :meth:`~tokenizers.Tokenizer.enable_truncation`)
            2. Apply the :class:`~tokenizers.processors.PostProcessor`
            3. Pad according to the set padding params (provided with
               :meth:`~tokenizers.Tokenizer.enable_padding`)

        Args:
            encoding (:class:`~tokenizers.Encoding`):
                The :class:`~tokenizers.Encoding` corresponding to the main sequence.

            pair (:class:`~tokenizers.Encoding`, `optional`):
                An optional :class:`~tokenizers.Encoding` corresponding to the pair sequence.

            add_special_tokens (:obj:`bool`):
                Whether to add the special tokens

        Returns:
            :class:`~tokenizers.Encoding`: The final post-processed encoding
        """
        pass
