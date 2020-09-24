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
    """ An Encoding as returned by the Tokenizer """

    @staticmethod
    def merge(encodings: List[Encoding], growing_offsets: bool = True) -> Encoding:
        """Merge the list of Encoding into one final Encoding

        Args:
            encodings: List[Encoding]:
                The list of encodings

            growing_offsets: bool:
                Whether the offsets should accumulate while merging

        Returns:
            The resulting Encoding
        """
        pass
    @property
    def ids(self) -> List[int]:
        """ The tokenized ids """
        pass
    @property
    def tokens(self) -> List[str]:
        """ The tokenized strings """
        pass
    @property
    def words(self) -> List[Optional[int]]:
        """ The tokenized words index """
        pass
    @property
    def type_ids(self) -> List[int]:
        """ The type ids """
        pass
    @property
    def offsets(self) -> List[Offsets]:
        """The offsets.
        These offsets can be used to index any `IndexableString` directly. If you want to
        index the original `str`, make sure to retrieve the converted offsets using the `.offsets`
        method on the `original_str`.
        """
        pass
    @property
    def special_tokens_mask(self) -> List[int]:
        """ The special tokens mask """
        pass
    @property
    def attention_mask(self) -> List[int]:
        """ The attention mask """
        pass
    @property
    def overflowing(self) -> Optional[Encoding]:
        """ The overflowing encoding, after truncation """
        pass
    def word_to_tokens(self, word_index: int) -> Optional[Tuple[int, int]]:
        """
        Get the encoded tokens corresponding to the word at the given index in the input
        sequence, with the form [start_token, end_token + 1]

        Args:
            word_index: int:
                The index of the word in the input sequence.

        Returns:
            The range of tokens with the form [start_token, end_token + 1]
        """
        pass
    def word_to_chars(self, word_index: int) -> Optional[Offsets]:
        """
        Get the offsets of the word at the given index in the input sequence.

        Args:
            word_index: int:
                The index of the word in the input sequence.

        Returns:
            The word offsets
        """
        pass
    def token_to_chars(self, token_index: int) -> Optional[Offsets]:
        """
        Get the offsets of the token at the given index

        Args:
            token_index: int:
                The index of the token in the encoded sequence.

        Returns:
            The token offsets
        """
        pass
    def token_to_word(self, token_index: int) -> Optional[int]:
        """
        Get the word that contains the token at the given index

        Args:
            token_index: int:
                The index of the token in the encoded sequence.

        Returns:
            The index of the word in the input sequence.
        """
        pass
    def char_to_token(self, pos: int) -> Optional[int]:
        """
        Get the token that contains the char at the given position

        Args:
            pos: int:
                The position of a char in the input string

        Returns:
            The index of the token that contains this char
        """
        pass
    def char_to_word(self, pos: int) -> Optional[int]:
        """
        Get the word that contains the given char.

        Args:
            pos: int:
                The position of a char in the input string

        Returns:
            The index of the word that contains this char
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
        """Pad the current Encoding at the given length

        Args:
            length: int:
                The length at which to pad

            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding
        """
        pass
    def truncate(self, max_length: int, stride: Optional[int] = 0):
        """Truncate the current Encoding at the given max_length

        Args:
            max_length: int:
                The maximum length to be kept

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence
        """
        pass

class AddedToken:
    """AddedToken represents a token to be added to a Tokenizer

    An AddedToken can have special options defining the way it should behave.
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
            content: str:
                The content of the token

            single_word: bool
                Whether this token should only match against single words. If True,
                this token will never match inside of a word. For example the token `ing` would
                match on `tokenizing` if this option if False, but not if this option is True.

            lstrip: bool
                Whether this token should strip all potential whitespaces on the left side.
                If True, this token will greedily match any whitespace on the left. For example,
                if we try to match the token `[MASK]` with lstrip=True, in the text `I saw a [MASK]`
                we will match on ` [MASK]`.

            rstrip: bool
                Whether this token should strip all potential whitespaces on the right side.
                If True, this token will greedily match any whitespace on the right. It works just
                like lstrip, but on the right.

            normalized: bool:
                Whether this token should be match the normalized version of the input text. For
                example, with the added token `yesterday` and a normalizer in charge of lowercasing
                the text, the token could be extract from the input `I saw a lion Yesterday`.
        """
        pass

class Tokenizer:
    """Tokenizer

    A Tokenizer works as a pipeline, it processes some raw text as input and outputs
    an `Encoding`.

    The various steps of the pipeline are:
        1. The `Normalizer`: in charge of normalizing the text. Common examples of
           normalization are the unicode normalization standards, such as NFD or NFKC.
        2. The `PreTokenizer`: in charge of creating initial words splits in the text.
           The most common way of splitting text is simply on whitespace.
        3. The `Model`: in charge of doing the actual tokenization. An example of a
           `Model` would be `BPE` or `WordPiece`.
        4. The `PostProcessor`: in charge of post-processing the `Encoding` to add anything
           relevant that, for example, a language model would need, such as special tokens.
    """

    def __new__(cls, model: models.Model) -> Tokenizer:
        """Instantiate a new Tokenizer using the given Model

        Args:
            model: models.Model:
                The model to be used with this Tokenizer

        Returns:
            Tokenizer
        """
        pass
    @staticmethod
    def from_str(s: str) -> Tokenizer:
        """Instantiate a new Tokenizer from the given JSON string

        Args:
            s: str:
                A JSON string representation of the Tokenizer

        Returns:
            Tokenizer
        """
        pass
    @staticmethod
    def from_file(path: str) -> Tokenizer:
        """Instantiate a new Tokenizer from the given file

        Args:
            path: str:
                Path to a file containing a Tokenizer

        Returns:
            Tokenizer
        """
        pass
    @staticmethod
    def from_buffer(buffer: bytes) -> Tokenizer:
        """Instantiate a new Tokenizer from the given buffer

        Args:
            buffer: bytes:
                A buffer used to instantiate a new Tokenizer

        Returns:
            Tokenizer
        """
        pass
    def to_str(self, pretty: bool = False) -> str:
        """Get a serialized JSON version of the Tokenizer as a str

        Args:
            pretty: bool:
                Whether the JSON string should be prettified

        Returns:
            str
        """
        pass
    def save(self, path: str, pretty: bool = False):
        """Save the Tokenizer as JSON to the given path

        Args:
            pretty: bool:
                Whether the JSON string should be prettified
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
        """Returns the vocabulary

        Args:
            with_added_tokens: boolean:
                Whether to include the added tokens in the vocabulary

        Returns:
            The vocabulary
        """
        pass
    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """Returns the size of the vocabulary

        Args:
            with_added_tokens: boolean:
                Whether to include the added tokens in the vocabulary's size

        Returns:
            The size of the vocabulary
        """
        pass
    def enable_truncation(self, max_length: int, stride: Optional[int], strategy: Optional[str]):
        """Enable the truncation

        Args:
            max_length: unsigned int:
                The maximum length at which to truncate

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence

            strategy: (`optional) str:
                Can be one of `longest_first`, `only_first` or `only_second`
        """
        pass
    def no_truncation(self):
        """ Disable truncation """
        pass
    @property
    def truncation(self) -> Optional[dict]:
        """Get the current truncation parameters

        Returns:
            None if truncation is disabled, a dict with the current truncation parameters if
            truncation is enabled
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
            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_to_multiple_of: (`optional`) unsigned int:
                If specified, the padding length should always snap to the next multiple of
                the given value. For example if we were going to pad with a length of 250 but
                `pad_to_multiple_of=8` then we will pad to 256.

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding

            length: (`optional`) unsigned int:
                If specified, the length at which to pad. If not specified
                we pad using the size of the longest sequence in a batch
        """
        pass
    def no_padding(self):
        """ Disable padding """
        pass
    @property
    def padding(self) -> Optional[dict]:
        """Get the current padding parameters

        Returns:
            None if padding is disabled, a dict with the currently set parameters
            if the padding is enabled.
        """
        pass
    def encode(
        self,
        sequence: InputSequence,
        pair: Optional[InputSequence],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> Encoding:
        """Encode the given sequence and pair. This method can process raw text sequences as well
        as already pre-tokenized sequences.

        Args:
            sequence: InputSequence:
                The sequence we want to encode. This sequence can be either raw text or
                pre-tokenized, according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            is_pretokenized: bool:
                Whether the input is already pre-tokenized

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            An Encoding
        """
        pass
    def encode_batch(
        self,
        inputs: List[EncodeInput],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> List[Encoding]:
        """Encode the given inputs. This method accept both raw text sequences as well as already
        pre-tokenized sequences.

        Args:
            inputs: List[EncodeInput]:
                A list of single sequences or pair sequences to encode. Each `EncodeInput` is
                expected to be of the following form:
                    `Union[InputSequence, Tuple[InputSequence, InputSequence]]`

                Each `InputSequence` can either be raw text or pre-tokenized,
                according to the `is_pretokenized` argument:

                - If `is_pretokenized=False`: `InputSequence` is expected to be `str`
                - If `is_pretokenized=True`: `InputSequence` is expected to be
                    `Union[List[str], Tuple[str]]`

            is_pretokenized: bool:
                Whether the input is already pre-tokenized.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding.

        Returns:
            A list of Encoding
        """
        pass
    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """Decode the given list of ids to a string sequence

        Args:
            ids: List[unsigned int]:
                A list of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output string

        Returns:
            The decoded string
        """
        pass
    def decode_batch(
        self, sequences: List[List[int]], skip_special_tokens: Optional[bool] = True
    ) -> str:
        """Decode the list of sequences to a list of string sequences

        Args:
            sequences: List[List[unsigned int]]:
                A list of sequence of ids to be decoded

            skip_special_tokens: (`optional`) boolean:
                Whether to remove all the special tokens from the output strings

        Returns:
            A list of decoded strings
        """
        pass
    def token_to_id(self, token: str) -> Optional[int]:
        """Convert the given token to its corresponding id

        Args:
            token: str:
                The token to convert

        Returns:
            The corresponding id if it exists, None otherwise
        """
        pass
    def id_to_token(self, id: int) -> Optional[str]:
        """Convert the given token id to its corresponding string

        Args:
            token: id:
                The token id to convert

        Returns:
            The corresponding string if it exists, None otherwise
        """
        pass
    def add_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """Add the given tokens to the vocabulary

        Args:
            tokens: List[Union[str, AddedToken]]:
                A list of tokens to add to the vocabulary. Each token can either be
                a string, or an instance of AddedToken

        Returns:
            The number of tokens that were added to the vocabulary
        """
        pass
    def add_special_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """Add the given special tokens to the vocabulary, and treat them as special tokens.

        The special tokens will never be processed by the model, and will be
        removed while decoding.

        Args:
            tokens: List[Union[str, AddedToken]]:
                The list of special tokens to add. Each token can either be a string
                or an instance of AddedToken

        Returns:
            The number of tokens that were added to the vocabulary
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
            1. Truncate according to global params (provided to `enable_truncation`)
            2. Apply the PostProcessor
            3. Pad according to global params. (provided to `enable_padding`)

        Args:
            encoding: Encoding:
                The main Encoding to post process

            pair: Optional[Encoding]:
                An optional pair Encoding

            add_special_tokens: bool:
                Whether to add special tokens

        Returns:
            The resulting Encoding
        """
        pass
