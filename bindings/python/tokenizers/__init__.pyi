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

from typing import Optional, Union, List, Tuple

Offsets = Tuple[int, int]

class Encoding:
    """ An Encoding as returned by the Tokenizer """

    @staticmethod
    def merge(encodings: List[Encoding], growing_offsets: bool = True) -> Encoding:
        """ Merge the list of Encoding into one final Encoding

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
        """ The offsets.
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
        """ Pad the current Encoding at the given length

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
        """ Truncate the current Encoding at the given max_length

        Args:
            max_length: int:
                The maximum length to be kept

            stride: (`optional`) unsigned int:
                The length of the previous first sequence to be included
                in the overflowing sequence
        """
        pass

class AddedToken:
    """ AddedToken represents a token to be added to a Tokenizer

    An AddedToken can have special options defining the way it should behave.
    """

    def __new__(
        cls, content: str, single_word: bool = False, lstrip: bool = False, rstrip: bool = False
    ) -> AddedToken:
        """ Instantiate a new AddedToken

        Args:
            content: str:
                The content of the token

            single_word: bool
                Whether this token should only match against single word. If True,
                this token will never match inside of a word.

            lstrip: bool
                Whether this token should strip all potential whitespaces on the left side.
                If True, this token will greedily match any whitespace on the left and then strip
                them out.

            rstrip: bool
                Whether this token should strip all potential whitespaces on the right side.
                If True, this token will greedily match any whitespace on the right and then strip
                them out.
        """
        pass

class Tokenizer:
    """ Tokenizer

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
        """ Instantiate a new Tokenizer using the given Model

        Args:
            model: models.Model:
                The model to be used with this Tokenizer

        Returns:
            Tokenizer
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
        """ Returns the vocabulary

        Args:
            with_added_tokens: boolean:
                Whether to include the added tokens in the vocabulary

        Returns:
            The vocabulary
        """
        pass
    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        """ Returns the size of the vocabulary

        Args:
            with_added_tokens: boolean:
                Whether to include the added tokens in the vocabulary's size

        Returns:
            The size of the vocabulary
        """
        pass
    def enable_truncation(self, max_length: int, stride: Optional[int], strategy: Optional[str]):
        """ Enable the truncation

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
    def enable_padding(
        self,
        direction: Optional[str] = "right",
        pad_id: Optional[int] = 0,
        pad_type_id: Optional[int] = 0,
        pad_token: Optional[str] = "[PAD]",
        max_length: Optional[int] = None,
    ):
        """ Enable the padding

        Args:
            direction: (`optional`) str:
                Can be one of: `right` or `left`

            pad_id: (`optional`) unsigned int:
                The indice to be used when padding

            pad_type_id: (`optional`) unsigned int:
                The type indice to be used when padding

            pad_token: (`optional`) str:
                The pad token to be used when padding

            max_length: (`optional`) unsigned int:
                If specified, the length at which to pad. If not specified
                we pad using the size of the longest sequence in a batch
        """
        pass
    def no_padding(self):
        """ Disable padding """
        pass
    def normalize(self, sequence: str) -> str:
        """ Normalize the given sequence

        Args:
            sequence: str:
                The sequence to normalize

        Returns:
            The normalized string
        """
        pass
    def encode(
        self, sequence: str, pair: Optional[str] = None, add_special_tokens: bool = True
    ) -> Encoding:
        """ Encode the given sequence

        Args:
            sequence: str:
                The sequence to encode

            pair: (`optional`) Optional[str]:
                The optional pair sequence

            add_special_tokens: bool:
                Whether to add the special tokens while encoding

        Returns:
            An Encoding
        """
        pass
    def encode_batch(
        self, sequences: List[Union[str, Tuple[str, str]]], add_special_tokens: bool = True
    ) -> List[Encoding]:
        """ Encode the given sequences or pair of sequences

        Args:
            sequences: List[Union[str, Tuple[str, str]]]:
                A list of sequences or pair of sequences. The list can contain both
                at the same time.

            add_special_tokens: bool:
                Whether to add the special tokens while encoding

        Returns:
            A list of Encoding
        """
        pass
    def decode(self, ids: List[int], skip_special_tokens: Optional[bool] = True) -> str:
        """ Decode the given list of ids to a string sequence

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
        """ Decode the list of sequences to a list of string sequences

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
        """ Convert the given token to its corresponding id

        Args:
            token: str:
                The token to convert

        Returns:
            The corresponding id if it exists, None otherwise
        """
        pass
    def id_to_token(self, id: int) -> Optional[str]:
        """ Convert the given token id to its corresponding string

        Args:
            token: id:
                The token id to convert

        Returns:
            The corresponding string if it exists, None otherwise
        """
        pass
    def add_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """ Add the given tokens to the vocabulary

        Args:
            tokens: List[Union[str, AddedToken]]:
                A list of tokens to add to the vocabulary. Each token can either be
                a string, or an instance of AddedToken

        Returns:
            The number of tokens that were added to the vocabulary
        """
        pass
    def add_special_tokens(self, tokens: List[Union[str, AddedToken]]) -> int:
        """ Add the given special tokens to the vocabulary, and treat them as special tokens.

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
        self, encoding: Encoding, pair: Optional[Encoding] = None, add_special_tokens: bool = True
    ) -> Encoding:
        """ Apply all the post-processing steps to the given encodings.

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
